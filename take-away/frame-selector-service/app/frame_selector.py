import os
import io
import time
import logging
import socket
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
from ultralytics import YOLO
import requests
from config_loader import load_config
from pathlib import Path
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Auto-detect STATION_ID from hostname if not set
# Docker Compose scales containers with names like: order-accuracy-frame-selector-1, order-accuracy-frame-selector-2
if 'STATION_ID' not in os.environ:
    hostname = socket.gethostname()
    # Extract number from hostname (e.g., order-accuracy-frame-selector-2 -> station_2)
    if hostname and hostname.split('-')[-1].isdigit():
        station_num = hostname.split('-')[-1]
        os.environ['STATION_ID'] = f'station_{station_num}'
        logger.info(f"Auto-detected STATION_ID from hostname: {os.environ['STATION_ID']}")
    else:
        os.environ['STATION_ID'] = 'station_1'
        logger.info(f"Using default STATION_ID: station_1")

cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]
FS_CFG = cfg["frame_selector"]
VLM_CFG = cfg["vlm"]

MINIO_ENDPOINT = MINIO["endpoint"]
FRAMES_BUCKET = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]

TOP_K = FS_CFG["top_k"]
POLL_INTERVAL = FS_CFG["poll_interval_sec"]
SKIP_LABELS = set(FS_CFG["skip_labels"])

# How many consecutive frames required to confirm a new order
MIN_FRAMES_PER_ORDER = FS_CFG.get("min_frames_per_order", 2)

VLM_ENDPOINT = VLM_CFG["endpoint"]

# Station ID from environment variable (for multi-station deployment)
STATION_ID = os.environ.get('STATION_ID', 'station_unknown')

processed_orders = set()

logger.info(f"Frame selector configuration: station_id={STATION_ID}, top_k={TOP_K}, poll_interval={POLL_INTERVAL}s, min_frames={MIN_FRAMES_PER_ORDER}")
logger.info(f"Skip labels: {SKIP_LABELS}")
logger.info(f"VLM endpoint: {VLM_ENDPOINT}")


# =====================================================
# VLM Caller
# =====================================================

def call_vlm(order_id, timeout=120):
    logger.info(f"[VLM-CALL] Calling VLM service for order_id={order_id}")
    payload = {"order_id": order_id}
    logger.debug(f"[VLM-CALL] Payload: {payload}")
    
    try:
        resp = requests.post(VLM_ENDPOINT, json=payload, timeout=timeout)
        resp.raise_for_status()
        result = resp.json()
        logger.info(f"[VLM-CALL] VLM service responded successfully for order_id={order_id}")
        logger.debug(f"[VLM-CALL] Response: {result}")
        return result
    except requests.exceptions.Timeout:
        logger.error(f"[VLM-CALL] Timeout after {timeout}s for order_id={order_id}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"[VLM-CALL] Request failed for order_id={order_id}: {e}")
        raise


# =====================================================
# Model + MinIO
# =====================================================

logger.info("Initializing YOLO model loading process")

# Define model paths (use /app/models for persistence)
model_dir = Path("/app/models")
model_dir.mkdir(exist_ok=True)

# Define dataset paths (use /app/datasets for persistence)
dataset_dir = Path("/app/datasets")
dataset_dir.mkdir(exist_ok=True)

# Set dataset directory for ultralytics
os.environ['YOLO_DATASETS_DIR'] = str(dataset_dir)

yolo_model = model_dir / "yolo11n.pt"
openvino_fp32_path = model_dir / "yolo11n_openvino_model"
openvino_int8_path = model_dir / "yolo11n_int8_openvino_model"

logger.info(f"Model directory: {model_dir}")
logger.info(f"Dataset directory: {dataset_dir}")

# Step 1: Download YOLOv11 model (if not exists)
if not yolo_model.exists():
    logger.info(f"Downloading YOLOv11 model to {yolo_model}")
    model_pt = YOLO(str(yolo_model))
    logger.info("YOLOv11 model downloaded successfully")
else:
    logger.info(f"YOLOv11 model already exists: {yolo_model}")

# Step 2: Convert to OpenVINO FP32 format (if not exists)
if not openvino_fp32_path.exists():
    logger.info("Converting YOLOv11 to OpenVINO FP32 format")
    # Change to model directory before export
    original_dir = os.getcwd()
    os.chdir(str(model_dir))
    
    model_pt = YOLO(str(yolo_model))
    model_pt.export(format="openvino", half=False)
    
    os.chdir(original_dir)
    logger.info("OpenVINO FP32 conversion complete")
else:
    logger.info(f"OpenVINO FP32 model already exists: {openvino_fp32_path}")

# Step 3: Quantize to INT8 (if not exists)
if not openvino_int8_path.exists():
    logger.info("Quantizing model to INT8")
    
    # Change to model directory before export
    original_dir = os.getcwd()
    os.chdir(str(model_dir))
    
    model_pt = YOLO(str(yolo_model))
    model_pt.export(format="openvino", int8=True, data="coco128.yaml")
    
    # Rename from default to int8 path
    default_output = Path("yolo11n_openvino_model")
    if default_output.exists() and not openvino_int8_path.exists():
        default_output.rename(openvino_int8_path.name)
    
    os.chdir(original_dir)
    logger.info("INT8 quantization complete")
else:
    logger.info(f"INT8 model already exists: {openvino_int8_path}")

# Step 4: Load the INT8 OpenVINO model
logger.info("Loading INT8 OpenVINO model")
model = YOLO(str(openvino_int8_path), task="detect")
logger.info("INT8 OpenVINO model loaded successfully")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO["access_key"],
    secret_key=MINIO["secret_key"],
    secure=MINIO.get("secure", False),
)

logger.info(f"MinIO client initialized: endpoint={MINIO_ENDPOINT}")


# =====================================================
# Helpers
# =====================================================

def wait_for_bucket(bucket):
    logger.info(f"Waiting for bucket to be available: {bucket}")
    while True:
        try:
            if client.bucket_exists(bucket):
                logger.info(f"Bucket available: {bucket}")
                return
        except Exception as e:
            logger.debug(f"Bucket check failed, retrying: {e}")
            pass
        time.sleep(1)


def ensure_buckets():
    logger.info("Ensuring MinIO buckets exist")
    wait_for_bucket(FRAMES_BUCKET)
    if not client.bucket_exists(SELECTED_BUCKET):
        logger.info(f"Creating bucket: {SELECTED_BUCKET}")
        client.make_bucket(SELECTED_BUCKET)
    else:
        logger.info(f"Bucket already exists: {SELECTED_BUCKET}")


def list_frames_sorted():
    logger.debug(f"Listing frames from bucket: {FRAMES_BUCKET} (station: {STATION_ID})")
    frames = []
    eos_seen = False

    for obj in client.list_objects(FRAMES_BUCKET, recursive=True):
        # Check for EOS marker for this station
        if obj.object_name == f"{STATION_ID}/__EOS__":
            eos_seen = True
            logger.info(f"EOS marker detected for station {STATION_ID}")
            continue

        # Only process frames for this station
        if not obj.object_name.startswith(f"{STATION_ID}/"):
            continue

        if obj.object_name.lower().endswith(".jpg"):
            # Extract: station_id/order_id/frame_X.jpg -> order_id
            parts = obj.object_name.split("/")
            if len(parts) >= 3 and parts[0] == STATION_ID:
                order_id = parts[1]
                frames.append((order_id, obj.object_name))

    frames.sort(key=lambda x: x[1])
    logger.debug(f"Found {len(frames)} frames for station {STATION_ID}, eos_seen={eos_seen}")
    return frames, eos_seen


def load_image(key):
    resp = client.get_object(FRAMES_BUCKET, key)
    data = resp.read()
    resp.close()
    resp.release_conn()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def count_items(frame):
    logger.debug("Running YOLO detection on frame")
    result = model(frame, conf=0.1, verbose=False)[0]

    # If ANY skip-label (like person/hand) is present → discard frame
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        if cls_name in SKIP_LABELS:
            logger.debug(f"Frame contains skip label: {cls_name}")
            return -1   # mark frame as invalid

    # Otherwise count valid objects
    count = 0
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        if cls_name not in SKIP_LABELS:
            count += 1

    logger.debug(f"Frame contains {count} valid objects")
    return count



# =====================================================
# Order Finalization
# =====================================================

def process_completed_order(order_id, keys):
    if not keys:
        logger.debug(f"Skipping empty order: order_id={order_id}")
        return

    # Prevent duplicate VLM calls
    if order_id in processed_orders:
        logger.warning(f"Order already processed, skipping: order_id={order_id}")
        return

    # Ignore tiny OCR-noise orders
    if len(keys) < MIN_FRAMES_PER_ORDER:
        logger.info(f"Ignoring order with insufficient frames: order_id={order_id}, frames={len(keys)}, min_required={MIN_FRAMES_PER_ORDER}")
        return

    logger.info(f"[ORDER-FINALIZE] Processing order_id={order_id} with {len(keys)} frames")

    scored = []
    for key in keys:
        logger.debug(f"[ORDER-FINALIZE] Loading and scoring frame: {key}")
        img = load_image(key)
        if img is None:
            logger.warning(f"[ORDER-FINALIZE] Failed to load image: {key}")
            continue
        items = count_items(img)

        # Skip frames containing person/hand etc.
        if items < 0:
            logger.debug(f"[ORDER-FINALIZE] Skipping frame with skip-labels: {key}")
            continue

        scored.append((items, key, img))

    if not scored:
        logger.warning(f"[ORDER-FINALIZE] No valid frames after filtering for order_id={order_id}")
        return

    # Pick TOP_K best frames
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    topk = scored[:min(TOP_K, len(scored))]
    
    logger.info(f"[ORDER-FINALIZE] Selected {len(topk)} top frames for order_id={order_id}")

    for rank, (items, key, frame) in enumerate(topk, 1):
        # Save to: {station_id}/{order_id}/rank_{rank}.jpg
        out_key = f"{STATION_ID}/{order_id}/rank_{rank}.jpg"
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            logger.error(f"[ORDER-FINALIZE] Failed to encode frame: {out_key}")
            continue

        client.put_object(
            SELECTED_BUCKET,
            out_key,
            io.BytesIO(buf.tobytes()),
            len(buf),
            content_type="image/jpeg",
        )

        logger.info(f"[ORDER-FINALIZE] Saved frame: {out_key} (items={items})")

    # Skip dummy order
    if order_id == "000":
        logger.debug("Skipping VLM call for dummy order 000")
        return

    logger.info(f"[ORDER-FINALIZE] Calling VLM service for order_id={order_id}")

    try:
        response = call_vlm(order_id)
        logger.info(f"[ORDER-FINALIZE] VLM call successful for order_id={order_id}")
        logger.debug(f"[ORDER-FINALIZE] VLM response: {response}")
        processed_orders.add(order_id)
        logger.info(f"[ORDER-FINALIZE] Processed orders so far: {processed_orders}")
    except Exception as e:
        logger.error(f"[ORDER-FINALIZE] VLM call failed for order_id={order_id}: {e}", exc_info=True)


# =====================================================
# MAIN LOOP
# =====================================================

if __name__ == "__main__":
    ensure_buckets()
    logger.info("=" * 60)
    logger.info("Frame selector service started")
    logger.info("Watching for frames in MinIO...")
    logger.info("=" * 60)

    processed_keys = set()

    current_order = None
    current_keys = []

    # For debouncing new order detection
    pending_order = None
    pending_count = 0

    poll_count = 0
    while True:
        try:
            frames, eos_seen = list_frames_sorted()
        except S3Error as e:
            logger.error(f"MinIO list error: {e}")
            time.sleep(POLL_INTERVAL)
            continue

        poll_count += 1
        if poll_count % 10 == 0:  # Log every 10 polls
            logger.debug(f"Polling iteration {poll_count}: {len(frames)} total frames, eos_seen={eos_seen}")

        for order_id, key in frames:
            if key in processed_keys:
                continue

            # First frame ever
            if current_order is None:
                logger.info(f"[MAIN-LOOP] First order detected: order_id={order_id}")
                current_order = order_id

            # Same order → normal collection
            if order_id == current_order:
                pending_order = None
                pending_count = 0

            # Potential new order
            else:
                if pending_order == order_id:
                    pending_count += 1
                else:
                    logger.debug(f"[MAIN-LOOP] New order candidate: {order_id}")
                    pending_order = order_id
                    pending_count = 1

                # New order not stable yet → ignore this frame
                if pending_count < MIN_FRAMES_PER_ORDER:
                    logger.debug(f"[MAIN-LOOP] Pending new order: {order_id} ({pending_count}/{MIN_FRAMES_PER_ORDER})")
                    processed_keys.add(key)
                    continue

                # New order confirmed stable → close current
                logger.info(f"[MAIN-LOOP] New order confirmed: {order_id}. Closing current order: {current_order}")
                process_completed_order(current_order, current_keys)

                current_order = order_id
                current_keys = []
                pending_order = None
                pending_count = 0

            # Collect frame into current order
            current_keys.append(key)
            processed_keys.add(key)

            logger.debug(f"[MAIN-LOOP] Collected frame: {key} (order_id={order_id}, total_frames={len(current_keys)})")

        # End-of-stream handling
        if eos_seen:
            if current_order and current_keys:
                logger.info(f"[MAIN-LOOP] EOS detected. Closing final order: {current_order}")
                process_completed_order(current_order, current_keys)
                current_order = None
                current_keys = []
            
            # Always delete EOS marker to prevent infinite loop
            try:
                client.remove_object(FRAMES_BUCKET, f"{STATION_ID}/__EOS__")
                logger.info(f"[MAIN-LOOP] EOS marker deleted for station {STATION_ID}")
            except Exception as e:
                logger.error(f"[MAIN-LOOP] Failed to delete EOS marker: {e}")
            
            # Clear state for next video
            logger.info(f"[MAIN-LOOP] Clearing state for next video: processed_keys={len(processed_keys)}, processed_orders={processed_orders}")
            processed_keys.clear()
            processed_orders.clear()
            pending_order = None
            pending_count = 0

        time.sleep(POLL_INTERVAL)