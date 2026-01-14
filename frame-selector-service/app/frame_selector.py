import os
import io
import time
import cv2
import numpy as np
import asyncio
from minio import Minio
from minio.error import S3Error
from ultralytics import YOLO
import requests
from config_loader import load_config
from pathlib import Path
import shutil

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

VLM_ENDPOINT = VLM_CFG["endpoint"]
VLM_RETRIES = VLM_CFG["retries"]
VLM_TIMEOUT = VLM_CFG["timeout_sec"]



def call_vlm(order_id, timeout=120):
    payload = {"order_id": order_id}

    resp = requests.post(
        VLM_ENDPOINT,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


print("[frame-selector] Loading YOLOv11 model...", flush=True)

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

# Step 1: Download YOLOv11 model (if not exists)
if not yolo_model.exists():
    print(f"[frame-selector] Downloading {yolo_model}...", flush=True)
    model_pt = YOLO(str(yolo_model))
    print("[frame-selector] YOLOv11 model downloaded.", flush=True)

# Step 2: Convert to OpenVINO FP32 format (if not exists)
if not openvino_fp32_path.exists():
    print("[frame-selector] Converting YOLOv11 to OpenVINO FP32 format...", flush=True)
    model_pt = YOLO(str(yolo_model))
    model_pt.export(format="openvino", half=False)
    # Export already saves to /app/models/yolo11n_openvino_model/
    print("[frame-selector] OpenVINO FP32 conversion complete.", flush=True)

# Step 3: Quantize to INT8 (if not exists)
if not openvino_int8_path.exists():
    print("[frame-selector] Quantizing model to INT8...", flush=True)
    model_pt = YOLO(str(yolo_model))
    model_pt.export(format="openvino", int8=True, data="coco128.yaml")
    
    # Rename from default to int8 path
    default_output = model_dir / "yolo11n_openvino_model"
    if default_output.exists() and not openvino_int8_path.exists():
        default_output.rename(openvino_int8_path)
    
    print("[frame-selector] INT8 quantization complete.", flush=True)

# Step 4: Load the INT8 OpenVINO model
print("[frame-selector] Loading INT8 OpenVINO model...", flush=True)
model = YOLO(str(openvino_int8_path), task="detect")
print("[frame-selector] INT8 OpenVINO model loaded successfully.", flush=True)

# =====================================================
# MINIO CLIENT
# =====================================================

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO["access_key"],
    secret_key=MINIO["secret_key"],
    secure=MINIO.get("secure", False),
)

# =====================================================
# HELPERS
# =====================================================

def wait_for_bucket(bucket: str):
    while True:
        try:
            if client.bucket_exists(bucket):
                print(f"[frame-selector] Bucket ready: {bucket}", flush=True)
                return
        except Exception as e:
            print(f"[frame-selector] Bucket check error: {e}", flush=True)
        time.sleep(1)


def ensure_buckets():
    wait_for_bucket(FRAMES_BUCKET)
    if not client.bucket_exists(SELECTED_BUCKET):
        print(f"[frame-selector] Creating bucket: {SELECTED_BUCKET}", flush=True)
        client.make_bucket(SELECTED_BUCKET)


def list_frames_sorted():
    frames = []
    eos_seen = False

    for obj in client.list_objects(FRAMES_BUCKET, recursive=True):
        if obj.object_name == "__EOS__":
            eos_seen = True
            continue

        if not obj.object_name.lower().endswith(".jpg"):
            continue

        parts = obj.object_name.split("/", 1)
        if len(parts) == 2:
            frames.append((parts[0], obj.object_name))

    frames.sort(key=lambda x: x[1])
    return frames, eos_seen



def load_image(key: str):
    try:
        resp = client.get_object(FRAMES_BUCKET, key)
        data = resp.read()
        resp.close()
        resp.release_conn()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"[frame-selector] ERROR loading {key}: {e}", flush=True)
        return None


def count_items(frame) -> int:
    result = model(frame, conf=0.1, verbose=False)[0]
    count = 0
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        if cls_name not in SKIP_LABELS:
            count += 1
    return count


# =====================================================
# ORDER FINALIZATION
# =====================================================

def process_completed_order(order_id, keys):
    """
    Called EXACTLY ONCE per order.
    All frames of the order must already be present.
    """
    if not keys:
        return

    print(
        f"\n[frame-selector] Finalizing order {order_id} "
        f"with {len(keys)} frames",
        flush=True
    )

    scored = []

    for key in keys:
        img = load_image(key)
        if img is None:
            continue

        items = count_items(img)
        scored.append((items, key, img))
        print(f"[frame-selector]   {key} → items={items}", flush=True)

    if not scored:
        print(f"[frame-selector] No usable frames for order {order_id}", flush=True)
        return

    # Sort by items DESC, frame index DESC
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    topk = scored[:min(TOP_K, len(scored))]

    selected_keys = []

    for rank, (items, key, frame) in enumerate(topk, 1):
        out_key = f"{order_id}/rank_{rank}.jpg"
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        client.put_object(
            SELECTED_BUCKET,
            out_key,
            io.BytesIO(buf.tobytes()),
            len(buf),
            content_type="image/jpeg",
        )

        selected_keys.append(out_key)
        print(
            f"[frame-selector]   Saved {out_key} (items={items})",
            flush=True
        )

    # -------------------------------------------------
    # VLM CALL (SKIP DUMMY ORDER 000)
    # -------------------------------------------------

    if order_id == "000":
        print("[frame-selector] Skipping VLM for order 000", flush=True)
        return

    print(f"[frame-selector] Calling VLM for order {order_id}", flush=True)

    # try:
    #     response = asyncio.run(run_vlm(order_id, selected_keys))
    #     print("[frame-selector] VLM response:", response, flush=True)
    # except Exception as e:
    #     print("[frame-selector] VLM call failed:", e, flush=True)
    try:
        response = call_vlm(order_id)
        print("[frame-selector] VLM response:", response, flush=True)
    except Exception as e:
        print("[frame-selector] VLM call failed:", e, flush=True)



# =====================================================
# MAIN LOOP
# =====================================================

if __name__ == "__main__":
    print("[frame-selector] Starting...", flush=True)
    ensure_buckets()

    processed_keys = set()
    current_order = None
    current_keys = []

    print("[frame-selector] Watching frames...", flush=True)

    LAST_FRAME_TIME = None
    ORDER_TIMEOUT_SEC = 3.0  # tweak if needed

    while True:
        try:
            frames, eos_seen = list_frames_sorted()
        except S3Error as e:
            print("[frame-selector] MinIO list error:", e, flush=True)
            time.sleep(POLL_INTERVAL)
            continue

        for order_id, key in frames:
            if key in processed_keys:
                continue

            # First frame
            if current_order is None:
                current_order = order_id

            # 🔁 ORDER SWITCH = ORDER COMPLETE
            if order_id != current_order:
                process_completed_order(current_order, current_keys)
                current_order = order_id
                current_keys = []

            current_keys.append(key)
            processed_keys.add(key)
            LAST_FRAME_TIME = time.time()

            print(
                f"[frame-selector] Collected {key} "
                f"(order={order_id}, total={len(current_keys)})",
                flush=True
            )

            # ⏱️ FINALIZE LAST ORDER ON INACTIVITY
            now = time.time()
            if current_order and LAST_FRAME_TIME:
                if now - LAST_FRAME_TIME > ORDER_TIMEOUT_SEC:
                    print(
                        f"[frame-selector] Timeout reached. Finalizing order {current_order}",
                        flush=True
                    )
                    process_completed_order(current_order, current_keys)
                    current_order = None
                    current_keys = []
                    LAST_FRAME_TIME = None
        if eos_seen and current_order and current_keys:
            print(
                f"[frame-selector] EOS detected. Finalizing last order {current_order}",
                flush=True
            )
            process_completed_order(current_order, current_keys)
            current_order = None
            current_keys = []

        time.sleep(POLL_INTERVAL)
