import os
import io
import time
import json
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
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Set, Optional

# RabbitMQ imports
try:
    import pika
    from pika.exceptions import AMQPConnectionError, AMQPChannelError
    PIKA_AVAILABLE = True
except ImportError:
    PIKA_AVAILABLE = False
    logging.warning("pika not installed, RabbitMQ disabled")

# Configure logging (initial setup - will be reconfigured after YOLO import)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Multi-station mode: single frame-selector handles ALL stations
# Set to False to use legacy per-station scaling mode
MULTI_STATION_MODE = os.environ.get('MULTI_STATION_MODE', 'true').lower() == 'true'

# Number of stations to monitor (only used in multi-station mode)
# Default to 1 if WORKERS is 0 or not set
NUM_STATIONS = max(1, int(os.environ.get('WORKERS', '1')))

# Legacy single-station mode: auto-detect STATION_ID from hostname
if not MULTI_STATION_MODE:
    if 'STATION_ID' not in os.environ:
        hostname = socket.gethostname()
        if hostname and hostname.split('-')[-1].isdigit():
            station_num = hostname.split('-')[-1]
            os.environ['STATION_ID'] = f'station_{station_num}'
            logger.info(f"Auto-detected STATION_ID from hostname: {os.environ['STATION_ID']}")
        else:
            os.environ['STATION_ID'] = 'station_1'
            logger.info(f"Using default STATION_ID: station_1")
    STATION_IDS = [os.environ.get('STATION_ID', 'station_1')]
else:
    # Multi-station mode: handle all stations
    STATION_IDS = [f'station_{i+1}' for i in range(NUM_STATIONS)]
    logger.info(f"Multi-station mode enabled: handling {NUM_STATIONS} stations: {STATION_IDS}")

cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]
FS_CFG = cfg["frame_selector"]
VLM_CFG = cfg["vlm"]

MINIO_ENDPOINT = MINIO["endpoint"]
FRAMES_BUCKET = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]

# Debug directory for saving received frames
DEBUG_FRAME_DIR = os.environ.get('DEBUG_FRAME_DIR', '/app/debug/frame-selector-in')
DEBUG_FRAMES_ENABLED = os.environ.get('DEBUG_FRAMES_ENABLED', 'true').lower() == 'true'

# Reconfigure logging after all imports (YOLO overwrites logging config)
# This must be AFTER ultralytics import
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force reconfiguration
)
logger = logging.getLogger(__name__)

TOP_K = FS_CFG.get("top_k", 3)  # Default to 5 if not specified
POLL_INTERVAL = FS_CFG.get("poll_interval_sec", 1.5)
# In a take-away packing scenario the worker is always present while placing
# items — every useful food frame has `person` detected.  Including 'person'
# causes ALL food-placement frames to be dropped, leaving only empty slip frames
# for the VLM.  Keep only 'hand' as a placeholder (COCO has no hand class;
# override via config 'skip_labels' if a dedicated hand-detection model is used).
SKIP_LABELS = set(FS_CFG.get("skip_labels", ["hand"]))
# YOLO classes that indicate non-food objects (order slips, phones, etc.).
# These are silently excluded from the item score so slip frames don't appear
# falsely rich — but unlike SKIP_LABELS they don't DROP the whole frame
# (a frame can contain both a slip and a real food item).
# YOLO commonly misidentifies order slips as 'remote' or 'cell phone'.
NON_FOOD_LABELS = set(FS_CFG.get("non_food_labels", ["remote", "cell phone", "book", "laptop"]))
MAX_BUCKET_SIZE = int(FS_CFG.get("max_bucket_size", 100))
# Minimum frames that must have YOLO item detections before invoking VLM.
# Buckets collected at a video-loop boundary (e.g. only 4-6 frames) often
# lack enough item-bearing frames for reliable detection — skip and wait
# for the next full cycle.
MIN_ITEM_FRAMES = int(FS_CFG.get("min_item_frames", 3))

# Confidence threshold for detecting skip-labels (e.g. a dedicated hand model).
SKIP_LABEL_CONF = float(os.environ.get("SKIP_LABEL_CONF",
                        str(FS_CFG.get("skip_label_conf", 0.1))))

# Confidence threshold for counting valid food items.
# Higher than SKIP_LABEL_CONF to suppress YOLO false-positives
# (e.g. 'clock', 'dining table', 'remote') that inflate frame scores.
ITEM_COUNT_CONF = float(os.environ.get("ITEM_COUNT_CONF",
                        str(FS_CFG.get("item_count_conf", 0.35))))

# How many consecutive frames required to confirm a new order
MIN_FRAMES_PER_ORDER = FS_CFG.get("min_frames_per_order", 2)

# Minimum frames before allowing finalization (prevents early finalization)
MIN_FRAMES_BEFORE_FINALIZE = FS_CFG.get("min_frames_before_finalize", 5)

# Inactivity timeout - finalize current order if no new frames after this many seconds
# This handles the case where the video ends (last order has no subsequent order)
INACTIVITY_TIMEOUT = FS_CFG.get("inactivity_timeout_sec", 15)

# Cooldown between re-processing the same order_id (seconds).
# Prevents duplicate VLM calls within the same video loop while allowing
# the order to be re-processed on the next loop iteration.
REPROCESS_COOLDOWN_SEC = int(os.environ.get('REPROCESS_COOLDOWN_SEC', '30'))

VLM_ENDPOINT = VLM_CFG["endpoint"]

# ── Order-aware diversity selection ──────────────────────────────────────────
# When selecting the best frame from each temporal bucket we also consider
# whether the frame contains expected items (even at lower YOLO confidence).
# This prevents situations where banana/apple detections at conf < ITEM_COUNT_CONF
# (0.35) are ignored during scoring, causing the VLM to never see those items.

def _load_orders_config():
    """Load expected orders from config/orders.json for diversity scoring."""
    try:
        config_dir = os.path.join(os.path.dirname(__file__), '..', 'config')
        orders_path = os.path.join(config_dir, 'orders.json')
        with open(orders_path, 'r') as f:
            data = json.load(f)
        result = {k: [item['name'].lower() for item in v] for k, v in data.items()}
        logger.info(f"[DIVERSITY] Loaded orders.json for diversity selection: {len(result)} orders")
        return result
    except Exception as e:
        logger.warning(f"[DIVERSITY] Could not load orders.json: {e} — diversity selection disabled")
        return {}

EXPECTED_ORDER_ITEMS = _load_orders_config()  # {order_id: [item_name_lower, ...]}

# Mapping from order item names → YOLO COCO class synonyms.
# Used to check if a frame contains an expected item at low confidence.
_ORDER_ITEM_YOLO_MAP = {
    "apple":               ["apple"],
    "green apple":         ["apple"],
    "banana":              ["banana"],
    "yellow banana":       ["banana"],
    "water bottle":        ["bottle"],
    "water":               ["bottle"],
    "coke bottle":         ["bottle"],
    "coke large bottle":   ["bottle"],
    "coke 2 liter bottle": ["bottle"],
    "pepsi can":           ["cup", "can", "bottle"],  # YOLO labels cans as 'bottle' more often than 'can'
}


def _get_expected_yolo_synonyms(order_id: str) -> set:
    """Return YOLO class names that correspond to expected items in this order."""
    synonyms = set()
    for item_name in EXPECTED_ORDER_ITEMS.get(order_id, []):
        synonyms.update(_ORDER_ITEM_YOLO_MAP.get(item_name, []))
    return synonyms
# ─────────────────────────────────────────────────────────────────────────────

# Per-station state tracking for multi-station mode
@dataclass
class StationState:
    """State tracking for a single station"""
    station_id: str
    current_order: Optional[str] = None
    current_keys: List[str] = field(default_factory=list)
    pending_order: Optional[str] = None
    pending_count: int = 0
    pending_keys: List[str] = field(default_factory=list)  # frames seen for the next‐order candidate
    processed_keys: Set[str] = field(default_factory=set)
    # Maps order_id → wall-clock of last VLM call (replaces the old Set).
    # Allows re-processing on the next video loop while preventing duplicate
    # calls within the same loop (guarded by REPROCESS_COOLDOWN_SEC).
    processed_order_times: Dict[str, float] = field(default_factory=dict)
    # Maps order_id → how many times it has been processed this session.
    processed_order_counts: Dict[str, int] = field(default_factory=dict)
    last_frame_time: float = field(default_factory=time.time)

    @property
    def processed_orders(self) -> set:
        """Backward-compatible view of all orders processed at least once."""
        return set(self.processed_order_times.keys())

    def reset(self):
        """Reset state for next video/stream"""
        self.current_order = None
        self.current_keys = []
        self.pending_order = None
        self.pending_count = 0
        self.pending_keys = []
        self.processed_keys.clear()
        self.processed_order_times.clear()
        self.processed_order_counts.clear()
        self.last_frame_time = time.time()

# Initialize per-station state
station_states: Dict[str, StationState] = {
    station_id: StationState(station_id=station_id)
    for station_id in STATION_IDS
}

# Legacy global state (for backwards compatibility)
processed_orders = set()
STATION_ID = STATION_IDS[0] if STATION_IDS else 'station_1'

# RabbitMQ configuration
RABBITMQ_HOST = os.environ.get('RABBITMQ_HOST', 'rabbitmq')
RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', '5672'))
RABBITMQ_USER = os.environ.get('RABBITMQ_USER', 'guest')
RABBITMQ_PASS = os.environ.get('RABBITMQ_PASS', 'guest')
USE_RABBITMQ = os.environ.get('USE_RABBITMQ', 'true').lower() == 'true' and PIKA_AVAILABLE
ORDER_QUEUE = 'order_processing'

logger.info(f"Frame selector configuration: stations={STATION_IDS}, top_k={TOP_K}, poll_interval={POLL_INTERVAL}s, min_frames={MIN_FRAMES_PER_ORDER}, min_before_finalize={MIN_FRAMES_BEFORE_FINALIZE}")
logger.info(f"Skip labels: {SKIP_LABELS}")
logger.info(f"Non-food labels (slip exclusion): {NON_FOOD_LABELS}")
logger.info(f"Max bucket size: {MAX_BUCKET_SIZE}, min item frames: {MIN_ITEM_FRAMES}")
logger.info(f"VLM endpoint: {VLM_ENDPOINT}")
logger.info(f"RabbitMQ enabled: {USE_RABBITMQ}")


# =====================================================
# RabbitMQ Producer
# =====================================================

@dataclass
class OrderMessage:
    """Message structure for order processing requests"""
    order_id: str
    station_id: str
    retry_count: int = 0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()
    
    def to_json(self) -> str:
        return json.dumps(asdict(self))


class RabbitMQProducer:
    """Simple producer for publishing order processing requests"""
    
    def __init__(self):
        self._connection = None
        self._channel = None
    
    def connect(self) -> bool:
        """Establish connection to RabbitMQ"""
        if self._connection and self._connection.is_open:
            return True
        
        try:
            credentials = pika.PlainCredentials(RABBITMQ_USER, RABBITMQ_PASS)
            parameters = pika.ConnectionParameters(
                host=RABBITMQ_HOST,
                port=RABBITMQ_PORT,
                credentials=credentials,
                heartbeat=600,
                blocked_connection_timeout=300,
            )
            
            logger.info(f"[QUEUE] Connecting to RabbitMQ at {RABBITMQ_HOST}:{RABBITMQ_PORT}")
            self._connection = pika.BlockingConnection(parameters)
            self._channel = self._connection.channel()
            
            # Declare transient, auto-delete queue
            self._channel.queue_declare(
                queue=ORDER_QUEUE,
                durable=False,
                auto_delete=True,
                arguments={'x-message-ttl': 300000}  # 5 min TTL
            )
            
            logger.info(f"[QUEUE] Connected to RabbitMQ, queue '{ORDER_QUEUE}' ready")
            return True
            
        except (AMQPConnectionError, AMQPChannelError) as e:
            logger.error(f"[QUEUE] Failed to connect to RabbitMQ: {e}")
            self._connection = None
            self._channel = None
            return False
    
    def publish(self, order_id: str, station_id: str) -> bool:
        """Publish order for processing"""
        message = OrderMessage(order_id=order_id, station_id=station_id)
        
        for attempt in range(3):
            try:
                if not self.connect():
                    time.sleep(1)
                    continue
                
                self._channel.basic_publish(
                    exchange='',
                    routing_key=ORDER_QUEUE,
                    body=message.to_json(),
                    properties=pika.BasicProperties(
                        delivery_mode=1,  # Transient
                        content_type='application/json',
                    )
                )
                
                logger.info(f"[QUEUE] Published order: order_id={order_id}, station_id={station_id}")
                return True
                
            except (AMQPConnectionError, AMQPChannelError) as e:
                logger.warning(f"[QUEUE] Publish attempt {attempt+1} failed: {e}")
                self._connection = None
                self._channel = None
                time.sleep(0.5)
        
        logger.error(f"[QUEUE] Failed to publish order after 3 attempts: order_id={order_id}")
        return False
    
    def close(self):
        """Close connection"""
        if self._connection and self._connection.is_open:
            self._connection.close()
        self._connection = None
        self._channel = None


# Global producer instance
_producer = None

def get_producer():
    global _producer
    if _producer is None and USE_RABBITMQ:
        _producer = RabbitMQProducer()
    return _producer


# =====================================================
# VLM Caller (with RabbitMQ support)
# =====================================================

def call_vlm(order_id, station_id=None, timeout=400):
    """
    Submit order for VLM processing.
    Uses RabbitMQ queue if available, falls back to direct HTTP call.
    """
    logger.info(f"[VLM-CALL] Submitting order for processing: order_id={order_id}, station_id={station_id}")
    
    # Try RabbitMQ first (non-blocking, guaranteed delivery)
    if USE_RABBITMQ:
        producer = get_producer()
        if producer and producer.publish(order_id, station_id or 'station_1'):
            logger.info(f"[VLM-CALL] Order queued successfully: order_id={order_id}")
            return {"status": "queued", "order_id": order_id}
        else:
            logger.warning(f"[VLM-CALL] Queue publish failed, falling back to HTTP")
    
    # Fallback to direct HTTP call (blocking)
    logger.info(f"[VLM-CALL] Using direct HTTP call for order_id={order_id}")
    payload = {"order_id": order_id}
    if station_id:
        payload["station_id"] = station_id
    
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

# Initialize debug directory
if DEBUG_FRAMES_ENABLED:
    Path(DEBUG_FRAME_DIR).mkdir(parents=True, exist_ok=True)
    logger.info(f"Debug frames enabled: saving to {DEBUG_FRAME_DIR}")


def save_debug_frame(station_id: str, order_id: str, key: str, img: np.ndarray):
    """Save frame to debug directory for debugging frame flow."""
    if not DEBUG_FRAMES_ENABLED:
        return
    
    try:
        # Create folder: /app/debug/frame-selector-in/{station_id}/{order_id}/
        debug_folder = Path(DEBUG_FRAME_DIR) / station_id / order_id
        debug_folder.mkdir(parents=True, exist_ok=True)
        
        # Extract frame name from key
        frame_name = key.split('/')[-1]  # e.g., frame_1.jpg
        debug_path = debug_folder / frame_name
        
        cv2.imwrite(str(debug_path), img)
        logger.debug(f"[DEBUG] Saved frame to: {debug_path}")
    except Exception as e:
        logger.warning(f"[DEBUG] Failed to save debug frame: {e}")


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


def list_frames_sorted(station_id: str):
    """List frames for a specific station, sorted by name.

    Returns:
        frames              : list of (order_id, object_name) sorted by name
        eos_seen            : True if the station-level __EOS__ marker exists
        finalized_order_ids : set of order_ids that have a per-order __EOS__ marker
                              (written by frame_pipeline on each order transition)
    """
    logger.debug(f"Listing frames from bucket: {FRAMES_BUCKET} (station: {station_id})")
    frames = []
    eos_seen = False
    finalized_order_ids = set()  # orders whose upload is complete

    for obj in client.list_objects(FRAMES_BUCKET, recursive=True):
        # Station-level EOS — full end-of-video cleanup signal
        if obj.object_name == f"{station_id}/__EOS__":
            eos_seen = True
            logger.info(f"EOS marker detected for station {station_id}")
            continue

        # Only process objects for this station
        if not obj.object_name.startswith(f"{station_id}/"):
            continue

        parts = obj.object_name.split("/")

        # Per-order EOS written by frame_pipeline: station_id/order_id/__EOS__
        if len(parts) == 3 and parts[2] == "__EOS__":
            order_id = parts[1]
            if order_id.lower() != "pending":
                finalized_order_ids.add(order_id)
                logger.info(f"[{station_id}] Per-order EOS detected: order_id={order_id}")
            continue

        if obj.object_name.lower().endswith(".jpg"):
            # Extract: station_id/order_id/frame_X.jpg -> order_id
            if len(parts) >= 3 and parts[0] == station_id:
                order_id = parts[1]
                # Skip "pending" folder - these are frames awaiting OCR, not real orders
                if order_id.lower() == "pending":
                    continue
                frames.append((order_id, obj.object_name))

    frames.sort(key=lambda x: x[1])
    logger.debug(f"Found {len(frames)} frames for station {station_id}, eos_seen={eos_seen}, "
                 f"finalized_orders={finalized_order_ids}")
    return frames, eos_seen, finalized_order_ids


def load_image(key):
    resp = None
    try:
        resp = client.get_object(FRAMES_BUCKET, key)
        data = resp.read()
        return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        # Key may have been overwritten or deleted between scan and load (race condition).
        # Log a warning and return None so the caller skips this frame gracefully.
        logger.warning(f"[load_image] Could not fetch {key}: {e}")
        return None
    finally:
        if resp is not None:
            resp.close()
            resp.release_conn()


def count_items(frame, frame_key="?"):
    """Score a frame by YOLO detection.

    Returns:
        (count, detected_label_set)
            count              : number of high-confidence items (>= ITEM_COUNT_CONF),
                                 or -1 if a skip-label was detected.
            detected_label_set : set of ALL YOLO class names seen at >= SKIP_LABEL_CONF
                                 (used for diversity scoring even at lower confidence).
    """
    # Run YOLO at the skip-label threshold.  Skip-labels default to ["hand"] only;
    # 'person' is intentionally excluded because the packing worker is always
    # visible in food-placement frames and must not cause them to be dropped.
    result = model(frame, conf=SKIP_LABEL_CONF, verbose=False)[0]

    # Collect ALL detected labels (including low-conf) for diversity scoring.
    detected_label_set = set()

    # Log every raw detection for diagnosis
    all_detections = []
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "unknown").lower()
        conf_val  = float(box.conf)
        all_detections.append(f"{cls_name}({conf_val:.2f})")
        detected_label_set.add(cls_name)
    logger.info(f"[YOLO][{frame_key}] Detections (conf>={SKIP_LABEL_CONF}): "
                f"{all_detections if all_detections else 'none'}")

    # If ANY skip-label is present → discard frame
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        if cls_name in SKIP_LABELS:
            logger.info(f"[YOLO][{frame_key}] DROPPED — skip label '{cls_name}' "
                        f"detected at conf={float(box.conf):.2f}")
            return -1, set()   # mark frame as invalid

    # Count valid objects only above the higher item-count threshold.
    # This suppresses false-positive classes (clock, remote, dining table etc.)
    # that YOLO hallucinates on food items at low confidence.
    count = 0
    valid_items = []
    low_conf_ignored = []
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        conf_val  = float(box.conf)
        if cls_name in SKIP_LABELS:
            continue   # already checked above
        if cls_name in NON_FOOD_LABELS:
            low_conf_ignored.append(f"{cls_name}(slip/{conf_val:.2f})")
            continue   # order-slip misdetection — ignore silently
        if conf_val >= ITEM_COUNT_CONF:
            count += 1
            valid_items.append(f"{cls_name}({conf_val:.2f})")
        else:
            low_conf_ignored.append(f"{cls_name}({conf_val:.2f})")

    if low_conf_ignored:
        logger.info(f"[YOLO][{frame_key}] Ignored low-conf items "
                    f"(conf<{ITEM_COUNT_CONF}): {low_conf_ignored}")
    logger.info(f"[YOLO][{frame_key}] ACCEPTED — {count} valid item(s): {valid_items}")
    return count, detected_label_set



# =====================================================
# Order Finalization
# =====================================================

def process_completed_order(station_id: str, order_id: str, keys: List[str], state: StationState):
    """Process a completed order for a specific station"""
    if not keys:
        logger.debug(f"[{station_id}] Skipping empty order: order_id={order_id}")
        return

    # Prevent duplicate VLM calls within the same video loop using a cooldown.
    # If the order was processed less than REPROCESS_COOLDOWN_SEC ago, skip it.
    # Once the cooldown expires (next video loop), it will be re-processed.
    last_processed = state.processed_order_times.get(order_id, 0)
    elapsed_since = time.time() - last_processed
    if last_processed > 0 and elapsed_since < REPROCESS_COOLDOWN_SEC:
        logger.warning(
            f"[{station_id}] Order {order_id} processed {elapsed_since:.0f}s ago "
            f"(cooldown={REPROCESS_COOLDOWN_SEC}s) — skipping duplicate call"
        )
        return
    run_number = state.processed_order_counts.get(order_id, 0) + 1
    logger.info(f"[{station_id}] Processing order_id={order_id} — run #{run_number}")

    # Ignore tiny OCR-noise orders
    if len(keys) < MIN_FRAMES_PER_ORDER:
        logger.info(f"[{station_id}] Ignoring order with insufficient frames: order_id={order_id}, frames={len(keys)}, min_required={MIN_FRAMES_PER_ORDER}")
        return

    logger.info(f"[{station_id}][ORDER-FINALIZE] Processing order_id={order_id} with {len(keys)} frames")
    
    # CRITICAL: Clean up any existing selected frames for this order before saving new ones
    # This prevents stale frames from previous video loops affecting VLM detection
    order_prefix = f"{station_id}/{order_id}/"
    try:
        stale_count = 0
        for obj in client.list_objects(SELECTED_BUCKET, prefix=order_prefix, recursive=True):
            try:
                client.remove_object(SELECTED_BUCKET, obj.object_name)
                stale_count += 1
            except Exception:
                pass
        if stale_count > 0:
            logger.info(f"[{station_id}][ORDER-FINALIZE] Cleaned {stale_count} stale frames from SELECTED_BUCKET for order_id={order_id}")
    except Exception as e:
        logger.warning(f"[{station_id}][ORDER-FINALIZE] Failed to cleanup stale frames for order_id={order_id}: {e}")

    scored = []

    logger.info(f"[{station_id}][ORDER-FINALIZE] Processing {len(keys)} frames for order_id={order_id}")
    logger.info(f"[{station_id}][ORDER-FINALIZE] Frame keys: {keys}")
    
    dropped_keys  = []
    accepted_keys = []

    for key in keys:
        img = load_image(key)
        if img is None:
            logger.warning(f"[{station_id}][ORDER-FINALIZE] Failed to load image: {key}")
            continue

        # Save debug frame
        save_debug_frame(station_id, order_id, key, img)

        frame_label = key.split('/')[-1]   # e.g. frame_12.jpg
        items, detected_labels = count_items(img, frame_key=f"{order_id}/{frame_label}")

        # Skip frames containing a detected skip-label (e.g. dedicated hand model).
        # (Option D: never use skip-label frames as fallback — always drop them
        #  to keep VLM input clean, matching the original design's accuracy guarantee.)
        if items < 0:
            dropped_keys.append(key)
            continue

        # detected_labels is used below for order-aware diversity scoring.
        scored.append((items, detected_labels, key, img))
        accepted_keys.append((items, key))

    logger.info(f"[{station_id}][SCORE-SUMMARY] order={order_id} | "
                f"total={len(keys)} | accepted={len(accepted_keys)} | dropped={len(dropped_keys)}")
    logger.info(f"[{station_id}][SCORE-SUMMARY] Accepted scores: "
                f"{[(k.split('/')[-1], s) for s, k in accepted_keys]}")
    if dropped_keys:
        logger.info(f"[{station_id}][SCORE-SUMMARY] Dropped (hand/person): "
                    f"{[k.split('/')[-1] for k in dropped_keys]}")

    if not scored:
        logger.warning(f"[{station_id}][ORDER-FINALIZE] No valid frames after filtering for order_id={order_id}")
        return

    item_bearing = sum(1 for s, _, _, _ in scored if s > 0)
    if item_bearing < MIN_ITEM_FRAMES:
        logger.warning(
            f"[{station_id}][ORDER-FINALIZE] Skipping VLM for order_id={order_id} — "
            f"only {item_bearing}/{len(scored)} frames have item detections "
            f"(need {MIN_ITEM_FRAMES}). Likely a partial cycle; next full loop will have more frames."
        )
        return

    # ── Order-aware temporal-spread frame selection ───────────────────────────
    # Primary goal: ensure the selected TOP_K frames collectively cover as many
    # expected order items as possible — even items YOLO detects at low confidence
    # (below ITEM_COUNT_CONF) such as bananas at 0.13–0.24 conf.
    #
    # Algorithm:
    #   1. Within each temporal bucket pick the frame with the highest
    #      (expected_item_coverage, high_conf_item_count) composite score.
    #   2. After temporal selection, if any expected items are still uncovered,
    #      swap the lowest-value selected frame for the best uncovered-item frame
    #      (forced coverage pass).
    #
    # Fallback: if fewer accepted frames than buckets, fall back to score-only
    # sort (avoids wasting slots on empty buckets).
    # ─────────────────────────────────────────────────────────────────────────

    # YOLO class synonyms for expected order items (used for diversity scoring).
    expected_yolo = _get_expected_yolo_synonyms(order_id)
    if expected_yolo:
        logger.info(f"[{station_id}][DIVERSITY] order={order_id} — expected YOLO synonyms: {expected_yolo}")

    def _diversity(detected_labels: set) -> int:
        """Number of expected YOLO synonyms visible in this frame."""
        return len(expected_yolo & detected_labels) if expected_yolo else 0

    # scored is currently in insertion order (same as keys order = frame number order)
    # Keep temporal order for bucket splitting.
    n = len(scored)
    topk = []

    if n >= TOP_K:
        # Split into TOP_K equal buckets and pick best from each
        bucket_size = n / TOP_K
        for b in range(TOP_K):
            start = int(b * bucket_size)
            end   = int((b + 1) * bucket_size)
            bucket = scored[start:end]
            if not bucket:
                continue
            # Best frame: primary = expected item coverage (diversity),
            # secondary = total high-conf item count, tertiary = latest key.
            best = max(bucket, key=lambda x: (_diversity(x[1]), x[0], x[2]))
            topk.append(best)
            logger.info(f"[{station_id}][TOP-K] bucket_{b+1} "
                        f"[frames {start}–{end-1} of {n}]: "
                        f"chose {best[2].split('/')[-1]} "
                        f"(yolo_items={best[0]}, diversity={_diversity(best[1])})")
    else:
        # Fewer frames than buckets — fall back to diversity+score sort
        logger.info(f"[{station_id}][TOP-K] only {n} accepted frame(s) < TOP_K={TOP_K}, "
                    f"using diversity+score fallback")
        scored_sorted = sorted(scored, key=lambda x: (_diversity(x[1]), x[0], x[2]), reverse=True)
        topk = scored_sorted[:n]

    # ── Forced coverage pass ─────────────────────────────────────────────────
    # After temporal selection, check if any expected items are still uncovered.
    # If so, replace the lowest-value selected frame with the best alternative
    # frame that covers the missing item.
    if expected_yolo:
        covered = set()
        for items_cnt, det_labels, _key, _frame in topk:
            covered |= (expected_yolo & det_labels)
        missing = expected_yolo - covered
        if missing:
            logger.info(f"[{station_id}][DIVERSITY] order={order_id} — uncovered synonyms after bucket selection: {missing}")
            # Find the best alternative frame for each missing synonym
            for missing_label in missing:
                best_alt = None
                for entry in scored:
                    if entry in topk:
                        continue
                    if missing_label in entry[1]:   # entry[1] = detected_labels
                        if best_alt is None or entry[0] > best_alt[0]:
                            best_alt = entry
                if best_alt is not None:
                    # Replace the lowest-score topk entry that doesn't cover this label
                    replaceable = [
                        (i, e) for i, e in enumerate(topk)
                        if missing_label not in e[1]   # e[1] = detected_labels
                    ]
                    if replaceable:
                        worst_idx = min(replaceable, key=lambda x: (x[1][0], x[1][2]))[0]
                        old_key = topk[worst_idx][2].split('/')[-1]
                        new_key = best_alt[2].split('/')[-1]
                        topk[worst_idx] = best_alt
                        logger.info(f"[{station_id}][DIVERSITY] Swapped {old_key} → {new_key} "
                                    f"to cover '{missing_label}'")

    logger.info(f"[{station_id}][TOP-K] order={order_id} — selected {len(topk)} frame(s):")
    for rank, (items, _labels, key, _) in enumerate(topk, 1):
        logger.info(f"[{station_id}][TOP-K]   rank_{rank} → {key.split('/')[-1]}  "
                    f"(yolo_items={items}, diversity={_diversity(_labels)})")

    for rank, (items, _labels, key, frame) in enumerate(topk, 1):
        # Save to: {station_id}/{order_id}/rank_{rank}.jpg
        out_key = f"{station_id}/{order_id}/rank_{rank}.jpg"
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            logger.error(f"[{station_id}][ORDER-FINALIZE] Failed to encode frame: {out_key}")
            continue

        client.put_object(
            SELECTED_BUCKET,
            out_key,
            io.BytesIO(buf.tobytes()),
            len(buf),
            content_type="image/jpeg",
        )

        logger.info(f"[{station_id}][ORDER-FINALIZE] Saved frame: {out_key} (items={items})")

    # Skip dummy order
    if order_id == "000":
        logger.debug(f"[{station_id}] Skipping VLM call for dummy order 000")
        return

    logger.info(f"[{station_id}][ORDER-FINALIZE] Calling VLM service for order_id={order_id}")

    try:
        response = call_vlm(order_id, station_id=station_id)
        logger.info(f"[{station_id}][ORDER-FINALIZE] VLM call successful for order_id={order_id} (run #{run_number})")
        logger.debug(f"[{station_id}][ORDER-FINALIZE] VLM response: {response}")
        state.processed_order_times[order_id] = time.time()
        state.processed_order_counts[order_id] = run_number
        logger.info(
            f"[{station_id}][ORDER-FINALIZE] Order {order_id} processed {run_number} time(s). "
            f"All processed: { {k: v for k, v in state.processed_order_counts.items()} }"
        )
    except Exception as e:
        logger.error(f"[{station_id}][ORDER-FINALIZE] VLM call failed for order_id={order_id}: {e}", exc_info=True)


def _cleanup_order(station_id: str, order_id: str, state: StationState) -> None:
    """Delete all frames and the per-order EOS marker for a completed order from
    FRAMES_BUCKET.  This prevents second-loop (video looping) frames from the
    same order_id accumulating in the same MinIO prefix and inflating counts.
    """
    deleted = 0
    # Remove frames already tracked in processed_keys
    keys_to_remove = [k for k in list(state.processed_keys)
                      if k.startswith(f"{station_id}/{order_id}/")]
    for key in keys_to_remove:
        try:
            client.remove_object(FRAMES_BUCKET, key)
            state.processed_keys.discard(key)
            deleted += 1
        except Exception as e:
            logger.debug(f"[{station_id}] Failed to delete tracked frame {key}: {e}")
    # Sweep MinIO for any frames that weren't yet in processed_keys
    try:
        for obj in client.list_objects(FRAMES_BUCKET,
                                        prefix=f"{station_id}/{order_id}/",
                                        recursive=True):
            if obj.object_name in state.processed_keys:
                continue
            try:
                client.remove_object(FRAMES_BUCKET, obj.object_name)
                deleted += 1
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[{station_id}] Error sweeping MinIO for order {order_id}: {e}")
    # Delete the per-order EOS marker itself
    try:
        client.remove_object(FRAMES_BUCKET, f"{station_id}/{order_id}/__EOS__")
    except Exception:
        pass
    logger.info(f"[{station_id}][CLEANUP] Removed {deleted} frames + EOS for order_id={order_id} "
                f"— next video loop will start with a clean bucket")


def process_station(station_id: str, state: StationState) -> bool:
    """
    Process frames for a single station.
    Returns True if EOS was detected for this station.
    """
    try:
        frames, eos_seen, finalized_order_ids = list_frames_sorted(station_id)
    except S3Error as e:
        logger.error(f"[{station_id}] MinIO list error: {e}")
        return False

    for order_id, key in frames:
        if key in state.processed_keys:
            continue

        # First frame ever for this station
        if state.current_order is None:
            logger.info(f"[{station_id}][MAIN-LOOP] First order detected: order_id={order_id}")
            state.current_order = order_id

        # Same order → normal collection
        if order_id == state.current_order:
            state.pending_order = None
            state.pending_count = 0

        # Potential new order
        else:
            if state.pending_order == order_id:
                state.pending_count += 1
            else:
                logger.debug(f"[{station_id}][MAIN-LOOP] New order candidate: {order_id}")
                state.pending_order = order_id
                state.pending_count = 1
                state.pending_keys = []  # reset buffer when candidate changes

            # New order not stable yet → buffer in pending_keys (NOT current_keys)
            if state.pending_count < MIN_FRAMES_PER_ORDER:
                logger.debug(f"[{station_id}][MAIN-LOOP] Pending new order: {order_id} ({state.pending_count}/{MIN_FRAMES_PER_ORDER})")
                state.pending_keys.append(key)  # hold for new order, never mix into current
                state.processed_keys.add(key)
                continue

            # New order confirmed stable → always close current order regardless of frame count.
            # (Option C fix: never silently discard an order just because it has few frames —
            #  min-frame filtering inside process_completed_order is sufficient.)
            current_frame_count = len(state.current_keys)
            logger.info(f"[{station_id}][MAIN-LOOP] New order confirmed: {order_id}. Closing current order: {state.current_order} ({current_frame_count} frames)")
            _prev_order = state.current_order
            process_completed_order(station_id, state.current_order, state.current_keys, state)
            _cleanup_order(station_id, _prev_order, state)
            state.current_order = order_id
            state.current_keys = list(state.pending_keys)  # seed new order with already-seen frames
            state.pending_keys = []
            state.pending_order = None
            state.pending_count = 0

        # Collect frame into current order (capped to avoid runaway accumulation
        # if VLM/OVMS hangs and the video loops for minutes)
        if len(state.current_keys) >= MAX_BUCKET_SIZE:
            logger.warning(f"[{station_id}][BUCKET-CAP] order={order_id} bucket full ({MAX_BUCKET_SIZE} frames) — dropping {key}")
        else:
            state.current_keys.append(key)
            state.processed_keys.add(key)
        state.last_frame_time = time.time()  # Update last frame time

        logger.debug(f"[{station_id}][MAIN-LOOP] Collected frame: {key} (order_id={order_id}, total_frames={len(state.current_keys)})")

    # ---- Per-order EOS: finalize as soon as frame_pipeline signals upload complete ----
    # frame_pipeline writes station_id/order_id/__EOS__ on every order transition.
    # Finalizing here (rather than waiting for frame-count transitions) means:
    #  • Each order is processed with only its own frames (no cross-loop contamination).
    #  • The bucket is cleaned before the next video loop writes to the same prefix.
    if state.current_order and state.current_order in finalized_order_ids:
        logger.info(f"[{station_id}][PER-ORDER-EOS] Order {state.current_order} upload complete — "
                    f"finalizing ({len(state.current_keys)} frames)")
        if state.current_keys:
            process_completed_order(station_id, state.current_order, state.current_keys, state)
        _cleanup_order(station_id, state.current_order, state)
        # Promote pending order if one is already confirmed, otherwise go idle
        if state.pending_order and state.pending_count >= MIN_FRAMES_PER_ORDER:
            logger.info(f"[{station_id}][PER-ORDER-EOS] Promoting pending order "
                        f"{state.pending_order} ({len(state.pending_keys)} buffered frames)")
            state.current_order = state.pending_order
            state.current_keys  = list(state.pending_keys)
        else:
            state.current_order = None
            state.current_keys  = []
        state.pending_order = None
        state.pending_count = 0
        state.pending_keys  = []
        state.last_frame_time = time.time()

    # Inactivity timeout handling - finalize current order if no new frames for a while
    # This handles the case where the video ends (last order has no subsequent order)
    if state.current_order and state.current_keys:
        time_since_last_frame = time.time() - state.last_frame_time
        if time_since_last_frame >= INACTIVITY_TIMEOUT:
            logger.info(f"[{station_id}][INACTIVITY] No new frames for {time_since_last_frame:.1f}s. Finalizing current order: {state.current_order}")
            _timeout_order = state.current_order
            process_completed_order(station_id, state.current_order, state.current_keys, state)
            _cleanup_order(station_id, _timeout_order, state)
            state.current_order = None
            state.current_keys = []
            state.last_frame_time = time.time()  # Reset timer

    # End-of-stream handling
    if eos_seen:
        if state.current_order and state.current_keys:
            logger.info(f"[{station_id}][MAIN-LOOP] EOS detected. Closing final order: {state.current_order}")
            process_completed_order(station_id, state.current_order, state.current_keys, state)
            state.current_order = None
            state.current_keys = []
        
        # Always delete EOS marker to prevent infinite loop
        try:
            client.remove_object(FRAMES_BUCKET, f"{station_id}/__EOS__")
            logger.info(f"[{station_id}][MAIN-LOOP] EOS marker deleted")
        except Exception as e:
            logger.error(f"[{station_id}][MAIN-LOOP] Failed to delete EOS marker: {e}")
        
        # Delete all station frames from FRAMES_BUCKET to prevent re-processing loop
        logger.info(f"[{station_id}][MAIN-LOOP] Cleaning up {len(state.processed_keys)} processed frames from MinIO")
        deleted_count = 0
        for key in list(state.processed_keys):
            try:
                client.remove_object(FRAMES_BUCKET, key)
                deleted_count += 1
            except Exception as e:
                logger.debug(f"[{station_id}][MAIN-LOOP] Failed to delete frame {key}: {e}")
        logger.info(f"[{station_id}][MAIN-LOOP] Deleted {deleted_count} frames from FRAMES_BUCKET")
        
        # CRITICAL: Also cleanup SELECTED_BUCKET to prevent stale frames affecting next video loop
        # Without this, old frames from previous order processing stay in MinIO and can cause
        # item detection mismatches when VLM picks up stale frames for the same order_id.
        logger.info(f"[{station_id}][MAIN-LOOP] Cleaning up all selected frames from SELECTED_BUCKET")
        selected_deleted = 0
        try:
            for obj in client.list_objects(SELECTED_BUCKET, prefix=f"{station_id}/", recursive=True):
                try:
                    client.remove_object(SELECTED_BUCKET, obj.object_name)
                    selected_deleted += 1
                except Exception as e:
                    logger.debug(f"[{station_id}][MAIN-LOOP] Failed to delete selected frame {obj.object_name}: {e}")
            logger.info(f"[{station_id}][MAIN-LOOP] Deleted {selected_deleted} frames from SELECTED_BUCKET")
        except Exception as e:
            logger.error(f"[{station_id}][MAIN-LOOP] Failed to cleanup SELECTED_BUCKET: {e}")
        
        # Clear state for next video
        logger.info(f"[{station_id}][MAIN-LOOP] Clearing state for next video")
        state.reset()
        
        return True  # EOS detected
    
    return False


# =====================================================
# MAIN LOOP
# =====================================================

if __name__ == "__main__":
    ensure_buckets()
    logger.info("=" * 60)
    logger.info("Frame selector service started")
    logger.info(f"Mode: {'Multi-station' if MULTI_STATION_MODE else 'Single-station'}")
    logger.info(f"Monitoring stations: {STATION_IDS}")
    logger.info("Watching for frames in MinIO...")
    logger.info("=" * 60)

    poll_count = 0
    while True:
        poll_count += 1
        
        # Process each station
        for station_id in STATION_IDS:
            state = station_states[station_id]
            try:
                process_station(station_id, state)
            except Exception as exc:
                logger.error(f"[{station_id}] Unhandled error in process_station, continuing: {exc}", exc_info=True)
        
        # Log status every 10 polls
        if poll_count % 10 == 0:
            active_stations = [sid for sid, s in station_states.items() if s.current_order or s.processed_keys]
            if active_stations:
                logger.debug(f"Polling iteration {poll_count}: Active stations: {active_stations}")

        time.sleep(POLL_INTERVAL)