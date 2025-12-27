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

# VLM_ENDPOINT = os.getenv(
#     "VLM_ENDPOINT",
#     "http://application-service:8000/run_vlm"
# )

def call_vlm(order_id, timeout=120):
    payload = {"order_id": order_id}

    resp = requests.post(
        VLM_ENDPOINT,
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()




# =====================================================
# CONFIG
# =====================================================

# MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
# MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS", "minioadmin")
# MINIO_SECRET_KEY = os.getenv("MINIO_SECRET", "minioadmin")

# FRAMES_BUCKET = os.getenv("MINIO_BUCKET", "frames")
# SELECTED_BUCKET = os.getenv("SELECTED_BUCKET", "selected")

# TOP_K = 3
# POLL_INTERVAL = 1.5  # seconds

# =====================================================
# YOLO MODEL
# =====================================================

print("[frame-selector] Loading YOLO model (yolov8n.pt)...", flush=True)
model = YOLO("yolov8n.pt")

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
    """
    Returns list of (order_id, key) sorted by frame index.
    Assumes upstream guarantees frames of one order are contiguous.
    """
    frames = []
    for obj in client.list_objects(FRAMES_BUCKET, recursive=True):
        key = obj.object_name
        if not key.lower().endswith(".jpg"):
            continue

        parts = key.split("/", 1)
        if len(parts) != 2:
            continue

        frames.append((parts[0], key))

    frames.sort(key=lambda x: x[1])
    return frames


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

    while True:
        try:
            frames = list_frames_sorted()
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

            print(
                f"[frame-selector] Collected {key} "
                f"(order={order_id}, total={len(current_keys)})",
                flush=True
            )

        time.sleep(POLL_INTERVAL)
