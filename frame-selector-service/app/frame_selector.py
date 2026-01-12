import os
import io
import time
import cv2
import numpy as np
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

# How many consecutive frames required to confirm a new order
MIN_FRAMES_PER_ORDER = FS_CFG.get("min_frames_per_order", 2)

VLM_ENDPOINT = VLM_CFG["endpoint"]

processed_orders = set()


# =====================================================
# VLM Caller
# =====================================================

def call_vlm(order_id, timeout=120):
    payload = {"order_id": order_id}
    resp = requests.post(VLM_ENDPOINT, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# =====================================================
# Model + MinIO
# =====================================================

print("[frame-selector] Loading YOLO model...", flush=True)
model = YOLO("yolov8n.pt")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO["access_key"],
    secret_key=MINIO["secret_key"],
    secure=MINIO.get("secure", False),
)


# =====================================================
# Helpers
# =====================================================

def wait_for_bucket(bucket):
    while True:
        try:
            if client.bucket_exists(bucket):
                return
        except:
            pass
        time.sleep(1)


def ensure_buckets():
    wait_for_bucket(FRAMES_BUCKET)
    if not client.bucket_exists(SELECTED_BUCKET):
        client.make_bucket(SELECTED_BUCKET)


def list_frames_sorted():
    frames = []
    eos_seen = False

    for obj in client.list_objects(FRAMES_BUCKET, recursive=True):
        if obj.object_name == "__EOS__":
            eos_seen = True
            continue

        if obj.object_name.lower().endswith(".jpg"):
            parts = obj.object_name.split("/", 1)
            if len(parts) == 2:
                frames.append((parts[0], obj.object_name))

    frames.sort(key=lambda x: x[1])
    return frames, eos_seen


def load_image(key):
    resp = client.get_object(FRAMES_BUCKET, key)
    data = resp.read()
    resp.close()
    resp.release_conn()
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def count_items(frame):
    result = model(frame, conf=0.1, verbose=False)[0]

    # If ANY skip-label (like person/hand) is present → discard frame
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        if cls_name in SKIP_LABELS:
            return -1   # mark frame as invalid

    # Otherwise count valid objects
    count = 0
    for box in result.boxes:
        cls_name = result.names.get(int(box.cls), "").lower()
        if cls_name not in SKIP_LABELS:
            count += 1

    return count



# =====================================================
# Order Finalization
# =====================================================

def process_completed_order(order_id, keys):
    if not keys:
        return

    # Prevent duplicate VLM calls
    if order_id in processed_orders:
        print(f"[frame-selector] Order {order_id} already processed — skipping", flush=True)
        return

    # Ignore tiny OCR-noise orders
    if len(keys) < MIN_FRAMES_PER_ORDER:
        print(f"[frame-selector] Ignoring order {order_id} (only {len(keys)} frame — OCR noise)", flush=True)
        return

    print(f"\n[frame-selector] Finalizing order {order_id} with {len(keys)} frames", flush=True)

    scored = []
    for key in keys:
        img = load_image(key)
        if img is None:
            continue
        items = count_items(img)

        # Skip frames containing person/hand etc.
        if items < 0:
            print(f"[frame-selector] Skipping {key} (contains person/hand)", flush=True)
            continue

        scored.append((items, key, img))

    if not scored:
        return

    # Pick TOP_K best frames
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    topk = scored[:min(TOP_K, len(scored))]

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

        print(f"[frame-selector]   Saved {out_key} (items={items})", flush=True)

    # Skip dummy order
    if order_id == "000":
        return

    print(f"[frame-selector] Calling VLM for order {order_id}", flush=True)

    try:
        response = call_vlm(order_id)
        print("[frame-selector] VLM response:", response, flush=True)
        processed_orders.add(order_id)
    except Exception as e:
        print("[frame-selector] VLM call failed:", e, flush=True)


# =====================================================
# MAIN LOOP
# =====================================================

if __name__ == "__main__":
    ensure_buckets()
    print("[frame-selector] Watching frames...", flush=True)

    processed_keys = set()

    current_order = None
    current_keys = []

    # For debouncing new order detection
    pending_order = None
    pending_count = 0

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

            # First frame ever
            if current_order is None:
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
                    pending_order = order_id
                    pending_count = 1

                # New order not stable yet → ignore this frame
                if pending_count < MIN_FRAMES_PER_ORDER:
                    print(f"[frame-selector] Pending new order {order_id} ({pending_count}/{MIN_FRAMES_PER_ORDER})", flush=True)
                    processed_keys.add(key)
                    continue

                # New order confirmed stable → close current
                print(f"[frame-selector] New order {order_id} confirmed. Closing order {current_order}", flush=True)
                process_completed_order(current_order, current_keys)

                current_order = order_id
                current_keys = []
                pending_order = None
                pending_count = 0

            # Collect frame into current order
            current_keys.append(key)
            processed_keys.add(key)

            print(f"[frame-selector] Collected {key} (order={order_id}, total={len(current_keys)})", flush=True)

        # End-of-stream closes last order
        if eos_seen and current_order and current_keys:
            print(f"[frame-selector] EOS closing order {current_order}", flush=True)
            process_completed_order(current_order, current_keys)
            current_order = None
            current_keys = []

        time.sleep(POLL_INTERVAL)
