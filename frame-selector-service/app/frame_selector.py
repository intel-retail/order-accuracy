import os
import io
import time
import cv2
import numpy as np
from minio import Minio
from minio.error import S3Error
from ultralytics import YOLO

# =====================================================
# CONFIG FROM ENV (matches docker-compose)
# =====================================================

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET", "minioadmin")

FRAMES_BUCKET = os.getenv("MINIO_BUCKET", "frames")
SELECTED_BUCKET = os.getenv("SELECTED_BUCKET", "selected")

TOP_K = 3  # max frames to pick per order


# =====================================================
# YOLO MODEL (for item counting)
# =====================================================

print("[frame-selector] Loading YOLO model (yolov8n.pt)...")
model = YOLO("yolov8n.pt")


# =====================================================
# MINIO CLIENT
# =====================================================

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def wait_for_bucket(bucket: str):
    """Wait until a bucket exists."""
    while True:
        try:
            if client.bucket_exists(bucket):
                print(f"[frame-selector] Bucket ready: {bucket}")
                return
        except Exception as e:
            print(f"[frame-selector] Bucket check error for {bucket}: {e}")
        time.sleep(1)


def ensure_buckets():
    wait_for_bucket(FRAMES_BUCKET)
    if not client.bucket_exists(SELECTED_BUCKET):
        print(f"[frame-selector] Creating bucket: {SELECTED_BUCKET}")
        client.make_bucket(SELECTED_BUCKET)


# =====================================================
# HELPERS
# =====================================================

def load_image(bucket: str, key: str):
    """Load an image from MinIO and decode using cv2."""
    try:
        resp = client.get_object(bucket, key)
        data = resp.read()
        resp.close()
        resp.release_conn()

        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[frame-selector] cv2.imdecode returned None for {key}")
        return img
    except Exception as e:
        print(f"[frame-selector] ERROR loading {key}: {e}")
        return None


def count_items(frame) -> int:
    """Count non-hand objects from YOLO detection."""
    # Use recommended call: model(frame)[0]
    result = model(frame, conf=0.1, verbose=False)[0]
    count = 0
    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = result.names.get(cls_id, "").lower()  # use result.names
        if cls_name in {"hand", "person"}:
            continue
        count += 1
    return count


def list_orders_and_frames():
    """
    Reads frames bucket and groups keys by order ID.
    Expects structure: frames/<order_id>/<frame>.jpg
    """
    objects = client.list_objects(FRAMES_BUCKET, recursive=True)

    orders = {}
    for obj in objects:
        key = obj.object_name

        if not key.lower().endswith(".jpg"):
            continue

        parts = key.split("/", 1)
        if len(parts) != 2:
            # malformed key, ignore
            continue

        order_id = parts[0]
        orders.setdefault(order_id, []).append(key)

    return orders


# =====================================================
# MAIN LOOP
# =====================================================

if __name__ == "__main__":
    print("[frame-selector] Starting...")
    ensure_buckets()
    print(f"[frame-selector] Connected to MinIO: {MINIO_ENDPOINT}")
    print("[frame-selector] Watching for frames...")

    processed_orders = set()

    while True:
        try:
            all_orders = list_orders_and_frames()
        except S3Error as e:
            print("[frame-selector] ERROR listing objects:", e)
            time.sleep(2)
            continue

        for order_id, keys in all_orders.items():
            if order_id in processed_orders:
                continue  # already handled once

            print(f"\n[frame-selector] Processing order {order_id}: {len(keys)} frames")

            scored_frames = []

            for key in sorted(keys):
                img = load_image(FRAMES_BUCKET, key)
                if img is None:
                    continue

                items = count_items(img)
                scored_frames.append((items, key, img))
                print(f"[frame-selector]   frame={key}  items={items}")

            if not scored_frames:
                print(f"[frame-selector] No valid frames for order {order_id}")
                processed_orders.add(order_id)
                continue

            # Sort: item_count DESC, then filename DESC (later frames win on tie)
            scored_frames.sort(key=lambda x: (x[0], x[1]), reverse=True)

            # Pick up to TOP_K frames (max 3 or fewer if not enough)
            k = min(TOP_K, len(scored_frames))
            topk = scored_frames[:k]

            print(f"[frame-selector] Selected top {k} frames for order {order_id}:")
            for items, key, _ in topk:
                print(f"    -> {key} (items={items})")

            # Upload selected frames to SELECTED_BUCKET
            for rank, (items, key, frame) in enumerate(topk, 1):
                out_key = f"{order_id}/rank_{rank}.jpg"
                ok, buf = cv2.imencode(".jpg", frame)
                if not ok:
                    print(f"[frame-selector]   ! Failed to encode {key}")
                    continue

                data = buf.tobytes()
                client.put_object(
                    SELECTED_BUCKET,
                    out_key,
                    io.BytesIO(data),
                    len(data),
                    content_type="image/jpeg",
                )

                print(f"[frame-selector]   Saved: {out_key} (items={items})")

            processed_orders.add(order_id)
            print(f"[frame-selector] Done for order {order_id}")

        time.sleep(2)
