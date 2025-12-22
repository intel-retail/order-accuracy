import os
import asyncio
from minio import Minio
from minio.error import S3Error
from config_loader import load_config
cfg = load_config()

MINIO = cfg["minio"]
BUCKETS = cfg["buckets"]

FRAMES_BUCKET = BUCKETS["frames"]
SELECTED_BUCKET = BUCKETS["selected"]


# -------------------------
# MinIO config
# -------------------------
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS = os.getenv("MINIO_ACCESS", "minioadmin")
MINIO_SECRET = os.getenv("MINIO_SECRET", "minioadmin")
SELECTED_BUCKET = os.getenv("SELECTED_BUCKET", "selected")

# -------------------------
# MinIO client
# -------------------------
client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=False,
)

# -------------------------
# VLM entrypoint
# -------------------------

async def run_vlm(order_id: str):
    print("[VLM] START", flush=True)
    print(f"[VLM] Order ID: {order_id}", flush=True)

    prefix = f"{order_id}/"
    frames = []

    try:
        for obj in client.list_objects(SELECTED_BUCKET, prefix=prefix, recursive=True):
            if obj.object_name.lower().endswith(".jpg"):
                frames.append(obj.object_name)
    except S3Error as e:
        print("[VLM] MinIO error while listing frames:", e, flush=True)
        return {
            "order_id": order_id,
            "items": [],
            "status": "error",
            "reason": "minio_list_failed",
        }

    if not frames:
        print("[VLM] No frames found for order", order_id, flush=True)
        return {
            "order_id": order_id,
            "items": [],
            "status": "no_frames",
        }

    frames.sort()

    print("[VLM] Frames loaded from selected bucket:", flush=True)
    for f in frames:
        print(f"   - {f}", flush=True)

    # -------------------------------------------------
    # TODO: real VLM logic goes here
    # -------------------------------------------------
    await asyncio.sleep(0.1)  # simulate processing

    print("[VLM] DONE", flush=True)

    return {
        "order_id": order_id,
        "frames_used": frames,
        "items": ["item1", "item2"],
        "status": "ok",
    }
