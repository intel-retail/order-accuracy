from minio import Minio
import os
import io
import json
from PIL import Image

def get_client():
    return Minio(
        os.environ["MINIO_ENDPOINT"],
        access_key=os.environ["MINIO_ACCESS"],
        secret_key=os.environ["MINIO_SECRET"],
        secure=False
    )

def ensure_bucket(client, bucket):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)

def upload_frame(client, bucket, order_id, frame_index, frame, item_count):
    img = Image.fromarray(frame[:, :, ::-1])
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    buf.seek(0)

    fname = f"{order_id}/{frame_index}.jpg"
    client.put_object(bucket, fname, buf, length=buf.getbuffer().nbytes)

    meta = {"frame_index": frame_index, "item_count": item_count}
    meta_bytes = json.dumps(meta).encode()
    client.put_object(bucket, fname + ".json", io.BytesIO(meta_bytes), len(meta_bytes))
