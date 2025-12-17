import cv2
import os
from io import BytesIO
from ultralytics import YOLO
from minio import Minio
from fastapi import FastAPI, Body
from ocr import read_order_id
from vlm_service import run_vlm

# =========================
# CONFIG
# =========================
VIDEO_SOURCE = "/videos/sample.mp4"
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.7
FPS_TARGET = 1
HAND_LABELS = {"hand", "person"}
BUCKET = "frames"

# =========================
# FASTAPI APP (API ONLY)
# =========================
app = FastAPI()

@app.post("/run_vlm")
async def run_vlm_endpoint(payload: dict = Body(...)):
    order_id = payload.get("order_id")
    if not order_id:
        return {
            "status": "error",
            "reason": "order_id_missing"
        }

    return await run_vlm(order_id)
# =========================
# VIDEO PIPELINE (ISOLATED)
# =========================
def run_video_pipeline():
    print("[application] Initializing MinIO client...")
    client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    print("[application] Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("Video cannot be opened.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip_interval = max(1, int(fps / FPS_TARGET))
    frame_idx = 0

    print(f"[application] Processing at {FPS_TARGET} FPS (skip={skip_interval})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_interval != 0:
            frame_idx += 1
            continue

        detections = model.predict(
            frame,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            verbose=False
        )[0]

        hand_present = False
        for box in detections.boxes:
            label = model.names[int(box.cls)].lower()
            if label in HAND_LABELS:
                hand_present = True
                break

        if hand_present:
            frame_idx += 1
            continue

        order_id = read_order_id(frame, frame_idx)
        if not order_id:
            frame_idx += 1
            continue

        _, jpeg = cv2.imencode(".jpg", frame)
        jpeg_bytes = jpeg.tobytes()

        objname = f"{order_id}/{frame_idx}.jpg"
        client.put_object(
            BUCKET,
            objname,
            data=BytesIO(jpeg_bytes),
            length=len(jpeg_bytes),
            content_type="image/jpeg",
        )

        print(f"[application] Stored frame {frame_idx} → order {order_id}")
        frame_idx += 1

    cap.release()
    print("[application] Completed frame extraction.")

# =========================
# ENTRYPOINT
# =========================
if __name__ == "__main__":
    run_video_pipeline()
