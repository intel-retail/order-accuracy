import cv2
import os
import time
from ultralytics import YOLO
from minio import Minio
from io import BytesIO
from ocr import read_order_id, now_ms
from fastapi import FastAPI, UploadFile, File, Form
from vlm_service import run_vlm

VIDEO_SOURCE = "/videos/sample.mp4"
CONF_THRESHOLD = 0.1
IOU_THRESHOLD = 0.7
FPS_TARGET = 1
HAND_LABELS = {"hand", "person"}

client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)


app = FastAPI()

@app.post("/run_vlm")
async def run_vlm_endpoint(order_id: str = Form(...), images: list[UploadFile] = File(...)):
    return await run_vlm(order_id, images)

BUCKET = "frames"

# Ensure bucket exists
if not client.bucket_exists(BUCKET):
    client.make_bucket(BUCKET)

print("[application] Loading YOLO model...")
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise RuntimeError("Video cannot be opened.")

fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30
skip_interval = int(fps / FPS_TARGET)

frame_idx = 0
print(f"[application] Processing at 1 FPS (skip interval = {skip_interval})")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % skip_interval != 0:
        frame_idx += 1
        continue

    # ---------- YOLO DETECTION ----------
    detections = model.predict(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]

    hand_present = False
    item_labels = set()

    for box in detections.boxes:
        label = model.names[int(box.cls)].lower()

        if label in HAND_LABELS:
            hand_present = True
        else:
            item_labels.add(label)

    if hand_present:
        frame_idx += 1
        continue

    # ---------- OCR ORDER ID ----------
    order_id = read_order_id(frame, frame_idx)
    if not order_id:
        frame_idx += 1
        continue

    # ---------- SAVE TO MINIO ----------
    folder = f"{order_id}"
    objname = f"{folder}/{frame_idx}.jpg"

    # Encode frame into JPEG bytes
    _, jpeg = cv2.imencode(".jpg", frame)
    jpeg_bytes = jpeg.tobytes()

    # Wrap in a file-like buffer
    jpeg_stream = BytesIO(jpeg_bytes)

    client.put_object(
        BUCKET,
        objname,
        data=jpeg_stream,
        length=len(jpeg_bytes),
        content_type="image/jpeg"
    )

    print(f"[application] Stored frame {frame_idx} under order {order_id}")

    frame_idx += 1

cap.release()
print("[application] Completed frame extraction.")
