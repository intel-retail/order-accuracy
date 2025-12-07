import cv2
import time
import easyocr
from ultralytics import YOLO
from minio import Minio
import io
from utils import CONFIG, preprocess_roi, read_order_id

# -------------------------
# MinIO inline client
# -------------------------
client = Minio(
    "minio:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False,
)

BUCKET = "frames"

def upload_to_minio(order_id, frame_idx, frame):
    if not client.bucket_exists(BUCKET):
        client.make_bucket(BUCKET)

    ok, buf = cv2.imencode(".jpg", frame)
    data = io.BytesIO(buf.tobytes())

    filename = f"{order_id}/{frame_idx}.jpg"

    client.put_object(
        BUCKET,
        filename,
        data,
        len(buf),
        content_type="image/jpeg"
    )

    print(f"[application] Uploaded frame → {filename}")


# -------------------------
# Main Video Processor
# -------------------------
def process_video(video_path):
    print("[application] Loading YOLO model...")
    model = YOLO("yolov8n.pt")

    print("[application] Initializing EasyOCR...")
    reader = easyocr.Reader(['en'], gpu=False)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    skip_interval = int(fps / CONFIG["fps_process"]) / 2
    frame_idx = 0

    print(f"[application] Processing at {CONFIG['fps_process']} FPS (skip={skip_interval})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_interval != 0:
            frame_idx += 1
            continue

        # YOLO detection
        detections = model.predict(frame, conf=CONFIG["conf_threshold"], verbose=False)[0]

        hand_detected = False
        for box in detections.boxes:
            if model.names[int(box.cls)].lower() in CONFIG["hand_labels"]:
                hand_detected = True

        if hand_detected:
            frame_idx += 1
            continue

        # OCR
        order_id = read_order_id(frame, reader, frame_idx)
        if not order_id:
            frame_idx += 1
            continue

        print(f"[application] PASS frame {frame_idx} → order={order_id}")

        # MinIO upload (inline)
        upload_to_minio(order_id, frame_idx, frame)

        frame_idx += 1

    cap.release()
    print("[application] Done extracting frames.")
