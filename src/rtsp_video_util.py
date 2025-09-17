import cv2
from ultralytics import YOLO
import os
import queue
import threading
import subprocess
from config import logger

def make_streamable(path):
    tmp_path = path.replace(".mp4", "_fast.mp4")
    process = subprocess.run(
        [
            "ffmpeg", "-y", "-i", path,
            "-c", "copy", "-movflags", "+faststart", tmp_path
        ]
    )
    os.replace(tmp_path, path)

# =========================
# Config
# =========================
MODEL_PATH = "yolo11n.pt"
SAMPLE_INTERVAL = 5  # seconds
OUTPUT_DIR = "clips"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load YOLO model
# =========================
def load_model(model_path=MODEL_PATH):
    return YOLO(model_path)

# =========================
# Check if table is empty in a frame
# =========================
def analyze_frame(model, frame, target_classes):
    results = model(frame, verbose=False)
    class_names = model.names
    detected_classes = set()
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls.cpu().numpy())
            detected_classes.add(class_names[cls_id])
    return detected_classes.isdisjoint(target_classes)  # True if empty table

# =========================
# Real-time streaming & splitting with queue + stop support
# =========================
def stream_and_split(rtsp_url, model, target_classes, clip_queue, stop_event, sample_interval=20):
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"Error: Cannot open RTSP stream {rtsp_url}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Change codec for MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')   # safer cross-platform
    # or: fourcc = cv2.VideoWriter_fourcc(*'avc1')   # H.264 if supported

    clip_idx = 1
    out, clip_name = None, None
    frame_count = 0
    last_sample_frame = 0
    clip_start_frame = 0

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        if out is None:
            clip_name = os.path.join(OUTPUT_DIR, f"clip_{clip_idx:03d}.mp4")
            out = cv2.VideoWriter(clip_name, fourcc, fps, (width, height))
            clip_start_frame = frame_count
            print(f"[Producer] Started new clip: {clip_name}")

        out.write(frame)
        frame_count += 1

        # Sample every N seconds
        if frame_count - last_sample_frame >= int(sample_interval * fps):
            last_sample_frame = frame_count
            if analyze_frame(model, frame, target_classes):
                # Calculate clip duration
                clip_duration = (frame_count - clip_start_frame) / fps

                # Close current clip
                out.release()
                
                print(f"[Producer] Empty table detected at ~{frame_count/fps:.2f}s → closing {clip_name}")

                # Push only if duration != interval
                if round(clip_duration) > sample_interval:
                    make_streamable(clip_name)
                    clip_queue.put(clip_name)
                    print(f"[Producer] Queued {clip_name} (duration {clip_duration:.2f}s)")
                else:
                    os.remove(clip_name)
                    print(f"[Producer] Skipped {clip_name} (exact {sample_interval}s empty clip)")

                clip_idx += 1
                out, clip_name = None, None

    # release last clip
    if out is not None:
        out.release()
        clip_duration = (frame_count - clip_start_frame) / fps
        if round(clip_duration) > sample_interval:
            make_streamable(clip_name)
            clip_queue.put(clip_name)
            print(f"[Producer] Queued {clip_name} (duration {clip_duration:.2f}s)")
        else:
            os.remove(clip_name)
            print(f"[Producer] Skipped {clip_name} (exact {sample_interval}s empty clip)")

    cap.release()
    print("[Producer] Streaming finished")

# =========================
# Helper functions to start/stop
# =========================
def start_stream(rtsp_url, model, target_classes, clip_queue, stop_event):
    t = threading.Thread(
        target=stream_and_split,
        args=(rtsp_url, model, target_classes, clip_queue, stop_event, SAMPLE_INTERVAL),
        daemon=True
    )
    t.start()
    return t

def stop_stream(stop_event, clip_queue):
    print("[System] Stop requested...")
    stop_event.set()
    clip_queue.put(None)  # sentinel for consumers
