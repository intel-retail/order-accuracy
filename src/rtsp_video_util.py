import cv2
from ultralytics import YOLO
import os
import queue
import threading
import subprocess
from config import logger

# =========================
# Config
# =========================
MODEL_PATH = "yolo11n.pt"
SAMPLE_INTERVAL = 8  # seconds
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
    """Return True if none of the target classes are detected."""
    results = model(frame, verbose=False)
    class_names = model.names
    for r in results:
        for box in r.boxes:
            if class_names[int(box.cls.cpu().numpy())] in target_classes:
                return False
    return True  # empty table

# =========================
# FFmpeg helpers
# =========================
def start_ffmpeg_writer(rtsp_url, output_path):
    """Start FFmpeg process to dump RTSP stream without re-encoding, streamable MP4."""
    # -fflags +genpts can help with some RTSP sources; kept minimal here
    return subprocess.Popen([
        "ffmpeg", "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-c", "copy",
        "-an",                        # drop audio if not needed
        "-movflags", "+faststart",    # make file streamable
        "-y", output_path
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def stop_ffmpeg_writer(proc):
    """Gracefully stop FFmpeg process."""
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()

# =========================
# Real-time streaming & splitting
# =========================
def stream_and_split(rtsp_url, model, target_classes, clip_queue, stop_event, sample_interval=SAMPLE_INTERVAL):
    """
    Producer: reads frames for detection and starts/stops ffmpeg subprocesses to produce lossless, faststart mp4 clips.
    When finished (stop_event set and producer cleaned up), places a single None sentinel into clip_queue.
    """
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        logger.error(f"Cannot open RTSP stream {rtsp_url}")
        # If we can't open the stream, signal consumer to stop
        try:
            clip_queue.put(None)
        except Exception:
            pass
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_step = max(1, int(sample_interval * fps))

    clip_idx = 1
    frame_count = 0
    ffmpeg_proc, clip_name, clip_start_frame = None, None, None

    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed (ret=False). Breaking producer loop.")
                break

            frame_count += 1

            # Start new clip if none active
            if ffmpeg_proc is None:
                clip_name = os.path.join(OUTPUT_DIR, f"clip_{clip_idx:03d}.mp4")
                ffmpeg_proc = start_ffmpeg_writer(rtsp_url, clip_name)
                clip_start_frame = frame_count
                logger.info(f"[Producer] Started new clip: {clip_name}")

            # Run YOLO only every SAMPLE_INTERVAL
            if frame_count % frame_step == 0:
                if analyze_frame(model, frame, target_classes):
                    # Close current recording
                    stop_ffmpeg_writer(ffmpeg_proc)
                    duration = (frame_count - clip_start_frame) / fps

                    if duration > sample_interval:
                        clip_queue.put(clip_name)
                        logger.info(f"[Producer] Queued {clip_name} ({duration:.2f}s)")
                    else:
                        try:
                            os.remove(clip_name)
                        except Exception:
                            pass
                        logger.info(f"[Producer] Skipped {clip_name} ({duration:.2f}s < {sample_interval}s)")

                    # Prepare for next clip
                    clip_idx += 1
                    ffmpeg_proc, clip_name, clip_start_frame = None, None, None

        # End while
    except Exception as e:
        logger.exception(f"[Producer] Unexpected error in stream_and_split: {e}")
    finally:
        # Cleanup last active clip if any
        try:
            if ffmpeg_proc is not None:
                stop_ffmpeg_writer(ffmpeg_proc)
                duration = (frame_count - clip_start_frame) / fps if clip_start_frame else 0
                if duration > sample_interval:
                    clip_queue.put(clip_name)
                    logger.info(f"[Producer] Queued last {clip_name} ({duration:.2f}s)")
                else:
                    try:
                        os.remove(clip_name)
                    except Exception:
                        pass
                    logger.info(f"[Producer] Skipped last {clip_name} ({duration:.2f}s < {sample_interval}s)")
        except Exception:
            logger.exception("[Producer] Error while cleaning up ffmpeg proc")

        cap.release()
        logger.info("[Producer] Streaming finished")

        # Signal consumer that no more clips will be produced
        try:
            clip_queue.put(None)
        except Exception:
            pass

# =========================
# Helper functions
# =========================
def start_stream(rtsp_url, model, target_classes, clip_queue, stop_event):
    t = threading.Thread(
        target=stream_and_split,
        args=(rtsp_url, model, target_classes, clip_queue, stop_event),
        daemon=True
    )
    t.start()
    return t

def stop_stream(stop_event):
    """
    Signal producer to stop. Producer will place None sentinel into the queue when done.
    """
    logger.info("[System] Stop requested...")
    stop_event.set()
