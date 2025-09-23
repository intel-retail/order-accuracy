import cv2
from ultralytics import YOLO
import os
import queue
import threading
import subprocess
import sys
from config import logger
 
# =========================
# Config
# =========================
MODEL_PATH = "yolo11n.pt"
SAMPLE_INTERVAL = 10  # seconds
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
def start_ffmpeg_writer(source, output_path, fps=30):
    """Start FFmpeg process to dump input (RTSP or webcam) into streamable MP4 with audio and better compatibility."""
    cmd = ["ffmpeg"]
 
    # Input source
    if isinstance(source, str) and source.startswith("rtsp://"):
        cmd += ["-rtsp_transport", "tcp", "-i", source]
    else:
        # Webcam input
        if sys.platform.startswith("linux"):
            cmd += ["-f", "v4l2", "-i", str(source)]
        elif sys.platform.startswith("win"):
            cmd += ["-f", "dshow", "-i", str(source)]
        else:
            raise RuntimeError("Unsupported platform for webcam input")
 
    # Add silent audio input (DLStreamer may require audio track)
    cmd += [
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"
    ]
 
    # Video encoding options
    cmd += [
        "-c:v", "libx264",              # Use H.264 codec
        "-profile:v", "baseline",       # Compatibility profile
        "-pix_fmt", "yuv420p",          # Standard pixel format
        "-preset", "ultrafast",         # Faster encoding
        "-crf", "23",                   # Quality setting (lower = better)
        "-g", str(int(fps * 2)),        # Keyframe every 2 seconds
        "-shortest",                    # Trim to shortest input (video or audio)
        "-movflags", "+faststart",      # Enable streaming playback
        "-y", output_path               # Overwrite output file
    ]
 
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
 
 
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
def stream_and_split(source, model, target_classes, clip_queue, stop_event, sample_interval=SAMPLE_INTERVAL):
    """
    Producer: reads frames for detection and starts/stops ffmpeg subprocesses to produce streamable mp4 clips.
    Works with RTSP or webcam input.
    """
    cap_source = source
    if isinstance(source, str) and source.startswith("/dev/video"):
        try:
            cap_source = int(source.replace("/dev/video", ""))
        except ValueError:
            logger.error(f"Invalid webcam device path: {source}")
            clip_queue.put(None)
            return
    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        logger.error(f"Cannot open source {source}")
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
                ffmpeg_proc = start_ffmpeg_writer(source, clip_name)
                clip_start_frame = frame_count
                logger.info(f"[Producer] Started new clip: {clip_name}")
 
            # Run YOLO only every SAMPLE_INTERVAL
            if frame_count % frame_step == 0:
                if analyze_frame(model, frame, target_classes):
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
 
                    clip_idx += 1
                    ffmpeg_proc, clip_name, clip_start_frame = None, None, None
 
    except Exception as e:
        logger.exception(f"[Producer] Unexpected error: {e}")
    finally:
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
 
        try:
            clip_queue.put(None)
        except Exception:
            pass
 
# =========================
# Helper functions
# =========================
def start_stream(source, model, target_classes, clip_queue, stop_event):
    t = threading.Thread(
        target=stream_and_split,
        args=(source, model, target_classes, clip_queue, stop_event),
        daemon=True
    )
    t.start()
    return t
 
def stop_stream(stop_event):
    logger.info("[System] Stop requested...")
    stop_event.set()
