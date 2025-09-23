# video_file_util.py
import cv2
import subprocess
import os
from ultralytics import YOLO
from config import logger

MODEL_PATH = "yolo11s.pt"
SAMPLE_INTERVAL = 8
OUTPUT_DIR = "clips"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model(model_path=MODEL_PATH):
    return YOLO(model_path)


def analyze_frame(model, frame, target_classes):
    results = model(frame, verbose=False)
    class_names = model.names
    for r in results:
        for box in r.boxes:
            if class_names[int(box.cls.cpu().numpy())] in target_classes:
                return False
    return True


def ffmpeg_clip(video_path, start, end, output_path, min_duration, reencode=True):
    if (end - start) < min_duration:
        logger.info(f"⏩ Skipping {output_path}, too short ({end - start:.2f}s)")
        return False

    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", video_path]
    if reencode:
        cmd += [
            "-c:v", "libx264", "-preset", "ultrafast",
            "-crf", "23", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart"
        ]
    else:
        cmd += ["-c", "copy"]

    cmd.append(output_path)
    subprocess.run(cmd, check=True)
    return True


def split_video_on_empty(video_path, model, target_classes, sample_interval=SAMPLE_INTERVAL):
    """
    Splits a video file into clips whenever the table becomes empty.
    Returns a list of clip file paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    frame_step = max(1, int(sample_interval * fps))
    frame_idx = 0
    clip_idx = 1
    in_clip = False
    clip_start_time = 0.0
    empty_detected = False
    clips = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = frame_idx / fps

        if frame_idx % frame_step == 0:
            empty = analyze_frame(model, frame, target_classes)

            if empty and not empty_detected:
                if in_clip:
                    clip_end_time = timestamp
                    out_file = os.path.join(OUTPUT_DIR, f"clip_{clip_idx:03d}.mp4")
                    if ffmpeg_clip(video_path, clip_start_time, clip_end_time,
                                   out_file, min_duration=sample_interval, reencode=True):
                        clips.append(out_file)
                        logger.info(f"✅ Saved {out_file}")
                        clip_idx += 1
                empty_detected = True
                in_clip = False

            elif not empty and not in_clip:
                clip_start_time = timestamp
                in_clip = True
                empty_detected = False

        frame_idx += 1

    if in_clip:
        clip_end_time = duration
        out_file = os.path.join(OUTPUT_DIR, f"clip_{clip_idx:03d}.mp4")
        if ffmpeg_clip(video_path, clip_start_time, clip_end_time,
                       out_file, min_duration=sample_interval, reencode=True):
            clips.append(out_file)
            logger.info(f"✅ Saved {out_file}")

    cap.release()
    return clips
