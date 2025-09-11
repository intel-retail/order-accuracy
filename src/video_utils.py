"""Video processing utilities for the grocery video app."""

import os
import subprocess
from typing import Tuple

try:
    from moviepy.editor import VideoFileClip  # type: ignore
except ImportError:  # Optional dependency
    VideoFileClip = None  # noqa: N816

from config import MAX_VIDEO_DURATION, logger


def _duration_with_moviepy(video_path: str) -> float:
    """Get video duration using moviepy."""
    if not VideoFileClip:
        raise ImportError("moviepy not installed")
    clip = VideoFileClip(video_path)
    try:
        return float(clip.duration)
    finally:
        clip.close()


def _duration_with_ffprobe(video_path: str) -> float:
    """Fallback using ffprobe (ffmpeg)."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15).decode().strip()
        return float(out)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"ffprobe duration extraction failed: {e}") from e


def get_video_orientation(video_path: str) -> str:
    """Detect video orientation using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=width,height", "-of", "csv=s=x:p=0", video_path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=15).decode().strip()
        width, height = map(int, out.split('x'))
        logger.debug(f"Video dimensions: {width}x{height}")
        
        if height > width:
            logger.info(f"Video is vertical: {width}x{height}")
            return "vertical"
        else:
            logger.info(f"Video is horizontal: {width}x{height}")
            return "horizontal"
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to detect video orientation: {e}, defaulting to horizontal")
        return "horizontal"


def get_video_duration(video_path: str) -> float:
    """Get duration preferring moviepy, falling back to ffprobe."""
    try:
        if VideoFileClip:
            return _duration_with_moviepy(video_path)
        return _duration_with_ffprobe(video_path)
    except Exception:
        # Try ffprobe as last resort
        return _duration_with_ffprobe(video_path)


def validate_video_duration(video_path: str) -> Tuple[bool, float, str]:
    """Validate video duration against maximum allowed duration."""
    try:
        duration = get_video_duration(video_path)
        if duration <= MAX_VIDEO_DURATION:
            return True, duration, ""
        return False, duration, f"Video too long: {duration:.2f}s (max {MAX_VIDEO_DURATION}s)."
    except Exception as e:  # noqa: BLE001
        hint = "Install moviepy via 'uv add moviepy' and ensure ffmpeg is installed (apt install ffmpeg)" if isinstance(e, ImportError) else "Ensure ffmpeg is installed."
        return False, 0.0, f"Failed to read video duration: {e}. {hint}"


def upload_video(video_path: str, original_name: str) -> Tuple[bool, str, str]:
    """Upload video file to the video service."""
    import requests
    from config import VIDEO_UPLOAD_URL
    
    logger.info(f"Starting video upload: {original_name}")
    try:
        name = os.path.splitext(os.path.basename(original_name))[0]
        tags = "grocery,upload"
        logger.debug(f"Upload parameters: name={name}, tags={tags}")
        
        with open(video_path, 'rb') as f:
            files = {'video': ('video.mp4', f, 'video/mp4')}
            data = {'name': name, 'tags': tags}
            logger.info(f"Uploading to {VIDEO_UPLOAD_URL}")
            resp = requests.post(VIDEO_UPLOAD_URL, files=files, data=data, timeout=120)
            
        logger.info(f"Upload response status: {resp.status_code}")
        if resp.status_code not in (200, 201):
            logger.error(f"Upload failed with status {resp.status_code}: {resp.text}")
            return False, "", f"Upload failed: {resp.status_code} {resp.text}"
        try:
            payload = resp.json()
            logger.debug(f"Upload response payload: {payload}")
        except Exception:
            logger.error(f"Invalid JSON response: {resp.text[:200]}")
            return False, "", f"Invalid upload response: {resp.text[:200]}"
        video_id = payload.get("uuid") or payload.get("id") or payload.get("videoId")
        if not video_id:
            logger.error(f"No video ID in response: {payload}")
            return False, "", f"UUID missing in upload response: {payload}"
        logger.info(f"Video uploaded successfully with ID: {video_id}")
        return True, video_id, ""
    except Exception as e:
        logger.exception(f"Exception during video upload: {e}")
        return False, "", f"Exception during upload: {e}"
