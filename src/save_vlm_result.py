import os
import sys
import subprocess
from pathlib import Path
import json
import io
from typing import Tuple
from config import MINIO_HOST, logger, VIDEO_MINIO_LOCATION_TEMPLATE


# Create a global MinIO client instance (singleton)
_minio_client = None
MINIO_BUCKET = "order-accuracy-vlm-results"

def get_minio_client():
    global _minio_client
    if _minio_client is None:
        try:
            from minio import Minio
        except ImportError:
            logger.error(
                "MinIO Python SDK is not installed. Please install it with:\n"
                "  pip install minio"
            )
            return None
        logger.info(f"############ MINIO_HOST =================={MINIO_HOST}")
        MINIO_ENDPOINT = os.environ.get("MINIO_HOST", MINIO_HOST)
        MINIO_ACCESS_KEY = os.environ.get("MINIO_ROOT_USER", "user")
        MINIO_SECRET_KEY = os.environ.get("MINIO_ROOT_PASSWORD", "passwd")
        _minio_client = Minio(
            MINIO_ENDPOINT,
            access_key=MINIO_ACCESS_KEY,
            secret_key=MINIO_SECRET_KEY,
            secure=False
        )
    return _minio_client

def save_to_minio(order_id: str, data: dict, video_id: str = None, bucket: str = None) -> Tuple[bool, str]:  
    try:
        from minio.error import S3Error
    except ImportError:
        logger.error(
            "MinIO Python SDK is not installed. Please install it with:\n"
            "  pip install minio"
        )
        return False, "MinIO SDK not installed"

    client = get_minio_client()
    if client is None:
        return False, "MinIO client not available"

    # Use provided bucket or default to MINIO_BUCKET
    target_bucket = bucket if bucket is not None else MINIO_BUCKET
    
    logger.info(f"Connected to MinIO:  bucket={target_bucket}")

    try:
        # Ensure bucket exists
        if not client.bucket_exists(target_bucket):
            client.make_bucket(target_bucket)
            logger.info(f"Created bucket: {target_bucket}")

        filename = f"{order_id}.json"
        data_to_save = dict(data)
        if video_id is not None:
            data_to_save["video_id"] = video_id
        json_bytes = json.dumps(data_to_save, indent=2).encode("utf-8")
        file_obj = io.BytesIO(json_bytes)

        logger.info(f"Preparing to upload JSON data to MinIO: {target_bucket}/{filename}")   
        client.put_object(
            target_bucket,
            filename,
            file_obj,
            length=len(json_bytes),
            content_type="application/json"
        )

        logger.info(f"Saved results to MinIO: {target_bucket}/{filename} (order_id={order_id}, video_id={video_id})")
        return True, f"{target_bucket}/{filename}"

    except S3Error as e:
        logger.error(f"MinIO S3 error: {e}")
        return False, str(e)
    except Exception as e:
        logger.error(f"Exception while saving to MinIO: {e}")
        return False, str(e)

def get_order_json_from_minio(order_id: str, bucket: str = None) -> dict:
    """
    Download and return the JSON (excluding video_id) from MinIO for the given order_id.
    """
    client = get_minio_client()
    if client is None:
        logger.error("MinIO client not available")
        return {"error": "MinIO client not available"}

    # Use provided bucket or default to MINIO_BUCKET
    target_bucket = bucket if bucket is not None else MINIO_BUCKET
    filename = f"{order_id}.json"
    
    try:
        response = client.get_object(target_bucket, filename)
        json_bytes = response.read()
        data = json.loads(json_bytes.decode("utf-8"))
        return data
    except Exception as e:
        logger.error(f"Failed to fetch {filename} from {target_bucket}: {e}")
        return {"error": f"Failed to fetch order: {e}"}

def get_video_url_from_minio(video_id: str, bucket: str = None) -> str:
    """
    Return a video URL or local file path for Gradio video preview.
    If Gradio blocks direct IP URLs, download the video from MinIO to a temp file and return the local path.
    """
    if not video_id:
        logger.error("No video_id provided for video preview.")
        return ""
    try:
        # Use provided bucket or default to MINIO_BUCKET for video location template
        target_bucket = bucket if bucket is not None else MINIO_BUCKET
        url = VIDEO_MINIO_LOCATION_TEMPLATE.format(video_id=video_id, bucket=target_bucket)
        logger.info(f"Generated video URL for preview: {url}")

        # Try to use HTTP/HTTPS URL first (works if Gradio allows)
        if url.startswith("http://") or url.startswith("https://"):
            # Gradio >=4 may block IPs; if so, fallback to local download
            parsed = urlparse(url)
            try:
                # If hostname is an IP address, fallback to download
                socket.inet_aton(parsed.hostname)
                is_ip = True
            except Exception:
                is_ip = False
            if not is_ip:
                return url

        # Fallback: Download video from MinIO to a temp file and return local path
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        logger.info(f"Downloading video from MinIO to local temp file: {tmp.name}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"Failed to generate or download video for preview: {e}")
        return ""

def get_video_url_from_minio(video_id: str) -> str:
    """
    Return a video URL or local file path for Gradio video preview.
    If Gradio blocks direct IP URLs, download the video from MinIO to a temp file and return the local path.
    """
    if not video_id:
        logger.error("No video_id provided for video preview.")
        return ""
    try:
        url = VIDEO_MINIO_LOCATION_TEMPLATE.format(video_id=video_id)
        logger.info(f"Generated video URL for preview: {url}")

        # Try to use HTTP/HTTPS URL first (works if Gradio allows)
        if url.startswith("http://") or url.startswith("https://"):
            # Gradio >=4 may block IPs; if so, fallback to local download
            import socket
            from urllib.parse import urlparse
            parsed = urlparse(url)
            try:
                # If hostname is an IP address, fallback to download
                socket.inet_aton(parsed.hostname)
                is_ip = True
            except Exception:
                is_ip = False
            if not is_ip:
                return url

        # Fallback: Download video from MinIO to a temp file and return local path
        import tempfile
        import requests
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        logger.info(f"Downloading video from MinIO to local temp file: {tmp.name}")
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception as e:
        logger.error(f"Failed to generate or download video for preview: {e}")
        return ""



