"""Pipeline management functions for object detection."""

import json
import time
from typing import Tuple

import requests

from config import (
    PIPELINE_TRIGGER_URL, 
    PIPELINE_TRIGGER_URL_VERTICAL,
    PIPELINES_BASE_URL, 
    VIDEO_MINIO_LOCATION_TEMPLATE,
    POLL_INTERVAL,
    logger
)
from video_utils import get_video_orientation


def trigger_pipeline(video_id: str, duration: float, video_path: str = None) -> Tuple[bool, str, str]:
    """Trigger pipeline and return pipeline instance UUID."""
    logger.info(f"Triggering pipeline for video {video_id} with duration {duration}s")
    
    # Determine which pipeline URL to use based on video orientation
    pipeline_url = PIPELINE_TRIGGER_URL  # Default to horizontal
    
    if video_path:
        orientation = get_video_orientation(video_path)
        if orientation == "vertical":
            pipeline_url = PIPELINE_TRIGGER_URL_VERTICAL
            logger.info(f"Using vertical pipeline for video {video_id}")
        else:
            logger.info(f"Using horizontal pipeline for video {video_id}")
    else:
        logger.warning(f"No video path provided, defaulting to horizontal pipeline for video {video_id}")
    
    body = {
        "source": {
            "element": "curlhttpsrc",
            "type": "gst",
            "properties": {
                "location": VIDEO_MINIO_LOCATION_TEMPLATE.format(video_id=video_id)
            }
        },
        "parameters": {
            "frame": 40,
            "chunk_duration": int(duration),
            "detection-properties": {
                "model": "/home/pipeline-server/models/yoloworld/FP32/yolov8l-worldv2.xml",
                "device": "CPU"
            },
            "publish": {
                "minio_bucket": "video-summary",
                "video_identifier": video_id,
                "topic": "topic/video_stream"
            }
        }
    }
    
    logger.debug(f"Pipeline trigger payload: {json.dumps(body, indent=2)}")
    
    try:
        logger.info(f"Sending pipeline trigger request to {pipeline_url}")
        resp = requests.post(pipeline_url, json=body, timeout=30)
        logger.info(f"Pipeline trigger response status: {resp.status_code}")
        
        if resp.status_code not in (200, 201, 202):
            logger.error(f"Pipeline trigger failed with status {resp.status_code}: {resp.text}")
            return False, "", f"Pipeline trigger failed: {resp.status_code} {resp.text}"
        try:
            payload = resp.json()
            logger.debug(f"Pipeline trigger response: {payload}")
        except Exception:
            logger.error(f"Invalid JSON in pipeline response: {resp.text[:200]}")
            return False, "", f"Invalid pipeline trigger response: {resp.text[:200]}"
        
        # Handle both string and object responses
        if isinstance(payload, str):
            pipeline_id = payload
        elif isinstance(payload, dict):
            pipeline_id = payload.get("id") or payload.get("uuid") or payload.get("pipeline_id") or payload.get("instance_id")
        else:
            pipeline_id = None
            
        if not pipeline_id:
            logger.error(f"No pipeline ID in response: {payload}")
            return False, "", f"Pipeline ID missing in trigger response: {payload}"
        logger.info(f"Pipeline triggered successfully with ID: {pipeline_id}")
        return True, pipeline_id, ""
    except Exception as e:
        logger.exception(f"Exception triggering pipeline: {e}")
        return False, "", f"Exception triggering pipeline: {e}"


def wait_for_pipeline_completion(pipeline_id: str, timeout: int = 600) -> Tuple[bool, str]:
    """Poll the specific pipeline instance URL until completion or failure."""
    status_url = f"{PIPELINES_BASE_URL}/{pipeline_id}"
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(status_url, timeout=10)
            if resp.status_code != 200:
                logger.warning("Status code %s while polling %s", resp.status_code, status_url)
            else:
                try:
                    status_json = resp.json()
                except Exception:
                    status_json = {}
                state = (status_json.get("state") or status_json.get("status") or "").lower()
                logger.debug(f"Pipeline {pipeline_id} current state: {state}")
                if state in ("completed", "finished", "done", "success"):
                    logger.info(f"Pipeline {pipeline_id} completed successfully")
                    return True, ""
                if state in ("error", "failed", "failure"):
                    logger.error(f"Pipeline {pipeline_id} failed with state: {state}")
                    return False, f"Pipeline failed: {status_json}"
            time.sleep(POLL_INTERVAL)
        except Exception as e:
            logger.error("Polling error for %s: %s", pipeline_id, e)
            time.sleep(POLL_INTERVAL)
    return False, f"Pipeline polling timed out for {pipeline_id}"
