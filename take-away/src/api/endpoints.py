"""
API Endpoints for Single Worker Mode
FastAPI REST endpoints for video upload and processing
"""
import os
import logging
import uuid
import shutil
from fastapi import FastAPI, Body, UploadFile, File, Query
from typing import Dict, Any, Optional, List

from core.pipeline_runner import run_pipeline_async
from core.order_results import get_results, get_statistics, clear_all_results
from core.config_loader import load_config
from core.vlm_service import run_vlm
from core.video_history import (
    start_video, complete_video, fail_video,
    get_video, get_video_history, get_video_summary,
    get_current_video_id, add_result_to_video, clear_video_history
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

cfg = load_config()


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(title="Order Accuracy Service - Single Worker Mode")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy",
            "mode": "single",
            "version": "1.0.0"
        }

    @app.post("/upload-video")
    async def upload_and_run_video(file: UploadFile = File(...)):
        """Upload video file and start processing pipeline"""
        logger.info(f"Received video upload request: filename={file.filename}")
        
        if not file.filename.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            logger.warning(f"Rejected unsupported file type: {file.filename}")
            return {
                "status": "error",
                "reason": "unsupported_file_type"
            }

        video_id = str(uuid.uuid4())
        save_path = f"/uploads/{video_id}_{file.filename}"
        logger.debug(f"Generated video_id={video_id}, save_path={save_path}")

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Video saved successfully: video_id={video_id}, path={save_path}")

        # Register video in history tracker
        start_video(video_id, file.filename, save_path)
        logger.info(f"Video registered in history: video_id={video_id}")

        # Trigger pipeline
        logger.info(f"Triggering GStreamer pipeline for video_id={video_id}")
        run_pipeline_async(
            source_type="file",
            source=save_path
        )

        logger.info(f"Pipeline started for video_id={video_id}")
        return {
            "status": "started",
            "video_id": video_id,
            "path": save_path,
            "filename": file.filename
        }

    @app.post("/run-video")
    def run_video(payload: Dict[str, Any] = Body(...)):
        """Run processing pipeline on video source"""
        source_type = payload.get("source_type")  # file | rtsp | webcam | http
        source = payload.get("source")
        
        logger.info(f"Received run-video request: source_type={source_type}, source={source}")

        if not source_type or not source:
            logger.warning("Missing source_type or source in payload")
            return {
                "status": "error",
                "reason": "source_type_or_source_missing"
            }

        # Trigger pipeline
        logger.info(f"Triggering pipeline: source_type={source_type}, source={source}")
        run_pipeline_async(
            source_type=source_type,
            source=source
        )

        return {
            "status": "started",
            "source_type": source_type,
            "source": source
        }

    @app.get("/results/{order_id}")
    def get_order_results(order_id: str):
        """Get validation results for a specific order"""
        logger.info(f"Fetching results for order_id={order_id}")
        all_results = get_results()
        
        # Find result for this order_id
        for result in all_results:
            if result.get("order_id") == order_id:
                logger.info(f"Returning results for order_id={order_id}")
                return result
        
        logger.warning(f"No results found for order_id={order_id}")
        return {
            "status": "not_found",
            "order_id": order_id
        }

    @app.get("/vlm/results")
    def get_all_results():
        """Get all VLM results (for Gradio UI compatibility)"""
        logger.info("Fetching all VLM results")
        all_results = get_results()
        logger.info(f"Returning {len(all_results)} results")
        return {
            "results": all_results
        }

    @app.post("/run_vlm")
    async def run_vlm_endpoint(payload: Dict[str, Any] = Body(...)):
        """Process order with VLM service"""
        order_id = payload.get("order_id")
        station_id = payload.get("station_id")
        logger.info(f"[API] Received VLM processing request for order_id={order_id}, station_id={station_id}")
        
        if not order_id:
            logger.warning("[API] Missing order_id in VLM request")
            return {
                "status": "error",
                "reason": "order_id_missing"
            }
        
        logger.debug(f"[API] Delegating order_id={order_id} to VLM service")
        result = await run_vlm(order_id, station_id=station_id)
        logger.info(f"[API] VLM processing completed for order_id={order_id}, status={result.get('status')}")
        return result

    @app.get("/statistics")
    def get_station_statistics():
        """Get processing statistics for this station"""
        logger.debug("[API] Retrieving station statistics")
        stats = get_statistics()
        logger.info(f"[API] Station statistics: {stats}")
        return stats

    # ==================== Video History Endpoints ====================

    @app.get("/videos/history")
    def get_videos_history(limit: int = 50):
        """Get history of processed videos"""
        logger.info(f"[API] Fetching video history, limit={limit}")
        history = get_video_history(limit=limit)
        logger.info(f"[API] Returning {len(history)} videos from history")
        return {
            "status": "success",
            "count": len(history),
            "videos": history
        }

    @app.get("/videos/summary")
    def get_videos_summary():
        """Get summary statistics for video processing history"""
        logger.info("[API] Fetching video history summary")
        summary = get_video_summary()
        logger.info(f"[API] Video summary: {summary}")
        return {
            "status": "success",
            "summary": summary
        }

    @app.get("/videos/current")
    def get_current_video():
        """Get currently processing video ID"""
        logger.info("[API] Fetching current video ID")
        video_id = get_current_video_id()
        if video_id:
            video = get_video(video_id)
            return {
                "status": "success",
                "video_id": video_id,
                "video": video
            }
        return {
            "status": "success",
            "video_id": None,
            "video": None
        }

    @app.get("/videos/{video_id}")
    def get_video_details(video_id: str):
        """Get details for a specific video by ID"""
        logger.info(f"[API] Fetching video details for video_id={video_id}")
        video = get_video(video_id)
        if video:
            logger.info(f"[API] Found video: {video_id}")
            return {
                "status": "success",
                "video": video
            }
        logger.warning(f"[API] Video not found: {video_id}")
        return {
            "status": "not_found",
            "video_id": video_id
        }

    @app.post("/videos/{video_id}/complete")
    def mark_video_complete(video_id: str):
        """Mark a video as completed"""
        logger.info(f"[API] Marking video complete: video_id={video_id}")
        complete_video(video_id)
        return {
            "status": "success",
            "video_id": video_id,
            "message": "Video marked as completed"
        }

    @app.post("/videos/{video_id}/fail")
    def mark_video_failed(video_id: str, error: str = "Unknown error"):
        """Mark a video as failed"""
        logger.info(f"[API] Marking video failed: video_id={video_id}, error={error}")
        fail_video(video_id, error)
        return {
            "status": "success",
            "video_id": video_id,
            "message": "Video marked as failed"
        }

    @app.delete("/videos/history")
    def delete_video_history():
        """Clear all video processing history and results"""
        logger.info("[API] Clearing video history and results")
        history_result = clear_video_history()
        results_result = clear_all_results()
        logger.info(f"[API] Video history cleared: {history_result}, Results cleared: {results_result}")
        return {
            "status": "success",
            "message": "Video history and results cleared",
            "videos_cleared": history_result.get('cleared_count', 0),
            "results_cleared": results_result.get('cleared_count', 0)
        }

    return app


# Export for module-level imports
__all__ = ['create_app']
