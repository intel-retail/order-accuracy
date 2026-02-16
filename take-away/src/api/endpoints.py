"""
API Endpoints for Single Worker Mode
FastAPI REST endpoints for video upload and processing
"""
import os
import logging
import uuid
import shutil
from fastapi import FastAPI, Body, UploadFile, File
from typing import Dict, Any

from core.pipeline_runner import run_pipeline_async
from core.order_results import get_results, get_statistics
from core.config_loader import load_config
from core.vlm_service import run_vlm

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
            "path": save_path
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

    return app


# Export for module-level imports
__all__ = ['create_app']
