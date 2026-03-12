"""
API Endpoints for Single Worker Mode
FastAPI REST endpoints for video upload and processing
"""
import os
import re
import logging
import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, Body, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse
from typing import Dict, Any, Optional, List
import cv2

from core.pipeline_runner import run_pipeline_async
from core.order_results import get_results, get_statistics, clear_all_results
from core.config_loader import load_config
from core.vlm_service import run_vlm
from core.video_history import (
    start_video, complete_video, fail_video,
    get_video, get_video_history, get_video_summary,
    get_current_video_id, add_result_to_video, clear_video_history,
    get_video_by_order_id
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

cfg = load_config()

SERVICE_MODE = os.getenv('SERVICE_MODE', 'single')

# ── Order Recall configuration ───────────────────────────────────────────────
# Directory where frame-selector saves ALL frames per order
#   structure: {FRAME_SELECTOR_DIR}/{station_id}/{order_id}/frame_N.jpg
FRAME_SELECTOR_DIR = Path(os.getenv('FRAME_SELECTOR_DEBUG_DIR', '/results/frame-selector-in'))
# Cached MP4s are written here so a second recall is instant
RECALL_CACHE_DIR = Path(os.getenv('RESULTS_DIR', '/results')) / 'recall_cache'
# Playback speed for the stitched replay video (10 fps matches 10-fps capture)
REPLAY_FPS = float(os.getenv('RECALL_REPLAY_FPS', '10'))


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
        """Run processing pipeline on any video source (file | rtsp | webcam | http).
        Always starts a new GStreamer pipeline for the given source.
        This is independent of any StationManager-managed workers.
        """
        source_type = payload.get("source_type")
        source = payload.get("source")

        logger.info(f"Received run-video request: source_type={source_type}, source={source}")

        if not source_type or not source:
            logger.warning("Missing source_type or source in payload")
            return {
                "status": "error",
                "reason": "source_type_or_source_missing"
            }

        # Register in video history so the Detected Orders tab can track it
        video_id = str(uuid.uuid4())
        display_name = source if source_type == "rtsp" else source.split("/")[-1]
        start_video(video_id, filename=display_name, path=source)
        logger.info(f"Registered source in history: video_id={video_id}")

        # Launch GStreamer pipeline in background thread
        run_pipeline_async(source_type=source_type, source=source)
        logger.info(f"Pipeline started: source_type={source_type}, source={source}")

        return {
            "status": "started",
            "video_id": video_id,
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

    @app.get("/mode")
    def get_service_mode():
        """Return current service mode (single | parallel) and worker count"""
        return {
            "service_mode": SERVICE_MODE,
            "workers": int(os.getenv('WORKERS', '1')),
            "station_id": os.getenv('STATION_ID', 'station_1')
        }

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

    # ==================== Order Recall Endpoints ====================

    @app.get("/orders/{order_id}/recall")
    def recall_order(order_id: str):
        """Look up order details from history with 24-hour TTL.

        Returns:
          200 + status='not_found'  – order ID was never processed
          200 + status='expired'    – order was processed but > 24 h ago
          200 + status='found'      – full result + frame availability flag
        """
        logger.info(f"[API] Order recall requested: order_id={order_id}")
        recall = get_video_by_order_id(order_id)

        if recall['status'] != 'found':
            logger.info(f"[API] Recall result for order {order_id}: {recall['status']}")
            return recall

        # Check whether frame-selector frames exist for this order
        station_id = recall.get('station_id', 'station_1')
        order_frame_dir = _find_order_frame_dir(station_id, order_id)
        frame_files = _sorted_frames(order_frame_dir) if order_frame_dir else []

        recall['frames_available'] = len(frame_files)
        recall['has_replay'] = len(frame_files) > 0
        logger.info(
            f"[API] Recall found for order {order_id}: "
            f"{len(frame_files)} frames, station={station_id}"
        )
        return recall

    @app.get("/orders/{order_id}/frames/{filename}")
    def serve_order_frame(order_id: str, filename: str):
        """Serve a single JPEG frame for an order.

        Bridges the volume gap: Gradio container cannot mount /results directly,
        so it fetches each frame via this HTTP endpoint.
        """
        logger.info(f"[API] Frame requested: order_id={order_id}, filename={filename}")

        recall = get_video_by_order_id(order_id)
        if recall['status'] == 'not_found':
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        if recall['status'] == 'expired':
            raise HTTPException(status_code=410, detail=f"Order {order_id} recall window has expired")

        station_id = recall.get('station_id', 'station_1')
        order_frame_dir = _find_order_frame_dir(station_id, order_id)
        if not order_frame_dir:
            raise HTTPException(status_code=404, detail=f"No frames found for order {order_id}")

        frame_path = order_frame_dir / filename
        if not frame_path.exists():
            raise HTTPException(status_code=404, detail=f"Frame {filename} not found")

        return FileResponse(path=str(frame_path), media_type="image/jpeg")

    @app.get("/orders/{order_id}/replay")
    def replay_order(order_id: str):
        """Generate (or serve cached) an MP4 replay for an order.

        Stitches all frame-selector JPEG frames for the order into an MP4
        at REPLAY_FPS (default 2 fps).  The result is cached under
        /results/recall_cache/ so subsequent requests are served instantly.
        """
        logger.info(f"[API] Replay requested: order_id={order_id}")

        recall = get_video_by_order_id(order_id)
        if recall['status'] == 'not_found':
            raise HTTPException(status_code=404, detail=f"Order {order_id} not found")
        if recall['status'] == 'expired':
            raise HTTPException(status_code=410, detail=f"Order {order_id} recall window has expired")

        station_id = recall.get('station_id', 'station_1')

        # Serve from cache if already built
        RECALL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_path = RECALL_CACHE_DIR / f"{station_id}_{order_id}.mp4"

        if not cached_path.exists():
            # Locate the frame directory
            order_frame_dir = _find_order_frame_dir(station_id, order_id)
            if not order_frame_dir:
                raise HTTPException(
                    status_code=404,
                    detail=f"No frames found for order {order_id} — replay unavailable"
                )

            frame_files = _sorted_frames(order_frame_dir)
            if not frame_files:
                raise HTTPException(
                    status_code=404,
                    detail=f"Frame directory for order {order_id} is empty"
                )

            # Read first frame to get video dimensions
            first_frame = cv2.imread(str(frame_files[0]))
            if first_frame is None:
                raise HTTPException(status_code=500, detail="Failed to read frames from disk")

            height, width = first_frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(cached_path), fourcc, REPLAY_FPS, (width, height))

            for fp in frame_files:
                frame = cv2.imread(str(fp))
                if frame is not None:
                    writer.write(frame)

            writer.release()
            logger.info(
                f"[API] Replay MP4 built for order {order_id}: "
                f"{len(frame_files)} frames → {cached_path}"
            )

        return FileResponse(
            path=str(cached_path),
            media_type="video/mp4",
            filename=f"order_{order_id}_replay.mp4"
        )

    return app


# ── Private helpers ───────────────────────────────────────────────────────────

def _find_order_frame_dir(station_id: str, order_id: str) -> Optional[Path]:
    """Return the Path to the frame-selector debug folder for this order.

    Tries {FRAME_SELECTOR_DIR}/{station_id}/{order_id} first (fast path).
    Falls back to scanning all station subdirectories so the code stays
    correct even if station_id stored in history differs from the folder name.
    """
    candidate = FRAME_SELECTOR_DIR / station_id / order_id
    if candidate.exists():
        return candidate

    # Broader scan across all station dirs
    if FRAME_SELECTOR_DIR.exists():
        for station_dir in FRAME_SELECTOR_DIR.iterdir():
            if station_dir.is_dir():
                alt = station_dir / order_id
                if alt.exists():
                    logger.debug(
                        f"[API] Found frames for order {order_id} in {alt} "
                        f"(expected station_id={station_id})"
                    )
                    return alt

    logger.warning(f"[API] No frame directory found for order {order_id}")
    return None


def _sorted_frames(order_dir: Path) -> List[Path]:
    """Return JPEG files in order_dir sorted numerically by the number in the filename."""
    files = [
        f for f in order_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png')
    ]
    return sorted(
        files,
        key=lambda f: int(re.search(r'(\d+)', f.stem).group(1))
        if re.search(r'(\d+)', f.stem) else 0
    )


# Export for module-level imports
__all__ = ['create_app']
