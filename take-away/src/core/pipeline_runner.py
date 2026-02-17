import subprocess
import os, io
import threading
import logging
from minio import Minio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def build_gstreamer_pipeline(source_type: str, source: str) -> str:
    logger.debug(f"Building GStreamer pipeline: source_type={source_type}, source={source}")
    if source_type == "file":
        src = f"filesrc location={source}"

    elif source_type == "rtsp":
        source = normalize_rtsp_url(source)
        src = f"rtspsrc location={source} protocols=tcp latency=200"

    elif source_type == "webcam":
        src = f"v4l2src device={source}"

    elif source_type == "http":
        src = f"souphttpsrc location={source}"

    else:
        logger.error(f"Unsupported source_type: {source_type}")
        raise ValueError(f"Unsupported source_type: {source_type}")

    pipeline = (
        f"{src} "
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! videorate "
        "! video/x-raw,framerate=1/1 "
        "! gvapython module=frame_pipeline function=process_frame "
        "! fakesink sync=true"  # sync=true ensures real-time playback so OCR can keep up
    )

    logger.info(f"GStreamer pipeline built: {pipeline[:100]}...")
    return pipeline

def run_pipeline(source_type: str, source: str):
    logger.info(f"Starting pipeline execution: source_type={source_type}, source={source}")
    pipeline = build_gstreamer_pipeline(source_type, source)

    cmd = f"gst-launch-1.0 -v -e {pipeline}"
    logger.debug(f"Executing GStreamer command: {cmd[:150]}...")

    # Explicitly pass environment to ensure PYTHONPATH is set for gvapython
    env = os.environ.copy()
    
    # Ensure app directory is in PYTHONPATH for gvapython to find frame_pipeline
    pythonpath = env.get('PYTHONPATH', '')
    
    # Use /app as base directory (frame_pipeline.py is at root)
    app_dir = '/app'
    
    if app_dir not in pythonpath:
        pythonpath = f"{app_dir}:{pythonpath}" if pythonpath else app_dir
    
    env['PYTHONPATH'] = pythonpath
    logger.debug(f"Updated PYTHONPATH for subprocess: {env['PYTHONPATH']}")

    try:
        # Run with explicit environment and working directory
        subprocess.run(
            cmd, 
            shell=True, 
            check=True, 
            env=env,
            cwd=app_dir  # Use detected working directory
        )
        logger.info("GStreamer pipeline completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"GStreamer pipeline failed with exit code {e.returncode}")
        raise
    
    # 🔔 EOS MARKER (CRITICAL)
    logger.info("Pipeline finished. Writing EOS marker to MinIO")

    # Get station ID from environment (same as frame_pipeline.py uses)
    station_id = os.environ.get('STATION_ID', 'station_1')
    
    client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    try:
        # Write station-specific EOS marker: {station_id}/__EOS__
        # This signals to frame_selector that all frames for this station are ready
        eos_key = f"{station_id}/__EOS__"
        client.put_object(
            "frames",
            eos_key,
            io.BytesIO(b"done"),
            length=4,
            content_type="text/plain",
        )
        logger.info(f"EOS marker written successfully to frames bucket: {eos_key}")
    except Exception as e:
        logger.error(f"Failed to write EOS marker: {e}")

def run_pipeline_async(source_type: str, source: str):
    logger.info(f"Starting pipeline in background thread: source_type={source_type}")
    t = threading.Thread(
        target=run_pipeline,
        args=(source_type, source),
        daemon=True
    )
    t.start()
    logger.debug(f"Pipeline thread started: thread_id={t.ident}")

def normalize_rtsp_url(url: str) -> str:
    if url.startswith("rtsp://localhost"):
        return url.replace("rtsp://localhost", "rtsp://host.docker.internal", 1)

    if url.startswith("rtsp://127.0.0.1"):
        return url.replace("rtsp://127.0.0.1", "rtsp://host.docker.internal", 1)

    return url
