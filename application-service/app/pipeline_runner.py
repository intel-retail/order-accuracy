import subprocess
import os, io
import threading
from minio import Minio

def build_gstreamer_pipeline(source_type: str, source: str) -> str:
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
        raise ValueError(f"Unsupported source_type: {source_type}")

    pipeline = (
        f"{src} "
        "! decodebin "
        "! videoconvert "
        "! video/x-raw,format=BGR "
        "! videorate "
        "! video/x-raw,framerate=1/1 "
        "! gvapython module=frame_pipeline function=process_frame "
        "! fakesink sync=false"
    )

    return pipeline

def run_pipeline(source_type: str, source: str):
    pipeline = build_gstreamer_pipeline(source_type, source)

    cmd = f"gst-launch-1.0 -v -e {pipeline}"

    subprocess.run(cmd, shell=True, check=True)
    
      # 🔔 EOS MARKER (CRITICAL)
    print("[application] Pipeline finished. Writing EOS marker.")

    client = Minio(
        "minio:9000",
        access_key="minioadmin",
        secret_key="minioadmin",
        secure=False,
    )

    client.put_object(
        "frames",
        "__EOS__",
        io.BytesIO(b"done"),
        length=4,
        content_type="text/plain",
    )

def run_pipeline_async(source_type: str, source: str):
    t = threading.Thread(
        target=run_pipeline,
        args=(source_type, source),
        daemon=True
    )
    t.start()

def normalize_rtsp_url(url: str) -> str:
    if url.startswith("rtsp://localhost"):
        return url.replace("rtsp://localhost", "rtsp://host.docker.internal", 1)

    if url.startswith("rtsp://127.0.0.1"):
        return url.replace("rtsp://127.0.0.1", "rtsp://host.docker.internal", 1)

    return url
