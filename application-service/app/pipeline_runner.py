import subprocess
import os
import threading





def build_gstreamer_pipeline(source_type: str, source: str) -> str:
    if source_type == "file":
        src = f"filesrc location={source}"

    elif source_type == "rtsp":
        src = f"rtspsrc location={source} latency=200"

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



def run_pipeline_async(source_type: str, source: str):
    t = threading.Thread(
        target=run_pipeline,
        args=(source_type, source),
        daemon=True
    )
    t.start()
