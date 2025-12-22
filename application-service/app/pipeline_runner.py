import subprocess
import os
import threading

def run_pipeline(video_path: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    cmd = [
        "gst-launch-1.0",
        "-v", "-e",
        f"filesrc location={video_path}",
        "!", "decodebin",
        "!", "videoconvert",
        "!", "video/x-raw,format=BGR",
        "!", "videorate",
        "!", "video/x-raw,framerate=1/1",
        "!", "gvapython",
        "module=frame_pipeline",
        "function=process_frame",
        "!", "fakesink",
        "sync=false",
    ]
    #add cpu/gpu device in gstreamer
    subprocess.run(" ".join(cmd), shell=True)


def run_pipeline_async(video_path: str):
    t = threading.Thread(
        target=run_pipeline,
        args=(video_path,),
        daemon=True
    )
    t.start()
