import gradio as gr
import requests

API_BASE = "http://application-service:8000"


# -----------------------------
# API HELPERS
# -----------------------------

def upload_video(file):
    if file is None:
        return "❌ No file selected"

    try:
        with open(file.name, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/upload-video",
                files={"file": f},
                timeout=60
            )

        if resp.status_code != 200:
            return f"❌ Upload failed: {resp.text}"

        data = resp.json()
        return (
            "✅ Video uploaded & pipeline started\n"
            f"Video ID: {data.get('video_id')}\n"
            f"Path: {data.get('path')}"
        )

    except Exception as e:
        return f"❌ Upload error: {e}"


def start_rtsp(rtsp_url):
    if not rtsp_url:
        return "❌ RTSP URL missing"

    payload = {
        "source_type": "rtsp",
        "source": rtsp_url
    }

    try:
        resp = requests.post(
            f"{API_BASE}/run-video",
            json=payload,
            timeout=10
        )

        if resp.status_code != 200:
            return f"❌ RTSP start failed: {resp.text}"

        return "✅ RTSP pipeline started"

    except Exception as e:
        return f"❌ RTSP error: {e}"


def fetch_results():
    try:
        resp = requests.get(
            f"{API_BASE}/vlm/results",
            timeout=5
        )
        if resp.status_code != 200:
            return []

        return resp.json().get("results", [])

    except Exception:
        return []


# -----------------------------
# UI
# -----------------------------

with gr.Blocks(title="Order Accuracy – Video Analyzer") as demo:

    gr.Markdown("## 📦 Order Accuracy – Video Analyzer")

    with gr.Tabs():

        # ======================
        # FILE UPLOAD TAB
        # ======================
        with gr.Tab("📁 Upload Video"):
            upload_file = gr.File(
                label="Upload Video File",
                file_types=[".mp4", ".avi", ".mkv", ".mov"]
            )

            upload_btn = gr.Button("🚀 Upload & Start")
            upload_status = gr.Textbox(label="Status", lines=4)

            upload_btn.click(
                upload_video,
                inputs=upload_file,
                outputs=upload_status
            )

        # ======================
        # RTSP TAB
        # ======================
        with gr.Tab("📡 RTSP Stream"):
            rtsp_url = gr.Textbox(
                label="RTSP URL",
                placeholder="rtsp://<ip>:<port>/stream"
            )

            rtsp_btn = gr.Button("🚀 Start RTSP Stream")
            rtsp_status = gr.Textbox(label="Status", lines=2)

            rtsp_btn.click(
                start_rtsp,
                inputs=rtsp_url,
                outputs=rtsp_status
            )

        # ======================
        # RESULTS TAB
        # ======================
        with gr.Tab("📊 Detected Orders"):
            results_box = gr.JSON(label="VLM Detected Orders")

            refresh_btn = gr.Button("🔄 Refresh Results")
            refresh_btn.click(
                fetch_results,
                outputs=results_box
            )


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
