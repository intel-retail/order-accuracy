import gradio as gr
import requests

API_BASE = "http://oa_service:8000"


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
# FORMAT RESULTS (NEW)
# -----------------------------

def format_detected_orders():
    results = fetch_results()

    if not results:
        return [], "No orders processed yet."

    rows = []
    summaries = []

    for r in results:
        order_id = r.get("order_id", "UNKNOWN")
        detected_items = r.get("detected_items", [])
        validation = r.get("validation", {})
        status = r.get("status", "unknown")

        missing = validation.get("missing", [])
        extra = validation.get("extra", [])
        qty_mismatch = validation.get("quantity_mismatch", [])

        item_lines = []

        for item in detected_items:
            name = item["name"]
            qty = item["quantity"]

            label = "OK"

            if any(m["name"] == name for m in missing):
                label = "Missing"
            elif any(e["name"] == name for e in extra):
                label = "Extra"
            elif any(q["name"] == name for q in qty_mismatch):
                label = "Qty Mismatch"

            item_lines.append(f"{name} x{qty} ({label})")

        rows.append([
            order_id,
            "\n".join(item_lines),
            "✅ VALIDATED" if status == "validated" else "❌ MISMATCH"
        ])

        summaries.append(
            f"### Order {order_id}\n"
            f"- Status: {'✅ VALIDATED' if status == 'validated' else '❌ MISMATCH'}\n"
            f"- Missing: {missing or 'None'}\n"
            f"- Extra: {extra or 'None'}\n"
            f"- Quantity Mismatch: {qty_mismatch or 'None'}"
        )

    return rows, "\n\n".join(summaries)

# -----------------------------
# UI
# -----------------------------

with gr.Blocks(title="Order Accuracy") as demo:

    gr.Markdown("## 📦 Order Accuracy")

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
        # RESULTS TAB (UPDATED)
        # ======================
        with gr.Tab("📊 Detected Orders"):
            results_table = gr.Dataframe(
                headers=["Order ID", "Items (Bill View)", "Order Status"],
                interactive=False
            )


            validation_summary = gr.Textbox(
                label="Validation Summary",
                lines=6
            )

            refresh_btn = gr.Button("🔄 Refresh Results")

            refresh_btn.click(
                format_detected_orders,
                outputs=[results_table, validation_summary]
            )


# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )
