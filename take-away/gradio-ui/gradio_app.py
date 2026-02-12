import gradio as gr
import requests
import cv2
import threading
import time
import numpy as np
from typing import Optional
from PIL import Image
import queue

API_BASE = "http://oa_service:8000"

# Global variables for RTSP streaming
STREAM_STOP_EVENT = threading.Event()
frame_queue = queue.Queue(maxsize=5)  # Buffer for smooth streaming

# -----------------------------
# OPTIMIZED RTSP STREAMING FUNCTIONS
# -----------------------------

class RTSPStreamReader:
    def __init__(self, rtsp_url):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.running = False
        self.thread = None
        
    def start(self):
        if self.running:
            return False
            
        print(f"[RTSP Reader] Starting stream: {self.rtsp_url}")
        self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        
        if not self.cap.isOpened():
            print(f"[RTSP Reader] Failed to connect")
            return False
            
        # Optimize capture settings for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get video dimensions for aspect ratio
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[RTSP Reader] Stream dimensions: {width}x{height}")
        
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        
        print(f"[RTSP Reader] Stream started successfully")
        return True
    
    def stop(self):
        print(f"[RTSP Reader] Stopping stream")
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.cap:
            self.cap.release()
        
        # Clear the queue
        while not frame_queue.empty():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                break
                
    def _read_frames(self):
        while self.running and self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                print("[RTSP Reader] Lost connection, reconnecting...")
                time.sleep(1)
                continue
            
            # Convert to RGB for Gradio
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Add to queue (drop old frames if queue is full)
            try:
                frame_queue.put_nowait(pil_image)
            except queue.Full:
                # Remove oldest frame and add new one
                try:
                    frame_queue.get_nowait()
                    frame_queue.put_nowait(pil_image)
                except queue.Empty:
                    pass
            
            # Small delay to prevent overwhelming
            time.sleep(0.02)  # ~50 FPS max

# Global stream reader instance
rtsp_reader = None

def start_smooth_stream(rtsp_url):
    """Start smooth RTSP streaming"""
    global rtsp_reader
    
    if not rtsp_url or not rtsp_url.startswith("rtsp://"):
        yield None, "❌ Invalid RTSP URL"
        return
    
    # Stop any existing stream
    if rtsp_reader:
        rtsp_reader.stop()
        rtsp_reader = None
    
    # Clear stop event
    STREAM_STOP_EVENT.clear()
    
    # Start new stream reader
    rtsp_reader = RTSPStreamReader(rtsp_url)
    
    if not rtsp_reader.start():
        yield None, "❌ Failed to start RTSP stream"
        return
    
    frame_count = 0
    last_update = time.time()
    fps_counter = 0
    
    try:
        while not STREAM_STOP_EVENT.is_set():
            try:
                # Get frame from queue with timeout
                frame = frame_queue.get(timeout=1.0)
                frame_count += 1
                fps_counter += 1
                
                # Calculate FPS every second
                current_time = time.time()
                if current_time - last_update >= 1.0:
                    fps = fps_counter / (current_time - last_update)
                    status = f"✅ Frame {frame_count} - Smooth stream active ({fps:.1f} FPS)"
                    fps_counter = 0
                    last_update = current_time
                else:
                    status = f"✅ Frame {frame_count} - Smooth stream active"
                
                yield frame, status
                
                # Small delay to control update rate in Gradio
                time.sleep(0.05)  # 20 FPS for UI updates
                
            except queue.Empty:
                yield None, f"❌ Frame {frame_count} - No frames available"
                continue
                
    except Exception as e:
        yield None, f"❌ Stream error: {str(e)}"
    
    finally:
        if rtsp_reader:
            rtsp_reader.stop()
            rtsp_reader = None
        yield None, f"Smooth stream stopped after {frame_count} frames"

def stop_smooth_stream():
    """Stop the smooth stream"""
    global rtsp_reader
    
    STREAM_STOP_EVENT.set()
    
    if rtsp_reader:
        rtsp_reader.stop()
        rtsp_reader = None
    
    return None, "Smooth stream stopped by user"

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

def start_rtsp_processing(rtsp_url):
    """Start RTSP processing pipeline"""
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
            return f"❌ RTSP processing failed: {resp.text}"

        return "✅ RTSP processing pipeline started"

    except Exception as e:
        return f"❌ RTSP processing error: {e}"

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
# FORMAT RESULTS
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
            name = item.get("name", "Unknown")
            qty = item.get("quantity", 0)

            label = "OK"

            if any(m.get("name") == name for m in missing):
                label = "Missing"
            elif any(e.get("name") == name for e in extra):
                label = "Extra"
            elif any(q.get("name") == name for q in qty_mismatch):
                label = "Qty Mismatch"

            item_lines.append(f"{name} x{qty} ({label})")

        rows.append([
            order_id,
            "\n".join(item_lines) if item_lines else "No items",
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
        with gr.TabItem("📁 Upload Video"):
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
        # SIMPLIFIED RTSP TAB
        # ======================
        with gr.TabItem("📡 RTSP Stream"):
            with gr.Row():
                # Left column: Controls
                with gr.Column(scale=1):
                    rtsp_url = gr.Textbox(
                        label="RTSP URL",
                        placeholder="rtsp://host.docker.internal:8554/test",
                        value="rtsp://host.docker.internal:8554/test"
                    )
                    
                    # Stream controls
                    gr.Markdown("### 🎥 Live Stream Preview")
                    
                    with gr.Row():
                        stream_start_btn = gr.Button("▶️ Start", variant="primary")
                        stream_stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                    
                    # Processing controls
                    gr.Markdown("### 🔄 Processing Pipeline")
                    process_btn = gr.Button("🚀 Start Processing", variant="primary")
                    
                    # Status displays
                    processing_status = gr.Textbox(label="Processing Status", lines=2)
                    
                    # Info section
                    # gr.Markdown("""
                    # ### 📝 How It Works:
                    
                    # **▶️ Start**:
                    # - Continuous background frame reading
                    # - Adaptive FPS display (~20 FPS)
                    # - Low latency buffering
                    # - Auto-reconnection on errors
                    
                    # **🚀 Start Processing**:
                    # - Begins order detection pipeline
                    # - Processes frames for VLM analysis
                    
                    # **Preview shows**: Raw input frames to your pipeline
                    # **Results**: Check "📊 Detected Orders" tab
                    
                    # **Stream Info:**
                    # - Source: `sample.mp4` (looped)
                    # - MediaMTX: `localhost:8554/test`
                    # """)
                
                # Right column: Stream display
                with gr.Column(scale=2):
                    stream_image = gr.Image(
                        label="RTSP Live Stream",
                        width=None,  # Let it auto-size
                        height=None,  # Let it auto-size
                        interactive=False,
                        show_label=True,
                        container=True
                    )
                    stream_status = gr.Textbox(label="Stream Status", lines=2)

            # Connect button functions - Smooth streaming only
            stream_start_btn.click(
                fn=start_smooth_stream,
                inputs=[rtsp_url],
                outputs=[stream_image, stream_status]
            )
            
            stream_stop_btn.click(
                fn=stop_smooth_stream,
                outputs=[stream_image, stream_status]
            )
            
            # Processing
            process_btn.click(
                fn=start_rtsp_processing,
                inputs=[rtsp_url],
                outputs=[processing_status]
            )

        # ======================
        # RESULTS TAB
        # ======================
        with gr.TabItem("📊 Detected Orders"):
            results_table = gr.Dataframe(
                headers=["Order ID", "Items (Bill View)", "Order Status"],
                datatype=["str", "str", "str"],
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
    print("[Gradio] Starting Order Accuracy UI with smooth RTSP streaming...")
    
    # Enable queue for generator support with higher concurrency
    demo.queue(concurrency_count=5, max_size=20)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True
    )