import gradio as gr
import requests
import cv2
import threading
import time
import math
import numpy as np
from typing import Optional, List, Dict
from PIL import Image
import queue
from datetime import datetime

API_BASE = "http://oa_service:8000"

# Global variables for RTSP streaming
STREAM_STOP_EVENT = threading.Event()
frame_queue = queue.Queue(maxsize=5)  # Buffer for smooth streaming

# Global processing state
is_processing = False

APP_TITLE = "📦 Take-Away Order Accuracy"
APP_DESCRIPTION = "AI-powered order validation for take-away operations"

# Intel Brand CSS (matching dine-in styling)
CUSTOM_CSS = """
/* Intel Theme Variables */
:root {
    --primary-500: #0071C5 !important;
    --primary-600: #005A9E !important;
    --primary-700: #00285A !important;
    --neutral-50: #f8fafc;
    --neutral-100: #f1f5f9;
    --neutral-200: #e2e8f0;
}

/* Full Width Layout */
.gradio-container {
    max-width: 100% !important;
    padding: 20px 40px !important;
    margin: 0 !important;
}

/* Hide Gradio Footer Elements */
footer {
    display: none !important;
}

.built-with {
    display: none !important;
}

#footer {
    display: none !important;
}

.gradio-container > footer {
    display: none !important;
}

div[class*="footer"] {
    display: none !important;
}

a[href*="gradio.app"] {
    display: none !important;
}

/* Header Banner - Intel Blue */
.header-banner {
    background: linear-gradient(135deg, #0071C5 0%, #00285A 100%);
    color: white;
    padding: 28px 40px;
    border-radius: 12px;
    margin-bottom: 28px;
    box-shadow: 0 4px 12px rgba(0, 113, 197, 0.3);
}

.header-banner h1 {
    margin: 0;
    font-size: 32px;
    font-weight: 700;
    letter-spacing: -0.5px;
}

.header-banner p {
    margin: 10px 0 0 0;
    opacity: 0.95;
    font-size: 15px;
}

/* Card Styling */
.card-panel {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    overflow: hidden;
}

/* Primary Button - Intel Blue */
.primary-btn {
    background: linear-gradient(135deg, #0071C5 0%, #00285A 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    padding: 14px 36px !important;
    box-shadow: 0 4px 12px rgba(0, 113, 197, 0.3) !important;
    transition: all 0.2s ease !important;
}

.primary-btn:hover {
    background: linear-gradient(135deg, #005A9E 0%, #001d47 100%) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 16px rgba(0, 113, 197, 0.4) !important;
}

.primary-btn:disabled {
    background: #9ca3af !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* Video History Card */
.video-history-card {
    background: white;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    margin-bottom: 16px;
    overflow: hidden;
}

.video-history-header {
    background: linear-gradient(135deg, #0071C5 0%, #00285A 100%);
    color: white;
    padding: 14px 20px;
    cursor: pointer;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.video-history-content {
    padding: 16px 20px;
}

/* Validation Summary Table */
.validation-table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 12px;
}

.validation-table th {
    background: #E6F3FB;
    padding: 12px 15px;
    text-align: left;
    font-weight: 600;
    color: #00285A;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.validation-table td {
    padding: 10px 15px;
    border-bottom: 1px solid #e9ecef;
}

/* Order Card */
.order-card {
    background: #f8fafc;
    border-radius: 8px;
    padding: 12px 16px;
    margin-top: 12px;
    border-left: 4px solid #0071C5;
}

.order-card.validated {
    border-left-color: #10b981;
}

.order-card.mismatch {
    border-left-color: #ef4444;
}

/* Status Badge */
.status-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
}

.status-badge.validated {
    background: #d1fae5;
    color: #10b981;
}

.status-badge.mismatch {
    background: #fee2e2;
    color: #ef4444;
}

/* Section Headers */
.section-header {
    font-weight: 600;
    color: #00285A;
    font-size: 14px;
    margin-bottom: 10px;
}

/* Footer */
.footer-info {
    text-align: center;
    color: #00285A;
    font-size: 13px;
    padding: 20px;
    border-top: 2px solid #E6F3FB;
    margin-top: 32px;
    background: #f8fafc;
}

/* Responsive Full Width */
@media (min-width: 1200px) {
    .gradio-container {
        padding: 24px 60px !important;
    }
}

/* Accordion styling */
.accordion-item {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 12px;
    overflow: hidden;
}
"""

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

def fetch_statistics():
    """Fetch processing statistics from backend"""
    try:
        resp = requests.get(
            f"{API_BASE}/statistics",
            timeout=5
        )
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None

def upload_video_with_progress(file, progress=gr.Progress()):
    """Upload video and track processing with button state management"""
    global is_processing
    
    if file is None:
        return "❌ No file selected", gr.update(interactive=True)
    
    video_name = file.name.split("/")[-1] if "/" in file.name else file.name
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    is_processing = True
    
    try:
        # Get initial statistics before upload
        initial_stats = fetch_statistics()
        initial_processed = initial_stats.get("total_processed", 0) if initial_stats else 0
        
        progress(0.1, desc="📤 Uploading video...")
        
        with open(file.name, "rb") as f:
            resp = requests.post(
                f"{API_BASE}/upload-video",
                files={"file": f},
                timeout=60
            )

        if resp.status_code != 200:
            is_processing = False
            return f"❌ Upload failed: {resp.text}", gr.update(interactive=True)

        data = resp.json()
        video_id = data.get('video_id', 'unknown')
        
        progress(0.2, desc="🎬 Processing video...")
        
        # Poll for results with real statistics tracking
        max_wait = 300  # 5 minutes max
        poll_interval = 2
        elapsed = 0
        last_processed = initial_processed
        orders_completed = 0
        
        while elapsed < max_wait:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            # Calculate a progress value that increases over time (max 90% while processing)
            # Using a logarithmic curve to slow progress over time
            time_progress = min(0.7, 0.2 + 0.5 * (1 - math.exp(-elapsed / 60)))
            
            # Check statistics for progress
            stats = fetch_statistics()
            if stats:
                current_processed = stats.get("total_processed", 0)
                orders_completed = current_processed - initial_processed
                
                if orders_completed > 0:
                    # Bump progress when orders complete
                    order_progress = min(0.9, time_progress + 0.1 * orders_completed)
                    progress(order_progress, desc=f"🔄 {orders_completed} order(s) detected")
                else:
                    progress(time_progress, desc=f"🔍 Analyzing video...")
            else:
                progress(time_progress, desc=f"🔄 Processing...")
            
            # Check if results are available
            results = fetch_results()
            new_results = [r for r in results if r.get("completed_at", "").replace("-", "").replace(":", "").replace(" ", "") >= upload_time.replace("-", "").replace(":", "").replace(" ", "")]
            
            # If we have new results and processing seems done (no new orders for 10 seconds)
            if new_results and len(new_results) >= orders_completed and orders_completed > 0:
                # Wait a bit more to catch any remaining orders
                time.sleep(5)
                results = fetch_results()
                break
        
        progress(1.0, desc="✅ Processing complete!")
        
        # Mark video as completed via API
        try:
            requests.post(f"{API_BASE}/videos/{video_id}/complete", timeout=5)
        except Exception as e:
            print(f"Failed to mark video complete: {e}")
        
        # Fetch final results
        results = fetch_results()
        
        # Filter to results from this session (after upload time)
        session_results = results  # Use all results for now
        
        is_processing = False
        
        stats = fetch_statistics()
        validated = stats.get("total_validated", 0) if stats else 0
        mismatch = stats.get("total_mismatch", 0) if stats else 0
        
        return (
            f"✅ Video processed successfully\n"
            f"Video: {video_name}\n"
            f"Time: {upload_time}\n"
            f"Orders detected: {len(session_results)}\n"
            f"Validated: {validated} | Mismatch: {mismatch}",
            gr.update(interactive=True)
        )

    except Exception as e:
        is_processing = False
        # Mark video as failed if we have a video_id
        if 'video_id' in dir():
            try:
                requests.post(f"{API_BASE}/videos/{video_id}/fail?error={str(e)}", timeout=5)
            except:
                pass
        return f"❌ Upload error: {e}", gr.update(interactive=True)

def generate_validation_summary(results):
    """Generate validation summary from results"""
    if not results:
        return {"total": 0, "validated": 0, "mismatch": 0}
    
    validated = sum(1 for r in results if r.get("status") == "validated")
    mismatch = sum(1 for r in results if r.get("status") != "validated")
    
    return {
        "total": len(results),
        "validated": validated,
        "mismatch": mismatch
    }

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
# FORMAT RESULTS AS HTML
# -----------------------------

def format_validation_summary_table(results):
    """Format validation summary as HTML table"""
    if not results:
        return ""
    
    validated = sum(1 for r in results if r.get("status") == "validated")
    mismatch = sum(1 for r in results if r.get("status") != "validated")
    total = len(results)
    
    return f'''
    <table class="validation-table" style="width: 100%; border-collapse: collapse; margin-top: 12px;">
        <thead>
            <tr>
                <th style="background: #E6F3FB; padding: 12px 15px; text-align: left; font-weight: 600; color: #00285A; font-size: 13px;">Metric</th>
                <th style="background: #E6F3FB; padding: 12px 15px; text-align: center; font-weight: 600; color: #00285A; font-size: 13px;">Count</th>
                <th style="background: #E6F3FB; padding: 12px 15px; text-align: center; font-weight: 600; color: #00285A; font-size: 13px;">Percentage</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef;">Total Orders</td>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center; font-weight: 600;">{total}</td>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center;">100%</td>
            </tr>
            <tr>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; color: #10b981;">✅ Validated</td>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center; font-weight: 600; color: #10b981;">{validated}</td>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center; color: #10b981;">{validated/total*100:.1f}%</td>
            </tr>
            <tr>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; color: #ef4444;">❌ Mismatch</td>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center; font-weight: 600; color: #ef4444;">{mismatch}</td>
                <td style="padding: 10px 15px; border-bottom: 1px solid #e9ecef; text-align: center; color: #ef4444;">{mismatch/total*100:.1f}%</td>
            </tr>
        </tbody>
    </table>
    '''

def format_order_card(result):
    """Format a single order result as HTML card"""
    order_id = result.get("order_id", "UNKNOWN")
    detected_items = result.get("detected_items", [])
    validation = result.get("validation", {})
    status = result.get("status", "unknown")
    
    missing = validation.get("missing", [])
    extra = validation.get("extra", [])
    qty_mismatch = validation.get("quantity_mismatch", [])
    
    status_class = "validated" if status == "validated" else "mismatch"
    status_icon = "✅" if status == "validated" else "❌"
    status_text = "VALIDATED" if status == "validated" else "MISMATCH"
    status_bg = "#d1fae5" if status == "validated" else "#fee2e2"
    status_color = "#10b981" if status == "validated" else "#ef4444"
    
    # Build items table
    items_rows = ""
    for item in detected_items:
        name = item.get("name", "Unknown")
        qty = item.get("quantity", 0)
        
        item_status = "OK"
        item_color = "#10b981"
        
        if any(m.get("name") == name for m in missing):
            item_status = "Missing"
            item_color = "#ef4444"
        elif any(e.get("name") == name for e in extra):
            item_status = "Extra"
            item_color = "#f59e0b"
        elif any(q.get("name") == name for q in qty_mismatch):
            item_status = "Qty Mismatch"
            item_color = "#f59e0b"
        
        items_rows += f'''
        <tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e9ecef;">{name}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e9ecef; text-align: center;">{qty}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #e9ecef; text-align: center; color: {item_color}; font-weight: 500;">{item_status}</td>
        </tr>
        '''
    
    # Build validation details
    validation_details = ""
    if missing:
        missing_items = ", ".join([f"{m.get('name', 'Unknown')} (×{m.get('quantity', 1)})" for m in missing])
        validation_details += f'<div style="color: #ef4444; font-size: 13px; margin-top: 8px;">⚠️ Missing: {missing_items}</div>'
    if extra:
        extra_items = ", ".join([f"{e.get('name', 'Unknown')} (×{e.get('quantity', 1)})" for e in extra])
        validation_details += f'<div style="color: #f59e0b; font-size: 13px; margin-top: 4px;">➕ Extra: {extra_items}</div>'
    if qty_mismatch:
        qty_items = ", ".join([f"{q.get('name', 'Unknown')}" for q in qty_mismatch])
        validation_details += f'<div style="color: #f59e0b; font-size: 13px; margin-top: 4px;">📊 Qty Mismatch: {qty_items}</div>'
    
    return f'''
    <div style="background: #f8fafc; border-radius: 8px; padding: 16px; margin-top: 12px; border-left: 4px solid {status_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
            <div style="font-weight: 600; color: #1f2937; font-size: 15px;">Order #{order_id}</div>
            <span style="background: {status_bg}; color: {status_color}; padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600;">{status_icon} {status_text}</span>
        </div>
        
        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
            <thead>
                <tr style="background: #e2e8f0;">
                    <th style="padding: 8px 12px; text-align: left; font-weight: 600; color: #475569;">Item</th>
                    <th style="padding: 8px 12px; text-align: center; font-weight: 600; color: #475569;">Qty</th>
                    <th style="padding: 8px 12px; text-align: center; font-weight: 600; color: #475569;">Status</th>
                </tr>
            </thead>
            <tbody>
                {items_rows if items_rows else '<tr><td colspan="3" style="padding: 12px; text-align: center; color: #6b7280;">No items detected</td></tr>'}
            </tbody>
        </table>
        
        {validation_details}
    </div>
    '''

def fetch_video_history():
    """Fetch video history from backend API"""
    try:
        response = requests.get(f"{API_BASE}/videos/history", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("videos", [])
    except Exception as e:
        print(f"Failed to fetch video history: {e}")
    return []

def clear_history():
    """Clear all video history via backend API"""
    try:
        response = requests.delete(f"{API_BASE}/videos/history", timeout=5)
        if response.status_code == 200:
            return format_history_html()
    except Exception as e:
        print(f"Failed to clear video history: {e}")
    return format_history_html()

def format_history_html():
    """Format complete history as expandable HTML sections"""
    # Fetch from backend API
    history = fetch_video_history()
    
    if not history:
        return '''
        <div style="background: #f8f9fa; border-radius: 12px; padding: 40px; text-align: center;">
            <div style="font-size: 48px; margin-bottom: 15px;">📹</div>
            <div style="color: #6c757d; font-size: 14px;">No videos processed yet. Upload a video to see results here.</div>
        </div>
        '''
    
    html_parts = []
    
    for idx, entry in enumerate(history):
        video_name = entry.get("filename", "Unknown")
        # Use upload_time or completed_time from the API response
        timestamp = entry.get("upload_time", entry.get("completed_time", ""))
        results = entry.get("results", [])
        status = entry.get("status", "unknown")
        
        # Calculate summary from results
        validated = sum(1 for r in results if r.get("status") == "validated")
        mismatch = sum(1 for r in results if r.get("status") == "mismatch")
        total = len(results)
        
        # Status badge for video
        if status == "processing":
            video_status = '<span style="background: #e0f2fe; color: #0284c7; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 8px;">⏳ Processing</span>'
        elif status == "completed":
            video_status = '<span style="background: #d1fae5; color: #10b981; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 8px;">✓ Completed</span>'
        elif status == "failed":
            video_status = f'<span style="background: #fee2e2; color: #ef4444; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600; margin-right: 8px;">✗ Failed</span>'
        else:
            video_status = ''
        
        # Summary badge
        if total > 0:
            if validated == total:
                summary_badge = f'<span style="background: #d1fae5; color: #10b981; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">All Validated</span>'
            elif mismatch == total:
                summary_badge = f'<span style="background: #fee2e2; color: #ef4444; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">All Mismatch</span>'
            else:
                summary_badge = f'<span style="background: #fef3c7; color: #d97706; padding: 4px 10px; border-radius: 12px; font-size: 11px; font-weight: 600;">{validated}/{total} Validated</span>'
        else:
            summary_badge = '<span style="background: #f3f4f6; color: #6b7280; padding: 4px 10px; border-radius: 12px; font-size: 11px;">No Results</span>'
        
        # Build validation summary table
        summary_table = format_validation_summary_table(results) if results else ""
        
        # Build order cards
        order_cards = ""
        for result in results:
            order_cards += format_order_card(result)
        
        # Use Gradio accordion-compatible structure
        html_parts.append(f'''
        <details style="background: white; border-radius: 12px; border: 1px solid #e2e8f0; margin-bottom: 16px; overflow: hidden;">
            <summary style="background: linear-gradient(135deg, #0071C5 0%, #00285A 100%); color: white; padding: 14px 20px; cursor: pointer; display: flex; justify-content: space-between; align-items: center; list-style: none;">
                <div>
                    <div style="font-weight: 600; font-size: 15px;">📹 {video_name}</div>
                    <div style="font-size: 12px; opacity: 0.9; margin-top: 4px;">{timestamp}</div>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    {video_status}{summary_badge}
                    <span style="font-size: 18px;">▼</span>
                </div>
            </summary>
            <div style="padding: 20px;">
                <div style="font-weight: 600; color: #00285A; margin-bottom: 8px; font-size: 14px;">📊 Validation Summary</div>
                {summary_table if summary_table else '<div style="color: #6b7280; font-size: 13px;">No results available</div>'}
                
                <div style="font-weight: 600; color: #00285A; margin: 20px 0 8px 0; font-size: 14px;">🧾 Order Details</div>
                {order_cards if order_cards else '<div style="color: #6b7280; font-size: 13px;">No orders detected</div>'}
            </div>
        </details>
        ''')
    
    return "".join(html_parts)

def format_detected_orders():
    """Legacy format function - now returns HTML history"""
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

def refresh_history():
    """Refresh history display"""
    return format_history_html()

# -----------------------------
# UI
# -----------------------------

with gr.Blocks(
    title=APP_TITLE,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=CUSTOM_CSS,
    fill_width=True,
) as demo:
    
    # Header Banner with Intel Branding
    gr.HTML(f'''
        <div class="header-banner">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="margin: 0; font-size: 32px; font-weight: 700;">{APP_TITLE}</h1>
                    <p style="margin: 10px 0 0 0; opacity: 0.95; font-size: 15px;">{APP_DESCRIPTION} • Powered by OpenVINO™ & Vision-Language Models</p>
                </div>
                <div style="text-align: right; font-size: 13px; opacity: 0.9;">
                    <div style="font-weight: 600; font-size: 16px;">intel</div>
                    <div>AI Solutions</div>
                </div>
            </div>
        </div>
    ''')

    with gr.Tabs():

        # ======================
        # FILE UPLOAD TAB
        # ======================
        with gr.TabItem("📁 Upload Video"):
            gr.HTML('<div style="height: 8px;"></div>')
            
            with gr.Row():
                with gr.Column(scale=2):
                    upload_file = gr.File(
                        label="📹 Select Video File",
                        file_types=[".mp4", ".avi", ".mkv", ".mov"],
                        elem_classes=["card-panel"]
                    )
                
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">🎬 Upload Controls</div>')
                    upload_btn = gr.Button(
                        "🚀 Upload & Start Processing",
                        variant="primary",
                        elem_classes=["primary-btn"],
                        size="lg",
                        interactive=True
                    )
                    upload_status = gr.Textbox(
                        label="Processing Status",
                        lines=5,
                        interactive=False
                    )
            
            # Connect upload function with button state management
            upload_btn.click(
                fn=lambda: gr.update(interactive=False),
                outputs=upload_btn
            ).then(
                fn=upload_video_with_progress,
                inputs=upload_file,
                outputs=[upload_status, upload_btn],
                show_progress="full"
            )

        # ======================
        # RTSP STREAM TAB
        # ======================
        with gr.TabItem("📡 RTSP Stream"):
            gr.HTML('<div style="height: 8px;"></div>')
            
            with gr.Row():
                # Left column: Controls
                with gr.Column(scale=1):
                    gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">🔗 Stream Configuration</div>')
                    rtsp_url = gr.Textbox(
                        label="RTSP URL",
                        placeholder="rtsp://host.docker.internal:8554/test",
                        value="rtsp://host.docker.internal:8554/test"
                    )
                    
                    gr.HTML('<div style="height: 16px;"></div>')
                    gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">🎥 Stream Controls</div>')
                    
                    with gr.Row():
                        stream_start_btn = gr.Button("▶️ Start Preview", variant="primary", elem_classes=["primary-btn"])
                        stream_stop_btn = gr.Button("⏹️ Stop", variant="secondary")
                    
                    gr.HTML('<div style="height: 16px;"></div>')
                    gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">🔄 Processing Pipeline</div>')
                    process_btn = gr.Button("🚀 Start Processing", variant="primary", elem_classes=["primary-btn"])
                    
                    processing_status = gr.Textbox(label="Pipeline Status", lines=2, interactive=False)
                
                # Right column: Stream display
                with gr.Column(scale=2):
                    gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">📺 Live Stream Preview</div>')
                    stream_image = gr.Image(
                        label="",
                        width=None,
                        height=400,
                        interactive=False,
                        show_label=False,
                        container=True
                    )
                    stream_status = gr.Textbox(label="Stream Status", lines=2, interactive=False)

            # Connect button functions
            stream_start_btn.click(
                fn=start_smooth_stream,
                inputs=[rtsp_url],
                outputs=[stream_image, stream_status]
            )
            
            stream_stop_btn.click(
                fn=stop_smooth_stream,
                outputs=[stream_image, stream_status]
            )
            
            process_btn.click(
                fn=start_rtsp_processing,
                inputs=[rtsp_url],
                outputs=[processing_status]
            )

        # ======================
        # RESULTS TAB - History View
        # ======================
        with gr.TabItem("📊 Detected Orders"):
            gr.HTML('<div style="height: 8px;"></div>')
            
            gr.HTML('<div style="font-weight: 600; color: #00285A; margin-bottom: 10px; font-size: 15px;">📜 Video Processing History</div>')
            gr.HTML('<p style="color: #6b7280; font-size: 13px; margin-bottom: 16px;">Click on any video entry to expand and view detailed validation results.</p>')
            
            results_history_display = gr.HTML(
                value=format_history_html(),
                elem_classes=["card-panel"]
            )
            
            with gr.Row():
                refresh_btn = gr.Button("🔄 Refresh Results", variant="secondary")
                clear_history_btn = gr.Button("🗑️ Clear History", variant="secondary")
            
            refresh_btn.click(
                fn=refresh_history,
                outputs=results_history_display
            )
            
            clear_history_btn.click(
                fn=clear_history,
                outputs=results_history_display
            )

    # Footer with Intel Branding
    gr.HTML('''
        <div class="footer-info">
            <div style="font-weight: 700; color: #0071C5; font-size: 15px; margin-bottom: 4px;">intel</div>
            <div><strong>AI Solutions</strong> • Take-Away Order Accuracy System</div>
            <div style="margin-top: 4px; font-size: 12px; color: #64748b;">Powered by Qwen2.5-VL Vision-Language Model on OpenVINO™</div>
        </div>
    ''')

# -----------------------------
# ENTRY POINT
# -----------------------------

if __name__ == "__main__":
    print("[Gradio] Starting Order Accuracy UI with Intel styling...")
    
    # Enable queue for generator support with higher concurrency
    demo.queue(concurrency_count=5, max_size=20)
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True,
    )