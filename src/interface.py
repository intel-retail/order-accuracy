"""Order Accuracy VLM Application"""

import os
import threading
import queue
import re, time
import requests
import tempfile
import rtsp_video_util
from typing import Generator, Tuple, Any, Dict
import cv2

import gradio as gr

from config import logger, VSS_IP
from video_utils import validate_video_duration, upload_video
from pipeline import trigger_pipeline, wait_for_pipeline_completion
from messaging import fetch_rabbitmq_metadata, get_top_frames_summary
from vlm import call_vlm
from save_vlm_result import get_order_json_from_minio, get_video_url_from_minio
from functools import lru_cache
from validate_addon import OrderValidator
from final_report import update_metrics, load_metrics_from_file
from collections import deque

TARGET_CLASSES = {
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
    "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "refrigerator", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
}

@lru_cache(maxsize=None)
def create_validate_agent_obj() -> OrderValidator:
    ov = OrderValidator()
    return ov

def call_validator_agent(vlm_op: Dict) -> Dict:
    ov = create_validate_agent_obj()
    validated_items = ov.validate_order(vlm_op)
    return validated_items

def fetch_frame_image(frame_url: str) -> str:
    """
    Fetch frame image from internal service and save to temporary file.
    Returns local file path that Gradio can safely access.
    """
    try:
        logger.info(f"Fetching frame image from: {frame_url}")
        response = requests.get(frame_url, timeout=10)
        response.raise_for_status()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
            
        logger.info(f"Frame image saved to temporary file: {tmp_path}")
        return tmp_path
        
    except Exception as e:
        logger.error(f"Failed to fetch frame image from {frame_url}: {e}")
        return None

def cleanup_temp_files(*file_paths):
    """Clean up temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {file_path}: {e}")

def process_video(video_file) -> Generator[Tuple[str, Any], None, None]:
    """Process uploaded video through the complete pipeline."""
    logger.info("Starting video processing workflow")
    
    if video_file is None:
        logger.warning("No video file provided")
        yield "Agent Status: Waiting for video input to begin analysis...", None
        return
        
    # Gradio may supply dict or path string depending on version
    if isinstance(video_file, dict):
        video_path = video_file.get('name') or video_file.get('path') or video_file.get('data')
    else:
        video_path = video_file
        
    logger.info(f"Processing video file: {video_path}")
    
    if not video_path or not os.path.exists(video_path):
        logger.error(f"Invalid video path: {video_path}")
        yield "🚨 Agent Error: Invalid video file detected. Please upload a valid video.", None
        return

    logger.info("Validating video duration")
    valid, duration, err = validate_video_duration(video_path)
    if not valid:
        logger.error(f"Video validation failed: {err}")
        yield f"🚨 Agent Validation Failed: {err}", None
        return
    yield f"✅ Agent Analysis: Video validated ({duration:.2f}s). Initiating secure upload...", None

    ok, video_id, err = upload_video(video_path, video_path)
    if not ok:
        logger.error(f"Video upload failed: {err}")
        yield f"🚨 Agent Upload Error: {err}", None
        return
    yield f"Agent Upload Complete: Video ID {video_id} assigned. Activating AI pipeline...", None

    ok, pipeline_id, err = trigger_pipeline(video_id, duration, video_path)
    if not ok:
        logger.error(f"Pipeline trigger failed: {err}")
        yield f"🚨 Agent Pipeline Error: {pipeline_id or err}", None
        return
    yield f"🚀 Agent Pipeline Activated: ID {pipeline_id}. Starting intelligent frame analysis...", None

    # Start consuming messages immediately after pipeline trigger
    logger.info("Starting MQTT message consumption in parallel with pipeline execution")
    
    # Start a background thread to collect messages
    message_queue = queue.Queue()
    stop_consuming = threading.Event()
    
    def background_consumer():
        try:
            logger.info("Background consumer started")
            # Collect more frames initially to have better selection of top frames with most objects
            ok, frames, err = fetch_rabbitmq_metadata(limit_frames=3, timeout=90)  # Collect top 3 frames for better selection
            message_queue.put((ok, frames, err))
            logger.info(f"Background consumer finished: ok={ok}, frames={len(frames) if frames else 0}")
        except Exception as e:
            logger.exception(f"Background consumer error: {e}")
            message_queue.put((False, [], f"Background consumer error: {e}"))
    
    consumer_thread = threading.Thread(target=background_consumer)
    consumer_thread.start()
    
    yield "🔄 Agent Processing: AI models analyzing video frames in real-time...", None

    ok, err = wait_for_pipeline_completion(pipeline_id)
    if not ok:
        logger.error(f"Pipeline completion failed: {err}")
        stop_consuming.set()
        yield f"🚨 Agent Pipeline Failed: {err}", None
        return
        
    yield "🧠 Agent Intelligence: Computer vision analysis complete. Collecting insights...", None
    
    # Wait for the consumer thread to finish (with timeout)
    consumer_thread.join(timeout=60)  # Wait up to 30 more seconds
    
    # Get the results from the background consumer
    try:
        ok, frames, err = message_queue.get_nowait()
    except queue.Empty:
        ok, frames, err = False, [], "Metadata collection timed out"
    
    if not ok:
        logger.error(f"Frame metadata fetch failed: {err}")
        yield f"🚨 Agent Metadata Error: {err}", None
        return
    
    logger.info(f"Retrieved {len(frames)} frames from metadata collection")
    
    # Select exactly top 3 frames with most objects for VLM analysis
    top_3_frames = frames[:3] if len(frames) >= 3 else frames
    logger.info(f"Selected top {len(top_3_frames)} frames with most objects for VLM analysis")
    
    # Generate and log summary of top frames
    frames_summary = get_top_frames_summary(frames, top_n=3)
    logger.info(f"Frame analysis summary:\n{frames_summary}")
    
    yield f"Agent Decision Making: Analyzed {len(frames)} frames, selecting optimal candidates for VLM...", None
    
    # Provide detailed feedback to user about selected frames
    if top_3_frames:
        top_frame_names = [frame.get('frame_name', f"frame_{frame.get('frame_id', 'unknown')}") for frame in top_3_frames]
        frame_counts = [frame.get('object_count', 0) for frame in top_3_frames]
        summary_msg = f"Agent Selection: Chose top {len(top_3_frames)} frames with highest object density: " + ", ".join([f"{name}({count} objects)" for name, count in zip(top_frame_names, frame_counts)])
        yield f"{summary_msg}. Engaging Vision-Language Model...", None
    else:
        yield f"Agent Analysis: No suitable frames detected for VLM processing.", None
        return

    ok, result_json, err = call_vlm(top_3_frames, video_id=video_id)
    if not ok:
        logger.error(f"VLM call failed: {err}")
        yield f" Agent VLM Error: {err}", None
        return
    
    logger.info("Video processing completed successfully")
    yield " Agent Success: Grocery items identified and receipt generation complete.", result_json

def webcam_stream():
    cap = cv2.VideoCapture("/dev/video0")  # direct webcam device
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(0.05)  # ~20 fps
    cap.release()

def build_interface():
    """Build and return the Gradio interface."""
    with gr.Blocks(css="""
        /* --- Layout restoration: full-height app & internal scroll --- */
        html, body, .gradio-container, #oa-shell {
            height:100%;
            margin:0;
            padding:0;
        }
        body {
            overflow:hidden; /* prevent body scrolling; we'll scroll #oa-content */
        }
        /* Estimate total header (title + tabs) height; adjust if needed */
        #oa-header { --oa-header-total: 150px; }
        /* Scrollable content area below sticky header */
        #oa-content {
            height:calc(100vh - 150px);
            overflow-y:auto;
            overflow-x:hidden;
            padding:4px 0 40px 0;
            scrollbar-width:thin;
        }
        #oa-content::-webkit-scrollbar { width:8px; }
        #oa-content::-webkit-scrollbar-track { background:transparent; }
        #oa-content::-webkit-scrollbar-thumb {
            background:#c2ccd4;
            border-radius:4px;
        }
        #oa-content::-webkit-scrollbar-thumb:hover { background:#a9b4bd; }

        /* Header & Title (already present, kept) */
        #oa-header {
            position:sticky;
            top:0;
            z-index:1000;
            background:linear-gradient(180deg,#ffffff 0%, #ffffffee 92%);
            backdrop-filter:blur(6px);
            -webkit-backdrop-filter:blur(6px);
            padding:14px 30px 4px;
            margin:0;
        }
        #oa-title {
            text-align:center;
            font-size:1.6rem;
            font-weight:600;
            letter-spacing:.35px;
            color:#1f2f33;
            margin:0 0 8px 0;
        }

        /* Tabs (unchanged visual pill style) */
        .oa-tabs {
            display:flex;
            justify-content:center;
            gap:12px;
            padding:6px 8px 10px;
            margin:0;
            background:linear-gradient(180deg,#ffffff 0%, #f5f7fa 100%);
            border-bottom:1px solid #d4dce1;
            position:relative;
            top:auto;
            z-index:950;
        }
        .oa-tab-btn {
            position:relative;
            background:linear-gradient(180deg,#ffffff 0%, #f2f5f7 100%) !important;
            border:1px solid #ccd5db !important;
            padding:8px 22px 10px 22px !important;
            font-size:.84rem !important;
            font-weight:600;
            letter-spacing:.3px;
            color:#4a5b66 !important;
            cursor:pointer;
            border-radius:20px;
            line-height:1;
            transition:background .18s, color .18s, box-shadow .18s, border-color .18s, transform .12s;
        }
        .oa-tab-btn:hover {
            background:#e6f0f6 !important;
            color:#0a6ca9 !important;
            border-color:#b9c5cc !important;
        }
        .oa-tab-btn:active { transform:translateY(1px); }
        .oa-tab-btn:focus-visible { outline:2px solid #0a6ca9; outline-offset:2px; }

        .oa-tabs[data-active="rt-tab"] #rt-tab,
        .oa-tabs[data-active="rc-tab"] #rc-tab,
        .oa-tabs[data-active="ar-tab"] #ar-tab {
            background:#d7edf7 !important;
            color:#0a5d90 !important;
            border-color:#7fb4d3 !important;
            box-shadow:0 2px 6px rgba(10,108,169,.28), 0 0 0 1px #b4d7e6 inset;
            font-weight:700 !important;
        }
        .oa-tabs[data-active="rt-tab"] #rt-tab::after,
        .oa-tabs[data-active="rc-tab"] #rc-tab::after,
        .oa-tabs[data-active="ar-tab"] #ar-tab::after {
            content:"";
            position:absolute;
            left:16px; right:16px; bottom:-7px;
            height:3px;
            background:#0a6ca9;
            border-radius:2px;
        }
i        /* ...existing CSS below remains unchanged... */
    """, title="Order Accuracy") as demo:
        with gr.Column(elem_id="oa-shell"):
            with gr.Column(elem_id="oa-header"):
                gr.Markdown("<div id='oa-title'>Order Accuracy</div>")
                with gr.Row(elem_id="oa-tabs-bar", elem_classes=["oa-tabs"]):
                    real_time_tab = gr.Button("Real-Time Tracking", elem_id="rt-tab", elem_classes=["oa-tab-btn"])
                    recall_order_tab = gr.Button("Recall Order", elem_id="rc-tab", elem_classes=["oa-tab-btn"])
                    accuracy_report_tab = gr.Button("Accuracy Report", elem_id="ar-tab", elem_classes=["oa-tab-btn"])
            with gr.Column(elem_id="oa-content"):
                with gr.Column(visible=True, elem_classes=["gradio-section"], elem_id="real-time-section") as get_order_col:
                    with gr.Row():
                        with gr.Column(scale=1):
                            live_webcam = gr.Image(label="Live Stream", streaming=True)
                            rtsp_input = gr.Textbox(label="Video Source", value="rtsp://localhost:8554/test")
                            run_rtsp_summary_btn = gr.Button("Analyze Stream")
                            stop_btn = gr.Button("Stop Stream")
                        with gr.Column(scale=1):
                            status = gr.Textbox(label="Order Summary Status", interactive=False, lines=8, max_lines=10)
                            result = gr.JSON(label="Order Accuracy Result")
                            validation_result = gr.JSON(label="Validation Results")

                with gr.Column(visible=False, elem_classes=["gradio-section"], elem_id="recall_order_col") as recall_order_col:
                    with gr.Row():
                        with gr.Column(scale=1):
                            recall_order_number = gr.Textbox(label="Recall Bill/Order Number", placeholder="Enter Bill/Order Number")
                            recall_btn = gr.Button("Recall Order", size="md")
                            recall_video = gr.Video(label="Order Video Preview", height=400, width=600, autoplay=True)
                        with gr.Column(scale=1):
                            recall_result = gr.JSON(label="Recalled Order Accuracy Result")
                            recall_validation_result = gr.JSON(label="Recalled Validation Results")

                with gr.Column(visible=False, elem_classes=["gradio-section","accuracy-report-section"], elem_id="final_report_col") as final_report_col:
                    with gr.Row():
                        final_report_btn = gr.Button("Refresh Final Report", size="md")
                    with gr.Row():
                        final_report_metrics = gr.JSON(label="Orders Final Report")

                    clip_queue = queue.Queue()
                    stop_event = threading.Event()

                    def clip_consumer():
                        logger.info("[Clips] Consumer started")
                        while True:
                            clip_path = clip_queue.get()
                            try:
                                if clip_path is None:
                                    logger.info("[Clips] Consumer received sentinel - exiting")
                                    break
                                logger.info(f"[Clips] Processing clip: {clip_path}")
                                for update in runner(clip_path):
                                    yield update
                                time.sleep(2)
                            except Exception as e:
                                logger.exception(f"[Clips] Error processing {clip_path}: {e}")
                                yield (f"🚨 Error processing clip {os.path.basename(clip_path)}: {e}", None, None)
                            finally:
                                try:
                                    clip_queue.task_done()
                                except Exception:
                                    pass
                        logger.info("[Clips] Consumer done")

                    def clips_runner(rtsp_url="rtsp://localhost:8554/test"):
                        model = rtsp_video_util.load_model()
                        stop_event.clear()
                        t = threading.Thread(
                            target=rtsp_video_util.start_stream,
                            args=(rtsp_url, model, TARGET_CLASSES, clip_queue, stop_event),
                            daemon=True
                        )
                        t.start()
                        logger.info("[Clips] Producer started.")
                        for update in clip_consumer():
                            yield update

                    def stop_workflow():
                        rtsp_video_util.stop_stream(stop_event)

                    def runner(video):
                        try:
                            status_tree = {
                                "Video Upload": "⏳ Pending",
                                "Chunking": "⏳ Pending",
                                "Object Detection": "⏳ Pending",
                                "Getting Best Frames": "⏳ Pending",
                                "Order Accuracy Results": "⏳ Pending",
                                "Validation Results": "⏳ Pending"
                            }

                            def format_tree():
                                return "\n".join([f"├── {k}: {v}" for k,v in status_tree.items()])

                            final_vlm_result = None

                            for status_text, result_json in process_video(video):
                                if status_text:
                                    st = status_text.lower()
                                    if "waiting for video input" in st:
                                        status_tree["Video Upload"] = "⏳ Waiting for input"
                                    elif "validated" in st:
                                        status_tree["Video Upload"] = "🔄 Validating..."
                                    elif "upload complete" in st:
                                        status_tree["Video Upload"] = "✅ Complete"
                                        status_tree["Chunking"] = "🔄 Processing..."
                                    elif "pipeline activated" in st or "frame analysis" in st:
                                        status_tree["Chunking"] = "✅ Complete"
                                        status_tree["Object Detection"] = "🔄 Running AI models..."
                                    elif "real-time" in st and "processing" in st:
                                        status_tree["Object Detection"] = "🔄 Analyzing frames..."
                                    elif "computer vision analysis complete" in st:
                                        status_tree["Object Detection"] = "✅ Complete"
                                        status_tree["Getting Best Frames"] = "🔄 Collecting insights..."
                                    elif "decision making" in st and "selecting optimal candidates" in st:
                                        status_tree["Getting Best Frames"] = "🔄 Analyzing frame quality..."
                                    elif "chose top" in st and "engaging" in st:
                                        status_tree["Getting Best Frames"] = "✅ Complete"
                                        status_tree["Order Accuracy Results"] = "🔄 Processing with VLM..."
                                    elif "success" in st and "receipt generation complete" in st:
                                        status_tree["Order Accuracy Results"] = "✅ Complete"
                                        status_tree["Validation Results"] = "🔄 Validating order..."
                                        final_vlm_result = result_json
                                    elif "error" in st or "failed" in st:
                                        for k,v in status_tree.items():
                                            if "⏳" in v or "🔄" in v:
                                                status_tree[k] = "❌ Failed"
                                                break
                                yield (format_tree(), result_json if isinstance(result_json,(dict,list)) else None, None)

                            if final_vlm_result:
                                try:
                                    status_tree["Validation Results"] = "🔄 Running validation agent..."
                                    yield (format_tree(), final_vlm_result, None)
                                    validated = call_validator_agent(final_vlm_result)
                                    extra_items = validated.get("extra_items", [])
                                    missing_items = validated.get("missing_items", [])
                                    missing_addons = validated.get("missing_addons", [])
                                    count_mismatches = validated.get("count_mismatches", [])
                                    passed = not (extra_items or missing_items or missing_addons or count_mismatches)
                                    update_metrics(success=passed)
                                    status_tree["Validation Results"] = "✅ Complete"
                                    yield (format_tree(), final_vlm_result, validated)
                                except Exception as ve:
                                    logger.exception(f"Validation error: {ve}")
                                    status_tree["Validation Results"] = "❌ Validation failed"
                                    yield (format_tree(), final_vlm_result, {"error": f"Validation failed: {ve}"})
                        except Exception as e:
                            logger.exception(f"Runner error: {e}")
                            yield (f"🚨 System Error: {e}", None, None)

                    def recall_order(order_number):
                        if not order_number or not str(order_number).strip():
                            return gr.update(value=None), gr.update(value=None), gr.update(value=None)
                        order_json = get_order_json_from_minio(order_number)
                        if isinstance(order_json, dict) and "error" in order_json:
                            return {"error": "Order Id doesn't exist"}, None, {"error": "No validation results available"}
                        if isinstance(order_json, dict) and "video_id" in order_json:
                            vid = order_json.pop("video_id")
                        else:
                            vid = None
                        video_url = get_video_url_from_minio(vid) if vid else None
                        try:
                            MINIO_BUCKET = "order-accuracy-validate-results"
                            validation_results = get_order_json_from_minio(order_number, bucket=MINIO_BUCKET)
                        except Exception as e:
                            logger.error(f"Failed to fetch validation results: {e}")
                            validation_results = {"error": f"Failed to fetch validation results: {e}"}
                        return order_json, video_url, validation_results

                    def switch_tab(tab_name):
                        return (
                            gr.update(visible=(tab_name == "Real-Time Tracking")),
                            gr.update(visible=(tab_name == "Recall Order")),
                            gr.update(visible=(tab_name == "Accuracy Report"))
                        )

                    real_time_tab.click(fn=lambda: switch_tab("Real-Time Tracking"),
                                        inputs=[], outputs=[get_order_col, recall_order_col, final_report_col])
                    recall_order_tab.click(fn=lambda: switch_tab("Recall Order"),
                                           inputs=[], outputs=[get_order_col, recall_order_col, final_report_col])
                    accuracy_report_tab.click(fn=lambda: switch_tab("Accuracy Report"),
                                              inputs=[], outputs=[get_order_col, recall_order_col, final_report_col])

                    demo.load(fn=webcam_stream, inputs=[], outputs=live_webcam)
                    run_rtsp_summary_btn.click(clips_runner, inputs=[rtsp_input], outputs=[status, result, validation_result])
                    stop_btn.click(stop_workflow)

                    def get_final_report_metrics():
                        load_metrics_from_file()
                        from final_report import get_metrics_dict
                        return get_metrics_dict()

                    final_report_btn.click(fn=get_final_report_metrics, outputs=[final_report_metrics], show_progress=True)
                    recall_btn.click(fn=recall_order, inputs=[recall_order_number],
                                     outputs=[recall_result, recall_video, recall_validation_result], show_progress=True)

        gr.HTML("""
<script>
(function(){
  const MAP = {
    'rt-tab':'real-time-section',
    'rc-tab':'recall_order_col',
    'ar-tab':'final_report_col'
  };
  const IDS = Object.keys(MAP);
  const KEY = 'oa_active_tab_simple';
  const bar = document.getElementById('oa-tabs-bar');

  function el(id){ return document.getElementById(id); }

  function visible(id){
    const n = el(id);
    if(!n) return false;
    if(n.hidden) return false;
    const cs = getComputedStyle(n);
    return cs.display !== 'none' && cs.visibility !== 'hidden' && cs.opacity !== '0';
  }

  function detectVisible(){
    for(const t of IDS){
      if(visible(MAP[t])) return t;
    }
    return IDS[0];
  }

  function stored(){
    const s = localStorage.getItem(KEY);
    return IDS.includes(s) ? s : null;
  }

  function apply(id, persist=true){
    if(!bar) return;
    if(bar.dataset.active !== id){
      bar.dataset.active = id;
    }
    if(persist) localStorage.setItem(KEY,id);
  }

  function handleClick(id){
    apply(id,true);
    setTimeout(()=>apply(detectVisible(),true),80);
  }

  function bind(){
    IDS.forEach(id=>{
      const b = el(id);
      if(b && !b.dataset.bound){
        b.dataset.bound='1';
        b.style.cursor='pointer';
        b.addEventListener('click', ()=>handleClick(id));
      }
    });
  }

  function init(){
    bind();
    const s = stored();
    if(s){
      apply(s,false);
      const vis = detectVisible();
      if(vis !== s) apply(vis,true);
    } else {
      apply(detectVisible(), true);
    }
  }

  const mo = new MutationObserver(()=>bind());
  mo.observe(document.body,{subtree:true, childList:true});

  setInterval(()=> {
    const s = stored();
    if(s) apply(s,false);
  }, 2500);

  if(document.readyState !== 'loading') init();
  else document.addEventListener('DOMContentLoaded', init);
})();
</script>
""")
        return demo
