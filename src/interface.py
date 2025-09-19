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
from final_report import update_metrics,load_metrics_from_file
from collections import deque

TARGET_CLASSES = { "bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","baseball bat","baseball glove","skateboard","surfboard","tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","refrigerator","clock","vase","scissors","teddy bear","hair drier","toothbrush"
}

@lru_cache(maxsize=None)
def create_validate_agent_obj()-> OrderValidator:
    ov = OrderValidator()
    return ov

def call_validator_agent(vlm_op : Dict)-> Dict:
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
        /* Hide Gradio footer bar */
        footer, .svelte-1ipelgc, .svelte-1ipelgc * {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
        }
        /* Style for metrics display */
        #metrics-display {
            font-size: 14px;
            color: #666;
            text-align: right;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            border: 1px solid #e9ecef;
        }
    """, title="Order Accuracy") as demo:
        with gr.Row():
            with gr.Column():
                gr.Markdown(
                    "<div style='text-align: center; font-size: 2em; font-weight: bold;'>Order Accuracy</div>",
                    elem_id="order-accuracy-title"
                )
            

        # Option selection row
        with gr.Row():
            with gr.Column(scale=1):
                option = gr.Radio(
                    choices=["Real-Time Tracking", "Recall Order", "Accuracy Report"],
                    value="Real-Time Tracking",
                    label="Select Feature",
                    interactive=True,
                )

        # Define tracking_col before it is used
        tracking_col = gr.Column(visible=False)  # Removed the invalid 'label' argument

        # Get Order Accuracy UI
        with gr.Column(visible=True) as get_order_col:
            with gr.Row():
                with gr.Column(scale=1):
                    live_webcam = gr.Image(label="Live Stream", streaming=True)
                    rtsp_input = gr.Textbox(label="Video Source")
                    run_rtsp_summary_btn = gr.Button("Analyze Stream")
                    stop_btn = gr.Button("Stop Stream")                
                with gr.Column(scale=1):
                    status = gr.Textbox(label="Order Summary Status", interactive=False, lines=6, max_lines=8)
                    result = gr.JSON(label="Order Accuracy Result")
                    validation_result = gr.JSON(label="Validation Results")  # Add validation results

        # Recall Order Accuracy UI
        with gr.Column(visible=False) as recall_order_col:
            with gr.Row():
                with gr.Column(scale=1):
                    recall_order_number = gr.Textbox(label="Recall Bill/Order Number", placeholder="Enter Bill/Order Number")
                    recall_btn = gr.Button("Recall Order", size="md")
                    recall_video = gr.Video(label="Order Video Preview", interactive=False, height=400, width=600, autoplay=True)
                with gr.Column(scale=1):
                    recall_result = gr.JSON(label="Recalled Order Accuracy Result")
                    recall_validation_result = gr.JSON(label="Recalled Validation Results")  # Add validation results for recall

        # Final Report UI
        with gr.Column(visible=False) as final_report_col:
            with gr.Row():
                with gr.Column(scale=1):
                    final_report_btn = gr.Button("Refresh Final Report", size="md")
                with gr.Column(scale=1):
                    final_report_metrics = gr.JSON(label="Orders Final Report")

        # Option logic: show/hide columns
        def toggle_option(selected):
            return (
                gr.update(visible=(selected == "Real-Time Tracking")),
                gr.update(visible=(selected == "Recall Order")),
                gr.update(visible=(selected == "Accuracy Report"))
            )

        option.change(
            fn=toggle_option,
            inputs=[option],
            outputs=[tracking_col, recall_order_col, final_report_col]
        )
        
        
        clip_queue = queue.Queue()
        stop_event = threading.Event()

        # =========================
        # Consumer Loop
        # =========================
        def clip_consumer():
            """
            Generator that yields updates from processing clips.
            It will exit only after receiving a None sentinel from producer.
            """
            logger.info("[Clips] Consumer started")
            while True:
                clip_path = clip_queue.get()  # blocks until an item is available
                try:
                    if clip_path is None:
                        # producer signaled completion
                        logger.info("[Consumer] received sentinel - exiting")
                        break

                    logger.info(f"[Consumer] Processing {clip_path}")
                    # Call your existing runner pipeline which itself yields progress updates
                    try:
                        for update in runner(clip_path):
                            yield update
                    except Exception as e:
                        logger.exception(f"[Consumer] Error while running pipeline on {clip_path}: {e}")
                        # yield an error status so UI shows it
                        yield (f"🚨 Error processing clip {os.path.basename(clip_path)}: {e}", None, None)
                    # small backoff to avoid busy loop downstream
                    time.sleep(8)
                finally:
                    # mark task done for this item (including sentinel)
                    try:
                        clip_queue.task_done()
                    except Exception:
                        pass

            logger.info("[Clips] Runner finished")
            return

        # =========================
        # Main Orchestration
        # =========================
        def clips_runner(rtsp_url="rtsp://localhost:8554/test"):
            
            # process_rtsp.start_workflow("rtsp://localhost:8554/test", TARGET_CLASSES)
            model = rtsp_video_util.load_model()
            stop_event.clear()
            # Start producer
            t = threading.Thread(
            target=rtsp_video_util.start_stream,  # calls stream_and_split internally
            args=(rtsp_url, model, TARGET_CLASSES, clip_queue, stop_event),
            daemon=True
            )
            t.start()
            logger.info("[Clips] Producer started.")
            for update in clip_consumer():
                yield update

        def stop_workflow():
            # signal producer to stop; producer will enqueue None sentinel when finished
            rtsp_video_util.stop_stream(stop_event)
            
        
        def runner(video):
            try:
                # Initialize status tree structure
                status_tree = {
                    "Video Upload": "⏳ Pending",
                    "Chunking": "⏳ Pending", 
                    "Object Detection": "⏳ Pending",
                    "Getting Best Frames": "⏳ Pending",
                    "Order Accuracy Results": "⏳ Pending",
                    "Validation Results": "⏳ Pending"  # Add validation stage
                }
                
                def format_status_tree():
                    tree_lines = []
                    for stage, status in status_tree.items():
                        tree_lines.append(f"├── {stage}: {status}")
                    return "\n".join(tree_lines)
                
                # Store the final VLM result for validation
                final_vlm_result = None
                
                # Iterate through pipeline progress updates
                for status_text, result_json in process_video(video):
                    if status_text is not None:
                        status_str = str(status_text)
                        # Update status tree based on current stage
                        if "video input" in status_str.lower() or "validation" in status_str.lower() or "upload" in status_str.lower():
                            if "waiting for video input" in status_str.lower():
                                status_tree["Video Upload"] = "⏳ Waiting for input"
                            elif "validated" in status_str.lower():
                                status_tree["Video Upload"] = "🔄 Validating..."
                            elif "upload complete" in status_str.lower():
                                status_tree["Video Upload"] = "✅ Complete"
                                status_tree["Chunking"] = "🔄 Processing..."
                        elif "pipeline activated" in status_str.lower() or "frame analysis" in status_str.lower():
                            status_tree["Chunking"] = "✅ Complete"
                            status_tree["Object Detection"] = "🔄 Running AI models..."
                        elif "processing" in status_str.lower() and "real-time" in status_str.lower():
                            status_tree["Object Detection"] = "🔄 Analyzing frames..."
                        elif "computer vision analysis complete" in status_str.lower():
                            status_tree["Object Detection"] = "✅ Complete"
                            status_tree["Getting Best Frames"] = "🔄 Collecting insights..."
                        elif "decision making" in status_str.lower() and "selecting optimal candidates" in status_str.lower():
                            status_tree["Getting Best Frames"] = "🔄 Analyzing frame quality..."
                        elif "agent selection" in status_str.lower() and "chose top" in status_str.lower():
                            status_tree["Getting Best Frames"] = "✅ Complete"
                        elif "engaging vision-language model" in status_str.lower():
                            status_tree["Order Accuracy Results"] = "🔄 Processing with VLM..."
                        elif "success" in status_str.lower() and "complete" in status_str.lower():
                            status_tree["Order Accuracy Results"] = "✅ Complete"
                            status_tree["Validation Results"] = "🔄 Validating order..."
                            final_vlm_result = result_json
                        elif "error" in status_str.lower() or "failed" in status_str.lower():
                            # Mark current stage as failed
                            for stage in status_tree:
                                if status_tree[stage] == "🔄 Processing..." or "🔄" in status_tree[stage]:
                                    status_tree[stage] = "❌ Failed"
                                    break
                    
                    status_output = format_status_tree()
                    result_output = result_json if isinstance(result_json, (dict, list)) else None

                    status_output = format_status_tree()
                    
                    yield (status_output, result_output, None)  # Add None for validation result
                
                # After VLM processing is complete, run validation
                if final_vlm_result:
                    try:
                        status_tree["Validation Results"] = "🔄 Running validation agent..."
                        status_output = format_status_tree()
                        yield (status_output, final_vlm_result, None)
                        
                        # Call validator agent
                        validated_result = call_validator_agent(final_vlm_result)
                        # Check if validation passed (no issues found)
                        extra_items = validated_result.get("extra_items", [])
                        missing_items = validated_result.get("missing_items", [])
                        missing_addons = validated_result.get("missing_addons", [])
                        count_mismatches = validated_result.get("count_mismatches", [])
                        
                        # Consider validation successful if no issues found
                        validation_passed = (
                            len(extra_items) == 0 and 
                            len(missing_items) == 0 and 
                            len(missing_addons) == 0 and
                            len(count_mismatches) == 0
                        )
                        
                        # Update metrics based on validation result
                        update_metrics(success=validation_passed)
                        
                        logger.info(f"Validation completed: {'PASSED' if validation_passed else 'FAILED'}")
                        logger.info(f"Issues found - Extra: {len(extra_items)}, Missing: {len(missing_items)}, Missing Addons: {len(missing_addons)}, Count Mismatches: {len(count_mismatches)}")
                        
                        status_tree["Validation Results"] = "✅ Complete"
                        status_output = format_status_tree()
                        # Yield final results with validation
                        yield (status_output, final_vlm_result, validated_result)
                        
                    except Exception as validation_error:
                        logger.exception(f"Validation error: {validation_error}")
                        status_tree["Validation Results"] = "❌ Validation failed"
                        status_output = format_status_tree()
                        validation_error_result = {"error": f"Validation failed: {str(validation_error)}"}
                        yield (status_output, final_vlm_result, validation_error_result)
                
            except Exception as e:
                logger.exception(f"Error in runner: {e}")
                # Mark all pending/processing stages as failed
                for stage in status_tree:
                    if "⏳" in status_tree[stage] or "🔄" in status_tree[stage]:
                        status_tree[stage] = "❌ Error occurred"
                error_tree = format_status_tree()
                yield (f"{error_tree}\n\n🚨 System Error: {str(e)}", None, None)

        def recall_order(order_number):
            if not order_number or not str(order_number).strip():
                return gr.update(value=None), gr.update(value=None), gr.update(value=None)
            
            # Fetch JSON (excluding video_id)
            order_json = get_order_json_from_minio(order_number)
            
            # Handle error for missing order id
            if (
                isinstance(order_json, dict)
                and "error" in order_json
            ):
                # Return a valid JSON object for the JSON component
                return {"error": "Order Id doesn't exist"}, None, {"error": "No validation results available"}
            
            # Remove video_id from JSON before showing on UI
            if isinstance(order_json, dict) and "video_id" in order_json:
                video_id = order_json.pop("video_id")
            else:
                video_id = None
            video_url = get_video_url_from_minio(video_id) if video_id else None
            
            # Fetch validation results from MinIO
            validation_results = None
            try:
                # Get validation results using the order number
                MINIO_BUCKET = "order-accuracy-validate-results"
                validation_results = get_order_json_from_minio(order_number,bucket=MINIO_BUCKET )
            except Exception as e:
                logger.error(f"Failed to fetch validation results: {e}")
                validation_results = {"error": f"Failed to fetch validation results: {str(e)}"}
            
            return order_json, video_url, validation_results

        recall_btn.click(
            fn=recall_order,
            inputs=[recall_order_number],
            outputs=[recall_result, recall_video, recall_validation_result],  # Add validation result output
            show_progress=True
        )
        demo.load(fn=webcam_stream, inputs=[], outputs=live_webcam)
        run_rtsp_summary_btn.click(clips_runner, inputs=[rtsp_input], outputs=[status, result, validation_result])
        stop_btn.click(stop_workflow)
        
        def get_final_report_metrics():
            load_metrics_from_file()
            from final_report import get_metrics_dict
            return get_metrics_dict()

        final_report_btn.click(
            fn=get_final_report_metrics,
            outputs=[final_report_metrics],
            show_progress=True
        )
    

    return demo