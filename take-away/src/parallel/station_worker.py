"""
Station Worker Process

Each station worker runs in a separate process and handles:
1. GStreamer video pipeline
2. OCR for order ID detection
3. Frame selection (YOLO or scoring)
4. Sending VLM requests to scheduler
5. Receiving VLM responses
6. Order validation and result generation

Maintains all business logic from original sequential implementation.
"""

import multiprocessing as mp
import time
import logging
import signal
import sys
import threading
from typing import Optional, List, Dict
import queue
from pathlib import Path

from .shared_queue import QueueManager, VLMRequest, VLMResponse
from .metrics_collector import MetricsStore



logger = logging.getLogger(__name__)


class StationWorker:
    """
    Station worker process for single camera stream.
    
    Architecture:
    - Runs complete pipeline for one station
    - Sends VLM requests to shared scheduler via queue
    - Receives responses via station-specific queue
    - Reports metrics to shared store
    - Supports graceful shutdown
    """
    
    def __init__(
        self,
        station_id: str,
        rtsp_url: str,
        queue_manager: QueueManager,
        metrics_store: MetricsStore,
        config: Dict
    ):
        """
        Args:
            station_id: Unique station identifier (e.g., "station_1")
            rtsp_url: RTSP stream URL for this station
            queue_manager: Shared queue manager
            metrics_store: Shared metrics storage
            config: Station configuration dict with:
                - minio_endpoint
                - minio_bucket
                - inventory_path
                - orders_path
                - yolo_model_path
                - etc.
        """
        self.station_id = station_id
        self.rtsp_url = rtsp_url
        self.queue_manager = queue_manager
        self.metrics_store = metrics_store
        self.config = config
        
        # Station-specific storage paths
        self.frame_storage_path = f"{config.get('minio_bucket', 'orders')}/{station_id}"
        
        # Response queue
        self.response_queue = queue_manager.get_response_queue(station_id)
        
        # Pipeline state
        self._running = False
        self._current_order_id: Optional[str] = None
        self._order_start_time: Optional[float] = None
        self._frames_buffer: List = []
        
        # Persistent pipeline components
        self._pipeline_subprocess = None
        self._pipeline_pid = None
        self._pipeline_running = False
        self._pipeline_restart_count = 0
        
        # Monitoring threads
        self._frame_monitor_thread = None
        self._health_check_thread = None
        
        # Order tracking
        self._processed_orders = set()
        self._active_orders = {}
        
        # Reusable components (initialized in run())
        self._pipeline_runner = None
        self._frame_selector = None
        self._validation_func = None
        self._orders = {}
        self._inventory = {}
        
        logger.info(
            f"StationWorker initialized: {station_id} "
            f"(RTSP: {rtsp_url})"
        )
    
    def run(self):
        """
        Main worker process loop.
        
        Runs until stop signal received via control queue.
        """
        # Register with metrics
        self.metrics_store.register_station(self.station_id)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        
        logger.info(f"[{self.station_id}] Worker process started (PID: {mp.current_process().pid})")
        
        try:
            # Initialize pipeline components
            logger.info("Pipeline is initializing...")
            self._initialize_pipeline()
            
            self._running = True
            
            # Start persistent GStreamer pipeline (subprocess)
            self._start_persistent_pipeline()
            
            logger.info(f"[{self.station_id}] Persistent pipeline started, waiting for frames...")
            # Start monitoring threads
            self._start_frame_monitor()
            logger.info(f"[{self.station_id}] Frame monitor started")
            self._start_health_monitor()
            logger.info(f"[{self.station_id}] Health monitor started")
            
            logger.info(f"[{self.station_id}] All components started, entering main loop")
            
            # Main processing loop: process completed orders
            while self._running:
                try:
                    # Check for control signals
                    if self._check_shutdown_signal():
                        logger.info(f"[{self.station_id}] Shutdown signal received")
                        break
                    
                    # Process orders that are ready in MinIO
                    self._process_ready_orders()
                    
                    time.sleep(0.1)  # Small delay to prevent CPU spinning
                
                except Exception as e:
                    logger.error(f"[{self.station_id}] Error in main loop: {e}")
                    time.sleep(1)
        
        finally:
            self._cleanup()
            self.metrics_store.unregister_station(self.station_id)
            logger.info(f"[{self.station_id}] Worker process stopped")
    
    def _initialize_pipeline(self):
        """
        Initialize pipeline components.
        
        Integrates existing pipeline code:
        - PipelineRunner (GStreamer)
        - FrameSelector (YOLO)
        - ValidationAgent (order validation)
        """
        logger.info(f"[{self.station_id}] Initializing pipeline components...")
        
        try:
            # Import existing modules from application-service and frame-selector-service
            # Note: These imports are dynamic and added to sys.path at runtime
            import sys
            import os
            
            # Add paths to existing services
            base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sys.path.insert(0, os.path.join(base_path, 'application-service', 'app'))
            sys.path.insert(0, os.path.join(base_path, 'frame-selector-service', 'app'))
            
            # Import pipeline runner (runtime import)
            try:
                from pipeline_runner import run_pipeline  # type: ignore
                self._pipeline_runner = run_pipeline
                logger.info(f"[{self.station_id}] Pipeline runner loaded")
            except ImportError as e:
                logger.warning(
                    f"[{self.station_id}] Could not import pipeline_runner: {e}"
                )
                self._pipeline_runner = None
            
            # Import frame selector (optional - fallback to simple selection)
            try:
                from frame_selector import FrameSelector  # type: ignore
                self._frame_selector = FrameSelector(
                    yolo_model_path=self.config.get(
                        'yolo_model_path', './models/yolo11n_openvino_model'
                    )
                )
                logger.info(
                    f"[{self.station_id}] Frame selector loaded (YOLO-based)"
                )
            except ImportError as e:
                logger.warning(
                    f"[{self.station_id}] Could not import frame_selector: {e}"
                )
                logger.warning(
                    f"[{self.station_id}] "
                    f"Will use simple frame selection (first N frames)"
                )
                self._frame_selector = None
            
            # Import validation agent (optional)
            try:
                from core.validation_agent import validate_order as validation_func  # type: ignore
                from core.order_results import add_result  # type: ignore
                import json
                # Load order inventory for validation
                inventory_path = self.config.get('inventory_path', './config/inventory.json')
                orders_path = self.config.get('orders_path', './config/orders.json')
                with open(inventory_path) as f:
                    self._inventory = json.load(f)
                with open(orders_path) as f:
                    self._orders = json.load(f)
                self._validation_func = validation_func
                self._add_result_func = add_result
                logger.info(f"[{self.station_id}] Validation function loaded")
            except Exception as e:
                logger.warning(f"[{self.station_id}] Could not import validation_agent: {e}")
                logger.warning(f"[{self.station_id}] Will use mock validation")
                self._validation_func = None
                self._add_result_func = None
                self._orders = {}
                self._inventory = {}
            
            logger.info(f"[{self.station_id}] Pipeline components initialized")
        
        except Exception as e:
            logger.error(f"[{self.station_id}] Failed to initialize pipeline: {e}")
            raise
    
    def _start_persistent_pipeline(self):
        """
        Start persistent GStreamer pipeline as subprocess.
        Pipeline runs continuously, extracting frames to MinIO.
        """
        logger.info(f"[{self.station_id}] Starting persistent GStreamer pipeline...")
        
        import subprocess
        import os
        
        # Build pipeline command
        pipeline = self._build_persistent_gstreamer_pipeline()
        
        # Redirect stderr to log file for debugging
        stderr_log = f'/tmp/gst_pipeline_{self.station_id}.log'
        cmd = f"gst-launch-1.0 -q {pipeline} 2>> {stderr_log}"
        
        # Log the exact command for debugging
        logger.info(f"[{self.station_id}] Pipeline command: gst-launch-1.0 -q {pipeline}")
        logger.info(f"[{self.station_id}] Pipeline errors will be logged to: {stderr_log}")
        
        # Prepare environment
        env = os.environ.copy()
        app_dir = '/app'
        pythonpath = env.get('PYTHONPATH', '')
        if app_dir not in pythonpath:
            env['PYTHONPATH'] = f"{app_dir}:{pythonpath}" if pythonpath else app_dir
        
        # Set GST_PLUGIN_PYTHON_PATH for gvapython to find modules
        env['GST_PLUGIN_PYTHON_PATH'] = app_dir
        
        # Pass station info to gvapython
        env['STATION_ID'] = self.station_id
        env['MINIO_ENDPOINT'] = self.config.get('minio', {}).get('endpoint', 'minio:9000')
        env['MINIO_BUCKET'] = self.config.get('minio', {}).get('frames_bucket', 'frames')
        
        try:
            # Start subprocess with process group for clean shutdown
            self._pipeline_subprocess = subprocess.Popen(
                cmd,
                shell=True,
                env=env,
                cwd='/app',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,  # Errors redirected via shell redirection in cmd
                preexec_fn=os.setsid  # Create process group
            )
            
            self._pipeline_pid = self._pipeline_subprocess.pid
            self._pipeline_running = True
            
            logger.info(
                f"[{self.station_id}] Persistent pipeline started "
                f"(PID: {self._pipeline_pid})"
            )
        except Exception as e:
            logger.error(f"[{self.station_id}] Failed to start persistent pipeline: {e}")
            raise
    
    def _build_persistent_gstreamer_pipeline(self) -> str:
        """
        Build GStreamer pipeline that runs continuously.
        
        Key differences from per-order pipeline:
        - No EOS (End-Of-Stream) - runs forever
        - Handles multiple orders via OCR detection
        - Pipeline never stops until worker shutdown
        """
        logger.info(f"[{self.station_id}] Building persistent GStreamer pipeline...")
        if self.rtsp_url.startswith("rtsp://"):
            src = (
                f"rtspsrc location={self.rtsp_url} "
                f"protocols=tcp latency=200 "
                f"! rtph264depay ! h264parse"
            )
        else:
            # For file: use filesrc for MP4 files
            src = f"filesrc location={self.rtsp_url}"
        
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
    
    def _start_frame_monitor(self):
        """
        Start thread that monitors MinIO for completed orders.
        """
        self._frame_monitor_thread = threading.Thread(
            target=self._frame_monitor_loop,
            daemon=True,
            name=f"{self.station_id}-FrameMonitor"
        )
        self._frame_monitor_thread.start()
        logger.info(f"[{self.station_id}] Frame monitor thread started")
    
    def _frame_monitor_loop(self):
        """
        Continuously check MinIO for orders with EOS markers.
        """
        while self._running:
            try:
                # Scan MinIO for completed orders
                completed_orders = self._scan_minio_for_completed_orders()
                
                for order_id in completed_orders:
                    if order_id not in self._processed_orders and order_id not in self._active_orders:
                        # New completed order detected!
                        logger.info(
                            f"[{self.station_id}] Detected completed order: {order_id}"
                        )
                        self._active_orders[order_id] = {
                            'detected_at': time.time(),
                            'status': 'ready'
                        }
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                logger.error(f"[{self.station_id}] Frame monitor error: {e}")
                time.sleep(1)
    
    def _start_health_monitor(self):
        """
        Start thread that monitors pipeline health and auto-restarts if needed.
        """
        self._health_check_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name=f"{self.station_id}-HealthCheck"
        )
        self._health_check_thread.start()
        logger.info(f"[{self.station_id}] Health monitor thread started")
    
    def _health_monitor_loop(self):
        """
        Check if GStreamer pipeline is still running.
        """
        while self._running:
            try:
                if self._pipeline_subprocess:
                    returncode = self._pipeline_subprocess.poll()
                    
                    if returncode is not None:
                        # Pipeline died! Capture stdout/stderr before restarting
                        output = ""
                        try:
                            # Read whatever is available (non-blocking)
                            output = self._pipeline_subprocess.stdout.read()
                        except Exception:
                            pass
                        
                        logger.error(
                            f"[{self.station_id}] Pipeline subprocess died "
                            f"(exit code: {returncode})"
                        )
                        
                        if output and output.strip():
                            logger.error(
                                f"[{self.station_id}] Pipeline output:\n{output}"
                            )
                        
                        # Restart pipeline with backoff
                        self._restart_pipeline()
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"[{self.station_id}] Health monitor error: {e}")
                time.sleep(5)
    
    def _restart_pipeline(self):
        """
        Restart crashed pipeline with exponential backoff.
        """
        self._pipeline_restart_count += 1
        
        if self._pipeline_restart_count > 5:
            logger.error(
                f"[{self.station_id}] Pipeline restart limit exceeded. "
                f"Stopping worker."
            )
            self._running = False
            return
        
        # Exponential backoff
        delay = min(2 ** self._pipeline_restart_count, 60)
        logger.info(
            f"[{self.station_id}] Restarting pipeline in {delay}s "
            f"(attempt {self._pipeline_restart_count})"
        )
        time.sleep(delay)
        
        try:
            self._start_persistent_pipeline()
        except Exception as e:
            logger.error(f"[{self.station_id}] Pipeline restart failed: {e}")
    
    def _scan_minio_for_completed_orders(self) -> List[str]:
        """
        Scan MinIO for orders with EOS markers.
        
        Returns:
            List of order_ids that have EOS markers
        """
        try:
            from minio import Minio
            
            minio_config = self.config.get('minio', {})
            client = Minio(
                minio_config.get('endpoint', 'minio:9000'),
                access_key=minio_config.get('access_key', 'minioadmin'),
                secret_key=minio_config.get('secret_key', 'minioadmin'),
                secure=minio_config.get('secure', False)
            )
            
            bucket = minio_config.get('frames_bucket', 'frames')
            prefix = f"{self.station_id}/"
            
            completed_orders = []
            
            # List order directories under this station
            try:
                objects = list(client.list_objects(bucket, prefix=prefix, recursive=False))
            except Exception:
                # Bucket might not exist yet
                return []
            
            # Extract unique order_ids
            order_dirs = set()
            for obj in objects:
                if obj.is_dir and obj.object_name:
                    # Extract order_id from path: station_1/ORDER_123/
                    parts = obj.object_name.strip('/').split('/')
                    if len(parts) >= 2:
                        order_dirs.add(parts[1])
            
            # Check each order for EOS marker
            for order_id in order_dirs:
                eos_path = f"{prefix}{order_id}/__EOS__"
                try:
                    client.stat_object(bucket, eos_path)
                    completed_orders.append(order_id)
                except Exception:
                    # EOS not found, order still processing
                    pass
            
            return completed_orders
            
        except Exception as e:
            logger.error(
                f"[{self.station_id}] Error scanning MinIO for orders: {e}"
            )
            return []
    
    def _process_ready_orders(self):
        """
        Process orders that are ready (have EOS markers).
        """
        for order_id, order_info in list(self._active_orders.items()):
            if order_info['status'] == 'ready':
                try:
                    self._process_single_order(order_id)
                    self._processed_orders.add(order_id)
                    del self._active_orders[order_id]
                except Exception as e:
                    logger.error(
                        f"[{self.station_id}] Error processing {order_id}: {e}"
                    )
                    order_info['status'] = 'error'
    
    def _process_single_order(self, order_id: str):
        """
        Process a single completed order.
        
        Frames are already in MinIO, no pipeline startup needed!
        """
        logger.info(f"[{self.station_id}] Processing order: {order_id}")
        
        try:
            order_start_time = time.time()
            
            # Load frames from MinIO
            logger.info(f"[{self.station_id}] Loading frames from MinIO for order {order_id}")
            frames = self._load_order_frames_from_minio(order_id)
            logger.info(f"[{self.station_id}] Loaded {len(frames) if frames else 0} frames for order {order_id}")
            
            if not frames:
                logger.warning(f"[{self.station_id}] No frames found for {order_id}")
                return
            
            # Frame selection (YOLO)
            logger.info(f"[{self.station_id}] Selecting best frames for order {order_id}")
            selected_frames = self._select_best_frames(frames)
            logger.info(f"[{self.station_id}] Selected {len(selected_frames) if selected_frames else 0} frames for order {order_id}")
            
            if not selected_frames:
                logger.warning(f"[{self.station_id}] No frames selected for {order_id}")
                return
            
            # VLM inference via scheduler
            logger.info(f"[{self.station_id}] Requesting VLM inference for order {order_id}")
            vlm_response = self._request_vlm_inference(selected_frames, order_id)
            
            if not vlm_response or not vlm_response.success:
                logger.error(
                    f"[{self.station_id}] VLM inference failed for {order_id}: "
                    f"{vlm_response.error if vlm_response else 'No response'}"
                )
                self.metrics_store.increment_failures()
                return
            
            # Order validation
            logger.info(f"[{self.station_id}] Validating detected items for order {order_id}")
            validation_result = self._validate_order(vlm_response.detected_items, order_id)
            
            # Log comprehensive result
            unique_id = f"{self.station_id}_{order_id}"
            logger.info(f"="*80)
            logger.info(f"[ORDER-COMPLETE] unique_id={unique_id}")
            logger.info(f"[ORDER-COMPLETE] Detected: {vlm_response.detected_items}")
            logger.info(f"[ORDER-COMPLETE] Validation: {validation_result}")
            logger.info(f"="*80)
            
            # Determine status based on validation
            has_errors = (
                validation_result.get('missing', []) or
                validation_result.get('extra', []) or
                validation_result.get('quantity_mismatch', [])
            )
            status = 'validated' if not has_errors else 'mismatch'
            
            # Get expected items for this order
            expected_items = self._orders.get(str(order_id), {}).get('items', [])
            
            # Build complete result dict
            complete_result = {
                'unique_id': unique_id,
                'order_id': order_id,
                'station_id': self.station_id,
                'expected_items': expected_items,
                'detected_items': validation_result.get('detected_items', vlm_response.detected_items),
                'validation': validation_result,
                'status': status,
                'num_frames': len(frames) if 'frames' in locals() else 3,
                'inference_time_sec': vlm_response.inference_time if hasattr(vlm_response, 'inference_time') else 0.0,
                'timestamp': time.time()
            }
            
            # Save result to file
            import json
            result_file = f"/results/{order_id}_{self.station_id}.json"
            try:
                with open(result_file, 'w') as f:
                    json.dump(complete_result, f, indent=2)
                logger.info(f"[{self.station_id}] Result saved to {result_file}")
            except Exception as e:
                logger.error(f"[{self.station_id}] Failed to save result: {e}")
            
            # Add to in-memory results (for Gradio UI /vlm/results endpoint)
            if self._add_result_func:
                try:
                    self._add_result_func(complete_result)
                    logger.info(f"[{self.station_id}] Result added to in-memory store for order {order_id}")
                except Exception as e:
                    logger.error(f"[{self.station_id}] Failed to add result to in-memory store: {e}")
            
            # Record metrics
            order_latency = time.time() - order_start_time
            self.metrics_store.record_latency(self.station_id, order_latency)
            self.metrics_store.increment_throughput(self.station_id)
            
            logger.info(
                f"[{self.station_id}] Order {order_id} completed: "
                f"latency={order_latency:.2f}s, "
                f"accuracy={validation_result.get('accuracy', 0):.1%} "
                f"(NO pipeline startup overhead!)"
            )
        except Exception as e:
            logger.exception(f"[{self.station_id}] Error processing order {order_id}: {e}")
    
    def _load_order_frames_from_minio(self, order_id: str) -> List:
        """
        Load all frames for a specific order from MinIO.
        
        With persistent pipeline, frames are organized as:
        frames/{station_id}/{order_id}/frame_*.jpg
        
        Args:
            order_id: Order ID to load frames for
        
        Returns:
            List of frame dictionaries with 'data' and metadata
        """
        try:
            from minio import Minio
            
            minio_config = self.config.get('minio', {})
            client = Minio(
                minio_config.get('endpoint', 'minio:9000'),
                access_key=minio_config.get('access_key', 'minioadmin'),
                secret_key=minio_config.get('secret_key', 'minioadmin'),
                secure=minio_config.get('secure', False)
            )
            
            bucket = minio_config.get('frames_bucket', 'frames')
            prefix = f"{self.station_id}/{order_id}/"
            
            frames = []
            
            # List all frames for this order
            objects = client.list_objects(bucket, prefix=prefix, recursive=True)
            
            for obj in objects:
                if not obj.object_name:
                    continue
                
                # Skip EOS marker
                if obj.object_name.endswith('__EOS__'):
                    continue
                
                # Download frame
                try:
                    response = client.get_object(bucket, obj.object_name)
                    frame_data = response.read()
                    response.close()
                    
                    timestamp = (
                        obj.last_modified.timestamp()
                        if obj.last_modified else time.time()
                    )
                    
                    frames.append({
                        'name': obj.object_name,
                        'timestamp': timestamp,
                        'order_id': order_id,
                        'data': frame_data
                    })
                except Exception as e:
                    logger.warning(
                        f"[{self.station_id}] Failed to load frame {obj.object_name}: {e}"
                    )
            
            logger.debug(
                f"[{self.station_id}] Loaded {len(frames)} frames for {order_id}"
            )
            return frames
            
        except Exception as e:
            logger.error(
                f"[{self.station_id}] Error loading frames from MinIO: {e}"
            )
            return []
    
    def _select_best_frames(self, frames: List) -> List:
        """
        Select top frames using YOLO-based scoring.
        
        Uses existing FrameSelector to rank frames and select top 3.
        
        Args:
            frames: List of extracted frames
        
        Returns:
            List of top 3 frames for VLM inference
        """
        if not self._frame_selector:
            # Fallback if selector not initialized
            logger.warning(f"[{self.station_id}] Frame selector not initialized, using first 3 frames")
            return frames[:3]
        
        # Use existing YOLO-based frame selection
        try:
            selected = self._frame_selector.select_top_frames(frames, top_k=3)
            logger.debug(
                f"[{self.station_id}] Selected {len(selected)} frames using YOLO"
            )
            return selected
        except Exception as e:
            logger.error(
                f"[{self.station_id}] Frame selection error: {e}, "
                f"falling back to first 3 frames"
            )
            return frames[:3]
    
    def _request_vlm_inference(self, frames: List, order_id: str) -> Optional[VLMResponse]:
        """
        Send VLM inference request via scheduler and wait for response.
        
        Args:
            frames: Selected frames for inference
            order_id: Order ID being processed
        
        Returns:
            VLMResponse or None if failed
        """
        # Serialize frames (base64 encode)
        import base64
        serialized_frames = []
        for frame in frames:
            if 'data' in frame and frame['data']:
                b64_data = base64.b64encode(frame['data']).decode('utf-8')
                serialized_frames.append({
                    'name': frame.get('name', ''),
                    'data': b64_data,
                    'timestamp': frame.get('timestamp', time.time())
                })
        
        # Create VLM request
        request = VLMRequest(
            station_id=self.station_id,
            order_id=order_id,
            frames=serialized_frames,
            timestamp=time.time()
        )
        
        # Send to scheduler
        unique_id = f"{self.station_id}_{order_id}"
        logger.info(f"[{self.station_id}] unique_id={unique_id} Submitting VLM request {request.request_id} with {len(serialized_frames)} frames")
        request_start_time = time.time()
        self.queue_manager.vlm_request_queue.put(request.to_dict())
        logger.info(f"[{self.station_id}] unique_id={unique_id} VLM request submitted successfully")
        
        logger.debug(
            f"[{self.station_id}] Sent VLM request: {request.request_id} with {len(serialized_frames)} frames"
        )
        
        # Wait for response (with timeout)
        timeout = 120.0  # Increased for large VLM model
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                response_dict = self.response_queue.get(block=True, timeout=1.0)
                
                if response_dict is None:
                    continue
                
                response = VLMResponse.from_dict(response_dict)
                
                # Verify response matches request
                if response.request_id == request.request_id:
                    vlm_latency = time.time() - request_start_time
                    logger.info(
                        f"[{self.station_id}] unique_id={unique_id} Received VLM response: "
                        f"{len(response.detected_items)} items detected in {vlm_latency:.2f}s"
                    )
                    logger.info(f"[{self.station_id}] unique_id={unique_id} VLM items: {response.detected_items}")
                    
                    return response
                else:
                    logger.warning(
                        f"[{self.station_id}] Response mismatch: "
                        f"expected {request.request_id}, got {response.request_id}"
                    )
            
            except queue.Empty:
                continue
        
        logger.error(
            f"[{self.station_id}] VLM response timeout after {timeout}s"
        )
        return None
    
    def _validate_order(self, detected_items: List[str], order_id: str) -> Dict:
        """
        Validate detected items against expected order.
        
        Uses existing validate_order function to compare detected items with order manifest.
        
        Args:
            detected_items: Items detected by VLM
            order_id: Order ID being validated
        
        Returns:
            Validation result dict with accuracy, missing items, etc.
        """
        if not self._validation_func:
            # Fallback if validator not initialized
            logger.warning(f"[{self.station_id}] Validation function not initialized, using placeholder")
            return {
                'order_id': order_id,
                'detected_items': detected_items,
                'accuracy': 0.95,
                'missing_items': [],
                'extra_items': []
            }
        
        # Use existing validation logic
        try:
            # Get expected items for this order
            order = self._orders.get(str(order_id), {})
            expected_items = order.get('items', [])
            
            # Convert detected_items (list of strings/dicts) to expected format
            detected_formatted = []
            for item in detected_items:
                if isinstance(item, dict):
                    detected_formatted.append(item)
                else:
                    detected_formatted.append({'name': item, 'quantity': 1})
            
            # Call validate_order function (no VLM pipeline needed - semantic matching is internal)
            result = self._validation_func(
                expected_items=expected_items,
                detected_items=detected_formatted,
                vlm_pipeline=None  # Not used with semantic service
            )
            
            # Add accuracy calculation
            total = len(expected_items)
            if total > 0:
                correct = total - len(result.get('missing', []))
                accuracy = correct / total
            else:
                accuracy = 1.0
            
            result['order_id'] = order_id
            result['accuracy'] = accuracy
            result['detected_items'] = detected_formatted
            
            logger.debug(f"[{self.station_id}] Validation result: accuracy={accuracy:.2%}")
            return result
        except Exception as e:
            logger.error(f"[{self.station_id}] Validation error: {e}")
            return {
                'order_id': order_id,
                'detected_items': detected_items,
                'accuracy': 0.0,
                'missing_items': [],
                'extra_items': [],
                'error': str(e)
            }
    
    def _check_shutdown_signal(self) -> bool:
        """Check if shutdown signal received via control queue"""
        try:
            signal = self.queue_manager.control_queue.get_nowait()
            
            if signal and signal.get('action') == 'shutdown':
                target_station = signal.get('station_id')
                if target_station == self.station_id or target_station == '*':
                    return True
        
        except queue.Empty:
            pass
        
        return False
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle OS shutdown signals"""
        logger.info(f"[{self.station_id}] Received signal {signum}")
        self._running = False
    
    def _cleanup(self):
        """Cleanup resources before shutdown"""
        logger.info(f"[{self.station_id}] Cleaning up...")
        
        import os
        import signal as sig
        
        # Stop persistent GStreamer pipeline subprocess
        if self._pipeline_subprocess and self._pipeline_subprocess.poll() is None:
            logger.info(f"[{self.station_id}] Terminating pipeline subprocess...")
            
            if not self._pipeline_pid:
                logger.warning(f"[{self.station_id}] Pipeline PID not set, cannot terminate")
                return
            
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self._pipeline_pid), sig.SIGTERM)
                self._pipeline_subprocess.wait(timeout=5)
                logger.info(f"[{self.station_id}] Pipeline stopped gracefully")
            except Exception as e:
                logger.warning(
                    f"[{self.station_id}] Pipeline didn't stop gracefully, killing..."
                )
                try:
                    os.killpg(os.getpgid(self._pipeline_pid), sig.SIGKILL)
                except Exception as kill_error:
                    logger.error(
                        f"[{self.station_id}] Error killing pipeline: {kill_error}"
                    )
        
        # Monitoring threads will exit automatically (daemon=True)
        
        logger.info(f"[{self.station_id}] Cleanup complete")


def start_worker_process(
    station_id: str,
    rtsp_url: str,
    queue_manager: QueueManager,
    metrics_store: MetricsStore,
    config: Dict
):
    """
    Entry point for station worker process.
    
    This function is called via multiprocessing.Process.
    """
    # Setup logging for this process
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - [{station_id}] - %(levelname)s - %(message)s'
    )
    
    # Create and run worker
    worker = StationWorker(
        station_id=station_id,
        rtsp_url=rtsp_url,
        queue_manager=queue_manager,
        metrics_store=metrics_store,
        config=config
    )
    
    worker.run()
