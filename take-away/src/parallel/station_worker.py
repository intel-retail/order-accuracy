"""
Station Worker Process - Production Ready Implementation

Each station worker runs in a separate process and handles:
1. GStreamer video pipeline with resilient RTSP connectivity
2. OCR for order ID detection
3. Frame selection (YOLO or scoring)
4. Sending VLM requests to scheduler
5. Receiving VLM responses
6. Order validation and result generation

Production Features:
- Circuit breaker pattern for unstable streams
- Exponential backoff for restarts
- Thread-safe restart handling
- Stall detection and recovery
- Comprehensive metrics and structured logging
- Graceful shutdown with subprocess cleanup

Maintains all business logic from original sequential implementation.
"""

import multiprocessing as mp
import os
import time
import logging
import signal
import sys
import threading
from typing import Optional, List, Dict, NamedTuple
from dataclasses import dataclass, field
import queue
from pathlib import Path
from enum import Enum
from contextlib import contextmanager

from .shared_queue import QueueManager, VLMRequest, VLMResponse
from .metrics_collector import MetricsStore


logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for GStreamer pipeline behavior."""
    
    # RTSP source hardening
    rtsp_latency_ms: int = 200
    rtsp_retry_count: int = 50  # Conservative retry count for RTSP reconnection
    rtsp_timeout_us: int = 5000000   # 5 seconds
    rtsp_keepalive: bool = True
    rtsp_drop_on_latency: bool = True
    rtsp_protocols: str = "tcp"
    
    # Restart behavior
    restart_base_delay_sec: float = 2.0
    restart_max_delay_sec: float = 60.0
    restart_stability_period_sec: float = 60.0  # Reset counter after stable for this long
    
    # Circuit breaker
    circuit_breaker_max_failures: int = 5
    circuit_breaker_window_sec: float = 300.0  # 5 minutes
    circuit_breaker_cooldown_sec: float = 30.0  # Wait before retrying after circuit opens
    
    # Health monitoring
    health_check_interval_sec: float = 5.0
    stall_detection_timeout_sec: float = 300.0  # No EOS markers for this long = stalled (based on order duration)
    
    # RTSP availability check
    rtsp_wait_timeout_sec: int = 120
    rtsp_poll_interval_sec: float = 1.0
    rtsp_probe_timeout_sec: int = 4
    
    @classmethod
    def from_dict(cls, config: Dict) -> 'PipelineConfig':
        """Create config from dictionary, using defaults for missing values."""
        pipeline_cfg = config.get('pipeline', {})
        return cls(
            rtsp_latency_ms=pipeline_cfg.get('rtsp_latency_ms', 200),
            rtsp_retry_count=pipeline_cfg.get('rtsp_retry_count', 50),
            rtsp_timeout_us=pipeline_cfg.get('rtsp_timeout_us', 5000000),
            rtsp_keepalive=pipeline_cfg.get('rtsp_keepalive', True),
            rtsp_drop_on_latency=pipeline_cfg.get('rtsp_drop_on_latency', True),
            rtsp_protocols=pipeline_cfg.get('rtsp_protocols', 'tcp'),
            restart_base_delay_sec=pipeline_cfg.get('restart_base_delay_sec', 2.0),
            restart_max_delay_sec=pipeline_cfg.get('restart_max_delay_sec', 60.0),
            restart_stability_period_sec=pipeline_cfg.get('restart_stability_period_sec', 60.0),
            circuit_breaker_max_failures=pipeline_cfg.get('circuit_breaker_max_failures', 5),
            circuit_breaker_window_sec=pipeline_cfg.get('circuit_breaker_window_sec', 300.0),
            circuit_breaker_cooldown_sec=pipeline_cfg.get('circuit_breaker_cooldown_sec', 30.0),
            health_check_interval_sec=pipeline_cfg.get('health_check_interval_sec', 5.0),
            stall_detection_timeout_sec=pipeline_cfg.get('stall_detection_timeout_sec', 300.0),
            rtsp_wait_timeout_sec=pipeline_cfg.get('rtsp_wait_timeout_sec', 120),
            rtsp_poll_interval_sec=pipeline_cfg.get('rtsp_poll_interval_sec', 1.0),
            rtsp_probe_timeout_sec=pipeline_cfg.get('rtsp_probe_timeout_sec', 4),
        )


class PipelineState(Enum):
    """Pipeline lifecycle states."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STALLED = "stalled"
    RESTARTING = "restarting"
    CIRCUIT_OPEN = "circuit_open"  # Circuit breaker tripped
    SHUTTING_DOWN = "shutting_down"


@dataclass
class PipelineMetrics:
    """Metrics counters for pipeline operations."""
    pipeline_restarts: int = 0
    pipeline_failures: int = 0
    rtsp_unavailable_events: int = 0
    successful_frames_processed: int = 0
    circuit_breaker_trips: int = 0
    stall_detections: int = 0
    
    # Timing metrics
    last_frame_time: float = 0.0
    pipeline_start_time: float = 0.0
    total_uptime_sec: float = 0.0
    
    def to_dict(self) -> Dict:
        """Export metrics as dictionary."""
        return {
            'pipeline_restarts': self.pipeline_restarts,
            'pipeline_failures': self.pipeline_failures,
            'rtsp_unavailable_events': self.rtsp_unavailable_events,
            'successful_frames_processed': self.successful_frames_processed,
            'circuit_breaker_trips': self.circuit_breaker_trips,
            'stall_detections': self.stall_detections,
            'total_uptime_sec': self.total_uptime_sec,
        }


@dataclass
class CircuitBreakerState:
    """State for circuit breaker pattern."""
    failure_timestamps: List[float] = field(default_factory=list)
    is_open: bool = False
    opened_at: float = 0.0
    
    def record_failure(self, window_sec: float, max_failures: int) -> bool:
        """
        Record a failure and check if circuit should open.
        
        Returns True if circuit breaker should trip.
        """
        now = time.time()
        self.failure_timestamps.append(now)
        
        # Remove failures outside the window
        cutoff = now - window_sec
        self.failure_timestamps = [t for t in self.failure_timestamps if t > cutoff]
        
        # Check if threshold exceeded
        if len(self.failure_timestamps) >= max_failures:
            self.is_open = True
            self.opened_at = now
            return True
        return False
    
    def should_retry(self, cooldown_sec: float) -> bool:
        """Check if enough time has passed to retry after circuit opened."""
        if not self.is_open:
            return True
        return time.time() - self.opened_at >= cooldown_sec
    
    def reset(self):
        """Reset circuit breaker state."""
        self.failure_timestamps.clear()
        self.is_open = False
        self.opened_at = 0.0


class StationWorker:
    """
    Station worker process for single camera stream.
    
    Production-Ready Architecture:
    - Runs complete pipeline for one station
    - Resilient RTSP connectivity with circuit breaker
    - Thread-safe restart handling with exponential backoff
    - Comprehensive health monitoring and stall detection
    - Structured logging for observability
    - Graceful shutdown with subprocess cleanup
    
    Lifecycle:
    1. Initialize → Wait for RTSP → Start Pipeline → Monitor Health
    2. On failure → Circuit Breaker Check → Backoff → Verify RTSP → Restart
    3. On shutdown → Stop Pipeline → Cleanup → Exit
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
        Initialize station worker with configuration.
        
        Args:
            station_id: Unique station identifier (e.g., "station_1")
            rtsp_url: RTSP stream URL for this station
            queue_manager: Shared queue manager for VLM requests/responses
            metrics_store: Shared metrics storage for performance tracking
            config: Station configuration dict containing:
                - minio_endpoint, minio_bucket
                - inventory_path, orders_path
                - yolo_model_path
                - pipeline: PipelineConfig settings
        """
        self.station_id = station_id
        self.rtsp_url = rtsp_url
        self.queue_manager = queue_manager
        self.metrics_store = metrics_store
        self.config = config
        
        # Parse pipeline configuration
        self.pipeline_config = PipelineConfig.from_dict(config)
        
        # Station-specific storage paths
        self.frame_storage_path = f"{config.get('minio_bucket', 'orders')}/{station_id}"
        
        # Response queue for VLM results
        self.response_queue = queue_manager.get_response_queue(station_id)
        
        # =================================================================
        # Pipeline State Management (Thread-Safe)
        # =================================================================
        self._state_lock = threading.RLock()  # Reentrant lock for nested calls
        self._pipeline_state = PipelineState.STOPPED
        self._running = False
        
        # Pipeline subprocess management
        self._pipeline_subprocess = None
        self._pipeline_pid: Optional[int] = None
        self._pipeline_running = False
        
        # Restart tracking with exponential backoff
        self._restart_count = 0
        self._last_restart_time: float = 0.0
        self._pipeline_stable_since: float = 0.0
        
        # =================================================================
        # Circuit Breaker
        # =================================================================
        self._circuit_breaker = CircuitBreakerState()
        
        # =================================================================
        # Metrics
        # =================================================================
        self._pipeline_metrics = PipelineMetrics()
        
        # =================================================================
        # Monitoring Threads
        # =================================================================
        self._frame_monitor_thread: Optional[threading.Thread] = None
        self._health_check_thread: Optional[threading.Thread] = None
        
        # =================================================================
        # Order Tracking
        # =================================================================
        self._processed_orders: set = set()
        self._active_orders: Dict = {}
        self._current_order_id: Optional[str] = None
        self._order_start_time: Optional[float] = None
        self._frames_buffer: List = []
        
        # =================================================================
        # Reusable Components (initialized in run())
        # =================================================================
        self._pipeline_runner = None
        self._frame_selector = None
        self._validation_func = None
        self._add_result_func = None
        self._orders: Dict = {}
        self._inventory: Dict = {}
        
        self._log_structured("info", "worker_initialized", {
            "rtsp_url": rtsp_url,
            "config_summary": {
                "restart_max_delay": self.pipeline_config.restart_max_delay_sec,
                "circuit_breaker_threshold": self.pipeline_config.circuit_breaker_max_failures,
                "stall_timeout": self.pipeline_config.stall_detection_timeout_sec,
            }
        })
    
    # =========================================================================
    # Structured Logging
    # =========================================================================
    
    def _log_structured(self, level: str, event: str, data: Dict = None):
        """
        Emit structured log message for observability.
        
        Args:
            level: Log level (debug, info, warning, error)
            event: Event type/name
            data: Additional structured data
        """
        log_data = {
            "station_id": self.station_id,
            "event": event,
            "pipeline_state": self._pipeline_state.value,
            "pipeline_pid": self._pipeline_pid,
            "restart_count": self._restart_count,
        }
        if data:
            log_data.update(data)
        
        message = f"[{self.station_id}] {event}: {data}" if data else f"[{self.station_id}] {event}"
        
        log_func = getattr(logger, level, logger.info)
        log_func(message)
    
    # =========================================================================
    # Thread-Safe State Management
    # =========================================================================
    
    @contextmanager
    def _state_transition(self, new_state: PipelineState):
        """
        Context manager for thread-safe state transitions.
        
        Usage:
            with self._state_transition(PipelineState.RESTARTING):
                # do restart work
        """
        with self._state_lock:
            old_state = self._pipeline_state
            self._pipeline_state = new_state
            self._log_structured("debug", "state_transition", {
                "from_state": old_state.value,
                "to_state": new_state.value
            })
        try:
            yield
        except Exception as e:
            with self._state_lock:
                self._pipeline_state = PipelineState.STOPPED
            raise
    
    def _get_state(self) -> PipelineState:
        """Thread-safe state getter."""
        with self._state_lock:
            return self._pipeline_state
    
    def _set_state(self, state: PipelineState):
        """Thread-safe state setter."""
        with self._state_lock:
            self._pipeline_state = state
    
    def run(self):
        """
        Main worker process loop with production-ready error handling.
        
        Lifecycle:
        1. Register with metrics store
        2. Setup signal handlers for graceful shutdown
        3. Initialize pipeline components
        4. Wait for RTSP stream availability
        5. Start persistent GStreamer pipeline
        6. Start monitoring threads
        7. Process orders until shutdown signal
        8. Cleanup and exit
        """
        # Register with metrics
        self.metrics_store.register_station(self.station_id)
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        
        self._log_structured("info", "worker_started", {
            "pid": mp.current_process().pid
        })
        
        try:
            # Initialize pipeline components
            self._log_structured("info", "pipeline_initializing")
            self._initialize_pipeline()
            
            self._running = True
            
            # Wait for RTSP stream to be available
            if not self._wait_for_rtsp_stream():
                self._log_structured("error", "rtsp_unavailable_at_startup", {
                    "timeout": self.pipeline_config.rtsp_wait_timeout_sec
                })
                self._pipeline_metrics.rtsp_unavailable_events += 1
                return
            
            # Start persistent GStreamer pipeline
            self._start_persistent_pipeline()
            
            self._log_structured("info", "pipeline_started_waiting_for_frames")
            
            # Start monitoring threads
            self._start_frame_monitor()
            self._log_structured("info", "frame_monitor_started")
            
            self._start_health_monitor()
            self._log_structured("info", "health_monitor_started")
            
            self._log_structured("info", "all_components_started_entering_main_loop")
            
            # Main processing loop
            while self._running:
                try:
                    # Check for control signals
                    if self._check_shutdown_signal():
                        self._log_structured("info", "shutdown_signal_received")
                        break
                    
                    # Handle circuit breaker state
                    if self._get_state() == PipelineState.CIRCUIT_OPEN:
                        self._handle_circuit_open_state()
                        continue
                    
                    # Process orders that are ready in MinIO
                    self._process_ready_orders()
                    
                    # Update stability tracking
                    self._check_pipeline_stability()
                    
                    time.sleep(0.1)  # Small delay to prevent CPU spinning
                
                except Exception as e:
                    self._log_structured("error", "main_loop_error", {"error": str(e)})
                    time.sleep(1)
        
        finally:
            self._set_state(PipelineState.SHUTTING_DOWN)
            self._cleanup()
            self.metrics_store.unregister_station(self.station_id)
            
            # Log final metrics
            self._log_structured("info", "worker_stopped", {
                "final_metrics": self._pipeline_metrics.to_dict()
            })
    
    def _handle_circuit_open_state(self):
        """
        Handle circuit breaker open state.
        
        Periodically check if RTSP stream is available and reset circuit
        if stream recovers.
        """
        cfg = self.pipeline_config
        
        if not self._circuit_breaker.should_retry(cfg.circuit_breaker_cooldown_sec):
            time.sleep(1)  # Wait before checking again
            return
        
        self._log_structured("info", "circuit_breaker_retry_check", {
            "cooldown_elapsed": time.time() - self._circuit_breaker.opened_at
        })
        
        # Try to verify RTSP availability
        if self._verify_rtsp_stream_quick():
            self._log_structured("info", "rtsp_recovered_resetting_circuit_breaker")
            self._circuit_breaker.reset()
            self._restart_count = 0
            self._set_state(PipelineState.STOPPED)
            
            # Try to restart pipeline
            self._safe_pipeline_restart("circuit_breaker_reset")
        else:
            self._log_structured("warning", "rtsp_still_unavailable", {
                "next_retry_in": cfg.circuit_breaker_cooldown_sec
            })
            self._circuit_breaker.opened_at = time.time()  # Reset cooldown timer
    
    def _check_pipeline_stability(self):
        """
        Check if pipeline has been stable long enough to reset restart counter.
        """
        if self._get_state() != PipelineState.RUNNING:
            return
        
        if self._pipeline_stable_since == 0:
            return
        
        stability_duration = time.time() - self._pipeline_stable_since
        
        if stability_duration >= self.pipeline_config.restart_stability_period_sec:
            if self._restart_count > 0:
                self._log_structured("info", "pipeline_stable_resetting_restart_count", {
                    "stability_duration_sec": stability_duration,
                    "previous_restart_count": self._restart_count
                })
                self._restart_count = 0
                self._circuit_breaker.reset()
    
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
                # Use absolute paths for Docker container
                inventory_path = self.config.get('inventory_path', '/config/inventory.json')
                orders_path = self.config.get('orders_path', '/config/orders.json')
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
    
    def _wait_for_rtsp_stream(self, timeout: int = None, poll_interval: float = None) -> bool:
        """
        Wait for RTSP stream to become available before starting GStreamer pipeline.
        
        This prevents repeated pipeline failures and EOS marker spam during startup.
        The pipeline will be started only after RTSP stream is confirmed available.
        
        Uses a short GStreamer test pipeline to verify the stream is accessible
        and actually streaming video data (not just RTSP server responding).
        
        Args:
            timeout: Maximum seconds to wait (uses config default if not specified)
            poll_interval: Seconds between checks (uses config default if not specified)
        
        Returns:
            True if stream is available, False if timeout exceeded
        """
        import subprocess
        import socket
        from urllib.parse import urlparse
        
        cfg = self.pipeline_config
        timeout = timeout or cfg.rtsp_wait_timeout_sec
        poll_interval = poll_interval or cfg.rtsp_poll_interval_sec
        
        if not self.rtsp_url.startswith("rtsp://"):
            self._log_structured("info", "non_rtsp_source_skipping_check")
            return True
        
        parsed = urlparse(self.rtsp_url)
        host = parsed.hostname or 'rtsp-streamer'
        port = parsed.port or 8554
        
        self._log_structured("info", "waiting_for_rtsp_stream", {
            "rtsp_url": self.rtsp_url,
            "timeout_sec": timeout
        })
        
        start_time = time.time()
        attempt = 0
        port_ready = False
        
        while time.time() - start_time < timeout:
            attempt += 1
            
            # Phase 1: Wait for RTSP port to be open
            if not port_ready:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(2)
                        result = sock.connect_ex((host, port))
                        if result == 0:
                            self._log_structured("debug", "rtsp_port_open", {
                                "attempt": attempt,
                                "host": host,
                                "port": port
                            })
                            port_ready = True
                except Exception:
                    pass
                
                if not port_ready:
                    time.sleep(poll_interval)
                    continue
            
            # Phase 2: Verify stream exists using quick GStreamer probe
            try:
                result = subprocess.run(
                    [
                        'timeout', str(cfg.rtsp_probe_timeout_sec),
                        'gst-launch-1.0', '-e',
                        'rtspsrc', f'location={self.rtsp_url}', 'protocols=tcp', 'latency=50',
                        '!', 'fakesink', 'sync=false'
                    ],
                    capture_output=True,
                    text=True,
                    timeout=cfg.rtsp_probe_timeout_sec + 2
                )
                
                combined_output = result.stdout + result.stderr
                
                # First check: Reject if 404 error (stream path doesn't exist yet)
                if 'Not Found' in combined_output or '404' in combined_output:
                    if attempt % 10 == 0:
                        self._log_structured("debug", "stream_path_not_found", {
                            "attempt": attempt
                        })
                    time.sleep(0.5)  # Quick retry
                    continue
                
                # Second check: Confirm stream opened successfully
                if 'PREROLLED' in combined_output or 'Opened Stream' in combined_output or 'PLAYING' in combined_output:
                    elapsed = time.time() - start_time
                    self._log_structured("info", "rtsp_stream_confirmed", {
                        "attempt": attempt,
                        "elapsed_sec": round(elapsed, 1)
                    })
                    return True
                    
            except subprocess.TimeoutExpired:
                pass  # Stream check timed out, retry
            except Exception as e:
                self._log_structured("debug", "stream_check_failed", {
                    "attempt": attempt,
                    "error": str(e)
                })
            
            if attempt % 10 == 0:
                elapsed = time.time() - start_time
                self._log_structured("info", "still_waiting_for_rtsp", {
                    "elapsed_sec": int(elapsed),
                    "timeout_sec": timeout,
                    "attempt": attempt
                })
            
            time.sleep(poll_interval)
        
        self._log_structured("warning", "rtsp_wait_timeout", {
            "timeout_sec": timeout,
            "attempts": attempt
        })
        self._pipeline_metrics.rtsp_unavailable_events += 1
        return False
    
    def _verify_rtsp_stream_quick(self, timeout_sec: int = 3) -> bool:
        """
        Quick verification that RTSP stream is currently available.
        
        Used by circuit breaker and restart logic to check stream status
        before attempting pipeline restart.
        
        Args:
            timeout_sec: Maximum seconds to wait for verification
        
        Returns:
            True if stream appears available, False otherwise
        """
        import subprocess
        
        if not self.rtsp_url.startswith("rtsp://"):
            return True
        
        try:
            result = subprocess.run(
                [
                    'timeout', str(timeout_sec),
                    'gst-launch-1.0', '-e',
                    'rtspsrc', f'location={self.rtsp_url}', 'protocols=tcp', 'latency=50',
                    '!', 'fakesink', 'sync=false'
                ],
                capture_output=True,
                text=True,
                timeout=timeout_sec + 2
            )
            
            combined_output = result.stdout + result.stderr
            
            # Check for 404 or other errors
            if 'Not Found' in combined_output or '404' in combined_output:
                return False
            
            # Check for successful connection
            if 'PREROLLED' in combined_output or 'Opened Stream' in combined_output:
                return True
            
            return False
            
        except Exception:
            return False

    def _start_persistent_pipeline(self):
        """
        Start persistent GStreamer frame capture as subprocess.
        
        Uses gvapython element for OCR and frame upload.
        Includes hardened RTSP source settings for production reliability.
        
        Thread-safe: Uses state lock to prevent concurrent starts.
        """
        with self._state_transition(PipelineState.STARTING):
            self._log_structured("info", "starting_persistent_pipeline")
            
            import subprocess
            import os
            
            # Build GStreamer pipeline command with hardened settings
            pipeline = self._build_persistent_gstreamer_pipeline()
            
            # Redirect both stdout and stderr to log file for debugging frame_pipeline output
            stderr_log = f'/tmp/gst_pipeline_{self.station_id}.log'
            
            # Export STATION_ID explicitly in shell command for gvapython
            cmd = f"export STATION_ID={self.station_id} && gst-launch-1.0 -q {pipeline} >> {stderr_log} 2>&1"
            
            # Log the exact command
            self._log_structured("info", "pipeline_command", {
                "command": f"gst-launch-1.0 -q {pipeline}",
                "stderr_log": stderr_log
            })
            
            # Prepare environment
            env = os.environ.copy()
            app_dir = '/app'
            pythonpath = env.get('PYTHONPATH', '')
            if app_dir not in pythonpath:
                env['PYTHONPATH'] = f"{app_dir}:{pythonpath}" if pythonpath else app_dir
            
            # Set GST_PLUGIN_PYTHON_PATH for gvapython
            env['GST_PLUGIN_PYTHON_PATH'] = app_dir
            
            # Also set STATION_ID in env (redundant but ensures availability)
            env['STATION_ID'] = self.station_id
            
            try:
                # Start subprocess with process group for clean shutdown
                self._pipeline_subprocess = subprocess.Popen(
                    cmd,
                    shell=True,
                    env=env,
                    cwd='/app',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid  # Create process group for clean termination
                )
                
                self._pipeline_pid = self._pipeline_subprocess.pid
                self._pipeline_running = True
                self._pipeline_stable_since = time.time()
                self._pipeline_metrics.pipeline_start_time = time.time()
                self._pipeline_metrics.last_frame_time = time.time()
                
                self._set_state(PipelineState.RUNNING)
                
                self._log_structured("info", "pipeline_started", {
                    "pid": self._pipeline_pid
                })
                
            except Exception as e:
                self._pipeline_metrics.pipeline_failures += 1
                self._log_structured("error", "pipeline_start_failed", {
                    "error": str(e)
                })
                raise
    
    def _build_persistent_gstreamer_pipeline(self) -> str:
        """
        Build GStreamer pipeline with production-hardened RTSP settings.
        
        RTSP Source Hardening:
        - do-rtsp-keep-alive=true: Maintain RTSP session
        - retry=N: Reconnection attempts on failure
        - timeout=N: Connection timeout in microseconds
        - drop-on-latency=true: Drop frames if falling behind
        - protocols=tcp: Force TCP transport for reliability
        
        Returns:
            GStreamer pipeline string ready for gst-launch-1.0
        """
        cfg = self.pipeline_config
        
        self._log_structured("debug", "building_pipeline", {
            "rtsp_url": self.rtsp_url,
            "is_rtsp": self.rtsp_url.startswith("rtsp://")
        })
        
        # Use module name (not file path) for gvapython - PYTHONPATH must include /app
        frame_pipeline_module = "frame_pipeline"
        
        # Use uridecodebin which handles both RTSP and file sources seamlessly
        # caps="video/x-raw" filters to video only (ignores audio stream)
        # Using 1fps for CPU-based EasyOCR (can increase if GPU available)
        uri = self.rtsp_url
        if not uri.startswith(("rtsp://", "file://", "http://", "https://")):
            # Local file path - convert to file:// URI
            uri = f"file://{self.rtsp_url}"
        
        # Capture framerate for the GStreamer pipeline.
        # Order 925 appears for only ~5s in the test video; at 1fps that gives
        # just 5 frames per loop cycle. 2fps doubles coverage without exceeding
        # the max_bucket_size=100 cap for the longer 651 / 384 sections.
        capture_fps = int(os.environ.get("CAPTURE_FPS", "2"))

        pipeline = (
            f'uridecodebin uri={uri} caps="video/x-raw" '
            "! videoconvert "
            "! video/x-raw,format=BGR "
            "! videorate "
            f"! video/x-raw,framerate={capture_fps}/1 "
            f"! gvapython module={frame_pipeline_module} function=process_frame "
            "! fakesink sync=false"
        )
        
        return pipeline
    
    def _safe_pipeline_restart(self, reason: str):
        """
        Safely restart pipeline with all production safeguards.
        
        Safeguards:
        1. Thread-safe state check (prevent concurrent restarts)
        2. Circuit breaker check
        3. RTSP availability verification
        4. Exponential backoff delay
        
        Args:
            reason: Human-readable reason for restart (for logging)
        """
        with self._state_lock:
            # Prevent concurrent restarts
            if self._pipeline_state in (PipelineState.RESTARTING, PipelineState.STARTING):
                self._log_structured("debug", "restart_skipped_already_in_progress", {
                    "current_state": self._pipeline_state.value
                })
                return
            
            # Check circuit breaker
            if self._pipeline_state == PipelineState.CIRCUIT_OPEN:
                self._log_structured("debug", "restart_skipped_circuit_open")
                return
        
        with self._state_transition(PipelineState.RESTARTING):
            self._restart_count += 1
            self._pipeline_metrics.pipeline_restarts += 1
            
            # Check circuit breaker threshold
            cfg = self.pipeline_config
            if self._circuit_breaker.record_failure(
                cfg.circuit_breaker_window_sec,
                cfg.circuit_breaker_max_failures
            ):
                self._pipeline_metrics.circuit_breaker_trips += 1
                self._log_structured("warning", "circuit_breaker_tripped", {
                    "failures_in_window": len(self._circuit_breaker.failure_timestamps),
                    "window_sec": cfg.circuit_breaker_window_sec,
                    "reason": reason
                })
                self._set_state(PipelineState.CIRCUIT_OPEN)
                return
            
            # Calculate exponential backoff delay
            delay = min(
                cfg.restart_base_delay_sec * (2 ** (self._restart_count - 1)),
                cfg.restart_max_delay_sec
            )
            
            self._log_structured("info", "pipeline_restart_scheduled", {
                "restart_count": self._restart_count,
                "delay_sec": delay,
                "reason": reason
            })
            
            time.sleep(delay)
            
            # Verify RTSP stream before restart
            self._log_structured("info", "verifying_rtsp_before_restart")
            
            if not self._wait_for_rtsp_stream(timeout=30):
                self._log_structured("warning", "rtsp_unavailable_for_restart", {
                    "restart_count": self._restart_count
                })
                self._pipeline_metrics.rtsp_unavailable_events += 1
                self._set_state(PipelineState.STOPPED)
                return
            
            # Attempt restart
            try:
                self._start_persistent_pipeline()
                self._log_structured("info", "pipeline_restart_successful", {
                    "restart_count": self._restart_count
                })
            except Exception as e:
                self._pipeline_metrics.pipeline_failures += 1
                self._log_structured("error", "pipeline_restart_failed", {
                    "error": str(e),
                    "restart_count": self._restart_count
                })
                self._set_state(PipelineState.STOPPED)
    
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
        self._log_structured("debug", "frame_monitor_thread_started")
    
    def _frame_monitor_loop(self):
        """
        Continuously check MinIO for orders with EOS markers.
        Also updates frame processing metrics for stall detection.
        """
        while self._running:
            try:
                # Scan MinIO for completed orders
                completed_orders = self._scan_minio_for_completed_orders()
                
                # Heartbeat: any EOS marker in MinIO means the pipeline is alive and
                # producing frames.  Update regardless of whether the order was already
                # processed so that loop-2+ restarts don't false-trigger stall detection.
                if completed_orders:
                    self._pipeline_metrics.last_frame_time = time.time()

                for order_id in completed_orders:
                    if order_id not in self._processed_orders and order_id not in self._active_orders:
                        # Skip orders not in orders.json to avoid wasted VLM inference
                        if self._orders and str(order_id) not in self._orders:
                            self._log_structured("debug", "skipping_unknown_order", {
                                "order_id": order_id,
                                "reason": "not in orders.json"
                            })
                            self._processed_orders.add(order_id)  # Mark as processed to avoid re-checking
                            continue
                        
                        self._pipeline_metrics.successful_frames_processed += 1
                        
                        self._log_structured("info", "completed_order_detected", {
                            "order_id": order_id
                        })
                        
                        self._active_orders[order_id] = {
                            'detected_at': time.time(),
                            'status': 'ready'
                        }
                
                time.sleep(0.5)  # Check every 500ms
                
            except Exception as e:
                self._log_structured("error", "frame_monitor_error", {"error": str(e)})
                time.sleep(1)
    
    def _start_health_monitor(self):
        """
        Start thread that monitors pipeline health and auto-restarts if needed.
        
        Health monitoring includes:
        - Pipeline subprocess exit detection
        - Stall detection (no frames for configurable timeout)
        - Automatic restart with safeguards
        """
        self._health_check_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name=f"{self.station_id}-HealthCheck"
        )
        self._health_check_thread.start()
        self._log_structured("debug", "health_monitor_thread_started")
    
    def _health_monitor_loop(self):
        """
        Comprehensive health monitoring loop.
        
        Monitors:
        1. Pipeline subprocess exit (crash detection)
        2. Frame processing stall (no new frames for timeout period)
        3. Pipeline state anomalies
        
        Triggers restart via _safe_pipeline_restart() with all safeguards.
        """
        cfg = self.pipeline_config
        
        while self._running:
            try:
                current_state = self._get_state()
                
                # Skip health checks during certain states
                if current_state in (
                    PipelineState.STOPPED,
                    PipelineState.STARTING,
                    PipelineState.RESTARTING,
                    PipelineState.SHUTTING_DOWN,
                    PipelineState.CIRCUIT_OPEN
                ):
                    time.sleep(cfg.health_check_interval_sec)
                    continue
                
                # Check 1: Pipeline subprocess exit
                if self._pipeline_subprocess:
                    returncode = self._pipeline_subprocess.poll()
                    
                    if returncode is not None:
                        # Pipeline died - capture any remaining output
                        output = ""
                        try:
                            output = self._pipeline_subprocess.stdout.read()
                            if isinstance(output, bytes):
                                output = output.decode('utf-8', errors='replace')
                        except Exception:
                            pass
                        
                        self._pipeline_metrics.pipeline_failures += 1
                        
                        self._log_structured("error", "pipeline_subprocess_died", {
                            "exit_code": returncode,
                            "output_snippet": output[:500] if output else None
                        })
                        
                        # Trigger safe restart
                        self._safe_pipeline_restart(f"subprocess_exit_code_{returncode}")
                        continue
                
                # Check 2: Stall detection (only when pipeline should be running)
                if current_state == PipelineState.RUNNING:
                    time_since_last_frame = time.time() - self._pipeline_metrics.last_frame_time
                    
                    if time_since_last_frame > cfg.stall_detection_timeout_sec:
                        self._pipeline_metrics.stall_detections += 1
                        
                        self._log_structured("warning", "pipeline_stall_detected", {
                            "seconds_since_last_frame": round(time_since_last_frame, 1),
                            "stall_timeout": cfg.stall_detection_timeout_sec
                        })
                        
                        self._set_state(PipelineState.STALLED)
                        
                        # Kill stalled pipeline
                        self._terminate_pipeline_subprocess()
                        
                        # Trigger safe restart
                        self._safe_pipeline_restart("stall_detected")
                        continue
                
                time.sleep(cfg.health_check_interval_sec)
                
            except Exception as e:
                self._log_structured("error", "health_monitor_error", {"error": str(e)})
                time.sleep(cfg.health_check_interval_sec)
    
    def _terminate_pipeline_subprocess(self):
        """
        Gracefully terminate pipeline subprocess.
        
        Termination sequence:
        1. Send SIGTERM to process group
        2. Wait for graceful exit (5 sec timeout)
        3. Send SIGKILL if still running
        4. Clean up subprocess handle
        
        Prevents zombie processes.
        """
        import os
        import signal as sig
        
        if not self._pipeline_subprocess:
            return
        
        if self._pipeline_subprocess.poll() is not None:
            # Already terminated
            self._pipeline_subprocess = None
            self._pipeline_pid = None
            return
        
        self._log_structured("info", "terminating_pipeline_subprocess", {
            "pid": self._pipeline_pid
        })
        
        try:
            # Send SIGTERM to process group
            if self._pipeline_pid:
                try:
                    os.killpg(os.getpgid(self._pipeline_pid), sig.SIGTERM)
                except ProcessLookupError:
                    pass  # Process already gone
            
            # Wait for graceful termination
            try:
                self._pipeline_subprocess.wait(timeout=5)
                self._log_structured("debug", "pipeline_terminated_gracefully")
            except Exception:
                # Force kill if still running
                self._log_structured("warning", "pipeline_force_killing")
                try:
                    if self._pipeline_pid:
                        os.killpg(os.getpgid(self._pipeline_pid), sig.SIGKILL)
                    self._pipeline_subprocess.wait(timeout=2)
                except Exception as e:
                    self._log_structured("error", "pipeline_kill_failed", {"error": str(e)})
        
        except Exception as e:
            self._log_structured("error", "pipeline_termination_error", {"error": str(e)})
        
        finally:
            self._pipeline_subprocess = None
            self._pipeline_pid = None
            self._pipeline_running = False
    
    def _scan_minio_for_completed_orders(self) -> List[str]:
        """
        Scan MinIO for orders with EOS markers.
        
        Returns:
            List of order_ids that have EOS markers (completed orders)
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
                    pass  # EOS not found
            
            return completed_orders
            
        except Exception as e:
            self._log_structured("error", "minio_scan_error", {"error": str(e)})
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
                    self._log_structured("error", "order_processing_error", {
                        "order_id": order_id,
                        "error": str(e)
                    })
                    order_info['status'] = 'error'
    
    def _process_single_order(self, order_id: str):
        """
        Process a single completed order.
        
        Frames are already in MinIO - no pipeline startup needed.
        """
        self._log_structured("info", "processing_order", {"order_id": order_id})

        # When oa_frame_selector service is running it owns YOLO frame selection
        # and the VLM call. Skip the duplicate path in station_worker to avoid
        # two concurrent VLM requests for the same order.
        import os
        if os.environ.get("EXTERNAL_FRAME_SELECTOR", "false").lower() == "true":
            self._log_structured("info", "skipping_order_external_frame_selector",
                                 {"order_id": order_id,
                                  "reason": "oa_frame_selector handles VLM calls"})
            return

        # Skip orders not in orders.json (invalid/partial OCR detections)
        if str(order_id) not in self._orders:
            self._log_structured("warning", "skipping_unknown_order", {
                "order_id": order_id,
                "reason": "not in orders.json"
            })
            return
        
        try:
            order_start_time = time.time()
            
            # Load frames from MinIO
            self._log_structured("debug", "loading_frames_from_minio", {"order_id": order_id})
            frames = self._load_order_frames_from_minio(order_id)
            self._log_structured("debug", "frames_loaded", {
                "order_id": order_id,
                "frame_count": len(frames) if frames else 0
            })
            
            if not frames:
                self._log_structured("warning", "no_frames_found", {"order_id": order_id})
                return
            
            # Frame selection (YOLO)
            self._log_structured("debug", "selecting_frames", {"order_id": order_id})
            selected_frames = self._select_best_frames(frames)
            self._log_structured("debug", "frames_selected", {
                "order_id": order_id,
                "selected_count": len(selected_frames) if selected_frames else 0
            })
            
            if not selected_frames:
                self._log_structured("warning", "no_frames_selected", {"order_id": order_id})
                return
            
            # VLM inference via scheduler
            self._log_structured("info", "requesting_vlm_inference", {"order_id": order_id})
            vlm_response = self._request_vlm_inference(selected_frames, order_id)
            
            if not vlm_response or not vlm_response.success:
                self._log_structured("error", "vlm_inference_failed", {
                    "order_id": order_id,
                    "error": vlm_response.error if vlm_response else "No response"
                })
                self.metrics_store.increment_failures()
                return
            
            # Order validation
            self._log_structured("info", "validating_order", {"order_id": order_id})
            validation_result = self._validate_order(vlm_response.detected_items, order_id)
            
            # Log comprehensive result
            unique_id = f"{self.station_id}_{order_id}"
            self._log_structured("info", "order_complete", {
                "unique_id": unique_id,
                "detected_items": vlm_response.detected_items,
                "validation": validation_result
            })
            
            # Determine status
            has_errors = (
                validation_result.get('missing', []) or
                validation_result.get('extra', []) or
                validation_result.get('quantity_mismatch', [])
            )
            status = 'validated' if not has_errors else 'mismatch'
            
            # Get expected items (orders.json has items directly as a list)
            expected_items = self._orders.get(str(order_id), [])
            
            # Build complete result
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
                self._log_structured("debug", "result_saved", {"file": result_file})
            except Exception as e:
                self._log_structured("error", "result_save_failed", {"error": str(e)})
            
            # Add to in-memory results
            if self._add_result_func:
                try:
                    self._add_result_func(complete_result)
                    self._log_structured("debug", "result_added_to_memory", {"order_id": order_id})
                except Exception as e:
                    self._log_structured("error", "result_memory_add_failed", {"error": str(e)})
            
            # Record metrics
            order_latency = time.time() - order_start_time
            self.metrics_store.record_latency(self.station_id, order_latency)
            self.metrics_store.increment_throughput(self.station_id)
            
            self._log_structured("info", "order_processing_complete", {
                "order_id": order_id,
                "latency_sec": round(order_latency, 2),
                "accuracy": validation_result.get('accuracy', 0)
            })
            
        except Exception as e:
            self._log_structured("error", "order_processing_exception", {
                "order_id": order_id,
                "error": str(e)
            })
            import traceback
            logger.exception(f"[{self.station_id}] Error processing order {order_id}")
    
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
                    self._log_structured("warning", "frame_load_failed", {
                        "object_name": obj.object_name,
                        "error": str(e)
                    })
            
            self._log_structured("debug", "frames_loaded_from_minio", {
                "order_id": order_id,
                "count": len(frames)
            })
            return frames
            
        except Exception as e:
            self._log_structured("error", "minio_frame_load_error", {
                "order_id": order_id,
                "error": str(e)
            })
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
            self._log_structured("warning", "frame_selector_not_initialized")
            return frames[:3]
        
        try:
            selected = self._frame_selector.select_top_frames(frames, top_k=3)
            self._log_structured("debug", "frames_selected_by_yolo", {
                "input_count": len(frames),
                "selected_count": len(selected)
            })
            return selected
        except Exception as e:
            self._log_structured("error", "frame_selection_error", {
                "error": str(e)
            })
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
        import base64
        
        # Serialize frames
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
        
        unique_id = f"{self.station_id}_{order_id}"
        request_start_time = time.time()
        
        self._log_structured("info", "vlm_request_submitting", {
            "unique_id": unique_id,
            "request_id": request.request_id,
            "frame_count": len(serialized_frames)
        })
        
        self.queue_manager.vlm_request_queue.put(request.to_dict())
        
        self._log_structured("debug", "vlm_request_submitted", {
            "unique_id": unique_id,
            "request_id": request.request_id
        })
        
        # Wait for response with timeout
        timeout = 120.0
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            try:
                response_dict = self.response_queue.get(block=True, timeout=1.0)
                
                if response_dict is None:
                    continue
                
                response = VLMResponse.from_dict(response_dict)
                
                if response.request_id == request.request_id:
                    vlm_latency = time.time() - request_start_time
                    self._log_structured("info", "vlm_response_received", {
                        "unique_id": unique_id,
                        "detected_items_count": len(response.detected_items),
                        "latency_sec": round(vlm_latency, 2)
                    })
                    return response
                else:
                    self._log_structured("warning", "vlm_response_mismatch", {
                        "expected": request.request_id,
                        "got": response.request_id
                    })
            
            except queue.Empty:
                continue
        
        self._log_structured("error", "vlm_response_timeout", {
            "unique_id": unique_id,
            "timeout_sec": timeout
        })
        return None
    
    def _validate_order(self, detected_items: List[str], order_id: str) -> Dict:
        """
        Validate detected items against expected order.
        
        Uses existing validate_order function for semantic comparison.
        
        Args:
            detected_items: Items detected by VLM
            order_id: Order ID being validated
        
        Returns:
            Validation result dict with accuracy, missing items, etc.
        """
        if not self._validation_func:
            self._log_structured("warning", "validation_func_not_initialized")
            return {
                'order_id': order_id,
                'detected_items': detected_items,
                'accuracy': 0.95,
                'missing_items': [],
                'extra_items': []
            }
        
        try:
            # Get expected items (orders.json has items directly as a list)
            expected_items = self._orders.get(str(order_id), [])
            
            # Format detected items
            detected_formatted = []
            for item in detected_items:
                if isinstance(item, dict):
                    detected_formatted.append(item)
                else:
                    detected_formatted.append({'name': item, 'quantity': 1})
            
            # Call validation function
            result = self._validation_func(
                expected_items=expected_items,
                detected_items=detected_formatted,
                vlm_pipeline=None
            )
            
            # Calculate accuracy
            total = len(expected_items)
            if total > 0:
                correct = total - len(result.get('missing', []))
                accuracy = correct / total
            else:
                accuracy = 1.0
            
            result['order_id'] = order_id
            result['accuracy'] = accuracy
            result['detected_items'] = detected_formatted
            
            self._log_structured("debug", "validation_complete", {
                "order_id": order_id,
                "accuracy": round(accuracy, 2)
            })
            return result
            
        except Exception as e:
            self._log_structured("error", "validation_error", {
                "order_id": order_id,
                "error": str(e)
            })
            return {
                'order_id': order_id,
                'detected_items': detected_items,
                'accuracy': 0.0,
                'missing_items': [],
                'extra_items': [],
                'error': str(e)
            }
    
    def _check_shutdown_signal(self) -> bool:
        """Check if shutdown signal received via control queue."""
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
        """
        Handle OS shutdown signals (SIGTERM, SIGINT).
        
        Sets running flag to False, triggering graceful shutdown
        in the main loop. The cleanup() method handles subprocess
        termination.
        """
        self._log_structured("info", "shutdown_signal_received", {
            "signal": signum
        })
        self._running = False
    
    def _cleanup(self):
        """
        Comprehensive cleanup before worker shutdown.
        
        Cleanup sequence:
        1. Set state to SHUTTING_DOWN
        2. Terminate pipeline subprocess gracefully
        3. Calculate final uptime metrics
        4. Log final state
        
        Prevents zombie processes and ensures clean exit.
        """
        self._log_structured("info", "cleanup_starting")
        
        # Calculate total uptime
        if self._pipeline_metrics.pipeline_start_time > 0:
            self._pipeline_metrics.total_uptime_sec = (
                time.time() - self._pipeline_metrics.pipeline_start_time
            )
        
        # Terminate pipeline subprocess
        self._terminate_pipeline_subprocess()
        
        # Monitoring threads exit automatically (daemon=True)
        
        self._log_structured("info", "cleanup_complete", {
            "final_metrics": self._pipeline_metrics.to_dict()
        })
    
    def update_frame_time(self):
        """
        Update last frame processing time.
        
        Called by frame_pipeline.py to signal successful frame processing.
        Used by stall detection to identify hung pipelines.
        """
        self._pipeline_metrics.last_frame_time = time.time()
        self._pipeline_metrics.successful_frames_processed += 1
    
    def get_metrics(self) -> Dict:
        """
        Get current pipeline metrics.
        
        Returns:
            Dictionary of all pipeline metrics
        """
        metrics = self._pipeline_metrics.to_dict()
        metrics['state'] = self._get_state().value
        metrics['restart_count'] = self._restart_count
        metrics['circuit_breaker_open'] = self._circuit_breaker.is_open
        return metrics


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
