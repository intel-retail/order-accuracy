# Take-Away Order Accuracy Pipeline - Comprehensive Code Analysis
## Table of Contents
1. [System Architecture Overview](#1-system-architecture-overview)
2. [Entry Point and Mode Selection](#2-entry-point-and-mode-selection)
3. [Parallel Mode Architecture](#3-parallel-mode-architecture)
4. [Inter-Process Communication (IPC)](#4-inter-process-communication-ipc)
5. [Frame Pipeline Flow](#5-frame-pipeline-flow)
6. [Frame Selector Service](#6-frame-selector-service)
7. [VLM Inference and Validation](#7-vlm-inference-and-validation)
8. [Data Flow Diagrams](#8-data-flow-diagrams)
9. [Key Design Patterns](#9-key-design-patterns)
10. [Configuration System](#10-configuration-system)
---
## 1. System Architecture Overview
### 1.1 High-Level Components
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         Take-Away Order Accuracy Pipeline                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ┌─────────────────┐    ┌──────────────────┐    ┌──────────────────────────┐   │
│   │  RTSP Streamer  │───▶│  Station Worker  │───▶│  Frame Selector Service  │   │
│   │  (Video Source) │    │  (GStreamer)     │    │  (YOLO Scoring)          │   │
│   └─────────────────┘    └──────────────────┘    └──────────────────────────┘   │
│                                   │                           │                  │
│                                   │                           │                  │
│                                   ▼                           ▼                  │
│                          ┌──────────────────┐    ┌──────────────────────────┐   │
│                          │  VLM Scheduler   │◀───│  MinIO Storage           │   │
│                          │  (Batching)      │    │  (Frame Buckets)         │   │
│                          └──────────────────┘    └──────────────────────────┘   │
│                                   │                                              │
│                                   ▼                                              │
│                          ┌──────────────────┐                                    │
│                          │  OVMS VLM Server │                                    │
│                          │  (Qwen2.5-VL-7B) │                                    │
│                          └──────────────────┘                                    │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```
### 1.2 Technology Stack
| Component | Technology | Purpose |
|-----------|------------|---------|
| Video Pipeline | GStreamer + gvapython | Frame extraction from RTSP/file |
| OCR | EasyOCR | Order ID detection from frames |
| Frame Scoring | YOLO11n INT8 OpenVINO | Quality scoring for frame selection |
| Object Storage | MinIO | Frame storage (raw + selected) |
| VLM Inference | OpenVINO Model Server | Qwen2.5-VL-7B model serving |
| IPC | Python multiprocessing | Inter-process communication |
| API Framework | FastAPI | REST endpoints |
### 1.3 Services Overview
The system consists of **three main services**:
1. **Main Application (`take-away`)**: Runs GStreamer pipeline, OCR detection, and VLM inference
2. **Frame Selector Service (`frame-selector-service`)**: Monitors MinIO for frames, scores them, selects top-k
3. **OVMS Service (`ovms-service`)**: Serves VLM model for inference
---
## 2. Entry Point and Mode Selection
### 2.1 Main Entry Point (`src/main.py`)
The application supports two operational modes controlled by `SERVICE_MODE` environment variable:
```python
# src/main.py (lines 195-241)
def main():
    """Main entry point for the application."""
    # Determine service mode
    service_mode = os.environ.get("SERVICE_MODE", "single").lower()
    
    if service_mode == "parallel":
        logger.info("Starting in PARALLEL mode with station workers")
        run_parallel_mode()
    else:
        logger.info("Starting in SINGLE worker mode")
        run_single_mode()
```
### 2.2 Single Mode
In single mode, the application:
- Starts a FastAPI server on port 8000
- Exposes REST endpoints for video upload and processing
- Uses `core/pipeline_runner.py` to run GStreamer pipeline
- Processes one video at a time
### 2.3 Parallel Mode
In parallel mode (`SERVICE_MODE=parallel`), the application:
- Creates multiple worker processes (one per station)
- Uses shared queues for VLM request batching
- Enables concurrent processing of multiple RTSP streams
```python
# src/main.py (lines 65-115)
def run_parallel_mode():
    """Run in parallel mode with multiple station workers."""
    
    # Initialize shared queue manager (IPC backbone)
    queue_manager = QueueManager()
    
    # Initialize VLM scheduler (request batching)
    vlm_scheduler = VLMScheduler(
        request_queue=queue_manager.vlm_request_queue,
        get_response_queue=queue_manager.get_response_queue,
        worker_count=int(os.getenv("VLM_WORKER_COUNT", "2"))
    )
    vlm_scheduler.start()
    
    # Initialize station manager (worker pool)
    station_manager = StationManager(
        queue_manager=queue_manager,
        station_configs=station_configs
    )
    station_manager.start_all()
    
    # Initialize metrics collector
    metrics_collector = MetricsCollector(
        queue_manager=queue_manager,
        station_manager=station_manager
    )
    metrics_collector.start()
    
    # Start FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=8000)
```
---
## 3. Parallel Mode Architecture
### 3.1 Station Manager (`src/parallel/station_manager.py`)
The `StationManager` manages a pool of worker processes, one per station:
```python
# src/parallel/station_manager.py (lines 80-150)
class StationManager:
    """Manages station worker processes."""
    
    def __init__(self, queue_manager: QueueManager, station_configs: List[Dict]):
        self.queue_manager = queue_manager
        self.station_configs = station_configs
        self.workers: Dict[str, mp.Process] = {}
        self.worker_status: Dict[str, str] = {}
        
    def start_all(self):
        """Start all station workers."""
        for config in self.station_configs:
            station_id = config["station_id"]
            self._start_worker(station_id, config)
            
    def _start_worker(self, station_id: str, config: Dict):
        """Start a single station worker process."""
        # Pre-create response queue for this station
        response_queue = self.queue_manager.get_response_queue(station_id)
        
        # Start worker process
        process = mp.Process(
            target=start_worker_process,
            args=(
                station_id,
                config,
                self.queue_manager.vlm_request_queue,
                response_queue
            ),
            daemon=False
        )
        process.start()
        self.workers[station_id] = process
```
**Key Features:**
- Uses `multiprocessing.Process` for true parallelism (bypasses GIL)
- Pre-creates response queues before spawning workers
- Supports worker health monitoring and automatic restart
- Graceful shutdown with process termination
### 3.2 Station Worker (`src/parallel/station_worker.py`)
Each station worker runs a GStreamer pipeline in its own process:
```python
# src/parallel/station_worker.py (lines 50-150)
class StationWorker:
    """Runs GStreamer pipeline for a single station."""
    
    def __init__(
        self,
        station_id: str,
        config: Dict[str, Any],
        vlm_request_queue: mp.Queue,
        vlm_response_queue: mp.Queue
    ):
        self.station_id = station_id
        self.config = config
        self.vlm_request_queue = vlm_request_queue
        self.vlm_response_queue = vlm_response_queue
        
        # Thread-safe state management
        self._lock = threading.RLock()
        self._pipeline_state = PipelineState.IDLE
        
        # Circuit breaker for failure handling
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=30.0
        )
```
**Key Features:**
- Circuit breaker pattern for fault tolerance
- Exponential backoff on failures
- Stall detection and automatic recovery
- Thread-safe state management with `RLock`
### 3.3 Worker Process Lifecycle
```
┌────────────────────────────────────────────────────────────────┐
│                    Worker Process Lifecycle                      │
├────────────────────────────────────────────────────────────────┤
│                                                                  │
│   start_worker_process()                                         │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────┐                                       │
│   │ Initialize Worker   │                                       │
│   │ - Create GStreamer  │                                       │
│   │ - Setup callbacks   │                                       │
│   └─────────────────────┘                                       │
│         │                                                        │
│         ▼                                                        │
│   ┌─────────────────────┐     ┌─────────────────────┐          │
│   │ Start Pipeline      │────▶│ GMainLoop.run()     │          │
│   │ - RTSP connection   │     │ (blocking)          │          │
│   └─────────────────────┘     └─────────────────────┘          │
│         │                            │                          │
│         │                            ▼                          │
│         │                     ┌─────────────────────┐          │
│         │                     │ Frame Callbacks     │          │
│         │                     │ - OCR detection     │          │
│         │                     │ - MinIO upload      │          │
│         │                     │ - VLM request       │          │
│         │                     └─────────────────────┘          │
│         │                            │                          │
│         ▼                            ▼                          │
│   ┌─────────────────────┐     ┌─────────────────────┐          │
│   │ EOS/Error Handler   │◀────│ Response Listener   │          │
│   │ - Cleanup           │     │ - Poll response_q   │          │
│   │ - Restart logic     │     │ - Store results     │          │
│   └─────────────────────┘     └─────────────────────┘          │
│                                                                  │
└────────────────────────────────────────────────────────────────┘
```
---
## 4. Inter-Process Communication (IPC)
### 4.1 SharedQueue Architecture (`src/parallel/shared_queue.py`)
The IPC system uses Python's `multiprocessing.Queue` for cross-process communication:
```python
# src/parallel/shared_queue.py (lines 30-100)
@dataclass
class VLMRequest:
    """Request message for VLM processing."""
    request_id: str
    station_id: str
    order_id: str
    frame_paths: List[str]
    order_items: List[Dict[str, Any]]
    timestamp: float = field(default_factory=time.time)
    
@dataclass
class VLMResponse:
    """Response message from VLM processing."""
    request_id: str
    station_id: str
    order_id: str
    status: str  # "validated" | "mismatch" | "error"
    detected_items: List[Dict[str, Any]]
    validation_result: Dict[str, Any]
    processing_time: float
    timestamp: float = field(default_factory=time.time)
```
### 4.2 Queue Manager
```python
# src/parallel/shared_queue.py (lines 150-220)
class QueueManager:
    """Manages shared queues for inter-process communication."""
    
    def __init__(self):
        # Single request queue shared by all workers
        self.vlm_request_queue = mp.Queue(maxsize=1000)
        
        # Per-station response queues
        self._response_queues: Dict[str, mp.Queue] = {}
        self._queue_lock = threading.Lock()
        
    def get_response_queue(self, station_id: str) -> mp.Queue:
        """Get or create response queue for a station."""
        with self._queue_lock:
            if station_id not in self._response_queues:
                self._response_queues[station_id] = mp.Queue(maxsize=100)
            return self._response_queues[station_id]
```
### 4.3 IPC Flow Diagram
```
┌────────────────────────────────────────────────────────────────────────────┐
│                        Inter-Process Communication Flow                      │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Station Worker 1                      VLM Scheduler                         │
│  ┌──────────────┐                     ┌──────────────┐                      │
│  │ Process 1    │                     │ Main Process │                      │
│  │              │   VLMRequest        │              │                      │
│  │  vlm_req_q ──┼────────────────────▶│  Collector   │                      │
│  │              │                     │  Thread      │                      │
│  │              │                     │      │       │                      │
│  │              │                     │      ▼       │                      │
│  │              │                     │  Batcher     │                      │
│  │              │                     │  Thread      │                      │
│  │              │                     │      │       │                      │
│  │              │                     │      ▼       │                      │
│  │              │   VLMResponse       │  Worker      │                      │
│  │  resp_q_1 ◀─┼─────────────────────│  Threads     │                      │
│  │              │                     │              │                      │
│  └──────────────┘                     └──────────────┘                      │
│                                              │                               │
│  Station Worker 2                            │                               │
│  ┌──────────────┐                            │                               │
│  │ Process 2    │                            │                               │
│  │              │   VLMRequest               │                               │
│  │  vlm_req_q ──┼────────────────────────────┤                               │
│  │              │                            │                               │
│  │              │   VLMResponse              │                               │
│  │  resp_q_2 ◀─┼─────────────────────────────┘                               │
│  │              │                                                            │
│  └──────────────┘                                                            │
│                                                                              │
│  Legend:                                                                     │
│  ────────▶  multiprocessing.Queue.put()                                     │
│  ◀────────  multiprocessing.Queue.get()                                     │
│                                                                              │
└────────────────────────────────────────────────────────────────────────────┘
```
---
## 5. Frame Pipeline Flow
### 5.1 Frame Pipeline (`src/frame_pipeline.py`)
The frame pipeline handles GStreamer frame extraction with async OCR:
```python
# src/frame_pipeline.py (lines 50-150)
class FramePipeline:
    """GStreamer pipeline with async OCR processing."""
    
    def __init__(self, station_id: str, source: str, source_type: str):
        self.station_id = station_id
        self.source = source
        
        # Async OCR processor (runs in separate thread)
        self.ocr_processor = AsyncOCRProcessor(
            sample_interval=int(os.getenv("OCR_LAG_SKIP_FRAMES", "3"))
        )
        
        # Order tracker (manages current order context)
        self.order_tracker = OrderTracker()
        
        # Pending frames buffer (CRITICAL for order transitions)
        self._pending_frames: Dict[str, List[bytes]] = {}
```
### 5.2 AsyncOCRProcessor
The OCR processor runs asynchronously to avoid blocking the video pipeline:
```python
# src/frame_pipeline.py (lines 200-280)
class AsyncOCRProcessor:
    """Async OCR processing in background thread."""
    
    def __init__(self, sample_interval: int = 3):
        self.sample_interval = sample_interval  # Skip N frames between OCR
        self._frame_count = 0
        self._ocr_queue = queue.Queue(maxsize=10)
        self._result_queue = queue.Queue(maxsize=10)
        self._ocr_thread = None
        self._running = False
        
    def submit_frame(self, frame_data: bytes):
        """Submit frame for async OCR processing."""
        self._frame_count += 1
        
        # Skip frames based on sample_interval
        if self._frame_count % self.sample_interval != 0:
            return
            
        # Add to OCR queue (non-blocking)
        try:
            self._ocr_queue.put_nowait(frame_data)
        except queue.Full:
            pass  # Drop frame if queue full
            
    def _ocr_worker(self):
        """Background thread for OCR processing."""
        reader = easyocr.Reader(['en'])
        
        while self._running:
            try:
                frame_data = self._ocr_queue.get(timeout=0.5)
                
                # Run OCR on frame
                results = reader.readtext(frame_data)
                
                # Extract order ID
                order_id = self._extract_order_id(results)
                
                if order_id:
                    self._result_queue.put(order_id)
            except queue.Empty:
                continue
```
### 5.3 Order Transition Handling
**Critical Fix Applied:** When order changes, pending frames are discarded to prevent frame mixing:
```python
# src/frame_pipeline.py (lines 350-420)
def on_frame_callback(self, frame: GstBuffer, frame_meta: Dict):
    """GStreamer callback for each frame."""
    
    # Check for new order from async OCR
    detected_order = self.ocr_processor.get_latest_order()
    
    if detected_order and detected_order != self.order_tracker.current_order:
        # ORDER CHANGE DETECTED
        
        # CRITICAL: Discard pending frames (they show OLD order content)
        if self.order_tracker.current_order is not None:
            old_order = self.order_tracker.current_order
            discarded_count = len(self._pending_frames.get(old_order, []))
            logger.info(
                f"Order change: {old_order} -> {detected_order}. "
                f"Discarding {discarded_count} pending frames."
            )
            self._pending_frames.pop(old_order, None)
        
        # Start new order tracking
        self.order_tracker.set_current_order(detected_order)
        
    # Add frame to pending buffer
    current_order = self.order_tracker.current_order
    if current_order:
        if current_order not in self._pending_frames:
            self._pending_frames[current_order] = []
        self._pending_frames[current_order].append(frame.data)
        
        # Upload to MinIO when buffer is full
        if len(self._pending_frames[current_order]) >= self.buffer_size:
            self._upload_pending_frames(current_order)
```
### 5.4 Frame Flow to MinIO
```
┌────────────────────────────────────────────────────────────────────────┐
│                         Frame Flow to MinIO                             │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   GStreamer                  Frame Pipeline                 MinIO       │
│   ┌─────────┐               ┌─────────────┐            ┌──────────────┐│
│   │ RTSP    │   frame       │ on_frame_   │            │              ││
│   │ Source  │──────────────▶│ callback()  │            │ FRAMES_      ││
│   └─────────┘               │             │            │ BUCKET       ││
│                             │  ┌────────┐ │   upload   │              ││
│                             │  │Pending │─┼───────────▶│ station_id/  ││
│                             │  │Frames  │ │            │ order_id/    ││
│                             │  └────────┘ │            │ frame_N.jpg  ││
│                             │      │      │            │              ││
│                             │      ▼      │            └──────────────┘│
│                             │  ┌────────┐ │                            │
│                             │  │ Async  │ │                            │
│                             │  │ OCR    │◀┼─── Order ID Detection      │
│                             │  └────────┘ │                            │
│                             │      │      │                            │
│                             │      ▼      │                            │
│                             │ Order Change │                            │
│                             │ Detection    │                            │
│                             └─────────────┘                            │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```
---
## 6. Frame Selector Service
### 6.1 Service Architecture (`frame-selector-service/app/frame_selector.py`)
The Frame Selector Service runs as a **separate container** that monitors MinIO for new frames:
```python
# frame-selector-service/app/frame_selector.py (lines 30-100)
class FrameSelectorService:
    """Service that monitors MinIO and selects best frames."""
    
    def __init__(self):
        # YOLO model for frame scoring
        self.yolo_model = YOLO("yolo11n_int8_openvino")
        
        # Per-station state tracking
        self.station_states: Dict[str, StationState] = {}
        
        # Configuration
        self.top_k = int(os.getenv("TOP_K_FRAMES", "3"))
        self.poll_interval = float(os.getenv("POLL_INTERVAL", "0.5"))
        
@dataclass
class StationState:
    """State tracking for a single station."""
    current_order_id: Optional[str] = None
    frames_received: int = 0
    scored_frames: List[Tuple[str, float]] = field(default_factory=list)
    order_start_time: Optional[float] = None
```
### 6.2 Frame Scoring with YOLO
```python
# frame-selector-service/app/frame_selector.py (lines 150-220)
def score_frame(self, frame_path: str) -> float:
    """Score frame using YOLO detection confidence."""
    
    # Download frame from MinIO
    frame_data = self.minio_client.get_object(
        FRAMES_BUCKET,
        frame_path
    ).read()
    
    # Decode image
    image = cv2.imdecode(
        np.frombuffer(frame_data, np.uint8),
        cv2.IMREAD_COLOR
    )
    
    # Run YOLO inference
    results = self.yolo_model(image)
    
    # Calculate score based on detection confidence
    if results and len(results[0].boxes) > 0:
        confidences = [box.conf.item() for box in results[0].boxes]
        return sum(confidences) / len(confidences)
    
    return 0.0
```
### 6.3 Top-K Frame Selection
```python
# frame-selector-service/app/frame_selector.py (lines 280-350)
def process_completed_order(self, station_id: str, order_id: str):
    """Process completed order - select top-k frames."""
    
    state = self.station_states[station_id]
    
    # Sort frames by score (descending)
    sorted_frames = sorted(
        state.scored_frames,
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top-k frames
    top_k_frames = sorted_frames[:self.top_k]
    
    # CRITICAL: Cleanup SELECTED_BUCKET before saving new frames
    self._cleanup_selected_bucket(station_id, order_id)
    
    # Copy top-k frames to SELECTED_BUCKET
    for frame_path, score in top_k_frames:
        src_path = frame_path
        dst_path = f"{station_id}/{order_id}/{os.path.basename(frame_path)}"
        
        # Copy from FRAMES_BUCKET to SELECTED_BUCKET
        self.minio_client.copy_object(
            SELECTED_BUCKET,
            dst_path,
            CopySource(FRAMES_BUCKET, src_path)
        )
        
    # Notify main application via API
    self._notify_vlm_service(station_id, order_id, top_k_frames)
```
### 6.4 Order Completion Detection
The service detects order completion in two ways:
1. **EOS (End of Stream)**: Video ends
2. **Order Change**: New order ID detected
```python
# frame-selector-service/app/frame_selector.py (lines 400-450)
def monitor_loop(self):
    """Main monitoring loop."""
    
    while self.running:
        # Poll MinIO for new frames
        for bucket_notification in self.minio_client.listen_bucket_notification(
            FRAMES_BUCKET,
            events=["s3:ObjectCreated:*"]
        ):
            event = bucket_notification['Records'][0]
            object_key = event['s3']['object']['key']
            
            # Parse: station_id/order_id/frame_N.jpg
            parts = object_key.split('/')
            station_id = parts[0]
            order_id = parts[1]
            
            # Check for order change
            state = self.get_station_state(station_id)
            
            if state.current_order_id and state.current_order_id != order_id:
                # ORDER CHANGE - process previous order
                self.process_completed_order(station_id, state.current_order_id)
                
            # Update state
            state.current_order_id = order_id
            
            # Score new frame
            score = self.score_frame(object_key)
            state.scored_frames.append((object_key, score))
```
---
## 7. VLM Inference and Validation
### 7.1 VLM Scheduler (`src/parallel/vlm_scheduler.py`)
The VLM Scheduler batches requests for efficient GPU utilization:
```python
# src/parallel/vlm_scheduler.py (lines 50-130)
class VLMScheduler:
    """Batches VLM requests for efficient processing."""
    
    def __init__(
        self,
        request_queue: mp.Queue,
        get_response_queue: Callable,
        worker_count: int = 2
    ):
        self.request_queue = request_queue
        self.get_response_queue = get_response_queue
        self.worker_count = worker_count
        
        # Internal batching
        self._batch_queue = queue.Queue()
        self._batch_timeout = 0.1  # 100ms batching window
        self._max_batch_size = 16
        
        # Threads
        self._collector_thread = None  # Collects from request_queue
        self._batcher_thread = None    # Creates batches
        self._worker_threads = []      # Processes batches
```
### 7.2 Request Collection and Batching
```
┌────────────────────────────────────────────────────────────────────────┐
│                         VLM Scheduler Threads                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌─────────────┐      ┌─────────────┐      ┌─────────────┐           │
│   │ Collector   │      │ Batcher     │      │ Worker      │           │
│   │ Thread      │      │ Thread      │      │ Thread(s)   │           │
│   └──────┬──────┘      └──────┬──────┘      └──────┬──────┘           │
│          │                    │                    │                   │
│          ▼                    │                    │                   │
│   Poll mp.Queue               │                    │                   │
│   (vlm_request_queue)         │                    │                   │
│          │                    │                    │                   │
│          │   VLMRequest       │                    │                   │
│          ├───────────────────▶│                    │                   │
│          │                    │                    │                   │
│          │                    ▼                    │                   │
│          │           Collect requests              │                   │
│          │           for 100ms or until            │                   │
│          │           max_batch_size=16             │                   │
│          │                    │                    │                   │
│          │                    │   Batch            │                   │
│          │                    ├───────────────────▶│                   │
│          │                    │                    │                   │
│          │                    │                    ▼                   │
│          │                    │           Call OVMSVLMClient          │
│          │                    │           for each request            │
│          │                    │                    │                   │
│          │                    │                    │   VLMResponse    │
│          │◀───────────────────┼────────────────────┤                   │
│          │                    │                    │                   │
│          ▼                    │                    │                   │
│   Put response in             │                    │                   │
│   station's resp_queue        │                    │                   │
│                                                                         │
└────────────────────────────────────────────────────────────────────────┘
```
### 7.3 OVMS VLM Client
```python
# src/parallel/vlm_scheduler.py (lines 200-280)
class OVMSVLMClient:
    """Client for OpenVINO Model Server VLM."""
    
    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.session = requests.Session()
        
    def generate(
        self,
        images: List[str],  # Base64 encoded images
        prompt: str,
        max_tokens: int = 512
    ) -> str:
        """Generate VLM response."""
        
        # Build messages with images
        content = []
        for img_b64 in images:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
            })
        content.append({"type": "text", "text": prompt})
        
        # Call OVMS API
        response = self.session.post(
            f"{self.endpoint}/v3/chat/completions",
            json={
                "model": "Qwen2.5-VL-7B-Instruct",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": max_tokens
            }
        )
        
        return response.json()["choices"][0]["message"]["content"]
```
### 7.4 VLM Service (`src/core/vlm_service.py`)
The VLM service handles inference and validation:
```python
# src/core/vlm_service.py (lines 100-180)
class VLMComponent:
    """Singleton VLM component for inference."""
    
    _instance = None
    
    def __init__(self):
        self.ovms_endpoint = os.getenv("OVMS_ENDPOINT", "http://ovms:8000")
        self.client = OVMSVLMClient(self.ovms_endpoint)
        
        # Sequential processing queue
        self._request_queue = asyncio.Queue()
        self._processing_task = None
        
    async def process_order(
        self,
        order_id: str,
        frame_paths: List[str],
        order_items: List[Dict]
    ) -> Dict:
        """Process order with VLM inference."""
        
        # Load frames from MinIO
        images_b64 = []
        for path in frame_paths:
            frame_data = self.minio_client.get_object(
                SELECTED_BUCKET,
                path
            ).read()
            images_b64.append(base64.b64encode(frame_data).decode())
        
        # Build prompt
        prompt = self._build_detection_prompt(order_items)
        
        # Call VLM
        response = self.client.generate(
            images=images_b64,
            prompt=prompt,
            max_tokens=512
        )
        
        # Parse detected items
        detected_items = self._parse_vlm_response(response)
        
        # Validate against order
        validation = self._validate_order(
            expected_items=order_items,
            detected_items=detected_items
        )
        
        return {
            "order_id": order_id,
            "status": "validated" if validation["match"] else "mismatch",
            "detected_items": detected_items,
            "validation": validation
        }
```
### 7.5 Validation Agent (`src/core/validation_agent.py`)
The validation agent performs semantic matching between expected and detected items:
```python
# src/core/validation_agent.py (lines 30-80)
class ValidationAgent:
    """Agent for semantic item matching."""
    
    def validate(
        self,
        expected_items: List[Dict],
        detected_items: List[Dict]
    ) -> Dict[str, Any]:
        """Validate detected items against expected items."""
        
        matched = []
        unmatched_expected = []
        unmatched_detected = []
        
        for expected in expected_items:
            best_match = None
            best_score = 0.0
            
            for detected in detected_items:
                # Semantic similarity matching
                score = self._semantic_similarity(
                    expected["name"],
                    detected["name"]
                )
                
                if score > best_score and score > 0.7:  # Threshold
                    best_score = score
                    best_match = detected
                    
            if best_match:
                matched.append({
                    "expected": expected,
                    "detected": best_match,
                    "confidence": best_score
                })
            else:
                unmatched_expected.append(expected)
                
        return {
            "match": len(unmatched_expected) == 0,
            "matched_items": matched,
            "missing_items": unmatched_expected,
            "extra_items": unmatched_detected,
            "accuracy": len(matched) / len(expected_items) if expected_items else 1.0
        }
```
---
## 8. Data Flow Diagrams
### 8.1 Complete Pipeline Flow (Parallel Mode)
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          Complete Pipeline Flow (Parallel Mode)                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│  1. VIDEO INGESTION                                                               │
│  ┌─────────────┐                                                                  │
│  │ RTSP Stream │                                                                  │
│  │ or File     │                                                                  │
│  └──────┬──────┘                                                                  │
│         │                                                                         │
│         ▼                                                                         │
│  2. STATION WORKER (Process per station)                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │ │
│  │ │ GStreamer   │───▶│ Frame       │───▶│ Async OCR   │───▶│ Order       │ │ │
│  │ │ Pipeline    │    │ Callback    │    │ Processor   │    │ Tracker     │ │ │
│  │ └─────────────┘    └──────┬──────┘    └─────────────┘    └─────────────┘ │ │
│  │                           │                                               │ │
│  │                           ▼                                               │ │
│  │                    ┌─────────────┐                                        │ │
│  │                    │ Upload to   │                                        │ │
│  │                    │ MinIO       │                                        │ │
│  │                    │ FRAMES_     │                                        │ │
│  │                    │ BUCKET      │                                        │ │
│  │                    └─────────────┘                                        │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│         │                                                                         │
│         ▼                                                                         │
│  3. FRAME SELECTOR SERVICE (Separate container)                                   │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │ │
│  │ │ Monitor     │───▶│ YOLO        │───▶│ Score       │───▶│ Select      │ │ │
│  │ │ FRAMES_     │    │ Inference   │    │ Frame       │    │ Top-K       │ │ │
│  │ │ BUCKET      │    │             │    │             │    │ Frames      │ │ │
│  │ └─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘ │ │
│  │                                                                  │        │ │
│  │                                                                  ▼        │ │
│  │                                                           ┌─────────────┐ │ │
│  │                                                           │ Copy to     │ │ │
│  │                                                           │ SELECTED_   │ │ │
│  │                                                           │ BUCKET      │ │ │
│  │                                                           └──────┬──────┘ │ │
│  │                                                                  │        │ │
│  │                                                                  ▼        │ │
│  │                                                           ┌─────────────┐ │ │
│  │                                                           │ Notify      │ │ │
│  │                                                           │ VLM Service │ │ │
│  │                                                           │ via API     │ │ │
│  │                                                           └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│         │                                                                         │
│         ▼                                                                         │
│  4. VLM SCHEDULER (Main process)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │ │
│  │ │ Collect     │───▶│ Batch       │───▶│ Worker      │───▶│ Send        │ │ │
│  │ │ Requests    │    │ Requests    │    │ Threads     │    │ Response    │ │ │
│  │ │ from Queue  │    │ (100ms)     │    │ (2)         │    │ to Station  │ │ │
│  │ └─────────────┘    └─────────────┘    └──────┬──────┘    └─────────────┘ │ │
│  │                                              │                            │ │
│  │                                              ▼                            │ │
│  │                                       ┌─────────────┐                     │ │
│  │                                       │ OVMS VLM    │                     │ │
│  │                                       │ Server      │                     │ │
│  │                                       │ Qwen2.5-VL  │                     │ │
│  │                                       │ -7B-Instruct│                     │ │
│  │                                       └─────────────┘                     │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│         │                                                                         │
│         ▼                                                                         │
│  5. VALIDATION & RESULTS                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐ │
│  │ ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐ │ │
│  │ │ Parse VLM   │───▶│ Validation  │───▶│ Store       │───▶│ Expose via  │ │ │
│  │ │ Response    │    │ Agent       │    │ Results     │    │ REST API    │ │ │
│  │ └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```
### 8.2 Order Lifecycle
```
┌────────────────────────────────────────────────────────────────────────────────┐
│                              Order Lifecycle                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Time ──────────────────────────────────────────────────────────────────────▶   │
│                                                                                  │
│  Order 384        Order 651            Order 925                                 │
│  ┌──────────┐     ┌──────────┐         ┌──────────┐                             │
│  │ Frames   │     │ Frames   │         │ Frames   │                             │
│  │ captured │     │ captured │         │ captured │                             │
│  └────┬─────┘     └────┬─────┘         └────┬─────┘                             │
│       │                │                    │                                    │
│       │ Order change   │ Order change       │ EOS                                │
│       │ detected       │ detected           │ detected                           │
│       ▼                ▼                    ▼                                    │
│  ┌─────────┐      ┌─────────┐          ┌─────────┐                              │
│  │ Discard │      │ Discard │          │ Process │                              │
│  │ pending │      │ pending │          │ final   │                              │
│  │ frames  │      │ frames  │          │ order   │                              │
│  └────┬────┘      └────┬────┘          └────┬────┘                              │
│       │                │                    │                                    │
│       ▼                ▼                    ▼                                    │
│  ┌─────────┐      ┌─────────┐          ┌─────────┐                              │
│  │ Frame   │      │ Frame   │          │ Frame   │                              │
│  │ Selector│      │ Selector│          │ Selector│                              │
│  │ Top-K   │      │ Top-K   │          │ Top-K   │                              │
│  └────┬────┘      └────┬────┘          └────┬────┘                              │
│       │                │                    │                                    │
│       ▼                ▼                    ▼                                    │
│  ┌─────────┐      ┌─────────┐          ┌─────────┐                              │
│  │ VLM     │      │ VLM     │          │ VLM     │                              │
│  │ Validate│      │ Validate│          │ Validate│                              │
│  └────┬────┘      └────┬────┘          └────┬────┘                              │
│       │                │                    │                                    │
│       ▼                ▼                    ▼                                    │
│   MISMATCH          VALIDATED           VALIDATED                                │
│                                                                                  │
└────────────────────────────────────────────────────────────────────────────────┘
```
---
## 9. Key Design Patterns
### 9.1 Multiprocessing for True Parallelism
**Why multiprocessing instead of threading?**
Python's Global Interpreter Lock (GIL) prevents true parallel execution of Python threads. For CPU-bound tasks like video processing + OCR + YOLO inference, `multiprocessing` is required:
```python
# Each station runs in its own process
process = mp.Process(
    target=start_worker_process,
    args=(station_id, config, vlm_request_queue, response_queue),
    daemon=False
)
process.start()
```
**Benefits:**
- True parallel execution across CPU cores
- Memory isolation between workers
- Fault isolation (one worker crash doesn't affect others)
### 9.2 Producer-Consumer Pattern with Queues
The system uses `multiprocessing.Queue` for thread-safe, process-safe communication:
```
Producer (Station Worker)     Consumer (VLM Scheduler)
         │                            │
         │  VLMRequest                │
         ├───────────────────────────▶│
         │                            │
         │  VLMResponse              │
         │◀───────────────────────────│
         │                            │
```
### 9.3 Circuit Breaker Pattern
Station workers implement circuit breaker for fault tolerance:
```python
class CircuitBreaker:
    """Prevents cascading failures."""
    
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing for recovery
    
    def __init__(self, failure_threshold: int, recovery_timeout: float):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.state = self.CLOSED
        self.last_failure_time = None
```
### 9.4 Async OCR Processing
OCR runs in a separate thread to avoid blocking video pipeline:
```
Main Thread (GStreamer)          OCR Thread
       │                              │
       │  Submit frame               │
       ├─────────────────────────────▶│
       │                              │ Run EasyOCR
       │                              │ (slow)
       │                              │
       │◀─────────────────────────────│
       │  Order ID result            │
       │                              │
       │ Continue processing         │
       │ without waiting             │
       ▼                              │
```
### 9.5 Singleton Pattern for VLM Component
```python
class VLMComponent:
    """Singleton for VLM inference."""
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
```
---
## 10. Configuration System
### 10.1 Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_MODE` | `single` | `single` or `parallel` mode |
| `OCR_LAG_SKIP_FRAMES` | `3` | Frames to skip between OCR attempts |
| `TOP_K_FRAMES` | `3` | Number of top frames to select |
| `VLM_WORKER_COUNT` | `2` | Number of VLM worker threads |
| `OVMS_ENDPOINT` | `http://ovms:8000` | OVMS server endpoint |
| `MINIO_ENDPOINT` | `minio:9000` | MinIO server endpoint |
| `FRAMES_BUCKET` | `frames` | Bucket for raw frames |
| `SELECTED_BUCKET` | `selected-frames` | Bucket for top-k frames |
### 10.2 Station Configuration
```json
// config/stations.json
{
  "stations": [
    {
      "station_id": "station-1",
      "source_type": "rtsp",
      "source": "rtsp://streamer:8554/station1",
      "name": "Station 1"
    },
    {
      "station_id": "station-2",
      "source_type": "rtsp",
      "source": "rtsp://streamer:8554/station2",
      "name": "Station 2"
    }
  ]
}
```
---
## Appendix A: File Structure
```
take-away/
├── src/
│   ├── main.py                    # Entry point
│   ├── frame_pipeline.py          # GStreamer + async OCR
│   ├── api/
│   │   └── endpoints.py           # REST API endpoints
│   ├── core/
│   │   ├── config_loader.py       # Configuration loading
│   │   ├── vlm_service.py         # VLM inference
│   │   ├── validation_agent.py    # Item validation
│   │   ├── order_results.py       # Results storage
│   │   └── video_history.py       # Video tracking
│   └── parallel/
│       ├── station_manager.py     # Worker pool management
│       ├── station_worker.py      # GStreamer worker process
│       ├── vlm_scheduler.py       # Request batching
│       └── shared_queue.py        # IPC queues
├── frame-selector-service/
│   └── app/
│       └── frame_selector.py      # Frame selection service
├── config/
│   └── stations.json              # Station configurations
├── docker-compose.yaml            # Service orchestration
├── Dockerfile                     # Container build
├── Makefile                       # Build automation
└── requirements.txt               # Python dependencies
```
---
## Appendix B: Known Issues and Fixes
### B.1 Frame Mixing Between Orders
**Problem:** Items from order 384 appearing in order 925's detection.
**Root Cause:** Pending frames captured before OCR detected order change contained old order items.
**Fix:** Discard ALL pending frames when order changes (frame_pipeline.py):
```python
if detected_order != self.order_tracker.current_order:
    # Discard pending frames
    self._pending_frames.pop(old_order, None)
```
### B.2 Stale Frames in SELECTED_BUCKET
**Problem:** Top-k frames from previous video loop persisted.
**Fix:** Cleanup SELECTED_BUCKET before saving new frames (frame_selector.py):
```python
def _cleanup_selected_bucket(self, station_id, order_id):
    prefix = f"{station_id}/{order_id}/"
    for obj in self.minio_client.list_objects(SELECTED_BUCKET, prefix=prefix):
        self.minio_client.remove_object(SELECTED_BUCKET, obj.object_name)
```
### B.3 OCR Lag Causing Too Few Frames
**Problem:** Order 925 only getting 1 frame due to aggressive OCR skipping.
**Fix:** Reduced `OCR_LAG_SKIP_FRAMES` from 10 to 3.
---
*Document generated for take-away order accuracy pipeline v1.0*
*Last updated: Based on code analysis session*


┌─────────────────────────────────────────────────────────────┐
│                    Main Process                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │ VLMScheduler │  │ QueueManager │  │ StationManager   │   │
│  │   (shared)   │  │   (shared)   │  │ (spawns workers) │   │
│  └──────┬───────┘  └───────┬──────┘  └──────────────────┘   │
└─────────┼──────────────────┼────────────────────────────────┘
          │                  │
          │    VLM Request Queue
          │         ▼
┌─────────┴──────────────────┴────────────────────────────────┐
│  Worker 1              Worker 2              Worker N       │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│  │ GStreamer   │      │ GStreamer   │      │ GStreamer   │  │
│  │ OCR         │      │ OCR         │      │ OCR         │  │
│  │ OrderTracker│      │ OrderTracker│      │ OrderTracker│  │
│  └─────────────┘      └─────────────┘      └─────────────┘  │
└─────────────────────────────────────────────────────────────┘