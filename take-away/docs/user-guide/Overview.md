# Take-Away Order Accuracy: Technical Overview

## Executive Summary

Take-Away Order Accuracy is an enterprise-grade AI vision system designed for real-time order validation in quick-service restaurant (QSR) environments. The system employs Vision Language Models (VLM) to analyze video feeds from drive-through stations, automatically identifying items in order bags and validating them against POS order data.

This document provides a comprehensive technical overview of the system architecture, component interactions, data flows, and design decisions.

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Service Modes](#service-modes)
3. [Core Components](#core-components)
4. [Data Flow Pipeline](#data-flow-pipeline)
5. [Video Processing Architecture](#video-processing-architecture)
6. [VLM Integration](#vlm-integration)
7. [Frame Selection Service](#frame-selection-service)
8. [Semantic Matching](#semantic-matching)
9. [Docker Services Topology](#docker-services-topology)
10. [Production Patterns](#production-patterns)
11. [Scalability Considerations](#scalability-considerations)

---

## System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                            TAKE-AWAY ORDER ACCURACY SYSTEM                           │
│                                                                                      │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐                  │
│  │    RTSP Video   │    │  Order Accuracy │    │     MinIO       │                  │
│  │     Streams     │───▶│    Service      │───▶│   (Frame        │                  │
│  │   (GStreamer)   │    │  (EasyOCR)      │    │    Storage)     │                  │
│  └─────────────────┘    └─────────────────┘    └────────┬────────┘                  │
│                                                          │                           │
│                                                          ▼                           │
│                                                 ┌─────────────────┐                  │
│                                                 │  Frame Selector │                  │
│                                                 │  (YOLO11n-CPU)  │                  │
│                                                 └────────┬────────┘                  │
│                                                          │ top 3 frames              │
│         ┌────────────────────────────────────────────────┤                           │
│         │                                                │                           │
│         ▼                                                ▼                           │
│  ┌─────────────────┐                          ┌─────────────────┐                   │
│  │   VLM Scheduler │                          │    Validation   │                   │
│  │  (ThreadPool)   │                          │      Agent      │                   │
│  └────────┬────────┘                          └────────┬────────┘                   │
│           │                                            │                             │
│           ▼                                            ▼                             │
│  ┌─────────────────┐                          ┌─────────────────┐                   │
│  │    OVMS VLM     │                          │    Semantic     │                   │
│  │  (Qwen2.5-VL)  │                          │    Service      │                   │
│  │    GPU-INT8     │                          └─────────────────┘                   │
│  └─────────────────┘                                                                │
│                                                                                      │
└──────────────────────────────────────────────────────────────────────────────────────┘

                              │
                              ▼
                      ┌─────────────┐
                      │  Gradio UI  │
                      │ (Interface) │
                      └─────────────┘
```

### Component Summary

| Component | Technology | Purpose |
|-----------|------------|---------|
| Order Accuracy Service | Python 3.10, FastAPI | Core orchestration and API |
| Station Workers | GStreamer, multiprocessing | RTSP video processing |
| VLM Scheduler | Threading, queue batching | Request optimization |
| Frame Selector | YOLO, OpenCV | Optimal frame detection |
| OVMS VLM | OpenVINO Model Server | Vision Language Model inference |
| Semantic Service | FastAPI, sentence-transformers | Text semantic matching |
| Gradio UI | Gradio 3.x | Web interface |
| MinIO | S3-compatible storage | Frame and result storage |

---

## Service Modes

The system supports two operational modes, each optimized for different deployment scenarios.

### Single Worker Mode

```
┌─────────────────────────────────────────────────────────────┐
│                    SINGLE WORKER MODE                        │
│                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌────────────┐  │
│  │  Gradio UI │────▶│  FastAPI REST  │────▶│  VLM       │  │
│  │            │     │  /upload-video │     │  Service   │  │
│  └────────────┘     └────────────────┘     └────────────┘  │
│                                                              │
│  Characteristics:                                            │
│  • Sequential video processing                              │
│  • Direct VLM calls (no batching)                          │
│  • Best for: Development, testing, demos                   │
└─────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
SERVICE_MODE=single
WORKERS=0
```

**Features:**
- Video upload via REST API
- Single order at a time
- Gradio UI integration
- FastAPI Swagger documentation

### Parallel Worker Mode

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PARALLEL WORKER MODE                                 │
│                                                                              │
│  ┌────────────┐   ┌────────────┐   ┌────────────┐                          │
│  │  Station   │   │  Station   │   │  Station   │  (independent,           │
│  │  Worker 1  │   │  Worker 2  │   │  Worker N  │   each with GStreamer    │
│  │ (GStr+OCR) │   │ (GStr+OCR) │   │ (GStr+OCR) │   + EasyOCR)            │
│  └─────┬──────┘   └─────┬──────┘   └─────┬──────┘                          │
│        │               │               │                                    │
│        ▼               ▼               ▼                                    │
│  ┌──────────────────────────────────────────────┐                          │
│  │              MinIO (Frame Storage)            │                          │
│  │         station_1/  station_2/  station_N/    │                          │
│  └─────────────────────┬────────────────────────┘                          │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────────┐                          │
│  │       Frame Selector (YOLO11n - CPU)          │                          │
│  │       Select top 3 frames per order           │                          │
│  └─────────────────────┬────────────────────────┘                          │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────────┐                          │
│  │     VLM Scheduler (ThreadPoolExecutor)        │                          │
│  │     Parallel requests to OVMS                 │                          │
│  └─────────────────────┬────────────────────────┘                          │
│                         │                                                    │
│                         ▼                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                           OVMS VLM (GPU)                              │  │
│  │              Qwen2.5-VL-7B / Continuous Batching                     │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  Characteristics:                                                            │
│  • Independent RTSP stream per station                                      │
│  • Shared EasyOCR, YOLO, and VLM models                                   │
│  • Parallel VLM requests via ThreadPoolExecutor                            │
│  • OVMS continuous batching on GPU                                          │
│  • Best for: Production, multi-camera deployments                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Configuration:**
```bash
SERVICE_MODE=parallel
WORKERS=4
SCALING_MODE=fixed  # or 'auto'
```

**Features:**
- Independent GStreamer pipeline per station
- Shared EasyOCR, YOLO, and VLM models across stations
- Parallel VLM requests via ThreadPoolExecutor
- OVMS continuous batching on GPU
- Circuit breaker pattern with exponential backoff

---

## Core Components

### 1. Main Entry Point (`src/main.py`)

The unified service entry point manages mode selection and service initialization.

```python
# Mode Selection Logic
SERVICE_MODE = os.getenv("SERVICE_MODE", "single")

if SERVICE_MODE == "single":
    # Start FastAPI with REST endpoints
    run_single_mode()
elif SERVICE_MODE == "parallel":
    # Start multi-worker orchestration
    run_parallel_mode()
```

**Responsibilities:**
- Environment configuration loading
- Mode-based service initialization
- Signal handling for graceful shutdown
- Worker process spawning (parallel mode)
- Hostname-based station detection

### 2. Station Worker (`src/parallel/station_worker.py`)

Production-ready worker process for single camera stream processing.

```
┌─────────────────────────────────────────────────────────────┐
│                    STATION WORKER LIFECYCLE                   │
│                                                               │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │Initialize│──▶│Wait RTSP │──▶│  Start   │──▶│ Monitor  │ │
│  │          │   │          │   │ Pipeline │   │ Health   │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬─────┘ │
│                                                     │        │
│       ┌──────────────────┬──────────────────────────┘        │
│       │                  │                                    │
│       ▼                  ▼                                    │
│  ┌──────────┐      ┌──────────┐      ┌──────────┐           │
│  │ Circuit  │      │ Backoff  │      │  Verify  │           │
│  │ Breaker  │─────▶│ & Retry  │─────▶│  RTSP    │──▶Restart │
│  │  Check   │      │          │      │          │           │
│  └──────────┘      └──────────┘      └──────────┘           │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**

| Feature | Implementation |
|---------|----------------|
| GStreamer Pipeline | RTSP → H.264 decode → Frame capture |
| Circuit Breaker | 5 failures in 300s (5 min) → 30s cooldown |
| Exponential Backoff | 2s → 4s → 8s → ... → 60s max |
| Stall Detection | No EOS markers for 300s triggers restart |
| Health Monitoring | Frame rate, pipeline state tracking |

**Configuration:**
```python
@dataclass
class PipelineConfig:
    rtsp_latency_ms: int = 0             # Zero buffering
    rtsp_retry_count: int = 50
    rtsp_timeout_us: int = 5000000       # 5 seconds
    restart_base_delay_sec: float = 2.0
    restart_max_delay_sec: float = 60.0
    circuit_breaker_max_failures: int = 5
    circuit_breaker_window_sec: float = 300.0   # 5 minutes
    circuit_breaker_cooldown_sec: float = 30.0
    stall_detection_timeout_sec: float = 300.0  # 5 minutes
```

### 3. VLM Scheduler (`src/parallel/vlm_scheduler.py`)

Request batching scheduler optimizing OVMS throughput.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VLM SCHEDULER ARCHITECTURE                       │
│                                                                          │
│  Worker 1 ──┐                                                            │
│             │     ┌─────────────┐     ┌───────────────┐                 │
│  Worker 2 ──┼────▶│  Collector  │────▶│               │                 │
│             │     │   Thread    │     │    Batch      │    ┌─────────┐  │
│  Worker N ──┘     │             │     │    Buffer     │───▶│  OVMS   │  │
│                   └─────────────┘     │   (50-100ms)  │    │   VLM   │  │
│                                       │               │    └────┬────┘  │
│                   ┌─────────────┐     └───────────────┘         │       │
│  Response ◀───────│  Response   │◀──────────────────────────────┘       │
│  Routing          │   Router    │                                        │
│                   └─────────────┘                                        │
│                                                                          │
│  Time-window batching: Collect requests for 50-100ms, send as batch     │
│  Fair scheduling: Round-robin across workers                             │
│  Backpressure: Queue limits prevent memory exhaustion                   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Batching Strategy:**
- **Time Window**: 50-100ms collection period
- **Max Batch Size**: Configurable (default: 16)
- **Fair Scheduling**: Round-robin request servicing
- **Response Routing**: Match responses to original requesters

### 4. VLM Component (`src/core/vlm_service.py`)

Vision Language Model processing with inventory detection and order validation.

```python
class VLMComponent:
    """
    Responsibilities:
    - Process selected frames through VLM
    - Generate item detection prompts
    - Parse VLM responses into structured data
    - Coordinate with validation agent
    """
    
    def extract_items(
        self,
        image_paths: List[str]
    ) -> dict:
        # 1. Encode frames for VLM
        # 2. Generate detection prompt
        # 3. Call OVMS VLM endpoint
        # 4. Parse JSON response
        # 5. Return detected items
        pass
```

**VLM Prompt Template:**
```
You are an order verification assistant. Analyze the image and identify all 
visible food items and their quantities.

Return JSON format:
{
  "detected_items": [
    {"name": "item_name", "quantity": N}
  ]
}
```

### 5. OVMS VLM Client (`src/core/ovms_client.py`)

OpenVINO Model Server client with OpenAI-compatible API.

```
┌─────────────────────────────────────────────────────────────┐
│                      OVMS CLIENT                             │
│                                                              │
│  ┌────────────┐     ┌────────────────┐     ┌────────────┐  │
│  │   Image    │────▶│  Base64        │────▶│   POST     │  │
│  │  (numpy)   │     │  Encoding      │     │/v3/chat/   │  │
│  └────────────┘     └────────────────┘     │completions │  │
│                                             └─────┬──────┘  │
│                                                   │         │
│  ┌────────────┐     ┌────────────────┐           │         │
│  │  Metrics   │◀────│   Response     │◀──────────┘         │
│  │  Logging   │     │   Parsing      │                      │
│  └────────────┘     └────────────────┘                      │
└─────────────────────────────────────────────────────────────┘
```

**API Integration:**
```python
# OpenAI-compatible chat/completions endpoint
response = requests.post(
    f"{OVMS_ENDPOINT}/v3/chat/completions",
    json={
        "model": OVMS_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        "max_completion_tokens": 100
    }
)
```

---

## Data Flow Pipeline

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW PIPELINE                                      │
│                                                                                      │
│  1. VIDEO CAPTURE                                                                    │
│  ┌─────────────┐                                                                    │
│  │ RTSP Camera │──▶ GStreamer Pipeline ──▶ Frame Buffer                            │
│  └─────────────┘                                                                    │
│                                        │                                             │
│  2. FRAME SELECTION                    ▼                                             │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Frame Selector (YOLO)                                                        │   │
│  │ • Object detection on raw frames                                             │   │
│  │ • Score frames by item visibility                                            │   │
│  │ • Select top K frames per order                                              │   │
│  │ • Store selected frames in MinIO                                             │   │
│  └───────────────────────────────────────────────────────────────┬──────────────┘   │
│                                                                   │                  │
│  3. VLM PROCESSING                                               ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ VLM Scheduler → OVMS (Qwen2.5-VL)                                            │   │
│  │ • Batch frames by time window                                                │   │
│  │ • Send to OVMS with detection prompt                                         │   │
│  │ • Parse structured item response                                             │   │
│  └───────────────────────────────────────────────────────────────┬──────────────┘   │
│                                                                   │                  │
│  4. ORDER VALIDATION                                             ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Validation Agent                                                             │   │
│  │ • Compare detected items with expected order                                 │   │
│  │ • Exact match → Semantic match → Flag mismatch                              │   │
│  │ • Generate validation result                                                 │   │
│  └───────────────────────────────────────────────────────────────┬──────────────┘   │
│                                                                   │                  │
│  5. RESULT OUTPUT                                                ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────────────┐   │
│  │ Output: { "matched": [...], "missing": [...], "extra": [...] }              │   │
│  └─────────────────────────────────────────────────────────────────────────────┘   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### State Transitions

```
┌──────────────────────────────────────────────────────────────┐
│                  WORKER STATE MACHINE                         │
│                                                               │
│   STOPPED ──▶ STARTING ──▶ RUNNING ──▶ STALLED               │
│      │            │            │           │                  │
│      │            │            │           │                  │
│      │            ▼            │           ▼                  │
│      │     CIRCUIT_OPEN ◀──────┴───▶ RESTARTING              │
│      │            │                        │                  │
│      │            │                        │                  │
│      ▼            ▼                        ▼                  │
│   SHUTTING_DOWN ◀──────────────────────────┘                 │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Video Processing Architecture

### GStreamer Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         GSTREAMER PIPELINE                                       │
│                                                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ rtspsrc  │──▶│ rtph264  │──▶│ h264parse│──▶│ avdec_   │──▶│ video    │     │
│  │          │   │ depay    │   │          │   │ h264     │   │ convert  │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│                                                                     │           │
│                                                                     ▼           │
│                                                              ┌──────────┐       │
│                                                              │ appsink  │       │
│                                                              │ (numpy)  │       │
│                                                              └──────────┘       │
│                                                                                  │
│  Pipeline String:                                                               │
│  rtspsrc location=rtsp://host/stream latency=100 !                             │
│  rtph264depay ! h264parse ! avdec_h264 ! videoconvert !                        │
│  video/x-raw,format=BGR ! appsink emit-signals=true max-buffers=1              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Frame Processing Flow

```python
# Frame capture callback
def on_new_sample(appsink):
    sample = appsink.emit("pull-sample")
    buffer = sample.get_buffer()
    array = buffer.extract_dup(0, buffer.get_size())
    
    # Convert to numpy array
    frame = np.frombuffer(array, dtype=np.uint8)
    frame = frame.reshape((height, width, 3))
    
    # Add to processing queue
    frame_queue.put((timestamp, frame))
```

---

## VLM Integration

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         OVMS VLM INTEGRATION                                     │
│                                                                                  │
│  Model: Qwen/Qwen2.5-VL-7B-Instruct                                             │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                         OVMS Model Server                               │    │
│  │                                                                         │    │
│  │  ┌────────────────┐    ┌────────────────┐    ┌────────────────┐       │    │
│  │  │  Vision        │    │  Language      │    │  Output        │       │    │
│  │  │  Encoder       │───▶│  Model         │───▶│  Decoder       │       │    │
│  │  │  (ViT-based)   │    │  (Qwen2.5)     │    │  (JSON)        │       │    │
│  │  └────────────────┘    └────────────────┘    └────────────────┘       │    │
│  │                                                                         │    │
│  │  API: OpenAI-compatible /v3/chat/completions                           │    │
│  │  Port: 8001 (configurable)                                             │    │
│  │  Precision: INT8 (optimized for inference)                             │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Request/Response Format

**Request:**
```json
{
  "model": "Qwen/Qwen2.5-VL-7B-Instruct",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Identify all food items in this image..."},
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
      ]
    }
  ],
  "max_completion_tokens": 100,
  "temperature": 0.2
}
```

**Response:**
```json
{
  "choices": [
    {
      "message": {
        "content": "{\"detected_items\": [{\"name\": \"burger\", \"quantity\": 2}]}"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 45,
    "total_tokens": 1295
  }
}
```

---

## Frame Selection Service

### YOLO-Based Frame Selection

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       FRAME SELECTOR SERVICE                                     │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                     frame_selector.py                                   │    │
│  │                                                                         │    │
│  │  1. Monitor frames bucket in MinIO                                      │    │
│  │  2. Run YOLO object detection on each frame                            │    │
│  │  3. Score frames by:                                                    │    │
│  │     - Object detection confidence                                       │    │
│  │     - Item visibility/occlusion                                         │    │
│  │     - Frame quality (blur, brightness)                                  │    │
│  │  4. Select TOP_K frames per order                                       │    │
│  │  5. Store selected frames in 'selected' bucket                         │    │
│  │  6. Trigger VLM processing via HTTP callback                           │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Configuration:                                                                  │
│  • TOP_K: 3 (frames per order)                                                  │
│  • POLL_INTERVAL: 1.5s                                                          │
│  • MIN_FRAMES_PER_ORDER: 1                                                      │
│  • YOLO_MODEL: yolo11n (INT8 OpenVINO)                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Selection Algorithm

```python
def select_best_frames(frames: List[Frame], top_k: int = 3) -> List[Frame]:
    """
    Select optimal frames for VLM processing.
    
    Scoring criteria:
    1. YOLO detection confidence (40%)
    2. Number of detected objects (30%)
    3. Frame quality metrics (30%)
    """
    scored_frames = []
    
    for frame in frames:
        # YOLO inference
        detections = yolo_model(frame.image)
        
        # Calculate score
        confidence_score = sum(d.confidence for d in detections) / len(detections)
        count_score = min(len(detections) / 10, 1.0)
        quality_score = calculate_frame_quality(frame.image)
        
        total_score = (0.4 * confidence_score + 
                       0.3 * count_score + 
                       0.3 * quality_score)
        
        scored_frames.append((total_score, frame))
    
    # Return top K frames
    scored_frames.sort(key=lambda x: x[0], reverse=True)
    return [frame for _, frame in scored_frames[:top_k]]
```

---

## Semantic Matching

### Matching Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SEMANTIC MATCHING SYSTEM                                  │
│                                                                                  │
│  ┌────────────────────────────────────────────────────────────────────────┐    │
│  │                    Validation Agent                                     │    │
│  │                                                                         │    │
│  │  Pass 1: EXACT MATCH                                                    │    │
│  │  ───────────────────                                                    │    │
│  │  "burger" == "burger" → MATCH                                          │    │
│  │                                                                         │    │
│  │  Pass 2: SEMANTIC MATCH (if exact fails)                               │    │
│  │  ─────────────────────────────────────────                              │    │
│  │  "quarter pounder" ≈ "quarterpounder" → MATCH (semantic similarity)   │    │
│  │                                                                         │    │
│  │  Pass 3: FLAG MISMATCH                                                  │    │
│  │  ────────────────────                                                   │    │
│  │  No match found → Add to "missing" or "extra"                          │    │
│  └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Semantic Service Integration:                                                   │
│  • External microservice: http://semantic-service:8080                          │
│  • Fallback: Local semantic matching (sentence-transformers)                    │
│  • Similarity threshold: 0.85 (configurable)                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Matching Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| `exact` | String equality comparison | Fast, deterministic |
| `semantic` | Embedding similarity | Handles variations |
| `hybrid` | Exact first, then semantic | Recommended default |

---

## Docker Services Topology

### Service Deployment

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DOCKER SERVICES TOPOLOGY                                 │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                         order-accuracy-net                               │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │   │
│  │  │   minio      │  │  ovms-vlm    │  │order-accuracy│                  │   │
│  │  │   :9000/9001 │  │   :8001      │  │   :8000      │                  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │   │
│  │                                                                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                  │   │
│  │  │frame-selector│  │  gradio-ui   │  │semantic-svc  │                  │   │
│  │  │  (internal)  │  │   :7860      │  │   :8080      │                  │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘                  │   │
│  │                                                                          │   │
│  │  ┌──────────────┐                                                      │   │
│  │  │rtsp-streamer │  (parallel profile)                                  │   │
│  │  │   :8554      │                                                      │   │
│  │  └──────────────┘                                                      │   │
│  │                                                                          │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Volumes:                                                                        │
│  • minio-data: S3-compatible object storage                                     │
│  • videos: Input video files                                                    │
│  • results: Output results and metrics                                          │
│  • models: OVMS model files                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Service Dependencies

```yaml
services:
  order-accuracy:
    depends_on:
      - minio
      - ovms-vlm
    
  frame-selector:
    depends_on:
      - minio
      - order-accuracy
    
  gradio-ui:
    depends_on:
      - order-accuracy
```

---

## Production Patterns

### Circuit Breaker Pattern

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         CIRCUIT BREAKER PATTERN                                  │
│                                                                                  │
│  Configuration:                                                                  │
│  • Failure threshold: 5 failures                                                │
│  • Time window: 300 seconds (5 minutes)                                          │
│  • Cooldown period: 30 seconds                                                   │
│                                                                                  │
│  States:                                                                         │
│  ┌──────────┐   5 failures    ┌──────────┐   30s elapsed   ┌──────────┐       │
│  │  CLOSED  │───────────────▶│   OPEN   │────────────────▶│HALF-OPEN │       │
│  │ (Normal) │                │(Blocking)│                 │(Testing) │       │
│  └──────────┘                └──────────┘                 └────┬─────┘       │
│       ▲                                                        │              │
│       │                            success                     │              │
│       └────────────────────────────────────────────────────────┘              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Exponential Backoff

```python
def calculate_backoff(attempt: int) -> float:
    """
    Exponential backoff with jitter.
    
    Backoff sequence: 2s, 4s, 8s, 16s, 32s, 60s (max)
    """
    base_delay = 2.0
    max_delay = 60.0
    
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = random.uniform(0, delay * 0.1)
    
    return delay + jitter
```

### Health Monitoring

```python
@dataclass
class PipelineMetrics:
    pipeline_restarts: int = 0
    pipeline_failures: int = 0
    rtsp_unavailable_events: int = 0
    successful_frames_processed: int = 0
    circuit_breaker_trips: int = 0
    stall_detections: int = 0
    last_frame_time: float = 0.0
    pipeline_start_time: float = 0.0
    total_uptime_sec: float = 0.0
```

---

## Scalability Considerations

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SCALING ARCHITECTURE                                     │
│                                                                                  │
│  Fixed Scaling (SCALING_MODE=fixed):                                            │
│  • Static worker count defined at startup                                       │
│  • Configuration: WORKERS=4                                                     │
│                                                                                  │
│  Auto Scaling (SCALING_MODE=auto):                                              │
│  • Dynamic worker adjustment based on metrics                                   │
│  • Scale up triggers:                                                           │
│    - GPU utilization > 80%                                                      │
│    - Average VLM latency > target                                               │
│    - Request queue depth > threshold                                            │
│  • Scale down triggers:                                                         │
│    - GPU utilization < 30%                                                      │
│    - Idle workers for > 5 minutes                                               │
│                                                                                  │
│  Bottleneck Analysis:                                                            │
│  • VLM inference: Primary bottleneck (~2-3s per request)                       │
│  • Mitigation: Request batching via VLM Scheduler                              │
│  • Target throughput: 20-30 orders/minute per GPU                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Performance Optimization

| Optimization | Implementation | Impact |
|--------------|----------------|--------|
| VLM Batching | 50-100ms time windows | +40% throughput |
| Frame Selection | YOLO pre-filtering | -60% VLM calls |
| INT8 Quantization | OpenVINO model optimization | +30% inference speed |
| Connection Pooling | Reuse HTTP connections | -20% latency |

---

## Summary

Take-Away Order Accuracy is a production-ready system designed for high-throughput, reliable order validation in QSR environments. Key architectural highlights:

1. **Dual-Mode Operation**: Single worker for development, parallel workers for production
2. **Resilient Video Processing**: GStreamer with circuit breaker and auto-recovery
3. **Optimized VLM Inference**: Request batching and INT8 quantization
4. **Intelligent Frame Selection**: YOLO-based filtering reduces unnecessary VLM calls
5. **Hybrid Matching**: Exact + semantic matching for robust item comparison
6. **Production Patterns**: Circuit breaker, exponential backoff, health monitoring

For deployment and configuration details, see the companion guides in this documentation suite.
