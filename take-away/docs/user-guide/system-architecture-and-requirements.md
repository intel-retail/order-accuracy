# System Architecture & Requirements

## Executive Summary

Take-Away Order Accuracy is an enterprise-grade AI vision system designed for real-time order validation in quick-service restaurant (QSR) environments. The system employs Vision Language Models (VLM) to analyze video feeds from drive-through stations, automatically identifying items in order bags and validating them against POS order data.

This document provides a comprehensive technical overview of the system architecture, component interactions, data flows, and design decisions.

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
| Order Accuracy Service | Python, FastAPI | Core orchestration and API |
| Station Workers | GStreamer (DL Streamer), multiprocessing | RTSP video processing |
| VLM Scheduler | Threading, queue batching | Request optimization |
| Frame Selector | YOLO, OpenCV | Optimal frame detection |
| OVMS VLM | OpenVINO Model Server | Vision Language Model inference |
| Semantic Service | FastAPI | Text semantic matching |
| Gradio UI | Gradio | Web interface |
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
WORKERS=N
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
| GStreamer Pipeline | RTSP → H.264 decode → `gvapython` frame processing |
| Circuit Breaker | 5 failures in 120s (2 min) → 10s cooldown |
| Exponential Backoff | 1s → 2s → 4s → ... → 15s max |
| Stall Detection | No EOS markers for 120s triggers restart |
| Health Monitoring | Frame rate, pipeline state tracking |

**Configuration (`src/parallel/station_worker.py`):**
```python
@dataclass
class PipelineConfig:
    rtsp_latency_ms: int = 0             # Zero buffering
    rtsp_retry_count: int = 50
    rtsp_timeout_us: int = 2000000       # 2 seconds
    restart_base_delay_sec: float = 1.0
    restart_max_delay_sec: float = 15.0
    circuit_breaker_max_failures: int = 5
    circuit_breaker_window_sec: float = 120.0   # 2 minutes
    circuit_breaker_cooldown_sec: float = 10.0
    stall_detection_timeout_sec: float = 120.0  # 2 minutes
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

**Responsibilities:**
- Process selected frames through VLM
- Generate item detection prompts
- Parse VLM responses into structured data
- Coordinate with validation agent

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

The pipeline is built in `src/parallel/station_worker.py` using DL Streamer's `gvapython` plugin for per-frame processing:

```
rtspsrc location=<url> latency=0 buffer-mode=0 protocols=tcp ntp-sync=false do-rtcp=false retry=5
! rtph264depay
! avdec_h264
! videoconvert
! video/x-raw,format=BGR
! videorate ! video/x-raw,framerate=<CAPTURE_FPS>/1
! queue max-size-buffers=200 leaky=no
! gvapython module=frame_pipeline function=process_frame
! fakesink sync=false
```

`CAPTURE_FPS` defaults to `10`. The `gvapython` element calls `frame_pipeline.process_frame()` per frame, which handles EasyOCR order slip detection and MinIO frame upload.

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

The selection algorithm scores each frame using YOLO detection confidence and item count, then selects the top `TOP_K` frames per order. Implementation is in `frame-selector-service/`.

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
│  • Fallback: Local semantic matching if service unavailable                      │
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
│  • Time window: 120 seconds (2 minutes)                                          │
│  • Cooldown period: 10 seconds                                                   │
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

Backoff sequence: 1s → 2s → 4s → 8s → 15s (max), with jitter. Implemented in `src/parallel/station_worker.py`.

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
│  • Configuration: WORKERS=N (set at startup)                                    │
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

| Optimization | Implementation |
|--------------|----------------|
| VLM Batching | 50-100ms time windows via VLM Scheduler |
| Frame Selection | YOLO pre-filtering reduces unnecessary VLM calls |
| INT8 Quantization | OpenVINO INT8 model served via OVMS |
| Connection Pooling | HTTP session reuse to OVMS |

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

---

## System Requirements

Hardware, software, and network requirements for deploying Take-Away Order Accuracy.

---

### Hardware Requirements

#### Development / Single-Station

| Component | Specification |
|-----------|---------------|
| **CPU** | 8+ cores |
| **RAM** | 16 GB min; 64 GB recommended for production / heavy model export workloads |
| **GPU** | Intel Arc A770 (16 GB) or equivalent Intel GPU |
| **Storage** | 50 GB SSD |

#### Production / Multi-Station

| Component | Specification |
|-----------|---------------|
| **CPU** | 16+ cores |
| **RAM** | 64 GB |
| **GPU** | Intel Data Center GPU Max (48 GB) — for 4+ concurrent stations |
| **Storage** | 200 GB NVMe SSD |

**GPU VRAM guidance:** The Qwen2.5-VL-7B INT8 model requires ~8 GB of VRAM. The default `cache_size=4` reserves an additional 4 GB VRAM for the KV cache. Total VRAM needed ≈ 12 GB — fits in Intel Arc A770 16 GB. On **integrated GPU** (iGPU) platforms such as Wildcat Lake and Meteor Lake, the KV cache is drawn from **system RAM** instead of dedicated VRAM — use a smaller value (e.g. `CACHE_SIZE=2`) to avoid exhausting system RAM. Set `export CACHE_SIZE=<N>` before running `setup_models.sh`. For a full per-platform sizing table and step-by-step instructions see [ovms-service/README.md — Tuning the KV Cache Size](../../ovms-service/README.md#tuning-the-kv-cache-size).

> **ℹ Model Export RAM Note:** 16 GB system RAM is sufficient for **inference-only** deployments. For first-time model export (`setup_models.sh` INT8 quantization), a higher-memory host (48–64 GB recommended) avoids potential OOM and corrupt IR files — export once there and copy `ovms-service/models/` to the target system. If you must export on 16 GB, set `export CACHE_SIZE=2` first. See [ovms-service/README.md — Tuning the KV Cache Size](../../ovms-service/README.md#tuning-the-kv-cache-size) for details.

---

### Software Requirements

#### Operating System

Ubuntu 22.04 LTS is the validated platform (matches the `intel/dlstreamer:2025.2.0-ubuntu22` base image).

#### Container Runtime

| Software | Minimum Version |
|----------|-----------------|
| Docker Engine | 24.0 |
| Docker Compose | V2 (2.20+) |

#### GPU Drivers

Intel GPU drivers must be installed from [packages.intel.com](https://packages.intel.com). Verify the GPU is accessible to Docker:

```bash
ls /dev/dri/
docker run --rm --device /dev/dri intel/openvino_dev:latest python3 -c \
  "from openvino.runtime import Core; print(Core().available_devices)"
```

Expected output includes `GPU`.

---

### Network Requirements

#### Port Configuration

| Service | Port | Purpose |
|---------|------|---------|
| Order Accuracy API | 8000 | REST API |
| OVMS VLM | 8001 | Model inference |
| Gradio UI | 7860 | Web interface |
| MinIO API | 9000 | S3-compatible storage |
| MinIO Console | 9001 | Storage admin UI |
| Semantic Service | 8080 | Semantic matching |
| RTSP Streamer | 8554 | Video streaming (parallel mode) |

#### RTSP Requirements (Parallel Mode)

| Requirement | Specification |
|-------------|---------------|
| Protocol | RTSP/RTP over TCP |
| Codec | H.264 |
| Resolution | 720p–1080p |
| Frame Rate | 15–30 FPS |

---

## Pre-Deployment Checklist

- [ ] Docker and Docker Compose installed and working
- [ ] Intel GPU drivers installed and GPU visible to Docker
- [ ] Required ports available (8000, 7860, 8001, 9000, 9001, 8080)
- [ ] At least 50 GB free disk space
- [ ] **16 GB+ RAM available** (sufficient for inference; for first-time model export 48–64 GB recommended — export on a high-RAM host and copy `ovms-service/models/` to the target system)
- [ ] VLM model downloaded (`setup_models.sh` completed)
- [ ] `.env` file configured
- [ ] Camera RTSP URLs accessible from host (parallel mode)
