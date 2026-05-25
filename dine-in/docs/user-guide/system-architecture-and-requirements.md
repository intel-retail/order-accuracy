# System Architecture & Requirements

Staff-triggered plate validation workflow designed for full-service restaurant expo operations. This solution demonstrates zero-training deployment with vision-language models (VLM), semantic order reconciliation, and latency instrumentation aligned to operational service windows.

## Introduction

The Dine-In Order Accuracy solution enables restaurant staff to validate food plates against customer orders before serving. Unlike take-away scenarios with conveyor-based automation, dine-in operations require manual triggering by expo staff when plates are ready for service.

### Use Case

In a full-service restaurant:
1. Kitchen prepares a dish for Table 12
2. Expo staff places the plate in the validation station
3. Staff triggers validation via Gradio UI or API
4. System analyzes plate contents using VLM
5. System compares detected items against the order manifest
6. Staff receives immediate feedback on order accuracy

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Zero-Training Deployment** | Uses pre-trained Qwen2.5-VL-7B model - no fine-tuning required |
| **Semantic Matching** | Fuzzy item matching handles naming variations (e.g., "Big Mac" ↔ "Maharaja Mac") |
| **Real-Time Validation** | Sub-15-second end-to-end latency for operational efficiency |
| **Circuit Breaker Pattern** | Fault-tolerant with automatic service recovery |
| **Connection Pooling** | Optimized HTTP/2 connections for high throughput |
| **Bounded Caching** | LRU cache prevents memory exhaustion under load |
| **Comprehensive Metrics** | CPU, GPU, memory utilization with token-level inference stats |

---

## How It Works

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DINE-IN ORDER ACCURACY                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌──────────────────┐      ┌─────────────────────┐    │
│  │             │      │                  │      │                     │    │
│  │  Gradio UI  │─────▶│   FastAPI API    │─────▶│   Validation        │    │
│  │  (Port 7861)│      │   (Port 8083)    │      │   Service           │    │
│  │             │      │                  │      │                     │    │
│  └─────────────┘      └────────┬─────────┘      └──────────┬──────────┘    │
│                                │                           │               │
│                                │                           │               │
│                    ┌───────────┴───────────┐               │               │
│                    │                       │               │               │
│                    ▼                       ▼               ▼               │
│           ┌────────────────┐     ┌─────────────────┐ ┌───────────────┐    │
│           │                │     │                 │ │               │    │
│           │  VLM Client    │     │ Semantic Client │ │ Metrics       │    │
│           │  (Circuit      │     │ (Circuit        │ │ Collector     │    │
│           │   Breaker)     │     │  Breaker)       │ │               │    │
│           │                │     │                 │ │               │    │
│           └───────┬────────┘     └────────┬────────┘ └───────────────┘    │
│                   │                       │                               │
└───────────────────┼───────────────────────┼───────────────────────────────┘
                    │                       │
                    ▼                       ▼
          ┌─────────────────┐     ┌─────────────────┐
          │                 │     │                 │
          │   OVMS VLM      │     │   Semantic      │
          │   (Qwen2.5-VL)  │     │   Service       │
          │   Port 8000     │     │   Port 8080     │
          │                 │     │                 │
          └─────────────────┘     └─────────────────┘
```

### Request Flow

```
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐
│  Staff   │    │ Gradio  │    │ FastAPI  │    │  VLM    │    │ Semantic │
│  Trigger │    │   UI    │    │   API    │    │ Client  │    │  Client  │
└────┬─────┘    └────┬────┘    └────┬─────┘    └────┬────┘    └────┬─────┘
     │               │              │               │              │
     │ Select Image  │              │               │              │
     │──────────────▶│              │               │              │
     │               │              │               │              │
     │ Click Validate│              │               │              │
     │──────────────▶│              │               │              │
     │               │              │               │              │
     │               │ POST /validate               │              │
     │               │─────────────▶│               │              │
     │               │              │               │              │
     │               │              │ Preprocess    │              │
     │               │              │ Image         │              │
     │               │              │───────────────│              │
     │               │              │               │              │
     │               │              │ analyze_plate()              │
     │               │              │──────────────▶│              │
     │               │              │               │              │
     │               │              │               │ OVMS POST    │
     │               │              │               │─────────────▶│
     │               │              │               │              │
     │               │              │               │◀─────────────│
     │               │              │               │ Detected Items
     │               │              │◀──────────────│              │
     │               │              │               │              │
     │               │              │ match_items()                │
     │               │              │─────────────────────────────▶│
     │               │              │                              │
     │               │              │◀─────────────────────────────│
     │               │              │            Similarity Scores │
     │               │              │               │              │
     │               │◀─────────────│               │              │
     │               │ Validation Result            │              │
     │◀──────────────│              │               │              │
     │ Display Results              │               │              │
     │               │              │               │              │
```

---

## System Architecture

### Docker Services

| Container | Image | Ports | Description |
|-----------|-------|-------|-------------|
| `dinein_app` | `intel/order-accuracy-dine-in:2026.0.0` | 7861, 8083 | Main application (Gradio + FastAPI) |
| `dinein_ovms_vlm` | `openvino/model_server:latest-gpu` | 8002 | Vision-Language Model server |
| `dinein_semantic_service` | `intel/semantic-search-agent:1.0.0` | 8081, 9091 | Semantic text matching |
| `metrics-collector` | `intel/hl-ai-metrics-collector:1.0.0` | 8084 | System metrics aggregation |

### Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Network: dinein-net               │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   dinein_app    │    │ dinein_ovms_vlm │                    │
│  │                 │    │                 │                    │
│  │  - Gradio:7861  │───▶│  - REST: 8000   │  (internal)        │
│  │  - API:8083     │    │  - Host: 8002   │  (external)        │
│  │                 │    │                 │                    │
│  └────────┬────────┘    └─────────────────┘                    │
│           │                                                     │
│           │             ┌─────────────────┐                    │
│           │             │ semantic_service│                    │
│           └────────────▶│                 │                    │
│                         │  - REST: 8080   │  (internal)        │
│                         │  - Host: 8081   │  (external)        │
│                         └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │metrics-collector│                                           │
│  │  - REST: 8084   │◀────── Prometheus-style metrics           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼ Host Network
        ┌───────────────────────┐
        │   localhost:7861      │  ← Gradio UI
        │   localhost:8083      │  ← REST API
        │   localhost:8083/docs │  ← Swagger Docs
        │   localhost:8002      │  ← OVMS VLM
        │   localhost:8084      │  ← Metrics API
        └───────────────────────┘
```

---

## Component Details

### 1. VLM Client (`vlm_client.py`)

The VLM Client handles communication with OpenVINO Model Server for visual inference.

**Features:**
- **Image Preprocessing**: Smart resizing (672px max), JPEG compression (82% quality), contrast enhancement
- **Circuit Breaker**: 5 failures → OPEN, 30s recovery → HALF_OPEN, 2 successes → CLOSED
- **Connection Pooling**: Shared `httpx.AsyncClient` with HTTP/2, 50 max connections
- **Inventory-Aware Prompts**: Includes known menu items for improved accuracy

```python
# Circuit Breaker States
class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests  
    HALF_OPEN = "half_open"  # Testing recovery
```

### 2. Semantic Client (`semantic_client.py`)

Handles fuzzy string matching for item comparison.

**Features:**
- **Similarity Threshold**: Default 0.7 (70% match required)
- **Fallback Matching**: Exact string match when service unavailable
- **Circuit Breaker**: 15s recovery timeout (faster than VLM)
- **Connection Pool**: Shared client with 20 max connections

### 3. Validation Service (`validation_service.py`)

Orchestrates the validation workflow using Strategy pattern.

**Validation Pipeline:**
1. VLM inference → detected items
2. Semantic matching → item correlations
3. Quantity analysis → mismatches
4. Accuracy calculation → final score

```python
# Accuracy Calculation
accuracy = matched_items / max(expected_items, detected_items)
order_complete = (missing == 0) and (quantity_errors == 0) and (extra == 0)
```

### 4. Configuration Manager (`config.py`)

Thread-safe singleton for application configuration.

**Features:**
- Double-checked locking pattern
- Environment variable driven
- Runtime benchmark mode toggle

### 5. API Layer (`api.py`)

FastAPI endpoints with bounded validation cache.

**Features:**
- **BoundedValidationCache**: LRU eviction, 10K max entries
- **Thread-safe service init**: Lock-protected lazy initialization
- **Async metrics collection**: Non-blocking system stats

---

## Data Flow

### Validation Request Processing

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VALIDATION PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. IMAGE PREPROCESSING                                            │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ Raw Image → Auto-Orient → Resize (672px) → Enhance →     │   │
│     │ Sharpen → JPEG Compress (82%) → Base64 Encode            │   │
│     └──────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  2. VLM INFERENCE                                                  │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ Prompt: "Analyze this food plate image..."               │   │
│     │ + Inventory list for context                             │   │
│     │ → OVMS POST /v3/chat/completions                         │   │
│     │ → Parse JSON response for detected items                 │   │
│     └──────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  3. SEMANTIC MATCHING                                              │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ For each expected item:                                  │   │
│     │   Find best match in detected items (similarity > 0.7)   │   │
│     │   Track: matched, missing, extra, quantity mismatches    │   │
│     └──────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  4. RESULT AGGREGATION                                             │
│     ┌──────────────────────────────────────────────────────────┐   │
│     │ {                                                        │   │
│     │   "order_complete": true/false,                          │   │
│     │   "accuracy_score": 0.0-1.0,                             │   │
│     │   "missing_items": [...],                                │   │
│     │   "extra_items": [...],                                  │   │
│     │   "metrics": { latency, tps, utilization }               │   │
│     │ }                                                        │   │
│     └──────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Metrics Collection

```
┌─────────────────────────────────────────────────────────────────────┐
│                        METRICS PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  VLM CLIENT                    METRICS COLLECTOR                   │
│  ┌────────────────┐           ┌────────────────┐                   │
│  │ log_start_time │──────────▶│ Start Timestamp│                   │
│  │ log_end_time   │──────────▶│ End Timestamp  │                   │
│  │ log_custom_event           │ TPS, Tokens    │                   │
│  │   - tps        │──────────▶│ Preprocess Time│                   │
│  │   - tokens     │           │ Items Detected │                   │
│  │   - latency    │           └────────┬───────┘                   │
│  └────────────────┘                    │                           │
│                                        ▼                           │
│                              ┌────────────────┐                    │
│                              │ JSON/CSV Export│                    │
│                              │ results/*.json │                    │
│                              │ results/*.csv  │                    │
│                              └────────────────┘                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Production Features

### Circuit Breaker Pattern

Prevents cascading failures when external services are unhealthy.

```
       ┌─────────────────────────────────────────────────────────────┐
       │                   CIRCUIT BREAKER FSM                       │
       └─────────────────────────────────────────────────────────────┘

                    5 consecutive failures
       ┌────────┐ ───────────────────────────▶ ┌────────┐
       │        │                               │        │
       │ CLOSED │                               │  OPEN  │
       │        │ ◀─────────────────────────── │        │
       └────────┘    2 successes in half_open  └────────┘
            ▲                                       │
            │                                       │
            │        ┌────────────┐                 │
            │        │            │                 │
            └────────│ HALF_OPEN  │◀────────────────┘
      2 successes    │            │    30s timeout
                     └────────────┘ 
                           │
                           │ 1 failure
                           ▼
                       Back to OPEN
```

### Connection Pooling

```python
# VLM Client Pool Configuration
limits = httpx.Limits(
    max_keepalive_connections=20,
    max_connections=50,
    keepalive_expiry=30.0
)
timeout = httpx.Timeout(
    connect=10.0,
    read=300.0,   # Extended for VLM inference
    write=10.0,
    pool=10.0
)
client = httpx.AsyncClient(limits=limits, timeout=timeout, http2=True)
```

### Bounded Cache (LRU)

```python
class BoundedValidationCache:
    """Thread-safe LRU cache with automatic eviction"""
    
    def __init__(self, maxsize: int = 10000):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._lock = threading.Lock()
    
    def __setitem__(self, key, value):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            # Evict oldest when full
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
```

---

## Performance Characteristics

### Latency Breakdown

| Stage | Typical Duration |
|-------|-----------------|
| Image Preprocessing | 50–100 ms |
| VLM Inference | 8–12 s |
| Semantic Matching | 20–50 ms |
| **Total E2E** | **9–15 s** |

Target: < 15 s end-to-end for operational efficiency.

---

---

## System Requirements

### Hardware Requirements

#### Development / Single Station

| Component | Requirement |
|-----------|-------------|
| CPU | 8+ cores |
| RAM | 48 GB min, 64 GB recommended (for first-time model export); 16 GB for inference only |
| GPU | Intel Arc A770 (16 GB) or equivalent Intel GPU |
| Storage | 50 GB SSD |

#### Production

| Component | Requirement |
|-----------|-------------|
| CPU | 16+ cores |
| RAM | 64 GB |
| GPU | Intel Data Center GPU (for concurrent validation) |
| Storage | 200 GB NVMe SSD |

**GPU VRAM guidance:** The Qwen2.5-VL-7B INT8 model requires ~8 GB of VRAM. The default `cache_size=4` reserves an additional 4 GB VRAM for the KV cache. Total VRAM needed ≈ 12 GB — fits in Intel Arc A770 16 GB. Set `export CACHE_SIZE=<N>` before running `setup_models.sh` to adjust (keep model + cache_size ≤ available VRAM).

> **⚠ Model Export RAM Requirement:** The initial `setup_models.sh` run performs INT8 quantization of the Qwen2.5-VL-7B model, which can temporarily require approximately 30–40 GB of peak system RAM (FP16 model ~15 GB + INT8 compressed ~8 GB + calibration buffers). Running this on 32 GB platforms (e.g. Wildcat Lake, Meteor Lake) causes OOM and corrupt model XML files. Use a system with at least 48 GB RAM for export. For inference-only, 16 GB system RAM is sufficient with the default cache_size=4 GB setting.

### Software Requirements

#### Operating System

Ubuntu 22.04 LTS is the validated platform (matches the `python:3.13-slim` base image running on the host GPU driver stack).

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

## Network Requirements

### Port Configuration

| Service | Port | Purpose |
|---------|------|---------|
| Gradio UI | 7861 | Web interface |
| REST API | 8083 | FastAPI endpoints |
| OVMS VLM | 8002 | Model inference (external) |
| Semantic Service | 8081 | Semantic matching (external) |
| Metrics Collector | 8084 | System metrics |

---

## Pre-Deployment Checklist

- [ ] Docker and Docker Compose installed and working
- [ ] Intel GPU drivers installed and GPU visible to Docker
- [ ] Required ports available (7861, 8083, 8002, 8081, 8084)
- [ ] At least 50 GB free disk space
- [ ] **48 GB+ RAM available** (required for `setup_models.sh` model export; 32 GB RAM platforms produce corrupt model files — use 48 GB min, 64 GB recommended)
- [ ] VLM model downloaded (`setup_models.sh` completed)
- [ ] `.env` file created (`make init-env`)
- [ ] Plate images placed in `images/` and `configs/orders.json` updated

---

## Next Steps

- [Get Started](get-started.md) - Set up and run the application
- [API Reference](api-reference.md) - REST endpoint documentation
- [How to Build](how-to-build-from-source.md) - Build from source code
