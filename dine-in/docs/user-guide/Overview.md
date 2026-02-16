# Dine-In Order Accuracy - Overview

Staff-triggered plate validation workflow designed for full-service restaurant expo operations. This solution demonstrates zero-training deployment with vision-language models (VLM), semantic order reconciliation, and latency instrumentation aligned to operational service windows.

## Table of Contents

- [Introduction](#introduction)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [System Architecture](#system-architecture)
- [Component Details](#component-details)
- [Data Flow](#data-flow)
- [Production Features](#production-features)
- [Performance Characteristics](#performance-characteristics)

---

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
| `dinein_app` | `dine-in-dine-in` | 7861, 8083 | Main application (Gradio + FastAPI) |
| `dinein_ovms_vlm` | `openvino/model_server` | 8000 | Vision-Language Model server |
| `dinein_semantic_service` | `semantic-comparison-service` | 8080 | Semantic text matching |
| `metrics-collector` | `metrics-collector` | 9000 | System metrics aggregation |

### Network Topology

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Network: dinein_network              │
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │   dinein_app    │    │ dinein_ovms_vlm │                    │
│  │                 │    │                 │                    │
│  │  - Gradio:7861  │───▶│  - gRPC: 8000   │                    │
│  │  - API:8083     │    │  - REST: 8000   │                    │
│  │                 │    │                 │                    │
│  └────────┬────────┘    └─────────────────┘                    │
│           │                                                     │
│           │             ┌─────────────────┐                    │
│           │             │ semantic_service│                    │
│           └────────────▶│                 │                    │
│                         │  - REST: 8080   │                    │
│                         │                 │                    │
│                         └─────────────────┘                    │
│                                                                 │
│  ┌─────────────────┐                                           │
│  │metrics-collector│                                           │
│  │  - REST: 9000   │◀────── Prometheus-style metrics           │
│  └─────────────────┘                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼ Host Network
        ┌───────────────────────┐
        │   localhost:7861      │  ← Gradio UI
        │   localhost:8083      │  ← REST API
        │   localhost:8083/docs │  ← Swagger Docs
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

| Stage | Typical Duration | Notes |
|-------|-----------------|-------|
| Image Preprocessing | 50-100ms | Resize, enhance, compress |
| VLM Inference | 8-12s | Qwen2.5-VL-7B on Intel GPU |
| Semantic Matching | 20-50ms | Per item comparison |
| **Total E2E** | **9-15s** | Target: < 15s |

### Resource Utilization

| Resource | Typical Usage | Notes |
|----------|--------------|-------|
| GPU | 80-100% | During VLM inference |
| CPU | 20-30% | Preprocessing + orchestration |
| Memory | 70-80% | Model weights + cache |

### Stream Density Results

```
Target Latency: 15000ms (15s)

┌──────────┬────────────┬─────────────┬────────┐
│ Density  │ Avg Latency│ P95 Latency │ Status │
├──────────┼────────────┼─────────────┼────────┤
│ 1 image  │ 11,726ms   │ 11,726ms    │ PASS   │
│ 2 images │ 14,808ms   │ 16,743ms    │ PASS   │
│ 3 images │ 19,509ms   │ 28,952ms    │ FAIL   │
└──────────┴────────────┴─────────────┴────────┘

Maximum sustainable density: 2 concurrent images
```

---

## Next Steps

- [Get Started](get-started.md) - Set up and run the application
- [System Requirements](system-requirements.md) - Hardware/software prerequisites
- [API Reference](api-reference.md) - REST endpoint documentation
- [How to Build](how-to-build-from-source.md) - Build from source code
