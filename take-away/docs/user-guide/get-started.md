# Getting Started with Take-Away Order Accuracy

This guide walks you through the installation, configuration, and first-run of the Take-Away Order Accuracy system.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Starting the Services](#starting-the-services)
5. [Verifying Installation](#verifying-installation)
6. [First Order Validation](#first-order-validation)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| CPU | Intel Xeon 8 cores | Intel Xeon 16+ cores |
| RAM | 16GB | 32GB+ |
| GPU | Intel Arc A770 (8GB) | Intel Arc |
| Storage | 50GB SSD | 200GB NVMe |
| Network | 1 Gbps | 10 Gbps |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | V2+ | Service orchestration |
| Intel GPU Driver | Latest | GPU support (if Intel) |
| Python | 3.10+ | Local development (optional) |

### Verify Prerequisites

```bash
# Docker version
docker --version
# Expected: Docker version 24.0.x or higher

# Docker Compose version
docker compose version
# Expected: Docker Compose version v2.x.x

# OR for Intel
clinfo | head -20
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/intel-retail/order-accuracy.git
cd order-accuracy/take-away
```

### Step 2: Initialize Git Submodules

The performance-tools repository is included as a git submodule for benchmarking:

```bash
# Initialize and update submodules
make update-submodules

# Or manually
cd ..
git submodule update --init --recursive
cd take-away
```

### Step 3: Setup OVMS Models (First Time Only)

The VLM model and EasyOCR models must be downloaded before running the application:

> **Note — `TARGET_DEVICE`**: To change the inference device mode, set `TARGET_DEVICE` in your `.env` file to `GPU`, `CPU`, or `AUTO`. After changing the device, re-run the setup script to update the model config:
> ```bash
> cd ../ovms-service && ./setup_models.sh --app take-away
> ```
> You can also pass the device explicitly: `./setup_models.sh --device CPU`


```bash
cd ../ovms-service
./setup_models.sh
```

This script:
- Downloads Qwen2.5-VL-7B-Instruct-ov-int8 from HuggingFace
- Downloads and quantizes YOLOv11 model (INT8 OpenVINO)
- Downloads EasyOCR detection and recognition models
- Generates `graph.pbtxt` from `config.json` graph_options

> **Note**: This only needs to be done once. The model files are shared across applications.

### Step 4: Create Environment File

```bash
# Create .env from template (backs up existing .env if present)
make init-env
```

### Step 5: Build and Start

```bash
# Pull images from registry (default)
make build
make up

# OR build locally from source
make build REGISTRY=false
make up
```

> **Note**: `make build` pulls pre-built images from Docker Hub by default. Use `REGISTRY=false` to build from source.

## RTSP Stream for Live Verification

To start a standalone RTSP streamer that loops video files for real-time order verification, run:

```bash
WORKERS=1 docker compose --profile parallel up -d --no-deps rtsp-streamer
```

> **Prerequisite:** Place a test video at `storage/videos/test.mp4` before starting the streamer. You can run `make download-sample-video` to get one.

Once the streamer is running, the following RTSP URL becomes available:

| Access From | URL |
|-------------|-----|
| Host machine / Gradio UI | `rtsp://localhost:8554/station_1` |
| Other containers (internal) | `rtsp://rtsp-streamer:8554/station_1` |

For multiple stations, increase `WORKERS` (e.g., `WORKERS=3`) to create `station_1`, `station_2`, and `station_3` streams.


---

## Configuration

### Basic Configuration (.env)

```bash
# =============================================================================
# VLM Backend
# =============================================================================
VLM_BACKEND=ovms
OVMS_ENDPOINT=http://ovms-vlm:8000
OVMS_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
TARGET_DEVICE=GPU            # 'GPU', 'CPU', or 'AUTO'

# =============================================================================
# Semantic Service
# =============================================================================
SEMANTIC_VLM_BACKEND=ovms
DEFAULT_MATCHING_STRATEGY=hybrid   # 'exact', 'semantic', or 'hybrid'
SIMILARITY_THRESHOLD=0.85
OVMS_TIMEOUT=60

# =============================================================================
# MinIO Storage
# =============================================================================
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=minio:9000
```

### Benchmarking Configuration (.env)

For stream density and performance testing, configure these variables:

```bash
# =============================================================================
# Benchmarking (Stream Density Testing)
# =============================================================================
TARGET_LATENCY_MS=25000      # Target latency threshold (ms)
LATENCY_METRIC=avg           # 'avg' or 'p95'
WORKER_INCREMENT=1           # Workers added per iteration
INIT_DURATION=10             # Init wait time (seconds)
MIN_TRANSACTIONS=3           # Min transactions before measuring
MAX_ITERATIONS=50            # Max scaling iterations
MAX_WAIT_SEC=600             # Max wait per iteration (seconds)
RESULTS_DIR=./results        # Results output directory
```

> **Note:** CLI arguments override environment variables. See [Benchmarking Guide](benchmarking-guide.md) for detailed usage.

### Configuration Validation

```bash
# Verify configuration
make check-env

# Show current configuration
make show-config
```

---

## Starting the Services

### Parallel Worker Mode (Production)

For multi-station processing, use parallel mode:

```bash
make up-parallel WORKERS=4
```

# View logs
make logs
```

### Stream Density Benchmark

To measure the maximum number of parallel workers the system can sustain under a latency target:

```bash
make benchmark-stream-density
```

This automatically scales workers up, measuring end-to-end latency at each level, and stops when the target latency (default 25s) is exceeded. Results are saved to `./results/`.

Override defaults via environment or CLI:

```bash
make benchmark-stream-density \
  BENCHMARK_TARGET_LATENCY_MS=30000 \
  BENCHMARK_INIT_DURATION=15
```


### Metrics Processing

After running benchmarks, consolidate and visualize metrics:

```bash
# Consolidate metrics from multiple runs
make consolidate-metrics

# Generate plots from benchmark metrics
make plot-metrics
```


See [Benchmarking Guide](benchmarking-guide.md) for full options.

### Service Status

```bash
# Check service status
make status

# Expected output:
# oa_service        Up 2 minutes   0.0.0.0:8000->8000/tcp
# oa_ovms_vlm       Up 2 minutes   0.0.0.0:8001->8001/tcp
# oa_gradio         Up 2 minutes   0.0.0.0:7860->7860/tcp
# oa_minio          Up 2 minutes   0.0.0.0:9000-9001->9000-9001/tcp
# oa_semantic_service  Up 2 minutes   0.0.0.0:8080->8080/tcp
```

---

## Verifying Installation

### Step 1: Health Check

```bash
# Test API health
make test-api

# Or directly
curl http://localhost:8000/health
# Expected: {"status": "healthy", "service": "order-accuracy"}
```

### Step 2: Verify OVMS Model

```bash
# Check OVMS configuration
curl http://localhost:8001/v1/config | jq .

# Expected: Model configuration with Qwen2.5-VL model details
```

### Step 3: Verify MinIO

Access MinIO Console at http://localhost:9001

- Username: `minioadmin`
- Password: `minioadmin`

Verify buckets:
- `frames` - Raw captured frames
- `selected` - YOLO-selected frames
- `results` - Validation results

### Step 4: Access Gradio UI

Open http://localhost:7860 in your browser.

You should see the Order Accuracy web interface with:
- Video upload section
- RTSP stream viewer
- Order entry form
- Results display

---

## First Order Validation

### Option 1: Via Gradio UI

1. Open http://localhost:7860
2. Upload a test video or enter RTSP URL
3. Click "Upload and Start Processing"
4. View results showing matched, missing, and extra items

### Option 2: Via REST API

```bash
# Upload video and validate
curl -X POST http://localhost:8000/upload-video \
  -F "file=@storage/videos/test.mp4" \
  -F "video_id=test_001"

# Response
{
  "status": "success",
  "video_id": "test_001",
  "message": "Video uploaded and queued for processing"
}

# Check results
curl http://localhost:8000/results/test_001
```

### Option 3: Using Make Target

> **Important**: Before running benchmarks, ensure a test video file is present at `storage/videos/test.mp4`. You can download a sample video using:
> ```bash
> make download-sample-video
> ```

```bash
# Run benchmark with test video
make benchmark
```

---

## Troubleshooting

### OVMS Not Starting

**Symptom**: `oa_ovms_vlm` container exits immediately

**Solution**:
```bash
# Check logs
docker logs oa_ovms_vlm

# Verify model path exists
ls -la models/vlm/


### Connection Refused to OVMS

**Symptom**: `Connection refused` errors to port 8001

**Solution**:
```bash
# Wait for OVMS to fully load model (can take 2-5 minutes)
docker logs -f oa_ovms_vlm | grep "Serving"

# Check network
docker network inspect order-accuracy-net
```

### MinIO Bucket Errors

**Symptom**: `Bucket does not exist` errors

**Solution**:
```bash
# Recreate MinIO with fresh volumes
make down
docker volume rm take-away_minio-data
make up
```

### Out of Memory

**Symptom**: Services crash with OOM errors

**Solution**:
```bash
# Reduce batch size
export VLM_BATCH_SIZE=1

# Use CPU instead of GPU (slower but less memory)
export TARGET_DEVICE=CPU

# Restart services
make down && make up
```

### GPU Not Detected

**Symptom**: `No GPU devices found`

**Solution**:
```bash

# For Intel
sudo usermod -aG render $USER
# Logout and login again
```

---

## Next Steps

After successful installation and first validation:

1. **Configure for Production**: See [System Requirements](system-requirements.md)
2. **Learn the API**: See [API Reference](api-reference.md)
3. **Run Benchmarks**: See [Benchmarking Guide](benchmarking-guide.md)
4. **Customize Settings**: See [How to Use Application](how-to-use-application.md)

---

## Quick Reference Commands

```bash
# Start services
make up                     # Single mode
make up-parallel WORKERS=4  # Parallel mode

# Check status
make status
make logs

# Stop services
make down

# Clean everything
make clean

# Run benchmark
make benchmark
make benchmark-stream-density

# Development
make shell                  # Shell into container
make test-api              # Test endpoints
```
