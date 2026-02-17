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
| GPU | Intel Arc A770 (8GB) | NVIDIA RTX 3080+ / Intel Arc |
| Storage | 50GB SSD | 200GB NVMe |
| Network | 1 Gbps | 10 Gbps |

### Software Requirements

| Software | Version | Purpose |
|----------|---------|---------|
| Docker | 24.0+ | Container runtime |
| Docker Compose | V2+ | Service orchestration |
| NVIDIA Driver | 535+ | GPU support (if NVIDIA) |
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

# GPU availability (NVIDIA)
nvidia-smi
# OR for Intel
clinfo | head -20
```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
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

### Step 3: Download VLM Model

The system uses Qwen2.5-VL-7B-Instruct optimized for OpenVINO.

```bash
# Using the model download script
./scripts/model-downloader.sh

# Or manually download
mkdir -p models/vlm
# Download from Hugging Face or Intel Model Zoo
```

### Step 4: Create Environment File

```bash
# Copy the example environment file
cp .env.example .env

# Or use make target
make init-env

# Edit configuration
nano .env
```

### Step 5: Initialize MinIO Storage

```bash
# Create storage directories
mkdir -p storage/videos storage/results minio-data
```

### Step 6: Build Benchmark Tools (Optional)

If you plan to run performance benchmarks:

```bash
# Build the benchmark Docker image locally
make build-benchmark

# Or fetch from registry (if available)
make fetch-benchmark
```

---

## Configuration

### Basic Configuration (.env)

```bash
# =============================================================================
# Service Mode
# =============================================================================
SERVICE_MODE=single          # 'single' for development, 'parallel' for production
WORKERS=0                    # Number of station workers (0 for single mode)
SCALING_MODE=fixed           # 'fixed' or 'auto'

# =============================================================================
# VLM Backend
# =============================================================================
VLM_BACKEND=ovms
OVMS_ENDPOINT=http://ovms-vlm:8000
OVMS_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
OPENVINO_DEVICE=GPU          # 'GPU', 'CPU', or 'AUTO'

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
TARGET_LATENCY_MS=15000      # Target latency threshold (ms)
LATENCY_METRIC=avg           # 'avg' or 'p95'
WORKER_INCREMENT=1           # Workers added per iteration
INIT_DURATION=120            # Init wait time (seconds)
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

### Single Worker Mode (Recommended for First Run)

```bash
# Build Docker images
make build

# Start all services
make up

# View logs
make logs
```

### Parallel Worker Mode

```bash
# Build Docker images
make build

# Start with 4 station workers
make up-parallel WORKERS=4

# View logs
make logs
```

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
3. Enter expected order items:
   ```
   2x Burger
   1x Fries (Large)
   1x Soda
   ```
4. Click "Validate Order"
5. View results showing matched, missing, and extra items

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

# Check GPU availability
nvidia-smi  # or clinfo for Intel
```

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
export OPENVINO_DEVICE=CPU

# Restart services
make down && make up
```

### GPU Not Detected

**Symptom**: `No GPU devices found`

**Solution**:
```bash
# For NVIDIA
nvidia-smi
sudo systemctl restart docker

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
make benchmark-oa-density

# Development
make shell                  # Shell into container
make test-api              # Test endpoints
```
