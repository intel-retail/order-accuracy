# Get Started

This guide walks you through the installation, configuration, and first-run of the Take-Away Order Accuracy system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Starting the Services](#starting-the-services)
5. [Verifying Installation](#verifying-installation)
6. [First Order Validation](#first-order-validation)

## Prerequisites

For detailed hardware and software requirements, see the [System Requirements](./get-started/system-requirements.md) guide.

### Hardware Requirements

| Component | Minimum              | Recommended          |
| --------- | -------------------- | -------------------- |
| CPU       | Intel Xeon 8 cores   | Intel Xeon 16+ cores |
| RAM       | 16GB                 | 64GB+                |
| GPU       | Intel Arc A770 (8GB) | Intel Arc            |
| Storage   | 50GB SSD             | 200GB NVMe           |

> **Note:** **RAM note** 16 GB system RAM is sufficient for **inference**. For first-time model
> export (`setup_models.sh`), a higher-memory host (48–64 GB recommended) avoids potential OOM
> — export there and copy `ovms-service/models/` to the target system. 64 GB+ is recommended
> for production or multi-station deployments.

> **KV Cache on iGPU / low-RAM systems:** On iGPU platforms the KV cache is allocated from
> **system RAM**. Set `export CACHE_SIZE=2` before running `setup_models.sh` to reduce KV cache
> to 2 GB (default is 4 GB). See [ovms-service/README.md — Tuning the KV Cache Size](https://github.com/intel-retail/order-accuracy/blob/main/ovms-service/README.md#tuning-the-kv-cache-size) for a full per-platform guide.

### Software Requirements

| Software         | Version | Purpose               |
| ---------------- | ------- | --------------------- |
| Docker           | 24.0+   | Container runtime     |
| Docker Compose   | V2+     | Service orchestration |
| Intel GPU Driver | Latest  | GPU support           |

### Verify Prerequisites

```bash
docker --version          # Docker version 24.0.x or higher
docker compose version    # Docker Compose version v2.x.x
```

## Installation

### Step 1: Clone the Repository

```bash
git clone -b <release-or-tag> --single-branch https://github.com/intel-retail/order-accuracy.git
cd order-accuracy/take-away
```

### Step 2: Configure the Environment

```bash
# Create .env from template
make init-env
# Edit .env if needed — defaults work for most setups

# Initialize git submodules (for benchmark tools)
make update-submodules
```

Edit the generated `.env` file as needed. See [Configuration](#configuration) below.

### Step 3: Setup OVMS Model (First Time Only)

Set `TARGET_DEVICE` in your `.env` **before** running this step. The script reads that value to export the model in the correct format for the target device.

```bash
cd ../ovms-service
./setup_models.sh --app take-away    # Downloads and exports model (~30-60 min first time)
cd ../take-away
```

This downloads and exports:

- MiniCPM-V-4_5 INT4 (OpenVINO™ format)
- YOLOv11 model (INT8 OpenVINO™)
- EasyOCR detection and recognition models

> **Note:** Re-run this step any time you change `TARGET_DEVICE` in `.env`.

### Step 4: Build and Start

```bash
# Pull images from registry (default)
make build
make up

# OR build locally from source
make build REGISTRY=false
make up
```

---

## Configuration

### Environment File (.env)

```bash
# =============================================================================
# VLM Backend
# =============================================================================
VLM_BACKEND=ovms
OVMS_ENDPOINT=http://ovms-vlm:8000
OVMS_MODEL_NAME=openbmb/MiniCPM-V-4_5-int4
VLM_PRECISION=int4           # must match the -int4 suffix in OVMS_MODEL_NAME
TARGET_DEVICE=GPU            # 'GPU' or 'CPU' — also set OPENVINO_DEVICE to match

# =============================================================================
# VLM Image / Decoding Options
# =============================================================================
VLM_IMAGE_MAX_SIZE=448       # frames are fitted into a 448x448 square canvas
VLM_ENABLE_THINKING=false    # keep MiniCPM's <think> reasoning disabled

# =============================================================================
# Inference Device (must match TARGET_DEVICE)
# =============================================================================
OPENVINO_DEVICE=GPU          # Used by benchmark targets

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
MINIO_ROOT_USER=<your-minio-username>
MINIO_ROOT_PASSWORD=<your-minio-password>
MINIO_ENDPOINT=minio:9000
```

> **Changing the inference device:** Set both `TARGET_DEVICE` and `OPENVINO_DEVICE` to the same value (`GPU` or `CPU`), then re-run `./setup_models.sh --app take-away` to re-export the model for that device.

> **`VLM_ENABLE_THINKING`:** MiniCPM-V-4.5 is a hybrid reasoning model. Leave this
> `false`. When enabled, the model spends the whole token budget on a `<think>`
> reasoning block and the detected-item list is truncated, which shows up as
> missing items and ~4x higher latency.

> **`VLM_IMAGE_MAX_SIZE`:** Each selected frame is fitted into a square canvas of
> this size (aspect ratio preserved, white padding), matching the dine-in
> pipeline. Larger values increase detail but also prompt tokens and latency.

### Validate Configuration

```bash
make check-env    # Validate required variables
make show-config  # Show current configuration
```

---

## Starting the Services

### Single Mode (default)

```bash
make up
```

### Parallel Worker Mode (multi-station)

```bash
make up-parallel WORKERS=4
```

### RTSP Streamer (live stream testing)

To feed a looping video as a live RTSP stream:

> **Prerequisite:** Place a test video at `storage/videos/test.mp4` first, or run `make download-sample-video`.

```bash
WORKERS=1 docker compose --profile parallel up -d --no-deps rtsp-streamer
```

| Access From      | URL                                   |
| ---------------- | ------------------------------------- |
| Host machine     | `rtsp://localhost:8554/station_1`     |
| Other containers | `rtsp://rtsp-streamer:8554/station_1` |

Increase `WORKERS` for multiple stations (`station_1`, `station_2`, etc.).

### View Logs

```bash
make logs
```

---

## Verifying Installation

### Health Check

```bash
make test-api
```

This checks both the order accuracy API (`localhost:8000`) and OVMS (`localhost:8001`).

### Service URLs

| Service            | URL                     |
| ------------------ | ----------------------- |
| Order Accuracy API | `http://localhost:8000` |
| OVMS VLM           | `http://localhost:8001` |
| Gradio UI          | `http://localhost:7860` |
| MinIO Console      | `http://localhost:9001` |
| Semantic Service   | `http://localhost:8080` |

### Service Status

```bash
make status
```

---

## First Order Validation

### Via Gradio UI

1. Open `http://localhost:7860`
2. Upload a test video or enter an RTSP URL
3. Click "Upload and Start Processing"
4. View results showing matched, missing, and extra items

### Via REST API

```bash
# Upload video and validate
curl -X POST http://localhost:8000/upload-video \
  -F "file=@storage/videos/test.mp4" \
  -F "video_id=test_001"

# Check results
curl http://localhost:8000/results/test_001
```

### Via Make Target

> **Prerequisite:** Ensure `storage/videos/test.mp4` exists. Run `make download-sample-video` if needed.

```bash
make benchmark
```

---

## Quick Reference

```bash
make update-submodules      # Init submodules (first time)
make download-sample-video  # Download sample video
make init-env               # Create .env from template
make build && make up       # Build images and start services
make status                 # Check service status
make logs                   # View logs
make test-api               # Test API health
make down                   # Stop services
make clean                  # Stop and remove volumes
make benchmark              # Run fixed-workers benchmark
make benchmark-stream-density  # Run stream density benchmark
```

## Next Steps

- [System Requirements](./get-started/system-requirements.md) - Check the detailed requirements
- [Build from Source](./get-started/build-from-source.md) - Build from source
- [How It Works](./how-it-works.md) - Learn about the architecture
- [How to Use](./how-to-use.md) - Customize settings
- [Benchmarking Guide](./ta-benchmarking.md) - Run benchmarks
- [API Reference](./api-reference.md) - Learn the API
- [Troubleshooting](./troubleshooting.md) - Resolve common issues
- [Release Notes](./release-notes.md) - Read about updates and improvements

<!--hide_directive
:::{toctree}
:hidden:

./get-started/system-requirements.md
./get-started/build-from-source.md

:::
hide_directive-->
