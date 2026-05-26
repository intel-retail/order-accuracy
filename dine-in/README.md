# Order Accuracy Dine-In

**Image-based Order Validation for Restaurant Dining Applications**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.0-blue.svg)](https://docs.openvino.ai)

---

## Quick Start

### Prerequisites

- Docker 24.0+ with Compose V2
- Intel GPU
- 16 GB RAM minimum (64 GB recommended for production)
- Intel Xeon or equivalent CPU

> **ℹ iGPU / low-RAM systems:** 16 GB RAM is sufficient for **inference**. For first-time model export (`setup_models.sh`), a higher-memory host (48–64 GB) is recommended — export the models there and copy the `ovms-service/models/` directory to your 16 GB system. If exporting on 16 GB, set `export CACHE_SIZE=2` first to reduce KV cache to 2 GB (default is 4 GB). On iGPU platforms the KV cache uses system RAM. See [ovms-service/README.md](../ovms-service/README.md#tuning-the-kv-cache-size) for details.

### Setup Test Data (Required)

Before running the application, you must prepare your test data:

1. **Add Images**: Place your food tray images in the `images/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Images should clearly show the food items on the tray

2. **Update Orders**: Edit `configs/orders.json` with your test orders
   - Each order should have an `order_id` and list of `items`
   - Order IDs should match your image filenames

3. **Update Inventory**: Edit `configs/inventory.json` to match your menu items
   - Define all possible food items that can appear in orders
   - Include item names, categories, and any relevant metadata

> **Note**: The `images/` folder does not contain sample images by default. You must add your own images before testing.

### 1. Configure Environment

```bash
cd order-accuracy/dine-in
make init-env
# Edit .env if needed (defaults work for most setups)

# Initialize git submodules (for benchmark tools)
make update-submodules
```

### 2. Setup OVMS Model (First Time Only)

The VLM model must be exported before running. The script reads `dine-in/.env`, so complete step 1 first.

```bash
cd ../ovms-service
./setup_models.sh --app dine-in    # Downloads and exports model (~30-60 min first time)
cd ../dine-in
```

This step:
- Downloads Qwen2.5-VL-7B-Instruct from HuggingFace (~7 GB)
- Converts to OpenVINO INT8 format

> **Note**: Only needed once. Model files are shared between dine-in and take-away.

### 3. Build and Start

**Option A: Using Registry Images (default)**
```bash
make build && make up
```

**Option B: Build Locally from Source**
```bash
make up REGISTRY=false
```

| Image | Tag |
|-------|-----|
| `intel/order-accuracy-dine-in` | `2026.0.0` |

### 4. Access Services

| Service | URL | Purpose |
|---------|-----|--------|
| Gradio UI | http://localhost:7861 | Interactive order validation |
| Order Accuracy API | http://localhost:8083 | REST API endpoints |
| API Docs | http://localhost:8083/docs | Swagger/OpenAPI documentation |
| OVMS VLM | http://localhost:8002 | VLM model server |

---

## Documentation

- **Overview**
  - [System Architecture & Requirements](docs/user-guide/system-architecture-and-requirements.md): Architecture, component details, hardware/software requirements, and pre-deployment checklist.

- **Getting Started**
  - [Get Started](docs/user-guide/get-started.md): Step-by-step guide to get started with the sample application.
  - [How to Use the Application](./docs/user-guide/how-to-use-application.md): Explore the application's features and verify its functionality.

- **Deployment**
  - [How to Build from Source](docs/user-guide/how-to-build-from-source.md): Instructions for building from source code.


- **API Reference**
  - [API Reference](docs/user-guide/api-reference.md): Comprehensive reference for the available REST API endpoints.

- **Release Notes**
  - [Release Notes](docs/user-guide/release-notes.md): Information on the latest updates, improvements, and bug fixes.

---

## Benchmarking

### Prerequisites

Initialize the performance-tools submodule before running benchmarks:

```bash
make update-submodules
```

### Quick Test

> **Prerequisite:** Services must be running. Start them first with `make up`.

```bash
make benchmark-single IMAGE_ID=MCD-1001    # Quick single image test
```

### Full Benchmark

```bash
make benchmark           # Run Order Accuracy benchmark
```

> **Note**: `make benchmark` uses Docker profiles to start worker containers. Both the `dine-in` app and `dinein-worker` services use the **same Docker image** (built from the same Dockerfile). The worker is simply the same container running `worker.py` instead of the UI.

### Stream Density Test

```bash
make benchmark-density   # Find max concurrent validations
```

> **Note**: `make benchmark-density` runs a Python script locally that sends concurrent HTTP requests to the running `dine-in` API. No separate worker containers are needed for this mode.

### Metrics Processing

```bash
make consolidate-metrics # Consolidate results to CSV
make plot-metrics        # Generate visualization plots
```

See [Get Started](docs/user-guide/get-started.md) for detailed benchmark configuration options.

---

## Cleanup

```bash
make down           # Stop all services
make clean          # Stop and remove volumes
make clean-images   # Remove dangling Docker images
make clean-all      # Remove all unused Docker resources
```
