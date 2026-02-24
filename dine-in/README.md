# Order Accuracy Dine-In

**Image-based Order Validation for Restaurant Dining Applications**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.6%2B-blue.svg)](https://docs.openvino.ai)

---

## Quick Start

### Prerequisites

- Docker 24.0+ with Compose V2
- NVIDIA GPU with 8GB+ VRAM (or Intel GPU)
- 32GB+ RAM recommended
- Intel Xeon or equivalent CPU

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

### 1. Setup OVMS Model (First Time Only)

The VLM model must be exported before running the application:

```bash
cd order-accuracy/ovms-service
./setup_models.sh    # Export model (30-60 min first time)
```

This step:
- Downloads Qwen2.5-VL-7B-Instruct from HuggingFace (~7GB)
- Converts to OpenVINO format with INT8 quantization
- Creates model files in `ovms-service/models/`

> **Note**: This only needs to be done once. The model files are shared between dine-in and take-away applications.

### 2. Configure Environment

```bash
cd ../dine-in
cp .env.example .env
# Edit .env if needed (defaults work for most setups)
```

### 3. Build and Start

```bash
make build
make up
```

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
  - [Overview](docs/user-guide/Overview.md): A high-level introduction.
  - [Overview Architecture](./docs/user-guide/Overview.md#how-it-works): Highlevel architecture.

- **Getting Started**
  - [Get Started](docs/user-guide/get-started.md): Step-by-step guide to get started with the sample application.
  - [System Requirements](docs/user-guide/system-requirements.md): Hardware and software requirements for running the sample application.
  - [How to Use the Application](./docs/user-guide/how-to-use-application.md): Explore the application's features and verify its functionality.

- **Deployment**
  - [How to Build from Source](docs/user-guide/how-to-build-from-source.md): Instructions for building from source code.
  - [How to Build using Helm](docs/user-guide/deploy-with-helm.md): Instructions for building using helm.


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

```bash
make benchmark-single    # Quick single image test
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