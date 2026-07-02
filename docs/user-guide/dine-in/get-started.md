# Get Started

This guide walks you through installation, configuration, and first run of the Dine-In Order Accuracy system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Verifying Installation](#verifying-installation)
4. [First Order Validation](#first-order-validation)

## Prerequisites

- Docker 24.0+ with Compose V2
- Intel GPU with drivers installed
- 16 GB RAM minimum (64 GB recommended for production)
- 50 GB free disk space

> **Notes:**
> **KV Cache on iGPU / low-RAM systems:** 16 GB RAM is sufficient for **inference**.
> For first-time model export, a higher-memory host (48–64 GB) is recommended.
> On iGPU platforms, the KV cache is allocated from **system RAM** — set `export CACHE_SIZE=2`
> before running `setup_models.sh` to reduce KV cache to 2 GB (default is 4 GB).
> See [ovms-service/README.md — Tuning the KV Cache Size](https://github.com/intel-retail/order-accuracy/blob/main/ovms-service/README.md#tuning-the-kv-cache-size) for a full per-platform guide.

```bash
docker --version
docker compose version
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/intel-retail/order-accuracy.git
cd order-accuracy/dine-in
```

### Step 2: Configure the Environment

```bash
# Create .env from template
make init-env
# Edit .env if needed — defaults work for most setups

# Initialize git submodules (for benchmark tools)
make update-submodules
```

### Step 3: Setup OVMS Model (First Time Only)

The setup script reads model configuration (device, precision, model name) from `dine-in/.env` (created in Step 2), so **complete Step 2 before running this step**.

```bash
cd ../ovms-service
./setup_models.sh --app dine-in    # Downloads and exports model (~30-60 min first time)
cd ../dine-in
```

> **Note:** Only needed once. Model files are shared between Dine-In and Take-Away.

This downloads Qwen2.5-VL-7B-Instruct (~7 GB) and converts it to OpenVINO™ INT8 format. This is only needed once — the model files are shared with Take-Away.

### Step 4: Prepare Test Data

Before running the application, you must prepare your test data:

1. **Add Images**: Place your food tray images in the `images/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Images should clearly show the food items on the tray

2. **Update Orders**: Edit `configs/orders.json` with your test orders
   - Each order should have an `order_id` and list of `items`
   - `order_id` should match your `image_id`

3. **Update Inventory**: Edit `configs/inventory.json` to match your menu items
   - Define all possible food items that can appear in orders
   - Include item names, categories, and any relevant metadata

### Step 5: Build and Start

```bash
# Pull images from registry (default)
make build && make up

# OR build locally from source
make build REGISTRY=false && make up
```

This starts 4 containers:

| Container                 | Ports      | Purpose                 |
| ------------------------- | ---------- | ----------------------- |
| `dinein_app`              | 7861, 8083 | Gradio UI + FastAPI     |
| `dinein_ovms_vlm`         | 8002       | VLM model server (OVMS) |
| `dinein_semantic_service` | 8081, 9091 | Semantic matching       |
| `metrics-collector`       | 8084       | System metrics          |

---

## Verifying Installation

```bash
# API health check
make test-api

# Or directly
curl http://localhost:8083/health

# Check OVMS model
curl http://localhost:8002/v1/config | jq .
```

Open `http://localhost:7861` for the Gradio UI, or `http://localhost:8083/docs` for the REST API docs.

---

## First Order Validation

### Via Gradio UI

1. Open `http://localhost:7861`
2. Select a scenario from the dropdown
3. Review the order manifest
4. Click **"Validate Plate"**
5. View accuracy score, matched/missing/extra items, and performance metrics

### Via REST API

The bundled `MCD-1001.png` image shows **Filet-O-Fish** and **Cheesy Fries** on the tray.
Two test scenarios are provided:

**Negative test case** — order does not match tray (demonstrates mismatch detection):

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={"items":[{"name":"Cheeseburger","quantity":1},{"name":"French Fries","quantity":1}]}'
# Expected: order_complete=false, accuracy_score=0.0
```

**Positive test case** — order matches tray (demonstrates successful validation):

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={"items":[{"name":"Filet-O-Fish","quantity":1},{"name":"Cheesy Fries","quantity":1}]}'
# Expected: order_complete=true, accuracy_score=1.0
```

### Via Make

```bash
# Services must be running first
make benchmark-single IMAGE_ID=MCD-1001
```

---

## Changing Inference Device

To switch between GPU and CPU, update `TARGET_DEVICE` in `.env` and re-run model setup:

```bash
# In .env
TARGET_DEVICE=CPU

cd ../ovms-service && ./setup_models.sh --app dine-in && cd ../dine-in
make down && make up
```

## Quick Reference

```bash
make up                              # Start services (registry image)
make up REGISTRY=false               # Start with locally built image
make down                            # Stop services
make logs                            # View logs
make test-api                        # Health check
make benchmark-single IMAGE_ID=...   # Quick test
make benchmark                       # Full benchmark
make benchmark-stream-density        # Stream density test
make clean                           # Stop and remove volumes
make clean-images                    # Remove dangling Docker images
make clean-all                       # Remove all unused Docker resources
make help                            # All commands
```

## Next Steps

- [System Requirements](./get-started/system-requirements.md) - Check the detailed requirements
- [Build from Source](./get-started/build-from-source.md) - Build from source
- [How It Works](./how-it-works.md) - Learn about the architecture
- [How to Use](./how-to-use.md) - Customize settings
- [Benchmarking Guide](./di-benchmarking.md) - Run benchmarks
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
