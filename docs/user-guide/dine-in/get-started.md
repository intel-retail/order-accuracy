# Get Started

This guide walks you through installation, configuration, and first run of the Dine-In Order Accuracy system.

---

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

---

## Step 1: Configure Environment

```bash
cd order-accuracy/dine-in

# Create .env from template
make init-env
# Edit .env if needed — defaults work for most setups

# Initialize git submodules (for benchmark tools)
make update-submodules
```

## Step 2: Setup OVMS Model (First Time Only)

The setup script reads model configuration (device, precision, model name) from `dine-in/.env` (created in Step 1), so **complete Step 1 before running this step**.

```bash
cd ../ovms-service
./setup_models.sh --app dine-in    # Downloads and exports model (~30-60 min first time)
cd ../dine-in
```

> **Note:** If you previously ran setup for take-away, the model files are already shared and this step will detect them automatically — no re-download needed.

This downloads Qwen2.5-VL-7B-Instruct (~7 GB) and converts it to OpenVINO INT8 format. This is only needed once — the model files are shared with Take-Away.

## Step 3: Prepare Test Data

The `images/` folder does not contain sample images. Add your own before testing:

1. Place plate images in `images/` (`.jpg`, `.jpeg`, or `.png`)
2. Edit `configs/orders.json` — add entries with `image_id` matching your filenames
3. Edit `configs/inventory.json` — define all possible menu items

## Step 4: Build and Start

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

## First Validation

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

## Benchmarking

### Single Image Test

```bash
make benchmark-single IMAGE_ID=MCD-1001
```

### Full Benchmark

```bash
make benchmark
```

Key variables:

| Variable                      | Default | Description            |
| ----------------------------- | ------- | ---------------------- |
| `BENCHMARK_WORKERS`           | 1       | Concurrent workers     |
| `BENCHMARK_DURATION`          | 180     | Duration (seconds)     |
| `BENCHMARK_TARGET_LATENCY_MS` | 25000   | Latency threshold (ms) |
| `TARGET_DEVICE`               | GPU     | Device: CPU, GPU       |

### Stream Density Test

Finds the maximum number of concurrent image validations within the latency target:

```bash
make benchmark-stream-density

# With overrides
make benchmark-stream-density BENCHMARK_TARGET_LATENCY_MS=20000 BENCHMARK_INIT_DURATION=30
```

### Metrics Processing

```bash
make consolidate-metrics   # Consolidate results to CSV
make plot-metrics          # Generate visualisation plots
```

---

## Changing Inference Device

To switch between GPU and CPU, update `TARGET_DEVICE` in `.env` and re-run model setup:

```bash
# In .env
TARGET_DEVICE=CPU

cd ../ovms-service && ./setup_models.sh && cd ../dine-in
make down && make up
```

---

## Troubleshooting

### OVMS Not Starting

```bash
docker logs dinein_ovms_vlm
ls -la ../ovms-service/models/
```

OVMS can take 2–5 minutes to load the model. Wait for `"Server started"` in the logs.

### GPU Not Detected

```bash
sudo usermod -aG render $USER
# Log out and back in, then restart services
make down && make up
```

### No Scenarios in UI

Ensure images are in `images/` and `configs/orders.json` has entries with matching `image_id` values (filename without extension).

---

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
make help                            # All commands
```

---

## Next Steps

- [System Requirements](./get-started/system-requirements.md) - Check the requirements
- [Build from Source](./get-started/build-from-source.md) - Build from source
- [How It Works](./how-it-works.md) - Learn about the architecture
- [How to Use](./how-to-use.md) - Customize settings
- [API Reference](./api-reference.md) - Learn the API
- [Release Notes](./release-notes.md) - Read about updates and improvements

<!--hide_directive
:::{toctree}
:hidden:

./get-started/system-requirements.md
./get-started/build-from-source.md

:::
hide_directive-->
