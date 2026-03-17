# Getting Started with Dine-In Order Accuracy

This guide walks you through the installation, configuration, and first-run of the Dine-In Order Accuracy system for image-based plate validation.

> **Note — `TARGET_DEVICE`**: To change the inference device mode, set `TARGET_DEVICE` in your `.env` file to `GPU`, `CPU`, or `AUTO`. After changing the device, re-run the setup script to update the model config:
> ```bash
> cd ../ovms-service && ./setup_models.sh --app dine-in
> ```
> You can also pass the device explicitly: `./setup_models.sh --device CPU`

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

```

---

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/intel-retail/order-accuracy.git
cd order-accuracy/dine-in
```

### Step 2: Initialize Git Submodules

The performance-tools repository is included as a git submodule for benchmarking:

```bash
# Initialize and update submodules
make update-submodules

# Or manually
cd ..
git submodule update --init --recursive
cd dine-in
```

### Step 3: Setup OVMS Models (First Time Only)

The VLM model must be downloaded before running the application:

```bash
cd ../ovms-service
./setup_models.sh
```

This script:
- Downloads Qwen2.5-VL-7B-Instruct from HuggingFace (~7GB)
- Converts to OpenVINO format with INT8 quantization
- Creates model files in `ovms-service/models/`

> **Note**: This only needs to be done once. The model files are shared between dine-in and take-away applications.

### Step 4: Prepare Test Data

Before running the application, prepare your test data:

1. **Add Images**: Place your food tray/plate images in the `images/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Images should clearly show the food items on the plate

2. **Update Orders**: Edit `configs/orders.json` with your test orders
   - Each order should have an `image_id` and list of `items_ordered`
   - Image IDs should match your image filenames (without extension)

3. **Update Inventory**: Edit `configs/inventory.json` to match your menu items
   - Define all possible food items that can appear in orders
   - Include item names, categories, and any relevant metadata

> **Note**: The `images/` folder does not contain sample images by default. You must add your own images before testing.

### Step 5: Create Environment File

```bash
# Copy .env template
cp .env.example .env
# Edit if needed (defaults work for most setups)
```

### Step 6: Build and Start

```bash
# Pull images from registry (default)
make build
make up

# OR build locally from source
make build REGISTRY=false
make up REGISTRY=false
```

> **Note**: `make build` pulls pre-built images from Docker Hub by default. Use `REGISTRY=false` to build from source.

---

## Configuration

### Basic Configuration (.env)

```bash
# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO

# =============================================================================
# Service Endpoints
# =============================================================================
OVMS_ENDPOINT=http://ovms-vlm:8000
OVMS_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
SEMANTIC_SERVICE_ENDPOINT=http://semantic-service:8080
API_TIMEOUT=60
```

### Benchmarking Configuration (.env)

For stream density and performance testing, configure these variables:

```bash
# =============================================================================
# Stream Density Benchmarking
# =============================================================================
BENCHMARK_TARGET_LATENCY_MS=15000      # Target latency threshold (ms)
BENCHMARK_LATENCY_METRIC=avg           # 'avg', 'p95', or 'max'
BENCHMARK_DENSITY_INCREMENT=1          # Concurrent images per iteration
BENCHMARK_INIT_DURATION=60             # Warmup time (seconds)
BENCHMARK_MIN_REQUESTS=3               # Min requests before measuring
BENCHMARK_REQUEST_TIMEOUT=300          # Individual request timeout (seconds)
BENCHMARK_API_ENDPOINT=http://localhost:8083
RESULTS_DIR=./results
```

> **Note:** Short aliases without the `BENCHMARK_` prefix (e.g. `TARGET_LATENCY_MS`, `LATENCY_METRIC`) are also accepted as CLI overrides for convenience.

> **Note:** CLI arguments override environment variables. See [Benchmark Configuration](#stream-density-configuration) for detailed usage.

---

## Starting the Services

### Standard Mode (Production)

```bash
# Start all services
make up

# View logs
make logs
```

### Using Local Build

```bash
# Build and run from local source
make build REGISTRY=false
make up REGISTRY=false
```

This starts 4 containers:
- `dinein_app` - Main application (Gradio UI + FastAPI)
- `dinein_ovms_vlm` - Vision-Language Model server (OVMS)
- `dinein_semantic_service` - Semantic matching service
- `metrics-collector` - System metrics collector

### Stream Density Benchmark

To measure the maximum number of concurrent image validations the system can sustain under a latency target:

```bash
make benchmark-stream-density
```

This automatically scales concurrent requests up, measuring end-to-end latency at each level, and stops when the target latency (default 15s) is exceeded. Results are saved to `./results/`.

Override defaults via environment or CLI:

```bash
make benchmark-stream-density \
  BENCHMARK_TARGET_LATENCY_MS=20000 \
  BENCHMARK_INIT_DURATION=30
```

### Metrics Processing

After running benchmarks, consolidate and visualize metrics:

```bash
# Consolidate metrics from multiple runs
make consolidate-metrics

# Generate plots from benchmark metrics
make plot-metrics
```

### Service Status

```bash
# Check running containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Expected output:
# dinein_app               Up 2 minutes   0.0.0.0:7861->7860/tcp, 0.0.0.0:8083->8080/tcp
# dinein_ovms_vlm          Up 2 minutes   0.0.0.0:8002->8000/tcp
# dinein_semantic_service  Up 2 minutes   0.0.0.0:8081->8080/tcp, 0.0.0.0:9091->9090/tcp
# metrics-collector        Up 2 minutes   0.0.0.0:8084->8084/tcp
```

---

## Verifying Installation

### Step 1: Health Check

```bash
# Test API health
make test-api

# Or directly
curl http://localhost:8083/health
# Expected: {"status": "healthy", ...}
```

### Step 2: Verify OVMS Model

```bash
# Check OVMS configuration
curl http://localhost:8002/v1/config | jq .

# Expected: Model configuration with Qwen2.5-VL model details
```

### Step 3: Access Gradio UI

Open http://localhost:7861 in your browser.

You should see the Order Accuracy web interface with:
- Scenario selection dropdown
- Order manifest display
- Validation results panel
- One-click "Validate Plate" button

### Step 4: Access API Documentation

Open http://localhost:8083/docs to view OpenAPI/Swagger documentation for all REST API endpoints.

---

## First Order Validation

### Option 1: Via Gradio UI

1. Open http://localhost:7861
2. Select a scenario from the dropdown (must have matching image in `images/` folder)
3. Review the order manifest displayed
4. Click "Validate Plate"
5. View results showing:
   - Order completion status
   - Accuracy score
   - Matched, missing, and extra items
   - Performance metrics

### Option 2: Via REST API

```bash
# Upload image and validate with order
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={"items":[{"name":"Cheeseburger","quantity":1},{"name":"French Fries","quantity":1}]}'

# Response
{
  "validation_id": "...",
  "order_complete": true,
  "accuracy_score": 100.0,
  "matched_items": [...],
  "missing_items": [],
  "extra_items": [],
  "metrics": {...}
}
```

### Option 3: Using Make Target

> **Prerequisite:** Services must be running (`make up`) before using this command.

```bash
# Quick single image test (IMAGE_ID from configs/orders.json)
make benchmark-single IMAGE_ID=MCD-1001
```

---

## Running Benchmarks

### Prerequisites

Before running benchmarks, initialize the performance-tools submodule:

```bash
make update-submodules
```

### Quick Single Image Test

For a quick validation test with a single image:

> **Prerequisite:** Services must be running. If not already started, run `make up` first.

```bash
# IMAGE_ID must match an entry in configs/orders.json and a file in images/
# Format: <PREFIX>-<NUMBER> (e.g., MCD-1001, MCD-1002)
make benchmark-single IMAGE_ID=MCD-1001
```

### Full Benchmark

Run the Order Accuracy benchmark using `benchmark_order_accuracy.py`:

```bash
make benchmark
```

Configuration options:

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_WORKERS` | 1 | Number of concurrent workers |
| `BENCHMARK_DURATION` | 180 | Benchmark duration (seconds) |
| `TARGET_DEVICE` | GPU | Target device: CPU, GPU, NPU |
| `RESULTS_DIR` | results | Output directory |

Example with custom settings:

```bash
make benchmark BENCHMARK_WORKERS=2 BENCHMARK_DURATION=600 TARGET_DEVICE=GPU
```

### Stream Density Test

```bash
make benchmark-stream-density
```

### Stream Density Configuration

All benchmark parameters can be configured via **environment variables** or **CLI arguments**. CLI arguments take precedence.

| Make Variable | Short Alias | CLI Argument | Default | Description |
|--------------|-------------|--------------|---------|-------------|
| `BENCHMARK_TARGET_LATENCY_MS` | `TARGET_LATENCY_MS` | `--target_latency_ms` | 25000 | Target latency threshold (ms) |
| `BENCHMARK_LATENCY_METRIC` | `LATENCY_METRIC` | `--latency_metric` | avg | Metric: `avg`, `p95`, or `max` |
| `BENCHMARK_DENSITY_INCREMENT` | `DENSITY_INCREMENT` | `--density_increment` | 1 | Concurrent images per iteration |
| `BENCHMARK_INIT_DURATION` | `INIT_DURATION` | `--init_duration` | 60 | Warmup time per iteration (s) |
| `BENCHMARK_MIN_REQUESTS` | `MIN_REQUESTS` | `--min_requests` | 3 | Min requests before measuring |
| `BENCHMARK_REQUEST_TIMEOUT` | `REQUEST_TIMEOUT` | `--request_timeout` | 300 | Request timeout (seconds) |
| `BENCHMARK_API_ENDPOINT` | `API_ENDPOINT` | `--api_endpoint` | http://localhost:8083 | API URL |
| `RESULTS_DIR` | — | `--results_dir` | ./results | Output directory |

> Both the full `BENCHMARK_*` name and the short alias work interchangeably on the `make` command line. `BENCHMARK_*` takes precedence if both are supplied.

**Using Environment Variables:**

```bash
# Set in .env file or export directly
export BENCHMARK_TARGET_LATENCY_MS=20000
export BENCHMARK_DENSITY_INCREMENT=2
export BENCHMARK_LATENCY_METRIC=p95

# Run benchmark (uses env vars)
make benchmark-stream-density

# Short aliases also work on the CLI:
make benchmark-stream-density TARGET_LATENCY_MS=20000 DENSITY_INCREMENT=2 LATENCY_METRIC=p95
```

**Using CLI Arguments (override env vars):**

```bash
python3 stream_density_oa_dine_in.py \
  --compose_file docker-compose.yaml \
  --target_latency_ms 15000 \
  --latency_metric p95 \
  --density_increment 1
```

> **Note:** In dine-in context, "density" = concurrent image validation requests through VLM.

---

## Troubleshooting

### OVMS Not Starting

**Symptom**: `dinein_ovms_vlm` container exits immediately or shows errors

**Solution**:
```bash
# Check logs
docker logs dinein_ovms_vlm

# Verify model path exists
ls -la ../ovms-service/models/

# Check GPU availability
clinfo | head -20  # Inte
```

### Connection Refused to OVMS

**Symptom**: `Connection refused` errors to port 8002

**Solution**:
```bash
# Wait for OVMS to fully load model (can take 2-5 minutes)
docker logs -f dinein_ovms_vlm | grep -i "serving\|ready\|loaded"

# Check network
docker network inspect dine-in_dinein-net
```

### Services Not Starting

**Symptom**: Containers fail to start or exit immediately

**Solution**:
```bash
# Check container logs
make logs

# Restart services
make down && make up

# Check for port conflicts
netstat -tulpn | grep -E "7861|8083|8002|8081"
```

### VLM Inference Slow

**Symptom**: Validations take longer than expected (>30s)

**Solution**:
- Ensure GPU drivers are installed
- Check GPU utilization: `intel_gpu_top`
- Verify OVMS is using GPU in logs: `docker logs dinein_ovms_vlm | grep -i gpu`
- Consider reducing image resolution in preprocessing

### Out of Memory

**Symptom**: Services crash with OOM errors

**Solution**:
```bash
# Check Docker memory limits
docker stats

# Increase Docker memory limit in Docker Desktop or daemon config

# Use CPU instead of GPU (slower but less memory)
# Edit docker-compose.yml to remove GPU device mapping temporarily

# Restart services
make down && make up
```

### GPU Not Detected

**Symptom**: `No GPU devices found` or model running on CPU

**Solution**:
```bash
# For Intel GPU
sudo usermod -aG render $USER
sudo usermod -aG video $USER
# Logout and login again

# Verify GPU access
ls -la /dev/dri/

```

### No Scenarios Available in UI

**Symptom**: Dropdown is empty in Gradio UI

**Solution**:
1. Ensure images are placed in `images/` folder
2. Ensure `configs/orders.json` has entries with matching `image_id` values
3. Image ID must match filename (e.g., `image_id: "MCD-1001"` matches `MCD-1001.png`)

---

## Stopping Services

```bash
make down
```

---

## Next Steps

After successful installation and first validation:

1. **Configure for Production**: See [System Requirements](system-requirements.md)
2. **Learn the API**: See [API Reference](api-reference.md)
3. **Run Benchmarks**: See [How to Use Application](how-to-use-application.md)
4. **Customize Settings**: Edit `.env` and `configs/` files

---

## Quick Reference Commands

```bash
# Start services
make up                        # Standard mode (registry image)
make up REGISTRY=false         # Using locally built image

# Build
make build                     # Pull from registry
make build REGISTRY=false      # Build locally

# Check status
docker ps
make logs

# Stop services
make down

# Clean everything
make clean

# Run benchmarks
make benchmark-single IMAGE_ID=MCD-1001  # Quick single image test
make benchmark                        # Full benchmark
make benchmark-stream-density          # Stream density test

# Development
make shell                     # Shell into container
make test-api                  # Test endpoints
make help                      # Show all commands
```

- [How to Use the Application](how-to-use-application.md)
- [API Reference](api-reference.md)
- [System Requirements](system-requirements.md)
