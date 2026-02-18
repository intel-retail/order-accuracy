# Get Started

Quick start guide for the Dine-In Order Accuracy application.

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.20+
- 16GB+ RAM
- Intel GPU (Arc/Flex recommended)

## Quick Start

### 1. Clone and Navigate

```bash
cd dine-in
```

### 2. Build Images

```bash
make build
```

### 3. Start Services

```bash
make up
```

This starts 4 containers:
- `dinein_app` - Main application (Gradio UI + API)
- `dinein_ovms_vlm` - Vision-Language Model server
- `dinein_semantic_service` - Semantic matching service
- `metrics-collector` - System metrics

### 4. Access the Application

| Interface | URL |
|-----------|-----|
| Gradio UI | http://localhost:7861 |
| REST API | http://localhost:8083 |
| API Docs | http://localhost:8083/docs |

### 5. Run Validation

**Via UI:**
1. Open http://localhost:7861
2. Select a scenario from dropdown
3. Click "Validate Plate"

**Via API:**
```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/DD-6993.jpg" \
  -F 'order={"order_id":"DD-6993","items":[{"name":"Maharaja Mac Chicken","quantity":1}]}'
```

## Running Benchmarks

### Prerequisites

Before running benchmarks, initialize the performance-tools submodule:

```bash
make update-submodules
```

### Quick Single Image Test

For a quick validation test with a single image:

```bash
make benchmark-single
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
| `BENCHMARK_DURATION` | 300 | Benchmark duration (seconds) |
| `BENCHMARK_TARGET_DEVICE` | GPU | Target device: CPU, GPU, NPU |
| `RESULTS_DIR` | results | Output directory |

Example with custom settings:

```bash
make benchmark BENCHMARK_WORKERS=2 BENCHMARK_DURATION=600
```

### Stream Density Test

```bash
make benchmark-density
```

### Metrics Processing

After running benchmarks, consolidate and visualize metrics:

```bash
# Consolidate metrics from multiple runs
make consolidate-metrics

# Generate plots from benchmark metrics
make plot-metrics
```

### Stream Density Configuration

All benchmark parameters can be configured via **environment variables** or **CLI arguments**. CLI arguments take precedence.

| Environment Variable | CLI Argument | Default | Description |
|---------------------|--------------|---------|-------------|
| `TARGET_LATENCY_MS` | `--target_latency_ms` | 15000 | Target latency threshold (ms) |
| `LATENCY_METRIC` | `--latency_metric` | avg | Metric: `avg`, `p95`, or `max` |
| `DENSITY_INCREMENT` | `--density_increment` | 1 | Concurrent images per iteration |
| `INIT_DURATION` | `--init_duration` | 60 | Warmup time per iteration (s) |
| `MIN_REQUESTS` | `--min_requests` | 3 | Min requests before measuring |
| `REQUEST_TIMEOUT` | `--request_timeout` | 300 | Request timeout (seconds) |
| `API_ENDPOINT` | `--api_endpoint` | http://localhost:8083 | API URL |
| `RESULTS_DIR` | `--results_dir` | ./results | Output directory |

**Using Environment Variables:**

```bash
# Set in .env file or export directly
export TARGET_LATENCY_MS=20000
export DENSITY_INCREMENT=2
export LATENCY_METRIC=p95

# Run benchmark (uses env vars)
make benchmark-density
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

## Stopping Services

```bash
make down
```

## Troubleshooting

### Services not starting

```bash
# Check container logs
make logs

# Restart services
make down && make up
```

### VLM inference slow

- Ensure GPU drivers are installed
- Check GPU utilization: `intel_gpu_top`
- Verify OVMS is using GPU in logs

### Out of memory

- Increase Docker memory limit
- Reduce cache size in `api.py`

## Next Steps

- [How to Use the Application](how-to-use-application.md)
- [API Reference](api-reference.md)
- [System Requirements](system-requirements.md)
