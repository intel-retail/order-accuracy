# Benchmarking Guide — Dine-In Order Accuracy

This guide covers performance testing, stream density benchmarking, and metrics collection for the Dine-In Order Accuracy system.

> **Note — Inference Device:** The default device is `GPU`. To switch to `CPU`, you must do **both** steps below, otherwise the model will be exported for the wrong device:
>
> 1. Set **both** variables in your `.env` file:
>
>    ```bash
>    TARGET_DEVICE=GPU      # used by setup_models.sh and docker-compose
>    OPENVINO_DEVICE=GPU    # used by the Makefile benchmark targets
>    ```
>
> 2. Re-export the model for the new device:
>
>    ```bash
>    cd ../ovms-service && ./setup_models.sh --app dine-in
>    ```
>
> `TARGET_DEVICE` is what `setup_models.sh` reads to export the model in the correct format. `OPENVINO_DEVICE` is what the Makefile passes to the benchmark script. Both must match.

## Prerequisites

```bash
# 1. Initialize git submodules (first time only)
make update-submodules

# 2. Start services
make up
```

> **Important:** The `images/` folder does not contain sample images. Add your own before testing:
>
> 1. Place plate images in `images/` (`.jpg`, `.jpeg`, or `.png`)
> 2. Edit `configs/orders.json` — add entries with `image_id` matching your filenames
> 3. Edit `configs/inventory.json` — define all possible menu items

## Benchmark Commands

### Single Image Test

```bash
make benchmark-single IMAGE_ID=MCD-1001
```

### Full Benchmark

```bash
make benchmark
```

> **Note:** `make benchmark` uses Docker profiles to start worker containers. Both the `dine-in` app and `dinein-worker` services use the **same Docker image** (built from the same Dockerfile). The worker is simply the same container running `worker.py` instead of the UI.

**Variables:**

| Variable                      | Default | Description            |
| ----------------------------- | ------- | ---------------------- |
| `BENCHMARK_WORKERS`           | 1       | Concurrent workers     |
| `BENCHMARK_DURATION`          | 180     | Duration (seconds)     |
| `BENCHMARK_TARGET_LATENCY_MS` | 25000   | Latency threshold (ms) |
| `TARGET_DEVICE`               | GPU     | Device: CPU, GPU       |

### Stream Density Benchmark

Finds the maximum number of concurrent image validations within the latency target:

```bash
make benchmark-stream-density

# With overrides
make benchmark-stream-density BENCHMARK_TARGET_LATENCY_MS=20000 BENCHMARK_INIT_DURATION=30
```

> **Note:** `make benchmark-density` runs a Python script locally that sends concurrent HTTP requests to the running `dine-in` API. No separate worker containers are needed for this mode.

## Metrics Processing

```bash
# Consolidate metrics from multiple runs into a single CSV
make consolidate-metrics

# Generate plots from consolidated metrics
make plot-metrics
```
