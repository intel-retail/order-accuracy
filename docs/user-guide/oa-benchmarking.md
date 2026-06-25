# Performance Testing & Benchmarking

Test your Order Accuracy pipeline performance on various hardware configurations. This guide covers everything from quick performance checks to comprehensive system capacity testing.

## Quick Start (5 minutes)

**Goal**: Run a basic performance test to verify your system works correctly

### 1. Initialize Performance Tools

```bash
make update-submodules
```

### 2. Run Quick Benchmark

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: dine-in hide_directive-->

```bash
cd dine-in
make benchmark
```

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

```bash
cd take-away
make benchmark
```

<!--hide_directive:::
::::hide_directive-->

**What this does:**

- Tests GPU/CPU performance for order validation
- Measures end-to-end latency
- Generates performance metrics
- Outputs results to `results/` directory

## Understanding Benchmark Types

<!--hide_directive::::::{tab-set}
:::::{tab-item}hide_directive--> **Dine-In Benchmarks**
<!--hide_directive:sync: dine-in hide_directive-->

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Single Request Benchmark**
<!--hide_directive:sync: singlehide_directive-->

```bash
make benchmark
```

Tests single image validation latency:

- Image preprocessing time
- VLM inference time
- Semantic matching time
- Total end-to-end latency

<!--hide_directive:::
:::{tab-item}hide_directive--> **Stream Density Benchmark**
<!--hide_directive:sync: density hide_directive-->

```bash
make benchmark-density
```

Finds maximum concurrent requests the system can handle under latency constraints:

- Target latency threshold (configurable)
- Progressive load increase
- Identifies performance ceiling

<!--hide_directive:::
::::
:::::
:::::{tab-item}hide_directive--> **Take-Away Benchmarks**
<!--hide_directive:sync: take-away hide_directive-->

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Single Video Benchmark**
<!--hide_directive:sync: singlehide_directive-->

```bash
make benchmark
```

Tests end-to-end latency for single order validation:

- Video upload time
- Frame extraction time
- VLM inference latency
- Validation time
- Total processing time

<!--hide_directive:::
:::{tab-item}hide_directive--> **Fixed Workers Benchmark**

```bash
make benchmark-oa BENCHMARK_WORKERS=4 BENCHMARK_DURATION=300
```

Tests system with fixed number of concurrent workers:

- Throughput (orders/minute)
- Latency percentiles (P50, P95, P99)
- GPU utilization
- Memory usage

<!--hide_directive:::
:::{tab-item}hide_directive--> **Stream Density Benchmark**
<!--hide_directive:sync: density hide_directive-->

```bash
make benchmark-stream-density
```

Finds maximum sustainable worker count under latency constraints:

- Maximum concurrent workers
- Latency at each worker count
- Point of degradation
- Resource utilization at capacity

<!--hide_directive
:::
::::
:::::
::::::hide_directive-->

## Environment Variables Reference

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In Configuration**
<!--hide_directive:sync: dine-in hide_directive-->

| Variable            | Default                 | Description                          |
| ------------------- | ----------------------- | ------------------------------------ |
| `TARGET_LATENCY_MS` | 15000                   | Target latency threshold (ms)        |
| `LATENCY_METRIC`    | avg                     | 'avg', 'p95', or 'max'               |
| `DENSITY_INCREMENT` | 1                       | Concurrent images per iteration      |
| `INIT_DURATION`     | 60                      | Warmup time (seconds)                |
| `MIN_REQUESTS`      | 3                       | Min requests before measuring        |
| `REQUEST_TIMEOUT`   | 300                     | Individual request timeout (seconds) |
| `API_ENDPOINT`      | `http://localhost:8083` | API endpoint URL                     |
| `RESULTS_DIR`       | `./results`             | Results output directory             |

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away Configuration**
<!--hide_directive:sync: take-away hide_directive-->

| Variable             | Default | Description                       |
| -------------------- | ------- | --------------------------------- |
| `TARGET_LATENCY_MS`  | 25000   | Target latency threshold (ms)     |
| `LATENCY_METRIC`     | avg     | 'avg', 'p95', or 'max'            |
| `WORKER_INCREMENT`   | 1       | Workers added per iteration       |
| `INIT_DURATION`      | 10      | Warmup time (seconds)             |
| `MIN_TRANSACTIONS`   | 3       | Min transactions before measuring |
| `MAX_ITERATIONS`     | 50      | Max scaling iterations            |
| `MAX_WAIT_SEC`       | 600     | Max wait per iteration (seconds)  |
| `BENCHMARK_WORKERS`  | 1       | Number of workers (fixed mode)    |
| `BENCHMARK_DURATION` | 60      | Test duration (seconds)           |

<!--hide_directive:::
::::hide_directive-->

## Hardware Testing Commands

### GPU Performance Testing

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: dine-in hide_directive-->

```bash
# Ensure GPU device is configured in .env
# OPENVINO_DEVICE=GPU
make benchmark
```

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

```bash
# Configure GPU in .env
# OPENVINO_DEVICE=GPU
make benchmark-oa BENCHMARK_WORKERS=4
```

<!--hide_directive:::
::::hide_directive-->

### Multi-Worker Stress Testing (Take-Away)

```bash
# Test with 2 parallel workers
make up-parallel WORKERS=2
make benchmark-oa BENCHMARK_WORKERS=2

# High stress test with 8 workers
make up-parallel WORKERS=8
make benchmark-oa BENCHMARK_WORKERS=8
```

### Progressive Load Testing

```bash
# Automatically find maximum sustainable workers
make benchmark-stream-density \
  BENCHMARK_TARGET_LATENCY_MS=25000 \
  BENCHMARK_WORKER_INCREMENT=1 \
  BENCHMARK_MAX_ITERATIONS=20
```

## Viewing Results

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In Results**
<!--hide_directive:sync: dine-in hide_directive-->

```bash
# View density benchmark results
make benchmark-density-results

# View raw results
cat results/benchmark_results.json
ls -la results/
```

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away Results**
<!--hide_directive:sync: take-away hide_directive-->

```bash
# View benchmark results
make benchmark-oa-results

# View density results
cat results/stream_density_results.json
ls -la results/
```

<!--hide_directive:::
::::hide_directive-->

### Consolidate Metrics

```bash
make consolidate-metrics
cat results/metrics_summary.csv
```

## Expected Performance

### Typical Latency Ranges

| Operation               | Dine-In   | Take-Away       |
| ----------------------- | --------- | --------------- |
| **Image Preprocessing** | 100-500ms | N/A             |
| **Frame Selection**     | N/A       | 200-500ms       |
| **VLM Inference**       | 5-10s     | 5-10s           |
| **Semantic Matching**   | 50-200ms  | 50-200ms        |
| **Total End-to-End**    | 8-15s     | 8-15s per order |

### Hardware Impact

| Configuration      | Typical Performance   |
| ------------------ | --------------------- |
| **CPU Only**       | 15-25s per validation |
| **Intel iGPU**     | 8-15s per validation  |
| **Intel Arc dGPU** | 5-10s per validation  |
| **NVIDIA RTX**     | 4-8s per validation   |

### Throughput Expectations

| Mode                               | Expected Throughput |
| ---------------------------------- | ------------------- |
| **Dine-In Single**                 | 4-6 orders/minute   |
| **Take-Away Single**               | 4-6 orders/minute   |
| **Take-Away Parallel (4 workers)** | 16-24 orders/minute |
| **Take-Away Parallel (8 workers)** | 30-40 orders/minute |

## Optimization Tips

### GPU Utilization

- Monitor GPU usage with `nvidia-smi -l 1` or `intel_gpu_top`
- Target 70-90% GPU utilization for optimal throughput
- If GPU is underutilized, increase worker count

### Memory Management

- Monitor container memory with `docker stats`
- VLM models require 8-16GB GPU memory
- Reduce batch size if out-of-memory errors occur

### Network Optimization (Take-Away)

- Use wired connections for RTSP streams
- Ensure 1Gbps+ network bandwidth per camera
- Consider local video storage for testing

### Latency Reduction

- Use INT8 model quantization
- Enable HTTP/2 for API connections
- Pre-warm VLM model before benchmarking

## Troubleshooting Performance Issues

### Low FPS / High Latency

- Check GPU driver installation
- Verify OPENVINO_DEVICE setting in .env
- Reduce image resolution or batch size
- Check for thermal throttling

### VLM Timeout Errors

- Increase API_TIMEOUT in .env
- Check GPU memory availability
- Consider using smaller model precision

### Memory Exhaustion

- Reduce number of parallel workers
- Lower batch size settings
- Monitor with `docker stats`

### Inconsistent Results

- Increase warmup duration (INIT_DURATION)
- Increase minimum transactions (MIN_TRANSACTIONS)
- Run multiple benchmark iterations
