# Benchmarking Guide for Take-Away Order Accuracy

This guide covers performance testing, stream density benchmarking, and metrics collection for the Take-Away Order Accuracy system.

---

## Table of Contents

1. [Overview](#overview)
2. [Benchmark Types](#benchmark-types)
3. [Running Benchmarks](#running-benchmarks)
4. [Stream Density Testing](#stream-density-testing)
5. [Performance Metrics](#performance-metrics)
6. [Results Analysis](#results-analysis)
7. [Optimization Recommendations](#optimization-recommendations)

---

## Overview

The benchmarking suite evaluates system performance under various conditions:

- **Single Video Benchmark**: Basic latency and throughput testing
- **Stream Density**: Maximum concurrent stations/streams
- **VLM Performance**: Model inference metrics
- **End-to-End Validation**: Complete order validation cycle

### Benchmark Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       BENCHMARKING ARCHITECTURE                              │
│                                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐                │
│  │   Benchmark  │────▶│  Order       │────▶│  Metrics     │                │
│  │   Script     │     │  Accuracy    │     │  Collector   │                │
│  └──────────────┘     │  Service     │     └──────────────┘                │
│                       └──────────────┘             │                        │
│                              │                     │                        │
│                              ▼                     ▼                        │
│                       ┌──────────────┐     ┌──────────────┐                │
│                       │  OVMS VLM    │     │  Results     │                │
│                       │  (GPU)       │     │  JSON/CSV    │                │
│                       └──────────────┘     └──────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Types

### 1. Single Video Benchmark

Tests end-to-end latency for a single order validation.

```bash
make benchmark
```

**Metrics Collected:**
- Video upload time
- Frame extraction time
- VLM inference latency
- Validation time
- Total end-to-end latency

### 2. Fixed Workers Benchmark

Tests system with fixed number of concurrent workers.

```bash
make benchmark-oa BENCHMARK_WORKERS=4
```

**Metrics Collected:**
- Throughput (orders/minute)
- Latency percentiles (P50, P95, P99)
- GPU utilization
- Memory usage

### 3. Stream Density Benchmark

Finds maximum sustainable stream count under latency constraints.

```bash
make benchmark-oa-density
```

**Metrics Collected:**
- Maximum concurrent streams
- Latency at each stream count
- Point of degradation
- Resource utilization at capacity

---

## Running Benchmarks

### Prerequisites

```bash
# 1. Initialize git submodules (first time only)
make update-submodules

# 2. Build benchmark tools
make build-benchmark

# 3. Ensure services are running
make up

# 4. Verify health
make test-api

# 5. Check GPU availability
nvidia-smi  # or clinfo for Intel
```

### Single Video Benchmark

```bash
# Basic benchmark
make benchmark

# With custom video
BENCHMARK_VIDEO=storage/videos/custom.mp4 make benchmark

# Multiple runs
for i in {1..10}; do make benchmark; done
```

**Output:**
```
=== Starting Benchmark ===
Video: storage/videos/test.mp4

=== Benchmark Results ===
{
  "status": "success",
  "video_id": "benchmark_test",
  "processing_time_ms": 3450
}

Latency: 3450ms
Results logged to results/benchmark_performance.log
```

### Fixed Workers Benchmark

```bash
# Configure and run
make benchmark-oa \
  BENCHMARK_WORKERS=4 \
  BENCHMARK_DURATION=300 \
  BENCHMARK_INIT_DURATION=30
```

**Configuration Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| BENCHMARK_WORKERS | 1 | Number of workers |
| BENCHMARK_DURATION | 60 | Test duration (seconds) |
| BENCHMARK_INIT_DURATION | 10 | Warmup time (seconds) |

**Output:**
```
╔═══════════════════════════════════════════════════════════════════╗
║       Order Accuracy Benchmark - Fixed Workers Mode               ║
╚═══════════════════════════════════════════════════════════════════╝
Workers: 4
Duration: 300s
Init Duration: 30s

Running benchmark...
[========================================] 100% (300/300s)

=== RESULTS ===
Total Orders:     125
Throughput:       25.0 orders/min
Latency P50:      2850ms
Latency P95:      4200ms
Latency P99:      5100ms
GPU Utilization:  72%
```

### Stream Density Benchmark

```bash
# Run stream density test
make benchmark-oa-density \
  BENCHMARK_TARGET_LATENCY_MS=3000 \
  BENCHMARK_MIN_TRANSACTIONS=3 \
  BENCHMARK_WORKER_INCREMENT=1
```

**Configuration Variables:**

All benchmark parameters can be configured via **environment variables** or **CLI arguments**. CLI arguments take precedence over environment variables.

| Environment Variable | CLI Argument | Default | Description |
|---------------------|--------------|---------|-------------|
| `TARGET_LATENCY_MS` | `--target_latency_ms` | 15000 | Target latency threshold (ms) |
| `LATENCY_METRIC` | `--latency_metric` | avg | Metric to use: `avg` or `p95` |
| `WORKER_INCREMENT` | `--worker_increment` | 1 | Workers added per iteration |
| `INIT_DURATION` | `--init_duration` | 120 | Warmup time per iteration (s) |
| `MIN_TRANSACTIONS` | `--min_transactions` | 3 | Min orders before measuring |
| `MAX_ITERATIONS` | - | 50 | Max scaling iterations |
| `MAX_WAIT_SEC` | - | 600 | Max wait per iteration (s) |
| `RESULTS_DIR` | `--results_dir` | ./results | Output directory |
| `OOM_PROTECTION` | - | 1 | Enable OOM protection |

**Using Environment Variables:**

```bash
# Set in .env file or export directly
export TARGET_LATENCY_MS=20000
export WORKER_INCREMENT=2
export LATENCY_METRIC=p95

# Run benchmark (uses env vars)
make benchmark-oa-density
```

**Using CLI Arguments (override env vars):**

```bash
# CLI args take precedence
TARGET_LATENCY_MS=20000 python3 stream_density_latency_oa.py \
  --compose_file docker-compose.yaml \
  --target_latency_ms 25000  # This value is used (25s)
```

**Output:**
```
╔═══════════════════════════════════════════════════════════════════╗
║       Order Accuracy Stream Density - Latency Based               ║
╚═══════════════════════════════════════════════════════════════════╝
Target Latency: 3000ms (3s)
Latency Metric: p95
Init Duration: 10s
Min Transactions: 3
Worker Increment: 1

=== Stream Density Results ===
Workers | Throughput | P50     | P95     | P99     | Status
--------|------------|---------|---------|---------|--------
1       | 12.5/min   | 2400ms  | 2800ms  | 3100ms  | PASS
2       | 22.3/min   | 2600ms  | 2950ms  | 3200ms  | PASS
3       | 28.1/min   | 2850ms  | 3150ms  | 3500ms  | FAIL

Maximum Sustainable Streams: 2
Limiting Factor: P95 latency exceeded 3000ms
```

---

## Stream Density Testing

### Understanding Stream Density

Stream density measures the maximum number of concurrent video streams the system can process while meeting latency requirements.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STREAM DENSITY PROGRESSION                                │
│                                                                              │
│  Latency                                                                     │
│    ^                                                                         │
│    |                                              X X X                      │
│    |                                        X                                │
│    |  Target ─────────────────────────X────────────────────                 │
│    |                              X                                          │
│    |                        X                                                │
│    |                  X                                                      │
│    |            X                                                            │
│    |      X                                                                  │
│    +──────────────────────────────────────────────────▶ Streams             │
│         1    2    3    4    5    6    7    8                                │
│                                                                              │
│         ↑_________________________↑                                          │
│             Acceptable Zone       Max Density                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Latency Metrics

| Metric | Description | Typical Target |
|--------|-------------|----------------|
| P50 | Median latency | 2-3 seconds |
| P95 | 95th percentile | 3-5 seconds |
| P99 | 99th percentile | 5-8 seconds |

### Running Density Test

**Option 1: Using Environment Variables (.env)**

Configure benchmarks in your `.env` file:

```bash
# .env file
TARGET_LATENCY_MS=5000
LATENCY_METRIC=p95
WORKER_INCREMENT=1
INIT_DURATION=120
MIN_TRANSACTIONS=3
MAX_ITERATIONS=50
RESULTS_DIR=./results
```

Then run:

```bash
make benchmark-oa-density
```

**Option 2: Using Export Commands**

```bash
# Export variables
export TARGET_LATENCY_MS=5000
export WORKER_INCREMENT=1
export LATENCY_METRIC=p95

# Run benchmark
make benchmark-oa-density
```

**Option 3: Inline Environment Variables**

```bash
# Conservative test (safe for production hardware)
TARGET_LATENCY_MS=5000 WORKER_INCREMENT=1 OOM_PROTECTION=1 \
  make benchmark-oa-density

# Aggressive test (may stress hardware)
TARGET_LATENCY_MS=3000 WORKER_INCREMENT=2 OOM_PROTECTION=0 \
  make benchmark-oa-density  # WARNING: Risk of OOM
```

**Option 4: Direct Script Invocation with CLI**

```bash
# Using CLI arguments (override env vars)
python3 benchmark-scripts/stream_density_latency_oa.py \
  --compose_file docker-compose.yaml \
  --target_latency_ms 5000 \
  --latency_metric p95 \
  --worker_increment 1 \
  --init_duration 120 \
  --min_transactions 3 \
  --results_dir ./results
```

### OOM Protection

```bash
# Enable OOM protection (recommended)
OOM_PROTECTION=1 make benchmark-oa-density

# Warning displayed if disabled
╔════════════════════════════════════════════════════════════╗
║ WARNING: OOM Protection DISABLED                           ║
║                                                            ║
║ This test may:                                             ║
║ • Exhaust system memory                                    ║
║ • Cause system instability                                 ║
║ • Require hard reboot                                      ║
╚════════════════════════════════════════════════════════════╝
```

---

## Performance Metrics

### VLM Metrics

View VLM-specific metrics:

```bash
make benchmark-oa-metrics
```

**Output:**
```
╔═══════════════════════════════════════════════════════════════════╗
║                     VLM Metrics Logs                              ║
╚═══════════════════════════════════════════════════════════════════╝

=== Application Metrics ===
Total Requests:        125
Successful:            123
Failed:                2
Success Rate:          98.4%

=== Performance Metrics ===
Avg Latency:           2850ms
P50 Latency:           2700ms
P95 Latency:           3500ms
P99 Latency:           4200ms
Avg Tokens/Request:    1350
GPU Utilization:       72%
```

### Metrics Files

Results are saved to `results/` directory:

```
results/
├── benchmark_performance.log      # Simple latency log
├── stream_density_results.json    # Detailed density results
├── vlm_application_metrics_*.txt  # VLM app metrics
└── vlm_performance_metrics_*.txt  # VLM performance metrics
```

### Metrics Schema

**stream_density_results.json:**
```json
{
  "test_config": {
    "target_latency_ms": 3000,
    "latency_metric": "p95",
    "worker_increment": 1,
    "timestamp": "2025-01-15T10:30:00Z"
  },
  "results": [
    {
      "workers": 1,
      "transactions": 25,
      "throughput_per_min": 12.5,
      "latency_p50_ms": 2400,
      "latency_p95_ms": 2800,
      "latency_p99_ms": 3100,
      "gpu_utilization_pct": 45,
      "memory_used_mb": 8500,
      "status": "pass"
    }
  ],
  "summary": {
    "max_streams": 2,
    "limiting_factor": "latency",
    "recommended_streams": 1
  }
}
```

---

## Results Analysis

### Viewing Results

```bash
# View all results
make benchmark-oa-results

# Parse JSON results
jq '.summary' results/stream_density_results.json

# Export to CSV
jq -r '.results[] | [.workers, .throughput_per_min, .latency_p95_ms] | @csv' \
  results/stream_density_results.json > results/summary.csv
```

### Consolidating Metrics

After running multiple benchmarks, consolidate metrics into a single CSV file:

```bash
# Consolidate all VLM metrics files
make consolidate-metrics
```

This generates:
- `results/consolidated_metrics.csv` - All metrics in tabular format
- `results/metrics_summary.json` - Summary statistics

### Generating Plots

Visualize benchmark results with built-in plotting:

```bash
# Generate plots from consolidated metrics
make plot-metrics
```

This generates:
- `results/latency_over_time.png` - Latency trends
- `results/throughput_scaling.png` - Throughput vs workers
- `results/resource_utilization.png` - CPU/GPU/Memory usage

### Manual Report Generation

```bash
# Generate performance report
cat results/benchmark_performance.log | \
  awk '{sum+=$NF; count++} END {print "Average:", sum/count, "ms"}'

# Plot latency distribution (requires matplotlib)
python3 -c "
import json
import matplotlib.pyplot as plt

with open('results/stream_density_results.json') as f:
    data = json.load(f)

workers = [r['workers'] for r in data['results']]
p95 = [r['latency_p95_ms'] for r in data['results']]

plt.plot(workers, p95, marker='o')
plt.xlabel('Workers')
plt.ylabel('P95 Latency (ms)')
plt.axhline(y=3000, color='r', linestyle='--', label='Target')
plt.legend()
plt.savefig('results/latency_plot.png')
"
```

### Interpreting Results

**Healthy System:**
```
Workers: 4
P95 Latency: 2800ms (under 3000ms target)
GPU Utilization: 70-75%
Throughput: 25 orders/min
```

**Overloaded System:**
```
Workers: 6
P95 Latency: 5500ms (exceeds target)
GPU Utilization: 95%+ 
Throughput: 18 orders/min (declining)
```

---

## Optimization Recommendations

### Based on Benchmark Results

| Symptom | Likely Cause | Recommendation |
|---------|--------------|----------------|
| High latency, low GPU util | Batching inefficient | Increase batch window |
| High GPU util, high latency | GPU saturated | Add GPU or reduce workers |
| Memory errors | OOM | Reduce batch size |
| Throughput plateau | VLM bottleneck | Enable request batching |

### Configuration Tuning

```bash
# For lower latency (fewer streams)
VLM_BATCH_SIZE=1
VLM_BATCH_TIMEOUT_MS=0

# For higher throughput (more streams)
VLM_BATCH_SIZE=4
VLM_BATCH_TIMEOUT_MS=100

# For memory-constrained systems
OPENVINO_DEVICE=CPU
VLM_MAX_CONCURRENT=2
```

### Hardware Scaling

| Current Capacity | To Achieve | Action |
|------------------|------------|--------|
| 2 streams | 4 streams | Add second GPU |
| 4 streams | 8 streams | Upgrade to A100 |
| 8 streams | 16 streams | Multi-node deployment |

---

## Quick Reference

```bash
# Setup (first time)
make update-submodules       # Initialize performance-tools submodule
make build-benchmark         # Build benchmark Docker image

# Basic benchmark
make benchmark

# Fixed workers benchmark
make benchmark-oa BENCHMARK_WORKERS=4

# Stream density test
make benchmark-oa-density

# View metrics
make benchmark-oa-metrics

# View all results
make benchmark-oa-results

# Metrics processing
make consolidate-metrics     # Consolidate metrics to CSV
make plot-metrics            # Generate plots

# Cleanup
make clean-metrics           # Remove metrics files only
make clean-results           # Remove all results
make clean                   # Stop containers and remove volumes

# Help
make benchmark-oa-help
make help                    # Show all commands
```
