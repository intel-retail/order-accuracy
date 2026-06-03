# How to Use the Application

Guide to using the Dine-In Order Accuracy application features.

> **Note — `TARGET_DEVICE`**: To change the inference device, set `TARGET_DEVICE` in `.env` to `GPU` or `CPU`, then re-run setup:
> ```bash
> cd ../ovms-service && ./setup_models.sh --app dine-in && cd ../dine-in
> make down && make up
> ```

## Gradio UI

Access the web interface at http://localhost:7861

### Interface Overview

> **Note — negative test case**: The default MCD-1001 scenario in the Gradio UI
> intentionally submits a mismatched order (Cheeseburger / French Fries) against a tray
> image that contains Filet-O-Fish and Cheesy Fries. This demonstrates the application's
> ability to detect an incorrect order. The result will show `order_complete: ✗`. To see a
> successful validation, select another scenario or update the order to match the tray.

```
┌─────────────────────────────────────────────────────────────┐
│  Dine-In Order Accuracy Benchmark                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scenario: [MCD-1001 – McDonald's Table T12   ▼]           │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │                     │  │ Order Manifest              │  │
│  │    [Plate Image]    │  │ ─────────────────           │  │
│  │                     │  │ items_ordered:              │  │
│  │                     │  │   - Cheeseburger            │  │
│  │                     │  │   - French Fries            │  │
│  │                     │  │                             │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
│  [Validate Plate]                                          │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │ Validation Result   │  │ Performance Metrics         │  │
│  │ ─────────────────   │  │ ───────────────────         │  │
│  │ order_complete: ✗   │  │ vlm_inference_ms: 9003      │  │
│  │ accuracy_score: 0.0 │  │ cpu_utilization: 27%        │  │
│  │ missing_items: [..] │  │ gpu_utilization: 100%       │  │
│  │ extra_items: [...]  │  │ memory_utilization: 80%     │  │
│  └─────────────────────┘  └─────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Usage Steps

1. **Select Scenario**: Choose a test scenario from the dropdown
2. **Review Order**: Verify the order manifest on the right
3. **Validate**: Click "Validate Plate" button
4. **Review Results**: Check validation outcome and metrics

## REST API

### Validate Single Image

The application supports two complementary test scenarios with the bundled `MCD-1001.png`
sample image (which depicts **Filet-O-Fish** and **Cheesy Fries** on the tray):

#### Negative test case — order does not match tray contents

This example intentionally submits an order that does **not** match the items visible in
`MCD-1001.png`. It demonstrates how the application detects a mismatch and reports missing
and extra items. This is a valid production scenario: a customer receives the wrong items.

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={
    "order_id": "MCD-1001",
    "table_number": "T12",
    "restaurant": "McDonald'\''s",
    "items": [
      {"name": "Cheeseburger", "quantity": 1},
      {"name": "French Fries", "quantity": 1}
    ]
  }'
```

Expected response (`order_complete: false` — tray has Filet-O-Fish/Cheesy Fries, not Cheeseburger/French Fries):

```json
{
  "validation_id": "939d830a-8335-4fea-a564-0b83b93b71ab",
  "image_id": "MCD-1001",
  "order_complete": false,
  "accuracy_score": 0.0,
  "missing_items": [
    {"name": "Cheeseburger", "quantity": 1},
    {"name": "French Fries", "quantity": 1}
  ],
  "extra_items": [
    {"name": "Filet-O-Fish", "quantity": 1},
    {"name": "Cheesy Fries", "quantity": 1}
  ],
  "quantity_mismatches": [],
  "matched_items": [],
  "timestamp": "2026-03-20T14:53:05.543315",
  "metrics": {
    "end_to_end_latency_ms": 21854,
    "vlm_inference_ms": 21747,
    "agent_reconciliation_ms": 32,
    "cpu_utilization": 23.21,
    "gpu_utilization": 0.0,
    "memory_utilization": 78.07
  }
}
```

#### Positive test case — order matches tray contents

This example submits the correct items for `MCD-1001.png` and demonstrates a successful
order validation (`order_complete: true`).

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={
    "order_id": "MCD-1001",
    "table_number": "T12",
    "restaurant": "McDonald'\''s",
    "items": [
      {"name": "Filet-O-Fish", "quantity": 1},
      {"name": "Cheesy Fries", "quantity": 1}
    ]
  }'
```

Expected response (`order_complete: true` — tray contents match the submitted order):

```json
{
  "validation_id": "c459c9e5-3b48-462a-8c09-6360d4fd76fa",
  "image_id": "MCD-1001",
  "order_complete": true,
  "accuracy_score": 1.0,
  "missing_items": [],
  "extra_items": [],
  "quantity_mismatches": [],
  "matched_items": [
    {
      "expected_name": "Filet-O-Fish",
      "detected_name": "Filet-O-Fish",
      "similarity": 1.0,
      "quantity": 1
    },
    {
      "expected_name": "Cheesy Fries",
      "detected_name": "Cheesy Fries",
      "similarity": 1.0,
      "quantity": 1
    }
  ],
  "timestamp": "2026-03-20T14:55:31.025273",
  "metrics": {
    "end_to_end_latency_ms": 14475,
    "vlm_inference_ms": 14438,
    "agent_reconciliation_ms": 6,
    "cpu_utilization": 16.02,
    "gpu_utilization": 0.0,
    "memory_utilization": 78.27
  }
}
```

### Get Validation by ID

```bash
curl "http://localhost:8083/api/validate/26eba3f8-276b-44ac-b553-74419f84c1ad"
```

### List All Validations

```bash
curl "http://localhost:8083/api/validate"
```

### Health Check

```bash
curl "http://localhost:8083/health"
```

## Benchmarking

### Prerequisites

Before running benchmarks, initialize the performance-tools submodule:

```bash
make update-submodules
```

Optionally build the benchmark Docker image:

```bash
make build-benchmark
```

Or fetch from registry (if `REGISTRY=true`):

```bash
make fetch-benchmark
```

### Quick Single Image Test

For a quick validation test with curl:

> **Prerequisite:** Services must be running. Start them first with `make up`.

```bash
# IMAGE_ID must match an entry in configs/orders.json
# Available IDs: MCD-1001, MCD-1002, MCD-1003, MCD-1004
make benchmark-single IMAGE_ID=MCD-1001
```

Output:
```
=== Benchmark Results ===
{
  "validation_id": "...",
  "accuracy_score": 0.5,
  "metrics": {
    "vlm_inference_ms": 9003,
    "gpu_utilization": 100.0
  }
}
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
| `BENCHMARK_TARGET_LATENCY_MS` | 25000 | Target latency threshold (ms) |
| `BENCHMARK_LATENCY_METRIC` | avg | Metric: `avg`, `p95`, or `max` |
| `BENCHMARK_DENSITY_INCREMENT` | 1 | Concurrent images per iteration |
| `BENCHMARK_INIT_DURATION` | 60 | Warmup time (seconds) |
| `BENCHMARK_MIN_REQUESTS` | 3 | Min requests before measuring |
| `BENCHMARK_REQUEST_TIMEOUT` | 300 | Request timeout (seconds) |
| `TARGET_DEVICE` | GPU | Target device: CPU, GPU |
| `RESULTS_DIR` | results | Output directory |
| `REGISTRY` | false | Use registry images (true/false) |

Example:
```bash
make benchmark BENCHMARK_WORKERS=2 BENCHMARK_DURATION=600 TARGET_DEVICE=GPU
```

### Stream Density Test

Tests maximum concurrent validations within latency target.

```bash
make benchmark-stream-density
```

Output:
```
Target Latency: 15000ms
Max Density: 2 concurrent images

Iteration 1: 1 image  → 11726ms ✓ PASSED
Iteration 2: 2 images → 14808ms ✓ PASSED
Iteration 3: 3 images → 19509ms ✗ FAILED
```

### Metrics Processing

After running benchmarks, consolidate and visualize metrics:

```bash
# Consolidate metrics from multiple runs to CSV
make consolidate-metrics

# Generate plots from benchmark metrics
make plot-metrics
```

## Understanding Results

### Validation Status

| Field | Description |
|-------|-------------|
| `order_complete` | `true` if all items match with correct quantities |
| `accuracy_score` | 0.0-1.0 ratio of matched to expected items |
| `missing_items` | Items in order but not detected on plate |
| `extra_items` | Items detected but not in order |
| `quantity_mismatches` | Items with wrong quantities |
| `matched_items` | Successfully matched items with similarity scores |

### Metrics Interpretation

| Metric | Good Value | Warning |
|--------|------------|---------|
| `vlm_inference_ms` | < 10,000 | > 15,000 |
| `gpu_utilization` | 80-100% | < 50% (not using GPU) |
| `cpu_utilization` | 20-40% | > 80% |
| `memory_utilization` | < 80% | > 90% |

## Adding Custom Test Scenarios

### 1. Add Image

Place image in `images/` directory:
```bash
cp my_plate.jpg images/
```

### 2. Update Orders Config

Edit `configs/orders.json`:
```json
{
  "orders": [
    {
      "image_id": "my_plate",
      "restaurant": "My Restaurant",
      "table_number": "5",
      "items_ordered": [
        {"item": "Burger", "quantity": 1},
        {"item": "Fries", "quantity": 1}
      ]
    }
  ]
}
```

### 3. Restart Application

```bash
make down && make up
```

The new scenario appears in the Gradio dropdown.
