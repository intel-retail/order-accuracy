# How to Use the Application

Guide to using the Dine-In Order Accuracy application features.

## Gradio UI

Access the web interface at http://localhost:7861

### Interface Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Dine-In Order Accuracy Benchmark                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Scenario: [DD-6993 – McDonald's Table 12    ▼]            │
│                                                             │
│  ┌─────────────────────┐  ┌─────────────────────────────┐  │
│  │                     │  │ Order Manifest              │  │
│  │    [Plate Image]    │  │ ─────────────────           │  │
│  │                     │  │ items_ordered:              │  │
│  │                     │  │   - Maharaja Mac Chicken    │  │
│  │                     │  │   - Cheese                  │  │
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

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/DD-6993.jpg" \
  -F 'order={
    "order_id": "DD-6993",
    "table_number": "12",
    "restaurant": "McDonald'\''s",
    "items": [
      {"name": "Maharaja Mac Chicken", "quantity": 1},
      {"name": "Cheese", "quantity": 1}
    ]
  }'
```

### Response Format

```json
{
  "validation_id": "26eba3f8-276b-44ac-b553-74419f84c1ad",
  "image_id": "DD-6993",
  "order_complete": false,
  "accuracy_score": 0.5,
  "missing_items": [
    {"name": "Cheese", "quantity": 1}
  ],
  "extra_items": [],
  "quantity_mismatches": [],
  "matched_items": [
    {
      "expected_name": "Maharaja Mac Chicken",
      "detected_name": "Big Mac",
      "similarity": 0.85,
      "quantity": 1
    }
  ],
  "timestamp": "2026-02-16T16:36:50.278369",
  "metrics": {
    "end_to_end_latency_ms": 9003,
    "vlm_inference_ms": 9003,
    "agent_reconciliation_ms": 35,
    "cpu_utilization": 27.07,
    "gpu_utilization": 100.0,
    "memory_utilization": 79.99
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

### Single Validation Benchmark

```bash
make benchmark
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

### Stream Density Test

Tests maximum concurrent validations within latency target.

```bash
make benchmark-density
```

Output:
```
Target Latency: 15000ms
Max Density: 2 concurrent images

Iteration 1: 1 image  → 11726ms ✓ PASSED
Iteration 2: 2 images → 14808ms ✓ PASSED
Iteration 3: 3 images → 19509ms ✗ FAILED
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
