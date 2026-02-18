# API Reference

Comprehensive REST API documentation for the Dine-In Order Accuracy service.

## Base URL

```
http://localhost:8083
```

## Authentication

Currently no authentication required (internal service).

---

## Endpoints

### Health Check

Check service health status.

```http
GET /health
```

**Response**

```json
{
  "status": "healthy",
  "timestamp": "2026-02-16T16:36:50.278369",
  "benchmark_mode": false
}
```

---

### Validate Plate

Validate a food plate image against an order manifest.

```http
POST /api/validate
```

**Request**

| Parameter | Type | Location | Description |
|-----------|------|----------|-------------|
| `image` | file | form-data | Plate image (JPEG/PNG) |
| `order` | JSON string | form-data | Order manifest |

**Order Schema**

```json
{
  "order_id": "string",
  "table_number": "string",
  "restaurant": "string",
  "items": [
    {
      "name": "string",
      "quantity": 1
    }
  ],
  "modifiers": []
}
```

**Example Request**

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@plate.jpg" \
  -F 'order={
    "order_id": "ORD-001",
    "table_number": "12",
    "restaurant": "Test Restaurant",
    "items": [
      {"name": "Burger", "quantity": 1},
      {"name": "Fries", "quantity": 1}
    ]
  }'
```

**Response**

```json
{
  "validation_id": "uuid-string",
  "image_id": "ORD-001",
  "order_complete": true,
  "accuracy_score": 1.0,
  "missing_items": [],
  "extra_items": [],
  "quantity_mismatches": [],
  "matched_items": [
    {
      "expected_name": "Burger",
      "detected_name": "Hamburger",
      "similarity": 0.92,
      "quantity": 1
    },
    {
      "expected_name": "Fries",
      "detected_name": "French Fries",
      "similarity": 0.95,
      "quantity": 1
    }
  ],
  "timestamp": "2026-02-16T16:36:50.278369",
  "metrics": {
    "end_to_end_latency_ms": 9500,
    "vlm_inference_ms": 9200,
    "agent_reconciliation_ms": 45,
    "within_operational_window": true,
    "cpu_utilization": 25.5,
    "gpu_utilization": 98.0,
    "memory_utilization": 78.2,
    "gpu_memory_utilization": 0.0
  }
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Validation successful |
| 400 | Invalid request (missing image or order) |
| 422 | Validation error (invalid order format) |
| 500 | Server error (VLM or semantic service failure) |

---

### Get Validation Result

Retrieve a validation result by ID.

```http
GET /api/validate/{validation_id}
```

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `validation_id` | string (path) | UUID of the validation |

**Example**

```bash
curl "http://localhost:8083/api/validate/26eba3f8-276b-44ac-b553-74419f84c1ad"
```

**Response**

Same schema as POST /api/validate response.

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Validation found |
| 404 | Validation not found |

---

### List Validations

Get all cached validation results.

```http
GET /api/validate
```

**Example**

```bash
curl "http://localhost:8083/api/validate"
```

**Response**

```json
[
  {
    "validation_id": "uuid-1",
    "image_id": "ORD-001",
    "order_complete": true,
    "accuracy_score": 1.0,
    ...
  },
  {
    "validation_id": "uuid-2",
    "image_id": "ORD-002",
    "order_complete": false,
    "accuracy_score": 0.5,
    ...
  }
]
```

---

### Delete Validation

Remove a validation from cache.

```http
DELETE /api/validate/{validation_id}
```

**Example**

```bash
curl -X DELETE "http://localhost:8083/api/validate/26eba3f8-276b-44ac-b553-74419f84c1ad"
```

**Response**

```json
{
  "message": "Validation 26eba3f8-276b-44ac-b553-74419f84c1ad deleted"
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Deleted successfully |
| 404 | Validation not found |

---

### Batch Validation

Validate multiple images in one request.

```http
POST /api/validate/batch
```

**Request**

| Parameter | Type | Description |
|-----------|------|-------------|
| `images` | file[] | Multiple plate images |
| `orders` | JSON string | Array of order manifests |

**Example**

```bash
curl -X POST "http://localhost:8083/api/validate/batch" \
  -F "images=@plate1.jpg" \
  -F "images=@plate2.jpg" \
  -F 'orders=[
    {"order_id": "ORD-001", "items": [{"name": "Burger", "quantity": 1}]},
    {"order_id": "ORD-002", "items": [{"name": "Pizza", "quantity": 1}]}
  ]'
```

**Response**

```json
{
  "batch_id": "batch-uuid",
  "results": [
    { "validation_id": "uuid-1", ... },
    { "validation_id": "uuid-2", ... }
  ],
  "summary": {
    "total": 2,
    "successful": 2,
    "failed": 0,
    "avg_latency_ms": 9800
  }
}
```

---

## Response Schemas

### ValidationResult

| Field | Type | Description |
|-------|------|-------------|
| `validation_id` | string | Unique identifier |
| `image_id` | string | Order/image identifier |
| `order_complete` | boolean | True if all items match |
| `accuracy_score` | float | 0.0-1.0 match ratio |
| `missing_items` | array | Items not detected |
| `extra_items` | array | Items detected but not ordered |
| `quantity_mismatches` | array | Quantity discrepancies |
| `matched_items` | array | Successfully matched items |
| `timestamp` | string | ISO 8601 timestamp |
| `metrics` | object | Performance metrics |

### Metrics

| Field | Type | Description |
|-------|------|-------------|
| `end_to_end_latency_ms` | int | Total request time |
| `vlm_inference_ms` | int | VLM processing time |
| `agent_reconciliation_ms` | int | Semantic matching time |
| `within_operational_window` | boolean | Under target latency |
| `cpu_utilization` | float | CPU usage percentage |
| `gpu_utilization` | float | GPU usage percentage |
| `memory_utilization` | float | Memory usage percentage |

---

## Error Responses

### 400 Bad Request

```json
{
  "detail": "Missing required field: image"
}
```

### 404 Not Found

```json
{
  "detail": "Validation with id '...' not found"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "order", "items"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 500 Internal Server Error

```json
{
  "detail": "Validation failed: Circuit breaker is OPEN for OVMS service"
}
```

---

## Interactive Documentation

- **Swagger UI**: http://localhost:8083/docs
- **ReDoc**: http://localhost:8083/redoc
- **OpenAPI JSON**: http://localhost:8083/openapi.json
