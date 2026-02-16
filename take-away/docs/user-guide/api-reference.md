# Take-Away Order Accuracy API Reference

Complete REST API documentation for the Take-Away Order Accuracy service.

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Base URL](#base-url)
4. [Health Endpoints](#health-endpoints)
5. [Video Processing Endpoints](#video-processing-endpoints)
6. [VLM Processing Endpoints](#vlm-processing-endpoints)
7. [Results Endpoints](#results-endpoints)
8. [Error Handling](#error-handling)
9. [Rate Limiting](#rate-limiting)

---

## Overview

The Order Accuracy API provides RESTful endpoints for video upload, VLM processing, and order validation. All responses are in JSON format.

### API Version

Current API version: `v1`

### Content Types

- Request: `application/json`, `multipart/form-data`
- Response: `application/json`

---

## Authentication

Currently, the API does not require authentication. For production deployments, configure API key or OAuth2 authentication.

### Future Authentication (Planned)

```bash
# API Key Header
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/...

# Bearer Token
curl -H "Authorization: Bearer your-token" http://localhost:8000/api/v1/...
```

---

## Base URL

| Environment | URL |
|-------------|-----|
| Development | `http://localhost:8000` |
| Docker Network | `http://oa_service:8000` |
| Production | `https://api.yourdomain.com` |

---

## Health Endpoints

### GET /health

Check service health status.

**Request:**
```bash
curl http://localhost:8000/health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "order-accuracy",
  "version": "1.0.0",
  "mode": "single",
  "uptime_seconds": 3600,
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| status | string | Health status: `healthy`, `degraded`, `unhealthy` |
| service | string | Service name |
| version | string | API version |
| mode | string | Service mode: `single`, `parallel` |
| uptime_seconds | integer | Seconds since service start |
| timestamp | string | ISO 8601 timestamp |

---

### GET /health/detailed

Get detailed health status including dependencies.

**Request:**
```bash
curl http://localhost:8000/health/detailed
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "service": "order-accuracy",
  "dependencies": {
    "ovms": {
      "status": "healthy",
      "latency_ms": 15,
      "endpoint": "http://ovms-vlm:8000"
    },
    "minio": {
      "status": "healthy",
      "latency_ms": 5,
      "endpoint": "minio:9000"
    },
    "semantic_service": {
      "status": "healthy",
      "latency_ms": 10,
      "endpoint": "http://oa_semantic_service:8080"
    }
  },
  "memory_mb": 512,
  "cpu_percent": 25.5
}
```

---

## Video Processing Endpoints

### POST /upload-video

Upload a video file for processing.

**Request:**
```bash
curl -X POST http://localhost:8000/upload-video \
  -F "file=@video.mp4" \
  -F "video_id=order_001" \
  -F "station_id=station_1"
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| file | file | Yes | Video file (MP4, AVI, MOV) |
| video_id | string | Yes | Unique identifier for the video |
| station_id | string | No | Station identifier (default: station_1) |

**Response (200 OK):**
```json
{
  "status": "success",
  "video_id": "order_001",
  "station_id": "station_1",
  "message": "Video uploaded successfully",
  "file_size_bytes": 15728640,
  "frames_estimated": 450,
  "upload_time_ms": 1250
}
```

**Error Responses:**

| Status | Description |
|--------|-------------|
| 400 | Invalid file format or missing parameters |
| 413 | File too large (max 500MB) |
| 500 | Server error during upload |

---

### POST /run-video

Process an uploaded video with VLM inference.

**Request:**
```bash
curl -X POST http://localhost:8000/run-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "order_001",
    "expected_items": [
      {"name": "burger", "quantity": 2},
      {"name": "fries", "quantity": 1},
      {"name": "drink", "quantity": 1}
    ],
    "options": {
      "matching_strategy": "hybrid",
      "similarity_threshold": 0.85
    }
  }'
```

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| video_id | string | Yes | Video ID from upload |
| expected_items | array | Yes | Array of expected order items |
| expected_items[].name | string | Yes | Item name |
| expected_items[].quantity | integer | Yes | Expected quantity |
| options | object | No | Processing options |
| options.matching_strategy | string | No | `exact`, `semantic`, `hybrid` |
| options.similarity_threshold | float | No | 0.0-1.0 (default: 0.85) |

**Response (202 Accepted):**
```json
{
  "status": "processing",
  "video_id": "order_001",
  "task_id": "task_abc123def456",
  "estimated_time_ms": 5000,
  "message": "Video queued for processing"
}
```

---

## VLM Processing Endpoints

### POST /run_vlm

Direct VLM inference on an image.

**Request:**
```bash
curl -X POST http://localhost:8000/run_vlm \
  -F "image=@frame.jpg" \
  -F "prompt=Identify all food items in this image"
```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | file | Yes | Image file (JPEG, PNG) |
| prompt | string | No | Custom prompt (default: item detection) |

**Response (200 OK):**
```json
{
  "status": "success",
  "detected_items": [
    {"name": "burger", "quantity": 2, "confidence": 0.95},
    {"name": "fries", "quantity": 1, "confidence": 0.88}
  ],
  "raw_response": "...",
  "processing_time_ms": 2850,
  "token_usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 85,
    "total_tokens": 1335
  }
}
```

---

### GET /vlm/results

Get VLM processing results.

**Request:**
```bash
curl "http://localhost:8000/vlm/results?video_id=order_001"
```

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| video_id | string | Yes | Video ID to retrieve results |

**Response (200 OK):**
```json
{
  "video_id": "order_001",
  "status": "completed",
  "frames_processed": 3,
  "detections": [
    {
      "frame_id": 1,
      "items": [
        {"name": "burger", "quantity": 2}
      ]
    },
    {
      "frame_id": 2,
      "items": [
        {"name": "burger", "quantity": 2},
        {"name": "fries", "quantity": 1}
      ]
    }
  ],
  "aggregated_detection": [
    {"name": "burger", "quantity": 2, "confidence": 0.95},
    {"name": "fries", "quantity": 1, "confidence": 0.88}
  ]
}
```

---

## Results Endpoints

### GET /results/{order_id}

Get validation results for an order.

**Request:**
```bash
curl http://localhost:8000/results/order_001
```

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| order_id | string | Order/video identifier |

**Response (200 OK):**
```json
{
  "order_id": "order_001",
  "station_id": "station_1",
  "status": "completed",
  "timestamp": "2025-01-15T10:35:00Z",
  "validation": {
    "matched": [
      {
        "name": "burger",
        "expected_quantity": 2,
        "detected_quantity": 2,
        "match_type": "exact",
        "confidence": 0.95
      },
      {
        "name": "fries",
        "expected_quantity": 1,
        "detected_quantity": 1,
        "match_type": "semantic",
        "confidence": 0.88,
        "detected_as": "french fries"
      }
    ],
    "missing": [],
    "extra": [
      {
        "name": "ketchup packet",
        "detected_quantity": 2,
        "reason": "not_in_order"
      }
    ],
    "quantity_mismatch": []
  },
  "summary": {
    "total_expected_items": 2,
    "total_expected_quantity": 3,
    "matched_items": 2,
    "matched_quantity": 3,
    "missing_items": 0,
    "extra_items": 1,
    "accuracy_score": 1.0,
    "order_complete": true
  },
  "metadata": {
    "video_duration_sec": 15,
    "frames_extracted": 450,
    "frames_selected": 3,
    "vlm_latency_ms": 3200,
    "total_processing_ms": 5500,
    "matching_strategy": "hybrid"
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| matched | array | Items correctly found |
| missing | array | Expected items not found |
| extra | array | Found items not in order |
| quantity_mismatch | array | Items with wrong quantity |
| accuracy_score | float | 0.0-1.0 validation score |

---

### GET /results/station/{station_id}

Get recent results for a station.

**Request:**
```bash
curl "http://localhost:8000/results/station/station_1?limit=10"
```

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| limit | integer | 10 | Max results to return |
| offset | integer | 0 | Pagination offset |
| since | string | - | ISO 8601 timestamp filter |

**Response (200 OK):**
```json
{
  "station_id": "station_1",
  "results": [
    {
      "order_id": "order_003",
      "timestamp": "2025-01-15T10:35:00Z",
      "accuracy_score": 1.0,
      "order_complete": true
    },
    {
      "order_id": "order_002",
      "timestamp": "2025-01-15T10:30:00Z",
      "accuracy_score": 0.67,
      "order_complete": false
    }
  ],
  "pagination": {
    "limit": 10,
    "offset": 0,
    "total": 45
  }
}
```

---

## Error Handling

### Error Response Format

All errors follow a standard format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid video format",
    "details": {
      "supported_formats": ["mp4", "avi", "mov"],
      "received": "webm"
    },
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_abc123"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| VALIDATION_ERROR | 400 | Invalid request parameters |
| NOT_FOUND | 404 | Resource not found |
| FILE_TOO_LARGE | 413 | Upload exceeds size limit |
| PROCESSING_ERROR | 500 | VLM or internal error |
| SERVICE_UNAVAILABLE | 503 | Dependency unavailable |
| TIMEOUT | 504 | Processing timeout |

### Common Error Responses

**400 Bad Request:**
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Missing required field: expected_items"
  }
}
```

**404 Not Found:**
```json
{
  "error": {
    "code": "NOT_FOUND",
    "message": "Video order_001 not found"
  }
}
```

**500 Internal Server Error:**
```json
{
  "error": {
    "code": "PROCESSING_ERROR",
    "message": "VLM inference failed",
    "details": {
      "ovms_status": "timeout"
    }
  }
}
```

---

## Rate Limiting

### Default Limits

| Endpoint | Limit | Window |
|----------|-------|--------|
| /upload-video | 10 | per minute |
| /run-video | 20 | per minute |
| /run_vlm | 30 | per minute |
| /results/* | 100 | per minute |
| /health | unlimited | - |

### Rate Limit Headers

```http
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 15
X-RateLimit-Reset: 1705312800
```

### Rate Limit Exceeded

**Response (429 Too Many Requests):**
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 45 seconds.",
    "retry_after": 45
  }
}
```

---

## OpenAPI Specification

Access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## SDK Examples

### Python

```python
import requests

class OrderAccuracyClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def upload_video(self, file_path, video_id):
        with open(file_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/upload-video",
                files={"file": f},
                data={"video_id": video_id}
            )
        return response.json()
    
    def run_validation(self, video_id, expected_items):
        response = requests.post(
            f"{self.base_url}/run-video",
            json={
                "video_id": video_id,
                "expected_items": expected_items
            }
        )
        return response.json()
    
    def get_results(self, order_id):
        response = requests.get(f"{self.base_url}/results/{order_id}")
        return response.json()

# Usage
client = OrderAccuracyClient()
client.upload_video("order.mp4", "order_001")
client.run_validation("order_001", [{"name": "burger", "quantity": 2}])
results = client.get_results("order_001")
print(results)
```

### cURL

```bash
# Complete workflow
VIDEO_ID="order_$(date +%s)"

# 1. Upload
curl -X POST http://localhost:8000/upload-video \
  -F "file=@video.mp4" \
  -F "video_id=${VIDEO_ID}"

# 2. Process
curl -X POST http://localhost:8000/run-video \
  -H "Content-Type: application/json" \
  -d '{
    "video_id": "'${VIDEO_ID}'",
    "expected_items": [{"name": "burger", "quantity": 2}]
  }'

# 3. Wait and get results
sleep 10
curl http://localhost:8000/results/${VIDEO_ID}
```
