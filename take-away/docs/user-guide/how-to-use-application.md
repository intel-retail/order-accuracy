# How to Use Take-Away Order Accuracy

This guide covers daily operational use of the Take-Away Order Accuracy system, including both the web interface and API integration patterns.

---

## Table of Contents

1. [Operational Modes](#operational-modes)
2. [Using the Gradio Web Interface](#using-the-gradio-web-interface)
3. [Using the REST API](#using-the-rest-api)
4. [Order Validation Workflow](#order-validation-workflow)
5. [Understanding Results](#understanding-results)
6. [Parallel Mode Operation](#parallel-mode-operation)
7. [Configuration Tuning](#configuration-tuning)
8. [Monitoring and Logging](#monitoring-and-logging)

---

## Operational Modes

### Single Worker Mode

Best for development, testing, and demonstrations.

```bash
# Start single worker mode (default)
make up
```

Features:
- Gradio web interface at http://localhost:7860
- REST API at http://localhost:8000
- Sequential order processing
- Video upload support

### Parallel Worker Mode

Best for production deployments with multiple camera stations.

```bash
# Start with 4 workers
make up-parallel WORKERS=4
```

Features:
- Multiple concurrent RTSP stream processing
- VLM request batching for throughput
- Auto-scaling (optional)
- Frame selection service active

---

## Using the Gradio Web Interface

Access the Gradio UI at http://localhost:7860

### 1. Video Upload Tab

Upload a pre-recorded video for order validation:

1. Click "Upload Video" button
2. Select MP4/AVI video file
3. Wait for upload completion
4. Video will be processed automatically

### 2. RTSP Stream Tab

Connect to a live camera feed:

1. Enter RTSP URL: `rtsp://camera-ip:554/stream`
2. Click "Connect"
3. Stream preview will appear
4. Use "Start Recording" to capture order

### 3. Order Entry

Enter expected order items:

```
Format: [quantity]x [item name]

Examples:
2x Burger
1x Fries (Large)
1x Coca-Cola
3x Chicken Nuggets (6pc)
```

### 4. Validation

1. Click "Validate Order" after video capture
2. System processes frames through VLM
3. Results display with matched/missing/extra items

### 5. Result Export

- Download results as JSON
- Save annotated frames
- Export validation report

---

## Using the REST API

### Base URL

```
http://localhost:8000
```

### Authentication

Currently no authentication required (configure for production).

### Common Endpoints

#### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "mode": "single",
  "version": "1.0.0"
}
```

#### Upload Video

Uploads a video file and automatically starts the GStreamer processing pipeline:

```bash
curl -X POST http://localhost:8000/upload-video \
  -F "file=@video.mp4"
```

Response:
```json
{
  "status": "started",
  "video_id": "<uuid>",
  "path": "/uploads/<uuid>_video.mp4",
  "filename": "video.mp4"
}
```

#### Run Video from Source

Start processing from any video source (file, RTSP, webcam, HTTP):

```bash
curl -X POST http://localhost:8000/run-video \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "rtsp",
    "source": "rtsp://camera:554/stream"
  }'
```

Response:
```json
{
  "status": "started",
  "video_id": "<uuid>",
  "source_type": "rtsp",
  "source": "rtsp://camera:554/stream"
}
```

#### Get Results

```bash
curl http://localhost:8000/results/<order_id>
```

#### Get All VLM Results

```bash
curl http://localhost:8000/vlm/results
```

#### Additional Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mode` | GET | Current service mode |
| `/statistics` | GET | Processing statistics |
| `/videos/history` | GET | All processed videos |
| `/videos/summary` | GET | Video processing summary |
| `/videos/current` | GET | Currently processing video |
| `/videos/{video_id}` | GET | Specific video details |
| `/videos/{video_id}/complete` | POST | Mark video complete |
| `/videos/{video_id}/fail` | POST | Mark video failed |
| `/videos/history` | DELETE | Clear video history |

---

## Order Validation Workflow

### Standard Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORDER VALIDATION WORKFLOW                             │
│                                                                              │
│  1. CAPTURE              2. SELECT              3. ANALYZE                  │
│  ┌──────────┐           ┌──────────┐           ┌──────────┐                │
│  │ Video/   │──────────▶│ Frame    │──────────▶│ VLM      │                │
│  │ RTSP     │           │ Selector │           │ Process  │                │
│  │ Upload   │           │ (YOLO)   │           │          │                │
│  └──────────┘           └──────────┘           └──────────┘                │
│                                                       │                      │
│  4. VALIDATE             5. REPORT                    │                      │
│  ┌──────────┐           ┌──────────┐                 │                      │
│  │ Compare  │◀──────────│ Detected │◀────────────────┘                      │
│  │ Orders   │           │ Items    │                                        │
│  └────┬─────┘           └──────────┘                                        │
│       │                                                                      │
│       ▼                                                                      │
│  ┌──────────┐                                                               │
│  │ Result:  │                                                               │
│  │ Match/   │                                                               │
│  │ Mismatch │                                                               │
│  └──────────┘                                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Processing Time Expectations

| Step | Single Mode | Parallel Mode |
|------|-------------|---------------|
| Video Upload | 2-5s | N/A |
| Frame Extraction | 1-3s | Real-time |
| Frame Selection | 1-2s | 0.5-1s |
| VLM Processing | 3-8s | 2-5s (batched) |
| Validation | <0.5s | <0.5s |
| **Total** | **7-18s** | **3-7s** |

---

## Understanding Results

### Result Structure

```json
{
  "order_id": "order_001",
  "station_id": "station_1",
  "timestamp": "2025-01-15T10:30:00Z",
  "validation": {
    "matched": [
      {
        "name": "burger",
        "expected_quantity": 2,
        "detected_quantity": 2,
        "match_type": "exact",
        "confidence": 0.95
      }
    ],
    "missing": [
      {
        "name": "fries",
        "expected_quantity": 1,
        "reason": "not_detected"
      }
    ],
    "extra": [
      {
        "name": "drink",
        "detected_quantity": 1,
        "reason": "not_in_order"
      }
    ],
    "quantity_mismatch": [
      {
        "name": "nuggets",
        "expected_quantity": 6,
        "detected_quantity": 4,
        "difference": -2
      }
    ]
  },
  "summary": {
    "total_expected": 4,
    "total_matched": 2,
    "total_missing": 1,
    "total_extra": 1,
    "accuracy_score": 0.50
  },
  "metadata": {
    "frames_processed": 3,
    "vlm_latency_ms": 3200,
    "matching_strategy": "hybrid"
  }
}
```

### Match Types

| Type | Description | Example |
|------|-------------|---------|
| `exact` | String exact match | "burger" == "burger" |
| `semantic` | Semantic similarity | "quarter pounder" ≈ "quarterpounder" |
| `fuzzy` | Approximate string | "chiken" ≈ "chicken" |

### Accuracy Score Calculation

```
accuracy = matched_items / total_expected_items

Where:
- matched_items = items correctly identified with correct quantity
- total_expected_items = sum of expected quantities
```

---

## Parallel Mode Operation

### Station Configuration

RTSP streams are configured via the `RTSP_STREAMS` environment variable (comma-separated):

```bash
# In .env
RTSP_STREAMS=rtsp://camera1:554/stream,rtsp://camera2:554/stream,rtsp://camera3:554/stream
```

The rtsp-streamer service can also loop local video files as RTSP streams for testing.

### Multi-Station Monitoring

```bash
# View all station logs
make logs

# View specific station
docker logs oa_service 2>&1 | grep "station_1"

# Check service health
curl http://localhost:8000/health
```

### Load Balancing

The VLM Scheduler automatically balances requests:

```
┌─────────────────────────────────────────────────────────────┐
│                  LOAD BALANCING                              │
│                                                              │
│  Station 1 ──┐                                               │
│              │     ┌────────────────┐     ┌────────────┐    │
│  Station 2 ──┼────▶│ VLM Scheduler  │────▶│ OVMS VLM   │    │
│              │     │ (Fair Queue)   │     │ (Batched)  │    │
│  Station 3 ──┘     └────────────────┘     └────────────┘    │
│                                                              │
│  • Round-robin scheduling                                    │
│  • 50-100ms batch windows                                    │
│  • Backpressure handling                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Configuration Tuning

### Matching Strategy

```bash
# Exact matching only (fastest)
DEFAULT_MATCHING_STRATEGY=exact

# Semantic matching only (most flexible)
DEFAULT_MATCHING_STRATEGY=semantic

# Hybrid (recommended - exact first, then semantic)
DEFAULT_MATCHING_STRATEGY=hybrid
```

### Similarity Threshold

```bash
# Lower threshold = more lenient matching (0.7)
SIMILARITY_THRESHOLD=0.70

# Higher threshold = stricter matching (0.9)
SIMILARITY_THRESHOLD=0.90

# Recommended balance (0.85)
SIMILARITY_THRESHOLD=0.85
```

### VLM Temperature

Configured in `config/application.yaml` under `vlm.temperature`:

```yaml
vlm:
  temperature: 0.2   # Lower = more deterministic (recommended)
```

### Frame Selection

Configured in `config/application.yaml` under `frame_selector`:

```yaml
frame_selector:
  top_k: 3                    # Frames selected per order
  min_frames_per_order: 1     # Minimum frames required
  poll_interval_sec: 1.5      # MinIO polling interval (seconds)
  min_frames_before_finalize: 5  # Wait for N frames before finalizing
  inactivity_timeout_sec: 8   # Seconds of inactivity before order ends
```

---

## Monitoring and Logging

### Log Levels

```bash
# Debug (verbose)
LOG_LEVEL=DEBUG

# Info (normal)
LOG_LEVEL=INFO

# Warning (minimal)
LOG_LEVEL=WARNING
```

### Key Log Messages

```
# Successful validation
INFO: [order_001] Validation complete: accuracy=0.95

# VLM processing
INFO: [order_001] VLM inference: latency=3200ms, tokens=1250

# Frame selection
INFO: [order_001] Selected 3/45 frames, best_score=0.87

# Circuit breaker trip
WARNING: [station_1] Circuit breaker opened after 5 failures

# Recovery
INFO: [station_1] Pipeline recovered, uptime reset
```

### Metrics Collection

```bash
# View VLM metrics
make benchmark-oa-metrics

# Output:
# VLM_LATENCY_MS=3245
# TOKEN_COUNT=1250
# BATCH_SIZE=2
# GPU_UTILIZATION=75%
```

### Health Monitoring

```bash
# Check all services
make test-api

# Check specific service
curl http://localhost:8000/health
curl http://localhost:8001/v1/config
curl http://localhost:8080/api/v1/health
```

---

## Best Practices

### For Production Deployment

1. **Use parallel mode** for multi-camera setups
2. **Set SCALING_MODE=fixed** for predictable performance
3. **Configure circuit breaker** for RTSP resilience
4. **Enable VLM batching** for throughput
5. **Monitor GPU utilization** to avoid OOM

### For Accuracy

1. **Use hybrid matching** for best results
2. **Set similarity threshold** based on item vocabulary
3. **Ensure good lighting** on cameras
4. **Position cameras** for clear bag visibility
5. **Test with representative orders** before deployment

### For Performance

1. **Use INT8 model** for faster inference
2. **Enable frame selection** to reduce VLM calls
3. **Configure appropriate batch size** for hardware
4. **Monitor and tune** based on actual workload
5. **Use GPU inference** whenever possible
