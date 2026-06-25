# How to Use

This guide covers daily operational use of the Take-Away Order Accuracy system.

---

## Operational Modes

### Single Worker Mode

Best for development, testing, and single-camera setups.

```bash
make up
```

Endpoints:

- Gradio UI: `http://localhost:7860`
- REST API: `http://localhost:8000`
- MinIO Console: `http://localhost:9001`

### Parallel Worker Mode

Best for multi-camera deployments. Pass `WORKERS` to set the number of RTSP stations.

```bash
make up-parallel WORKERS=3
```

This starts the `rtsp-streamer` service and enables the frame-selector service for all stations.

---

## Using the Gradio Web Interface

Access the Gradio UI at `http://localhost:7860`.

1. Upload a video file **or** enter an RTSP URL
2. Enter the expected order items (one per line)
3. Click **Validate Order**
4. Review matched, missing, and extra items in the results panel

---

## Using the REST API

Base URL: `http://localhost:8000`

### Health Check

```bash
curl http://localhost:8000/health
```

### Upload Video

Uploads a video file and starts the GStreamer processing pipeline:

```bash
curl -X POST http://localhost:8000/upload-video \
  -F "file=@video.mp4"
```

### Run From Any Source

Start processing from a file, RTSP stream, or other video source:

```bash
curl -X POST http://localhost:8000/run-video \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "rtsp",
    "source": "rtsp://camera:554/stream"
  }'
```

### Get Results

```bash
curl http://localhost:8000/results/<order_id>
curl http://localhost:8000/vlm/results
```

### Additional Endpoints

| Endpoint                      | Method | Description                           |
| ----------------------------- | ------ | ------------------------------------- |
| `/mode`                       | GET    | Current service mode and worker count |
| `/statistics`                 | GET    | Processing statistics                 |
| `/videos/history`             | GET    | All processed videos                  |
| `/videos/summary`             | GET    | Video processing summary              |
| `/videos/current`             | GET    | Currently processing video            |
| `/videos/{video_id}`          | GET    | Specific video details                |
| `/videos/{video_id}/complete` | POST   | Mark video complete                   |
| `/videos/{video_id}/fail`     | POST   | Mark video failed                     |
| `/videos/history`             | DELETE | Clear video history                   |
| `/orders/{order_id}/recall`   | GET    | Recall frames for an order            |
| `/orders/{order_id}/replay`   | GET    | Replay order processing               |

---

## Configuration Tuning

### Frame Selector (`config/application.yaml`)

```yaml
frame_selector:
  top_k: 3 # Frames selected per order
  min_frames_per_order: 1 # Minimum frames required before processing
  poll_interval_sec: 1.5 # MinIO polling interval (seconds)
  min_frames_before_finalize: 5 # Wait for N frames before finalizing
  inactivity_timeout_sec: 8 # Seconds of inactivity before order ends
```

### VLM (`config/application.yaml`)

```yaml
vlm:
  temperature: 0.2 # Lower = more deterministic (recommended)
  max_new_tokens: 100 # Token limit for VLM output
```

### RTSP Streams (parallel mode)

Configure station streams via the `RTSP_STREAMS` environment variable in `.env`:

```bash
RTSP_STREAMS=rtsp://camera1:554/stream,rtsp://camera2:554/stream
```

### Scaling Mode

```bash
# Fixed worker count (default, predictable performance)
SCALING_MODE=fixed

# Auto-scaling
SCALING_MODE=auto
```

---

## Monitoring and Logging

```bash
# Check all services
make status

# Test API health
make test-api

# View logs
make logs               # order-accuracy service
make logs-vlm           # OVMS VLM service
make logs-gradio        # Gradio UI
make logs-all           # All services combined
```

### Service Health Endpoints

```bash
curl http://localhost:8000/health       # order-accuracy API
curl http://localhost:8001/v1/config    # OVMS VLM
curl http://localhost:9001              # MinIO Console
```
