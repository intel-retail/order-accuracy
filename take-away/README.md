# Unified Order Accuracy Service

## 🎯 Architecture

Single codebase that supports two modes:

### **Mode 1: Single Worker** (FastAPI)
- Video upload via REST API
- One order at a time
- Gradio UI support
- Best for: Testing, development, single-user scenarios

### **Mode 2: Parallel Workers** (Multi-station)
- Multiple concurrent RTSP streams
- VLM request batching
- Autoscaling support
- Best for: Production, multi-camera deployments

## 📁 Directory Structure

```
order-accuracy-service/
├── main.py                    # Entry point with mode switching
├── Dockerfile                 # Unified Docker image
├── requirements.txt           # Python dependencies
├── api/                       # Single worker mode (FastAPI)
│   ├── __init__.py
│   └── endpoints.py          # REST endpoints
├── core/                      # Shared components
│   ├── pipeline_runner.py    # GStreamer pipeline
│   ├── frame_pipeline.py     # Frame processing + OCR
│   ├── validation_agent.py   # Order validation
│   ├── ovms_client.py        # OVMS VLM client
│   ├── semantic_client.py    # Semantic comparison
│   └── vlm_service.py        # VLM inference
├── parallel/                  # Parallel mode components
│   ├── station_manager.py    # Worker orchestration
│   ├── station_worker.py     # Worker process
│   ├── vlm_scheduler.py      # Request batching
│   ├── metrics_collector.py  # System metrics
│   └── scaling_policy.py     # Auto-scaling logic
└── config/                    # Configuration files
```

## 🚀 Quick Start

### Single Worker Mode (Default)

```bash
# 1. Copy environment file
cp .env.unified .env

# 2. Start with OVMS backend
docker compose --profile ovms -f docker-compose.unified.yaml up -d

# 3. Access Gradio UI
open http://localhost:7860
```

### Parallel Mode (2 Workers)

```bash
# 1. Set environment
cat > .env << 'EOF'
SERVICE_MODE=parallel
WORKERS=2
VLM_BACKEND=ovms
SCALING_MODE=fixed
EOF

# 2. Start services
docker compose --profile ovms -f docker-compose.unified.yaml up -d

# 3. Check logs
docker logs -f oa_service
```

### Parallel Mode with Autoscaling

```bash
# Environment
SERVICE_MODE=parallel
WORKERS=2
SCALING_MODE=auto
VLM_BACKEND=ovms

# Services will scale 1-8 workers based on:
# - GPU utilization
# - CPU usage
# - VLM request latency
```

## ⚙️ Configuration

### Environment Variables

| Variable | Values | Description |
|----------|--------|-------------|
| `SERVICE_MODE` | `single`, `parallel` | Operating mode |
| `WORKERS` | `1-8` | Number of workers (parallel mode) |
| `SCALING_MODE` | `fixed`, `auto` | Worker scaling policy |
| `VLM_BACKEND` | `embedded`, `ovms` | VLM inference backend |
| `OVMS_ENDPOINT` | URL | OVMS server address |
| `OVMS_MODEL_NAME` | String | Model name in OVMS |

## 🔧 Development

### Running Locally (Single Mode)

```bash
cd order-accuracy-service

# Install dependencies
pip install -r requirements.txt

# Set environment
export SERVICE_MODE=single
export VLM_BACKEND=ovms
export OVMS_ENDPOINT=http://localhost:8001

# Run
python main.py
```

### Running Locally (Parallel Mode)

```bash
export SERVICE_MODE=parallel
export WORKERS=2
export SCALING_MODE=fixed

python main.py
```

## 📊 Monitoring

### Single Mode
- FastAPI docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

### Parallel Mode
- Logs: `docker logs -f oa_service`
- Metrics: Monitor GPU/CPU/latency in logs
- Autoscaling events: Watch for "Scaling UP/DOWN" messages

## 🔄 Migration from Old Services

### Before (Separate Services)
```bash
# Two separate containers
docker compose up -d application-service  # Single mode
docker compose --profile parallel up -d   # Parallel mode
```

### After (Unified Service)
```bash
# One container, mode via env var
SERVICE_MODE=single docker compose -f docker-compose.unified.yaml up -d
SERVICE_MODE=parallel WORKERS=2 docker compose -f docker-compose.unified.yaml up -d
```

## ✅ Benefits

1. **Single Codebase** - Fix once, benefits both modes
2. **Consistent Behavior** - Same logic everywhere
3. **Easier Testing** - Test core components once
4. **Simpler Deployment** - One Docker image
5. **Flexible Scaling** - Change mode via environment variable
6. **Better Maintenance** - No code duplication

## 🧪 Testing

```bash
# Test single mode
docker compose -f docker-compose.unified.yaml up -d
curl http://localhost:8000/health

# Test parallel mode
SERVICE_MODE=parallel WORKERS=2 docker compose -f docker-compose.unified.yaml up -d
docker logs oa_service | grep "Started worker"
```

## 📝 Examples

### Example 1: Video Upload (Single Mode)
```bash
curl -X POST http://localhost:8000/upload-video \
  -F "file=@video.mp4"
```

### Example 2: RTSP Streams (Parallel Mode)
```bash
# Configured in parallel/config.py
# Automatically processes streams from:
# - rtsp://camera1:8554/stream
# - rtsp://camera2:8554/stream
```

### Example 3: Switching Modes
```bash
# Stop current
docker compose -f docker-compose.unified.yaml down

# Switch to parallel
SERVICE_MODE=parallel WORKERS=3 \
  docker compose -f docker-compose.unified.yaml up -d
```

## 🆘 Troubleshooting

### Issue: "Module not found"
```bash
# Fix: Ensure PYTHONPATH includes /app
export PYTHONPATH=/app:$PYTHONPATH
```

### Issue: OVMS not connecting
```bash
# Check OVMS is running
docker logs oa_ovms_vlm

# Verify endpoint
curl http://localhost:8001/v1/config
```

### Issue: Workers not starting
```bash
# Check logs
docker logs oa_service

# Verify RTSP sources are accessible
gst-launch-1.0 rtspsrc location=rtsp://... ! fakesink
```

## 🔐 Security

- Health checks on port 8000
- OVMS model validation
- Graceful shutdown handlers
- Resource limits via docker-compose

## 📚 Related Documentation

- [COPILOT_SETUP_GUIDE.md](../COPILOT_SETUP_GUIDE.md) - Complete setup guide
- [Original Application Service](../application-service/) - Legacy single mode
- [Original Parallel Pipeline](../parallel-pipeline/) - Legacy parallel mode
