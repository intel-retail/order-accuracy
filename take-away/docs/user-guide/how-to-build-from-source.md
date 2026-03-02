# Building Take-Away Order Accuracy from Source

This guide provides detailed instructions for building the Take-Away Order Accuracy system from source, including Docker images, local development setup, and model compilation.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Repository Structure](#repository-structure)
3. [Docker Build](#docker-build)
4. [Local Development Setup](#local-development-setup)
5. [Model Preparation](#model-preparation)
6. [Service-Specific Builds](#service-specific-builds)
7. [Development Workflow](#development-workflow)
8. [Troubleshooting Build Issues](#troubleshooting-build-issues)

---

## Prerequisites

### Build Tools

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y \
    build-essential \
    python3-dev \
    python3-pip \
    python3-venv \
    git \
    curl \
    wget

# Docker
sudo apt install -y docker.io docker-compose-v2
sudo usermod -aG docker $USER
```

### GPU Support (Optional)

```bash
# NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Python Environment

```bash
python3 --version  # Should be 3.10+
pip3 --version     # Should be 23.0+
```

---

## Repository Structure

```
take-away/
├── src/                        # Main application source
│   ├── main.py                 # Entry point
│   ├── api/                    # REST API endpoints
│   ├── core/                   # Core business logic
│   └── parallel/               # Parallel mode components
├── frame-selector-service/     # YOLO frame selection
│   ├── app/                    # Frame selector code + requirements.txt
│   └── dockerfile              # Service Dockerfile
├── gradio-ui/                  # Web interface
│   ├── gradio_app.py           # Gradio application
│   └── dockerfile              # UI Dockerfile (deps inline)
├── rtsp-streamer/              # RTSP video streamer (MediaMTX + FFmpeg)
│   ├── start.sh                # Entry point with 2PC sync
│   └── Dockerfile              # Multi-stage Alpine image
├── model/                      # YOLO models (pre-exported)
├── models/                     # EasyOCR models (from setup_models.sh)
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── docker-compose.yaml         # Service orchestration
├── Dockerfile                  # Main service Dockerfile
├── Makefile                    # Build automation
└── requirements.txt            # Python dependencies
```

---

## Docker Build

### Full Build

Build all Docker images:

```bash
# Navigate to project directory
cd take-away

# Create environment file
make init-env

# Pull images from registry (default)
make build

# OR build locally from source
make build REGISTRY=false
```

### Individual Service Build

```bash
# Build only the order-accuracy service
docker compose build order-accuracy

# Build only the frame-selector
docker compose build frame-selector

# Build only the gradio-ui
docker compose build gradio-ui
```

### Build Arguments

```bash
# Build with no cache (clean rebuild)
docker compose build --no-cache

# Build with progress output
docker compose build --progress=plain

# Pass proxy settings (handled automatically by docker-compose.yaml)
HTTP_PROXY=http://proxy:8080 make build REGISTRY=false
```

### Build Verification

```bash
# List built images
docker images | grep intel/order-accuracy

# Expected output:
# intel/order-accuracy-take-away           2026.0-rc1   ...   9.64GB
# intel/order-accuracy-frame-selector      2026.0-rc1   ...   1.96GB
# intel/order-accuracy-take-away-ui        2026.0-rc1   ...   1.3GB
# intel/order-accuracy-take-away-rtsp      2026.0-rc1   ...   227MB
```

---

## Local Development Setup

### Step 1: Create Virtual Environment

```bash
cd take-away

# Create virtual environment
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### Step 2: Install Dependencies

```bash
# Main application dependencies
pip install -r requirements.txt
```

### Step 3: Install GStreamer (for video processing)

```bash
# Ubuntu/Debian
sudo apt install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-gi \
    gir1.2-gst-plugins-base-1.0 \
    gir1.2-gstreamer-1.0
```

### Step 4: Set Environment Variables

```bash
# Create local .env or export directly
export SERVICE_MODE=single
export VLM_BACKEND=ovms
export OVMS_ENDPOINT=http://localhost:8001
export LOG_LEVEL=DEBUG
```

### Step 5: Run Locally

```bash
# Single mode
python src/main.py

# Or with uvicorn for development
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

---

## Model Preparation

### Download VLM and EasyOCR Models

Use the shared OVMS setup script (downloads the pre-converted OpenVINO INT8 model and EasyOCR models):

```bash
cd ../ovms-service
./setup_models.sh
```

This downloads:
- **Qwen2.5-VL-7B-Instruct-ov-int8** from HuggingFace (pre-converted OpenVINO model)
- **EasyOCR** detection and recognition models
- Generates `graph.pbtxt` from `config.json` graph_options

### Model Directory Structure

After running `setup_models.sh`:

```
ovms-service/models/
├── config.json                         # OVMS model server config
├── graph.pbtxt                         # Auto-generated mediapipe graph
└── Qwen/
    └── Qwen2.5-VL-7B-Instruct/
        ├── config.json
        ├── openvino_*.bin / .xml        # OpenVINO IR model files
        └── tokenizer/
```

EasyOCR models (mounted into order-accuracy container):

```
take-away/models/
└── easyocr/
    ├── craft_mlt_25k.pth               # Detection model
    └── english_g2.pth                  # Recognition model
```

### YOLO Model (for Frame Selector)

The YOLO model is pre-exported and mounted at runtime from the `model/` directory. No separate download is needed — the models are included in the repository:

```
model/
├── yolo11n.pt                      # YOLOv11 nano (PyTorch)
├── yolo11n_openvino_model/         # OpenVINO FP32 export
└── yolo11n_int8_openvino_model/    # OpenVINO INT8 quantized
```

> **Note**: The frame-selector container mounts `./model:/app/models` and uses the INT8 OpenVINO model for inference. If models are missing, the frame-selector will auto-export them on first startup (requires torch).

---

## Service-Specific Builds

### Order Accuracy Service

```dockerfile
# Dockerfile excerpt
FROM intel/dlstreamer:2025.2.0-ubuntu22

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata git curl ca-certificates jq \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/core/ /app/core/
COPY src/api/ /app/api/
COPY src/parallel/ /app/parallel/
COPY config/ /app/config/
COPY src/main.py src/frame_pipeline.py config_loader.py /app/

CMD ["python3", "/app/main.py"]
```

> **Note**: Uses `intel/dlstreamer:2025.2.0-ubuntu22` base image which includes GStreamer, DL Streamer plugins, and OpenVINO runtime.

Build and test:

```bash
docker compose build order-accuracy
docker compose up order-accuracy -d
```

### Frame Selector Service

```dockerfile
# frame-selector-service/dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY app/requirements.txt .
# Install CPU-only torch (ultralytics needs torch but inference uses OpenVINO)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY app /app

CMD ["python", "frame_selector.py"]
```

Build and test:

```bash
docker compose build frame-selector
```

### Gradio UI

```dockerfile
# gradio-ui/dockerfile
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies installed inline (no separate requirements.txt)
RUN pip install --no-cache-dir \
    numpy==1.24.3 pillow==10.0.0 \
    opencv-python-headless==4.8.1.78 \
    requests==2.31.0 gradio==3.50.2

COPY gradio_app.py /app/gradio_app.py

EXPOSE 7860
```

Build and test:

```bash
docker compose build gradio-ui
```

---

## Development Workflow

### Code Changes

```bash
# 1. Make code changes
vim src/core/vlm_service.py

# 2. Run linting
flake8 src/
black src/ --check
mypy src/

# 3. Run tests
pytest tests/ -v

# 4. Build and test locally
docker compose build order-accuracy
docker compose up order-accuracy -d
docker logs -f oa_service
```

### Hot Reload Development

```bash
# Mount source code for hot reload
docker compose -f docker-compose.yaml -f docker-compose.dev.yaml up

# docker-compose.dev.yaml
# services:
#   order-accuracy:
#     volumes:
#       - ./src:/app/src:ro
#     environment:
#       - DEBUG=true
```

### Testing Individual Components

```bash
# Test OVMS client
python -c "
from core.ovms_client import OVMSVLMClient
client = OVMSVLMClient(endpoint='http://localhost:8001')
print('OVMS client initialized')
"

# Test VLM component
python -c "
from core.vlm_service import VLMComponent
print('VLM component importable')
"
```

### Debugging

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debugger
python -m debugpy --listen 5678 --wait-for-client src/main.py

# Attach VS Code debugger (launch.json)
{
    "name": "Attach to Container",
    "type": "python",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
    }
}
```

---

## Troubleshooting Build Issues

### Docker Build Fails

**Issue**: `pip install` fails with network error

```bash
# Solution: Use pip cache mount
docker build --build-arg PIP_CACHE_DIR=/root/.cache/pip .

# Or disable cache
docker build --no-cache .
```

**Issue**: GStreamer not found

```bash
# Solution: Install gstreamer in Dockerfile
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    python3-gi \
    gir1.2-gstreamer-1.0
```

### Python Import Errors

**Issue**: `ModuleNotFoundError`

```bash
# Solution: Ensure PYTHONPATH is set
export PYTHONPATH=/app:$PYTHONPATH

# Or in Dockerfile
ENV PYTHONPATH="/app:${PYTHONPATH}"
```

### GPU Not Available

**Issue**: `CUDA not available`

```bash
# Solution: Use nvidia runtime
docker run --gpus all -it oa_service:dev

# Or in compose
services:
  order-accuracy:
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### Model Loading Fails

**Issue**: `Model file not found`

```bash
# Solution: Check volume mounts
docker compose config | grep -A5 volumes

# Verify model path
docker exec oa_service ls -la /app/models/vlm/
```

---

## Build Artifacts

### Export Docker Images

```bash
# Save images to tar
docker save intel/order-accuracy-take-away:2026.0-rc1 | gzip > order-accuracy.tar.gz
docker save intel/order-accuracy-frame-selector:2026.0-rc1 | gzip > frame-selector.tar.gz
docker save intel/order-accuracy-take-away-ui:2026.0-rc1 | gzip > gradio-ui.tar.gz
docker save intel/order-accuracy-take-away-rtsp:2026.0-rc1 | gzip > rtsp-streamer.tar.gz

# Load on another machine
docker load < order-accuracy.tar.gz
```

### Build Information

```bash
# Tag with version
docker tag intel/order-accuracy-take-away:2026.0-rc1 myregistry/order-accuracy:latest

# Push to custom registry
docker push myregistry/order-accuracy:latest
```
