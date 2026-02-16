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
│   ├── app/                    # Frame selector code
│   ├── Dockerfile              # Service Dockerfile
│   └── requirements.txt        # Python dependencies
├── gradio-ui/                  # Web interface
│   ├── gradio_app.py           # Gradio application
│   ├── Dockerfile              # UI Dockerfile
│   └── requirements.txt        # Gradio dependencies
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

# Copy and configure environment
cp .env.example .env
# Edit .env as needed

# Build all images
make build

# Equivalent to:
docker compose -f docker-compose.yaml build
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
# Build with specific Python version
docker compose build \
    --build-arg PYTHON_VERSION=3.11 \
    order-accuracy

# Build with no cache (clean rebuild)
docker compose build --no-cache

# Build with progress output
docker compose build --progress=plain
```

### Build Verification

```bash
# List built images
docker images | grep oa_

# Expected output:
# oa_service         latest    abc123...   5 minutes ago   2.1GB
# oa_gradio          latest    def456...   5 minutes ago   1.8GB
# oa_frame_selector  latest    ghi789...   5 minutes ago   1.5GB
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

# Development dependencies
pip install -r requirements-dev.txt  # If available

# Or install editable mode
pip install -e .
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

### Download VLM Model

```bash
# Using provided script
./scripts/model-downloader.sh

# Or manually download from Hugging Face
mkdir -p models/vlm
cd models/vlm

# Option 1: Using huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct-ov-int8

# Option 2: Using git-lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct-ov-int8
```

### Convert Model to OpenVINO (if needed)

```bash
# Install OpenVINO tools
pip install openvino openvino-dev

# Convert from PyTorch
optimum-cli export openvino \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --task text-generation-with-past \
    --weight-format int8 \
    models/vlm/qwen2.5-vl-7b-ov-int8
```

### Model Directory Structure

```
models/
└── vlm/
    └── Qwen/
        └── Qwen2.5-VL-7B-Instruct-ov-int8/
            ├── config.json
            ├── openvino_model.bin
            ├── openvino_model.xml
            └── tokenizer/
                ├── tokenizer.json
                └── vocab.json
```

### YOLO Model (for Frame Selector)

```bash
# Download YOLOv8 model
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Move to models directory
mkdir -p frame-selector-service/models
mv yolov8n.pt frame-selector-service/models/
```

---

## Service-Specific Builds

### Order Accuracy Service

```dockerfile
# Dockerfile excerpt
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
WORKDIR /app

CMD ["python", "src/main.py"]
```

Build and test:

```bash
docker build -t oa_service:dev -f Dockerfile .
docker run --rm -p 8000:8000 oa_service:dev
```

### Frame Selector Service

```dockerfile
# frame-selector-service/Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ /app/
WORKDIR /app

CMD ["python", "frame_selector.py"]
```

Build and test:

```bash
docker build -t oa_frame_selector:dev -f frame-selector-service/Dockerfile frame-selector-service/
```

### Gradio UI

```dockerfile
# gradio-ui/Dockerfile
FROM python:3.10-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gradio_app.py /app/
WORKDIR /app

CMD ["python", "gradio_app.py"]
```

Build and test:

```bash
docker build -t oa_gradio:dev -f gradio-ui/Dockerfile gradio-ui/
docker run --rm -p 7860:7860 oa_gradio:dev
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
from src.core.ovms_client import OVMSClient
client = OVMSClient('http://localhost:8001')
print(client.health_check())
"

# Test VLM service
python -c "
from src.core.vlm_service import VLMService
service = VLMService()
print(service.test_inference())
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
docker save oa_service:latest | gzip > oa_service.tar.gz
docker save oa_gradio:latest | gzip > oa_gradio.tar.gz

# Load on another machine
docker load < oa_service.tar.gz
```

### Build Information

```bash
# Tag with version
docker build -t oa_service:v1.0.0 .
docker tag oa_service:v1.0.0 registry/oa_service:v1.0.0

# Add build labels
docker build \
    --label "version=1.0.0" \
    --label "build.date=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    -t oa_service:latest .
```
