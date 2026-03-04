# Take-Away Order Accuracy

**Real-time Order Validation System for Quick Service Restaurants**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.6%2B-blue.svg)](https://docs.openvino.ai)

---

## Overview

Take-Away Order Accuracy is an AI-powered vision system that validates drive-through and take-away orders in real-time using Vision Language Models (VLM). The system processes video feeds from multiple stations simultaneously, detecting items in order bags and validating them against expected orders.

### Key Capabilities

- **Real-Time Video Processing**: GStreamer-based pipeline with RTSP support
- **Multi-Station Parallel Processing**: Concurrent order validation across multiple stations
- **VLM-Based Item Detection**: Qwen2.5-VL-7B for visual product identification
- **Intelligent Frame Selection**: YOLO-powered frame selection for optimal VLM input
- **Semantic Matching**: Hybrid exact/semantic matching for robust item comparison
- **Production-Ready Architecture**: Circuit breaker, exponential backoff, health monitoring

---



### Service Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Single** | Single worker with Gradio UI | Development, testing, demos |
| **Parallel** | Multi-worker with VLM scheduler | Production, high throughput |

---

## Quick Start

### Prerequisites

- Docker 24.0+ with Compose V2
- Intel hardware (CPU, iGPU, dGPU, NPU)
- 32GB+ RAM recommended
- [Docker](https://docs.docker.com/engine/install/)
- [Make](https://www.gnu.org/software/make/) (`sudo apt install make`)
- **Python 3** (`sudo apt install python3`) - required for video download and validation scripts
- Sufficient disk space for models, videos, and results

### 1. Setup OVMS Model (First Time Only)

The VLM model must be exported before running the application:

```bash
cd order-accuracy/ovms-service
./setup_models.sh    # Export model (30-60 min first time)
```

This step:
- Downloads Qwen2.5-VL-7B-Instruct from HuggingFace (~7GB)
- Converts to OpenVINO format with INT8 quantization
- Creates model files in `ovms-service/models/`

> **Note**: This only needs to be done once. The model files are shared between dine-in and take-away applications.

### 2. Clone and Configure

```bash
cd ../take-away

# Initialize git submodules (for benchmark tools)
make update-submodules

cp .env.example .env
# Edit .env for your configuration
```

### 3. Build and Start

```bash
# Pull images from registry (default)
make build
make up

# OR build locally from source
make build REGISTRY=false
make up REGISTRY=false
```

### 4. Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| Gradio UI | http://localhost:7860 | Interactive order validation |
| Order Accuracy API | http://localhost:8000 | REST API endpoints |
| MinIO Console | http://localhost:9001 | Frame storage management |
| OVMS VLM | http://localhost:8001 | VLM model server |
| Semantic Service | http://localhost:8080 | Semantic matching API |

---

## Documentation

### User Guides

| Document | Description |
|----------|-------------|
| [Overview](docs/user-guide/Overview.md) | Comprehensive architecture and design |
| [Getting Started](docs/user-guide/get-started.md) | Installation and setup guide |
| [System Requirements](docs/user-guide/system-requirements.md) | Hardware and software requirements |
| [How to Use](docs/user-guide/how-to-use-application.md) | Usage instructions and workflows |
| [Build from Source](docs/user-guide/how-to-build-from-source.md) | Source build instructions |
| [API Reference](docs/user-guide/api-reference.md) | Complete REST API documentation |
| [Benchmarking Guide](docs/user-guide/benchmarking-guide.md) | Performance testing guide |
| [Release Notes](docs/user-guide/release-notes.md) | Version history and changes |

---

## Related Projects

- **Dine-In Order Accuracy**: Image-based order validation for dining applications
- **Semantic Comparison Service**: Microservice for semantic text matching
- **Performance Tools**: Benchmarking scripts for stream density testing (git submodule)

> **Note**: Performance tools are included as a git submodule. Run `make update-submodules` to initialize.

---

## License

Copyright © 2025 Intel Corporation

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Support

For issues, questions, or contributions:

1. Review the [documentation](docs/user-guide/)
2. Check existing [issues](issues/)
3. Submit a detailed bug report or feature request
