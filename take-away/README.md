# Take-Away Order Accuracy

**Real-time Order Validation System for Quick Service Restaurants**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.0-blue.svg)](https://docs.openvino.ai)

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

| Mode         | Description                     | Use Case                    |
| ------------ | ------------------------------- | --------------------------- |
| **Single**   | Single worker with Gradio UI    | Development, testing, demos |
| **Parallel** | Multi-worker with VLM scheduler | Production, high throughput |

---

## Quick Start

### Prerequisites

- Docker 24.0+ with Compose V2
- Intel hardware (CPU, iGPU, dGPU)
- 16 GB RAM minimum (64 GB recommended for production)
- [Docker](https://docs.docker.com/engine/install/)
- [Make](https://www.gnu.org/software/make/) (`sudo apt install make`)
- **Python 3** (`sudo apt install python3`) - required for video download and validation scripts
- Sufficient disk space for models, videos, and results

> **ℹ iGPU / low-RAM systems:** 16 GB RAM is sufficient for **inference**. For first-time model export (`setup_models.sh`), a higher-memory host (48–64 GB) is recommended — export the models there and copy the `ovms-service/models/` directory to your 16 GB system. If exporting on 16 GB, set `export CACHE_SIZE=2` first to reduce KV cache to 2 GB (default is 4 GB). On iGPU platforms the KV cache uses system RAM. See [ovms-service/README.md](../ovms-service/README.md#tuning-the-kv-cache-size) for details.

### 1. Configure

```bash
cd take-away

cp .env.example .env
# Edit .env — set TARGET_DEVICE, OPENVINO_DEVICE, and other settings

# Initialize git submodules (for benchmark tools)
make update-submodules
```

### 2. Setup OVMS Model (First Time Only)

The VLM model must be exported before running the application. The script reads `take-away/.env`, so complete step 1 first.

```bash
cd ../ovms-service
./setup_models.sh --app take-away    # Downloads and exports model (~30-60 min first time)
cd ../take-away
```

This step:

- Downloads Qwen2.5-VL-7B-Instruct from HuggingFace (~7 GB)
- Converts to OpenVINO™ INT8 format
- Downloads YOLO and EasyOCR models
- Creates model files in `ovms-service/models/` and `take-away/models/`

> **Note:** Only needed once. Model files are shared between Dine-In and Take-Away.

### 3. Build and Start

```bash
# Pull images from registry (default)
make build && make up

# OR build locally from source
make up REGISTRY=false
```

### 4. Access Services

| Service            | URL                   | Purpose                      |
| ------------------ | --------------------- | ---------------------------- |
| Gradio UI          | http://localhost:7860 | Interactive order validation |
| Order Accuracy API | http://localhost:8000 | REST API endpoints           |
| MinIO Console      | http://localhost:9001 | Frame storage management     |
| OVMS VLM           | http://localhost:8001 | VLM model server             |
| Semantic Service   | http://localhost:8080 | Semantic matching API        |

---

## Documentation

| Document                                                                               | Description                                                              |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ |
| [Getting Started](../docs/user-guide/take-away/get-started.md)                         | Installation and setup guide                                             |
| [System Requirements](../docs/user-guide/take-away/get-started/system-requirements.md) | Hardware/software requirements and pre-deployment checklist              |
| [System Architecture](../docs/user-guide/take-away/how-it-works.md)                    | Architecture, design and component details of the Take-Away application. |
| [How to Use](../docs/user-guide/take-away/how-to-use.md)                               | Usage instructions and workflows                                         |
| [Build from Source](../docs/user-guide/take-away/get-started/build-from-source.md)     | Source build instructions                                                |
| [API Reference](../docs/user-guide/take-away/api-reference.md)                         | Complete REST API documentation                                          |
| [Benchmarking Guide](../docs/user-guide/take-away/ta-benchmarking.md)                  | Performance testing guide                                                |
| [Troubleshooting](../docs/user-guide/take-away/troubleshooting.md)                     | Common issues and resolutions                                            |
| [Release Notes](../docs/user-guide/take-away/release-notes.md)                         | Version history and changes                                              |

---

## Related Projects

- **Dine-In Order Accuracy**: Image-based order validation for dining applications
- **Semantic Comparison Service**: Microservice for semantic text matching
- **Performance Tools**: Benchmarking scripts for stream density testing (git submodule)

> **Note:** Performance tools are included as a git submodule. Run `make update-submodules` to initialize.

---

## License

Copyright © 2026 Intel Corporation

Licensed under the Apache License, Version 2.0. See [LICENSE](../LICENSE) for details.

---

## Support

For issues, questions, or contributions:

1. Review the [documentation](../docs/user-guide/take-away/troubleshooting.md) and [release notes](../docs/user-guide/take-away/release-notes.md)
2. Check existing [issues](https://github.com/intel-retail/order-accuracy/issues)
3. Submit a detailed bug report or feature request. See the [Support](../README.md#support) section of the main README for guidance.
