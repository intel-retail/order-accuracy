# Order Accuracy Dine-In

**Image-based Order Validation for Restaurant Dining Applications**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.6%2B-blue.svg)](https://docs.openvino.ai)

---

## Quick Start

### Prerequisites

- Docker 24.0+ with Compose V2
- NVIDIA GPU with 8GB+ VRAM (or Intel GPU)
- 32GB+ RAM recommended
- Intel Xeon or equivalent CPU

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

### 2. Build and Start

```bash
cd ../dine-in
make build
make up
```

### 3. Access Services

| Service | URL | Purpose |
|---------|-----|--------|
| Gradio UI | http://localhost:7860 | Interactive order validation |
| Order Accuracy API | http://localhost:8000 | REST API endpoints |
| OVMS VLM | http://localhost:8002 | VLM model server |

---

## Documentation

- **Overview**
  - [Overview](docs/user-guide/Overview.md): A high-level introduction.
  - [Overview Architecture](./docs/user-guide/Overview.md#how-it-works): Highlevel architecture.

- **Getting Started**
  - [Get Started](docs/user-guide/get-started.md): Step-by-step guide to get started with the sample application.
  - [System Requirements](docs/user-guide/system-requirements.md): Hardware and software requirements for running the sample application.
  - [How to Use the Application](./docs/user-guide/how-to-use-application.md): Explore the application's features and verify its functionality.

- **Deployment**
  - [How to Build from Source](docs/user-guide/how-to-build-from-source.md): Instructions for building from source code.
  - [How to Build using Helm](docs/user-guide/deploy-with-helm.md): Instructions for building using helm.


- **API Reference**
  - [API Reference](docs/user-guide/api-reference.md): Comprehensive reference for the available REST API endpoints.

- **Release Notes**
  - [Release Notes](docs/user-guide/release-notes.md): Information on the latest updates, improvements, and bug fixes.