# Order Accuracy

**AI-Powered Order Validation Platform for Quick Service Restaurants**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2024.6%2B-blue.svg)](https://docs.openvino.ai)

---

## Overview

Order Accuracy is an enterprise AI vision platform that validates food orders in real-time using Vision Language Models (VLM). The platform automatically detects items in food trays, bags, or containers, compares them against expected order data, and identifies discrepancies before orders reach customers.

### Platform Applications

The platform provides two specialized applications optimized for different restaurant scenarios:

| Application | Use Case | Input Type |
|-------------|----------|------------|
| **[Dine-In](#dine-in-order-accuracy)** | Restaurant table service validation | Static images |
| **[Take-Away](#take-away-order-accuracy)** | Drive-through and counter service | Video streams (RTSP) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         ORDER ACCURACY PLATFORM                                  │
│                                                                                  │
│  ┌──────────────────────────────┐    ┌──────────────────────────────┐          │
│  │        DINE-IN               │    │        TAKE-AWAY             │          │
│  │   (Image-Based Validation)   │    │  (Video Stream Validation)   │          │
│  │                              │    │                              │          │
│  │  • Single image capture      │    │  • Real-time RTSP streams    │          │
│  │  • Tray/table validation     │    │  • Multi-station parallel    │          │
│  │  • REST API integration      │    │  • Frame selection (YOLO)    │          │
│  │  • Gradio web interface      │    │  • VLM request batching      │          │
│  └──────────────┬───────────────┘    └──────────────┬───────────────┘          │
│                 │                                   │                           │
│                 └─────────────┬─────────────────────┘                           │
│                               │                                                  │
│                               ▼                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐  │
│  │                        SHARED PLATFORM SERVICES                          │  │
│  │                                                                          │  │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌──────────┐ │  │
│  │  │  OVMS VLM   │    │  Semantic   │    │   MinIO     │    │  Gradio  │ │  │
│  │  │ (Qwen2.5-VL)│    │  Service    │    │  Storage    │    │    UI    │ │  │
│  │  │   :8001     │    │   :8080     │    │   :9000     │    │  :7860   │ │  │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └──────────┘ │  │
│  └──────────────────────────────────────────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dine-In Order Accuracy

**Image-based order validation for restaurant dining applications**

Optimized for validating food trays at serving stations before delivery to tables. Uses single image capture and VLM analysis for fast, accurate item detection.

### Key Features

- Single image capture and analysis
- Food tray/plate item detection
- REST API for POS integration
- Gradio web interface for manual validation
- Hybrid semantic matching

### Quick Start

```bash
# 1. Setup OVMS Model (first time only - takes 30-60 min)
cd ovms-service
./setup_models.sh

# 2. Build and start dine-in
cd ../dine-in
make build
make up
# Access UI at http://localhost:7861
```

> **Note**: The OVMS model setup only needs to be done once. Model files are shared between dine-in and take-away applications.

### Documentation

| Document | Description |
|----------|-------------|
| [Overview](dine-in/docs/user-guide/Overview.md) | Architecture and design |
| [Getting Started](dine-in/docs/user-guide/get-started.md) | Installation guide |
| [System Requirements](dine-in/docs/user-guide/system-requirements.md) | Hardware/software requirements |
| [How to Use](dine-in/docs/user-guide/how-to-use-application.md) | Usage instructions |
| [Build from Source](dine-in/docs/user-guide/how-to-build-from-source.md) | Build instructions |
| [Deploy with Helm](dine-in/docs/user-guide/deploy-with-helm.md) | Kubernetes deployment |
| [API Reference](dine-in/docs/user-guide/api-reference.md) | REST API documentation |
| [Release Notes](dine-in/docs/user-guide/release-notes.md) | Version history |

📖 **Full Documentation**: [dine-in/README.md](dine-in/README.md)

---

## Take-Away Order Accuracy

**Real-time video stream validation for drive-through and counter service**

Optimized for high-throughput drive-through environments with multiple camera stations. Processes RTSP video streams in parallel using intelligent frame selection and VLM batching.

### Key Features

- Real-time RTSP video stream processing
- Multi-station parallel processing (up to 8 workers)
- GStreamer-based video pipeline
- YOLO-powered frame selection
- VLM request batching for throughput
- Circuit breaker and auto-recovery

### Service Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Single** | Single worker, Gradio UI | Development, testing |
| **Parallel** | Multi-worker, VLM scheduler | Production deployment |

### Quick Start

```bash
cd take-away

# Single worker mode (development)
make build
make up

# Parallel mode (production)
make build
make up-parallel WORKERS=4

# Access UI at http://localhost:7860
```

### Documentation

| Document | Description |
|----------|-------------|
| [Overview](take-away/docs/user-guide/Overview.md) | Architecture and design |
| [Getting Started](take-away/docs/user-guide/get-started.md) | Installation guide |
| [System Requirements](take-away/docs/user-guide/system-requirements.md) | Hardware/software requirements |
| [How to Use](take-away/docs/user-guide/how-to-use-application.md) | Usage instructions |
| [Build from Source](take-away/docs/user-guide/how-to-build-from-source.md) | Build instructions |
| [API Reference](take-away/docs/user-guide/api-reference.md) | REST API documentation |
| [Benchmarking Guide](take-away/docs/user-guide/benchmarking-guide.md) | Performance testing |
| [Release Notes](take-away/docs/user-guide/release-notes.md) | Version history |

📖 **Full Documentation**: [take-away/README.md](take-away/README.md)

---

## Choosing the Right Application

| Criteria | Dine-In | Take-Away |
|----------|---------|-----------|
| **Input Type** | Static images | Video streams (RTSP) |
| **Throughput** | Low-medium | High (parallel) |
| **Latency Priority** | Accuracy over speed | Speed and accuracy |
| **Camera Setup** | Fixed position | Multi-station |
| **Typical Use** | Table service | Drive-through, counter |
| **Processing** | Single request | Batch processing |

### Recommendation

- **Choose Dine-In** if you need to validate orders from captured images at serving stations
- **Choose Take-Away** if you need real-time validation from continuous video streams

---

## Shared Platform Components

### VLM Backend (OVMS)

Both applications use OpenVINO Model Server with Qwen2.5-VL for vision-language inference:

```bash
# OVMS provides OpenAI-compatible API
curl http://localhost:8001/v3/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8",
    "messages": [...]
  }'
```

### Semantic Comparison Service

AI-powered semantic matching microservice for intelligent item comparison:

- **Matching Strategies**: Exact → Semantic → Hybrid
- **Example**: Matches "green apple" ↔ "apple" using semantic reasoning
- **Fallback**: Automatic fallback to local matching if service unavailable

### MinIO Storage

S3-compatible object storage for frames and results:

- **frames bucket**: Raw captured frames
- **selected bucket**: YOLO-selected frames
- **results bucket**: Validation results

Access MinIO Console: http://localhost:9001 (minioadmin/minioadmin)

---

## System Requirements

### Minimum Configuration

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon 8+ cores |
| RAM | 16 GB |
| GPU | Intel Arc A770 8GB / NVIDIA RTX 3060 |
| Storage | 50 GB SSD |
| Docker | 24.0+ with Compose V2 |

### Recommended Configuration

| Component | Specification |
|-----------|---------------|
| CPU | Intel Xeon 16+ cores |
| RAM | 32 GB |
| GPU | NVIDIA RTX 3080+ / Intel Data Center GPU |
| Storage | 200 GB NVMe SSD |
| Network | 10 Gbps (for Take-Away RTSP) |

---

## Project Structure

```
order-accuracy/
├── dine-in/                     # Dine-In application
│   ├── src/                     # Application source code
│   ├── docs/                    # Documentation
│   ├── docker-compose.yaml      # Service orchestration
│   ├── Makefile                 # Build automation
│   └── README.md                # Dine-In documentation
│
├── take-away/                   # Take-Away application
│   ├── src/                     # Application source code
│   ├── frame-selector-service/  # YOLO frame selection
│   ├── gradio-ui/               # Web interface
│   ├── docs/                    # Documentation
│   ├── docker-compose.yaml      # Service orchestration
│   ├── Makefile                 # Build automation
│   └── README.md                # Take-Away documentation
│
├── ovms-service/                # Shared OVMS configuration
├── performance-tools/           # Benchmarking scripts
├── config/                      # Shared configuration
└── README.md                    # This file
```

---

## Quick Reference

### Dine-In Commands

```bash
cd dine-in
make build                  # Build Docker images
make up                     # Start services
make down                   # Stop services
make logs                   # View logs
make update-submodules      # Initialize performance-tools (required before benchmarking)
make benchmark              # Run benchmark
make benchmark-density      # Run stream density test
make benchmark-density-results  # View density benchmark results
```

### Take-Away Commands

```bash
cd take-away
make build                  # Build Docker images
make up                     # Start (single mode)
make up-parallel WORKERS=4  # Start (parallel mode)
make down                   # Stop services
make logs                   # View logs
make update-submodules      # Initialize performance-tools (required before benchmarking)
make benchmark-oa-density   # Stream density test
```

---

## Service Endpoints

| Service | Port | URL |
|---------|------|-----|
| Order Accuracy API | 8000 | http://localhost:8000 |
| OVMS VLM | 8001 | http://localhost:8001 |
| Gradio UI | 7860 | http://localhost:7860 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 |
| Semantic Service | 8080 | http://localhost:8080 |

---

## License

Copyright © 2025 Intel Corporation

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

## Support

For application-specific issues, refer to the respective documentation:

- **Dine-In Issues**: See [dine-in/docs/](dine-in/docs/user-guide/)
- **Take-Away Issues**: See [take-away/docs/](take-away/docs/user-guide/)

For platform-wide issues or feature requests, submit an issue with:
1. Application name (dine-in/take-away)
2. Steps to reproduce
3. Expected vs actual behavior
4. Logs (`make logs`)
