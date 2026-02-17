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

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TAKE-AWAY ORDER ACCURACY                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
   ┌────▼────┐                  ┌─────▼─────┐                 ┌─────▼─────┐
   │ Gradio  │                  │   Order   │                 │  Frame    │
   │   UI    │◄────────────────►│ Accuracy  │                 │ Selector  │
   │ :7860   │                  │  Service  │                 │  (YOLO)   │
   └─────────┘                  │  :8000    │                 └─────┬─────┘
                                └─────┬─────┘                       │
           ┌─────────────────────────┬┴─────────────────────────┐   │
           │                        │                           │   │
      ┌────▼────┐             ┌─────▼─────┐              ┌──────▼───▼──┐
      │ Station │             │ Station   │              │    VLM      │
      │Worker 1 │             │ Worker N  │              │  Scheduler  │
      │(Process)│             │ (Process) │              │  (Batcher)  │
      └────┬────┘             └─────┬─────┘              └──────┬──────┘
           │                        │                          │
           └────────────────────────┼──────────────────────────┘
                                    │
                              ┌─────▼─────┐
                              │  OVMS VLM │
                              │  :8001    │
                              │(Qwen2.5-VL)
                              └─────┬─────┘
                                    │
           ┌────────────────────────┼────────────────────────┐
           │                        │                        │
      ┌────▼────┐             ┌─────▼─────┐           ┌──────▼──────┐
      │  MinIO  │             │ Semantic  │           │   RTSP      │
      │  :9000  │             │  Service  │           │  Streamer   │
      │ (S3)    │             │  :8080    │           │   :8554     │
      └─────────┘             └───────────┘           └─────────────┘
```

### Service Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Single** | Single worker with Gradio UI | Development, testing, demos |
| **Parallel** | Multi-worker with VLM scheduler | Production, high throughput |

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

### 2. Clone and Configure

```bash
cd ../take-away
cp .env.example .env
# Edit .env for your configuration
```

### 3. Build and Start

```bash
# Single worker mode (development)
make build
make up

# Parallel mode (production)
make build
make up-parallel WORKERS=4
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

## Key Commands

```bash
# Service Management
make up                    # Start services (single mode)
make up-parallel          # Start services (parallel mode)
make down                 # Stop all services
make status               # Show service status

# Logs
make logs                 # Order accuracy service logs
make logs-vlm             # OVMS VLM logs
make logs-all             # All service logs

# Benchmarking
make benchmark            # Single video benchmark
make benchmark-oa-density # Stream density test

# Development
make shell                # Shell into container
make test-api             # Test API endpoints
make show-config          # Show current configuration
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_MODE` | `single` | Service mode (`single`, `parallel`) |
| `WORKERS` | `0` | Number of station workers |
| `VLM_BACKEND` | `ovms` | VLM backend type |
| `OVMS_ENDPOINT` | `http://ovms-vlm:8000` | OVMS server endpoint |
| `OVMS_MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct-ov-int8` | Model name |
| `DEFAULT_MATCHING_STRATEGY` | `hybrid` | Matching strategy |
| `SIMILARITY_THRESHOLD` | `0.85` | Semantic similarity threshold |

### Benchmark Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_TARGET_LATENCY_MS` | `3000` | Target latency threshold |
| `BENCHMARK_MIN_TRANSACTIONS` | `3` | Minimum transactions per level |
| `BENCHMARK_WORKER_INCREMENT` | `1` | Workers added per iteration |

---

## Project Structure

```
take-away/
├── src/
│   ├── main.py                 # Service entry point
│   ├── api/
│   │   └── endpoints.py        # REST API endpoints
│   ├── core/
│   │   ├── vlm_service.py      # VLM processing logic
│   │   ├── ovms_client.py      # OVMS client
│   │   ├── validation_agent.py # Order validation
│   │   ├── semantic_client.py  # Semantic service client
│   │   └── semantic_matcher.py # Local semantic matching
│   └── parallel/
│       ├── station_worker.py   # Station worker process
│       ├── vlm_scheduler.py    # VLM request batcher
│       └── shared_queue.py     # Inter-process queue
├── frame-selector-service/
│   └── app/
│       └── frame_selector.py   # YOLO frame selection
├── gradio-ui/
│   └── gradio_app.py          # Web interface
├── config/                     # Configuration files
├── storage/                    # Videos and results
├── docker-compose.yaml         # Docker services
├── Makefile                    # Build automation
└── README.md                   # This file
```

---

## Related Projects

- **Dine-In Order Accuracy**: Image-based order validation for dining applications
- **Semantic Comparison Service**: Microservice for semantic text matching
- **Performance Tools**: Benchmarking scripts for stream density testing

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
