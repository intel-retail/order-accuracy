# Order Accuracy

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.0-blue.svg)](https://docs.openvino.ai)

A suite of Intel® edge AI applications for real-time order validation in Quick Service Restaurant (QSR)
environments, powered by Vision Language Models (VLM) and Intel® OpenVINO™ inference.

Order Accuracy automatically detects items in food trays, bags, or containers, compares them
against expected order data, and identifies discrepancies before orders reach customers.

## Platform Applications

The platform provides two specialized applications optimized for different restaurant scenarios:

| Application                                | Use Case                            |
| ------------------------------------------ | ----------------------------------- |
| **[Dine-In](#dine-in-order-accuracy)**     | Restaurant table service validation |
| **[Take-Away](#take-away-order-accuracy)** | Drive-through and counter service   |

### Choosing the Right Application

| Criteria             | Dine-In             | Take-Away              |
| -------------------- | ------------------- | ---------------------- |
| **Input Type**       | Static images       | Video streams (RTSP)   |
| **Throughput**       | Low-medium          | High (parallel)        |
| **Latency Priority** | Accuracy over speed | Speed and accuracy     |
| **Camera Setup**     | Fixed position      | Multi-station          |
| **Typical Use**      | Table service       | Drive-through, counter |
| **Processing**       | Single request      | Batch processing       |

---

### Dine-In Order Accuracy

Image-based order validation for full-service restaurant expo operations. Staff
place a plate or tray in the validation station and trigger a check via the
Gradio UI or REST API. The system analyzes plate contents with a pre-trained
Qwen2.5-VL model and reconciles detected items against the order manifest —
no model fine-tuning required.

**Best for:** table service, fixed-position cameras, accuracy-prioritized
single-image validation.

For full details see the [Dine-In User Guide](./docs/user-guide/dine-in/index.md).
Alternatively, see the [Dine-In README](./dine-in/README.md) for a quick overview of the application.

### Take-Away Order Accuracy

Real-time video stream validation for drive-through and counter service. The
service ingests RTSP streams from multiple stations, selects relevant frames
with a YOLO model, batches VLM requests, and validates orders in parallel
across up to 8 workers — with circuit-breaker fault tolerance and automatic
recovery for high-throughput deployments.

**Best for:** drive-through and counter service, multi-station camera setups,
speed-and-accuracy continuous validation.

For full details see the [Take-Away User Guide](./docs/user-guide/take-away/index.md).
Alternatively, see the [Take-Away README](./take-away/README.md) for a quick overview of the application.

### Documentation

| Dine-In                                                         | Take-Away                                                         |
| --------------------------------------------------------------- | ----------------------------------------------------------------- |
| [Get Started](./docs/user-guide/dine-in/get-started.md)         | [Get Started](./docs/user-guide/take-away/get-started.md)         |
| [How It Works](./docs/user-guide/dine-in/how-it-works.md)       | [How It Works](./docs/user-guide/take-away/how-it-works.md)       |
| [How to Use](./docs/user-guide/dine-in/how-to-use.md)           | [How to Use](./docs/user-guide/take-away/how-to-use.md)           |
| [Benchmarking](./docs/user-guide/dine-in/di-benchmarking.md)    | [Benchmarking](./docs/user-guide/take-away/ta-benchmarking.md)    |
| [API Reference](./docs/user-guide/dine-in/api-reference.md)     | [API Reference](./docs/user-guide/take-away/api-reference.md)     |
| [Troubleshooting](./docs/user-guide/dine-in/troubleshooting.md) | [Troubleshooting](./docs/user-guide/take-away/troubleshooting.md) |
| [Release Notes](./docs/user-guide/dine-in/release-notes.md)     | [Release Notes](./docs/user-guide/take-away/release-notes.md)     |

---

### Shared Platform Services

Both applications share a common set of services:

| Service              | Description                                                              |
| -------------------- | ------------------------------------------------------------------------ |
| **OVMS VLM**         | OpenVINO™ Model Server running Qwen2.5-VL with an OpenAI-compatible API  |
| **Semantic Service** | AI-powered item matching (exact → semantic → hybrid) with local fallback |
| **MinIO Storage**    | S3-compatible object storage for frames, selections, and results         |
| **Gradio UI**        | Web interface for manual validation and demos                            |

The OVMS model files are exported once and shared by both applications. See the
[OVMS Service README](./ovms-service/README.md) for model setup and KV cache
tuning guidance.

## Get Started

**Choose an application** and follow its Get Started guide:

- [Dine-In — Get Started](./docs/user-guide/dine-in/get-started.md)
- [Take-Away — Get Started](./docs/user-guide/take-away/get-started.md)

For hardware and software prerequisites, see the system requirements for the
application you plan to deploy:

- [Dine-In — System Requirements](./docs/user-guide/dine-in/get-started/system-requirements.md)
- [Take-Away — System Requirements](./docs/user-guide/take-away/get-started/system-requirements.md)

> **⚠ Model Export RAM Requirement:** `setup_models.sh` performs INT8 quantization of Qwen2.5-VL-7B, which temporarily requires up to 40 GB of system RAM (FP16 model ~15 GB + INT8 compressed ~8 GB + calibration buffers ~8–15 GB). On platforms with 32 GB RAM (e.g. Wildcat Lake, Meteor Lake), the export OOMs and writes partial, corrupt XML files, causing the `oa_ovms_vlm` container to fail at startup with "Unable to read the model" errors. Always run `setup_models.sh` on a system with at least 48 GB RAM (64 GB recommended). The exported model files can then be copied to lower-memory systems for inference-only deployments.
>
> **ℹ OVMS KV Cache (`cache_size`):** The default `CACHE_SIZE=4` reserves 4 GB of VRAM for the KV cache. The INT8 model itself uses ~8 GB VRAM, so total VRAM ≈ 12 GB (fits in Intel Arc A770 16 GB). On **integrated GPU** platforms (Wildcat Lake, Meteor Lake), the KV cache is allocated from **system RAM** — on a 32 GB system this can exhaust all available memory. Use a smaller value (`CACHE_SIZE=2`) on iGPU platforms. Set `export CACHE_SIZE=<N>` before running `setup_models.sh`, or edit `graph.pbtxt` directly after export. See [OVMS Service README — Tuning the KV Cache Size](./ovms-service/README.md#tuning-the-kv-cache-size) for a full sizing guide and per-platform recommendations.

---

## Project Structure

```
order-accuracy/
├── dine-in/                     # Dine-In application
│   ├── src/                     # Application source code
│   ├── docker-compose.yaml      # Service orchestration
│   ├── Makefile                 # Build automation
│   └── README.md                # Dine-In documentation
│
├── docs/user-guide
│   ├── dine-in/                 # Dine-In user guide
│   └── take-away/               # Take-Away user guide
│
├── take-away/                   # Take-Away application
│   ├── src/                     # Application source code
│   ├── frame-selector-service/  # YOLO frame selection
│   ├── gradio-ui/               # Web interface
│   ├── docker-compose.yaml      # Service orchestration
│   ├── Makefile                 # Build automation
│   └── README.md                # Take-Away documentation
│
├── ovms-service/                # Shared OVMS configuration
├── performance-tools/           # Benchmarking scripts
└── README.md                    # This file
```

---

## License

Copyright © 2025 Intel Corporation

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

## Support

For application-specific issues, refer to the respective documentation:

- **Dine-In Issues**: See [Dine-In Troubleshooting](./docs/user-guide/dine-in/troubleshooting.md)
- **Take-Away Issues**: See [Take-Away Troubleshooting](./docs/user-guide/take-away/troubleshooting.md)

For platform-wide issues or feature requests, [submit an issue](https://github.com/intel-retail/order-accuracy/issues) with:

1. Application name (dine-in / take-away)
2. Steps to reproduce
3. Expected vs. actual behavior
4. Logs (`make logs`)
