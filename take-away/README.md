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
- **VLM-Based Item Detection**: MiniCPM-V-4.5 (INT4) for visual product identification
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

This downloads and exports:

- MiniCPM-V-4_5 INT4 (OpenVINO™ format)
- YOLOv11 model (INT8 OpenVINO™)
- EasyOCR detection and recognition models

> **Note:** Re-run this step any time you change `TARGET_DEVICE` in `.env`.

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
| MCP Server         | http://localhost:8010/mcp | Agent-facing events, durable log, read tools (see below) |
| MinIO Console      | http://localhost:9001 | Frame storage management     |
| OVMS VLM           | http://localhost:8001 | VLM model server             |
| Semantic Service   | http://localhost:8080 | Semantic matching API        |

---

## MCP Server (Events, Durable Log, Read Tools)

Order Accuracy is agentic-ready: it is a **sensor** (detect/report only, no
runtime actions) that exposes an [MCP](https://modelcontextprotocol.io)
server over `mcp-service-sdk`, alongside a durable, restart-safe event log.

**Events emitted** (one per completed order, right after validation):

| Event             | Trigger                                   | Payload                                                                                     |
| ------------------ | ------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `order_validated` | Order passes validation                   | `order_id`, `station`, `run_number`, `num_frames`, `inference_time_sec`                       |
| `order_failed`     | Order fails validation (missing/extra/qty) | above, plus `missing_items`, `extra_items`, `quantity_mismatch`, `reason`                      |

Every event is appended to a durable SQLite log (`MCP_LOG_PATH`, default
`/results/order_accuracy_events.db`) **before** any delivery is attempted, is
idempotent on `ref_id` (`{station}:{order_id}:{run_number}`), and survives
container restarts via the same `/results` volume already used for other
results. The container ships with one day of seeded baseline history so
comparative queries ("today vs. baseline") work immediately after `make up`
— this seeding is a demo/UX convenience that satisfies Issue #102 AC4
("ships preloaded with enough history that comparative behaviour works").
The Dine-In application ships the equivalent seeding, adapted to its own
event schema (see the [Dine-In README](../dine-in/README.md#mcp-server-events-durable-log-read-tools)).

**Read tools** (no action tools — Order Accuracy never acts):

| Tool                 | Purpose                                                        |
| --------------------- | --------------------------------------------------------------- |
| `get_rework_rate`     | Rework rate for a period (`today`/`yesterday`/`all`/date), vs. a baseline period |
| `get_order_history`   | Order validation history, optionally filtered by station/order_id |
| `get_station_totals`  | Per-station pass/fail totals and rework rate                   |
| `describe`            | Self-description of event types, schemas and tools (SDK built-in) |

**Try it yourself** (from a clone, no code reading required):

```bash
pip install fastmcp
python -c "
import asyncio
from fastmcp import Client

async def main():
    async with Client('http://localhost:8010/mcp') as client:
        print([t.name for t in await client.list_tools()])
        print(await client.call_tool('get_rework_rate', {'period': 'today'}))

asyncio.run(main())
"
```

Set `MCP_SERVICE_ENABLED=false` in `.env` to disable events and the MCP
server entirely for clean benchmark runs. See `.env.example` for all
`MCP_*` options (transport, host/port, log backend/path, optional webhook
delivery URL).

**Replaying a recorded day** (Issue #102 AC5 — "a recorded day replays
identically from the log"): a small developer/validation script,
`scripts/replay_log.py`, replays every event in the durable log in its
original order. It is a read-only diagnostic workflow — **not** an MCP
tool — since Order Accuracy's MCP surface is read/detect only:

```bash
make replay
# or directly:
docker exec -it oa_service python3 scripts/replay_log.py
docker exec -it oa_service python3 scripts/replay_log.py --format json
```

Replay only reads the existing SQLite log (via the SDK's own
`DurableLog.replay()`) and prints each event — it never calls `emit()`, so
it cannot create duplicate events, and it has no effect on delivery or
business logic. Running it twice against the same log always produces
identical output.

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
