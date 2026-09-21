# Order Accuracy Dine-In

**Image-based Order Validation for Restaurant Dining Applications**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://docker.com)
[![OpenVINO](https://img.shields.io/badge/OpenVINO-2026.0-blue.svg)](https://docs.openvino.ai)

---

## Quick Start

### Prerequisites

- Docker 24.0+ with Compose V2
- Intel GPU with drivers installed
- 16 GB RAM minimum (64 GB recommended for production)
- Intel Xeon or equivalent CPU

> **ℹ iGPU / low-RAM systems:** 16 GB RAM is sufficient for **inference**. For first-time model export (`setup_models.sh`), a higher-memory host (48–64 GB) is recommended — export the models there and copy the `ovms-service/models/` directory to your 16 GB system. If exporting on 16 GB, set `export CACHE_SIZE=2` first to reduce KV cache to 2 GB (default is 4 GB). On iGPU platforms the KV cache uses system RAM. See [ovms-service/README.md](../ovms-service/README.md#tuning-the-kv-cache-size) for details.

### Setup Test Data (Required)

Before running the application, you must prepare your test data:

1. **Add Images**: Place your food tray images in the `images/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Images should clearly show the food items on the tray

2. **Update Orders**: Edit `configs/orders.json` with your test orders
   - Each order should have an `order_id` and list of `items`
   - `order_id` should match your `image_id`

3. **Update Inventory**: Edit `configs/inventory.json` to match your menu items
   - Define all possible food items that can appear in orders
   - Include item names, categories, and any relevant metadata

> **Note:** The `images/` folder does not contain sample images by default. You must add your own images before testing.

### 1. Configure the Environment

```bash
cd order-accuracy/dine-in
make init-env
# Edit .env if needed (defaults work for most setups)

# Initialize git submodules (for benchmark tools)
make update-submodules
```

### 2. Setup OVMS Model (First Time Only)

The VLM model must be exported before running. The script reads `dine-in/.env`, so complete step 1 first.

```bash
cd ../ovms-service
./setup_models.sh --app dine-in    # Downloads and exports model (~30-60 min first time)
cd ../dine-in
```

> **Note:** Only needed once. Model files are shared between Dine-In and Take-Away.

This step:

- Reads `OVMS_MODEL_NAME` from `dine-in/.env` (default: `openbmb/MiniCPM-V-4_5`)
- Downloads the selected model from HuggingFace
- Converts to OpenVINO™ INT8 format

If the selected model is gated on Hugging Face, add `HF_TOKEN=<your_token>` to `dine-in/.env` (or run `huggingface-cli login`). Public models such as `Qwen/Qwen2.5-VL-7B-Instruct` do not require authentication.

### 3. Build and Start

**Option A: Using Registry Images (default)**

```bash
make build && make up
```

**Option B: Build Locally from Source**

```bash
make up REGISTRY=false
```

| Image                          | Tag        |
| ------------------------------ | ---------- |
| `intel/order-accuracy-dine-in` | `2026.2.0-rc2` |

### 4. Access Services

| Service            | URL                        | Purpose                       |
| ------------------ | -------------------------- | ----------------------------- |
| Gradio UI          | http://localhost:7861      | Interactive order validation  |
| Order Accuracy API | http://localhost:8083      | REST API endpoints            |
| API Docs           | http://localhost:8083/docs | Swagger/OpenAPI documentation |
| OVMS VLM           | http://localhost:8002      | VLM model server              |
| MCP Server         | http://localhost:8011/mcp  | Agent-facing events, durable log, read tools (see below) |

---

## MCP Server (Events, Durable Log, Read Tools)

Order Accuracy is agentic-ready: it is a **sensor** (detect/report only, no
runtime actions) that exposes an [MCP](https://modelcontextprotocol.io)
server over `mcp-service-sdk`, alongside a durable, restart-safe event log.

**Events emitted** (one per completed plate validation, right after `/api/validate`
or `/api/validate/batch` returns a result):

| Event              | Trigger                                    | Payload                                                                 |
| ------------------ | ------------------------------------------- | ------------------------------------------------------------------------ |
| `order_validated`  | Order passes validation                    | `order_id`, `station`, `image_id`, `accuracy_score`                       |
| `order_failed`     | Order fails validation (missing/extra/qty) | above, plus `missing_items`, `extra_items`, `quantity_mismatches`, `reason` |

`station` is derived from the order's `table_number` (falls back to `"unknown"`
if not present). Every event is appended to a durable SQLite log (`MCP_LOG_PATH`,
default `/app/results/order_accuracy_events.db`) **before** any delivery is
attempted, is idempotent on `ref_id` (`{order_id}:{image_id}`), and survives
container restarts via the same `results/` volume already used for other
results. The container ships with one day of seeded baseline history
(tables `T1`/`T2`, ~25% fail rate) so comparative queries ("today vs.
baseline") work immediately after `make up` — mirroring Take-Away's
approach, seeded once on first run and never duplicated on restart.

Only the real validation endpoints (`/api/validate`, `/api/validate/batch`)
emit events. The separate benchmark/stream-density worker (`dinein-worker`)
intentionally does not, since it does not represent real order traffic.

**Read tools** (no action tools — Order Accuracy never acts):

| Tool                 | Purpose                                                            |
| --------------------- | -------------------------------------------------------------------- |
| `get_rework_rate`     | Rework rate for a period (`today`/`yesterday`/`all`/date), vs. a baseline period |
| `get_order_history`   | Order validation history, optionally filtered by station/order_id |
| `get_station_totals`  | Per-station pass/fail totals and rework rate                        |
| `describe`            | Self-description of event types, schemas and tools (SDK built-in)   |

**Try it yourself** (from a clone, no code reading required):

```bash
pip install fastmcp
python -c "
import asyncio
from fastmcp import Client

async def main():
    async with Client('http://localhost:8011/mcp') as client:
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
docker exec -it dinein_app python3 scripts/replay_log.py
docker exec -it dinein_app python3 scripts/replay_log.py --format json
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
| [Getting Started](../docs/user-guide/dine-in/get-started.md)                           | Installation and setup guide                                             |
| [System Requirements](../docs/user-guide/dine-in/get-started/system-requirements.md)   | Hardware/software requirements and pre-deployment checklist              |
| [System Architecture](../docs/user-guide/dine-in/how-it-works.md)                      | Architecture, design and component details of the Dine-In application.   |
| [How to Use](../docs/user-guide/dine-in/how-to-use.md)                                 | Usage instructions and workflows                                         |
| [Build from Source](../docs/user-guide/dine-in/get-started/build-from-source.md)       | Source build instructions                                                |
| [API Reference](../docs/user-guide/dine-in/api-reference.md)                           | Complete REST API documentation                                          |
| [Benchmarking Guide](../docs/user-guide/dine-in/di-benchmarking.md)                    | Performance testing guide                                                |
| [Troubleshooting](../docs/user-guide/dine-in/troubleshooting.md)                       | Common issues and resolutions                                            |
| [Release Notes](../docs/user-guide/dine-in/release-notes.md)                           | Version history and changes                                              |

## Support

For issues, questions, or contributions:

1. Review the [documentation](../docs/user-guide/dine-in/troubleshooting.md) and [release notes](../docs/user-guide/dine-in/release-notes.md)
2. Check existing [issues](https://github.com/intel-retail/order-accuracy/issues)
3. Submit a detailed bug report or feature request. See the [Support](../README.md#support) section of the main README for guidance.
