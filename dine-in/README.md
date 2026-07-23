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

- Downloads Qwen2.5-VL-7B-Instruct from HuggingFace (~7 GB)
- Converts to OpenVINO™ INT8 format

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
| `intel/order-accuracy-dine-in` | `2026.1.0` |

### 4. Access Services

| Service            | URL                        | Purpose                       |
| ------------------ | -------------------------- | ----------------------------- |
| Gradio UI          | http://localhost:7861      | Interactive order validation  |
| Order Accuracy API | http://localhost:8083      | REST API endpoints            |
| API Docs           | http://localhost:8083/docs | Swagger/OpenAPI documentation |
| OVMS VLM           | http://localhost:8002      | VLM model server              |

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
