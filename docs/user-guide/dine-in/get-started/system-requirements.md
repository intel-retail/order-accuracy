# System Requirements

Hardware, software, and network requirements for deploying Dine-In Order Accuracy.

---

## Hardware Requirements

### Development / Single Station

| Component | Specification                                                              |
| --------- | -------------------------------------------------------------------------- |
| CPU       | 8+ cores                                                                   |
| RAM       | 16 GB min; 64 GB recommended for production / heavy model export workloads |
| GPU       | Intel® Arc™ A770 (16 GB) or equivalent Intel GPU                           |
| Storage   | 50 GB SSD                                                                  |

### Production / Multi-Station

| Component | Specification                                      |
| --------- | -------------------------------------------------- |
| CPU       | 16+ cores                                          |
| RAM       | 64 GB                                              |
| GPU       | Intel® Data Center GPU (for concurrent validation) |
| Storage   | 200 GB NVMe SSD                                    |

**GPU VRAM guidance:** The Qwen2.5-VL-7B INT8 model requires ~8 GB of VRAM.
The default `cache_size=4` reserves an additional 4 GB VRAM for the KV cache. Total VRAM needed
is around 12 GB, which fits in an Intel® Arc™ A770 16 GB. On **integrated GPU** (iGPU)
platforms such as Wildcat Lake and Meteor Lake, the KV cache is drawn from **system RAM**
instead of dedicated VRAM; in such a case, use a smaller value (e.g. `CACHE_SIZE=2`) to avoid
exhausting system RAM. Set `export CACHE_SIZE=<N>` before running `setup_models.sh`. For a
full per-platform sizing table and step-by-step instructions see [ovms-service/README.md — Tuning the KV Cache Size](https://github.com/intel-retail/order-accuracy/blob/main/ovms-service/README.md#tuning-the-kv-cache-size).

> **Model Export RAM Note:** 16 GB system RAM is sufficient for **inference-only**
> deployments. For first-time model export (`setup_models.sh` INT8 quantization), a
> higher-memory host (48–64 GB recommended) avoids potential OOM and corrupt IR files — export
> once there and copy `ovms-service/models/` to the target system. If you must export on 16 GB,
> set `export CACHE_SIZE=2` first. See [ovms-service/README.md — Tuning the KV Cache Size](https://github.com/intel-retail/order-accuracy/blob/main/ovms-service/README.md#tuning-the-kv-cache-size) for details.

## Software Requirements

### Operating System

Ubuntu 22.04 LTS is the validated platform (matches the `python:3.13-slim` base image running on the host GPU driver stack).

### Container Runtime

| Software       | Minimum Version |
| -------------- | --------------- |
| Docker Engine  | 24.0            |
| Docker Compose | V2 (2.20+)      |

### GPU Drivers

Intel GPU drivers must be installed from [packages.intel.com](https://packages.intel.com). Verify the GPU is accessible to Docker:

```bash
ls /dev/dri/
docker run --rm --device /dev/dri intel/openvino_dev:latest python3 -c \
  "from openvino.runtime import Core; print(Core().available_devices)"
```

Expected output includes `GPU`.

---

## Network Requirements

### Port Configuration

| Service           | Port | Purpose                      |
| ----------------- | ---- | ---------------------------- |
| Gradio UI         | 7861 | Web interface                |
| REST API          | 8083 | FastAPI endpoints            |
| OVMS VLM          | 8002 | Model inference (external)   |
| Semantic Service  | 8081 | Semantic matching (external) |
| Metrics Collector | 8084 | System metrics               |

---

## Pre-Deployment Checklist

- [ ] Docker and Docker Compose installed and working
- [ ] Intel GPU drivers installed and GPU visible to Docker
- [ ] Required ports available (7861, 8083, 8002, 8081, 8084)
- [ ] At least 50 GB free disk space
- [ ] **16 GB+ RAM available** (sufficient for inference; for first-time model export 48–64 GB recommended — export on a high-RAM host and copy `ovms-service/models/` to the target system)
- [ ] VLM model downloaded (`setup_models.sh` completed)
- [ ] `.env` file created (`make init-env`)
- [ ] Plate images placed in `images/` and `configs/orders.json` updated

---

## Next Steps

- [Get Started](../get-started.md) - Set up and run the application
- [API Reference](../api-reference.md) - REST endpoint documentation
- [How to Build](./build-from-source.md) - Build from source code
