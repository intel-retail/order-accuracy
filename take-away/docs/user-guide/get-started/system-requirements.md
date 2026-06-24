# System Requirements

Hardware, software, and network requirements for deploying Take-Away Order Accuracy.

---

## Hardware Requirements

### Development / Single Station

| Component   | Specification                                                              |
| ----------- | -------------------------------------------------------------------------- |
| **CPU**     | 8+ cores                                                                   |
| **RAM**     | 16 GB min; 64 GB recommended for production / heavy model export workloads |
| **GPU**     | Intel® Arc™ A770 (16 GB) or equivalent Intel GPU                           |
| **Storage** | 50 GB SSD                                                                  |

### Production / Multi-Station

| Component   | Specification                                                   |
| ----------- | --------------------------------------------------------------- |
| **CPU**     | 16+ cores                                                       |
| **RAM**     | 64 GB                                                           |
| **GPU**     | Intel® Data Center GPU Max (48 GB) — for 4+ concurrent stations |
| **Storage** | 200 GB NVMe SSD                                                 |

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

---

## Software Requirements

### Operating System

Ubuntu 22.04 LTS is the validated platform (matches the `intel/dlstreamer:2025.2.0-ubuntu22` base image).

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

| Service            | Port | Purpose                         |
| ------------------ | ---- | ------------------------------- |
| Order Accuracy API | 8000 | REST API                        |
| OVMS VLM           | 8001 | Model inference                 |
| Gradio UI          | 7860 | Web interface                   |
| MinIO API          | 9000 | S3-compatible storage           |
| MinIO Console      | 9001 | Storage admin UI                |
| Semantic Service   | 8080 | Semantic matching               |
| RTSP Streamer      | 8554 | Video streaming (parallel mode) |

### RTSP Requirements (Parallel Mode)

| Requirement | Specification     |
| ----------- | ----------------- |
| Protocol    | RTSP/RTP over TCP |
| Codec       | H.264             |
| Resolution  | 720p–1080p        |
| Frame Rate  | 15–30 FPS         |

---

## Pre-Deployment Checklist

- [ ] Docker and Docker Compose installed and working
- [ ] Intel GPU drivers installed and GPU visible to Docker
- [ ] Required ports available (8000, 7860, 8001, 9000, 9001, 8080)
- [ ] At least 50 GB free disk space
- [ ] **16 GB+ RAM available** (sufficient for inference; for first-time model export 48–64 GB recommended — export on a high-RAM host and copy `ovms-service/models/` to the target system)
- [ ] VLM model downloaded (`setup_models.sh` completed)
- [ ] `.env` file configured
- [ ] Camera RTSP URLs accessible from host (parallel mode)

## Next Steps

- [Get Started](../get-started.md) - Set up and run the application
- [API Reference](../api-reference.md) - REST endpoint documentation
- [How to Build](./build-from-source.md) - Build from source code
