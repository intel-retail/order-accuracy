# Build from Source

This guide covers building Docker images locally instead of pulling them from the registry.

---

## Prerequisites

- Docker 24.0+ and Docker Compose V2+
- Git

```bash
docker --version
docker compose version
```

---

## Repository Structure

```text
take-away/
├── src/                        # Main order-accuracy service source
├── frame-selector-service/     # YOLO frame selection service
├── gradio-ui/                  # Web UI
├── rtsp-streamer/              # RTSP video streamer
├── config/                     # Configuration files
├── models/                     # YOLO + EasyOCR models (from setup_models.sh)
├── docker-compose.yaml
├── Dockerfile                  # order-accuracy service
├── Makefile
└── requirements.txt
```

---

## Building Images

### Build All Services

```bash
# Pull images from registry (default)
make build

# Build all images locally from source
make build REGISTRY=false
```

This builds:

| Service        | Image                                          |
| -------------- | ---------------------------------------------- |
| order-accuracy | `intel/order-accuracy-take-away:2026.1.0`      |
| frame-selector | `intel/order-accuracy-frame-selector:2026.1.0` |
| gradio-ui      | `intel/order-accuracy-take-away-ui:2026.1.0`   |
| rtsp-streamer  | `intel/order-accuracy-take-away-rtsp:2026.1.0` |

> **Note:** `semantic-service` and OVMS (`openvino/model_server`) are always pulled — they have no local build context.

### Build a Single Service

```bash
docker compose build order-accuracy
docker compose build frame-selector
docker compose build gradio-ui
docker compose build rtsp-streamer
```

### Rebuild Without Cache

```bash
docker compose build --no-cache
```

---

## Model Preparation

Before starting services, download and export models:

```bash
cd ../ovms-service
./setup_models.sh
cd ../take-away
```

This downloads and exports:

- **Qwen2.5-VL-7B-Instruct** (OpenVINO INT8) → `ovms-service/models/`
- **EasyOCR** models → `take-away/models/easyocr/`
- **YOLO11n** (FP32 + INT8 OpenVINO) → `take-away/models/`

---

## Start Services

```bash
make up
```

Verify services are running:

```bash
make status
make test-api
```

---

## Troubleshooting

### Build Fails (network / pip)

```bash
docker compose build --no-cache
```

### Model File Not Found

```bash
# Verify models were correctly set up
ls ../ovms-service/models/
ls models/easyocr/
ls models/yolo11n_int8_openvino_model/
```

### GPU Not Detected

```bash
sudo usermod -aG render $USER
# Log out and back in, then restart services
make down && make up
```
