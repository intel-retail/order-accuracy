# Building Dine-In Order Accuracy from Source

Instructions for building Docker images locally instead of pulling from the registry.

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

```
dine-in/
├── src/
│   ├── app.py                        # Gradio UI
│   ├── api.py                        # FastAPI REST endpoints
│   ├── main.py                       # Launcher (API + Gradio)
│   ├── config.py                     # Configuration manager
│   └── services/
│       ├── vlm_client.py             # OVMS VLM client
│       ├── semantic_client.py        # Semantic matching client
│       ├── validation_service.py     # Validation orchestration
│       └── benchmark_service.py      # Benchmark mode
├── configs/
│   ├── orders.json                   # Test order manifests
│   └── inventory.json               # Known menu items
├── images/                          # Test plate images (user-supplied)
├── results/                         # Benchmark output
├── docker-compose.yaml
├── Dockerfile                       # python:3.13-slim base
├── Makefile
└── requirements.txt
```

---

## Building Images

### Build All Services

```bash
# Pull images from registry (default)
make build

# Build the dine-in image locally from source
make build REGISTRY=false
```

`make build REGISTRY=false` builds `intel/order-accuracy-dine-in:2026.1.0` from the local Dockerfile.

> `ovms-vlm`, `semantic-service`, and `metrics-collector` are always pulled from their registries — they have no local build context.

### Build a Custom Tag

```bash
make build REGISTRY=false TAG=my-custom-tag
```

### Rebuild Without Cache

```bash
docker compose build --no-cache
```

---

## Model Preparation

Before starting services, download and export the VLM model:

```bash
cd ../ovms-service
./setup_models.sh --app dine-in
cd ../dine-in
```

---

## Start Services

```bash
make up REGISTRY=false
```

Verify:

```bash
make test-api
curl http://localhost:8083/health
```

---

## Troubleshooting

### Build Fails (network / pip)

```bash
docker compose build --no-cache
```

### GPU Not Detected

```bash
sudo usermod -aG render $USER
# Log out and back in, then restart
make down && make up
```
