# Get Started

Quick start guide for the Dine-In Order Accuracy application.

## Prerequisites

- Docker Engine 24.0+
- Docker Compose 2.20+
- 16GB+ RAM
- Intel GPU (Arc/Flex recommended)

## Quick Start

### 1. Clone and Navigate

```bash
cd dine-in
```

### 2. Build Images

```bash
make build
```

### 3. Start Services

```bash
make up
```

This starts 4 containers:
- `dinein_app` - Main application (Gradio UI + API)
- `dinein_ovms_vlm` - Vision-Language Model server
- `dinein_semantic_service` - Semantic matching service
- `metrics-collector` - System metrics

### 4. Access the Application

| Interface | URL |
|-----------|-----|
| Gradio UI | http://localhost:7861 |
| REST API | http://localhost:8083 |
| API Docs | http://localhost:8083/docs |

### 5. Run Validation

**Via UI:**
1. Open http://localhost:7861
2. Select a scenario from dropdown
3. Click "Validate Plate"

**Via API:**
```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/DD-6993.jpg" \
  -F 'order={"order_id":"DD-6993","items":[{"name":"Maharaja Mac Chicken","quantity":1}]}'
```

## Running Benchmarks

### Single Validation

```bash
make benchmark
```

### Stream Density Test

```bash
make benchmark-density
```

## Stopping Services

```bash
make down
```

## Troubleshooting

### Services not starting

```bash
# Check container logs
make logs

# Restart services
make down && make up
```

### VLM inference slow

- Ensure GPU drivers are installed
- Check GPU utilization: `intel_gpu_top`
- Verify OVMS is using GPU in logs

### Out of memory

- Increase Docker memory limit
- Reduce cache size in `api.py`

## Next Steps

- [How to Use the Application](how-to-use-application.md)
- [API Reference](api-reference.md)
- [System Requirements](system-requirements.md)
