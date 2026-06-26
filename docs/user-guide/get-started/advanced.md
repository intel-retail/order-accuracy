# Advanced Settings

## System Requirements

### Minimum Configuration

| Component | Specification         |
| --------- | --------------------- |
| CPU       | 8+ cores              |
| RAM       | 16 GB                 |
| GPU       | Intel© Arc™ A770 8GB  |
| Storage   | 50 GB SSD             |
| Docker    | 24.0+ with Compose V2 |

### Recommended Configuration

| Component | Specification                |
| --------- | ---------------------------- |
| CPU       | 16+ cores                    |
| RAM       | 64 GB                        |
| GPU       | Intel© Data Center GPU       |
| Storage   | 200 GB NVMe SSD              |
| Network   | 10 Gbps (for Take-Away RTSP) |

## Build Local Image

By default, the application uses pre-built Docker images for faster setup. If you need to build images locally (for customization or development):

```bash
# Build and run locally instead of using pre-built images
make build REGISTRY=false
make up

# Examples for both applications:
# Dine-In
cd dine-in && make build REGISTRY=false && make up

# Take-Away
cd take-away && make build REGISTRY=false && make up
```

**When to use local building:**

- Modifying source code or configurations
- Development and testing changes
- Air-gapped environments without internet access
- Custom hardware optimizations

> **Note:** Local building takes significantly longer (15-30 minutes) compared to pre-built images (2-5 minutes).

---

## Configuration Options

### Dine-In Configuration

#### Environment Configuration (.env)

```bash
# =============================================================================
# Logging
# =============================================================================
LOG_LEVEL=INFO

# =============================================================================
# Service Endpoints
# =============================================================================
OVMS_ENDPOINT=http://ovms-vlm:8000
OVMS_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
SEMANTIC_SERVICE_ENDPOINT=http://semantic-service:8080
API_TIMEOUT=60
```

#### Test Data Configuration

1. **Add Images**: Place food tray/plate images in `images/` folder
   - Supported formats: `.jpg`, `.jpeg`, `.png`
   - Images should clearly show food items on the plate

2. **Update Orders**: Edit `configs/orders.json` with test orders
   - Each order needs `image_id` and list of `items_ordered`
   - Image IDs should match filenames (without extension)

3. **Update Inventory**: Edit `configs/inventory.json` with menu items
   - Define all possible food items
   - Include item names, categories, and metadata

#### Dine-In Docker Services

| Container                 | Ports      | Description                         |
| ------------------------- | ---------- | ----------------------------------- |
| `dinein_app`              | 7861, 8083 | Main application (Gradio + FastAPI) |
| `dinein_ovms_vlm`         | 8002       | Vision-Language Model server        |
| `dinein_semantic_service` | 8081, 9091 | Semantic text matching              |
| `metrics-collector`       | 8084       | System metrics aggregation          |

---

### Take-Away Configuration

#### Environment Configuration (.env)

```bash
# =============================================================================
# VLM Backend
# =============================================================================
VLM_BACKEND=ovms
OVMS_ENDPOINT=http://ovms-vlm:8000
OVMS_MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
OPENVINO_DEVICE=GPU          # 'GPU', 'CPU', or 'AUTO'

# =============================================================================
# Semantic Service
# =============================================================================
SEMANTIC_VLM_BACKEND=ovms
DEFAULT_MATCHING_STRATEGY=hybrid   # 'exact', 'semantic', or 'hybrid'
SIMILARITY_THRESHOLD=0.85
OVMS_TIMEOUT=60

# =============================================================================
# MinIO Storage
# =============================================================================
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_ENDPOINT=minio:9000
```

#### Service Modes

| Mode         | Configuration                     | Use Case              |
| ------------ | --------------------------------- | --------------------- |
| **Single**   | `SERVICE_MODE=single`             | Development, testing  |
| **Parallel** | `SERVICE_MODE=parallel WORKERS=4` | Production deployment |

**Start in Different Modes:**

```bash
# Single mode (default)
make up

# Parallel mode with 4 workers
make up-parallel WORKERS=4

# Parallel mode with auto-scaling
make up-parallel WORKERS=4 SCALING_MODE=auto
```

#### Take-Away Docker Services

| Container          | Ports      | Description                         |
| ------------------ | ---------- | ----------------------------------- |
| `takeaway_app`     | 7860, 8080 | Main application (Gradio + FastAPI) |
| `ovms-vlm`         | 8001       | Vision-Language Model server        |
| `frame-selector`   | 8085       | YOLO-based frame selection          |
| `semantic-service` | 8081, 9091 | Semantic text matching              |
| `minio`            | 9000, 9001 | S3-compatible storage               |

### Configure System Proxy

Please follow these steps to configure proxy settings:

#### 1. Configure Proxy for Current Shell Session

```bash
export http_proxy=http://<proxy-host>:<port>
export https_proxy=http://<proxy-host>:<port>
export HTTP_PROXY=http://<proxy-host>:<port>
export HTTPS_PROXY=http://<proxy-host>:<port>
export NO_PROXY=localhost,127.0.0.1,::1
export no_proxy=localhost,127.0.0.1,::1
```

#### 2. Docker Daemon Proxy Configuration

Create directory if missing:

```bash
sudo mkdir -p /etc/systemd/system/docker.service.d
sudo nano /etc/systemd/system/docker.service.d/http-proxy.conf
```

Add configuration:

```ini
[Service]
Environment="http_proxy=http://<proxy-host>:<port>"
Environment="https_proxy=http://<proxy-host>:<port>"
Environment="no_proxy=localhost,127.0.0.1,::1"
```

Reload and restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart docker
```

---

## Benchmarking

### Dine-In Benchmarking

**Initialize Performance Tools:**

```bash
cd dine-in
make update-submodules
```

**Run Benchmark:**

```bash
make benchmark
```

**Stream Density Test:**

```bash
make benchmark-density
```

**View Results:**

```bash
make benchmark-density-results
cat results/benchmark_results.json
```

### Take-Away Benchmarking

**Initialize Performance Tools:**

```bash
cd take-away
make update-submodules
```

**Single Video Benchmark:**

```bash
make benchmark
```

**Fixed Workers Benchmark:**

```bash
make benchmark-oa BENCHMARK_WORKERS=4 BENCHMARK_DURATION=300
```

**Stream Density Benchmark:**

```bash
make benchmark-stream-density \
  BENCHMARK_TARGET_LATENCY_MS=25000 \
  BENCHMARK_MIN_TRANSACTIONS=3 \
  BENCHMARK_WORKER_INCREMENT=1
```

### Benchmark Configuration Variables

| Variable            | Default   | Description                       |
| ------------------- | --------- | --------------------------------- |
| `TARGET_LATENCY_MS` | 25000     | Target latency threshold (ms)     |
| `LATENCY_METRIC`    | avg       | 'avg', 'p95', or 'max'            |
| `WORKER_INCREMENT`  | 1         | Workers added per iteration       |
| `INIT_DURATION`     | 10        | Warmup time (seconds)             |
| `MIN_TRANSACTIONS`  | 3         | Min transactions before measuring |
| `MAX_ITERATIONS`    | 50        | Max scaling iterations            |
| `RESULTS_DIR`       | ./results | Results output directory          |

---

## Useful Make Commands

### Dine-In Commands

```bash
make build                  # Build Docker images
make up                     # Start services
make down                   # Stop services
make logs                   # View logs
make update-submodules      # Initialize performance-tools
make benchmark              # Run benchmark
make benchmark-density      # Run stream density test
```

### Take-Away Commands

```bash
make build                  # Build Docker images
make up                     # Start (single mode)
make up-parallel WORKERS=4  # Start (parallel mode)
make down                   # Stop services
make logs                   # View logs
make update-submodules      # Initialize performance-tools
make benchmark              # Run benchmark
make benchmark-stream-density  # Stream density test
```

### Common Commands

- `make clean-images` — Remove dangling Docker images
- `make clean-all` — Remove all unused Docker resources
- `make check-env` — Verify configuration
- `make show-config` — Display current configuration

---

## Troubleshooting

### Common Issues

**OVMS Not Loading:**

- Ensure GPU drivers are installed
- Check model files exist in `ovms-service/models/`
- Verify OVMS endpoint in `.env`

**VLM Timeout Errors:**

- Increase `API_TIMEOUT` in `.env`
- Check GPU memory utilization
- Consider using a smaller model precision (INT8)

**Stream Processing Issues (Take-Away):**

- Verify RTSP stream URLs are accessible
- Check network bandwidth
- Consider reducing number of parallel workers

### Debug Commands

```bash
# Check container logs
docker logs <container_name>

# Check GPU utilization
nvidia-smi -l 1

# Check network connectivity
curl http://localhost:8001/v2/models

# Verify service health
curl http://localhost:8083/health
```
