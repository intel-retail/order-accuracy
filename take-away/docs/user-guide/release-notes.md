# Release Notes

Version history and changelog for Take-Away Order Accuracy.

---

## Version 2026.0-rc1 (March 2026)

**Initial Release Candidate**

### Highlights

- **AI-Powered Order Validation**: Real-time take-away order verification using Qwen2.5-VL-7B Vision Language Model
- **Multi-Station Parallel Processing**: Concurrent order validation across multiple stations via RTSP streams
- **Intelligent Frame Selection**: YOLO11-based frame selection with OpenVINO INT8 inference for optimal VLM input
- **Semantic Matching**: Hybrid exact/semantic item matching via dedicated microservice
- **Docker Registry Support**: Pre-built images published to `intel/` Docker Hub namespace
- **Stream Density Benchmarking**: Automated latency-based stream density testing

### Published Images

| Image | Tag | Size |
|-------|-----|------|
| `intel/order-accuracy-take-away` | `2026.0-rc1` | 9.64GB |
| `intel/order-accuracy-frame-selector` | `2026.0-rc1` | 1.96GB |
| `intel/order-accuracy-take-away-ui` | `2026.0-rc1` | 1.3GB |
| `intel/order-accuracy-take-away-rtsp` | `2026.0-rc1` | 227MB |

### Features

#### Core Functionality
- **Dual Service Mode**: Single worker mode for development, parallel worker mode for production
- **VLM Integration**: Qwen2.5-VL-7B-Instruct via OpenVINO Model Server (OVMS) with GPU acceleration
- **Video Processing**: GStreamer-based pipeline with RTSP support and configurable FPS
- **Frame Selection**: YOLO11 nano model with OpenVINO INT8 inference for hand/object detection and frame filtering
- **Semantic Matching**: Hybrid exact/semantic item matching with configurable similarity threshold
- **EasyOCR Integration**: Order number detection from video frames

#### Architecture
- **Station Workers**: Production-ready multi-process workers with per-station isolation
- **VLM Scheduler**: Time-window batching for throughput optimization
- **2PC Pipeline Sync**: Two-phase commit synchronization between RTSP streamer and processing pipelines
- **Circuit Breaker**: Resilient RTSP connectivity with auto-recovery
- **Exponential Backoff**: Configurable retry with jitter for transient failures

#### User Interface
- **Gradio UI**: Web-based interface for video upload and order validation
- **REST API**: FastAPI-based endpoints with OpenAPI documentation
- **MinIO Integration**: S3-compatible storage for frames and results

#### Build & Deployment
- **Registry Mode**: `make build` pulls pre-built images from Docker Hub
- **Local Build Mode**: `make build REGISTRY=false` builds all images locally from source
- **OVMS Auto-Config**: `graph.pbtxt` auto-generated from `config.json` graph_options for tester-friendly tuning
- **Model Setup Script**: `setup_models.sh` handles VLM model download, EasyOCR model download, and graph configuration

#### Benchmarking
- **Stream Density Test**: `make benchmark-stream-density` — automated latency-based stream scaling
- **Fixed Workers Benchmark**: `make benchmark-oa` — throughput testing with configurable workers
- **Single Video Benchmark**: `make benchmark` — end-to-end latency testing
- **VLM Metrics Logger**: Detailed performance metrics collection and consolidation

### Components

| Component | Image | Description |
|-----------|-------|-------------|
| Order Accuracy Service | `intel/order-accuracy-take-away` | Core orchestration, GStreamer pipelines, VLM scheduling |
| Frame Selector | `intel/order-accuracy-frame-selector` | YOLO11 OpenVINO INT8 frame selection |
| Gradio UI | `intel/order-accuracy-take-away-ui` | Web interface for order validation |
| RTSP Streamer | `intel/order-accuracy-take-away-rtsp` | Video-to-RTSP stream conversion with 2PC sync |
| OVMS VLM | `openvino/model_server:latest-gpu` | Qwen2.5-VL-7B model serving |
| Semantic Service | `intel/semantic-search-agent:1.0.0` | Semantic text matching microservice |
| MinIO | `minio/minio` | S3-compatible object storage |

### Configuration Defaults

| Variable | Default | Description |
|----------|---------|-------------|
| `BENCHMARK_TARGET_LATENCY_MS` | `25000` | Target latency threshold (25s) |
| `BENCHMARK_INIT_DURATION` | `10` | Warmup time (seconds) |
| `OCR_WARMUP_FRAMES` | `2` | Frames before OCR ready signal |
| `REGISTRY` | `true` | Pull from registry; set `false` to build locally |

### Known Issues

1. **RTSP Reconnection Delay**: Initial RTSP connection may take 5-10 seconds
2. **Large Video Upload**: Videos >500MB may timeout on slow connections
3. **Order Accuracy Image Size**: 9.64GB due to torch+CUDA dependencies (required by EasyOCR)

---