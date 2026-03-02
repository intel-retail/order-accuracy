# Release Notes

## Version 2026.0-rc1 (March 2026)

### New Features

- **Flexible Image Deployment**: New `REGISTRY` flag for build and deployment control
  - `make build REGISTRY=false` - Build image locally from source
  - `make build` - Pull pre-built image from registry (default)
  - `make up REGISTRY=false` - Run with locally built image
  - `make up` - Run with registry image (default)

- **Docker Image Tagging**: Standardized image naming
  - Image: `intel/order-accuracy-dine-in:2026.0-rc1`
  - Configurable via `TAG` and `DINEIN_IMAGE_NAME` variables

### Configuration Changes

| Variable | Old Default | New Default |
|----------|-------------|-------------|
| `TAG` | 1.0.0 | 2026.0-rc1 |
| `REGISTRY` | false | true |
| `DINEIN_IMAGE` | - | intel/order-accuracy-dine-in:2026.0-rc1 |

### Migration Notes

- Update deployment scripts to use new `REGISTRY` flag
- Image name changed from `dine-in-dine-in` to `intel/order-accuracy-dine-in`
- Use `REGISTRY=false` for local development workflows

---

## Version 2.0.0 (February 2026)

### New Features

- **Circuit Breaker Pattern**: Added fault tolerance for VLM and Semantic services
  - 5 consecutive failures trigger circuit OPEN state
  - 30s recovery timeout for VLM, 15s for Semantic service
  - Automatic recovery with half-open state testing

- **Connection Pooling**: Shared HTTP clients with optimized settings
  - Up to 50 concurrent connections for VLM client
  - HTTP/2 support enabled for improved performance
  - Keepalive connections (30s expiry)

- **Bounded Validation Cache**: LRU cache prevents memory exhaustion
  - Maximum 10,000 entries
  - Thread-safe operations
  - Automatic eviction of oldest entries

- **Image Preprocessing Pipeline**: Optimized images for faster VLM inference
  - Smart resizing (672px max dimension)
  - Adaptive contrast enhancement
  - Light sharpening for food detail
  - JPEG compression (82% quality)

- **Stream Density Benchmark**: New testing mode for concurrent validation
  - Automatic density scaling
  - Latency-based pass/fail criteria
  - Comprehensive results export (JSON/CSV)

### Improvements

- **Thread-safe Singleton**: Config manager uses double-checked locking
- **Async Metrics Collection**: Non-blocking system stats retrieval
- **Token Usage Logging**: Detailed TPS and token metrics
- **Enhanced VLM Prompts**: Inventory-aware prompts for better accuracy

### Bug Fixes

- Fixed race condition in service initialization
- Fixed unbounded Dict causing OOM under load
- Fixed blocking `psutil.cpu_percent()` call
- Fixed sync HTTP in Gradio callback

### Configuration Changes

| Variable | Old Default | New Default |
|----------|-------------|-------------|
| `API_TIMEOUT` | 60 | 300 |
| `BENCHMARK_TARGET_LATENCY_MS` | 1000 | 2000 |

### Dependencies

- Added: `httpx[http2]>=0.25.0` (HTTP/2 support)
- Added: `aiofiles>=23.0.0` (async file operations)
- Removed: `requests` (replaced by httpx)

### Known Issues

- GPU memory utilization metric always reports 0.0 (metrics collector limitation)
- First `psutil.cpu_percent()` call returns 0.0 (expected behavior)

---

## Version 1.0.0 (January 2026)

### Initial Release

- Gradio UI for interactive validation
- FastAPI REST endpoints
- OVMS VLM integration (Qwen2.5-VL-7B)
- Semantic matching service
- Basic metrics collection
- Docker Compose deployment

### Features

- Single image validation
- Batch validation endpoint
- Order manifest comparison
- Accuracy scoring
- Performance metrics

