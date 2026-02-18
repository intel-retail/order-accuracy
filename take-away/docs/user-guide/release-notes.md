# Release Notes

Version history and changelog for Take-Away Order Accuracy.

---

## Version 1.0.0 (January 2025)

**Initial Production Release**

### Features

#### Core Functionality
- **Dual Service Mode**: Support for single worker and parallel worker modes
- **VLM Integration**: Qwen2.5-VL-7B-Instruct via OpenVINO Model Server
- **Video Processing**: GStreamer-based pipeline with RTSP support
- **Frame Selection**: YOLO-powered intelligent frame selection
- **Semantic Matching**: Hybrid exact/semantic item matching

#### Architecture
- **Station Workers**: Production-ready worker processes with isolation
- **VLM Scheduler**: Time-window batching for throughput optimization
- **Circuit Breaker**: Resilient RTSP connectivity with auto-recovery
- **Exponential Backoff**: Configurable retry with jitter

#### User Interface
- **Gradio UI**: Web-based interface for video upload and validation
- **REST API**: FastAPI-based endpoints with OpenAPI documentation
- **MinIO Integration**: S3-compatible storage for frames and results

#### Benchmarking
- **Single Video Benchmark**: End-to-end latency testing
- **Stream Density Test**: Maximum concurrent stream detection
- **VLM Metrics Logger**: Detailed performance metrics collection

### Components

| Component | Version | Description |
|-----------|---------|-------------|
| Order Accuracy Service | 1.0.0 | Core orchestration |
| OVMS VLM | 2024.6 | Model inference |
| Frame Selector | 1.0.0 | YOLO-based selection |
| Gradio UI | 1.0.0 | Web interface |
| Semantic Service | 1.0.0 | Text matching |

### Performance

| Metric | Single Mode | Parallel (4 workers) |
|--------|-------------|---------------------|
| Latency P50 | 3-5s | 2-4s |
| Throughput | 8-12/min | 20-30/min |
| GPU Memory | 8-10 GB | 10-14 GB |

### Known Issues

1. **RTSP Reconnection Delay**: Initial RTSP connection may take 5-10 seconds
2. **Large Video Upload**: Videos >500MB may timeout on slow connections
3. **INT4 Model**: INT4 quantization not yet supported

### Migration Notes

This is the initial release. No migration required.

---

## Roadmap

### Version 1.1.0 (Planned Q2 2025)

#### Planned Features
- [ ] INT4 model quantization support
- [ ] Multi-GPU inference distribution
- [ ] RabbitMQ integration for async processing
- [ ] Kubernetes/Helm deployment
- [ ] Prometheus metrics export
- [ ] Grafana dashboard templates

#### Planned Improvements
- [ ] Reduced VLM latency via streaming inference
- [ ] Enhanced semantic matching with context
- [ ] Auto-tuning for optimal batch size
- [ ] Improved error messages and diagnostics

### Version 1.2.0 (Planned Q3 2025)

#### Planned Features
- [ ] Multi-language item recognition
- [ ] Receipt OCR integration
- [ ] Analytics dashboard
- [ ] A/B testing framework
- [ ] Model hot-swap capability

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 1.0.0 | Jan 2025 | Initial production release |

---

## Support

### Reporting Issues

When reporting issues, please include:

1. Version number (`make show-config`)
2. Service mode (single/parallel)
3. Hardware configuration
4. Logs (`make logs`)
5. Steps to reproduce

### Getting Help

- Review [documentation](../user-guide/)
- Check [System Requirements](system-requirements.md)
- Submit detailed issue report

---

## Acknowledgments

- OpenVINO team for model optimization
- Qwen team for VLM model
- GStreamer community
- FastAPI and Gradio maintainers

---

## License

Copyright © 2025 Intel Corporation

Licensed under the Apache License, Version 2.0.
