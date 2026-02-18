# System Requirements

This document outlines the hardware, software, and network requirements for deploying Take-Away Order Accuracy in various configurations.

---

## Table of Contents

1. [Hardware Requirements](#hardware-requirements)
2. [Software Requirements](#software-requirements)
3. [Network Requirements](#network-requirements)
4. [GPU Support](#gpu-support)
5. [Deployment Configurations](#deployment-configurations)
6. [Resource Estimation](#resource-estimation)

---

## Hardware Requirements

### Minimum Configuration

Suitable for development and testing with single worker mode.

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Xeon 8 cores @ 2.4 GHz |
| **RAM** | 16 GB DDR4 |
| **GPU** | Intel Arc A770 8GB / NVIDIA RTX 3060 12GB |
| **Storage** | 50 GB SSD |
| **Network** | 1 Gbps Ethernet |

### Recommended Configuration

Suitable for production with 2-4 station workers.

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Xeon 16 cores @ 3.0 GHz |
| **RAM** | 32 GB DDR4 |
| **GPU** | Intel Data Center GPU Max / NVIDIA RTX 3080 10GB |
| **Storage** | 200 GB NVMe SSD |
| **Network** | 10 Gbps Ethernet |

### High-Performance Configuration

Suitable for production with 8+ station workers.

| Component | Specification |
|-----------|---------------|
| **CPU** | Intel Xeon 32+ cores @ 3.0 GHz |
| **RAM** | 64 GB DDR4/DDR5 |
| **GPU** | 2x NVIDIA RTX 4090 / Intel Data Center GPU Flex |
| **Storage** | 500 GB NVMe SSD RAID |
| **Network** | 25 Gbps Ethernet |

---

## Software Requirements

### Operating System

| OS | Version | Support Level |
|----|---------|---------------|
| Ubuntu | 22.04 LTS | Fully supported |
| Ubuntu | 24.04 LTS | Fully supported |
| RHEL | 8.x / 9.x | Supported |
| Windows | WSL2 + Docker | Development only |

### Container Runtime

| Software | Minimum Version | Recommended |
|----------|-----------------|-------------|
| Docker Engine | 24.0.0 | 25.0+ |
| Docker Compose | 2.20.0 | 2.24+ |
| NVIDIA Container Toolkit | 1.14.0 | Latest |
| containerd | 1.6.0 | 1.7+ |

### GPU Drivers

#### NVIDIA
| Driver | Minimum Version |
|--------|-----------------|
| NVIDIA Driver | 535.x |
| CUDA Toolkit | 12.0 |
| cuDNN | 8.9 |

#### Intel
| Driver | Minimum Version |
|--------|-----------------|
| Intel GPU Driver | Latest from packages.intel.com |
| oneAPI | 2024.0+ |
| Level Zero | 1.8+ |

### Python (for local development)

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| pip | 23.0+ |
| venv | Built-in |

---

## Network Requirements

### Port Configuration

| Service | Port | Protocol | Purpose |
|---------|------|----------|---------|
| Order Accuracy API | 8000 | HTTP | REST API |
| OVMS VLM | 8001 | HTTP | Model inference |
| Gradio UI | 7860 | HTTP | Web interface |
| MinIO API | 9000 | HTTP | S3-compatible storage |
| MinIO Console | 9001 | HTTP | Storage admin UI |
| Semantic Service | 8080 | HTTP | Semantic matching |
| RTSP Streamer | 8554 | RTSP | Video streaming (parallel mode) |

### Firewall Rules

```bash
# Allow required ports
sudo ufw allow 8000/tcp   # Order Accuracy API
sudo ufw allow 7860/tcp   # Gradio UI
sudo ufw allow 9001/tcp   # MinIO Console (admin only)
```

### Network Bandwidth

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Single worker | 100 Mbps | 1 Gbps |
| 4 workers (RTSP) | 500 Mbps | 5 Gbps |
| 8 workers (RTSP) | 1 Gbps | 10 Gbps |

### RTSP Requirements (Parallel Mode)

| Requirement | Specification |
|-------------|---------------|
| Protocol | RTSP/RTP over TCP |
| Codec | H.264 (Main/High profile) |
| Resolution | 720p minimum, 1080p recommended |
| Frame Rate | 15-30 FPS |
| Bitrate | 2-8 Mbps per stream |

---

## GPU Support

### NVIDIA GPUs

| GPU | VRAM | Workers Supported | Notes |
|-----|------|-------------------|-------|
| RTX 3060 | 12 GB | 1-2 | Development |
| RTX 3080 | 10 GB | 2-4 | Recommended |
| RTX 4080 | 16 GB | 4-6 | High performance |
| RTX 4090 | 24 GB | 6-8 | Best performance |
| A100 | 40/80 GB | 10+ | Data center |

### Intel GPUs

| GPU | VRAM | Workers Supported | Notes |
|-----|------|-------------------|-------|
| Arc A770 | 16 GB | 2-4 | Consumer |
| Arc A750 | 8 GB | 1-2 | Budget |
| Data Center GPU Max | 48 GB | 8+ | Enterprise |
| Data Center GPU Flex | 12 GB | 2-4 | Flexible |

### GPU Memory Allocation

```
┌─────────────────────────────────────────────────────────────┐
│                   GPU MEMORY USAGE                           │
│                                                              │
│  VLM Model (Qwen2.5-VL-7B INT8):     ~4-6 GB                │
│  CUDA/OpenVINO Runtime:               ~1-2 GB                │
│  Per-Request Overhead:                ~0.5-1 GB              │
│  Buffer/Workspace:                    ~1-2 GB                │
│  ─────────────────────────────────────────────               │
│  Total per Worker:                    ~8-11 GB               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Configurations

### Development Configuration

```yaml
# docker-compose.override.yaml
services:
  order-accuracy:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
  
  ovms-vlm:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 12G
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
```

### Production Configuration (Single GPU)

```yaml
services:
  order-accuracy:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 16G
  
  ovms-vlm:
    deploy:
      resources:
        limits:
          cpus: '8'
          memory: 24G
        reservations:
          devices:
            - capabilities: [gpu]
              count: 1
```

### Production Configuration (Multi-GPU)

```yaml
services:
  ovms-vlm-1:
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ['0']
  
  ovms-vlm-2:
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
              device_ids: ['1']
```

---

## Resource Estimation

### Memory Usage Estimates

| Component | Single Mode | Parallel (4 workers) |
|-----------|-------------|---------------------|
| Order Accuracy Service | 2 GB | 4 GB |
| OVMS VLM | 8-12 GB | 12-16 GB |
| Frame Selector | 1 GB | 2 GB |
| Gradio UI | 0.5 GB | 0.5 GB |
| MinIO | 0.5 GB | 1 GB |
| Semantic Service | 1 GB | 1 GB |
| **Total** | **13-17 GB** | **20-25 GB** |

### CPU Usage Estimates

| Component | Single Mode | Parallel (4 workers) |
|-----------|-------------|---------------------|
| Video Processing | 2 cores | 8 cores |
| VLM Pre/Post | 1 core | 4 cores |
| API/Routing | 1 core | 2 cores |
| Frame Selection | 1 core | 2 cores |
| **Total** | **5 cores** | **16 cores** |

### Storage Requirements

| Data Type | Size per Order | Daily (500 orders) |
|-----------|----------------|-------------------|
| Raw Frames | 5-10 MB | 2.5-5 GB |
| Selected Frames | 1-2 MB | 0.5-1 GB |
| Results JSON | 5 KB | 2.5 MB |
| Logs | 1 MB | 500 MB |
| **Total** | - | **3-7 GB/day** |

### Throughput Estimates

| Configuration | Orders/Minute | Latency (P95) |
|---------------|---------------|---------------|
| Single Worker (CPU) | 2-3 | 20-30s |
| Single Worker (GPU) | 8-12 | 5-8s |
| 4 Workers (GPU) | 20-30 | 3-5s |
| 8 Workers (GPU) | 40-60 | 2-4s |

---

## Capacity Planning

### Workers per GPU Formula

```
workers_per_gpu = floor(gpu_vram_gb / 8)

Examples:
- RTX 3080 (10 GB): 1 worker
- RTX 4090 (24 GB): 2-3 workers
- A100 (40 GB): 4-5 workers
```

### Stream Density Formula

```
max_streams = min(
    cpu_cores / 2,
    ram_gb / 4,
    gpu_vram_gb / 8,
    network_mbps / 100
)

Example (32 cores, 64 GB RAM, 24 GB VRAM, 10 Gbps):
max_streams = min(16, 16, 3, 100) = 3 streams per GPU
```

---

## Checklist

Before deployment, verify:

- [ ] Docker and Docker Compose installed
- [ ] GPU drivers installed and verified
- [ ] Required ports available
- [ ] Sufficient disk space
- [ ] Network connectivity to cameras (parallel mode)
- [ ] VLM model downloaded
- [ ] Environment file configured
- [ ] Storage directories created
