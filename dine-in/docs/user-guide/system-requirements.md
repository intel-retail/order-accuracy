# System Requirements

Hardware and software requirements for running the Dine-In Order Accuracy application.

## Hardware Requirements

### Minimum

| Component | Requirement |
|-----------|-------------|
| CPU | Intel Core i7 (8+ cores) |
| RAM | 16 GB |
| GPU | Intel Arc A380 or Intel Data Center GPU Flex 140 |
| Storage | 20 GB available |

### Recommended

| Component | Requirement |
|-----------|-------------|
| CPU | Intel Xeon (16+ cores) |
| RAM | 32 GB |
| GPU | Intel Arc A770 or Intel Data Center GPU Flex 170 |
| Storage | 50 GB SSD |

## Software Requirements

### Operating System

| OS | Version |
|----|---------|
| Ubuntu | 22.04 LTS or 24.04 LTS |
| RHEL | 8.x or 9.x |
| Windows | WSL2 with Ubuntu 22.04+ |

### Container Runtime

| Software | Version |
|----------|---------|
| Docker Engine | 24.0+ |
| Docker Compose | 2.20+ |

### GPU Drivers (Intel)

| Driver | Version |
|--------|---------|
| Intel GPU Drivers | Latest stable |
| Level Zero | 1.8+ |

#### Install Intel GPU Drivers (Ubuntu)

```bash
# Add Intel repositories
wget -qO - https://repositories.intel.com/gpu/intel-graphics.key | \
  sudo gpg --dearmor --output /usr/share/keyrings/intel-graphics.gpg

echo "deb [arch=amd64 signed-by=/usr/share/keyrings/intel-graphics.gpg] \
  https://repositories.intel.com/gpu/ubuntu jammy unified" | \
  sudo tee /etc/apt/sources.list.d/intel-gpu-jammy.list

# Install drivers
sudo apt update
sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero
```

## Network Requirements

### Ports

| Port | Service | Direction |
|------|---------|-----------|
| 7861 | Gradio UI | Inbound |
| 8083 | REST API | Inbound |
| 8000 | OVMS (internal) | Internal |
| 8080 | Semantic Service (internal) | Internal |
| 9000 | Metrics Collector (internal) | Internal |

### Docker Networks

The application creates a bridge network (`dinein_network`) for inter-container communication.

## Model Requirements

### Qwen2.5-VL-7B-Instruct-ov-int8

| Property | Value |
|----------|-------|
| Size | ~7B parameters |
| Format | OpenVINO INT8 |
| VRAM | 8-12 GB |
| Disk | ~15 GB |

Model is automatically downloaded by OVMS on first startup.

## Performance Expectations

| Configuration | Expected Latency |
|--------------|------------------|
| Intel Arc A380 | 12-15s per image |
| Intel Arc A770 | 8-10s per image |
| Intel Flex 170 | 6-8s per image |

## Verifying System Readiness

### Check GPU

```bash
# Intel GPU
clinfo | grep "Device Name"
intel_gpu_top
```

### Check Docker

```bash
docker --version
docker compose version
```

### Check Memory

```bash
free -h
# Ensure at least 16GB available
```

### Check Ports

```bash
# Ensure ports are not in use
netstat -tulpn | grep -E '7861|8083|8000|8080|9000'
```
