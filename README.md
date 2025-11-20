# Order Accuracy

## Overview

The Order Accuracy Pipeline System is an open-source reference implementation for building and deploying video analytics pipelines for retail order accuracy in Quick Servce Restaurant(QSR) use cases. It leverages Intel® hardware and software, GStreamer, and OpenVINO™ to enable scalable, real-time object detection and classification at the edge.

## 📋 Prerequisites

- Ubuntu 24.04 or newer (Linux recommended)
- [Docker](https://docs.docker.com/engine/install/)
- [Make](https://www.gnu.org/software/make/) (`sudo apt install make`)
- Intel hardware (CPU, iGPU, dGPU, NPU)
- Intel drivers (see [Intel GPU drivers](https://dgpu-docs.intel.com/driver/client/overview.html))
- Sufficient disk space for models, videos, and results

## 🚀 QuickStart

Clone the repo with the below command
```
git clone -b <release-or-tag> --single-branch https://github.com/intel-retail/automated-self-checkout
```
>Replace <release-or-tag> with the version you want to clone (for example, **v1.1.0**).
```
git clone -b v1.1.0 --single-branch https://github.com/intel-retail/automated-self-checkout
```

### **NOTE:** 

By default the application runs by pulling the pre-built images. If you want to build the images locally and then run the application, set the flag:

```bash
REGISTRY=false

usage: make <command> REGISTRY=false (applicable for all commands like benchmark, benchmark-stream-density..)
Example: make run-demo REGISTRY=false
```

(If this is the first time, it will take some time to download videos, models, docker images and build images)

### 1. Step by step instructions:

1.1 Download the models using download_models/downloadModels.sh

  ```bash
  make download-models
  ```

1.2 Update github submodules

  ```bash
  make update-submodules
  ```

1.3 Download sample videos used by the performance tools

  ```bash
  make download-sample-videos
  ```

1.4 Start Order Accuracy using the Docker Compose file.

  ```bash
  make run-render-mode
  ```

- The above series of commands can be executed using only one command:
    
  ```bash
  make run-demo
  ```
### 2. To build the images locally step by step:
- Follow the following steps:
  ```bash
  make download-models REGISTRY=false
  make update-submodules REGISTRY=false
  make download-sample-videos
  make run-render-mode REGISTRY=false
  ```
- The above series of commands can be executed using only one command:
    ```bash
    make run-demo REGISTRY=false
    ```

### 3. Stop all containers

```sh
make down
```

### 4. Run benchmarking on CPU/NPU/GPU.

```sh
make benchmark
```

- By default, the configuration is set to use the CPU. If you want to benchmark the application on GPU or NPU, please update the `DEVICE_ENV` variable.

  ```sh
  make benchmark DEVICE_ENV=res/all-gpu.env
  ```

### 5. See the benchmarking results.

```sh
make consolidate-metrics

cat benchmark/metrics.csv
```


## 🛠️ Other Useful Make Commands.

- `make clean-images` — Remove dangling Docker images
- `make clean-models` — Remove all the downloaded models from the system
- `make clean-all` — Remove all unused Docker resources

## 📁 Project Structure

- `configs/` — Configuration files (txt file with sample video URLs for inference)
- `docker/` — Dockerfiles for downloader and pipeline containers
- `download-scripts/` — Scripts for downloading models and videos
- `src/` — Main source code and pipeline runner scripts
- `Makefile` — Build automation and workflow commands

## ⓘ Learn More

For detailed documentation and a comprehensive guide, please visit our [project website](https://intel-retail.github.io/documentation/use-cases/order-accuracy/order-accuracy.html).

---
