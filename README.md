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

> The first run will download models, videos, and build Docker images. This may take some time.


### 1. Download models and videos, and run the Order Accuracy application.

```sh
make download-models
make update-submodules
make download-sample-videos
make run-render-mode
```


> **User can directly run single make command that internally called all above command and run the Order Accuracy application.**


### 3. Run Order Accuracy appliaction with single command.


```sh
make run-demo

```

### 4. Stop all containers

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

---
