# Get Started

## Prerequisites

- Ubuntu 24.04 or newer (Linux recommended), Desktop edition (or Server edition with GUI installed)
- [Docker](https://docs.docker.com/engine/install/) 24.0+
- [Docker Compose](https://docs.docker.com/compose/install/) V2+
- [Make](https://www.gnu.org/software/make/) (`sudo apt install make`)
- Intel hardware (CPU, iGPU, dGPU)
- Intel drivers:
  - [Intel GPU drivers](https://dgpu-docs.intel.com/driver/client/overview.html)
- Sufficient disk space for models, videos, and results (50GB minimum)

> [!NOTE]
> First-time setup downloads AI models (~7GB) and Docker images - this may take 30-60 minutes depending on your internet connection.

> [!NOTE]
> **KV Cache on iGPU / low-RAM systems:** 16 GB RAM is sufficient for **inference**.
> For first-time model export, a higher-memory host (48–64 GB) is recommended.
> On iGPU platforms, the KV cache is allocated from **system RAM**. Set `export CACHE_SIZE=2`
> before running `setup_models.sh` to reduce KV cache to 2 GB (default is 4 GB).
> See [ovms-service/README.md — Tuning the KV Cache Size](https://github.com/intel-retail/order-accuracy/blob/main/ovms-service/README.md#tuning-the-kv-cache-size) for a full per-platform guide.

## Choose Your Application

| Criteria | Dine-In Order Accuracy                                             | Take-Away Order Accuracy                                         |
| -------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Purpose  | Validate food plates at serving stations before delivery to tables | Real-time order validation for drive-through and counter service |
| Use When | You need image-based validation for restaurant table service       | You need continuous video stream validation at multiple stations |
| Input    | Static images of food trays/plates                                 | RTSP video streams                                               |
| Features | Gradio web interface, REST API for POS integration                 | Multi-station parallel processing, VLM request batching          |

### What You Will See When Working

|                | Dine-In Results                                          | Take-Away Results                              |
| -------------- | -------------------------------------------------------- | ---------------------------------------------- |
| **Visual**     | Gradio UI displays detected items with confidence scores | Real-time frame processing with item detection |
| **Validation** | Order match/mismatch status with detailed comparison     | Continuous order validation against POS data   |
| **Logging**    | Detection results and semantic matching scores           | Frames and results stored in MinIO buckets     |

### Expected Performance

- **Startup Time**: 2-5 minutes (first run includes model loading)
- **Processing**: Sub-15-second validation latency (Dine-In), real-time stream processing (Take-Away)
- **Results**: JSON files appear in `results/` directory

## Step-by-Step Instructions

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In Order Accuracy**
<!--hide_directive:sync: dine-in hide_directive-->

1. **Clone the Repository**

   ```bash
   git clone -b main --single-branch https://github.com/intel-retail/order-accuracy.git
   cd order-accuracy/dine-in
   ```

2. **Configure the Environment**

   ```bash
   # Create .env from template
   make init-env
   # Edit .env if needed — defaults work for most setups

   # Initialize git submodules (for benchmark tools)
   make update-submodules
   ```

3. **Setup OVMS Models (First Time Only)**

   The setup script reads model configuration (device, precision, model name) from `dine-in/.env` (created in Step 2), so **complete Step 2 before running this step**.

   ```bash
   cd ../ovms-service
   ./setup_models.sh --app dine-in    # Downloads and exports model (~30-60 min first time)
   cd ../dine-in
   ```

   This downloads MiniCPM-V-4.5 and converts it to OpenVINO™ INT4 format. This is only needed once — the model files are shared with Take-Away.

4. **Prepare Test Data**
   Before running the application, you must prepare your test data:

   - **Add Images**: Place your food tray images in the `images/` folder
     - Supported formats: `.jpg`, `.jpeg` or `.png`
     - Images should clearly show the food items on the tray

   - **Update Orders**: Edit `configs/orders.json` with your test orders
     - Each order should have an `items_ordered` entry, each with `item` and `quantity`
     - `image_id` should match your image filenames in the `images/` folder

   - **Update Inventory**: Edit `configs/inventory.json` to match your menu items
     - Define all possible food items that can appear in orders
     - Include item names, categories, and any relevant metadata

5. **Build and Start Services**

   ```bash
   # Pull images from registry (default)
   make build && make up

   # OR build locally from source
   make build REGISTRY=false && make up
   ```

   This starts 4 containers:

   | Container                 | Ports      | Purpose                 |
   | ------------------------- | ---------- | ----------------------- |
   | `dinein_app`              | 7861, 8083 | Gradio UI + FastAPI     |
   | `dinein_ovms_vlm`         | 8002       | VLM model server (OVMS) |
   | `dinein_semantic_service` | 8081, 9091 | Semantic matching       |
   | `metrics-collector`       | 9000       | System metrics          |

6. **Access the Application**
   - **Gradio UI**: `http://localhost:7861`
   - **REST API Docs**: `http://localhost:8083/docs`

<!--hide_directive:::
:::{tab-item}hide_directive-->  **Take-Away Order Accuracy**
<!--hide_directive:sync: take-away hide_directive-->

1. **Clone the Repository**

   ```bash
   git clone -b main --single-branch https://github.com/intel-retail/order-accuracy.git
   cd order-accuracy/take-away
   ```

2. **Configure the Environment**

   ```bash
   # Create .env from template
   make init-env
   # Edit .env if needed — defaults work for most setups

   # Initialize git submodules (for benchmark tools)
   make update-submodules
   ```

   Edit the generated `.env` file as needed. See [Configuration](./take-away/get-started.md#configuration) in the dedicated Take-Away guide.

3. **Setup OVMS Models (First Time Only)**

   Set `TARGET_DEVICE` in your `.env` **before** running this step. The script reads that value to export the model in the correct format for the target device.

   ```bash
   cd ../ovms-service
   ./setup_models.sh --app take-away    # Downloads and exports model (~30-60 min first time)
   cd ../take-away
   ```

   This downloads and exports:

   - MiniCPM-V-4_5 INT4 (OpenVINO™ format)
   - YOLOv11 model (INT8 OpenVINO™)
   - EasyOCR detection and recognition models

   > [!NOTE]
   > Re-run this step any time you change `TARGET_DEVICE` in `.env`.

4. **Build and Start Services**

   ```bash
   # Pull images from registry (default)
   make build
   make up

   # OR build locally from source
   make build REGISTRY=false
   make up
   ```

5. **Access the Application**
   - **Gradio UI**: `http://localhost:7860`
   - **MinIO Console**: `http://localhost:9001` (`minioadmin/minioadmin`)

<!--hide_directive:::
::::hide_directive-->

## Verify the Installation

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: dine-in hide_directive-->

```bash
# API health check
make test-api

# Or directly
curl http://localhost:8083/health

# Check OVMS model
curl http://localhost:8002/v1/config | jq .
```

Open `http://localhost:7861` for the Gradio UI, or `http://localhost:8083/docs` for the REST API docs.

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

- Health Check

  ```bash
  make test-api
  ```

  This checks both the order accuracy API (`localhost:8000`) and OVMS (`localhost:8001`).

- Service URLs

  | Service            | URL                     |
  | ------------------ | ----------------------- |
  | Order Accuracy API | `http://localhost:8000` |
  | OVMS VLM           | `http://localhost:8001` |
  | Gradio UI          | `http://localhost:7860` |
  | MinIO Console      | `http://localhost:9001` |
  | Semantic Service   | `http://localhost:8080` |

- Service Status

  ```bash
  make status
  ```

<!--hide_directive:::
::::hide_directive-->

## First Order Validation

### Via Gradio UI

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: dine-in hide_directive-->

1. Open `http://localhost:7861`
2. Select a scenario from the dropdown
3. Review the order manifest
4. Click **"Validate Plate"**
5. View accuracy score, matched/missing/extra items, and performance metrics

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

1. Open `http://localhost:7860`
2. Upload a test video or enter an RTSP URL
3. Click "Upload and Start Processing"
4. View results showing matched, missing, and extra items

<!--hide_directive:::
::::hide_directive-->

### Via REST API

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: dine-in hide_directive-->

The bundled `MCD-1001.png` image shows **Filet-O-Fish** and **Cheesy Fries** on the tray.
Two test scenarios are provided:

**Negative test case** — order does not match tray (demonstrates mismatch detection):

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={"items":[{"name":"Cheeseburger","quantity":1},{"name":"French Fries","quantity":1}]}'
# Expected: order_complete=false, accuracy_score=0.0
```

**Positive test case** — order matches tray (demonstrates successful validation):

```bash
curl -X POST "http://localhost:8083/api/validate" \
  -F "image=@images/MCD-1001.png" \
  -F 'order={"items":[{"name":"Filet-O-Fish","quantity":1},{"name":"Cheesy Fries","quantity":1}]}'
# Expected: order_complete=true, accuracy_score=1.0
```

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

```bash
# Upload video and validate
curl -X POST http://localhost:8000/upload-video \
  -F "file=@storage/videos/test.mp4" \
  -F "video_id=test_001"

# Check results
curl http://localhost:8000/results/test_001
```

<!--hide_directive:::
::::hide_directive-->

### Via Make Target

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: dine-in hide_directive-->

```bash
# Services must be running first
make benchmark-single IMAGE_ID=MCD-1001
```

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

> [!NOTE]
> **Prerequisite:** Ensure `storage/videos/test.mp4` exists. Run `make download-sample-video` if needed.

```bash
make benchmark
```

<!--hide_directive:::
::::hide_directive-->

## Stop the Services

```bash
# Stop all services
make down

# Stop and remove volumes (clean restart)
make clean
```

## Quick Start Reference

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In Commands**
<!--hide_directive:sync: dine-in hide_directive-->

| Configuration      | Command                     | Description                |
| ------------------ | --------------------------- | -------------------------- |
| **Start Services** | `make up`                   | Start all Dine-In services |
| **Build Locally**  | `make build REGISTRY=false` | Build images from source   |
| **View Logs**      | `make logs`                 | View service logs          |
| **Stop Services**  | `make down`                 | Stop all services          |

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away Commands**
<!--hide_directive:sync: take-away hide_directive-->

| Configuration     | Command                      | Description                                |
| ----------------- | ---------------------------- | ------------------------------------------ |
| **Single Mode**   | `make up`                    | Start in single worker mode (development)  |
| **Parallel Mode** | `make up-parallel WORKERS=4` | Start with 4 parallel workers (production) |
| **Build Locally** | `make build REGISTRY=false`  | Build images from source                   |
| **View Logs**     | `make logs`                  | View service logs                          |

> [!NOTE]
> **Single Mode** is best for development and testing. **Parallel Mode** is recommended for production with multiple camera stations.

<!--hide_directive:::
::::hide_directive-->

## Advanced Settings

See the [Advanced Settings](./get-started/advanced.md) guide for detailed configuration options, including environment variables, service modes, and troubleshooting tips.

<!--hide_directive
:::{toctree}
:hidden:

./get-started/advanced.md

:::
hide_directive-->
