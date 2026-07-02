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

> **Note:** First-time setup downloads AI models (~7GB) and Docker images - this may take 30-60 minutes depending on your internet connection.

## Choose Your Application

| Criteria | Dine-In Order Accuracy                                             | Take-Away Order Accuracy                                         |
| -------- | ------------------------------------------------------------------ | ---------------------------------------------------------------- |
| Purpose  | Validate food plates at serving stations before delivery to tables | Real-time order validation for drive-through and counter service |
| Use When | You need image-based validation for restaurant table service       | You need continuous video stream validation at multiple stations |
| Input    | Static images of food trays/plates                                 | RTSP video streams                                               |
| Features | Gradio web interface, REST API for POS integration                 | Multi-station parallel processing, VLM request batching          |

## Step-by-Step Instructions

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In Order Accuracy**
<!--hide_directive:sync: Dine-In hide_directive-->

1. **Clone the Repository**

   ```bash
   git clone -b <release-or-tag> --single-branch https://github.com/intel-retail/order-accuracy
   ```

   > Replace `<release-or-tag>` with the version you want to clone (for example, **v2026.1**).

   ```bash
   git clone -b v2026.1 --single-branch https://github.com/intel-retail/order-accuracy
   cd order-accuracy/dine-in
   ```

2. **Setup OVMS Models (First Time Only)**

   ```bash
   cd ../ovms-service
   ./setup_models.sh -app dine-in
   cd ../dine-in
   ```

   This downloads and converts the Qwen2.5-VL-7B model (~7GB). This only needs to be done once.

3. **Prepare Test Data**
   - Add your food tray/plate images to the `images/` folder
   - Update `configs/orders.json` with test orders
   - Update `configs/inventory.json` with your menu items

4. **Build and Start Services**

   ```bash
   # Using pre-built images (recommended for first run)
   make build
   make up

   # OR build locally from source
   make build REGISTRY=false
   make up
   ```

5. **Access the Application**
   - **Gradio UI**: `http://localhost:7861`
   - **REST API Docs**: `http://localhost:8083/docs`

<!--hide_directive:::
:::{tab-item}hide_directive-->  **Take-Away Order Accuracy**
<!--hide_directive:sync: take-away hide_directive-->

1. **Clone the Repository**

   ```bash
   git clone -b <release-or-tag> --single-branch https://github.com/intel-retail/order-accuracy
   ```

   > Replace `<release-or-tag>` with the version you want to clone (for example, **v2026.1**).

   ```bash
   git clone -b v2026.1 --single-branch https://github.com/intel-retail/order-accuracy
   cd order-accuracy/take-away
   ```

2. **Setup OVMS Models (First Time Only)**

   ```bash
   cd ../ovms-service
   ./setup_models.sh -app take-away
   cd ../take-away
   ```

   This downloads the VLM and EasyOCR models. This only needs to be done once.

3. **Initialize Environment**

   ```bash
   make init-env
   ```

4. **Build and Start Services**

   ```bash
   # Single worker mode (development/testing)
   make build
   make up

   # OR Parallel mode (production)
   make build
   make up-parallel WORKERS=4
   ```

5. **Access the Application**
   - **Gradio UI**: `http://localhost:7860`
   - **MinIO Console**: `http://localhost:9001` (`minioadmin/minioadmin`)

<!--hide_directive:::
::::hide_directive-->

## What You Will See When Working

|                | Dine-In Results                                          | Take-Away Results                              |
| -------------- | -------------------------------------------------------- | ---------------------------------------------- |
| **Visual**     | Gradio UI displays detected items with confidence scores | Real-time frame processing with item detection |
| **Validation** | Order match/mismatch status with detailed comparison     | Continuous order validation against POS data   |
| **Logging**    | Detection results and semantic matching scores           | Frames and results stored in MinIO buckets     |

### Expected Performance

- **Startup Time**: 2-5 minutes (first run includes model loading)
- **Processing**: Sub-15-second validation latency (Dine-In), real-time stream processing (Take-Away)
- **Results**: JSON files appear in `results/` directory

## Verify Results

After starting the application, you can verify it is working:

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In**
<!--hide_directive:sync: Dine-In hide_directive-->

```bash
# Check running containers
docker ps

# View application logs
make logs

# Test the API
curl http://localhost:8083/health
```

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away**
<!--hide_directive:sync: take-away hide_directive-->

```bash
# Check running containers
docker ps

# View application logs
make logs

# Check MinIO storage
# Visit http://localhost:9001
```

<!--hide_directive:::
::::hide_directive-->

## Stop the Services

```bash
# Stop all services
make down

# Stop and remove volumes (clean restart)
make down-volumes
```

## Quick Start Reference

<!--hide_directive::::{tab-set}
:::{tab-item}hide_directive--> **Dine-In Commands**
<!--hide_directive:sync: Dine-In hide_directive-->

| Configuration      | Command                     | Description                |
| ------------------ | --------------------------- | -------------------------- |
| **Start Services** | `make up`                   | Start all Dine-In services |
| **Build Locally**  | `make build REGISTRY=false` | Build images from source   |
| **View Logs**      | `make logs`                 | View service logs          |
| **Stop Services**  | `make down`                 | Stop all containers        |

<!--hide_directive:::
:::{tab-item}hide_directive--> **Take-Away Commands**
<!--hide_directive:sync: take-away hide_directive-->

| Configuration     | Command                      | Description                                |
| ----------------- | ---------------------------- | ------------------------------------------ |
| **Single Mode**   | `make up`                    | Start in single worker mode (development)  |
| **Parallel Mode** | `make up-parallel WORKERS=4` | Start with 4 parallel workers (production) |
| **Build Locally** | `make build REGISTRY=false`  | Build images from source                   |
| **View Logs**     | `make logs`                  | View service logs                          |

> **Note:** **Single Mode** is best for development and testing. **Parallel Mode** is recommended for production with multiple camera stations.

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
