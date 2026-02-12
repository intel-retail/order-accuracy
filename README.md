# **Order Accuracy**

This project processes a video or RTSP stream, extracts **valid order-ID frames**, uploads them to **MinIO**, selects the **top frames per order**, and runs **VLM inference** to extract ordered items.

## 🔧 **VLM Backend Support**

The system supports **two VLM backends**:

1. **Embedded VLM** (Default) - OpenVINO GenAI running directly in application container
   - Model: Qwen2.5-VL-7B-Instruct (int8, ~7GB)
   - Device: GPU (Intel Arc iGPU)
   - Best for: Single deployment, lower latency

2. **OVMS Backend** - External OpenVINO Model Server
   - Model: Qwen2-VL-2B-Instruct (int4, ~2GB)
   - Device: GPU via OVMS service
   - Best for: Multiple applications, resource efficiency, scalability

**Quick Backend Switch**: See [QUICK_START_BACKEND_SWITCH.md](QUICK_START_BACKEND_SWITCH.md)

## 🧠 **Semantic Comparison Service**

Integrated AI-powered semantic matching microservice for intelligent item comparison:

- **Multiple Matching Strategies**: Exact → Semantic → Hybrid
- **VLM-Powered**: Uses OVMS for semantic reasoning
- **Automatic Fallback**: Falls back to local matching if service unavailable
- **Caching**: Memory/Redis cache for performance
- **Metrics**: Prometheus metrics at port 9090

**Example:** Matches "green apple" ↔ "apple" using semantic reasoning

See [SEMANTIC_SERVICE_INTEGRATION.md](SEMANTIC_SERVICE_INTEGRATION.md) for details.

---

## 📦 **What the system does**

* Accepts **video file uploads** or **RTSP streams**
* Extracts frames using **GStreamer + gvapython**
* Detects **order ID using OCR**
* Stores frames in **MinIO**
* Selects **Top-K frames** per order using **YOLO**
* Runs **VLM (OpenVINO GenAI)** for item & quantity extraction
* Provides a **Gradio UI** for interaction

---

## 📁 **Project Structure**

```
order-accuracy/
│
├── docker-compose.yaml           # Multi-service orchestration
├── config/
│   └── application.yaml          # Backend configuration
│
├── ovms-service/                 # OVMS model server (optional)
│   ├── setup_models.sh           # Model setup script
│   ├── export_model.py           # Export HF models to OpenVINO
│   ├── export_requirements.txt   # Model export dependencies
│   ├── models_vlm/               # OVMS model repository
│   │   ├── config.json           # OVMS configuration
│   │   └── Qwen/                 # Model files (after setup)
│   └── README.md                 # OVMS setup documentation
│
├── application-service/
│   ├── Dockerfile
│   └── app/
│       ├── main.py               # API + pipeline trigger
│       ├── vlm_service.py        # VLM inference service
│       ├── vlm_backend_factory.py # Backend factory pattern
│       ├── ovms_client.py        # OVMS HTTP client
│       ├── pipeline_runner.py    # GStreamer launcher
│       ├── frame_pipeline.py     # OCR + frame upload
│       └── requirements.txt
│
├── frame-selector-service/
│   ├── Dockerfile
│   └── app/
│       ├── frame_selector.py     # Selects top frames
│       └── requirements.txt
│
├── gradio-ui/
│   ├── Dockerfile
│   └── gradio_app.py             # Web UI
│
├── model/                        # Embedded VLM model (optional)
│   └── Qwen2.5-VL-7B-Instruct-ov-int8/
│
└── storage/
    ├── videos/
    └── uploads/
```

---

## ▶️ **How to Run**

### **Option 1: Embedded VLM (Default)**

```bash
# Start all services
docker-compose up --build -d
```

### **Option 2: OVMS Backend**

```bash
# 1. Set up OVMS models (first time only)
cd ovms-service
./setup_models.sh
cd ..

# 2. Change backend in config/application.yaml:
#    vlm:
#      backend: ovms

# 3. Change environment in docker-compose.yaml:
#    VLM_BACKEND: ovms

# 4. Start services with OVMS
docker-compose --profile ovms up --build -d
```

**Verify OVMS is running:**
```bash
curl http://localhost:8001/v1/config
curl http://localhost:8001/v1/models
```

This launches:

* **MinIO** (frame storage)
* **Application Service** (GStreamer + OCR + VLM API)
* **Frame Selector Service** (YOLO ranking)
* **Gradio UI**
* **OVMS VLM Service** (when using OVMS backend)

---

Login for MinIO:

```
minioadmin / minioadmin
```

---

## 🎥 **How to Use**

### **Upload a Video (UI)**

1. Open Gradio UI
2. Upload `.mp4 / .avi / .mkv`
3. Click **Upload & Start**

The pipeline starts automatically.

---

### **RTSP Stream**

RTSP example:

```
rtsp://192.168.1.5:8554/test
```

API call:

```bash
curl -X POST http://localhost:8000/run-video \
  -H "Content-Type: application/json" \
  -d '{"source_type":"rtsp","source":"rtsp://192.168.1.5:8554/test"}'
```

> If `localhost` is provided in RTSP, the backend safely normalizes it for Docker.

---

## 🖼 **View Frames in MinIO**

### Extracted Frames

```
frames/
 └── <order_id>/
      ├── 11.jpg
      ├── 42.jpg
      └── 76.jpg
```

### Selected Frames

```
selected/
 └── <order_id>/
      ├── rank_1.jpg
      ├── rank_2.jpg
      └── rank_3.jpg
```

---

## 🔄 **Clean Restart (Recommended)**

```bash
docker compose down --remove-orphans
docker volume rm order-accuracy_minio_data
docker compose up --build
```

⚠️ This deletes all stored frames.

---

## ✅ **TL;DR**

```bash
docker compose up --build
open http://localhost:7860
```

Upload video or RTSP → frames extracted → top frames selected → VLM results available.
