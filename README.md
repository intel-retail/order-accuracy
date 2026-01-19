# **Order Accuracy**

This project processes a video or RTSP stream, extracts **valid order-ID frames**, uploads them to **MinIO**, selects the **top frames per order**, and runs **VLM inference** to extract ordered items.

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
├── docker-compose.yaml
│
├── application-service/
│   ├── Dockerfile
│   └── app/
│       ├── main.py               # API + pipeline trigger
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
├── config/
│   └── application.yaml
│
├── model/
│   └── Qwen2.5-VL-7B-Instruct-ov-int8/
│
└── storage/
    ├── videos/
    └── uploads/
```

---

## ▶️ **How to Run**

### **1. Start all services**

```bash
docker compose up --build
```

This launches:

* **MinIO** (frame storage)
* **Application Service** (GStreamer + OCR + VLM API)
* **Frame Selector Service** (YOLO ranking)
* **Gradio UI**

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
