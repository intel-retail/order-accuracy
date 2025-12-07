# **Order Accuracy**

This project processes a video, extracts valid order-ID frames, uploads them to **MinIO**, and then selects the **top frames** per order based on item count.

---

## 📦 **Project Structure**

```
order-accuracy/
│
├── docker-compose.yaml
│
├── application-service/
│   ├── Dockerfile
│   └── app/
│       ├── main.py               # Extracts frames, OCR, YOLO item detection
│       ├── ocr_reader.py
│       ├── frame_preprocessor.py
│       └── requirements.txt
│
├── frame-selector-service/
│   ├── Dockerfile
│   └── app/
│       ├── frame_selector.py     # Selects top frames
│       └── requirements.txt
│
└── storage/
    └── videos/
        └── sample.mp4            # Input video (replace with your own)
```

---

## ▶️ **How to Run**

### **1. Add your input video**

Place your video here:

```
storage/videos/sample.mp4
```

(or modify `VIDEO_SOURCE` in the compose file)

---

### **2. Start all services**

```bash
docker compose up --build
```

This launches:

* **MinIO**
* **Application Service** → Extracts valid frames to MinIO
* **Frame Selector Service** → Picks best frames and writes to MinIO

---

## 🖼 View frames in MinIO UI

Open:

```
http://localhost:9001
```

Login:

```
minioadmin / minioadmin
```

### **Extracted Frames (input frames)**

```
frames/
 └── <order_id>/
      ├── 11.jpg
      ├── 42.jpg
      └── 76.jpg
```

### **Selected Frames (top frames)**

```
selected/
 └── <order_id>/
      ├── rank_1.jpg
      ├── rank_2.jpg
      └── rank_3.jpg
```

---

## 🔄 **Clean Restart (recommended)**

Sometimes MinIO retains old state.
Use these commands for a fresh restart.

### **1. Stop all services**

```bash
docker compose down
```

### **2. Remove orphan containers**

```bash
docker compose down --remove-orphans
```

### **3. Remove dangling images/containers**

```bash
docker system prune -f
```

### **4. Remove MinIO volume completely**

> **WARNING:** This deletes all previously stored frames.

```bash
docker volume rm order-accuracy_minio_data
```

### **5. Restart the complete system**

```bash
docker compose up --build
```

---

## 🔁 **Run again with a different video**

1. Replace `storage/videos/sample.mp4`
2. Clean restart (optional)
3. Run `docker compose up --build`

---