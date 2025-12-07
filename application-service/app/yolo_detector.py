from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name="yolov8n.pt"):
        self.model = YOLO(model_name)

    def count_items(self, frame):
        result = self.model.predict(frame, conf=0.2, verbose=False)[0]
        count = 0
        for box in result.boxes:
            label = self.model.names[int(box.cls)].lower()
            if label not in ("hand", "person"):
                count += 1
        return count
