"""Configuration settings for the grocery video app."""

import os
import logging

# ---------------- Base Configuration -----------------
VSS_IP = os.environ.get("VSS_IP", "localhost")

# ---------------- API Endpoints -----------------
VIDEO_UPLOAD_URL = f"http://{VSS_IP}:12345/manager/videos"
PIPELINE_TRIGGER_URL = f"http://{VSS_IP}:8090/pipelines/user_defined_pipelines/object_detection"
PIPELINE_TRIGGER_URL_VERTICAL = f"http://{VSS_IP}:8090/pipelines/user_defined_pipelines/object_detection_vertical"
PIPELINES_BASE_URL = f"http://{VSS_IP}:8090/pipelines"
FRAME_BASE_URL_TEMPLATE = f"http://{VSS_IP}:12345/datastore/video-summary/{{frame_id}}/frame_{{chunk}}_{{frame}}.jpeg"
VIDEO_MINIO_LOCATION_TEMPLATE = f"http://{VSS_IP}:4001/video-summary/{{video_id}}/video.mp4"
MINIO_HOST = f"{VSS_IP}:4001"
VLM_URL = f"http://{VSS_IP}:9766/v1/chat/completions"

# ---------------- Model Configuration -----------------
VLM_MODEL = os.environ.get("VLM_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

# ---------------- RabbitMQ Configuration -----------------
RABBITMQ_HOST = os.environ.get("RABBITMQ_HOST", VSS_IP)
RABBITMQ_USER = os.environ.get("RABBITMQ_USER", "RabbitMQ")
RABBITMQ_PASSWORD = os.environ.get("RABBITMQ_PASSWORD", "RabbitPass")
RABBITMQ_TOPIC = os.environ.get("RABBITMQ_TOPIC", "topic/video_stream")
RABBITMQ_EXCHANGE = "amq.topic"  # Default MQTT topic exchange in RabbitMQ
# Convert topic from slash format to dot format for AMQP routing
RABBITMQ_ROUTING_KEY = RABBITMQ_TOPIC.replace("/", ".")

# ---------------- Application Settings -----------------
POLL_INTERVAL = 5  # seconds
MAX_VIDEO_DURATION = 70  # seconds

# ---------------- Server Configuration -----------------
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 7860

# ---------------- Logging Configuration -----------------
def setup_logging():
    """Configure logging with detailed format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('order_accuracy_app.log')
        ]
    )
    return logging.getLogger("order_accuracy_app")

# Create logger instance
logger = setup_logging()
