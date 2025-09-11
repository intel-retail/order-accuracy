"""
Order Accuracy App - A modular application for processing grocery checkout videos.

This package provides functionality to:
- Upload videos to a processing service
- Trigger object detection pipelines
- Collect metadata via RabbitMQ messaging
- Analyze frames using Vision Language Models
- Present results through a Gradio web interface

Modules:
- config: Configuration and settings
- video_utils: Video processing utilities  
- pipeline: Pipeline management functions
- messaging: RabbitMQ/MQTT messaging functionality
- vlm: Vision Language Model integration
- interface: Gradio user interface components
- main: Application entry point
"""

__version__ = "1.0.0"
__author__ = "Retail Agent"

# Import main components for easy access
from .config import logger
from .main import main
from .interface import build_interface

__all__ = ["main", "build_interface", "logger"]
