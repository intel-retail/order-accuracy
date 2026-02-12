"""
Configuration Management

Centralized configuration for parallel pipeline system.
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ScalingConfig:
    """Autoscaling configuration"""
    enabled: bool = True
    scale_up_gpu_threshold: float = 85.0
    scale_up_cpu_threshold: float = 80.0
    scale_up_latency_threshold: float = 5.0
    scale_down_gpu_threshold: float = 95.0
    scale_down_cpu_threshold: float = 90.0
    scale_down_latency_threshold: float = 5.0
    hysteresis_window: float = 30.0
    min_stations: int = 1
    max_stations: int = 8


@dataclass
class VLMConfig:
    """VLM inference configuration"""
    ovms_url: str = "http://localhost:8000"
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"  # Must match OVMS config
    batch_window_ms: int = 100
    max_batch_size: int = 16
    max_workers: int = 4
    timeout_seconds: float = 30.0


@dataclass
class StorageConfig:
    """Storage configuration"""
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "orders"
    minio_secure: bool = False
    frames_bucket: str = "frames"  # Added for frame storage


@dataclass
class SystemConfig:
    """Complete system configuration"""
    
    # RTSP streams
    rtsp_urls: list = None
    
    # Model paths
    yolo_model_path: str = "./models/yolo11n_openvino_model"
    
    # Data paths
    inventory_path: str = "./config/inventory.json"
    orders_path: str = "./config/orders.json"
    
    # Queue backend
    queue_backend: str = "multiprocessing"  # or "redis"
    redis_host: str = "localhost"
    redis_port: int = 6379
    
    # Metrics
    metrics_sample_interval: float = 1.0
    metrics_window_size: int = 30
    
    # Control loop
    control_loop_interval: float = 5.0
    
    # Nested configs
    scaling: ScalingConfig = None
    vlm: VLMConfig = None
    storage: StorageConfig = None
    
    def __post_init__(self):
        if self.rtsp_urls is None:
            self.rtsp_urls = []
        if self.scaling is None:
            self.scaling = ScalingConfig()
        if self.vlm is None:
            self.vlm = VLMConfig()
        if self.storage is None:
            self.storage = StorageConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'rtsp_urls': self.rtsp_urls,
            'yolo_model_path': self.yolo_model_path,
            'inventory_path': self.inventory_path,
            'orders_path': self.orders_path,
            'queue_backend': self.queue_backend,
            'redis_host': self.redis_host,
            'redis_port': self.redis_port,
            'metrics_sample_interval': self.metrics_sample_interval,
            'metrics_window_size': self.metrics_window_size,
            'control_loop_interval': self.control_loop_interval,
            'scaling': asdict(self.scaling),
            'vlm': asdict(self.vlm),
            'storage': asdict(self.storage)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Create from dictionary"""
        scaling = ScalingConfig(**data.get('scaling', {}))
        vlm = VLMConfig(**data.get('vlm', {}))
        storage = StorageConfig(**data.get('storage', {}))
        
        return cls(
            rtsp_urls=data.get('rtsp_urls', []),
            yolo_model_path=data.get('yolo_model_path', './models/yolo11n_openvino_model'),
            inventory_path=data.get('inventory_path', './config/inventory.json'),
            orders_path=data.get('orders_path', './config/orders.json'),
            queue_backend=data.get('queue_backend', 'multiprocessing'),
            redis_host=data.get('redis_host', 'localhost'),
            redis_port=data.get('redis_port', 6379),
            metrics_sample_interval=data.get('metrics_sample_interval', 1.0),
            metrics_window_size=data.get('metrics_window_size', 30),
            control_loop_interval=data.get('control_loop_interval', 5.0),
            scaling=scaling,
            vlm=vlm,
            storage=storage
        )
    
    @classmethod
    def from_yaml(cls, filepath: str) -> 'SystemConfig':
        """Load from YAML file"""
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    @classmethod
    def from_json(cls, filepath: str) -> 'SystemConfig':
        """Load from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def save_yaml(self, filepath: str):
        """Save to YAML file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def save_json(self, filepath: str):
        """Save to JSON file"""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


def create_default_config() -> SystemConfig:
    """Create default configuration using environment variables"""
    # Get RTSP server configuration from environment
    rtsp_host = os.getenv('RTSP_STREAMER_HOST', 'rtsp-streamer')
    rtsp_port = os.getenv('RTSP_STREAMER_PORT', '8554')
    
    # Available RTSP streams (based on video files in rtsp-streamer)
    # These are the actual stream names published by the RTSP server
    available_streams = ['384-651-925', 'video1', 'video2']
    
    # Generate RTSP URLs for multiple stations, cycling through available streams
    num_stations = int(os.getenv('WORKERS', '4'))
    rtsp_urls = [
        f'rtsp://{rtsp_host}:{rtsp_port}/{available_streams[i % len(available_streams)]}'
        for i in range(num_stations)
    ]
    
    return SystemConfig(
        rtsp_urls=rtsp_urls,
        yolo_model_path='./models/yolo11n_openvino_model',
        inventory_path='./config/inventory.json',
        orders_path='./config/orders.json',
        queue_backend='multiprocessing'
    )


if __name__ == "__main__":
    # Create and save default config
    config = create_default_config()
    config.save_yaml('./config/system_config.yaml')
    print("Default configuration saved to ./config/system_config.yaml")
