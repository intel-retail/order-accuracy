"""
Configuration management using Singleton pattern for application settings.
"""

import os
from typing import Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ServiceConfig:
    """Service endpoint configurations"""
    ovms_endpoint: str
    ovms_model_name: str
    semantic_service_endpoint: str
    metrics_collector_endpoint: str
    api_timeout: int


@dataclass
class BenchmarkConfig:
    """Benchmark mode configurations"""
    enabled: bool
    initial_workers: int
    max_workers: int
    min_workers: int
    target_latency_ms: float
    max_latency_ms: float
    cpu_threshold_percent: float
    gpu_threshold_percent: float
    scale_up_threshold: float
    scale_down_threshold: float
    check_interval_seconds: int
    warmup_requests: int


@dataclass
class AppConfig:
    """Application configuration"""
    log_level: str
    service: ServiceConfig
    benchmark: BenchmarkConfig


class ConfigManager:
    """
    Singleton configuration manager.
    Provides centralized access to application configuration.
    """
    _instance: Optional['ConfigManager'] = None
    _config: Optional[AppConfig] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._config is None:
            self._load_config()

    def _load_config(self):
        """Load configuration from environment variables with defaults"""
        logger.info("Loading application configuration")
        
        service_config = ServiceConfig(
            ovms_endpoint=os.getenv("OVMS_ENDPOINT", "http://ovms-vlm:8000"),
            ovms_model_name=os.getenv("OVMS_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct-ov-int8"),
            semantic_service_endpoint=os.getenv("SEMANTIC_SERVICE_ENDPOINT", "http://semantic-service:8080"),
            metrics_collector_endpoint=os.getenv("METRICS_COLLECTOR_ENDPOINT", "http://metrics-collector:8084"),
            api_timeout=int(os.getenv("API_TIMEOUT", "300"))  # Extended for 7B model with inventory
        )
        
        benchmark_config = BenchmarkConfig(
            enabled=os.getenv("BENCHMARK_MODE", "false").lower() == "true",
            initial_workers=int(os.getenv("BENCHMARK_INITIAL_WORKERS", "1")),
            max_workers=int(os.getenv("BENCHMARK_MAX_WORKERS", "10")),
            min_workers=int(os.getenv("BENCHMARK_MIN_WORKERS", "1")),
            target_latency_ms=float(os.getenv("BENCHMARK_TARGET_LATENCY_MS", "2000")),
            max_latency_ms=float(os.getenv("BENCHMARK_MAX_LATENCY_MS", "5000")),
            cpu_threshold_percent=float(os.getenv("BENCHMARK_CPU_THRESHOLD", "80.0")),
            gpu_threshold_percent=float(os.getenv("BENCHMARK_GPU_THRESHOLD", "80.0")),
            scale_up_threshold=float(os.getenv("BENCHMARK_SCALE_UP_THRESHOLD", "0.8")),
            scale_down_threshold=float(os.getenv("BENCHMARK_SCALE_DOWN_THRESHOLD", "0.5")),
            check_interval_seconds=int(os.getenv("BENCHMARK_CHECK_INTERVAL", "10")),
            warmup_requests=int(os.getenv("BENCHMARK_WARMUP_REQUESTS", "5"))
        )
        
        self._config = AppConfig(
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            service=service_config,
            benchmark=benchmark_config
        )
        
        logger.info(f"Configuration loaded: OVMS={service_config.ovms_endpoint}, "
                   f"Semantic={service_config.semantic_service_endpoint}, "
                   f"Benchmark={benchmark_config.enabled}")

    @property
    def config(self) -> AppConfig:
        """Get application configuration"""
        return self._config

    def update_benchmark_mode(self, enabled: bool):
        """Update benchmark mode at runtime"""
        logger.info(f"Updating benchmark mode: {enabled}")
        self._config.benchmark.enabled = enabled


# Global config instance
config_manager = ConfigManager()
