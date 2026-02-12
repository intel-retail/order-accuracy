"""
Benchmark Service implementing dynamic worker scaling.
Uses Observer pattern for metrics monitoring and auto-scaling decisions.
"""

import asyncio
import logging
import time
import psutil
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import statistics

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    """Metrics for benchmark monitoring"""
    timestamp: datetime
    active_workers: int
    completed_validations: int
    failed_validations: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    cpu_percent: float
    gpu_percent: float
    throughput_per_sec: float


@dataclass
class WorkerStats:
    """Statistics for individual worker"""
    worker_id: int
    validations_completed: int = 0
    total_time_ms: float = 0.0
    last_latency_ms: float = 0.0
    errors: int = 0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_time_ms / self.validations_completed if self.validations_completed > 0 else 0.0


class MetricsCollector:
    """Collects and analyzes performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)
        self.worker_stats: Dict[int, WorkerStats] = {}
        self.total_completed = 0
        self.total_failed = 0
    
    def record_validation(self, worker_id: int, latency_ms: float, success: bool):
        """Record validation completion"""
        self.latencies.append(latency_ms)
        self.timestamps.append(time.time())
        
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
        
        stats = self.worker_stats[worker_id]
        if success:
            stats.validations_completed += 1
            stats.total_time_ms += latency_ms
            stats.last_latency_ms = latency_ms
            self.total_completed += 1
        else:
            stats.errors += 1
            self.total_failed += 1
    
    def get_metrics(self) -> Optional[BenchmarkMetrics]:
        """Calculate current metrics"""
        if not self.latencies:
            return None
        
        # Calculate latency percentiles
        sorted_latencies = sorted(self.latencies)
        avg_latency = statistics.mean(sorted_latencies)
        p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)] if len(sorted_latencies) > 1 else avg_latency
        p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)] if len(sorted_latencies) > 1 else avg_latency
        
        # Calculate throughput
        if len(self.timestamps) > 1:
            time_window = self.timestamps[-1] - self.timestamps[0]
            throughput = len(self.latencies) / time_window if time_window > 0 else 0
        else:
            throughput = 0
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        # Try to get GPU utilization (if available)
        gpu_percent = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_percent = gpu_util.gpu
        except:
            # GPU monitoring not available
            pass
        
        return BenchmarkMetrics(
            timestamp=datetime.now(),
            active_workers=len(self.worker_stats),
            completed_validations=self.total_completed,
            failed_validations=self.total_failed,
            avg_latency_ms=avg_latency,
            p95_latency_ms=p95_latency,
            p99_latency_ms=p99_latency,
            cpu_percent=cpu_percent,
            gpu_percent=gpu_percent,
            throughput_per_sec=throughput
        )
    
    def get_worker_stats(self) -> List[Dict[str, Any]]:
        """Get statistics for all workers"""
        return [
            {
                "worker_id": stats.worker_id,
                "completed": stats.validations_completed,
                "avg_latency_ms": round(stats.avg_latency_ms, 2),
                "last_latency_ms": round(stats.last_latency_ms, 2),
                "errors": stats.errors
            }
            for stats in self.worker_stats.values()
        ]


class AutoScaler:
    """Implements auto-scaling logic based on metrics"""
    
    def __init__(self, config):
        self.config = config
        self.last_scale_time = time.time()
        self.scale_cooldown = 30  # seconds
    
    def should_scale_up(self, metrics: BenchmarkMetrics) -> bool:
        """Determine if workers should be scaled up"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale up if latency exceeds threshold
        if metrics.avg_latency_ms > self.config.target_latency_ms * self.config.scale_up_threshold:
            logger.info(f"Scale up triggered: latency {metrics.avg_latency_ms:.2f}ms > "
                       f"{self.config.target_latency_ms * self.config.scale_up_threshold:.2f}ms")
            return True
        
        # Scale up if CPU/GPU below threshold (underutilized)
        if (metrics.cpu_percent < self.config.cpu_threshold_percent * 0.5 and
            metrics.active_workers < self.config.max_workers):
            logger.info(f"Scale up triggered: CPU underutilized ({metrics.cpu_percent:.1f}%)")
            return True
        
        return False
    
    def should_scale_down(self, metrics: BenchmarkMetrics) -> bool:
        """Determine if workers should be scaled down"""
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale down if latency well below target and multiple workers
        if (metrics.avg_latency_ms < self.config.target_latency_ms * self.config.scale_down_threshold and
            metrics.active_workers > self.config.min_workers):
            logger.info(f"Scale down triggered: latency {metrics.avg_latency_ms:.2f}ms < "
                       f"{self.config.target_latency_ms * self.config.scale_down_threshold:.2f}ms")
            return True
        
        # Scale down if resources overutilized
        if metrics.cpu_percent > self.config.cpu_threshold_percent:
            logger.info(f"Scale down triggered: CPU overutilized ({metrics.cpu_percent:.1f}%)")
            return True
        
        return False
    
    def mark_scaled(self):
        """Mark that scaling action was taken"""
        self.last_scale_time = time.time()


class BenchmarkService:
    """
    Benchmark service with dynamic worker scaling.
    Implements Observer pattern for metrics monitoring.
    """
    
    def __init__(self, config, validation_service, test_images: List[bytes], test_orders: List[Dict]):
        self.config = config.benchmark
        self.validation_service = validation_service
        self.test_images = test_images
        self.test_orders = test_orders
        
        self.metrics_collector = MetricsCollector()
        self.auto_scaler = AutoScaler(self.config)
        
        self.active_workers: List[asyncio.Task] = []
        self.should_stop = False
        self.worker_id_counter = 0
        
        logger.info(f"Benchmark Service initialized: "
                   f"initial_workers={self.config.initial_workers}, "
                   f"max_workers={self.config.max_workers}")
    
    async def _worker(self, worker_id: int):
        """Individual worker processing validations"""
        logger.info(f"Worker {worker_id} started")
        
        try:
            while not self.should_stop:
                # Round-robin through test images
                for img_idx, (image_bytes, order) in enumerate(zip(self.test_images, self.test_orders)):
                    if self.should_stop:
                        break
                    
                    start_time = time.time()
                    try:
                        # Perform validation
                        image_id = f"benchmark_w{worker_id}_i{img_idx}_{int(time.time())}"
                        result = await self.validation_service.validate_plate(
                            image_bytes, order, image_id
                        )
                        
                        latency_ms = (time.time() - start_time) * 1000
                        self.metrics_collector.record_validation(worker_id, latency_ms, True)
                        
                        logger.debug(f"Worker {worker_id} completed validation: "
                                   f"latency={latency_ms:.2f}ms, "
                                   f"accuracy={result.accuracy_score:.2f}")
                        
                    except Exception as e:
                        latency_ms = (time.time() - start_time) * 1000
                        self.metrics_collector.record_validation(worker_id, latency_ms, False)
                        logger.error(f"Worker {worker_id} validation failed: {e}")
                    
                    # Small delay between validations
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info(f"Worker {worker_id} cancelled")
        except Exception as e:
            logger.exception(f"Worker {worker_id} crashed: {e}")
        finally:
            logger.info(f"Worker {worker_id} stopped")
    
    async def _monitor_and_scale(self):
        """Monitor metrics and adjust worker count"""
        logger.info("Metrics monitor started")
        
        while not self.should_stop:
            await asyncio.sleep(self.config.check_interval_seconds)
            
            metrics = self.metrics_collector.get_metrics()
            if not metrics:
                continue
            
            # Log current metrics
            logger.info(f"Benchmark Metrics: "
                       f"workers={metrics.active_workers}, "
                       f"completed={metrics.completed_validations}, "
                       f"failed={metrics.failed_validations}, "
                       f"avg_latency={metrics.avg_latency_ms:.2f}ms, "
                       f"p95={metrics.p95_latency_ms:.2f}ms, "
                       f"cpu={metrics.cpu_percent:.1f}%, "
                       f"throughput={metrics.throughput_per_sec:.2f}/s")
            
            # Check if scaling needed
            current_workers = len(self.active_workers)
            
            if self.auto_scaler.should_scale_up(metrics) and current_workers < self.config.max_workers:
                await self._add_worker()
                self.auto_scaler.mark_scaled()
            elif self.auto_scaler.should_scale_down(metrics) and current_workers > self.config.min_workers:
                await self._remove_worker()
                self.auto_scaler.mark_scaled()
    
    async def _add_worker(self):
        """Add a new worker"""
        worker_id = self.worker_id_counter
        self.worker_id_counter += 1
        
        task = asyncio.create_task(self._worker(worker_id))
        self.active_workers.append(task)
        
        logger.info(f"Added worker {worker_id}, total workers: {len(self.active_workers)}")
    
    async def _remove_worker(self):
        """Remove a worker"""
        if self.active_workers:
            task = self.active_workers.pop()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            
            logger.info(f"Removed worker, remaining workers: {len(self.active_workers)}")
    
    async def start(self):
        """Start benchmark with initial workers"""
        logger.info("Starting benchmark service")
        
        if not self.test_images:
            raise ValueError("No test images available for benchmarking")
        
        # Start initial workers
        for _ in range(self.config.initial_workers):
            await self._add_worker()
        
        # Start monitoring
        monitor_task = asyncio.create_task(self._monitor_and_scale())
        
        # Wait for completion (or manual stop)
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
    
    async def stop(self):
        """Stop all workers"""
        logger.info("Stopping benchmark service")
        self.should_stop = True
        
        # Cancel all workers
        for task in self.active_workers:
            task.cancel()
        
        # Wait for all to finish
        await asyncio.gather(*self.active_workers, return_exceptions=True)
        self.active_workers.clear()
        
        logger.info("Benchmark service stopped")
    
    def get_report(self) -> Dict[str, Any]:
        """Get comprehensive benchmark report"""
        metrics = self.metrics_collector.get_metrics()
        worker_stats = self.metrics_collector.get_worker_stats()
        
        return {
            "status": "running" if not self.should_stop else "stopped",
            "current_metrics": metrics.__dict__ if metrics else {},
            "worker_stats": worker_stats,
            "config": {
                "initial_workers": self.config.initial_workers,
                "max_workers": self.config.max_workers,
                "target_latency_ms": self.config.target_latency_ms,
                "cpu_threshold": self.config.cpu_threshold_percent,
                "gpu_threshold": self.config.gpu_threshold_percent
            }
        }
