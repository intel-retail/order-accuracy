"""
Metrics Collection and Monitoring

Tracks system metrics for autoscaling decisions:
- CPU utilization
- GPU utilization (pluggable interface)
- Order latency (rolling average)
- Queue depths
- Throughput

Thread-safe metrics storage using multiprocessing.Manager.
"""

import multiprocessing as mp
from multiprocessing.managers import SyncManager
from typing import Dict, Optional, List
import time
import psutil
import logging
from collections import deque
from abc import ABC, abstractmethod
import threading

logger = logging.getLogger(__name__)


class GPUMetricsProvider(ABC):
    """Abstract interface for GPU metrics collection"""
    
    @abstractmethod
    def get_utilization(self) -> float:
        """Return GPU utilization percentage (0-100)"""
        pass
    
    @abstractmethod
    def get_memory_used(self) -> float:
        """Return GPU memory used in GB"""
        pass
    
    @abstractmethod
    def get_memory_total(self) -> float:
        """Return total GPU memory in GB"""
        pass


class IntelGPUMetrics(GPUMetricsProvider):
    """Intel GPU metrics using intel_gpu_top or similar tools"""
    
    def __init__(self):
        self._available = self._check_availability()
        self._last_utilization = 0.0
        self._last_memory_used = 0.0
        
        if self._available:
            logger.info("Intel GPU metrics provider initialized")
        else:
            logger.warning("Intel GPU metrics not available, using placeholder")
    
    def _check_availability(self) -> bool:
        """Check if GPU monitoring tools are available"""
        try:
            import subprocess
            result = subprocess.run(
                ['intel_gpu_top', '-h'],
                capture_output=True,
                timeout=2
            )
            return result.returncode == 0
        except Exception as e:
            logger.debug(f"intel_gpu_top not available: {e}")
            return False
    
    def get_utilization(self) -> float:
        """Get GPU utilization from intel_gpu_top"""
        if not self._available:
            # Return simulated value based on system load
            return min(psutil.cpu_percent() * 1.2, 100.0)
        
        try:
            import subprocess
            # Run intel_gpu_top for 1 second sample
            result = subprocess.run(
                ['intel_gpu_top', '-s', '100', '-n', '1'],
                capture_output=True,
                text=True,
                timeout=2
            )
            
            # Parse output (example: "Render/3D: 45.6%")
            for line in result.stdout.split('\n'):
                if 'Render/3D' in line or 'GPU' in line:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        percent_str = parts[1].strip().rstrip('%')
                        try:
                            self._last_utilization = float(percent_str)
                            return self._last_utilization
                        except ValueError:
                            pass
            
            return self._last_utilization
        
        except Exception as e:
            logger.debug(f"Failed to get GPU utilization: {e}")
            return self._last_utilization
    
    def get_memory_used(self) -> float:
        """Get GPU memory usage (placeholder - Intel doesn't expose this easily)"""
        # Intel GPUs use unified memory, approximate based on system memory
        mem = psutil.virtual_memory()
        # Assume GPU uses ~10% of system memory for inference workloads
        self._last_memory_used = (mem.total - mem.available) / (1024**3) * 0.1
        return self._last_memory_used
    
    def get_memory_total(self) -> float:
        """Get total GPU memory (placeholder)"""
        # Intel Arc GPUs typically have 8-16GB dedicated
        return 16.0


class PlaceholderGPUMetrics(GPUMetricsProvider):
    """Placeholder GPU metrics when hardware access unavailable"""
    
    def get_utilization(self) -> float:
        """Estimate GPU utilization from CPU utilization"""
        # Assume GPU scales with CPU for inference workloads
        cpu_util = psutil.cpu_percent(interval=0.1)
        return min(cpu_util * 1.5, 100.0)
    
    def get_memory_used(self) -> float:
        """Placeholder memory used"""
        return 4.0
    
    def get_memory_total(self) -> float:
        """Placeholder total memory"""
        return 16.0


class MetricsStore:
    """
    Thread-safe metrics storage using multiprocessing.Manager.
    
    Stores:
    - CPU utilization (rolling average)
    - GPU utilization (rolling average)
    - Order latencies per station
    - Queue depths
    - Throughput counters
    """
    
    def __init__(self, window_size: int = 30):
        """
        Args:
            window_size: Number of samples for rolling averages (30 = 30 seconds at 1Hz)
        """
        manager = SyncManager()
        manager.start()
        
        # Shared dictionaries
        self._data = manager.dict()
        self._lock = manager.Lock()
        
        self.window_size = window_size
        
        # Initialize metrics
        with self._lock:
            self._data['cpu_samples'] = manager.list()
            self._data['gpu_samples'] = manager.list()
            self._data['latencies'] = manager.dict()  # {station_id: [latency_samples]}
            self._data['throughput'] = manager.dict()  # {station_id: order_count}
            self._data['queue_depths'] = manager.dict()
            self._data['active_stations'] = manager.list()
            self._data['total_orders'] = 0
            self._data['failed_orders'] = 0
        
        logger.info(f"MetricsStore initialized (window_size={window_size})")
    
    def record_cpu(self, utilization: float):
        """Record CPU utilization sample"""
        with self._lock:
            samples = list(self._data['cpu_samples'])
            samples.append(utilization)
            if len(samples) > self.window_size:
                samples.pop(0)
            self._data['cpu_samples'] = samples
    
    def record_gpu(self, utilization: float):
        """Record GPU utilization sample"""
        with self._lock:
            samples = list(self._data['gpu_samples'])
            samples.append(utilization)
            if len(samples) > self.window_size:
                samples.pop(0)
            self._data['gpu_samples'] = samples
    
    def record_latency(self, station_id: str, latency: float):
        """Record order latency for station"""
        with self._lock:
            latencies = dict(self._data['latencies'])
            if station_id not in latencies:
                latencies[station_id] = []
            
            station_latencies = latencies[station_id]
            station_latencies.append(latency)
            if len(station_latencies) > self.window_size:
                station_latencies.pop(0)
            
            latencies[station_id] = station_latencies
            self._data['latencies'] = latencies
    
    def increment_throughput(self, station_id: str):
        """Increment order count for station"""
        with self._lock:
            throughput = dict(self._data['throughput'])
            throughput[station_id] = throughput.get(station_id, 0) + 1
            self._data['throughput'] = throughput
            self._data['total_orders'] = self._data['total_orders'] + 1
    
    def increment_failures(self):
        """Increment failed order count"""
        with self._lock:
            self._data['failed_orders'] = self._data['failed_orders'] + 1
    
    def update_queue_depth(self, queue_name: str, depth: int):
        """Update queue depth metric"""
        with self._lock:
            depths = dict(self._data['queue_depths'])
            depths[queue_name] = depth
            self._data['queue_depths'] = depths
    
    def register_station(self, station_id: str):
        """Register active station"""
        with self._lock:
            stations = list(self._data['active_stations'])
            if station_id not in stations:
                stations.append(station_id)
                self._data['active_stations'] = stations
    
    def unregister_station(self, station_id: str):
        """Unregister stopped station"""
        with self._lock:
            stations = list(self._data['active_stations'])
            if station_id in stations:
                stations.remove(station_id)
                self._data['active_stations'] = stations
    
    def get_cpu_avg(self) -> float:
        """Get average CPU utilization"""
        with self._lock:
            samples = list(self._data['cpu_samples'])
            return sum(samples) / len(samples) if samples else 0.0
    
    def get_gpu_avg(self) -> float:
        """Get average GPU utilization"""
        with self._lock:
            samples = list(self._data['gpu_samples'])
            return sum(samples) / len(samples) if samples else 0.0
    
    def get_latency_avg(self, station_id: Optional[str] = None) -> float:
        """
        Get average latency.
        
        Args:
            station_id: If provided, return latency for specific station.
                       Otherwise, return global average across all stations.
        """
        with self._lock:
            latencies_dict = dict(self._data['latencies'])
            
            if station_id:
                station_latencies = latencies_dict.get(station_id, [])
                return sum(station_latencies) / len(station_latencies) if station_latencies else 0.0
            else:
                # Global average
                all_latencies = []
                for station_latencies in latencies_dict.values():
                    all_latencies.extend(station_latencies)
                return sum(all_latencies) / len(all_latencies) if all_latencies else 0.0
    
    def get_latency_p95(self, station_id: Optional[str] = None) -> float:
        """Get 95th percentile latency"""
        with self._lock:
            latencies_dict = dict(self._data['latencies'])
            
            if station_id:
                station_latencies = latencies_dict.get(station_id, [])
                all_latencies = station_latencies
            else:
                all_latencies = []
                for station_latencies in latencies_dict.values():
                    all_latencies.extend(station_latencies)
            
            if not all_latencies:
                return 0.0
            
            sorted_latencies = sorted(all_latencies)
            idx = int(len(sorted_latencies) * 0.95)
            return sorted_latencies[idx]
    
    def get_throughput(self, station_id: Optional[str] = None) -> int:
        """Get throughput (total orders processed)"""
        with self._lock:
            if station_id:
                throughput = dict(self._data['throughput'])
                return throughput.get(station_id, 0)
            else:
                return self._data['total_orders']
    
    def get_active_stations(self) -> List[str]:
        """Get list of active station IDs"""
        with self._lock:
            return list(self._data['active_stations'])
    
    def get_snapshot(self) -> Dict:
        """Get complete metrics snapshot"""
        with self._lock:
            return {
                'cpu_avg': self.get_cpu_avg(),
                'gpu_avg': self.get_gpu_avg(),
                'latency_avg': self.get_latency_avg(),
                'latency_p95': self.get_latency_p95(),
                'total_orders': self._data['total_orders'],
                'failed_orders': self._data['failed_orders'],
                'active_stations': len(list(self._data['active_stations'])),
                'queue_depths': dict(self._data['queue_depths']),
                'throughput': dict(self._data['throughput'])
            }


class MetricsCollector:
    """
    Background metrics collector.
    
    Runs in separate thread, periodically samples system metrics
    and updates shared MetricsStore.
    """
    
    def __init__(
        self,
        metrics_store: MetricsStore,
        gpu_provider: Optional[GPUMetricsProvider] = None,
        sample_interval: float = 1.0
    ):
        """
        Args:
            metrics_store: Shared metrics storage
            gpu_provider: GPU metrics provider (default: auto-detect)
            sample_interval: Sampling interval in seconds
        """
        self.metrics_store = metrics_store
        self.sample_interval = sample_interval
        
        # Initialize GPU provider
        if gpu_provider is None:
            try:
                self.gpu_provider = IntelGPUMetrics()
            except Exception as e:
                logger.warning(f"Failed to initialize Intel GPU metrics: {e}")
                self.gpu_provider = PlaceholderGPUMetrics()
        else:
            self.gpu_provider = gpu_provider
        
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        logger.info(f"MetricsCollector initialized (interval={sample_interval}s)")
    
    def start(self):
        """Start background metrics collection"""
        if self._running:
            logger.warning("MetricsCollector already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()
        
        logger.info("MetricsCollector started")
    
    def stop(self):
        """Stop background metrics collection"""
        if not self._running:
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info("MetricsCollector stopped")
    
    def _collect_loop(self):
        """Background collection loop"""
        logger.info("Metrics collection loop started")
        
        while self._running:
            try:
                # Sample CPU
                cpu_util = psutil.cpu_percent(interval=0.1)
                self.metrics_store.record_cpu(cpu_util)
                
                # Sample GPU
                gpu_util = self.gpu_provider.get_utilization()
                self.metrics_store.record_gpu(gpu_util)
                
                # Log current metrics
                logger.debug(
                    f"Metrics: CPU={cpu_util:.1f}% GPU={gpu_util:.1f}% "
                    f"Stations={len(self.metrics_store.get_active_stations())}"
                )
                
                time.sleep(self.sample_interval)
            
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.sample_interval)
        
        logger.info("Metrics collection loop stopped")
