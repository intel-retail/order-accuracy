"""
Station Manager with Autoscaling

Manages pool of station worker processes with dynamic scaling based on:
- CPU utilization
- GPU utilization  
- Order latency
- Scaling policy

Responsibilities:
- Start/stop station workers
- Monitor system metrics
- Apply scaling policy
- Graceful worker lifecycle management
"""

import multiprocessing as mp
import time
import logging
from typing import Dict, List, Optional
import signal
import sys

from .shared_queue import QueueManager, QueueBackend
from .metrics_collector import MetricsCollector, MetricsStore
from .station_worker import start_worker_process

logger = logging.getLogger(__name__)


class StationManager:
    """
    Manages pool of station worker processes with autoscaling.
    
    Architecture:
    1. Maintains pool of worker processes (one per station)
    2. Monitors metrics via MetricsCollector
    3. Evaluates scaling decisions via ScalingPolicy
    4. Dynamically starts/stops workers
    5. Handles graceful shutdown
    """
    
    def __init__(
        self,
        config: Dict,
        queue_manager: QueueManager,
        initial_stations: int = 1
    ):
        """
        Initialize the station manager with configuration and scaling policy.
        
        Args:
            config: Global configuration dictionary with settings for:
                - rtsp_urls: List of RTSP stream URLs for cameras
                - minio_endpoint: MinIO storage endpoint
                - minio_bucket: Bucket name for frame storage
                - inventory_path: Path to inventory JSON file
                - orders_path: Path to orders JSON file
                - yolo_model_path: Path to YOLO model for frame selection
            queue_manager: Shared queue manager instance (REQUIRED)
            initial_stations: Number of station workers to start initially
        """
        self.config = config
        
        # Use shared infrastructure
        self.queue_manager = queue_manager
        self.metrics_store = MetricsStore(window_size=30)
        self.metrics_collector = MetricsCollector(
            metrics_store=self.metrics_store,
            sample_interval=1.0
        )
        
        # Worker pool
        self.workers: Dict[str, mp.Process] = {}
        self.worker_configs: Dict[str, Dict] = {}
        self._next_station_id = 1
        
        # Control
        self._running = False
        
        # Check scaling mode from config
        scaling_config = config.get('scaling', {})
        scaling_mode = scaling_config.get('mode', 'auto') if isinstance(scaling_config, dict) else 'auto'
        self._autoscaling_enabled = (scaling_mode != 'fixed')
        
        self._control_loop_interval = 5.0  # Check every 5 seconds
        
        # Statistics
        self._scaling_events = []
        
        logger.info(
            f"StationManager initialized: "
            f"initial_stations={initial_stations}, "
            f"scaling_mode={scaling_mode}, "
            f"autoscaling_enabled={self._autoscaling_enabled}"
        )
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        signal.signal(signal.SIGINT, self._handle_shutdown_signal)
        
        # Pre-create response queues for all potential stations
        # This ensures queues are created BEFORE worker processes spawn
        max_stations = config.get('scaling', {}).get('max_stations', 8) if isinstance(config.get('scaling'), dict) else 8
        station_ids = [f"station_{i+1}" for i in range(max_stations)]
        for station_id in station_ids:
            self.queue_manager.get_response_queue(station_id)
        logger.info(f"Pre-created response queues for {len(station_ids)} stations")
        
        # Start initial stations
        for i in range(initial_stations):
            self._start_worker()
    
    def start(self):
        """Start station manager"""
        if self._running:
            logger.warning("StationManager already running")
            return
        
        self._running = True
        
        # Start metrics collector
        self.metrics_collector.start()
        
        logger.info(
            f"StationManager started with {len(self.workers)} stations"
        )
        
        # Run control loop
        try:
            self._control_loop()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received")
        finally:
            self.stop()
    
    def stop(self):
        """Stop station manager and all workers"""
        if not self._running:
            return
        
        logger.info("Stopping StationManager...")
        self._running = False
        
        # Stop metrics collector
        self.metrics_collector.stop()
        
        # Stop all workers
        self._stop_all_workers()
        
        # Cleanup queues
        self.queue_manager.shutdown()
        
        logger.info("StationManager stopped")
    
    def _control_loop(self):
        """Main control loop for autoscaling"""
        logger.info("Control loop started")
        
        while self._running:
            try:
                time.sleep(self._control_loop_interval)
                
                # Update queue depth metrics only (no auto-scaling)
                self._update_queue_metrics()
            
            except Exception as e:
                logger.error(f"Error in control loop: {e}")
                time.sleep(self._control_loop_interval)
        
        logger.info("Control loop stopped")
    
    def _get_current_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            'cpu_avg': self.metrics_store.get_cpu_avg(),
            'gpu_avg': self.metrics_store.get_gpu_avg(),
            'latency_avg': self.metrics_store.get_latency_avg(),
            'latency_p95': self.metrics_store.get_latency_p95(),
            'active_stations': len(self.metrics_store.get_active_stations()),
            'total_orders': self.metrics_store.get_throughput()
        }
    
    def _start_worker(self) -> str:
        """
        Start new station worker process.
        
        Returns:
            Station ID of started worker
        """
        station_id = f"station_{self._next_station_id}"
        self._next_station_id += 1
        
        # Get RTSP URL
        # Cycle through available RTSP URLs or use simulation
        rtsp_urls = self.config.get('rtsp_urls', [])
        if rtsp_urls:
            rtsp_url = rtsp_urls[(self._next_station_id - 1) % len(rtsp_urls)]
        else:
            rtsp_url = f"rtsp://simulation/{station_id}"
        
        # Create worker configuration
        worker_config = {
            'minio': {
                'endpoint': self.config.get('minio_endpoint', 'minio:9000'),
                'bucket': self.config.get('minio_bucket', 'orders'),
                'frames_bucket': self.config.get('frames_bucket', 'frames'),
                'access_key': self.config.get('minio_access_key', 'minioadmin'),
                'secret_key': self.config.get('minio_secret_key', 'minioadmin'),
                'secure': self.config.get('minio_secure', False)
            },
            'inventory_path': self.config.get('inventory_path'),
            'orders_path': self.config.get('orders_path'),
            'yolo_model_path': self.config.get('yolo_model_path')
        }
        
        # Start worker process
        process = mp.Process(
            target=start_worker_process,
            args=(
                station_id,
                rtsp_url,
                self.queue_manager,
                self.metrics_store,
                worker_config
            ),
            name=f"Worker-{station_id}",
            daemon=False
        )
        
        process.start()
        
        self.workers[station_id] = process
        self.worker_configs[station_id] = {
            'rtsp_url': rtsp_url,
            'config': worker_config,
            'start_time': time.time()
        }
        
        logger.info(
            f"Started worker: {station_id} "
            f"(PID: {process.pid}, RTSP: {rtsp_url})"
        )
        
        return station_id
    
    def _stop_worker(self, station_id: Optional[str] = None):
        """
        Stop station worker process gracefully.
        
        Args:
            station_id: Specific station to stop, or None to stop oldest
        """
        if not self.workers:
            logger.warning("No workers to stop")
            return
        
        # Select worker to stop
        if station_id is None:
            # Stop oldest worker
            station_id = min(
                self.workers.keys(),
                key=lambda sid: self.worker_configs[sid]['start_time']
            )
        
        if station_id not in self.workers:
            logger.warning(f"Worker {station_id} not found")
            return
        
        process = self.workers[station_id]
        
        # Send graceful shutdown signal
        shutdown_signal = {
            'action': 'shutdown',
            'station_id': station_id
        }
        self.queue_manager.control_queue.put(shutdown_signal)
        
        logger.info(f"Stopping worker: {station_id} (PID: {process.pid})")
        
        # Wait for graceful shutdown
        process.join(timeout=10.0)
        
        if process.is_alive():
            logger.warning(
                f"Worker {station_id} did not stop gracefully, terminating..."
            )
            process.terminate()
            process.join(timeout=5.0)
            
            if process.is_alive():
                logger.error(
                    f"Worker {station_id} did not terminate, killing..."
                )
                process.kill()
                process.join()
        
        # Remove from pool
        del self.workers[station_id]
        del self.worker_configs[station_id]
        
        logger.info(f"Worker {station_id} stopped")
    
    def _stop_all_workers(self):
        """Stop all worker processes"""
        logger.info(f"Stopping all {len(self.workers)} workers...")
        
        # Send broadcast shutdown signal
        shutdown_signal = {
            'action': 'shutdown',
            'station_id': '*'  # Wildcard
        }
        self.queue_manager.control_queue.put(shutdown_signal)
        
        # Wait for all workers
        for station_id, process in list(self.workers.items()):
            logger.info(f"Waiting for worker: {station_id}")
            process.join(timeout=10.0)
            
            if process.is_alive():
                logger.warning(f"Terminating worker: {station_id}")
                process.terminate()
                process.join(timeout=5.0)
                
                if process.is_alive():
                    logger.error(f"Killing worker: {station_id}")
                    process.kill()
                    process.join()
        
        self.workers.clear()
        self.worker_configs.clear()
        
        logger.info("All workers stopped")
    
    def _update_queue_metrics(self):
        """Update queue depth metrics"""
        try:
            vlm_queue_depth = self.queue_manager.vlm_request_queue.qsize()
            self.metrics_store.update_queue_depth(
                'vlm_requests',
                vlm_queue_depth
            )
        except Exception as e:
            logger.debug(f"Failed to update queue metrics: {e}")
    
    def _handle_shutdown_signal(self, signum, frame):
        """Handle OS shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self._running = False
    
    def enable_autoscaling(self):
        """Enable autoscaling"""
        self._autoscaling_enabled = True
        logger.info("Autoscaling enabled")
    
    def disable_autoscaling(self):
        """Disable autoscaling (for benchmark mode)"""
        self._autoscaling_enabled = False
        logger.info("Autoscaling disabled")
    
    def set_station_count(self, count: int):
        """
        Manually set number of stations (for benchmark mode).
        
        Args:
            count: Target number of stations
        """
        current = len(self.workers)
        
        if count > current:
            # Add workers
            for i in range(count - current):
                self._start_worker()
        
        elif count < current:
            # Remove workers
            for i in range(current - count):
                self._stop_worker()
        
        logger.info(f"Manually set station count: {current} → {count}")
    
    def get_status(self) -> Dict:
        """Get current manager status"""
        metrics = self._get_current_metrics()
        
        return {
            'running': self._running,
            'autoscaling_enabled': self._autoscaling_enabled,
            'active_stations': len(self.workers),
            'worker_pids': {
                sid: proc.pid for sid, proc in self.workers.items()
            },
            'metrics': metrics,
            'scaling_events': len(self._scaling_events),
            'queue_depths': self.metrics_store.get_snapshot()['queue_depths']
        }
    
    def get_scaling_history(self) -> List[Dict]:
        """Get history of scaling events"""
        return self._scaling_events.copy()
