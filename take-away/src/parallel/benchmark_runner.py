"""
Benchmark Runner

Runs performance benchmarks to determine maximum supported stations
and system capacity under various scenarios.

Features:
- Configurable number of stations
- Synthetic load generation
- Metrics collection and reporting
- CSV/JSON output for analysis
"""

import time
import logging
import json
import csv
from typing import Dict, List, Optional
from pathlib import Path
import statistics

from .station_manager import StationManager
from .shared_queue import QueueBackend, QueueManager
from .vlm_scheduler import VLMScheduler

logger = logging.getLogger(__name__)


class BenchmarkScenario:
    """Defines a benchmark test scenario"""
    
    def __init__(
        self,
        name: str,
        num_stations: int,
        duration_seconds: int,
        description: str = ""
    ):
        self.name = name
        self.num_stations = num_stations
        self.duration = duration_seconds
        self.description = description


class BenchmarkResults:
    """Stores benchmark results"""
    
    def __init__(self, scenario: BenchmarkScenario):
        self.scenario = scenario
        self.start_time = time.time()
        self.end_time: Optional[float] = None
        
        # Metrics snapshots
        self.snapshots: List[Dict] = []
        
        # Aggregated results
        self.total_orders = 0
        self.failed_orders = 0
        self.avg_latency = 0.0
        self.p95_latency = 0.0
        self.p99_latency = 0.0
        self.avg_cpu = 0.0
        self.avg_gpu = 0.0
        self.max_cpu = 0.0
        self.max_gpu = 0.0
        self.throughput = 0.0  # Orders per second
        
        # Resource usage
        self.cpu_samples: List[float] = []
        self.gpu_samples: List[float] = []
        self.latency_samples: List[float] = []
    
    def finalize(self):
        """Calculate final statistics"""
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.total_orders > 0:
            self.throughput = self.total_orders / duration
        
        if self.cpu_samples:
            self.avg_cpu = statistics.mean(self.cpu_samples)
            self.max_cpu = max(self.cpu_samples)
        
        if self.gpu_samples:
            self.avg_gpu = statistics.mean(self.gpu_samples)
            self.max_gpu = max(self.gpu_samples)
        
        if self.latency_samples:
            self.avg_latency = statistics.mean(self.latency_samples)
            sorted_latencies = sorted(self.latency_samples)
            self.p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            self.p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
    
    def to_dict(self) -> Dict:
        """Export results as dictionary"""
        return {
            'scenario': {
                'name': self.scenario.name,
                'num_stations': self.scenario.num_stations,
                'duration': self.scenario.duration,
                'description': self.scenario.description
            },
            'timing': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration': self.end_time - self.start_time if self.end_time else 0
            },
            'throughput': {
                'total_orders': self.total_orders,
                'failed_orders': self.failed_orders,
                'success_rate': (
                    (self.total_orders - self.failed_orders) / self.total_orders
                    if self.total_orders > 0 else 0
                ),
                'orders_per_second': self.throughput
            },
            'latency': {
                'avg': self.avg_latency,
                'p95': self.p95_latency,
                'p99': self.p99_latency
            },
            'resource_usage': {
                'cpu': {
                    'avg': self.avg_cpu,
                    'max': self.max_cpu
                },
                'gpu': {
                    'avg': self.avg_gpu,
                    'max': self.max_gpu
                }
            }
        }


class BenchmarkRunner:
    """
    Runs performance benchmarks with configurable scenarios.
    
    Usage:
        runner = BenchmarkRunner(config)
        runner.add_scenario("baseline", num_stations=1, duration=60)
        runner.add_scenario("scale_test", num_stations=4, duration=120)
        results = runner.run_all()
        runner.export_results("benchmark_results.json")
    """
    
    def __init__(
        self,
        config: Dict,
        output_dir: str = "./benchmark_results",
        queue_backend: QueueBackend = QueueBackend.MULTIPROCESSING
    ):
        """
        Args:
            config: System configuration
            output_dir: Directory for results output
            queue_backend: Queue backend type
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.queue_backend = queue_backend
        
        # Scenarios to run
        self.scenarios: List[BenchmarkScenario] = []
        
        # Results
        self.results: List[BenchmarkResults] = []
        
        logger.info(f"BenchmarkRunner initialized (output: {output_dir})")
    
    def add_scenario(
        self,
        name: str,
        num_stations: int,
        duration_seconds: int,
        description: str = ""
    ):
        """Add benchmark scenario"""
        scenario = BenchmarkScenario(
            name=name,
            num_stations=num_stations,
            duration_seconds=duration_seconds,
            description=description
        )
        self.scenarios.append(scenario)
        
        logger.info(
            f"Added scenario: {name} "
            f"(stations={num_stations}, duration={duration_seconds}s)"
        )
    
    def run_all(self) -> List[BenchmarkResults]:
        """Run all benchmark scenarios"""
        logger.info(f"Starting benchmark with {len(self.scenarios)} scenarios")
        
        for i, scenario in enumerate(self.scenarios, 1):
            logger.info(
                f"\n{'='*60}\n"
                f"Scenario {i}/{len(self.scenarios)}: {scenario.name}\n"
                f"{'='*60}"
            )
            
            result = self._run_scenario(scenario)
            self.results.append(result)
            
            # Cooldown between scenarios
            if i < len(self.scenarios):
                cooldown = 10
                logger.info(f"Cooldown: {cooldown}s before next scenario...")
                time.sleep(cooldown)
        
        logger.info("Benchmark completed")
        return self.results
    
    def _run_scenario(self, scenario: BenchmarkScenario) -> BenchmarkResults:
        """Run single benchmark scenario"""
        logger.info(
            f"Running scenario: {scenario.name}\n"
            f"  Stations: {scenario.num_stations}\n"
            f"  Duration: {scenario.duration}s\n"
            f"  Description: {scenario.description}"
        )
        
        # Initialize results
        results = BenchmarkResults(scenario)
        
        # Create queue manager
        queue_manager = QueueManager(backend=self.queue_backend)
        
        # Create station manager
        manager = StationManager(
            config=self.config,
            queue_manager=queue_manager,
            initial_stations=0  # Start with 0, manually set
        )
        
        # Start VLM scheduler
        vlm_scheduler = VLMScheduler(
            queue_manager=manager.queue_manager,
            ovms_url=self.config.get('ovms_url', 'http://localhost:8000'),
            model_name=self.config.get('vlm_model_name', 'vlm'),
            batch_window_ms=self.config.get('batch_window_ms', 100),
            max_batch_size=self.config.get('max_batch_size', 16)
        )
        vlm_scheduler.start()
        
        try:
            # Set target station count
            manager.set_station_count(scenario.num_stations)
            
            # Wait for workers to stabilize
            logger.info("Waiting for workers to initialize (5s)...")
            time.sleep(5)
            
            # Run benchmark for specified duration
            logger.info(f"Benchmark running for {scenario.duration}s...")
            
            start_time = time.time()
            sample_interval = 1.0  # Sample every second
            
            while time.time() - start_time < scenario.duration:
                # Collect metrics snapshot
                snapshot = manager.metrics_store.get_snapshot()
                results.snapshots.append({
                    'timestamp': time.time(),
                    'metrics': snapshot
                })
                
                # Update samples
                results.cpu_samples.append(snapshot['cpu_avg'])
                results.gpu_samples.append(snapshot['gpu_avg'])
                
                # Collect latency samples
                for station_id in manager.metrics_store.get_active_stations():
                    station_latency = manager.metrics_store.get_latency_avg(station_id)
                    if station_latency > 0:
                        results.latency_samples.append(station_latency)
                
                # Log progress
                elapsed = time.time() - start_time
                remaining = scenario.duration - elapsed
                logger.info(
                    f"Progress: {elapsed:.0f}/{scenario.duration}s "
                    f"(CPU={snapshot['cpu_avg']:.1f}% "
                    f"GPU={snapshot['gpu_avg']:.1f}% "
                    f"Orders={snapshot['total_orders']})"
                )
                
                time.sleep(sample_interval)
            
            # Final metrics
            final_snapshot = manager.metrics_store.get_snapshot()
            results.total_orders = final_snapshot['total_orders']
            results.failed_orders = final_snapshot['failed_orders']
            
            # Finalize statistics
            results.finalize()
            
            # Log summary
            self._log_results_summary(results)
        
        finally:
            # Cleanup
            logger.info("Cleaning up scenario...")
            vlm_scheduler.stop()
            manager.stop()
        
        return results
    
    def _log_results_summary(self, results: BenchmarkResults):
        """Log benchmark results summary"""
        logger.info(
            f"\n{'='*60}\n"
            f"Scenario: {results.scenario.name}\n"
            f"{'='*60}\n"
            f"Stations: {results.scenario.num_stations}\n"
            f"Duration: {results.end_time - results.start_time:.1f}s\n"
            f"\n"
            f"Throughput:\n"
            f"  Total Orders: {results.total_orders}\n"
            f"  Failed Orders: {results.failed_orders}\n"
            f"  Success Rate: {(results.total_orders - results.failed_orders) / results.total_orders * 100 if results.total_orders > 0 else 0:.1f}%\n"
            f"  Orders/sec: {results.throughput:.2f}\n"
            f"\n"
            f"Latency:\n"
            f"  Average: {results.avg_latency:.2f}s\n"
            f"  P95: {results.p95_latency:.2f}s\n"
            f"  P99: {results.p99_latency:.2f}s\n"
            f"\n"
            f"Resource Usage:\n"
            f"  CPU: avg={results.avg_cpu:.1f}% max={results.max_cpu:.1f}%\n"
            f"  GPU: avg={results.avg_gpu:.1f}% max={results.max_gpu:.1f}%\n"
            f"{'='*60}"
        )
    
    def export_results(self, filename: Optional[str] = None):
        """
        Export results to JSON file.
        
        Args:
            filename: Output filename (default: timestamp-based)
        """
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Prepare export data
        export_data = {
            'benchmark_info': {
                'timestamp': time.time(),
                'num_scenarios': len(self.results),
                'config': self.config
            },
            'scenarios': [result.to_dict() for result in self.results]
        }
        
        # Write JSON
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Results exported to: {filepath}")
        
        # Also export CSV summary
        self._export_csv_summary(filename.replace('.json', '.csv'))
    
    def _export_csv_summary(self, filename: str):
        """Export results summary as CSV"""
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Scenario',
                'Stations',
                'Duration (s)',
                'Total Orders',
                'Failed Orders',
                'Success Rate (%)',
                'Throughput (orders/s)',
                'Avg Latency (s)',
                'P95 Latency (s)',
                'P99 Latency (s)',
                'Avg CPU (%)',
                'Max CPU (%)',
                'Avg GPU (%)',
                'Max GPU (%)'
            ])
            
            # Data rows
            for result in self.results:
                writer.writerow([
                    result.scenario.name,
                    result.scenario.num_stations,
                    result.end_time - result.start_time if result.end_time else 0,
                    result.total_orders,
                    result.failed_orders,
                    (result.total_orders - result.failed_orders) / result.total_orders * 100 if result.total_orders > 0 else 0,
                    result.throughput,
                    result.avg_latency,
                    result.p95_latency,
                    result.p99_latency,
                    result.avg_cpu,
                    result.max_cpu,
                    result.avg_gpu,
                    result.max_gpu
                ])
        
        logger.info(f"CSV summary exported to: {filepath}")


def run_standard_benchmark(config: Dict, output_dir: str = "./benchmark_results"):
    """
    Run standard benchmark suite.
    
    Tests 1, 2, 4, 6, 8 stations to determine capacity.
    """
    runner = BenchmarkRunner(config, output_dir)
    
    # Add scenarios
    runner.add_scenario(
        name="baseline_1_station",
        num_stations=1,
        duration_seconds=60,
        description="Baseline with single station"
    )
    
    runner.add_scenario(
        name="scale_2_stations",
        num_stations=2,
        duration_seconds=90,
        description="Scale to 2 stations"
    )
    
    runner.add_scenario(
        name="scale_4_stations",
        num_stations=4,
        duration_seconds=120,
        description="Scale to 4 stations"
    )
    
    runner.add_scenario(
        name="scale_6_stations",
        num_stations=6,
        duration_seconds=120,
        description="Scale to 6 stations"
    )
    
    runner.add_scenario(
        name="scale_8_stations",
        num_stations=8,
        duration_seconds=120,
        description="Maximum capacity test (8 stations)"
    )
    
    # Run benchmarks
    results = runner.run_all()
    
    # Export results
    runner.export_results()
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Sample configuration
    config = {
        'rtsp_urls': [
            'rtsp://simulation/station_1',
            'rtsp://simulation/station_2'
        ],
        'ovms_url': 'http://localhost:8000',
        'vlm_model_name': 'vlm',
        'batch_window_ms': 100,
        'max_batch_size': 16,
        'minio_endpoint': 'localhost:9000',
        'minio_bucket': 'orders',
        'inventory_path': '/config/inventory.json',
        'orders_path': '/config/orders.json',
        'yolo_model_path': './models/yolo11n_openvino_model'
    }
    
    # Run standard benchmark
    run_standard_benchmark(config)
