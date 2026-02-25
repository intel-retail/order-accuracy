#!/usr/bin/env python3
"""
Dine-In Worker Service for Stream Density Benchmarking.

This worker continuously processes images from the images folder,
validates them using VLM, and stores results. The number of workers
is configurable via WORKERS environment variable.

Usage:
    WORKERS=3 python worker.py
    
Stream Density Testing:
    Increase WORKERS count to test system throughput under load.
"""

import os
import sys
import json
import time
import asyncio
import logging
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import config_manager
from services import ValidationService, VLMClient, SemanticClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [Worker-%(process)d] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class WorkerResult:
    """Result from a single worker validation"""
    worker_id: int
    iteration: int
    image_id: str
    order_id: str
    timestamp: str
    success: bool
    order_complete: bool = False
    accuracy_score: float = 0.0
    vlm_latency_ms: float = 0.0
    semantic_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    items_detected: int = 0
    items_expected: int = 0
    missing_items: int = 0
    extra_items: int = 0
    missing_items_list: List[Dict] = field(default_factory=list)
    extra_items_list: List[Dict] = field(default_factory=list)
    error: Optional[str] = None
    tps: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class WorkerStats:
    """Aggregated statistics for a worker"""
    worker_id: int
    total_iterations: int = 0
    successful_iterations: int = 0
    failed_iterations: int = 0
    total_latency_ms: float = 0.0
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    avg_tps: float = 0.0
    total_tokens: int = 0
    start_time: str = ""
    last_update: str = ""
    results: List[WorkerResult] = field(default_factory=list)


class DineInWorker:
    """
    Worker that processes images for dine-in order validation.
    
    Each worker:
    1. Loads orders from orders.json
    2. Iterates through images in images folder
    3. Validates each image against corresponding order
    4. Stores results in results folder
    """
    
    def __init__(
        self,
        worker_id: int,
        images_dir: Path,
        orders_file: Path,
        results_dir: Path,
        iterations: int = 0,  # 0 = infinite
        loop_images: bool = True,
        delay_between_requests: float = 0.0
    ):
        self.worker_id = worker_id
        self.images_dir = images_dir
        self.orders_file = orders_file
        self.results_dir = results_dir
        self.iterations = iterations
        self.loop_images = loop_images
        self.delay_between_requests = delay_between_requests
        
        self.stats = WorkerStats(worker_id=worker_id)
        self.running = True
        self.validation_service: Optional[ValidationService] = None
        
        # Load orders
        self.orders: Dict[str, Dict] = {}
        self._load_orders()
        
        # Get available images
        self.images: List[Path] = []
        self._load_images()
        
        logger.info(f"Worker {worker_id} initialized: "
                   f"{len(self.images)} images, {len(self.orders)} orders")
    
    def _load_orders(self):
        """Load orders from orders.json"""
        try:
            with open(self.orders_file, 'r') as f:
                data = json.load(f)
            
            # Index orders by image_id for quick lookup
            for order in data.get("orders", []):
                image_id = order.get("image_id")
                if image_id:
                    # Convert items_ordered to items format expected by validation
                    items = []
                    for item in order.get("items_ordered", []):
                        items.append({
                            "name": item.get("item"),
                            "quantity": item.get("quantity", 1)
                        })
                    self.orders[image_id] = {
                        "order_id": order.get("order_id", image_id),
                        "items": items,
                        "restaurant": order.get("restaurant", "Unknown"),
                        "table_number": order.get("table_number", "")
                    }
            
            logger.info(f"Loaded {len(self.orders)} orders from {self.orders_file}")
            
        except Exception as e:
            logger.error(f"Failed to load orders: {e}")
            raise
    
    def _load_images(self):
        """Load available images from images directory"""
        if not self.images_dir.exists():
            logger.warning(f"Images directory not found: {self.images_dir}")
            return
        
        # Get all jpg/jpeg/png images
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            self.images.extend(self.images_dir.glob(ext))
        
        # Sort for consistent ordering
        self.images.sort(key=lambda p: p.stem)
        
        # Filter to only images that have corresponding orders
        self.images = [img for img in self.images if img.stem in self.orders]
        
        logger.info(f"Found {len(self.images)} images with matching orders")
    
    async def _wait_for_ovms_ready(self, max_retries: int = 30, retry_interval: int = 5):
        """Wait for OVMS VLM endpoint to be ready"""
        import httpx
        
        cfg = config_manager.config
        endpoint = f"{cfg.service.ovms_endpoint}/v3/chat/completions"
        
        logger.info(f"Worker {self.worker_id}: Waiting for OVMS VLM to be ready at {endpoint}...")
        
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    # Send a minimal test request
                    response = await client.post(
                        endpoint,
                        json={
                            "model": cfg.service.ovms_model_name,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 1
                        }
                    )
                    if response.status_code == 200:
                        logger.info(f"Worker {self.worker_id}: OVMS VLM is ready (attempt {attempt + 1})")
                        return True
                    else:
                        logger.warning(f"Worker {self.worker_id}: OVMS returned {response.status_code}, retrying...")
            except Exception as e:
                logger.warning(f"Worker {self.worker_id}: OVMS not ready (attempt {attempt + 1}/{max_retries}): {e}")
            
            await asyncio.sleep(retry_interval)
        
        raise RuntimeError(f"OVMS VLM not ready after {max_retries * retry_interval} seconds")
    
    def _initialize_services(self):
        """Initialize validation services"""
        cfg = config_manager.config
        
        logger.info(f"Worker {self.worker_id}: Initializing services...")
        
        # Create VLM client
        vlm_client = VLMClient(
            endpoint=cfg.service.ovms_endpoint,
            model_name=cfg.service.ovms_model_name,
            timeout=cfg.service.api_timeout
        )
        
        # Create semantic client  
        semantic_client = SemanticClient(
            endpoint=cfg.service.semantic_service_endpoint,
            timeout=cfg.service.api_timeout
        )
        
        # Create validation service
        self.validation_service = ValidationService(
            vlm_client=vlm_client,
            semantic_client=semantic_client
        )
        
        logger.info(f"Worker {self.worker_id}: Services initialized")
    
    async def process_image(self, image_path: Path, iteration: int) -> WorkerResult:
        """Process a single image through validation pipeline"""
        image_id = image_path.stem
        order_data = self.orders.get(image_id, {})
        order_id = order_data.get("order_id", image_id)
        
        result = WorkerResult(
            worker_id=self.worker_id,
            iteration=iteration,
            image_id=image_id,
            order_id=order_id,
            timestamp=datetime.now().isoformat(),
            success=False
        )
        
        start_time = time.time()
        
        try:
            # Ensure validation service is initialized
            if self.validation_service is None:
                raise RuntimeError("Validation service not initialized")
            
            # Read image
            with open(image_path, 'rb') as f:
                image_bytes = f.read()
            
            # Generate request ID for tracking
            request_id = f"worker{self.worker_id}_{order_id}_{iteration}"
            
            logger.info(f"Worker {self.worker_id}: Processing {image_id} (iteration {iteration})")
            
            # Validate plate
            validation_result = await self.validation_service.validate_plate(
                image_bytes=image_bytes,
                order_manifest={"items": order_data.get("items", [])},
                image_id=image_id,
                request_id=request_id
            )
            
            # Extract metrics
            total_latency = (time.time() - start_time) * 1000
            
            result.success = True
            result.order_complete = validation_result.order_complete
            result.accuracy_score = validation_result.accuracy_score
            result.total_latency_ms = total_latency
            result.items_expected = len(order_data.get("items", []))
            result.missing_items = len(validation_result.missing_items)
            result.extra_items = len(validation_result.extra_items)
            result.missing_items_list = validation_result.missing_items
            result.extra_items_list = validation_result.extra_items
            
            # Extract VLM metrics if available
            if validation_result.metrics:
                metrics = validation_result.metrics
                result.vlm_latency_ms = getattr(metrics, 'vlm_inference_time_ms', 0)
                result.semantic_latency_ms = getattr(metrics, 'semantic_matching_time_ms', 0)
                result.items_detected = getattr(metrics, 'items_detected', 0)
                result.tps = getattr(metrics, 'tps', 0)
                result.prompt_tokens = getattr(metrics, 'prompt_tokens', 0)
                result.completion_tokens = getattr(metrics, 'completion_tokens', 0)
            
            logger.info(f"Worker {self.worker_id}: Completed {image_id} in {total_latency:.0f}ms "
                       f"(accuracy={result.accuracy_score:.2f}, complete={result.order_complete})")
            
        except Exception as e:
            result.error = str(e)
            result.total_latency_ms = (time.time() - start_time) * 1000
            logger.error(f"Worker {self.worker_id}: Error processing {image_id}: {e}")
        
        return result
    
    def _update_stats(self, result: WorkerResult):
        """Update worker statistics with result"""
        self.stats.total_iterations += 1
        self.stats.last_update = datetime.now().isoformat()
        
        if result.success:
            self.stats.successful_iterations += 1
            self.stats.total_latency_ms += result.total_latency_ms
            self.stats.avg_latency_ms = self.stats.total_latency_ms / self.stats.successful_iterations
            self.stats.min_latency_ms = min(self.stats.min_latency_ms, result.total_latency_ms)
            self.stats.max_latency_ms = max(self.stats.max_latency_ms, result.total_latency_ms)
            self.stats.total_tokens += result.prompt_tokens + result.completion_tokens
            if result.tps > 0:
                # Running average of TPS
                n = self.stats.successful_iterations
                self.stats.avg_tps = ((n - 1) * self.stats.avg_tps + result.tps) / n
        else:
            self.stats.failed_iterations += 1
        
        self.stats.results.append(result)
    
    def _save_results(self):
        """Save worker results to file"""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"worker_{self.worker_id}_results_{timestamp}.json"
        
        # Convert stats to dict
        stats_dict = asdict(self.stats)
        
        # Also save individual results
        stats_dict["results"] = [asdict(r) for r in self.stats.results]
        
        with open(results_file, 'w') as f:
            json.dump(stats_dict, f, indent=2, default=str)
        
        logger.info(f"Worker {self.worker_id}: Results saved to {results_file}")
        return results_file
    
    async def run(self):
        """Main worker loop"""
        # Wait for OVMS to be ready before initializing services
        await self._wait_for_ovms_ready()
        
        self._initialize_services()
        
        self.stats.start_time = datetime.now().isoformat()
        iteration = 0
        image_index = 0
        
        logger.info(f"Worker {self.worker_id}: Starting processing loop")
        
        try:
            while self.running:
                # Check iteration limit
                if self.iterations > 0 and iteration >= self.iterations:
                    logger.info(f"Worker {self.worker_id}: Reached iteration limit ({self.iterations})")
                    break
                
                # Get next image (loop if needed)
                if image_index >= len(self.images):
                    if self.loop_images:
                        image_index = 0
                        logger.info(f"Worker {self.worker_id}: Looping back to first image")
                    else:
                        logger.info(f"Worker {self.worker_id}: All images processed")
                        break
                
                image_path = self.images[image_index]
                
                # Process image
                result = await self.process_image(image_path, iteration)
                self._update_stats(result)
                
                iteration += 1
                image_index += 1
                
                # Optional delay between requests
                if self.delay_between_requests > 0:
                    await asyncio.sleep(self.delay_between_requests)
                    
        except asyncio.CancelledError:
            logger.info(f"Worker {self.worker_id}: Cancelled")
        except Exception as e:
            logger.error(f"Worker {self.worker_id}: Fatal error: {e}")
        finally:
            # Save results
            self._save_results()
            logger.info(f"Worker {self.worker_id}: Finished - "
                       f"{self.stats.successful_iterations}/{self.stats.total_iterations} successful, "
                       f"avg_latency={self.stats.avg_latency_ms:.0f}ms")
    
    def stop(self):
        """Signal worker to stop"""
        self.running = False


class WorkerManager:
    """
    Manages multiple dine-in workers for stream density testing.
    
    Usage:
        manager = WorkerManager(num_workers=4)
        await manager.start()
    """
    
    def __init__(
        self,
        num_workers: int = 1,
        images_dir: Optional[Path] = None,
        orders_file: Optional[Path] = None,
        results_dir: Optional[Path] = None,
        iterations_per_worker: int = 0,  # 0 = infinite
        delay_between_requests: float = 0.0
    ):
        self.num_workers = num_workers
        
        # Default paths
        base_dir = Path(__file__).parent.parent
        self.images_dir = images_dir or base_dir / "images"
        self.orders_file = orders_file or base_dir / "configs" / "orders.json"
        self.results_dir = results_dir or base_dir / "results"
        
        self.iterations_per_worker = iterations_per_worker
        self.delay_between_requests = delay_between_requests
        
        self.workers: List[DineInWorker] = []
        self.tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info(f"WorkerManager initialized: {num_workers} workers")
    
    async def start(self):
        """Start all workers"""
        self.running = True
        
        logger.info(f"Starting {self.num_workers} workers...")
        
        # Create workers
        for i in range(self.num_workers):
            worker = DineInWorker(
                worker_id=i,
                images_dir=self.images_dir,
                orders_file=self.orders_file,
                results_dir=self.results_dir,
                iterations=self.iterations_per_worker,
                loop_images=True,
                delay_between_requests=self.delay_between_requests
            )
            self.workers.append(worker)
        
        # Start worker tasks
        for worker in self.workers:
            task = asyncio.create_task(worker.run())
            self.tasks.append(task)
        
        logger.info(f"All {self.num_workers} workers started")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("Worker manager cancelled")
        
        self._print_summary()
    
    def stop(self):
        """Stop all workers"""
        logger.info("Stopping all workers...")
        self.running = False
        
        for worker in self.workers:
            worker.stop()
        
        for task in self.tasks:
            task.cancel()
    
    def _print_summary(self):
        """Print summary of all worker results"""
        print("\n" + "=" * 70)
        print("                    DINE-IN WORKER SUMMARY")
        print("=" * 70)
        
        total_iterations = 0
        total_successful = 0
        total_latency = 0.0
        all_latencies = []
        
        for worker in self.workers:
            stats = worker.stats
            total_iterations += stats.total_iterations
            total_successful += stats.successful_iterations
            
            if stats.successful_iterations > 0:
                total_latency += stats.total_latency_ms
                all_latencies.extend([r.total_latency_ms for r in stats.results if r.success])
            
            print(f"\n  Worker {stats.worker_id}:")
            print(f"    Iterations: {stats.successful_iterations}/{stats.total_iterations}")
            print(f"    Avg Latency: {stats.avg_latency_ms:.0f}ms")
            print(f"    Min/Max: {stats.min_latency_ms:.0f}ms / {stats.max_latency_ms:.0f}ms")
            print(f"    Avg TPS: {stats.avg_tps:.2f}")
        
        print("\n" + "-" * 70)
        print("  AGGREGATE:")
        print(f"    Total Workers: {self.num_workers}")
        print(f"    Total Iterations: {total_successful}/{total_iterations}")
        
        if all_latencies:
            avg_latency = sum(all_latencies) / len(all_latencies)
            print(f"    Overall Avg Latency: {avg_latency:.0f}ms")
            print(f"    Overall Min Latency: {min(all_latencies):.0f}ms")
            print(f"    Overall Max Latency: {max(all_latencies):.0f}ms")
        
        print("=" * 70 + "\n")


def signal_handler(signum, frame, manager: WorkerManager):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    manager.stop()


async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Dine-In Worker for Stream Density Testing")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "1")),
                       help="Number of workers to run (default: WORKERS env or 1)")
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ITERATIONS", "0")),
                       help="Iterations per worker, 0=infinite (default: ITERATIONS env or 0)")
    parser.add_argument("--delay", type=float, default=float(os.environ.get("REQUEST_DELAY", "0")),
                       help="Delay between requests in seconds (default: 0)")
    parser.add_argument("--images-dir", type=str, default=os.environ.get("IMAGES_DIR"),
                       help="Path to images directory")
    parser.add_argument("--orders-file", type=str, default=os.environ.get("ORDERS_FILE"),
                       help="Path to orders.json file")
    parser.add_argument("--results-dir", type=str, default=os.environ.get("RESULTS_DIR"),
                       help="Path to results directory")
    
    args = parser.parse_args()
    
    logger.info(f"Starting Dine-In Worker Service")
    logger.info(f"  Workers: {args.workers}")
    logger.info(f"  Iterations: {args.iterations if args.iterations > 0 else 'infinite'}")
    logger.info(f"  Delay: {args.delay}s")
    
    # Create manager
    manager = WorkerManager(
        num_workers=args.workers,
        images_dir=Path(args.images_dir) if args.images_dir else None,
        orders_file=Path(args.orders_file) if args.orders_file else None,
        results_dir=Path(args.results_dir) if args.results_dir else None,
        iterations_per_worker=args.iterations,
        delay_between_requests=args.delay
    )
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s, None, manager))
    
    # Run
    await manager.start()


if __name__ == "__main__":
    asyncio.run(main())
