#!/usr/bin/env python3
"""
Unified Order Accuracy Service
Supports both single-worker and parallel multi-worker modes
"""
import os
import sys
import logging
import signal
import socket
import threading
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Auto-detect STATION_ID from hostname if not set
# Docker Compose scales containers with names like: order-accuracy-order-accuracy-1, order-accuracy-order-accuracy-2
if 'STATION_ID' not in os.environ:
    hostname = socket.gethostname()
    # Extract number from hostname (e.g., order-accuracy-order-accuracy-2 -> station_2)
    if hostname and hostname.split('-')[-1].isdigit():
        station_num = hostname.split('-')[-1]
        os.environ['STATION_ID'] = f'station_{station_num}'
        logger.info(f"Auto-detected STATION_ID from hostname: {os.environ['STATION_ID']}")
    else:
        os.environ['STATION_ID'] = 'station_1'
        logger.info(f"Using default STATION_ID: station_1")

# Service configuration from environment
SERVICE_MODE = os.getenv('SERVICE_MODE', 'single')  # 'single' or 'parallel'
WORKERS = int(os.getenv('WORKERS', '1'))
SCALING_MODE = os.getenv('SCALING_MODE', 'fixed')  # 'fixed' or 'auto'

logger.info(f"======================================")
logger.info(f"Order Accuracy Service Starting")
logger.info(f"======================================")
logger.info(f"Station ID: {os.environ.get('STATION_ID')}")
logger.info(f"Mode: {SERVICE_MODE}")
logger.info(f"Workers: {WORKERS}")
if SERVICE_MODE == 'parallel':
    logger.info(f"Scaling: {SCALING_MODE}")
logger.info(f"======================================")

# RabbitMQ configuration
USE_RABBITMQ = os.getenv('USE_RABBITMQ', 'true').lower() == 'true'
logger.info(f"RabbitMQ enabled: {USE_RABBITMQ}")


def start_rabbitmq_consumer():
    """Start RabbitMQ consumer for processing orders from queue"""
    if not USE_RABBITMQ:
        logger.info("RabbitMQ disabled, consumer not started")
        return None
    
    try:
        from core.message_queue import OrderConsumer
        from core.vlm_service import run_vlm
        import asyncio
        
        # Create a persistent event loop for the consumer thread
        _consumer_loop = None
        
        def process_order_sync(order_id: str, station_id: str) -> bool:
            """Synchronous wrapper for async run_vlm"""
            nonlocal _consumer_loop
            try:
                logger.info(f"[CONSUMER] Processing order: order_id={order_id}, station_id={station_id}")
                
                # Reuse event loop or create new one if needed
                if _consumer_loop is None or _consumer_loop.is_closed():
                    _consumer_loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(_consumer_loop)
                
                result = _consumer_loop.run_until_complete(run_vlm(order_id, station_id))
                status = result.get('status', 'unknown')
                logger.info(f"[CONSUMER] Order processed: order_id={order_id}, status={status}")
                return status in ('validated', 'mismatch', 'success')
                    
            except Exception as e:
                logger.error(f"[CONSUMER] Failed to process order: order_id={order_id}, error={e}")
                return False
        
        consumer = OrderConsumer(process_callback=process_order_sync)
        consumer.start()
        logger.info("[CONSUMER] RabbitMQ consumer started")
        return consumer
        
    except ImportError as e:
        logger.warning(f"RabbitMQ consumer not available: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to start RabbitMQ consumer: {e}")
        return None


def run_single_mode():
    """Run in single-worker mode with FastAPI"""
    logger.info("Starting Single Worker Mode with FastAPI API")
    
    # Start RabbitMQ consumer in background
    consumer = start_rabbitmq_consumer()
    
    from api import create_app
    import uvicorn
    
    app = create_app()
    
    # Get configuration
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    logger.info(f"Starting API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def run_parallel_mode():
    """Run in parallel multi-worker mode"""
    logger.info(f"Starting Parallel Mode with {WORKERS} workers")
    
    # Start RabbitMQ consumer in background (for frame-selector communication)
    consumer = start_rabbitmq_consumer()
    
    from parallel import StationManager, VLMScheduler, MetricsCollector, MetricsStore, QueueManager
    from parallel.shared_queue import QueueBackend
    from parallel.config import create_default_config
    
    # Load configuration
    config = create_default_config()
    config.scaling.mode = SCALING_MODE
    config.scaling.initial_stations = WORKERS
    
    logger.info("Initializing parallel processing components...")
    
    # Initialize shared queue manager
    queue_mgr = QueueManager()
    
    # Initialize metrics store and collector
    metrics_store = MetricsStore()
    metrics = MetricsCollector(metrics_store=metrics_store, sample_interval=1.0)
    
    # VLM worker count - defaults to WORKERS count but can be overridden
    vlm_workers = int(os.getenv('VLM_WORKERS', WORKERS))
    
    # Initialize VLM scheduler  
    scheduler = VLMScheduler(
        queue_manager=queue_mgr,
        ovms_url=os.getenv('OVMS_ENDPOINT', 'http://ovms-vlm:8000'),
        model_name=os.getenv('OVMS_MODEL_NAME', 'Qwen/Qwen2.5-VL-7B-Instruct'),
        batch_window_ms=100,
        max_batch_size=16,
        max_workers=vlm_workers
    )
    
    # Initialize station manager with SHARED queue manager
    manager = StationManager(
        config=config.to_dict(),
        queue_manager=queue_mgr,  # Pass the same instance used by scheduler!
        initial_stations=WORKERS
    )
    
    # Start API server in background thread
    api_host = os.getenv('API_HOST', '0.0.0.0')
    api_port = int(os.getenv('API_PORT', '8000'))
    
    def run_api_server():
        """Run API server in background thread"""
        import uvicorn
        from api.endpoints import create_app
        
        app = create_app()
        logger.info(f"Starting API server on {api_host}:{api_port}")
        uvicorn.run(app, host=api_host, port=api_port, log_level="info")
    
    api_thread = threading.Thread(
        target=run_api_server,
        name="APIServer",
        daemon=True
    )
    api_thread.start()
    logger.info("API server started in background thread")
    
    # Start components
    logger.info("Starting VLM scheduler...")
    scheduler.start()
    
    logger.info("Starting metrics collector...")
    metrics.start()
    
    logger.info(f"Starting station manager with {WORKERS} workers...")
    manager.start()
    
    logger.info("All components started successfully")
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        manager.stop()
        scheduler.stop()
        metrics.stop()
        logger.info("Shutdown complete")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Keep main thread alive
    try:
        signal.pause()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down...")
        manager.stop()
        scheduler.stop()
        metrics.stop()


def main():
    """Main entry point with mode selection"""
    try:
        if SERVICE_MODE == 'single':
            run_single_mode()
        elif SERVICE_MODE == 'parallel':
            run_parallel_mode()
        else:
            logger.error(f"Invalid SERVICE_MODE: {SERVICE_MODE}")
            logger.error("Valid modes are: 'single' or 'parallel'")
            sys.exit(1)
            
    except Exception as e:
        logger.exception(f"Fatal error in {SERVICE_MODE} mode: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
