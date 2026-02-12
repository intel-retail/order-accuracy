"""
FastAPI endpoints for Dine-In Order Accuracy Validation.
Production-ready implementation with proper service architecture.
"""

import json
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from io import BytesIO

from fastapi import FastAPI, File, UploadFile, HTTPException, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

# Import services and config
from config import config_manager
from services import ValidationService, VLMClient, SemanticClient
from services.benchmark_service import BenchmarkService

# Configure logging
logging.basicConfig(
    level=config_manager.config.log_level,
    format='%(asctime)s - [%(name)s] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Dine-In Order Accuracy API",
    description="Production API for validating food plate orders using VLM and semantic matching",
    version="2.0.0"
)

# Pydantic models
class OrderItem(BaseModel):
    """Order item model"""
    name: str = Field(..., description="Item name")
    quantity: int = Field(1, ge=1, description="Item quantity")


class OrderManifest(BaseModel):
    """Order manifest containing expected items"""
    items: List[OrderItem]


class ValidationResult(BaseModel):
    """Validation result model"""
    validation_id: str
    image_id: str
    order_complete: bool
    accuracy_score: float
    missing_items: List[Dict]
    extra_items: List[Dict]
    quantity_mismatches: List[Dict]
    matched_items: List[Dict]
    timestamp: str
    metrics: Optional[Dict] = None


class BenchmarkStatus(BaseModel):
    """Benchmark status response"""
    enabled: bool
    status: str
    current_metrics: Dict
    worker_stats: List[Dict]
    config: Dict


# Service initialization
def initialize_services():
    """Initialize all services (lazy initialization)"""
    cfg = config_manager.config
    
    logger.info("Initializing services...")
    
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
    validation_service = ValidationService(
        vlm_client=vlm_client,
        semantic_client=semantic_client
    )
    
    logger.info("Services initialized successfully")
    return validation_service


# Global services (initialized on first request)
_validation_service: Optional[ValidationService] = None
_benchmark_service: Optional[BenchmarkService] = None


def get_validation_service() -> ValidationService:
    """Get or create validation service (singleton pattern)"""
    global _validation_service
    if _validation_service is None:
        _validation_service = initialize_services()
    return _validation_service


def get_benchmark_service() -> Optional[BenchmarkService]:
    """Get benchmark service if enabled"""
    global _benchmark_service
    return _benchmark_service


# In-memory storage for validation results
validation_store: Dict[str, ValidationResult] = {}


# Helper functions for metrics collection

async def call_metrics_collector() -> Dict:
    """
    Get CPU/GPU utilization metrics from dedicated metrics-collector service.
    
    Returns:
        Dict with CPU and GPU metrics
    """
    try:
        import httpx
        
        # Metrics collector serves on port 9000, not 8084
        metrics_url = "http://metrics-collector:9000/metrics"
        logger.info(f"[METRICS] Calling metrics collector at {metrics_url}")
        
        # Call metrics collector service
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(metrics_url)
            response.raise_for_status()
            
            metrics_data = response.json()
            logger.debug(f"[METRICS] Raw response keys: {list(metrics_data.keys())}")
            
            # Extract latest values from time-series arrays
            # Response format: {"cpu_utilization": [[timestamp, value], ...], ...}
            cpu_series = metrics_data.get('cpu_utilization', [])
            gpu_series = metrics_data.get('gpu_utilization', [])
            memory_series = metrics_data.get('memory', [])
            
            # Get latest values (last element in each array)
            cpu_util = cpu_series[-1][1] if cpu_series else 0.0
            gpu_util = gpu_series[-1][1] if gpu_series else 0.0
            # Memory array format: [timestamp, total_gb, used_gb, avail_gb, percent]
            memory_util = memory_series[-1][4] if memory_series and len(memory_series[-1]) > 4 else 0.0
            
            metrics_response = {
                "cpu_utilization": round(cpu_util, 2),
                "gpu_utilization": round(gpu_util, 2),
                "memory_utilization": round(memory_util, 2),
                "gpu_memory_utilization": 0.0  # Not available in current metrics
            }
            
            logger.info(f"[METRICS] System metrics collected from service: {metrics_response}")
            
            return metrics_response
            
    except Exception as e:
        logger.error(f"[METRICS] Error getting metrics from collector service: {e}")
        # Return zeros if metrics service is unavailable
        return {
            "cpu_utilization": 0.0,
            "gpu_utilization": 0.0,
            "memory_utilization": 0.0,
            "gpu_memory_utilization": 0.0
        }


# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "benchmark_mode": config_manager.config.benchmark.enabled
    }


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Dine-In Order Accuracy API",
        "version": "2.0.0",
        "endpoints": {
            "validate": "/api/validate",
            "validate_batch": "/api/validate/batch",
            "get_orders": "/api/orders",
            "health": "/health"
        }
    }


@app.get("/api/orders")
async def get_orders():
    """
    Get all orders with image paths and metadata.
    Returns a mapping of order_id to order details.
    """
    try:
        # Since api.py is in /app/src/, go up one level to /app/
        base_dir = Path(__file__).resolve().parent.parent
        configs_dir = base_dir / "configs"
        images_dir = base_dir / "images"
        orders_file = configs_dir / "orders.json"
        
        if not orders_file.exists():
            raise HTTPException(status_code=404, detail="Orders file not found")
        
        with open(orders_file, 'r') as f:
            orders_data = json.load(f)
        
        orders_mapping = {}
        
        for order in orders_data.get('orders', []):
            order_id = order.get('order_id', '')
            image_id = order.get('image_id', '')
            
            if not order_id or not image_id:
                continue
            
            # Find image file (try different extensions)
            image_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                potential_path = images_dir / f"{image_id}{ext}"
                if potential_path.exists():
                    image_path = str(potential_path)
                    break
            
            # Create label for UI display
            label = (
                f"{image_id} – {order.get('restaurant', 'Unknown')} "
                f"Table {order.get('table_number', '?')}"
            )
            
            orders_mapping[label] = {
                "order_id": order_id,
                "image_id": image_id,
                "image_path": image_path,
                "restaurant": order.get('restaurant', ''),
                "table_number": order.get('table_number', ''),
                "items_ordered": order.get('items_ordered', [])
            }
        
        logger.info(f"[API] Loaded {len(orders_mapping)} orders")
        return {
            "success": True,
            "count": len(orders_mapping),
            "orders": orders_mapping
        }
        
    except Exception as e:
        logger.error(f"[API] Error loading orders: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error loading orders: {str(e)}")


@app.post("/api/validate", response_model=ValidationResult)
async def validate_plate(
    image: UploadFile = File(...),
    order: str = Body(...)
):
    """
    Validate a single plate image against order manifest.
    
    This endpoint performs:
    1. VLM inference to detect items in the image
    2. Semantic matching against expected order items
    3. Accuracy calculation and validation result generation
    
    Args:
        image: Uploaded image file
        order: JSON string of order manifest
        
    Returns:
        ValidationResult with detailed analysis
    """
    logger.info(f"[API] Validation request received: image={image.filename}")
    validation_id = str(uuid.uuid4())
    
    try:
        # Parse order manifest
        try:
            order_data = json.loads(order)
            order_manifest = OrderManifest(**order_data)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in order data: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Invalid order manifest format: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid order format: {e}")
        
        # Read image bytes
        image_bytes = await image.read()
        logger.debug(f"Image read: {len(image_bytes)} bytes")
        
        # Validate image
        try:
            img = Image.open(BytesIO(image_bytes))
            logger.debug(f"Image validated: format={img.format}, size={img.size}, mode={img.mode}")
        except Exception as e:
            logger.error(f"Invalid image file: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
        
        # Get validation service
        validation_service = get_validation_service()
        
        # Extract image_id from filename and order_id from manifest
        image_id = Path(image.filename).stem
        order_id = order_data.get('order_id', image_id)
        
        # Generate unique request ID with station prefix
        request_id = f"station1_{order_id}"
        
        # Perform validation
        logger.info(f"[API] Starting validation: validation_id={validation_id}, image_id={image_id}, request_id={request_id}")
        
        result = await validation_service.validate_plate(
            image_bytes=image_bytes,
            order_manifest=order_manifest.model_dump(),
            image_id=image_id,
            request_id=request_id
        )
        
        # Collect system metrics
        logger.info(f"[API] Calling metrics collector for {request_id}")
        system_metrics = await call_metrics_collector()
        logger.info(f"[API] Metrics collector response for {request_id}: {system_metrics}")
        
        # Enhance metrics with system data
        enhanced_metrics = None
        if result.metrics:
            base_metrics = result.metrics.to_dict()
            vlm_latency_ms = int(base_metrics.get("vlm_inference_ms", 0))
            enhanced_metrics = {
                "end_to_end_latency_ms": vlm_latency_ms,  # Use VLM latency as primary metric
                "vlm_inference_ms": vlm_latency_ms,
                "agent_reconciliation_ms": int(base_metrics.get("semantic_matching_ms", 0)),
                "within_operational_window": vlm_latency_ms < 2000,
                "cpu_utilization": system_metrics.get("cpu_utilization", 0.0),
                "gpu_utilization": system_metrics.get("gpu_utilization", 0.0),
                "memory_utilization": system_metrics.get("memory_utilization", 0.0),
                "gpu_memory_utilization": system_metrics.get("gpu_memory_utilization", 0.0)
            }
            logger.info(f"[API] Metrics for {request_id}: {enhanced_metrics}")
        
        # Build response
        validation_result = ValidationResult(
            validation_id=validation_id,
            image_id=result.image_id,
            order_complete=result.order_complete,
            accuracy_score=result.accuracy_score,
            missing_items=result.missing_items,
            extra_items=result.extra_items,
            quantity_mismatches=result.quantity_mismatches,
            matched_items=result.matched_items,
            timestamp=datetime.now().isoformat(),
            metrics=enhanced_metrics
        )
        
        # Store result
        validation_store[validation_id] = validation_result
        
        logger.info(f"[API] Validation completed: validation_id={validation_id}, "
                   f"accuracy={result.accuracy_score:.2f}, "
                   f"complete={result.order_complete}")
        
        return validation_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[API] Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@app.post("/api/validate/batch", response_model=List[ValidationResult])
async def validate_batch(
    images: List[UploadFile] = File(...),
    orders: str = Body(...)
):
    """
    Validate multiple plate images in batch.
    
    Args:
        images: List of uploaded image files
        orders: JSON string mapping image names to order manifests
        
    Returns:
        List of ValidationResult for each image
    """
    logger.info(f"[API] Batch validation request: {len(images)} images")
    
    try:
        # Parse orders mapping
        try:
            orders_map = json.loads(orders)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in orders data: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        
        validation_service = get_validation_service()
        results = []
        
        # Process each image
        for image in images:
            validation_id = str(uuid.uuid4())
            image_id = Path(image.filename).stem
            
            # Get order for this image
            order_data = orders_map.get(image_id)
            if not order_data:
                logger.warning(f"No order found for image: {image_id}")
                continue
            
            try:
                # Parse and validate order
                order_manifest = OrderManifest(**order_data)
                
                # Read and validate image
                image_bytes = await image.read()
                img = Image.open(BytesIO(image_bytes))
                
                # Perform validation
                logger.info(f"[API] Processing batch item: image_id={image_id}")
                
                result = await validation_service.validate_plate(
                    image_bytes=image_bytes,
                    order_manifest=order_manifest.model_dump(),
                    image_id=image_id
                )
                
                # Build response
                validation_result = ValidationResult(
                    validation_id=validation_id,
                    image_id=result.image_id,
                    order_complete=result.order_complete,
                    accuracy_score=result.accuracy_score,
                    missing_items=result.missing_items,
                    extra_items=result.extra_items,
                    quantity_mismatches=result.quantity_mismatches,
                    matched_items=result.matched_items,
                    timestamp=datetime.now().isoformat(),
                    metrics=result.metrics.to_dict() if result.metrics else None
                )
                
                # Store and collect result
                validation_store[validation_id] = validation_result
                results.append(validation_result)
                
                logger.info(f"[API] Batch item completed: image_id={image_id}, "
                           f"accuracy={result.accuracy_score:.2f}")
                
            except Exception as e:
                logger.error(f"[API] Failed to process image {image_id}: {e}")
                continue
        
        logger.info(f"[API] Batch validation completed: {len(results)}/{len(images)} successful")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[API] Batch validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch validation failed: {str(e)}")


@app.get("/api/validate/{validation_id}", response_model=ValidationResult)
async def get_validation_result(validation_id: str):
    """Get validation result by ID"""
    logger.debug(f"[API] Retrieving validation result: {validation_id}")
    
    if validation_id not in validation_store:
        logger.warning(f"[API] Validation not found: {validation_id}")
        raise HTTPException(status_code=404, detail="Validation not found")
    
    return validation_store[validation_id]


@app.get("/api/validate", response_model=List[ValidationResult])
async def list_validations():
    """List all validation results"""
    logger.debug(f"[API] Listing all validations: {len(validation_store)} total")
    return list(validation_store.values())


@app.delete("/api/validate/{validation_id}")
async def delete_validation(validation_id: str):
    """Delete validation result by ID"""
    logger.info(f"[API] Deleting validation: {validation_id}")
    
    if validation_id not in validation_store:
        logger.warning(f"[API] Validation not found: {validation_id}")
        raise HTTPException(status_code=404, detail="Validation not found")
    
    del validation_store[validation_id]
    return {"status": "deleted", "validation_id": validation_id}


# Benchmark endpoints

@app.post("/api/benchmark/start")
async def start_benchmark(background_tasks: BackgroundTasks):
    """
    Start benchmark mode with dynamic worker scaling.
    
    This endpoint starts multiple workers that continuously process
    images and automatically scale based on latency and resource utilization.
    """
    global _benchmark_service
    
    if not config_manager.config.benchmark.enabled:
        raise HTTPException(
            status_code=400,
            detail="Benchmark mode is not enabled. Set BENCHMARK_MODE=true"
        )
    
    if _benchmark_service is not None:
        raise HTTPException(status_code=400, detail="Benchmark already running")
    
    logger.info("[API] Starting benchmark mode")
    
    try:
        # Load test images and orders
        base_dir = Path(__file__).resolve().parent.parent
        images_dir = base_dir / "images"
        orders_dir = base_dir / "orders"
        
        test_images = []
        test_orders = []
        
        for img_path in sorted(images_dir.glob("*.png"))[:5]:  # Use first 5 images
            with open(img_path, "rb") as f:
                test_images.append(f.read())
            
            order_path = orders_dir / f"{img_path.stem}.json"
            if order_path.exists():
                with open(order_path) as f:
                    test_orders.append(json.load(f))
        
        if not test_images:
            raise HTTPException(status_code=500, detail="No test images found")
        
        # Create benchmark service
        validation_service = get_validation_service()
        _benchmark_service = BenchmarkService(
            config=config_manager.config,
            validation_service=validation_service,
            test_images=test_images,
            test_orders=test_orders
        )
        
        # Start in background
        background_tasks.add_task(_benchmark_service.start)
        
        logger.info("[API] Benchmark mode started")
        return {
            "status": "started",
            "message": "Benchmark service started with dynamic scaling",
            "config": {
                "initial_workers": config_manager.config.benchmark.initial_workers,
                "max_workers": config_manager.config.benchmark.max_workers,
                "target_latency_ms": config_manager.config.benchmark.target_latency_ms
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[API] Failed to start benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start benchmark: {str(e)}")


@app.post("/api/benchmark/stop")
async def stop_benchmark():
    """Stop benchmark mode"""
    global _benchmark_service
    
    if _benchmark_service is None:
        raise HTTPException(status_code=400, detail="Benchmark not running")
    
    logger.info("[API] Stopping benchmark mode")
    
    try:
        await _benchmark_service.stop()
        report = _benchmark_service.get_report()
        _benchmark_service = None
        
        logger.info("[API] Benchmark mode stopped")
        return {
            "status": "stopped",
            "final_report": report
        }
        
    except Exception as e:
        logger.exception(f"[API] Failed to stop benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop benchmark: {str(e)}")


@app.get("/api/benchmark/status", response_model=BenchmarkStatus)
async def get_benchmark_status():
    """Get current benchmark status and metrics"""
    benchmark_service = get_benchmark_service()
    
    if benchmark_service is None:
        return BenchmarkStatus(
            enabled=config_manager.config.benchmark.enabled,
            status="not_running",
            current_metrics={},
            worker_stats=[],
            config={
                "initial_workers": config_manager.config.benchmark.initial_workers,
                "max_workers": config_manager.config.benchmark.max_workers,
                "target_latency_ms": config_manager.config.benchmark.target_latency_ms
            }
        )
    
    report = benchmark_service.get_report()
    
    return BenchmarkStatus(
        enabled=config_manager.config.benchmark.enabled,
        status=report["status"],
        current_metrics=report["current_metrics"],
        worker_stats=report["worker_stats"],
        config=report["config"]
    )


@app.on_event("startup")
async def startup_event():
    """Application startup"""
    logger.info("=" * 60)
    logger.info("Dine-In Order Accuracy API Starting")
    logger.info(f"Version: 2.0.0")
    logger.info(f"Log Level: {config_manager.config.log_level}")
    logger.info(f"OVMS Endpoint: {config_manager.config.service.ovms_endpoint}")
    logger.info(f"Semantic Endpoint: {config_manager.config.service.semantic_service_endpoint}")
    logger.info(f"Benchmark Mode: {config_manager.config.benchmark.enabled}")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown"""
    global _benchmark_service
    
    logger.info("Shutting down API...")
    
    # Stop benchmark if running
    if _benchmark_service is not None:
        logger.info("Stopping benchmark service...")
        await _benchmark_service.stop()
    
    logger.info("API shutdown complete")
