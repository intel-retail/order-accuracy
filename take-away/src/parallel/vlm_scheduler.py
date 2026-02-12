"""
VLM Scheduler with Request Batching

Collects VLM inference requests from multiple station workers,
performs time-window batching, sends batched requests to OVMS,
and distributes responses back to workers.

Key features:
- Small time-window batching (50-100ms) to improve OVMS throughput
- Fair scheduling across stations
- Graceful error handling and retry logic
- Maintains order accuracy by preserving request-response pairing
"""

import time
import threading
import logging
import sys
import os
from typing import List, Dict, Optional
import numpy as np
import requests
import json
from queue import Queue, Empty
import base64
from io import BytesIO
from PIL import Image

from .shared_queue import QueueManager, VLMRequest, VLMResponse

# Import OVMS client from core module
import sys
import os
# Add core to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from ovms_client import OVMSVLMClient  # type: ignore
from config_loader import load_config  # type: ignore

logger = logging.getLogger(__name__)

# Load inventory for prompt
cfg = load_config()
INVENTORY = cfg.get("inventory", [])

if not INVENTORY:
    logger.warning("Inventory list is empty - VLM will have no product constraints")
else:
    logger.info(f"VLM Scheduler loaded inventory: {len(INVENTORY)} items")
    logger.debug(f"Inventory items: {INVENTORY}")

INVENTORY_TEXT = "\n".join(f"- {item}" for item in INVENTORY)


class VLMScheduler:
    """
    VLM inference request scheduler with batching.
    
    Architecture:
    1. Continuously polls vlm_request_queue for incoming requests
    2. Accumulates requests in time window (e.g., 50-100ms)
    3. When window expires or batch size reached, sends to OVMS
    4. Parses responses and routes back to station-specific response queues
    
    This enables efficient GPU utilization through OVMS continuous batching.
    """
    
    def __init__(
        self,
        queue_manager: QueueManager,
        ovms_url: str = "http://localhost:8000",
        model_name: str = "vlm",
        batch_window_ms: int = 100,
        max_batch_size: int = 16,
        max_workers: int = 4
    ):
        """
        Args:
            queue_manager: Shared queue manager
            ovms_url: OVMS server URL
            model_name: Model name in OVMS
            batch_window_ms: Time window for batching in milliseconds
            max_batch_size: Maximum batch size
            max_workers: Number of parallel OVMS request threads
        """
        self.queue_manager = queue_manager
        self.ovms_url = ovms_url.rstrip('/')
        self.model_name = model_name
        self.batch_window = batch_window_ms / 1000.0  # Convert to seconds
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        
        # Internal request buffer
        self._request_buffer: List[VLMRequest] = []
        self._buffer_lock = threading.Lock()
        self._last_batch_time = time.time()
        
        # Threading
        self._running = False
        self._collector_thread: Optional[threading.Thread] = None
        self._batcher_thread: Optional[threading.Thread] = None
        self._worker_threads: List[threading.Thread] = []
        
        # Work queue for batched requests
        self._work_queue: Queue = Queue(maxsize=100)
        
        # Statistics
        self._total_requests = 0
        self._total_batches = 0
        self._total_errors = 0
        
        # Initialize VLM backend
        self._vlm_client = None
        self._initialize_vlm_backend(ovms_url, model_name)
        
        logger.info(
            f"VLMScheduler initialized:\n"
            f"  - OVMS endpoint: {ovms_url}\n"
            f"  - Model name: {model_name}\n"
            f"  - Batch window: {batch_window_ms}ms\n"
            f"  - Max batch size: {max_batch_size}\n"
            f"  - Worker threads: {max_workers}"
        )
    
    def _initialize_vlm_backend(self, ovms_url: str, model_name: str):
        """Initialize OVMS VLM backend"""
        try:
            logger.info(f"Initializing OVMS VLM client: {ovms_url}")
            self._vlm_client = OVMSVLMClient(
                endpoint=ovms_url,
                model_name=model_name,
                timeout=300,  # 5 minutes for large VLM model
                max_new_tokens=512,
                temperature=0.2
            )
            logger.info("OVMS VLM client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize OVMS VLM backend: {e}")
            raise
    
    def start(self):
        """Start scheduler threads"""
        if self._running:
            logger.warning("VLMScheduler already running")
            return
        
        self._running = True
        
        # Start request collector thread
        self._collector_thread = threading.Thread(
            target=self._collect_requests_loop,
            daemon=True,
            name="VLM-Collector"
        )
        self._collector_thread.start()
        
        # Start batching thread
        self._batcher_thread = threading.Thread(
            target=self._batching_loop,
            daemon=True,
            name="VLM-Batcher"
        )
        self._batcher_thread.start()
        
        # Start worker threads
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                daemon=True,
                name=f"VLM-Worker-{i}"
            )
            worker.start()
            self._worker_threads.append(worker)
        
        logger.info(f"VLMScheduler started with {self.max_workers} workers")
    
    def stop(self):
        """Stop scheduler threads"""
        if not self._running:
            return
        
        logger.info("Stopping VLMScheduler...")
        self._running = False
        
        # Wait for threads
        if self._collector_thread:
            self._collector_thread.join(timeout=5)
        if self._batcher_thread:
            self._batcher_thread.join(timeout=5)
        
        for worker in self._worker_threads:
            worker.join(timeout=5)
        
        logger.info(
            f"VLMScheduler stopped. "
            f"Stats: {self._total_requests} requests, "
            f"{self._total_batches} batches, "
            f"{self._total_errors} errors"
        )
    
    def _collect_requests_loop(self):
        """Continuously collect requests from queue"""
        logger.info("Request collector thread started")
        logger.info(f"[COLLECT-THREAD] Polling vlm_request_queue: {id(self.queue_manager.vlm_request_queue)}")
        
        while self._running:
            try:
                # Non-blocking get with timeout
                logger.debug("[COLLECT-THREAD] Waiting for request...")
                request_dict = self.queue_manager.vlm_request_queue.get(
                    block=True,
                    timeout=0.1
                )
                
                if request_dict is None:
                    logger.debug("❌ Got None from queue, skipping")
                    continue
                
                # Deserialize request
                logger.info(f"✓ GOT REQUEST FROM QUEUE: {request_dict.get('request_id', 'unknown')}")
                request = VLMRequest.from_dict(request_dict)
                
                # Debug logging
                logger.info(f"[COLLECT-DEBUG] ⚡ Request {request.request_id}: {len(request.frames)} frames")
                if request.frames:
                    logger.info(f"[COLLECT-DEBUG] First frame type: {type(request.frames[0])}, keys: {request.frames[0].keys() if isinstance(request.frames[0], dict) else 'N/A'}")
                
                # Add to buffer
                with self._buffer_lock:
                    self._request_buffer.append(request)
                    self._total_requests += 1
                
                logger.debug(
                    f"Collected request: {request.request_id} "
                    f"(buffer size: {len(self._request_buffer)})"
                )
            
            except Empty:
                continue
            
            except Exception as e:
                logger.error(f"Error collecting request: {e}")
                time.sleep(0.1)
        
        logger.info("Request collector thread stopped")
    
    def _batching_loop(self):
        """Batch requests based on time window"""
        logger.info("Batching thread started")
        
        while self._running:
            try:
                time.sleep(0.01)  # Check every 10ms
                
                with self._buffer_lock:
                    if not self._request_buffer:
                        continue
                    
                    time_since_last_batch = time.time() - self._last_batch_time
                    buffer_size = len(self._request_buffer)
                    
                    # Trigger batch if:
                    # 1. Time window expired, OR
                    # 2. Max batch size reached
                    should_batch = (
                        time_since_last_batch >= self.batch_window or
                        buffer_size >= self.max_batch_size
                    )
                    
                    if should_batch:
                        # Extract batch
                        batch = self._request_buffer[:self.max_batch_size]
                        self._request_buffer = self._request_buffer[self.max_batch_size:]
                        self._last_batch_time = time.time()
                        
                        # Submit to work queue
                        self._work_queue.put(batch)
                        self._total_batches += 1
                        
                        logger.debug(
                            f"Created batch: {len(batch)} requests, "
                            f"waited {time_since_last_batch*1000:.1f}ms"
                        )
            
            except Exception as e:
                logger.error(f"Error in batching loop: {e}")
                time.sleep(0.1)
        
        logger.info("Batching thread stopped")
    
    def _worker_loop(self):
        """Worker thread that processes batched requests"""
        worker_name = threading.current_thread().name
        logger.info(f"{worker_name} started")
        
        while self._running:
            try:
                # Get batch from work queue
                batch = self._work_queue.get(timeout=0.5)
                
                if batch is None:
                    continue
                
                # Process batch
                self._process_batch(batch)
            
            except Empty:
                continue
            
            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                time.sleep(0.1)
        
        logger.info(f"{worker_name} stopped")
    
    def _process_batch(self, batch: List[VLMRequest]):
        """
        Process batch of VLM requests.
        
        For simplicity, this implementation sends requests individually
        to OVMS. OVMS will batch them internally if they arrive concurrently.
        
        For true batching, modify to use OVMS batch inference API.
        """
        batch_start = time.time()
        
        logger.debug(f"Processing batch of {len(batch)} requests")
        
        # Process each request in batch
        # Note: OVMS handles internal batching if requests arrive together
        for request in batch:
            try:
                response = self._send_to_ovms(request)
                
                # Log response details
                unique_id = f"{request.station_id}_{request.order_id}"
                logger.info(
                    f"[VLM-RESPONSE] unique_id={unique_id} "
                    f"items={len(response.detected_items)} "
                    f"time={response.inference_time:.2f}s "
                    f"success={response.success}"
                )
                if response.detected_items:
                    logger.info(f"[VLM-ITEMS] unique_id={unique_id} items={response.detected_items}")
                
                # Route response back to station
                response_queue = self.queue_manager.get_response_queue(
                    request.station_id
                )
                logger.debug(f"[VLM-ROUTE] Routing response {response.request_id} to {request.station_id}")
                response_queue.put(response.to_dict())
                logger.debug(f"[VLM-ROUTE] Response queued successfully")
            
            except Exception as e:
                logger.error(
                    f"Error processing request {request.request_id} "
                    f"for station {request.station_id}: {e}"
                )
                self._total_errors += 1
                
                # Send error response
                error_response = VLMResponse(
                    request_id=request.request_id,
                    station_id=request.station_id,
                    order_id=request.order_id,
                    detected_items=[],
                    inference_time=0.0,
                    success=False,
                    error=str(e)
                )
                
                response_queue = self.queue_manager.get_response_queue(
                    request.station_id
                )
                response_queue.put(error_response.to_dict())
        
        batch_time = time.time() - batch_start
        logger.debug(
            f"Batch completed: {len(batch)} requests in {batch_time*1000:.1f}ms "
            f"({batch_time/len(batch)*1000:.1f}ms per request)"
        )
    
    def _send_to_ovms(self, request: VLMRequest) -> VLMResponse:
        """
        Send single request to VLM service using existing OVMS client.
        
        Integrates with existing ovms_client.py and vlm_service.py.
        """
        inference_start = time.time()
        
        try:
            if not self._vlm_client:
                raise Exception("VLM client not initialized")
            
            # Debug logging with unique_id
            unique_id = f"{request.station_id}_{request.order_id}"
            logger.info(f"[VLM-DEBUG] unique_id={unique_id} request_id={request.request_id} frames={len(request.frames)}")
            logger.info(f"[VLM-DEBUG] Frame types: {[type(f).__name__ for f in request.frames]}")
            if request.frames:
                logger.info(f"[VLM-DEBUG] First frame sample: {str(request.frames[0])[:200]}")
            
            # Prepare frames as numpy arrays
            # request.frames should contain image data (numpy arrays or base64)
            images = []
            for frame_data in request.frames:
                if isinstance(frame_data, str):
                    # Base64 encoded string - decode to numpy
                    import base64
                    from PIL import Image
                    from io import BytesIO
                    img_data = base64.b64decode(frame_data)
                    img = Image.open(BytesIO(img_data))
                    images.append(np.array(img))
                elif isinstance(frame_data, dict) and 'data' in frame_data:
                    # Frame dict with data field - could be base64 string or numpy
                    data = frame_data['data']
                    if isinstance(data, str):
                        # Base64 string in dict
                        import base64
                        from PIL import Image
                        from io import BytesIO
                        img_data = base64.b64decode(data)
                        img = Image.open(BytesIO(img_data))
                        images.append(np.array(img))
                    elif isinstance(data, np.ndarray):
                        images.append(data)
                    else:
                        logger.warning(f"Unknown data type in frame dict: {type(data)}")
                elif isinstance(frame_data, np.ndarray):
                    images.append(frame_data)
                else:
                    logger.warning(f"Unknown frame data type: {type(frame_data)}")
            
            if not images:
                raise Exception("No valid images in request")
            
            # Build prompt using existing format from vlm_service.py
            prompt = self._build_vlm_prompt(len(images))
            
            # Call OVMS VLM client
            output = self._vlm_client.generate(
                prompt,
                images=images,
                generation_config=None
            )
            
            # Extract text from output
            raw_text = output.texts[0]
            
            # Log raw VLM output
            logger.info(f"[VLM-RAW] unique_id={unique_id} raw_output_length={len(raw_text)}")
            logger.info(f"[VLM-RAW] unique_id={unique_id} raw_text={raw_text[:500]}")  # First 500 chars
            
            # Extract detected items from VLM output text
            detected_items = self._parse_vlm_output(raw_text)
            
            logger.info(f"[VLM-PARSED] unique_id={unique_id} parsed_items={detected_items}")
            
            inference_time = time.time() - inference_start
            
            logger.debug(
                f"OVMS inference: {request.request_id} "
                f"completed in {inference_time*1000:.1f}ms, "
                f"detected {len(detected_items)} items"
            )
            
            return VLMResponse(
                request_id=request.request_id,
                station_id=request.station_id,
                order_id=request.order_id,
                detected_items=detected_items,
                inference_time=inference_time,
                success=True
            )
        
        except requests.exceptions.Timeout:
            raise Exception("OVMS request timeout")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"OVMS request failed: {e}")
        
        except Exception as e:
            raise Exception(f"VLM inference error: {e}")
    
    def _build_vlm_prompt(self, num_images: int) -> str:
        """
        Build VLM prompt with inventory constraints.
        
        Matches the EXACT prompt format used in vlm_service.py.
        """
        # OVMS automatically associates images with the prompt
        img_tags = ""
        
        prompt = (
            f"You will receive {num_images} frames.\n\n"
            f"Recognize products ONLY from this inventory list:\n"
            f"{INVENTORY_TEXT}\n\n"
            f"Rules:\n"
            f"- Always choose the closest matching inventory item name.\n"
            f"- Never invent new product names outside the list.\n"
            f"- Format strictly as: inventory_item_name x quantity\n"
            f"- If no inventory items are visible, output NO_ITEMS.\n"
            f"{img_tags}"
        )
        return prompt
    
    def _parse_vlm_output(self, raw_text: str) -> List[str]:
        """
        Parse VLM text output to extract detected items.
        
        Uses same logic as VLMComponent.extract_items() from vlm_service.py.
        
        Args:
            raw_text: Raw VLM text output
        
        Returns:
            List of detected food item names
        """
        blacklist = {
            "total", "total items", "items", "quantity",
            "subtotal", "tax", "bill", "amount", "price"
        }
        
        try:
            # Split by commas and clean
            parts = [p.strip() for p in raw_text.split(',')]
            
            # Filter out blacklisted terms and empty strings
            items = []
            for item in parts:
                item_lower = item.lower()
                if item and item_lower not in blacklist:
                    # Additional filtering
                    if not any(skip in item_lower for skip in ['$', 'price', 'total', 'quantity']):
                        items.append(item)
            
            if items:
                logger.debug(
                    f"Parsed {len(items)} items from VLM output: {items}"
                )
            else:
                logger.warning("No items parsed from VLM output")
            return items
        
        except Exception as e:
            logger.warning(f"Failed to parse VLM output: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get scheduler statistics"""
        with self._buffer_lock:
            buffer_size = len(self._request_buffer)
        
        return {
            'total_requests': self._total_requests,
            'total_batches': self._total_batches,
            'total_errors': self._total_errors,
            'buffer_size': buffer_size,
            'work_queue_size': self._work_queue.qsize(),
            'avg_batch_size': (
                self._total_requests / self._total_batches
                if self._total_batches > 0 else 0
            ),
            'error_rate': (
                self._total_errors / self._total_requests
                if self._total_requests > 0 else 0
            )
        }
