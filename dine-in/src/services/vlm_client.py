"""
VLM Client Service for OpenVINO Model Server interaction.
Implements Adapter pattern for VLM inference abstraction.
"""

import asyncio
import base64
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from io import BytesIO
import httpx
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from vlm_metrics_logger import (
    log_start_time, 
    log_end_time, 
    log_custom_event
)

logger = logging.getLogger(__name__)


# ============================================================================
# Circuit Breaker Pattern for Fault Tolerance
# ============================================================================

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5      # Failures before opening circuit
    recovery_timeout: float = 30.0  # Seconds before trying half-open
    success_threshold: int = 2      # Successes in half-open to close


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and rejecting requests"""
    pass


class CircuitBreaker:
    """
    Thread-safe circuit breaker for OVMS service.
    
    Prevents cascading failures by failing fast when service is unhealthy.
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if request can proceed through circuit breaker."""
        async with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            
            if self._state == CircuitState.OPEN:
                # Check if recovery timeout has passed
                if self._last_failure_time:
                    elapsed = (datetime.now() - self._last_failure_time).total_seconds()
                    if elapsed >= self.config.recovery_timeout:
                        self._state = CircuitState.HALF_OPEN
                        self._success_count = 0
                        logger.info("Circuit breaker transitioning to HALF_OPEN")
                        return True
                return False
            
            # HALF_OPEN state - allow limited requests
            return True
    
    async def record_success(self):
        """Record successful request."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.config.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._failure_count = 0
                    logger.info("Circuit breaker CLOSED - service recovered")
            elif self._state == CircuitState.CLOSED:
                # Decay failure count on success
                self._failure_count = max(0, self._failure_count - 1)
    
    async def record_failure(self):
        """Record failed request."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = datetime.now()
            
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning("Circuit breaker OPEN - service still failing")
            elif self._failure_count >= self.config.failure_threshold:
                self._state = CircuitState.OPEN
                logger.warning(f"Circuit breaker OPEN after {self._failure_count} failures")
    
    @property
    def state(self) -> CircuitState:
        return self._state


class ImagePreprocessor:
    """
    High-performance image preprocessor optimized for VLM inference.
    
    Applies intelligent preprocessing to reduce inference time while
    maintaining visual quality for accurate food item detection.
    """
    
    # Optimal dimensions for Qwen2-VL models (balance quality vs speed)
    DEFAULT_MAX_SIZE = 672  # Sweet spot for VLM quality/speed tradeoff
    MIN_SIZE = 224
    
    # JPEG quality for base64 encoding (80-85 is optimal for VLM)
    JPEG_QUALITY = 82
    
    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        jpeg_quality: int = JPEG_QUALITY,
        enhance_contrast: bool = True,
        sharpen: bool = True
    ):
        """
        Initialize preprocessor with configurable parameters.
        
        Args:
            max_size: Maximum dimension (width or height) for resizing
            jpeg_quality: JPEG compression quality (1-100)
            enhance_contrast: Apply adaptive contrast enhancement
            sharpen: Apply light sharpening for edge clarity
        """
        self.max_size = max_size
        self.jpeg_quality = jpeg_quality
        self.enhance_contrast = enhance_contrast
        self.sharpen = sharpen
        logger.info(f"ImagePreprocessor initialized: max_size={max_size}, "
                   f"quality={jpeg_quality}, contrast={enhance_contrast}, sharpen={sharpen}")
    
    def preprocess(self, image_bytes: bytes) -> Tuple[bytes, Dict[str, Any]]:
        """
        Apply full preprocessing pipeline to optimize image for VLM.
        
        Pipeline:
        1. Load and convert to RGB (remove alpha channel)
        2. Auto-orient based on EXIF
        3. Smart resize maintaining aspect ratio
        4. Contrast enhancement (adaptive)
        5. Light sharpening for food detail
        6. Optimized JPEG compression
        
        Args:
            image_bytes: Raw input image bytes
            
        Returns:
            Tuple of (processed_bytes, metadata_dict)
        """
        preprocess_start = time.time()
        metadata: Dict[str, Any] = {"original_size": len(image_bytes)}
        
        try:
            # Load image
            img = Image.open(BytesIO(image_bytes))
            metadata["original_dimensions"] = img.size
            metadata["original_format"] = img.format
            metadata["original_mode"] = img.mode
            
            # Step 1: Auto-orient based on EXIF data
            img = ImageOps.exif_transpose(img)
            
            # Step 2: Convert to RGB (handles RGBA, P, L modes)
            if img.mode != 'RGB':
                # Handle transparency by compositing on white background
                if img.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', img.size, (255, 255, 255))
                    if img.mode == 'P':
                        img = img.convert('RGBA')
                    background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                    img = background
                else:
                    img = img.convert('RGB')
            
            # Step 3: Smart resize with aspect ratio preservation
            img, resize_info = self._smart_resize(img)
            metadata.update(resize_info)
            
            # Step 4: Adaptive contrast enhancement (improves food item visibility)
            if self.enhance_contrast:
                img = self._enhance_contrast(img)
                metadata["contrast_enhanced"] = True
            
            # Step 5: Light sharpening (improves text and edges)
            if self.sharpen:
                img = self._apply_sharpening(img)
                metadata["sharpened"] = True
            
            # Step 6: Optimized JPEG encoding
            output_buffer = BytesIO()
            img.save(
                output_buffer,
                format='JPEG',
                quality=self.jpeg_quality,
                optimize=True,
                progressive=True  # Progressive JPEG for better compression
            )
            processed_bytes = output_buffer.getvalue()
            
            # Calculate metrics
            preprocess_time_ms = (time.time() - preprocess_start) * 1000
            compression_ratio = len(image_bytes) / len(processed_bytes) if processed_bytes else 1
            
            metadata.update({
                "processed_size": len(processed_bytes),
                "processed_dimensions": img.size,
                "compression_ratio": round(compression_ratio, 2),
                "preprocess_time_ms": round(preprocess_time_ms, 2),
                "size_reduction_percent": round((1 - len(processed_bytes)/len(image_bytes)) * 100, 1)
            })
            
            logger.info(f"[PREPROCESS] {metadata['original_dimensions']} -> {img.size}, "
                       f"compression={compression_ratio:.1f}x, time={preprocess_time_ms:.1f}ms, "
                       f"size: {len(image_bytes)//1024}KB -> {len(processed_bytes)//1024}KB")
            
            return processed_bytes, metadata
            
        except Exception as e:
            logger.error(f"[PREPROCESS] Error: {e}, returning original image")
            return image_bytes, {"error": str(e), "fallback": True}
    
    def _smart_resize(self, img: Image.Image) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        Intelligently resize image while preserving aspect ratio.
        
        Uses high-quality LANCZOS resampling for downscaling.
        Only resizes if image exceeds max_size.
        """
        original_width, original_height = img.size
        info: Dict[str, Any] = {"resize_applied": False}
        
        # Check if resize needed
        if max(original_width, original_height) <= self.max_size:
            info["resize_reason"] = "already_optimal"
            return img, info
        
        # Calculate new dimensions preserving aspect ratio
        if original_width > original_height:
            new_width = self.max_size
            new_height = int(original_height * (self.max_size / original_width))
        else:
            new_height = self.max_size
            new_width = int(original_width * (self.max_size / original_height))
        
        # Ensure minimum dimensions
        new_width = max(new_width, self.MIN_SIZE)
        new_height = max(new_height, self.MIN_SIZE)
        
        # Use LANCZOS for high-quality downsampling
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        info.update({
            "resize_applied": True,
            "scale_factor": round(new_width / original_width, 3),
            "resize_reason": f"exceeded_max_{self.max_size}"
        })
        
        return img, info
    
    def _enhance_contrast(self, img: Image.Image) -> Image.Image:
        """
        Apply adaptive contrast enhancement optimized for food images.
        
        Uses a moderate enhancement factor to improve item visibility
        without over-saturating colors.
        """
        # Moderate contrast boost (1.0 = no change, 1.2 = 20% increase)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.15)
        
        # Slight color saturation boost for food items
        color_enhancer = ImageEnhance.Color(img)
        img = color_enhancer.enhance(1.08)
        
        return img
    
    def _apply_sharpening(self, img: Image.Image) -> Image.Image:
        """
        Apply light sharpening to improve edge detection.
        
        Uses UnsharpMask which is superior to simple Sharpen filter
        for preserving natural appearance while enhancing details.
        """
        # UnsharpMask: radius=1, percent=50, threshold=3
        # Light sharpening that doesn't introduce artifacts
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))


class VLMResponse:
    """Value object for VLM inference results"""
    
    def __init__(self, raw_response: Dict[str, Any]):
        self.raw_response = raw_response
        self.detected_items: List[Dict[str, Any]] = []
        self.performance_metadata: Dict[str, Any] = {}  # Set by VLMClient after inference
        self._parse_response()
    
    def _parse_response(self):
        """Parse VLM response to extract detected items"""
        try:
            # Extract content from OpenAI-compatible response
            if "choices" in self.raw_response:
                content = self.raw_response["choices"][0]["message"]["content"]
                logger.info(f"[PARSE] VLM content: {content[:500]}")  # Log first 500 chars
                
                # Strip markdown code blocks if present (```json ... ```)
                content_stripped = content.strip()
                if content_stripped.startswith("```"):
                    # Remove opening ```json or ```
                    lines = content_stripped.split("\n")
                    if lines[0].startswith("```"):
                        lines = lines[1:]  # Remove first line
                    if lines and lines[-1].strip() == "```":
                        lines = lines[:-1]  # Remove last line
                    content_stripped = "\n".join(lines)
                    logger.info(f"[PARSE] Stripped markdown code blocks")
                
                # Try to parse as JSON first (structured output)
                try:
                    parsed_content = json.loads(content_stripped)
                    logger.info(f"[PARSE] Successfully parsed JSON: {parsed_content}")
                    if isinstance(parsed_content, dict) and "items" in parsed_content:
                        self.detected_items = parsed_content["items"]
                        logger.info(f"[PARSE] Extracted {len(self.detected_items)} items from JSON dict")
                    elif isinstance(parsed_content, list):
                        self.detected_items = parsed_content
                        logger.info(f"[PARSE] Extracted {len(self.detected_items)} items from JSON list")
                    else:
                        logger.warning(f"[PARSE] Unexpected JSON structure: {parsed_content}")
                except json.JSONDecodeError as je:
                    logger.info(f"[PARSE] JSON decode failed: {je}, trying to recover truncated JSON")
                    # Try to recover items from truncated JSON response
                    recovered_items = self._recover_truncated_json(content_stripped)
                    if recovered_items:
                        self.detected_items = recovered_items
                        logger.info(f"[PARSE] Recovered {len(self.detected_items)} items from truncated JSON")
                    else:
                        # Fallback: parse natural language response
                        self._parse_natural_language(content)
                    
                logger.info(f"Parsed {len(self.detected_items)} items from VLM response")
            else:
                logger.error(f"Unexpected VLM response format: {self.raw_response}")
        except Exception as e:
            logger.exception(f"Error parsing VLM response: {e}")
    
    def _parse_natural_language(self, content: str):
        """Fallback parser for natural language VLM responses"""
        # Simple pattern matching for common food item descriptions
        # Format: "- item_name (quantity: N)" or "- item_name x N"
        import re
        patterns = [
            r'-\s*([^(]+)\s*\(quantity:\s*(\d+)\)',
            r'-\s*([^x]+)\s*x\s*(\d+)',
            r'(\d+)\s*x\s*([^,\n]+)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                for match in matches:
                    if len(match) == 2:
                        name, quantity = match if pattern.startswith(r'-\s*([^(]') else (match[1], match[0])
                        self.detected_items.append({
                            "name": name.strip(),
                            "quantity": int(quantity)
                        })
                break
        
        logger.info(f"Parsed {len(self.detected_items)} items from natural language")
    
    def _recover_truncated_json(self, content: str) -> List[Dict[str, Any]]:
        """
        Recover complete items from truncated JSON array response.
        
        When VLM hits max_tokens limit, the JSON may be truncated mid-object.
        This method extracts all complete JSON objects before the truncation.
        
        Args:
            content: Potentially truncated JSON string
            
        Returns:
            List of successfully parsed item dictionaries
        """
        import re
        
        recovered_items = []
        
        try:
            # Find all complete JSON objects in the array
            # Pattern matches: {"item": "...", "quantity": N} or {"name": "...", "quantity": N}
            item_pattern = r'\{\s*"(?:item|name)"\s*:\s*"([^"]+)"\s*,\s*"quantity"\s*:\s*(\d+)\s*\}'
            
            matches = re.findall(item_pattern, content, re.IGNORECASE | re.DOTALL)
            
            for match in matches:
                item_name, quantity = match
                recovered_items.append({
                    "item": item_name.strip(),
                    "quantity": int(quantity)
                })
            
            if recovered_items:
                logger.info(f"[PARSE] Recovered {len(recovered_items)} complete items from truncated JSON")
            else:
                logger.warning("[PARSE] No complete items found in truncated JSON")
                
        except Exception as e:
            logger.warning(f"[PARSE] Error recovering truncated JSON: {e}")
        
        return recovered_items


class VLMClient:
    """
    VLM Client implementing Adapter pattern with connection pooling and circuit breaker.
    Provides abstraction over OpenVINO Model Server VLM endpoint.
    
    Features:
    - Persistent HTTP connection pool (avoids TCP handshake per request)
    - Circuit breaker for fault tolerance
    - Configurable timeouts per operation stage
    """
    
    # Class-level HTTP client pool (shared across instances)
    _http_client: Optional[httpx.AsyncClient] = None
    _client_lock = asyncio.Lock()
    
    def __init__(self, endpoint: str, model_name: str, timeout: int = 60):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.chat_endpoint = f"{endpoint}/v3/chat/completions"
        self.inventory_items = self._load_inventory()
        
        # Circuit breaker for OVMS service
        self._circuit_breaker = CircuitBreaker(
            config=CircuitBreakerConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                success_threshold=2
            )
        )
        
        # Initialize image preprocessor for optimized VLM inference
        # Balanced settings for 7B model - good quality with reasonable speed
        self.preprocessor = ImagePreprocessor(
            max_size=672,      # Good quality for 7B model
            jpeg_quality=82,   # High quality compression
            enhance_contrast=True,
            sharpen=True
        )
        
        logger.info(f"VLM Client initialized: endpoint={endpoint}, model={model_name}, "
                   f"inventory_items={len(self.inventory_items)}, preprocessing=enabled, "
                   f"circuit_breaker=enabled")
    
    @classmethod
    async def get_http_client(cls) -> httpx.AsyncClient:
        """
        Get or create shared HTTP client with connection pooling.
        Thread-safe initialization using async lock.
        """
        if cls._http_client is None or cls._http_client.is_closed:
            async with cls._client_lock:
                if cls._http_client is None or cls._http_client.is_closed:
                    # Configure connection pool limits
                    limits = httpx.Limits(
                        max_keepalive_connections=20,
                        max_connections=50,
                        keepalive_expiry=30.0
                    )
                    # Extended timeout for VLM inference
                    timeout = httpx.Timeout(
                        connect=10.0,
                        read=300.0,
                        write=10.0,
                        pool=10.0
                    )
                    cls._http_client = httpx.AsyncClient(
                        limits=limits,
                        timeout=timeout,
                        http2=True  # Enable HTTP/2 for better performance
                    )
                    logger.info("Created shared HTTP client with connection pooling")
        return cls._http_client
    
    @classmethod
    async def close_http_client(cls):
        """Close the shared HTTP client (call on shutdown)"""
        if cls._http_client is not None:
            await cls._http_client.aclose()
            cls._http_client = None
            logger.info("Closed shared HTTP client")
    
    def _load_inventory(self) -> List[str]:
        """Load inventory items from inventory.json"""
        try:
            # Since vlm_client.py is in /app/src/services/, go up to /app/
            base_dir = Path(__file__).resolve().parent.parent.parent
            inventory_path = base_dir / "configs" / "inventory.json"
            
            if not inventory_path.exists():
                logger.warning(f"Inventory file not found at {inventory_path}, using empty inventory")
                return []
            
            with open(inventory_path, 'r') as f:
                items = json.load(f)
            
            logger.info(f"Loaded {len(items)} inventory items from {inventory_path}")
            return items
        except Exception as e:
            logger.error(f"Error loading inventory: {e}")
            return []
    
    def _encode_image(self, image_bytes: bytes, skip_preprocessing: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Preprocess and encode image to base64 for VLM input.
        
        Args:
            image_bytes: Raw image bytes
            skip_preprocessing: If True, skip optimization (for debugging)
            
        Returns:
            Tuple of (base64_encoded_string, preprocessing_metadata)
        """
        try:
            preprocess_metadata = {}
            
            # Apply preprocessing pipeline unless skipped
            if not skip_preprocessing:
                processed_bytes, preprocess_metadata = self.preprocessor.preprocess(image_bytes)
            else:
                processed_bytes = image_bytes
                preprocess_metadata = {"preprocessing_skipped": True}
            
            # Validate processed image can be opened
            img = Image.open(BytesIO(processed_bytes))
            logger.debug(f"Image validated: format={img.format}, size={img.size}")
            
            # Encode to base64
            encoded = base64.b64encode(processed_bytes).decode('utf-8')
            
            return f"data:image/jpeg;base64,{encoded}", preprocess_metadata
            
        except Exception as e:
            logger.exception(f"Error encoding image: {e}")
            raise
    
    def _build_prompt(self) -> str:
        """Build ultra-compact prompt for fast inference on iGPU"""
        if self.inventory_items:
            # Compact format: comma-separated items (faster than numbered list)
            inventory_text = ", ".join(self.inventory_items[:30])  # Limit to 30 items
            prompt = f"""Food items to detect: {inventory_text}

List food items visible in image with quantities.
JSON: {{"items":[{{"name":"item","quantity":1}}]}}"""
        else:
            prompt = """List food items in image with quantities.
JSON: {"items":[{"name":"item","quantity":1}]}"""
        
        logger.info(f"[PROMPT] Built compact prompt with {len(self.inventory_items)} inventory items, length={len(prompt)} chars")
        return prompt
    
    async def analyze_plate(
        self, 
        image_bytes: bytes, 
        order_id: Optional[str] = None,
        request_id: Optional[str] = None
    ) -> VLMResponse:
        """
        Analyze food plate image using VLM with optimized preprocessing.
        
        Pipeline:
        1. Image preprocessing (resize, enhance, compress)
        2. Base64 encoding
        3. VLM inference via OVMS
        4. Response parsing
        
        Args:
            image_bytes: Raw image bytes
            order_id: Optional order identifier for tracking
            request_id: Optional unique request identifier for tracking (deprecated, use order_id)
            
        Returns:
            VLMResponse with detected items
            
        Raises:
            httpx.HTTPError: On network or API errors
        """
        # Generate unique ID for metrics logging using order_id
        # Format: dine_in_{order_id} or dine_in_{uuid} if no order_id provided
        if order_id:
            unique_id = f"dine_in_{order_id}"
        elif request_id:
            unique_id = request_id  # Backward compatibility
        else:
            unique_id = f"dine_in_{uuid.uuid4().hex[:12]}"
        req_id = unique_id
        logger.info(f"[VLM] Starting analysis for request_id={req_id}, input_size={len(image_bytes)//1024}KB")
        total_start = time.time()
        
        try:
            # Step 1: Preprocess and encode image
            encode_start = time.time()
            encoded_image, preprocess_meta = self._encode_image(image_bytes)
            encode_time_ms = (time.time() - encode_start) * 1000
            
            logger.info(f"[VLM] Preprocessing completed for {req_id}: "
                       f"time={encode_time_ms:.1f}ms, "
                       f"compression={preprocess_meta.get('compression_ratio', 'N/A')}x, "
                       f"dims={preprocess_meta.get('processed_dimensions', 'N/A')}")
            
            # Step 2: Build request payload (OpenAI-compatible format)
            payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_prompt()},
                            {"type": "image_url", "image_url": {"url": encoded_image}}
                        ]
                    }
                ],
                "max_tokens": 512,  # Increased to handle longer item lists
                "temperature": 0.0  # Greedy decoding for fastest inference
            }
            
            # Step 3: Check circuit breaker before making request
            if not await self._circuit_breaker.can_execute():
                raise CircuitOpenError(f"Circuit breaker is OPEN for OVMS service. Service may be unavailable.")
            
            logger.info(f"[VLM_REQUEST] Endpoint: {self.chat_endpoint}")
            logger.info(f"[VLM_REQUEST] Model: {self.model_name}")
            logger.info(f"[VLM_REQUEST] Payload size: {len(str(payload))//1024}KB")
            
            # Step 4: Make async request using shared client with connection pooling
            client = await self.get_http_client()
            try:
                logger.info(f"[VLM_REQUEST] Sending POST to {self.chat_endpoint} for {req_id}")
                inference_start = time.time()
                
                # Log start time for metrics
                log_start_time("ovms_vlm_request", unique_id)
                
                response = await client.post(
                    self.chat_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                # Record success for circuit breaker
                await self._circuit_breaker.record_success()
                
                inference_time_ms = (time.time() - inference_start) * 1000
                total_time_ms = (time.time() - total_start) * 1000
                
                # Log end time for metrics
                log_end_time("ovms_vlm_request", unique_id)
                
                result = response.json()
                
                # Step 4: Create response with enhanced metadata
                vlm_response = VLMResponse(result)
                
                # Attach performance metadata to response
                vlm_response.performance_metadata = {
                    "preprocess_time_ms": round(encode_time_ms, 2),
                    "inference_time_ms": round(inference_time_ms, 2),
                    "total_time_ms": round(total_time_ms, 2),
                    "input_size_kb": len(image_bytes) // 1024,
                    "processed_size_kb": preprocess_meta.get("processed_size", 0) // 1024,
                    "compression_ratio": preprocess_meta.get("compression_ratio", 1.0),
                    "image_dimensions": preprocess_meta.get("processed_dimensions", None)
                }
                
                # Extract token usage for metrics
                usage = result.get("usage", {})
                logger.info(f"[OVMS-CLIENT] result: {result}")
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                elapsed_sec = inference_time_ms / 1000
                tps = completion_tokens / elapsed_sec if elapsed_sec > 0 else 0
                logger.info(f"[OVMS-CLIENT] Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
                logger.info(f"[OVMS-CLIENT] Tokens per second (TPS): {tps:.2f}")
                
                # Log custom metrics event
                log_custom_event(
                    "ovms_metrics", "ovms_vlm_request", unique_id,
                    tps=tps,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    elapsed_sec=elapsed_sec,
                    preprocess_ms=encode_time_ms,
                    items_detected=len(vlm_response.detected_items)
                )
                
                logger.info(f"[VLM_RESPONSE] Completed for {req_id}: "
                           f"preprocess={encode_time_ms:.1f}ms, "
                           f"inference={inference_time_ms:.1f}ms, "
                           f"total={total_time_ms:.1f}ms, "
                           f"items_detected={len(vlm_response.detected_items)}, "
                           f"tps={tps:.2f}")
                
                return vlm_response
            
            except httpx.HTTPError as e:
                # Record failure for circuit breaker
                await self._circuit_breaker.record_failure()
                logger.error(f"HTTP error during VLM analysis: {e}")
                raise
                
        except CircuitOpenError as e:
            logger.error(f"Circuit breaker prevented request: {e}")
            raise
        except Exception as e:
            # Record failure for circuit breaker on any exception
            await self._circuit_breaker.record_failure()
            logger.exception(f"Unexpected error during VLM analysis: {e}")
            raise
