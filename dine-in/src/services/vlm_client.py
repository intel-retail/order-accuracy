"""
VLM Client Service for OpenVINO Model Server interaction.
Implements Adapter pattern for VLM inference abstraction.
"""

import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from io import BytesIO
import httpx
from PIL import Image

logger = logging.getLogger(__name__)


class VLMResponse:
    """Value object for VLM inference results"""
    
    def __init__(self, raw_response: Dict[str, Any]):
        self.raw_response = raw_response
        self.detected_items: List[Dict[str, Any]] = []
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
                    logger.info(f"[PARSE] JSON decode failed: {je}, falling back to natural language parsing")
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


class VLMClient:
    """
    VLM Client implementing Adapter pattern.
    Provides abstraction over OpenVINO Model Server VLM endpoint.
    """
    
    def __init__(self, endpoint: str, model_name: str, timeout: int = 60):
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout = timeout
        self.chat_endpoint = f"{endpoint}/v3/chat/completions"
        self.inventory_items = self._load_inventory()
        logger.info(f"VLM Client initialized: endpoint={endpoint}, model={model_name}, inventory_items={len(self.inventory_items)}")
    
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
    
    def _encode_image(self, image_bytes: bytes) -> str:
        """Encode image to base64 for VLM input"""
        try:
            # Validate image can be opened
            img = Image.open(BytesIO(image_bytes))
            logger.debug(f"Image validated: format={img.format}, size={img.size}")
            
            # Encode to base64
            encoded = base64.b64encode(image_bytes).decode('utf-8')
            return f"data:image/jpeg;base64,{encoded}"
        except Exception as e:
            logger.exception(f"Error encoding image: {e}")
            raise
    
    def _build_prompt(self) -> str:
        """Build inventory-aware structured prompt for food plate analysis"""
        # Format inventory items as numbered list
        if self.inventory_items:
            inventory_text = "\n".join([f"  {i+1}. {item}" for i, item in enumerate(self.inventory_items)])
            prompt = f"""Analyze this food plate image carefully and recognize products ONLY from this inventory list:

{inventory_text}

Rules:
- Always choose the closest matching inventory item name (exact case-sensitive match preferred)
- Never invent new product names outside the inventory list
- If you see an item similar to one in inventory, use the inventory name
- Count quantities accurately

Return the result as a JSON object with this exact structure:
{{
  "items": [
    {{"name": "exact_inventory_item_name", "quantity": number}},
    ...
  ]
}}

If no inventory items are visible in the image, return: {{"items": []}}"""
        else:
            # Fallback if inventory not loaded
            prompt = """Analyze this food plate image carefully. List all food items you can identify with their quantities.

Return the result as a JSON object with this exact structure:
{
  "items": [
    {"name": "item_name", "quantity": number},
    ...
  ]
}

Be specific with item names. Count quantities accurately."""
        
        logger.info(f"[PROMPT] Built prompt with {len(self.inventory_items)} inventory items")
        logger.info(f"[PROMPT] Full prompt: {prompt}")
        return prompt
    
    async def analyze_plate(self, image_bytes: bytes, request_id: str = None) -> VLMResponse:
        """
        Analyze food plate image using VLM.
        
        Args:
            image_bytes: Raw image bytes
            request_id: Optional unique request identifier for tracking
            
        Returns:
            VLMResponse with detected items
            
        Raises:
            httpx.HTTPError: On network or API errors
        """
        req_id = request_id or "unknown"
        logger.info(f"Starting VLM analysis for request_id={req_id}")
        vlm_request_start = time.time()
        
        try:
            # Encode image
            encoded_image = self._encode_image(image_bytes)
            
            # Build request payload (OpenAI-compatible format)
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
                "max_tokens": 500,
                "temperature": 0.1  # Low temperature for consistent structured output
            }
            
            # Make async request with extended timeout for 7B model
            # Use separate timeouts: connect=10s, read=300s for long inference
            timeout_config = httpx.Timeout(connect=10.0, read=300.0, write=10.0, pool=10.0)
            logger.info(f"[VLM_REQUEST] Endpoint: {self.chat_endpoint}")
            logger.info(f"[VLM_REQUEST] Model: {self.model_name}")
            logger.info(f"[VLM_REQUEST] Timeout config: connect=10s, read=300s")
            
            async with httpx.AsyncClient(timeout=timeout_config) as client:
                logger.info(f"[VLM_REQUEST] Sending POST to {self.chat_endpoint} for {req_id}")
                request_sent_time = time.time()
                
                response = await client.post(
                    self.chat_endpoint,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                
                response_received_time = time.time()
                vlm_latency_ms = (response_received_time - request_sent_time) * 1000
                
                result = response.json()
                logger.info(f"[VLM_RESPONSE] Received from OVMS for {req_id}, latency={vlm_latency_ms:.2f}ms")
                logger.info(f"[VLM_RESPONSE] Raw response: {result}")  # Changed from debug to info
                
                return VLMResponse(result)
                
        except httpx.HTTPError as e:
            logger.error(f"HTTP error during VLM analysis: {e}")
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during VLM analysis: {e}")
            raise
