"""OVMS VLM Client - OpenAI-compatible API client"""

import requests
import base64
import time
import logging
from io import BytesIO
from typing import List, Optional
import numpy as np
from PIL import Image
from vlm_metrics_logger import (
    log_start_time, 
    log_end_time, 
    log_custom_event
)

logger = logging.getLogger(__name__)


class OVMSVLMClient:
    """
    OVMS VLM Client using OpenAI-compatible Chat Completions API.
    Drop-in replacement for openvino_genai.VLMPipeline.
    """

    def __init__(
        self,
        endpoint: str,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        timeout: int = 120,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        """
        Initialize OVMS client.

        Args:
            endpoint: OVMS endpoint (e.g., http://ovms-vlm:8000)
            model_name: Model name in OVMS config
            timeout: Request timeout in seconds
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
        """
        self.endpoint = f"{endpoint}/v3/chat/completions"  # OVMS uses v3 API
        self.model_name = model_name
        self.timeout = timeout
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        
        logger.info(f"[OVMS-CLIENT] Initialized: {endpoint}, model: {model_name}")

    def _encode_image(self, image: np.ndarray) -> str:
        """
        Convert numpy array to base64 data URL.

        Args:
            image: numpy array (HWC, uint8)

        Returns:
            Base64 data URL string
        """
        # Handle BGR to RGB conversion (OpenCV uses BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = image[:, :, ::-1]  # BGR -> RGB
        else:
            image_rgb = image

        pil_img = Image.fromarray(image_rgb.astype('uint8'))
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return f"data:image/png;base64,{img_b64}"

    def generate(
        self,
        prompt: str,
        images: List,
        generation_config=None,
        unique_id: Optional[str] = None,
    ):
        """
        Generate response using OVMS Chat Completions API.
        Compatible with openvino_genai.VLMPipeline.generate() signature.

        Args:
            prompt: Text prompt
            images: List of ov.Tensor or np.ndarray images
            generation_config: GenerationConfig object (for compatibility)
            unique_id: Transaction ID for metrics logging (station_id_order_id)

        Returns:
            Object with .texts[0] attribute (mimics openvino_genai output)
        """
        # Convert images to base64 data URLs
        content = [{"type": "text", "text": prompt}]

        for img in images:
            # Handle ov.Tensor or np.ndarray
            if hasattr(img, 'data'):
                # ov.Tensor -> numpy
                img_array = np.array(img.data).reshape(img.shape)
            else:
                img_array = img

            content.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(img_array)}
            })

        # Build request
        request_data = {
            "model": self.model_name,
            "messages": [{
                "role": "user",
                "content": content
            }],
            "max_completion_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

        # Send request
        try:
            logger.info(f"[OVMS-CLIENT] Sending request to: {self.endpoint}")
            logger.info(f"[OVMS-CLIENT] Request data: model={request_data['model']}, images={len(images)}, prompt_len={len(prompt)}")
            logger.debug(f"[OVMS-CLIENT] Full request: {request_data}")
            
            if unique_id:
                log_start_time("USECASE_1", unique_id)

            request_start = time.time()
            response = requests.post(
                self.endpoint,
                headers={"Content-Type": "application/json"},
                json=request_data,
                timeout=self.timeout,
            )
            logger.info(f"[OVMS-CLIENT] Response status: {response.status_code}")
            
            # Log response body for debugging
            if response.status_code != 200:
                logger.error(f"[OVMS-CLIENT] Error response: {response.text[:500]}")
            
            response.raise_for_status()

            result = response.json()
            total_latency = time.time() - request_start
            
            # Extract text from response
            text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            logger.info(f"[OVMS-CLIENT] Response received in {total_latency:.2f}s")
            logger.debug(f"[OVMS-CLIENT] Generated text: {text[:200]}...")

            # Extract token usage from response
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            generated_tokens = completion_tokens
            
            # Calculate VLM performance metrics
            tpot = (total_latency / generated_tokens) if generated_tokens > 0 else 0.0
            throughput_mean = (generated_tokens / total_latency) if total_latency > 0 else 0.0
            
            # Log VLM metrics
            logger.info(f"[OVMS-CLIENT] ========== VLM METRICS ==========")
            logger.info(f"[OVMS-CLIENT] Generated_tokens: {generated_tokens}")
            logger.info(f"[OVMS-CLIENT] Total_latency: {total_latency:.4f}s")
            logger.info(f"[OVMS-CLIENT] TPOT (Time per output token): {tpot:.4f}s")
            logger.info(f"[OVMS-CLIENT] Throughput_mean (tokens/sec): {throughput_mean:.2f}")
            logger.info(f"[OVMS-CLIENT] Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}")
            logger.info(f"[OVMS-CLIENT] =================================")
            
            if unique_id:
                log_end_time("USECASE_1", unique_id)
                log_custom_event("USECASE_1", "ovms_vlm_request", unique_id, 
                                 generated_tokens=generated_tokens,
                                 total_latency_sec=total_latency,
                                 tpot_sec=tpot,
                                 throughput_mean=throughput_mean,
                                 prompt_tokens=prompt_tokens, 
                                 completion_tokens=completion_tokens)

            # Mimic openvino_genai output format
            class GenerationResult:
                def __init__(self, text):
                    self.texts = [text]

            return GenerationResult(text)

        except requests.exceptions.Timeout:
            logger.error(f"[OVMS-CLIENT] Timeout after {self.timeout}s")
            raise TimeoutError(f"OVMS request timeout after {self.timeout}s")
        except requests.exceptions.RequestException as e:
            logger.error(f"[OVMS-CLIENT] Request failed: {e}")
            raise RuntimeError(f"OVMS request failed: {e}")
        except (KeyError, IndexError) as e:
            logger.error(f"[OVMS-CLIENT] Invalid response format: {e}")
            raise RuntimeError(f"Invalid OVMS response: {e}")


class MockGenerationConfig:
    """Mock GenerationConfig for compatibility."""
    def __init__(self, **kwargs):
        self.max_new_tokens = kwargs.get('max_new_tokens', 512)
        self.temperature = kwargs.get('temperature', 0.2)
        self.do_sample = kwargs.get('do_sample', False)
