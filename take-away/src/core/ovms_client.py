"""OVMS VLM Client - OpenAI-compatible API client"""

import requests
import base64
import time
import logging
from io import BytesIO
from typing import List, Optional
import numpy as np
from PIL import Image

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
    ):
        """
        Generate response using OVMS Chat Completions API.
        Compatible with openvino_genai.VLMPipeline.generate() signature.

        Args:
            prompt: Text prompt
            images: List of ov.Tensor or np.ndarray images
            generation_config: GenerationConfig object (for compatibility)

        Returns:
            Object with .texts[0] attribute (mimics openvino_genai output)
        """
        start_time = time.time()

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
            elapsed = time.time() - start_time

            # Extract text
            text = result["choices"][0]["message"]["content"]
            logger.info(f"[OVMS-CLIENT] Response received in {elapsed:.2f}s")
            logger.debug(f"[OVMS-CLIENT] Generated text: {text[:200]}...")

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
