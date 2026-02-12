"""VLM Backend Factory - OVMS-only backend"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class VLMBackendFactory:
    """
    Factory for creating OVMS VLM backend instances.
    """
    
    @staticmethod
    def create_backend(
        backend_type: str,
        config: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
    ):
        """
        Create OVMS VLM backend instance.

        Args:
            backend_type: Must be 'ovms'
            config: Configuration dict with OVMS endpoint and model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Tuple of (OVMSVLMClient, MockGenerationConfig)
        """
        backend_type = backend_type.lower()
        
        if backend_type != "ovms":
            raise ValueError(f"Only 'ovms' backend is supported, got: {backend_type}")
        
        return VLMBackendFactory._create_ovms_backend(config, max_new_tokens, temperature)

    @staticmethod
    def _create_ovms_backend(config: dict, max_new_tokens: int, temperature: float):
        """Create OVMS backend."""
        from .ovms_client import OVMSVLMClient, MockGenerationConfig
        
        endpoint = config.get("ovms_endpoint", "http://ovms-vlm:8000")
        model_name = config.get("ovms_model", "Qwen/Qwen2-VL-2B-Instruct")
        timeout = config.get("timeout_sec", 120)
        
        logger.info(f"[BACKEND-FACTORY] Creating OVMS backend: {endpoint}")
        
        client = OVMSVLMClient(
            endpoint=endpoint,
            model_name=model_name,
            timeout=timeout,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        
        # Return tuple: (client, generation_config)
        gen_config = MockGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )
        
        logger.info("[BACKEND-FACTORY] OVMS backend created successfully")
        
        return client, gen_config
