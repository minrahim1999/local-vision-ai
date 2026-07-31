"""Backend factory — automatically selects the best available backend per platform.

Usage:
    from services.backends.factory import create_t2i_backend, create_i2t_backend

    t2i = create_t2i_backend(config)
    i2t = create_i2t_backend(config)
"""
from __future__ import annotations

import logging
from typing import Any

from services.backends.base import ImageToTextBackend, TextToImageBackend
from services.backends.i2t_apple import AppleSmolVLMBackend
from services.backends.i2t_transformers import TransformersVLMBackend
from services.backends.t2i_apple import AppleFlux2Backend
from services.backends.t2i_diffusers import DiffusersBackend

logger = logging.getLogger(__name__)

T2I_BACKENDS = [AppleFlux2Backend, DiffusersBackend]
I2T_BACKENDS = [AppleSmolVLMBackend, TransformersVLMBackend]


def create_t2i_backend(config: dict[str, Any]) -> TextToImageBackend:
    """Create the best available T2I backend for the current platform.

    Priority:
      1. Apple Silicon MLX (FLUX.2) — best quality on Mac
      2. Diffusers (SD 2.1 / SD 1.5) — universal fallback
    """
    for cls in T2I_BACKENDS:
        if cls.is_available():
            logger.info("Selected T2I backend: %s", cls.__name__)
            if cls is AppleFlux2Backend:
                return cls(
                    outputs_dir=config.get("outputs_dir", "outputs/text_to_image"),
                    hf_home=config.get("hf_home"),
                    low_ram=config.get("low_ram", True),
                    mlx_cache_limit_gb=config.get("mlx_cache_limit_gb", 8.0),
                )
            # DiffusersBackend
            return cls(
                model_id=config.get("model_id"),
                device=config.get("device"),
                dtype=config.get("dtype"),
                outputs_dir=config.get("outputs_dir", "outputs/text_to_image"),
                cpu_offload=config.get("cpu_offload", True),
            )
    raise RuntimeError("No T2I backend available on this platform")


def create_i2t_backend(config: dict[str, Any]) -> ImageToTextBackend:
    """Create the best available I2T backend for the current platform.

    Priority:
      1. Apple Silicon MLX (SmolVLM) — smallest, fastest on Mac
      2. Transformers (Qwen2.5-VL) — universal fallback
    """
    for cls in I2T_BACKENDS:
        if cls.is_available():
            logger.info("Selected I2T backend: %s", cls.__name__)
            if cls is AppleSmolVLMBackend:
                return cls(
                    model_id=config.get("model_id", "mlx-community/SmolVLM-256M-Instruct-4bit"),
                    max_tokens=config.get("max_tokens", 512),
                    temperature=config.get("temperature", 0.2),
                    top_p=config.get("top_p", 0.9),
                    json_retries=config.get("json_retries", 3),
                    max_file_size_mb=config.get("max_file_size_mb", 10),
                    max_image_pixels=config.get("max_image_pixels", 2_097_152),
                )
            # TransformersVLMBackend
            return cls(
                model_id=config.get("model_id"),
                max_tokens=config.get("max_tokens", 512),
                temperature=config.get("temperature", 0.2),
                top_p=config.get("top_p", 0.9),
                json_retries=config.get("json_retries", 3),
                max_file_size_mb=config.get("max_file_size_mb", 10),
                max_image_pixels=config.get("max_image_pixels", 2_097_152),
                device=config.get("device"),
            )
    raise RuntimeError("No I2T backend available on this platform")
