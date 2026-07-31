"""FastAPI dependencies and lifespan."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import yaml
from fastapi import FastAPI

from services.image_to_text import ImageToTextService
from services.memory_manager import MemoryManager
from services.text_to_image import TextToImageService

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared services on startup and clean up on shutdown."""
    t2i_cfg = _load_yaml(CONFIG_DIR / "text_to_image.yaml").get("t2i", {})
    i2t_cfg = _load_yaml(CONFIG_DIR / "image_to_text.yaml").get("i2t", {})

    mm = MemoryManager(
        memory_warn_mb=int(os.getenv("MEMORY_WARN_MB", "12000")),
        memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB", "14000")),
        idle_timeout_seconds=float(os.getenv("IDLE_TIMEOUT_SECONDS", "300")),
    )

    t2i = TextToImageService(
        model_id=t2i_cfg.get("model_id", "runwayml/stable-diffusion-v1-5"),
        device=t2i_cfg.get("device", "mps"),
        dtype=t2i_cfg.get("dtype", "float16"),
        safety_checker=t2i_cfg.get("safety_checker", False),
        attention_slicing=t2i_cfg.get("attention_slicing", True),
        vae_slicing=t2i_cfg.get("vae_slicing", True),
        cpu_offload=t2i_cfg.get("cpu_offload", False),
        outputs_dir="outputs/text_to_image",
    )

    i2t = ImageToTextService(
        model_id=i2t_cfg.get("model_id", "mlx-community/smolvlm-256m-8bit"),
        max_tokens=i2t_cfg.get("max_tokens", 512),
        temperature=i2t_cfg.get("temperature", 0.2),
        top_p=i2t_cfg.get("top_p", 0.9),
        json_retries=i2t_cfg.get("json_retries", 3),
        max_file_size_mb=i2t_cfg.get("max_file_size_mb", 10),
        max_image_pixels=i2t_cfg.get("max_image_pixels", 2_097_152),
    )

    mm.register("text_to_image", t2i)
    mm.register("image_to_text", i2t)

    app.state.memory_manager = mm
    app.state.text_to_image = t2i
    app.state.image_to_text = i2t

    logger.info("Services registered. Ready to accept requests.")
    yield
    logger.info("Shutting down. Releasing all models.")
    mm.release_all()
