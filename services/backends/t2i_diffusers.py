"""Text-to-Image backend: Diffusers + PyTorch for NVIDIA CUDA or CPU fallback.

Works on:
  - Windows + NVIDIA GPU (CUDA)
  - Linux + NVIDIA GPU (CUDA)
  - Any platform with CPU-only (very slow)
"""
from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, StableDiffusionPipeline
from PIL import Image

from api.schemas import GeneratedImage
from services.backends.base import TextToImageBackend

logger = logging.getLogger(__name__)


class DiffusersBackend(TextToImageBackend):
    """T2I via Diffusers. Supports CUDA, MPS, or CPU.

    Automatically selects the best available device and dtype.
    """

    # Prefer smaller models for consumer GPUs (4-12 GB VRAM)
    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-1"  # ~2.6B, runs on 8GB
    FALLBACK_MODEL = "runwayml/stable-diffusion-v1-5"   # ~1.2B, runs on 6GB

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        outputs_dir: str = "outputs/text_to_image",
        cpu_offload: bool = True,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL
        self.device_str = device or self._auto_device()
        self.dtype_str = dtype or self._auto_dtype(self.device_str)
        self.dtype = getattr(torch, self.dtype_str)
        self.device = torch.device(self.device_str)
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.cpu_offload = cpu_offload
        self._pipe: DiffusionPipeline | None = None
        self._ready = False

    @staticmethod
    def is_available() -> bool:
        """Always available — Diffusers works on any platform."""
        try:
            import diffusers
            import torch
            return True
        except ImportError:
            return False

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        try:
            if torch.backends.mps.is_available():
                return "mps"
        except AttributeError:
            pass
        return "cpu"

    @staticmethod
    def _auto_dtype(device: str) -> str:
        if device == "cuda":
            return "float16"
        if device == "mps":
            return "float32"  # fp16 VAE NaN on MPS
        return "float32"

    def load(self) -> None:
        if self._ready:
            return

        logger.info(
            "Loading Diffusers model: %s on %s (%s)",
            self.model_id,
            self.device,
            self.dtype,
        )
        start = time.monotonic()

        try:
            self._pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )
        except Exception as exc:
            logger.warning("Failed to load %s: %s. Falling back to %s", self.model_id, exc, self.FALLBACK_MODEL)
            self.model_id = self.FALLBACK_MODEL
            self._pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                safety_checker=None,
                requires_safety_checker=False,
            )

        if hasattr(self._pipe, "enable_attention_slicing"):
            self._pipe.enable_attention_slicing()
        if hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "enable_slicing"):
            try:
                self._pipe.vae.enable_slicing()
            except Exception:
                pass

        if self.cpu_offload and self.device_str != "cpu":
            self._pipe.enable_sequential_cpu_offload()
        else:
            self._pipe = self._pipe.to(self.device)

        elapsed = time.monotonic() - start
        logger.info("Diffusers model loaded in %.2f s", elapsed)
        self._ready = True

    def unload(self) -> None:
        if not self._ready:
            return
        logger.info("Unloading Diffusers model")
        del self._pipe
        self._pipe = None
        self._ready = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except AttributeError:
            pass
        gc.collect()
        logger.info("Diffusers model unloaded")

    def is_loaded(self) -> bool:
        return self._ready

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.0,
        seed: int | None = None,
    ) -> GeneratedImage:
        if not self._ready or self._pipe is None:
            raise RuntimeError("Backend not loaded")
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device_str).manual_seed(seed)

        start = time.monotonic()
        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        image: Image.Image = result.images[0]
        duration = time.monotonic() - start

        timestamp = int(time.time())
        filename = f"diffusers_{timestamp}_{seed or 'rnd'}.png"
        output_path = self.outputs_dir / filename

        metadata = {
            "model": self.model_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed,
            "duration_seconds": round(duration, 3),
        }
        from PIL.PngImagePlugin import PngInfo

        png_info = PngInfo()
        png_info.add_text("Description", str(metadata))
        image.save(output_path, pnginfo=png_info)

        return GeneratedImage(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed or 0,
            duration_seconds=round(duration, 3),
            model_id=self.model_id,
            output_path=str(output_path.resolve()),
        )
