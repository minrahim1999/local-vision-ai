"""Text-to-Image inference service using Stable Diffusion via Diffusers + MPS."""
from __future__ import annotations

import gc
import logging
import os
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image

from api.schemas import GeneratedImage

logger = logging.getLogger(__name__)


class TextToImageService:
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: str = "mps",
        dtype: str = "float16",
        safety_checker: bool = False,
        attention_slicing: bool = True,
        vae_slicing: bool = True,
        cpu_offload: bool = False,
        outputs_dir: str = "outputs/text_to_image",
    ) -> None:
        self.model_id = model_id
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.safety_checker = safety_checker
        self.attention_slicing = attention_slicing
        self.vae_slicing = vae_slicing
        self.cpu_offload = cpu_offload
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self._pipe: StableDiffusionPipeline | None = None
        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._is_loaded:
            logger.info("T2I model already loaded")
            return

        logger.info("Loading T2I model: %s", self.model_id)
        start = time.monotonic()

        kwargs: dict = {
            "pretrained_model_name_or_path": self.model_id,
            "torch_dtype": self.dtype,
            "safety_checker": None if not self.safety_checker else "",
        }

        # Remove safety_checker key entirely if disabled, to avoid noisy warnings
        if not self.safety_checker:
            kwargs.pop("safety_checker", None)

        self._pipe = StableDiffusionPipeline.from_pretrained(
            kwargs.pop("pretrained_model_name_or_path"),
            torch_dtype=kwargs.get("torch_dtype"),
            safety_checker=None,
            requires_safety_checker=False,
        )

        if self.attention_slicing:
            if hasattr(self._pipe, "enable_attention_slicing"):
                self._pipe.enable_attention_slicing()
                logger.info("Attention slicing enabled")

        if self.vae_slicing:
            if hasattr(self._pipe, "vae") and hasattr(self._pipe.vae, "enable_slicing"):
                try:
                    self._pipe.vae.enable_slicing()
                    logger.info("VAE slicing enabled")
                except Exception:
                    logger.warning("VAE slicing not supported on this model")

        if self.cpu_offload:
            self._pipe.enable_sequential_cpu_offload()
            logger.info("CPU offload enabled")
        else:
            self._pipe = self._pipe.to(self.device)

        # Warm-up to avoid first-run MPS compilation overhead
        logger.info("Warming up T2I pipeline with a tiny forward pass...")
        try:
            with torch.no_grad():
                _ = self._pipe(
                    "a cat",
                    num_inference_steps=1,
                    height=64,
                    width=64,
                    guidance_scale=1.0,
                )
        except Exception as exc:
            logger.warning("T2I warm-up failed (non-fatal): %s", exc)

        elapsed = time.monotonic() - start
        logger.info("T2I model loaded in %.2f s", elapsed)
        self._is_loaded = True

    def unload(self) -> None:
        if not self._is_loaded:
            return
        logger.info("Unloading T2I model")
        del self._pipe
        self._pipe = None
        self._is_loaded = False
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()
        logger.info("T2I model unloaded")

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
        if not self._is_loaded or self._pipe is None:
            raise RuntimeError("TextToImageService is not loaded. Call load() first.")

        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")

        if width > 768 or height > 768:
            raise ValueError(
                f"Dimensions {width}x{height} exceed safe limit of 768x768"
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        start = time.monotonic()
        try:
            result = self._pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
        except NotImplementedError as exc:
            logger.error("MPS operation not supported: %s", exc)
            raise RuntimeError(
                "An operation required by the diffusion pipeline is not supported on MPS. "
                "Try lowering resolution or switching to a CPU fallback (not recommended)."
            ) from exc
        except torch.cuda.OutOfMemoryError as exc:
            # Should not happen on MPS, but kept for completeness
            raise MemoryError("Out of memory during image generation") from exc
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise MemoryError("Out of memory during image generation") from exc
            raise

        image: Image.Image = result.images[0]
        duration = time.monotonic() - start

        # Save with metadata
        timestamp = int(time.time())
        filename = f"t2i_{timestamp}_{seed or 'rnd'}.png"
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
