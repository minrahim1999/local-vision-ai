"""Image-to-Text backend: Qwen2.5-VL via Transformers on NVIDIA CUDA.

Works on:
  - Windows + NVIDIA GPU (CUDA)
  - Linux + NVIDIA GPU (CUDA)
  - Any platform with CPU (very slow)
"""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

from api.schemas import VisionResult
from services.backends.base import ImageToTextBackend

logger = logging.getLogger(__name__)


class TransformersVLMBackend(ImageToTextBackend):
    """Vision-Language model via Hugging Face Transformers.

    Automatically selects device (cuda > mps > cpu).
    Uses Qwen2.5-VL-Instruct by default (3B fits on 8GB VRAM).
    """

    DEFAULT_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    FALLBACK_MODEL = "microsoft/Phi-3-vision-128k-instruct"  # smaller fallback

    def __init__(
        self,
        model_id: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        json_retries: int = 3,
        max_file_size_mb: int = 10,
        max_image_pixels: int = 2_097_152,
        device: str | None = None,
    ) -> None:
        self.model_id = model_id or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.json_retries = json_retries
        self.max_file_size_mb = max_file_size_mb
        self.max_image_pixels = max_image_pixels
        self.device = device or self._auto_device()
        self._model = None
        self._processor = None
        self._ready = False

    @staticmethod
    def is_available() -> bool:
        try:
            import transformers
            return True
        except ImportError:
            return False

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def load(self) -> None:
        if self._ready:
            return

        logger.info(
            "Loading Transformers VLM: %s on %s", self.model_id, self.device
        )
        start = time.monotonic()

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "mps" else None,
            )
            if self.device == "mps":
                self._model = self._model.to("mps")
        except Exception as exc:
            logger.warning(
                "Failed to load %s: %s. Falling back to %s",
                self.model_id,
                exc,
                self.FALLBACK_MODEL,
            )
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            self.model_id = self.FALLBACK_MODEL
            self._processor = AutoProcessor.from_pretrained(self.model_id)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device if self.device != "mps" else None,
            )
            if self.device == "mps":
                self._model = self._model.to("mps")

        elapsed = time.monotonic() - start
        logger.info("Transformers VLM loaded in %.2f s", elapsed)
        self._ready = True

    def unload(self) -> None:
        if not self._ready:
            return
        self._model = None
        self._processor = None
        self._ready = False
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass
        logger.info("Transformers VLM unloaded")

    def is_loaded(self) -> bool:
        return self._ready

    def _validate_image(self, image_path: str) -> None:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(f"Image too large: {size_mb:.1f} MB > {self.max_file_size_mb} MB")
        from PIL import Image as PILImage
        img = PILImage.open(image_path)
        if img.format and img.format.lower() not in {"png", "jpeg", "jpg", "webp", "gif", "bmp"}:
            raise ValueError(f"Unsupported image format: {img.format}")
        w, h = img.size
        if w * h > self.max_image_pixels:
            raise ValueError(f"Image too large: {w}x{h} > {self.max_image_pixels} pixels")

    def analyze(
        self,
        image_path: str,
        prompt: str,
        response_format: str = "text",
        max_tokens: int = 512,
    ) -> VisionResult:
        if not self._ready:
            raise RuntimeError("Backend not loaded")
        self._validate_image(image_path)

        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")

        try:
            import torch
            if self.device != "cpu":
                inputs = inputs.to(self.device)
        except Exception:
            pass

        start = time.monotonic()
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            do_sample=True,
        )
        duration = time.monotonic() - start

        content = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]

        if response_format == "json":
            try:
                content = json.dumps(json.loads(content))
            except json.JSONDecodeError:
                pass

        return VisionResult(
            image_path=image_path,
            prompt=prompt,
            content=content,
            response_format=response_format,
            model_id=self.model_id,
            duration_seconds=round(duration, 3),
        )

    def extract(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> VisionResult:
        if not self._ready:
            raise RuntimeError("Backend not loaded")
        self._validate_image(image_path)

        from PIL import Image as PILImage
        image = PILImage.open(image_path).convert("RGB")
        full_prompt = f"{prompt}\n\nReturn only valid JSON."
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": full_prompt}],
            }
        ]

        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], return_tensors="pt")

        try:
            import torch
            if self.device != "cpu":
                inputs = inputs.to(self.device)
        except Exception:
            pass

        last_content = ""
        for attempt in range(self.json_retries):
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
            )
            content = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
            last_content = content
            try:
                parsed = json.loads(content)
                return VisionResult(
                    image_path=image_path,
                    prompt=prompt,
                    content=json.dumps(parsed),
                    response_format="json",
                    model_id=self.model_id,
                    duration_seconds=0.0,
                )
            except json.JSONDecodeError:
                logger.warning("JSON parse failed (attempt %s): %s", attempt + 1, content[:200])
                continue

        return VisionResult(
            image_path=image_path,
            prompt=prompt,
            content=last_content,
            response_format="json",
            model_id=self.model_id,
            duration_seconds=0.0,
        )
