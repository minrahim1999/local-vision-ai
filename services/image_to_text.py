"""Image-to-Text inference service using MLX-VLM."""
from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_vlm import generate, load
from PIL import Image

from api.schemas import VisionResult

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "webp", "gif", "bmp"}


class ImageToTextService:
    def __init__(
        self,
        model_id: str = "mlx-community/smolvlm-256m-8bit",
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        json_retries: int = 3,
        max_file_size_mb: int = 10,
        max_image_pixels: int = 2_097_152,
    ) -> None:
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.json_retries = json_retries
        self.max_file_size_mb = max_file_size_mb
        self.max_image_pixels = max_image_pixels
        self._model: Any = None
        self._processor: Any = None
        self._is_loaded = False

    def is_loaded(self) -> bool:
        return self._is_loaded

    def load(self) -> None:
        if self._is_loaded:
            logger.info("I2T model already loaded")
            return
        logger.info("Loading I2T model: %s", self.model_id)
        start = time.monotonic()
        self._model, self._processor = load(self.model_id)
        elapsed = time.monotonic() - start
        logger.info("I2T model loaded in %.2f s", elapsed)
        self._is_loaded = True

    def unload(self) -> None:
        if not self._is_loaded:
            return
        logger.info("Unloading I2T model")
        del self._model
        del self._processor
        self._model = None
        self._processor = None
        self._is_loaded = False
        gc.collect()
        logger.info("I2T model unloaded")

    def _validate_image(self, image_path: str) -> Image.Image:
        path = Path(image_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        ext = path.suffix.lstrip(".").lower()
        if ext not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported image format: {ext}")

        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > self.max_file_size_mb:
            raise ValueError(
                f"File too large: {size_mb:.1f} MB > {self.max_file_size_mb} MB"
            )

        try:
            img = Image.open(path)
        except Exception as exc:
            raise ValueError(f"Cannot open image: {exc}") from exc

        if img.format is None or img.format.lower() not in {
            "PNG",
            "JPEG",
            "JPG",
            "WEBP",
            "GIF",
            "BMP",
        }:
            raise ValueError(f"Unsupported or unknown image format: {img.format}")

        w, h = img.size
        pixels = w * h
        if pixels > self.max_image_pixels:
            raise ValueError(
                f"Image dimensions too large: {w}x{h} ({pixels} pixels) > {self.max_image_pixels}"
            )

        return img

    def analyze(
        self,
        image_path: str,
        prompt: str,
        response_format: str = "text",
        max_tokens: int = 512,
    ) -> VisionResult:
        if not self._is_loaded or self._model is None or self._processor is None:
            raise RuntimeError("ImageToTextService is not loaded. Call load() first.")

        img = self._validate_image(image_path)

        start = time.monotonic()
        try:
            output = generate(
                self._model,
                self._processor,
                image=image_path,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=self.temperature,
                top_p=self.top_p,
                verbose=False,
            )
        except Exception as exc:
            logger.exception("I2T generation failed")
            raise RuntimeError(f"Image-to-text generation failed: {exc}") from exc

        duration = time.monotonic() - start
        response_text = output.strip()

        if response_format == "json":
            response_text = self._extract_json(response_text)

        return VisionResult(
            prompt=prompt,
            response=response_text,
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
        structured_prompt = (
            f"{prompt}\n\n"
            "Respond ONLY with valid JSON. Do not include markdown formatting or extra text."
        )
        result = self.analyze(
            image_path=image_path,
            prompt=structured_prompt,
            response_format="json",
            max_tokens=max_tokens,
        )
        return result

    def _extract_json(self, text: str) -> str:
        """Attempt to extract and validate JSON with limited retries."""
        for attempt in range(1, self.json_retries + 1):
            candidate = text
            # Strip markdown fences if present
            candidate = candidate.strip()
            if candidate.startswith("```"):
                lines = candidate.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                candidate = "\n".join(lines).strip()

            try:
                parsed = json.loads(candidate)
                return json.dumps(parsed, ensure_ascii=False, indent=2)
            except json.JSONDecodeError:
                logger.warning("JSON parse attempt %s/%s failed", attempt, self.json_retries)
                # If there is a substring that looks like JSON, try that next
                start_idx = text.find("{")
                end_idx = text.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    text = text[start_idx : end_idx + 1]
                else:
                    break

        # If all retries fail, return the raw text but log it
        logger.error("Failed to extract valid JSON after %s attempts", self.json_retries)
        return text
