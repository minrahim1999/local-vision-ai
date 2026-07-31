"""Image-to-Text backend: SmolVLM via MLX-VLM on Apple Silicon."""
from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path

from api.schemas import VisionResult
from services.backends.base import ImageToTextBackend

logger = logging.getLogger(__name__)


class AppleSmolVLMBackend(ImageToTextBackend):
    """SmolVLM 256M int4 via mlx_vlm.

    Requirements:
      - macOS
      - Apple Silicon (M1/M2/M3/M4+)
      - mlx_vlm package installed
    """

    def __init__(
        self,
        model_id: str = "mlx-community/SmolVLM-256M-Instruct-4bit",
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
        self._model = None
        self._processor = None
        self._ready = False

    @staticmethod
    def is_available() -> bool:
        try:
            import platform as plat
            if plat.system() != "Darwin":
                return False
            import subprocess as sp
            result = sp.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "Apple" in result.stdout
        except Exception:
            return False

    def load(self) -> None:
        if self._ready:
            return
        try:
            import mlx_vlm
            self._model, self._processor = mlx_vlm.load(self.model_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to load SmolVLM: {exc}") from exc
        self._ready = True
        logger.info("AppleSmolVLMBackend loaded: %s", self.model_id)

    def unload(self) -> None:
        if not self._ready:
            return
        self._model = None
        self._processor = None
        self._ready = False
        gc.collect()
        logger.info("AppleSmolVLMBackend unloaded")

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

        import mlx_vlm
        full_prompt = f"<image> {prompt}"
        start = time.monotonic()
        output = mlx_vlm.generate(
            self._model,
            self._processor,
            prompt=full_prompt,
            image=image_path,
            max_tokens=max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            verbose=False,
        )
        duration = time.monotonic() - start

        content = str(output) if not isinstance(output, str) else output

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

        import mlx_vlm
        full_prompt = f"{prompt}\n\nReturn only valid JSON."
        last_content = ""

        for attempt in range(self.json_retries):
            output = mlx_vlm.generate(
                self._model,
                self._processor,
                prompt=full_prompt,
                image=image_path,
                max_tokens=max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                verbose=False,
            )
            content = str(output) if not isinstance(output, str) else output
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
