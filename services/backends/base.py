"""Abstract backend interfaces for cross-platform model inference.

Backends are selected at runtime based on platform and available hardware.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from api.schemas import GeneratedImage, VisionResult


class TextToImageBackend(ABC):
    """Abstract text-to-image generation backend."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Return True if this backend can run on the current platform."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load or verify model weights."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if ready for generation."""
        ...

    @abstractmethod
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
        """Generate an image from the given prompt."""
        ...


class ImageToTextBackend(ABC):
    """Abstract image-to-text analysis backend."""

    @staticmethod
    @abstractmethod
    def is_available() -> bool:
        """Return True if this backend can run on the current platform."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load or verify model weights."""
        ...

    @abstractmethod
    def unload(self) -> None:
        """Release model resources."""
        ...

    @abstractmethod
    def is_loaded(self) -> bool:
        """Return True if ready for analysis."""
        ...

    @abstractmethod
    def analyze(
        self,
        image_path: str,
        prompt: str,
        response_format: str = "text",
        max_tokens: int = 512,
    ) -> VisionResult:
        """Analyze an image and return text or JSON."""
        ...

    @abstractmethod
    def extract(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int = 512,
    ) -> VisionResult:
        """Extract structured JSON from an image."""
        ...
