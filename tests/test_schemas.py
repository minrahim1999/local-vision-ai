"""Tests for API request/response schemas."""
import pytest
from pydantic import ValidationError

from api.schemas import (
    ExtractRequest,
    GeneratedImage,
    ImageToTextRequest,
    TextToImageRequest,
)


class TestTextToImageRequest:
    def test_valid_request(self) -> None:
        req = TextToImageRequest(prompt="a cat on a sofa", width=512, height=512)
        assert req.prompt == "a cat on a sofa"
        assert req.width == 512
        assert req.height == 512

    def test_missing_prompt(self) -> None:
        with pytest.raises(ValidationError):
            TextToImageRequest(width=512, height=512)

    def test_width_not_multiple_of_64(self) -> None:
        with pytest.raises(ValidationError):
            TextToImageRequest(prompt="a cat", width=500, height=512)

    def test_height_not_multiple_of_64(self) -> None:
        with pytest.raises(ValidationError):
            TextToImageRequest(prompt="a cat", width=512, height=500)

    def test_seed_optional(self) -> None:
        req = TextToImageRequest(prompt="a cat", seed=None)
        assert req.seed is None


class TestImageToTextRequest:
    def test_valid(self) -> None:
        req = ImageToTextRequest(prompt="describe this")
        assert req.response_format == "text"

    def test_invalid_format(self) -> None:
        with pytest.raises(ValidationError):
            ImageToTextRequest(prompt="describe", response_format="xml")

    def test_max_tokens_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ImageToTextRequest(prompt="x", max_tokens=0)
        with pytest.raises(ValidationError):
            ImageToTextRequest(prompt="x", max_tokens=3000)


class TestExtractRequest:
    def test_valid(self) -> None:
        req = ExtractRequest(prompt="extract json")
        assert req.json_schema is None


class TestGeneratedImage:
    def test_fields(self) -> None:
        img = GeneratedImage(
            prompt="a cat",
            width=512,
            height=512,
            steps=20,
            guidance_scale=7.0,
            seed=42,
            duration_seconds=1.2,
            model_id="test-model",
            output_path="/tmp/out.png",
        )
        assert img.seed == 42
