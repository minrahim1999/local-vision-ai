"""API request/response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class TextToImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)
    negative_prompt: str | None = Field(default=None, max_length=2048)
    width: int = Field(default=512, ge=256, le=768)
    height: int = Field(default=512, ge=256, le=768)
    steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.0, ge=1.0, le=50.0)
    seed: int | None = Field(default=None)

    @field_validator("width", "height")
    @classmethod
    def _multiple_of_64(cls, v: int) -> int:
        if v % 64 != 0:
            raise ValueError("Dimensions must be a multiple of 64")
        return v


class GeneratedImage(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int
    duration_seconds: float
    model_id: str
    output_path: str


class ImageToTextRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    response_format: str = Field(default="text", pattern="^(text|json)$")
    max_tokens: int = Field(default=512, ge=1, le=2048)


class VisionResult(BaseModel):
    prompt: str
    response: str
    response_format: str
    model_id: str
    duration_seconds: float


class ExtractRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4096)
    json_schema: dict[str, Any] | None = Field(default=None)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class SystemStatus(BaseModel):
    current_pipeline: str | None = None
    rss_mb: int = 0
    vms_mb: int = 0
    system_percent: float = 0.0
    swap_used_mb: int = 0
    uptime_seconds: float = 0.0


class ModelStatus(BaseModel):
    pipelines: dict[str, bool] = Field(default_factory=dict)
    current_pipeline: str | None = None
