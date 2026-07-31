"""FastAPI application for Local Vision AI — cross-platform backend support."""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from api.schemas import (
    ExtractRequest,
    GeneratedImage,
    ImageToTextRequest,
    ModelStatus,
    SystemStatus,
    TextToImageRequest,
    VisionResult,
)
from services.backends.factory import create_i2t_backend, create_t2i_backend
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
    t2i_cfg = _load_yaml(CONFIG_DIR / "text_to_image.yaml").get("t2i", {})
    i2t_cfg = _load_yaml(CONFIG_DIR / "image_to_text.yaml").get("i2t", {})

    mm = MemoryManager(
        memory_warn_mb=int(os.getenv("MEMORY_WARN_MB", "12000")),
        memory_limit_mb=int(os.getenv("MEMORY_LIMIT_MB", "14000")),
        idle_timeout_seconds=float(os.getenv("IDLE_TIMEOUT_SECONDS", "300")),
    )

    hf_home = os.getenv("HF_HOME", str(Path("models/huggingface").resolve()))

    # Use factory for auto platform detection
    t2i = create_t2i_backend({**t2i_cfg.get("flux2", {}), **t2i_cfg.get("sd15", {}), "hf_home": hf_home})
    i2t = create_i2t_backend({**i2t_cfg.get("smolvlm", {}), **i2t_cfg.get("qwen", {}), "hf_home": hf_home})

    mm.register("text_to_image", t2i)
    mm.register("image_to_text", i2t)

    app.state.memory_manager = mm
    app.state.text_to_image = t2i
    app.state.image_to_text = i2t

    logger.info("Services registered. T2I: %s, I2T: %s", type(t2i).__name__, type(i2t).__name__)
    yield
    logger.info("Shutting down. Releasing all models.")
    mm.release_all()


app = FastAPI(
    title="Local Vision AI",
    description="Local multimodal AI API for Apple Silicon, Windows + CUDA, and Linux + CUDA",
    version="0.2.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health(request: Request) -> dict:
    mm: MemoryManager = request.app.state.memory_manager
    return {"status": "ok", "pipeline": mm.current_pipeline}


@app.get("/v1/system/memory")
async def system_memory(request: Request) -> SystemStatus:
    mm: MemoryManager = request.app.state.memory_manager
    data = mm.get_memory_status()
    return SystemStatus(
        current_pipeline=data.get("current_pipeline"),
        rss_mb=data.get("rss_mb", 0),
        vms_mb=data.get("vms_mb", 0),
        system_percent=data.get("system_percent", 0.0),
        swap_used_mb=data.get("swap_used_mb", 0),
        uptime_seconds=time.monotonic(),
    )


@app.get("/v1/models/status")
async def models_status(request: Request) -> ModelStatus:
    mm: MemoryManager = request.app.state.memory_manager
    data = mm.get_memory_status()
    t2i = request.app.state.text_to_image
    i2t = request.app.state.image_to_text
    return ModelStatus(
        pipelines={
            "text_to_image": t2i.is_loaded(),
            "image_to_text": i2t.is_loaded(),
        },
        current_pipeline=data.get("current_pipeline"),
    )


@app.post("/v1/images/generate", response_model=GeneratedImage)
async def images_generate(request: Request, body: TextToImageRequest) -> GeneratedImage:
    mm: MemoryManager = request.app.state.memory_manager
    mm.acquire("text_to_image")
    svc = request.app.state.text_to_image
    try:
        result = svc.generate(
            prompt=body.prompt,
            negative_prompt=body.negative_prompt,
            width=body.width,
            height=body.height,
            steps=body.steps,
            guidance_scale=body.guidance_scale,
            seed=body.seed,
        )
        return result
    except MemoryError as exc:
        logger.error("Memory error during T2I generation: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.error("Runtime error during T2I generation: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during T2I generation")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/vision/analyze", response_model=VisionResult)
async def vision_analyze(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    response_format: str = Form(default="text"),
    max_tokens: int = Form(default=512),
) -> VisionResult:
    if response_format not in {"text", "json"}:
        raise HTTPException(status_code=422, detail="response_format must be 'text' or 'json'")

    upload_path = Path("outputs/temp_uploads")
    upload_path.mkdir(parents=True, exist_ok=True)
    local_file = upload_path / f"{int(time.time())}_{image.filename}"
    try:
        with open(local_file, "wb") as f:
            content = await image.read()
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    mm: MemoryManager = request.app.state.memory_manager
    mm.acquire("image_to_text")
    svc = request.app.state.image_to_text
    try:
        result = svc.analyze(
            image_path=str(local_file),
            prompt=prompt,
            response_format=response_format,
            max_tokens=max_tokens,
        )
        return result
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during I2T analysis")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            local_file.unlink(missing_ok=True)
        except Exception:
            pass


@app.post("/v1/vision/extract", response_model=VisionResult)
async def vision_extract(
    request: Request,
    image: UploadFile = File(...),
    prompt: str = Form(...),
    max_tokens: int = Form(default=512),
) -> VisionResult:
    upload_path = Path("outputs/temp_uploads")
    upload_path.mkdir(parents=True, exist_ok=True)
    local_file = upload_path / f"{int(time.time())}_{image.filename}"
    try:
        with open(local_file, "wb") as f:
            content = await image.read()
            f.write(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save upload: {exc}") from exc

    mm: MemoryManager = request.app.state.memory_manager
    mm.acquire("image_to_text")
    svc = request.app.state.image_to_text
    try:
        result = svc.extract(
            image_path=str(local_file),
            prompt=prompt,
            max_tokens=max_tokens,
        )
        return result
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during I2T extraction")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        try:
            local_file.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
