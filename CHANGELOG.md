# Changelog

All notable changes to Local Vision AI are documented in this file.

## [0.2.0] — 2026-07-31

### Added
- Cross-platform backend auto-detection system
  - `AppleFlux2Backend` for macOS Apple Silicon (FLUX.2 via mflux)
  - `DiffusersBackend` for Windows/Linux/NVIDIA (SD 2.1 via Diffusers)
  - `AppleSmolVLMBackend` for macOS Apple Silicon (SmolVLM via MLX-VLM)
  - `TransformersVLMBackend` for Windows/Linux/NVIDIA (Qwen2.5-VL via Transformers)
  - Backend factory with runtime platform detection
- New documentation:
  - `docs/api-reference.md` — Full REST API documentation with curl examples
  - `docs/backends.md` — Backend architecture and extension guide
  - `scripts/check_cross_platform.sh` — Platform detection script
- Config split: `apple_silicon` extras separated from base dependencies in `pyproject.toml`

### Changed
- `README.md` updated with platform support matrix and backend comparison
- `config/text_to_image.yaml` now includes both `flux2` and `sd15` sections
- `config/image_to_text.yaml` now includes both `smolvlm` and `qwen` sections
- `Makefile` `audit` target now runs both environment and cross-platform checks

## [0.1.0] — 2026-07-31

### Added
- Initial release with Apple Silicon-first design
- Text-to-Image pipeline:
  - FLUX.2 [klein] 4B int4 via mflux (primary)
  - Stable Diffusion 1.5 via Diffusers + MPS (legacy fallback)
- Image-to-Text pipeline:
  - SmolVLM 256M int4 via MLX-VLM
- FastAPI with unified endpoints:
  - `POST /v1/images/generate`
  - `POST /v1/vision/analyze`
  - `POST /v1/vision/extract`
  - `GET /health`, `GET /v1/system/memory`, `GET /v1/models/status`
- Thread-safe `MemoryManager` with automatic model switching
- Dataset preparation scripts for both pipelines
- LoRA training placeholders:
  - Image-to-Text via Unsloth `FastVisionModel`
  - Text-to-Image via Diffusers + PEFT
- FLUX.2 LoRA training config (`config/flux2_lora.json`)
- Training feasibility report (`docs/text-to-image-training-feasibility.md`)
- Environment audit (`docs/environment-report.md`)
- Model selection document (`docs/model-selection.md`)
- Benchmark suite with real execution results (`benchmarks/report.md`)
- Test suite (15 tests covering schemas and memory manager)
- Makefile with standard dev commands

### Verified
- FLUX.2 generates 512×512 images in ~30–40s (peak ~4.8 GB)
- FLUX.2 generates 1024×1024 images in ~60–70s (peak ~12.4 GB)
- SmolVLM captions images in ~2–5s (peak ~1.3 GB)
- All 15 pytest tests pass
- Memory manager correctly unloads models before switching

---

*Format based on [Keep a Changelog](https://keepachangelog.com/)*
