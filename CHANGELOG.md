# Changelog

All notable changes to Local Vision AI are documented in this file.

## [Unreleased]

### Added
- `.github/workflows/manual-release.yml` — manual trigger workflow with tag auto-creation
- `scripts/update_changelog.sh` — auto-generate changelog entries from git commits
- `docs/CHANGELOG_TEMPLATE.md` — release notes template and checklist

### Changed
- `.github/workflows/release.yml` — improved: shared metadata job, version injection, DMG creation, artifact verification
- Release workflows now checkout at tag ref for reproducible builds
- `softprops/action-gh-release` upgraded from v1 to v2
- `astral-sh/setup-uv@v5` replaces manual curl install

## [0.2.0] — 2026-07-31

### Added
- **Release build system** — standalone binaries for all platforms
  - `app/standalone.py` — auto-starts API then opens GUI in one process
  - `app/desktop.spec` — PyInstaller spec with icon bundling
  - `scripts/build_releases.sh` — local build script (macOS DMG, Windows zip, Linux tar.gz)
  - `.github/workflows/release.yml` — GitHub Actions CI for automated releases
  - Desktop app window icon (`page.window_icon`) for Windows/Linux
- **Custom app icon** — generated icon set (16×16 to 1024×1024) in `assets/icons/`
  - macOS `.icns` bundle, Windows `.ico`, and PNG set for all resolutions
  - Icon bundled into PyInstaller `.app` via `CFBundleIconFile`
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
- `README.md` completely rewritten with hero section, badges, quick-start, download table, and troubleshooting
- `config/text_to_image.yaml` now includes both `flux2` and `sd15` sections
- `config/image_to_text.yaml` now includes both `smolvlm` and `qwen` sections
- `Makefile` `audit` target now runs both environment and cross-platform checks

### Fixed
- Desktop GUI title bar now shows app name correctly on all platforms
- Standalone launcher properly shuts down API on exit (`terminate` + `kill` fallback)

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
