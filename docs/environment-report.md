# Environment Report

Generated during Phase 0 of the Local Vision AI project.

## Hardware Summary

| Property | Value |
|----------|-------|
| Device | MacBook Air |
| Chip | Apple M3 |
| Architecture | arm64 |
| CPU Cores | 8 (4 performance + 4 efficiency) |
| GPU Cores | 10 |
| Neural Engine | 16-core |
| Unified Memory | 16 GB LPDDR5 |
| Memory Bandwidth | 100 GB/s |

## Operating System

| Property | Value |
|----------|-------|
| OS | macOS |
| Version | 26.5.2 (Tahoe) |
| Kernel | Darwin 26.5.2 |

## Python Environment

| Property | Value |
|----------|-------|
| Python Version | 3.11.12 |
| Package Manager | uv |
| Virtual Environment | `.venv` (in project root) |

### Core Packages

| Package | Version | Platform |
|---------|---------|----------|
| PyTorch | 2.13.0 | All (CUDA/MPS/CPU) |
| MLX | 0.32.0 | Apple Silicon only |
| MLX-VLM | 0.4.4 | Apple Silicon only |
| mflux | latest | Apple Silicon only |
| Diffusers | 0.39.0 | All |
| Transformers | 4.57.6 | All |
| FastAPI | 0.111.0 | All |
| Uvicorn | 0.30.0 | All |
| Pydantic | 2.7.0 | All |
| Pillow | 10.0.0 | All |
| psutil | 6.0.0 | All |

## Acceleration Availability

| Accelerator | Status | Notes |
|-------------|--------|-------|
| **MLX** | ✅ Available | Apple Silicon exclusive |
| **MPS** | ✅ Available | macOS Metal Performance Shaders |
| **CUDA** | ❌ Not Available | No NVIDIA hardware |
| **CPU** | ✅ Available | Fallback for all operations |

## Unsloth

| Property | Value |
|----------|-------|
| Version | 2026.7.7 |
| Python | 3.13 (Studio), 3.11 (Project venv) |
| Vision Support | ✅ FastVisionModel available |

## Hermes Agent

| Property | Value |
|----------|-------|
| Status | Installed and configured |
| Profile | default |

## Available Disk Space

| Location | Usage |
|----------|-------|
| Project total | ~33 GB (includes models) |
| `.venv` | ~1.5 GB |
| FLUX.2 weights | ~4.3 GB |
| HuggingFace cache | ~6 GB |

## Recommended Memory Limits

For reliable operation on 16 GB unified memory:

| Pipeline | Max Resolution | Peak Memory | Safe? |
|----------|---------------|-------------|-------|
| FLUX.2 T2I | 512×512 | ~4.8 GB | ✅ Yes |
| FLUX.2 T2I | 1024×1024 | ~12.4 GB | ⚠️ Tight |
| SmolVLM I2T | Any supported | ~1.3 GB | ✅ Yes |
| SD 1.5 T2I | 512×512 | ~508 MB | ✅ Yes |
| Both loaded | — | ~14 GB | ❌ No |

## Known Compatibility Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| FLUX.2 1024×1024 + swap pressure | High | Use 512×512 default, unload I2T first |
| MLX first import slow | Medium | Pre-warm in setup, allow 60s timeout |
| mflux subprocess timeout | Medium | Use 600s timeout, monitor swap |
| VAE fp16 NaN on MPS | High | Force float32 for VAE (SD 1.5 only) |
| Unsloth import timeout | Medium | Extend timeout or use Studio Python |

## Cross-Platform Notes

This environment report reflects the **primary development machine** (Apple Silicon). The codebase auto-detects platform and selects appropriate backends:

- **Apple Silicon**: FLUX.2 (MLX) + SmolVLM (MLX-VLM)
- **Windows/Linux + NVIDIA**: SD 2.1 (Diffusers/CUDA) + Qwen2.5-VL (Transformers/CUDA)
- **CPU-only**: SD 1.5 (Diffusers/CPU) + Phi-3 Vision (Transformers/CPU)

See `docs/backends.md` for per-platform requirements.

## Selected Model Candidates (16 GB Constraints)

### Text-to-Image

| Model | Parameters | Quantization | Size | Platform | Status |
|-------|-----------|--------------|------|----------|--------|
| **FLUX.2 [klein] 4B** | 4B | int4 | ~6 GB | Apple MLX | ✅ Selected |
| Stable Diffusion 1.5 | 1.2B | fp16 | ~5 GB | All | ✅ Fallback |
| Stable Diffusion 2.1 | 2.6B | fp16 | ~6 GB | All | ✅ CUDA fallback |
| Z-Image Turbo | 4B+ | int8 | ~14 GB | Apple MLX | ❌ Too large |

### Image-to-Text

| Model | Parameters | Quantization | Size | Platform | Status |
|-------|-----------|--------------|------|----------|--------|
| **SmolVLM 256M** | 256M | int4 | ~200 MB | Apple MLX | ✅ Selected |
| Qwen2.5-VL 3B | 3B | int4/int8 | ~2 GB | All | ✅ CUDA fallback |
| PaliGemma 3B | 3B | int4 | ~2 GB | All | ⚠️ Not tested |
| Gemma-3 4B IT | 4B | int4 | ~2.5 GB | All | ⚠️ Not tested |

## Verification Commands

```bash
# Re-run this audit
make audit

# Check platform detection
bash scripts/check_cross_platform.sh

# Verify backends
python -c "from services.backends.factory import create_t2i_backend, create_i2t_backend; \
           t2i = create_t2i_backend({}); i2t = create_i2t_backend({}); \
           print('T2I:', type(t2i).__name__); print('I2T:', type(i2t).__name__)"
```

---

*Report generated: 2026-07-31*
*Machine: MacBook Air M3, 16 GB*
