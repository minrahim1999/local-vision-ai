# Environment Report

Generated: 2026-07-31

## Hardware Summary

| Property | Value |
|----------|-------|
| Device | MacBook Air |
| Chip | Apple M3 |
| Architecture | arm64 (Apple Silicon) |
| CPU Cores | 8 (4 Performance + 4 Efficiency) |
| Unified Memory | 16 GB |
| Model Identifier | Mac15,12 |

## Operating System

| Property | Value |
|----------|-------|
| Name | macOS |
| Version | 26.5.2 (Tahoe) |
| Build | 25F84 |

## Python Versions

| Interpreter | Version | Notes |
|-------------|---------|-------|
| System `python3` | 3.9.6 | Bundled with macOS; outdated for modern ML |
| `python3.10` (pyenv) | 3.10.17 | Has PyTorch 2.13.0 + MPS |
| Unsloth Studio | 3.13.13 | Separate venv in `~/.unsloth/studio` |

## Package Manager

- **uv** 0.11.29 (aarch64-apple-darwin) — installed and preferred.

## Unsloth

- **CLI version:** 2026.7.6 (2026.7.7 in studio venv)
- **Commands:** `train`, `inference`, `chat`, `export`, `list-checkpoints`, `run`, `studio`, `start`
- **Supported model families in installed version:**
  - `gemma`, `gemma2`, `gemma3`, `gemma4`
  - `qwen2`, `qwen3`, `qwen3_moe`
  - `llama`, `llama4`
  - `mistral`
  - `vision` (vision-language wrapper)
  - `diffusion` (text-to-image)
- **Studio venv location:** `~/.unsloth/studio/unsloth_studio/bin/python`
- **PyTorch in studio:** 2.10.0 (MPS built & available)
- **Transformers in studio:** 4.57.6
- **Diffusers in studio:** 0.39.0
- **Accelerate in studio:** 1.14.0
- **PEFT in studio:** 0.18.1

## MLX (Apple Silicon Native)

| Package | Version |
|---------|---------|
| `mlx` | 0.32.0 |
| `mlx-metal` | 0.32.0 |
| `mlx-lm` | 0.31.2 |
| `mlx-vlm` | 0.4.4 |

**Available `mlx-vlm` model families** (relevant to this project):
- `gemma3`, `gemma3n`, `gemma4`
- `qwen2_vl`, `qwen2_5_vl`, `qwen3_vl`, `qwen3_5`
- `smolvlm`
- `paligemma`
- `moondream3`
- `llava`, `llava_next`
- `phi4_siglip`
- `mistral3`

## PyTorch MPS

- **MPS built:** Yes
- **MPS available:** Yes
- **Notes:** PyTorch MPS is functional but has narrower operator coverage than MLX. Some operations silently fall back to CPU or raise `NotImplementedError`.

## CUDA Status

- **CUDA available:** No
- **NVIDIA libraries:** Not present
- **Impact:** Any package requiring CUDA (e.g., CUDA Triton, CUDA-only `bitsandbytes`) will fail or must be avoided.

## Disk Space

| Volume | Size | Used | Available |
|--------|------|------|-----------|
| `/System/Volumes/Data` | 460 GB | 333 GB | ~92 GB |

**Recommendation:** Reserve at least 20–30 GB for model weights, datasets, and generated outputs.

## Recommended Memory Limits (16 GB Unified)

| Scenario | Safe Peak | Caution Zone |
|----------|-----------|--------------|
| Inference (single model) | ≤ 10 GB | 10–13 GB |
| Training (QLoRA, rank 8) | ≤ 12 GB | 12–14 GB |
| Two models loaded simultaneously | **Avoid** | — |

**Guidelines:**
- Keep a safety margin of ~3–4 GB for macOS and other processes.
- Batch size should remain **1** unless proven safe.
- Use gradient accumulation to simulate larger batches.
- Start image generation at **512×512**; 768×768 may push into swap.
- Use 4-bit quantization where supported.

## Known Compatibility Risks

1. **Python 3.9 (system):** Too old for modern `transformers`/`diffusers`/`mlx`. The project should use a dedicated uv virtual environment with Python ≥ 3.10.
2. **Unsloth Studio venv isolation:** The studio environment is fully isolated. We should not attempt to modify it. Instead, create a fresh uv-managed virtual environment for `local-vision-ai`.
3. **MPS operator gaps:** Certain diffusion schedulers and attention implementations may not be fully supported on MPS. MLX-native diffusion is preferred when available.
4. **MLX diffusion maturity:** As of MLX 0.32, MLX-native Stable Diffusion exists but may lag behind Diffusers in model coverage. Diffusers + MPS is a reliable fallback for SD 1.5/2.1.
5. **Unsloth text-to-image training:** The installed Unsloth version includes a `diffusion` module, but its Apple Silicon support is less battle-tested than language-model training. We will treat T2I LoRA training as *feasibility-test only* in Milestone 1.
6. **Unified memory swap:** Sustained workloads near 14 GB will force heavy swap usage and degrade performance. We must explicitly unload models and call `gc.collect()` / `torch.mps.empty_cache()` between pipeline switches.

## Selected Model Candidates

### Image-to-Text (Vision-Language)

| Candidate | Size | Quantization | Why / Why Not |
|-----------|------|--------------|---------------|
| **SmolVLM-256M-Instruct** (mlx-vlm) | ~256M–500M | 4-bit built-in | Extremely small, fast, good for captioning/VQA. Fits comfortably in < 2 GB. |
| **Qwen2.5-VL-3B-Instruct** (mlx-vlm) | ~3B | 4-bit built-in | Strong OCR, JSON structured output, Unsloth supports Qwen-VL fine-tuning. Good balance of capability and memory. |
| **Gemma-3-4B-IT** (mlx-vlm) | ~4B | 4-bit built-in | Good quality, but 4B is near the upper comfort limit for 16 GB when training. |

**First-choice:** `SmolVLM-256M-Instruct` for baseline inference (lowest risk, fastest). `Qwen2.5-VL-3B-Instruct` as the primary fine-tuning target if Unsloth supports it in the installed version.

### Text-to-Image

| Candidate | Size | Why / Why Not |
|-----------|------|---------------|
| **Stable Diffusion 1.5** (Diffusers + MPS) | ~1.2B (fp32) / ~4 GB (fp16) | Proven on Apple Silicon, low memory, good LoRA ecosystem. |
| **Stable Diffusion XL** (Diffusers + MPS) | ~3.5B / ~7 GB (fp16) | Better quality, but pushes memory limits. Optional for later. |
| **MLX Stable Diffusion** (native MLX) | ~1.2B | Fastest on Apple Silicon, but fewer pretrained variants and LoRA tools. |

**First-choice:** Stable Diffusion 1.5 via Diffusers with MPS, because it is the most stable, widely supported, and has the best LoRA tooling on Apple Silicon.

## Next Steps

1. Create a dedicated uv virtual environment with Python 3.10/3.11.
2. Install `mlx`, `mlx-vlm`, `diffusers`, `transformers`, `accelerate`, `fastapi`, `uvicorn`, `pillow`, `psutil`, `pyyaml`, `pydantic`.
3. Download chosen models during setup, not at runtime.
4. Verify single-image inference before any training attempt.
