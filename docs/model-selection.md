# Model Selection

Date: 2026-07-31
Updated: 2026-07-31 (post-FLUX.2 integration)

## Criteria

Models evaluated against:
- Platform support (MLX native > Diffusers/MPS > Diffusers/CUDA > CPU)
- Peak memory consumption under target RAM
- Inference speed
- Fine-tuning support (LoRA / QLoRA)
- Output quality
- License permissiveness
- Download size

## Image-to-Text (Vision-Language)

### Candidate 1: SmolVLM-256M-Instruct (Selected for Apple Silicon)
- **Size:** ~256M parameters
- **MLX-VLM support:** ✅ `mlx-community/SmolVLM-256M-Instruct-4bit`
- **Memory:** ~1.3 GB peak during inference (int4)
- **Speed:** Very fast on Apple Silicon (~3–5s per caption)
- **Capabilities:** Image captioning, VQA, basic OCR, structured JSON
- **Fine-tuning:** Unsloth `FastVisionModel` supports SmolVLM-style models. `mlx_vlm.trainer` does not exist, so Unsloth is the path.
- **License:** Apache 2.0
- **Verdict:** ✅ **Selected** for Apple Silicon — lowest memory, fastest, fits with massive headroom.

### Candidate 2: Qwen2.5-VL-3B-Instruct (Selected for CUDA/Linux/Windows)
- **Size:** ~3B parameters
- **Transformers support:** ✅ `Qwen/Qwen2.5-VL-3B-Instruct`
- **Memory:** ~4–6 GB peak during inference (fp16 on CUDA)
- **Speed:** Moderate on CUDA, slow on CPU
- **Capabilities:** Strong OCR, structured JSON output, detailed captioning
- **Fine-tuning:** Unsloth has `qwen2_vl` support. Best path for LoRA/QLoRA training on NVIDIA.
- **License:** Qwen License (research/commercial permissible with conditions)
- **Verdict:** ✅ **Selected as cross-platform fallback** — best balance of quality and portability.

### Candidate 3: Gemma-3-4B-IT
- **Size:** ~4B parameters
- **MLX/Transformers support:** Yes
- **Memory:** ~6–8 GB peak during inference
- **Capabilities:** Good VQA, decent OCR
- **Fine-tuning:** Unsloth lists `gemma3` support. 4B + training overhead pushes close to 16 GB limit on Apple Silicon.
- **License:** Gemma Terms of Use (permissive)
- **Verdict:** ❌ Not selected — too large for comfortable training on 16 GB.

### Final Choice — I2T
| Platform | Hardware | Selected Model | Backend | Quantization |
|----------|----------|---------------|---------|-------------|
| **macOS** | Apple Silicon | SmolVLM 256M int4 | MLX-VLM | int4 |
| **Windows/Linux** | NVIDIA CUDA | Qwen2.5-VL 3B | Transformers | fp16 |
| **Any** | CPU only | Phi-3 Vision | Transformers | fp32 |

---

## Text-to-Image

### Candidate 1: FLUX.2 [klein] 4B (Selected Primary for Apple Silicon)
- **Size:** ~4B parameters
- **Apple Silicon support:** ✅ `mlx-community/FLUX.2-Klein-4B-4bit` via mflux
- **Memory:** ~4.8 GB at 512×512, ~12.4 GB at 1024×1024 (int4)
- **Speed:** ~65s at 1024×1024, ~30–40s at 512×512
- **Quality:** Excellent photorealism, correct multi-subject composition, readable text
- **Fine-tuning:** `mflux-train` supports LoRA with rank/alpha/dropout config
- **License:** Apache 2.0
- **Verdict:** ✅ **Selected as primary** — best quality on Apple Silicon, fits 16 GB at 512×512.

### Candidate 2: Stable Diffusion 2.1 (Selected for CUDA/Linux/Windows)
- **Size:** ~2.6B parameters
- **Diffusers support:** ✅ `stabilityai/stable-diffusion-2-1`
- **Memory:** ~6 GB at 512×512 (fp16 on CUDA)
- **Speed:** ~10–20s per image on RTX 3060
- **Quality:** Good general-purpose generation
- **Fine-tuning:** Fully supported via Diffusers + PEFT on CUDA and MPS
- **License:** CreativeML Open RAIL-M (permissive with safety restrictions)
- **Verdict:** ✅ **Selected as cross-platform fallback** — reliable, well-supported, lower memory than FLUX.2.

### Candidate 3: Stable Diffusion 1.5 (Legacy Fallback)
- **Size:** ~1.2B parameters
- **Diffusers support:** ✅ `runwayml/stable-diffusion-v1-5`
- **Memory:** ~508 MB–4 GB at 512×512
- **Speed:** Fastest of all candidates
- **Quality:** Adequate for simple subjects; struggles with complex multi-subject prompts
- **Fine-tuning:** Fully supported via Diffusers + PEFT
- **License:** CreativeML Open RAIL-M
- **Verdict:** ⚠️ **Retained as legacy/emergency fallback** — smallest, fastest, lowest quality.

### Candidate 4: Z-Image Turbo
- **Size:** ~4B+ parameters
- **MLX support:** `mlx-community/Z-Image-Turbo` via mflux
- **Memory:** ~14 GB raw cache — **too large for 16 GB MacBook Air**
- **Verdict:** ❌ Rejected — causes severe swap pressure, generation hangs.

### Final Choice — T2I
| Platform | Hardware | Selected Model | Backend | Quantization |
|----------|----------|---------------|---------|-------------|
| **macOS** | Apple Silicon | FLUX.2 klein 4B | mflux (MLX) | int4 |
| **Windows/Linux** | NVIDIA CUDA | SD 2.1 | Diffusers | fp16 |
| **Any** | CPU only | SD 1.5 | Diffusers | fp32 |

---

## Cross-Platform Backend Architecture

```
┌─────────────────────────────────────────────────────┐
│                  FastAPI (uvicorn)                   │
├─────────────────────────────────────────────────────┤
│              services/backends/factory.py            │
│              (auto-detects platform)                 │
├─────────────────┬─────────────────┬─────────────────┤
│   Apple Silicon │   NVIDIA CUDA   │   CPU Fallback  │
│   (macOS M1–M4) │ (Windows/Linux) │ (Any platform)  │
├─────────────────┴─────────────────┴─────────────────┤
│  T2I: FLUX.2     │  T2I: SD 2.1    │  T2I: SD 1.5   │
│  (mflux + MLX)   │  (Diffusers)    │  (Diffusers)    │
│                  │                 │                 │
│  I2T: SmolVLM    │  I2T: Qwen2.5   │  I2T: Phi-3    │
│  (MLX-VLM)       │  (Transformers) │  (Transformers)│
└─────────────────────────────────────────────────────┘
```

## Summary Table

| Pipeline | Primary Model | Fallback Model | Backend | Peak Memory | Training Ready |
|----------|--------------|----------------|---------|-------------|----------------|
| **T2I (Apple)** | FLUX.2 klein 4B int4 | SD 1.5 | mflux/MLX | 4.8–12.4 GB | ✅ mflux-train LoRA |
| **T2I (CUDA)** | SD 2.1 fp16 | SD 1.5 | Diffusers | 4–6 GB | ✅ Diffusers + PEFT |
| **I2T (Apple)** | SmolVLM 256M int4 | — | MLX-VLM | ~1.3 GB | ✅ Unsloth FastVisionModel |
| **I2T (CUDA)** | Qwen2.5-VL 3B | Phi-3 Vision | Transformers | 4–6 GB | ✅ Unsloth / TRL |

## How to Replace Models

Edit `config/text_to_image.yaml` and `config/image_to_text.yaml`, then re-run `make download-models`.

### T2I Replacement Rules
- **mflux models:** Must be MLX-compatible (`mlx-community/*` prefix recommended)
- **Diffusers models:** Any `StableDiffusionPipeline` or `DiffusionPipeline` compatible checkpoint
- **Memory check:** Always verify with `scripts/benchmark.sh` after swapping

### I2T Replacement Rules
- **MLX-VLM models:** Must be `mlx-community` quantized VLMs with `<image>` token support
- **Transformers models:** Must be vision-language models with `AutoModelForVision2Seq` support
- **JSON extraction:** Models with strong instruction-following work best for `/v1/vision/extract`

---

*Last updated: 2026-07-31*
