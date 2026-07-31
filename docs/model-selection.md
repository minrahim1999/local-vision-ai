# Model Selection

Date: 2026-07-31

## Criteria

Models were evaluated against:
- Apple Silicon native support (MLX > MPS > CPU)
- Peak memory consumption under 16 GB unified memory
- Inference speed
- Fine-tuning support (LoRA / QLoRA)
- Output quality for the target use cases
- License permissiveness
- Download size

## Image-to-Text (Vision-Language)

### Candidate 1: SmolVLM-256M-Instruct
- **Size:** ~256M parameters (8-bit quantized by mlx-community)
- **MLX-VLM support:** Yes — `mlx-community/smolvlm-256m-8bit`
- **Memory:** ~2 GB peak during inference
- **Speed:** Very fast on Apple Silicon
- **Capabilities:** Image captioning, VQA, basic OCR
- **Fine-tuning:** Unsloth does not officially list SmolVLM for fine-tuning yet. QLoRA via `trl` + `peft` on MLX is possible but less documented.
- **License:** Apache 2.0
- **Verdict:** Best for **baseline inference** — lowest risk, fastest, fits with room to spare.

### Candidate 2: Qwen2.5-VL-3B-Instruct
- **Size:** ~3B parameters
- **MLX-VLM support:** Yes — `mlx-community/Qwen2.5-VL-3B-Instruct-8bit`
- **Memory:** ~4–6 GB peak during inference
- **Speed:** Moderate (slower than SmolVLM but acceptable)
- **Capabilities:** Strong OCR, structured JSON output, detailed captioning
- **Fine-tuning:** Unsloth has `qwen2_vl` / `qwen3_vl` support in the installed version. This is the best path for **LoRA/QLoRA training**.
- **License:** Qwen License (research / commercial permissible with conditions)
- **Verdict:** Best for **fine-tuning** if Unsloth supports it. Higher memory footprint but still safe on 16 GB.

### Candidate 3: Gemma-3-4B-IT
- **Size:** ~4B parameters
- **MLX-VLM support:** Yes — `mlx-community/gemma-3-4b-it-8bit`
- **Memory:** ~6–8 GB peak during inference
- **Speed:** Moderate
- **Capabilities:** Good VQA, decent OCR
- **Fine-tuning:** Unsloth lists `gemma3` / `gemma4` support. However, 4B + training overhead pushes close to the 16 GB limit.
- **License:** Gemma Terms of Use (permissive for research and commercial)
- **Verdict:** Too large for comfortable training on this machine. Risk of swap pressure.

### Final Choice
- **Baseline inference:** `SmolVLM-256M-Instruct` (mlx-community 8-bit)
- **Fine-tuning target:** `Qwen2.5-VL-3B-Instruct` (mlx-community 8-bit or 4-bit) using Unsloth if compatible, otherwise MLX-VLM native fine-tuning via `mlx-vlm.trainer`.

## Text-to-Image

### Candidate 1: Stable Diffusion 1.5 (Diffusers + MPS)
- **Size:** ~1.2B parameters (~4 GB in fp16)
- **Apple Silicon support:** Excellent via Diffusers + PyTorch MPS
- **Memory:** ~4–6 GB peak at 512×512 with attention slicing
- **Speed:** ~5–15 steps/sec depending on scheduler
- **Quality:** Good for general subjects; well understood tradeoffs
- **LoRA training:** Fully supported via `diffusers` + `peft` on MPS. No CUDA dependencies needed.
- **License:** CreativeML Open RAIL-M (permissive with safety restrictions)
- **Verdict:** Most reliable choice for 16 GB. Best ecosystem support.

### Candidate 2: Stable Diffusion XL (Diffusers + MPS)
- **Size:** ~3.5B parameters (~7 GB in fp16)
- **Memory:** ~8–10 GB peak at 512×512; may push into swap at 768×768
- **Speed:** Slower than SD 1.5
- **Quality:** Better photorealism and text rendering
- **Verdict:** Marginal on 16 GB unified memory. Risk of severe memory pressure during inference or training.

### Candidate 3: MLX Stable Diffusion (native MLX)
- **Size:** Same SD 1.2B backbone
- **MLX diffusion support:** As of MLX 0.32, MLX has `mlx_lm` but diffusion support is more limited than Diffusers.
- **Memory:** Potentially lower due to native MLX memory management
- **Verdict:** Less mature ecosystem for pretrained LoRA adapters and community models. Not chosen for first milestone.

### Final Choice
- **Baseline inference:** `runwayml/stable-diffusion-v1-5` via Diffusers with MPS, fp16, attention slicing, and VAE slicing.
- **LoRA feasibility:** Will be tested in Phase 8 using Diffusers + PEFT on MPS.

## Summary Table

| Pipeline | Selected Model | Backend | Quantization | Peak Memory (est.) | Training Ready |
|----------|----------------|---------|--------------|-------------------|----------------|
| Image-to-Text | SmolVLM-256M-Instruct | MLX-VLM | 8-bit | ~2 GB | TBD (smoke test) |
| Image-to-Text (FT) | Qwen2.5-VL-3B-Instruct | Unsloth / MLX | 8-bit | ~5–7 GB | Yes (if Unsloth supports) |
| Text-to-Image | SD 1.5 | Diffusers + MPS | fp16 | ~4–6 GB | Yes (Diffusers LoRA) |
