# Text-to-Image LoRA Training Feasibility Report

Date: 2026-07-31

## Selected Model

- **Model:** `runwayml/stable-diffusion-v1-5`
- **Size:** ~1.2B parameters
- **Backend:** Diffusers + PyTorch MPS (Apple Silicon)

## Training Framework

- **Primary framework:** Diffusers `train_text_to_image_lora.py` (or `peft` + custom loop)
- **Secondary option:** `diffusers` `StableDiffusionLoraLoaderMixin` with `peft.LoraConfig`

## Apple Silicon Support Assessment

### What Works
- Diffusers supports PyTorch MPS backend.
- `torch.backends.mps.is_available()` returns `True` on this machine.
- SD 1.5 UNet + VAE fit in ~4–6 GB at fp16.
- `peft` LoRA injection into UNet attention blocks works without CUDA-specific code.

### Risks / Unsupported CUDA Dependencies
- **xFormers:** CUDA-only; not available on MPS. We will disable xFormers and rely on attention slicing / SDPA.
- **bitsandbytes 8-bit optimizers:** CUDA-only. Use standard AdamW (fp32) or `torch.optim.AdamW` with fp16.
- **torch.compile:** Not supported on MPS. Leave disabled.
- **Some VAE operations:** A few VAE convolutions may silently fall back to CPU on MPS, causing slowdowns but not crashes.
- **Dataloader multiprocessing:** `num_workers > 0` can be unstable with MPS tensors in subprocesses. Keep `num_workers=0`.

## Estimated Peak Memory

| Component | Memory (fp16) |
|-----------|--------------|
| UNet (~860M params) | ~1.7 GB |
| VAE (~83M params) | ~170 MB |
| Text encoder (~123M params) | ~250 MB |
| LoRA weights (rank 8, ~30% of UNet) | ~50 MB |
| Optimizer states (AdamW, fp32) | ~3.4 GB |
| Activations + gradients (512×512, BS=1, grad checkpointing) | ~2–3 GB |
| **Total estimated peak** | **~8–10 GB** |

This is within the 12 GB safe zone for 16 GB unified memory, but leaves little margin.

## Estimated Storage Usage

- Base model weights (cached): ~4 GB
- LoRA adapter output (per checkpoint): ~50–150 MB
- Dataset images: depends on dataset size
- Training logs / tensorboard: ~100 MB

## Safe Resolution and Batch Size

| Resolution | Batch Size | Status |
|------------|-----------|--------|
| 256×256 | 1 | Very safe |
| 512×512 | 1 | Safe, recommended start |
| 512×512 | 2 | Risk of OOM without CPU offload |
| 768×768 | 1 | Marginal; may push into swap |

**Recommendation:** Start at **512×512, batch size 1**, with gradient accumulation = 8.

## Fine-Tuning Verdict

- **Local LoRA training is feasible** on this MacBook Air M3 with 16 GB, but it is memory-constrained.
- Expected step time: ~5–20 seconds/step (MPS is slower than CUDA for diffusion training).
- A 1,000-step run would take ~2–6 hours.
- Swapping risk exists if any other large app is running.

## Cloud GPU Comparison

| Factor | MacBook Air M3 (16 GB) | Rent NVIDIA A10G / L4 (24 GB) |
|--------|------------------------|-------------------------------|
| Step time (512×512) | ~5–20 s | ~1–3 s |
| Batch size flexibility | 1 only | 2–4 |
| Memory safety margin | Tight | Comfortable |
| Cost | $0 (existing hardware) | ~$0.50–$1.50/hour |
| Setup complexity | Low (local env) | Medium (cloud setup) |

**Recommendation:** For serious LoRA training (e.g., > 2,000 steps on a custom dataset), a rented NVIDIA GPU is more practical. Use local training only for **small experiments, smoke tests, and adapter prototyping**.

## Required Smoke Test Before Full Run

Before committing to a full training run:

1. Verify `diffusers` + `peft` can inject LoRA into SD 1.5 UNet on MPS.
2. Run **one training step** at 512×512, batch size 1.
3. Confirm loss is finite and not NaN.
4. Save a checkpoint and reload it.
5. Run one inference with the loaded adapter.
6. Monitor memory with Activity Monitor or `psutil`.

Only proceed if peak memory stays below ~12 GB and swap usage is minimal.

## Next Command (if smoke test passes)

```bash
uv run python -m training.text_to_image.train_lora --config config/text_to_image.yaml --smoke-test
```

## Conclusion

- **Local training:** Feasible for small-scale LoRA experiments.
- **Production training:** Consider cloud GPU for speed and safety.
- **Do not run a full training job until the smoke test above succeeds.**
