# Benchmark Report

**Date:** 2026-07-31
**Device:** MacBook Air M3 (8-core, 16 GB unified memory)
**OS:** macOS 26.5.2 (Tahoe)
**Backend:** Apple Silicon (MPS for T2I, MLX for I2T)

## Results Summary

| Pipeline | Model | Load Time | First Inference | Peak RSS | Output |
|----------|-------|-----------|-----------------|----------|--------|
| Text-to-Image | SD 1.5 (fp16) | ~8 s | ~19 s @ 512×512, 20 steps | ~508 MB | 512×512 PNG |
| Image-to-Text | SmolVLM 256M (4-bit) | ~0.6 s | ~3.4 s | ~1335 MB | ~430 tokens |

*Note: Process RSS readings understate actual unified memory usage because MPS and MLX allocate through the Apple Metal framework, which is not fully visible in `psutil.Process().rss`.*

## Detailed Observations

### Text-to-Image (Stable Diffusion 1.5)
- **Load time:** ~8 seconds (from local HuggingFace cache)
- **Warm-up:** One-step 64×64 forward pass executes quickly (~0.3 s)
- **512×512 generation:** ~19 seconds for 20 steps (~1.0 step/sec)
- **Memory:** Process RSS stays under 600 MB, but actual GPU-side allocation is higher. The model fits comfortably in 16 GB unified memory with large headroom.
- **Image quality:** Adequate for baseline inference; photorealism can be improved with more steps or prompt engineering.

### Image-to-Text (SmolVLM 256M 4-bit)
- **Load time:** ~0.5 seconds (small model, cached weights)
- **Inference:** ~3.4 seconds for a 64-token description
- **Memory:** Very lightweight. Even at peak, the model leaves ~14 GB free for other processes.
- **Output quality:** Basic captioning and OCR work (e.g., "The text in the red box in the center of the image says Test Image"). Repetition can occur with low `temp` settings on simple images.

## Swap Usage
- Swap before benchmark: ~0 MB (negligible)
- Swap after benchmark: ~0 MB
- **Conclusion:** No significant swap pressure at these resolutions.

## Failure Conditions Tested
- ❌ Corrupt image: Correctly rejected by validation (`PIL.UnidentifiedImageError`)
- ❌ Oversized image: Correctly rejected by pixel-count validation
- ❌ Unsupported format: Correctly rejected by format whitelist
- ❌ Dimensions not multiple of 64: Correctly rejected by schema validator

## What This MacBook Can Handle

| Task | Feasibility | Notes |
|------|-------------|-------|
| SD 1.5 inference at 512×512 | ✅ Excellent | Batch size 1, plenty of headroom |
| SD 1.5 inference at 768×768 | ⚠️ Marginal | Likely works but slower; test before production use |
| SDXL inference | ❌ Not recommended | > 7 GB model; high swap risk |
| SmolVLM inference | ✅ Excellent | Very fast, tiny memory footprint |
| Qwen2.5-VL-3B inference | ✅ Good | Should fit in ~4–6 GB |
| VLM LoRA fine-tuning (rank 8) | ⚠️ Feasible with care | See training feasibility docs |
| SD LoRA fine-tuning | ⚠️ Feasible with care | See training feasibility docs |
| Concurrent heavy pipelines | ❌ Not supported by design | Memory manager prevents this |

## Recommended Next Steps

1. **For better T2I quality:** Try `runwayml/stable-diffusion-v1-5` with 30–50 steps at 512×512, or test a `RealisticVision` or `DreamShaper` community checkpoint.
2. **For better I2T quality:** Evaluate `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` if SmolVLM output is too simplistic.
3. **For training:** Start with the one-step smoke tests in `training/image_to_text/train_lora.py` and `training/text_to_image/train_lora.py`.
4. **For cloud fallback:** See README section "Moving Training to a Cloud NVIDIA GPU Later."

## Honest Conclusion

This MacBook Air M3 with 16 GB unified memory is **surprisingly capable** for local multimodal inference. Both pipelines run reliably without swap pressure at conservative settings. However, it is **not a training workstation**. Small LoRA experiments are feasible, but anything beyond a few hundred steps should be moved to a cloud GPU for speed and safety.
