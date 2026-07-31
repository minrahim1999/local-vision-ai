# Local Vision AI

Production-structured local multimodal AI for Apple Silicon.

## Purpose

Local Vision AI provides two local inference pipelines on your Mac:

1. **Text-to-Image** — Generate images from prompts using Stable Diffusion 1.5.
2. **Image-to-Text** — Analyze images, caption them, answer visual questions, and extract structured JSON using SmolVLM (via MLX-VLM).

Everything runs on-device. No cloud API keys required.

## Architecture

```
┌──────────────────────────────────────────┐
│            FastAPI (uvicorn)             │
├──────────────┬───────────────────────────┤
│   POST       │    POST       POST        │
│ /v1/images   │ /v1/vision  /v1/vision   │
│ /generate    │ /analyze    /extract      │
└──────┬───────┴──────┬────────────┬───────┘
       │              │            │
┌──────▼───────┐ ┌────▼────────┐  │
│ TextToImage  │ │ ImageToText │  │
│   Service    │ │   Service   │  │
└──────┬───────┘ └────┬────────┘  │
       │              │           │
┌──────▼──────────────▼───────────▼────────┐
│          Memory Manager                  │
│  (One model loaded at a time, thread-  │
│   safe switching, telemetry)           │
└──────────────────────────────────────────┘
```

## Hardware Assumptions

- Apple Silicon Mac (M1/M2/M3+)
- macOS 14+
- 16 GB unified memory (minimum comfortable config)
- No CUDA / NVIDIA hardware

## Installation

Requires **uv** (Python package manager) and **git**.

```bash
cd local-vision-ai
make setup
```

Or manually:

```bash
uv venv --python 3.11
uv pip install -e ".[text_to_image,image_to_text,dev]"
cp .env.example .env
# Edit .env with your preferred settings
```

## Model Download

Download weights before running the API (avoids runtime stalls):

```bash
make download-models
```

Or:

```bash
uv run python scripts/download_models.py
```

Models are cached to `./models/huggingface/`.

## Running the API

```bash
make run
```

Or:

```bash
uv run python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

## Example curl Commands

### Text-to-Image

```bash
curl -X POST http://localhost:8000/v1/images/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A small futuristic robot working at a wooden desk",
    "negative_prompt": "blurry, deformed, low quality",
    "width": 512,
    "height": 512,
    "steps": 20,
    "guidance_scale": 7.0,
    "seed": 3407
  }'
```

### Image-to-Text (Analyze)

```bash
curl -X POST http://localhost:8000/v1/vision/analyze \
  -F "image=@sample.jpg" \
  -F "prompt=Describe this image in detail." \
  -F "response_format=text" \
  -F "max_tokens=512"
```

### Image-to-Text (Extract JSON)

```bash
curl -X POST http://localhost:8000/v1/vision/extract \
  -F "image=@sample.jpg" \
  -F "prompt=Extract the objects and colors as JSON." \
  -F "max_tokens=512"
```

## Dataset Preparation

### Image-to-Text

Format (JSONL):

```json
{"image":"images/001.jpg","messages":[{"role":"user","content":"Describe this image."},{"role":"assistant","content":"A black sensor on a desk."}]}
```

Prepare and split:

```bash
uv run python -m training.image_to_text.prepare_dataset \
  --input-jsonl datasets/image_to_text/dataset.jsonl \
  --images-dir datasets/image_to_text/images
```

### Text-to-Image

Format (JSONL):

```json
{"file_name":"001.jpg","text":"A black smart sensor placed on a white desk"}
```

Prepare and split:

```bash
uv run python -m training.text_to_image.prepare_dataset \
  --input-jsonl datasets/text_to_image/metadata.jsonl \
  --images-dir datasets/text_to_image/images
```

## Fine-Tuning

### Image-to-Text (VLM LoRA)

```bash
# Smoke test first
make train-vlm-smoke

# Full run (if smoke test passes and memory is safe)
make train-vlm
```

### Text-to-Image LoRA Feasibility

See [`docs/text-to-image-training-feasibility.md`](docs/text-to-image-training-feasibility.md).

Local training is feasible for small experiments, but a cloud GPU is recommended for production LoRA training.

## Evaluation

```bash
make evaluate-vlm
```

## Memory Limitations

- Only **one** large model is kept in memory at a time.
- The memory manager automatically unloads the previous pipeline before loading the next.
- Peak safe memory usage: ~12 GB. The system will reject requests if memory exceeds the configured limit.
- Start image generation at **512×512**.
- Do not run other heavy apps (e.g., video editing, Xcode builds) simultaneously.

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| `NotImplementedError` during T2I | Unsupported MPS op | Lower resolution or steps |
| `MemoryError` | Model already loaded + another request | Wait for idle unload or restart API |
| Slow first inference | MPS shader compilation | Warm-up run on startup |
| Corrupt image upload | Wrong format or size | Use PNG/JPG under 10 MB |
| JSON parse failure from I2T | Model output raw text | Retry or tighten prompt |

## How Unsloth Is Used

Unsloth is available in the project for vision-language fine-tuning. The installed version supports:

- `FastVisionModel.from_pretrained(...)` with 4-bit loading
- PEFT LoRA injection
- Gradient checkpointing

It is **not** used for text-to-image in Milestone 1 (Diffusers + MPS is more reliable for SD 1.5).

## Replacing Selected Models

Edit `config/text_to_image.yaml` and `config/image_to_text.yaml`:

- **T2I:** Change `model_id` to any SD 1.5/2.1-compatible Diffusers checkpoint.
- **I2T:** Change `model_id` to any `mlx-community` quantized VLM (e.g., Qwen2.5-VL-3B, Gemma-3-4B).

Then run `make download-models` again.

## Moving Training to a Cloud NVIDIA GPU Later

1. Export the adapter:
   ```bash
   python -c "from peft import PeftModel; ..."
   ```
2. Upload the adapter + dataset to the cloud instance.
3. Install CUDA PyTorch (`torch>=2.2+cu121`).
4. Increase batch size (e.g., 4) and resolution (e.g., 768×768).
5. Run the same training scripts — they are framework-agnostic except for device config.

## License Considerations

- **SmolVLM:** Apache 2.0
- **Stable Diffusion 1.5:** CreativeML Open RAIL-M
- **Qwen2.5-VL:** Qwen License (read terms before commercial use)
- **Generated outputs:** Subject to model license terms; review before redistribution.

## Makefile Commands

| Command | Purpose |
|---------|---------|
| `make setup` | Create venv and install dependencies |
| `make audit` | Run environment audit script |
| `make download-models` | Download model weights |
| `make run` | Start API server |
| `make test` | Run pytest suite |
| `make benchmark` | Run benchmarks |
| `make train-vlm-smoke` | One-step VLM LoRA smoke test |
| `make train-vlm` | Full VLM LoRA training |
| `make evaluate-vlm` | Evaluate VLM adapter |
| `make clean-cache` | Remove generated cache files |

## Development

```bash
uv run pytest tests/ -v
uv run ruff check .
uv run mypy api/ services/ training/
```
