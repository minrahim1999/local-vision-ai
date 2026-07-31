# Backend Architecture

This document describes the cross-platform backend system that powers Local Vision AI.

## Design Philosophy

The backend system follows three principles:

1. **Platform Auto-Detection** — Code detects hardware and picks the best implementation
2. **Common Interface** — All backends implement identical methods; callers don't care which runs
3. **Graceful Degradation** — If the best backend fails, fall back to the next best

## Architecture Diagram

```
                    ┌─────────────────────┐
                    │   FastAPI Routes    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Backend Factory   │
                    │  (factory.py)       │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
     ┌────────▼─────┐  ┌──────▼───────┐  ┌─────▼──────┐
     │   Apple      │  │   NVIDIA     │  │   CPU      │
     │   Silicon    │  │   CUDA       │  │   Fallback │
     └──────┬───────┘  └──────┬───────┘  └─────┬──────┘
            │                 │                 │
     ┌──────▼───────┐  ┌─────▼──────┐  ┌──────▼──────┐
     │ AppleFlux2   │  │ Diffusers  │  │ Diffusers   │
     │ Backend      │  │ Backend    │  │ Backend     │
     │ (mflux/MLX)  │  │ (SD 2.1)   │  │ (SD 1.5)    │
     └──────────────┘  └────────────┘  └─────────────┘
            │                 │                 │
     ┌──────▼───────┐  ┌─────▼──────┐  ┌──────▼──────┐
     │ AppleSmolVLM │  │ Transformers│  │ Transformers │
     │ Backend      │  │ VLM Backend │  │ VLM Backend  │
     │ (MLX-VLM)    │  │ (Qwen2.5)   │  │ (Phi-3)     │
     └──────────────┘  └────────────┘  └─────────────┘
```

## Backend Registry

### Text-to-Image Backends

| Backend Class | File | Platform | Model | Quantization |
|--------------|------|----------|-------|-------------|
| `AppleFlux2Backend` | `t2i_apple.py` | macOS Apple Silicon | FLUX.2 klein 4B | int4 |
| `DiffusersBackend` | `t2i_diffusers.py` | All (best on CUDA) | SD 2.1 / SD 1.5 | fp16/fp32 |

### Image-to-Text Backends

| Backend Class | File | Platform | Model | Quantization |
|--------------|------|----------|-------|-------------|
| `AppleSmolVLMBackend` | `i2t_apple.py` | macOS Apple Silicon | SmolVLM 256M | int4 |
| `TransformersVLMBackend` | `i2t_transformers.py` | All (best on CUDA) | Qwen2.5-VL / Phi-3 | fp16/fp32 |

## Selection Priority

```python
# T2I selection order:
1. AppleFlux2Backend.is_available()   # macOS + Apple Silicon?
2. DiffusersBackend.is_available()    # Always true if diffusers installed

# I2T selection order:
1. AppleSmolVLMBackend.is_available() # macOS + Apple Silicon?
2. TransformersVLMBackend.is_available()  # Always true if transformers installed
```

## Interface Contract

All backends must implement:

### TextToImageBackend

```python
class TextToImageBackend(ABC):
    @staticmethod
    def is_available() -> bool: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def generate(self, prompt: str, ...) -> GeneratedImage: ...
```

### ImageToTextBackend

```python
class ImageToTextBackend(ABC):
    @staticmethod
    def is_available() -> bool: ...
    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def analyze(self, image_path: str, prompt: str, ...) -> VisionResult: ...
    def extract(self, image_path: str, prompt: str, ...) -> VisionResult: ...
```

## Adding a New Backend

To add support for a new platform or model:

1. Create a new file in `services/backends/` (e.g., `t2i_onnx.py`)
2. Inherit from `TextToImageBackend` or `ImageToTextBackend`
3. Implement all abstract methods
4. Add to `T2I_BACKENDS` or `I2T_BACKENDS` in `factory.py`
5. Register in priority order (best first)

Example:

```python
# services/backends/t2i_onnx.py
from services.backends.base import TextToImageBackend

class OnnxBackend(TextToImageBackend):
    @staticmethod
    def is_available() -> bool:
        try:
            import onnxruntime
            return "CUDA" in onnxruntime.get_available_providers()
        except ImportError:
            return False

    def load(self) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...
    def generate(self, prompt: str, ...) -> GeneratedImage: ...
```

Then in `factory.py`:
```python
from services.backends.t2i_onnx import OnnxBackend

T2I_BACKENDS = [AppleFlux2Backend, OnnxBackend, DiffusersBackend]
```

## Platform-Specific Notes

### Apple Silicon (MLX)

- **Memory:** Unified memory is shared between CPU and GPU. `gc.collect()` is not sufficient — model references must be explicitly deleted.
- **Subprocess:** `AppleFlux2Backend` runs mflux in a subprocess to isolate memory. The parent process stays light.
- **MLX Cache:** Set via `--mlx-cache-limit-gb` or `mx.set_cache_limit()` to prevent unbounded growth.

### NVIDIA CUDA (Diffusers)

- **Device:** Auto-detected via `torch.cuda.is_available()`. Falls back to MPS on macOS Intel, then CPU.
- **Dtype:** fp16 on CUDA for speed, float32 on MPS/CPU for stability.
- **Offloading:** `enable_sequential_cpu_offload()` reduces VRAM usage at the cost of speed.

### CPU-Only

- **Expectation:** Very slow. SD 1.5 at 512×512 takes 5–10 minutes per image.
- **Use case:** Emergency fallback, testing, or very small models only.

## Configuration

Backends read from YAML config files:

```yaml
# config/text_to_image.yaml
t2i:
  flux2:
    model_id: "mlx-community/FLUX.2-Klein-4B-4bit"
    default_width: 512
    default_height: 512
  sd15:
    model_id: "stabilityai/stable-diffusion-2-1"
    device: null  # auto-detect
```

The factory passes the appropriate config section to the selected backend.

---

*Last updated: 2026-07-31*
