# API Reference

Base URL: `http://localhost:8000`

## Endpoints Overview

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + current pipeline status |
| GET | `/v1/system/memory` | Memory telemetry (RSS, swap, system %) |
| GET | `/v1/models/status` | Which models are loaded |
| POST | `/v1/images/generate` | Generate image from text prompt |
| POST | `/v1/vision/analyze` | Analyze image, return text/JSON |
| POST | `/v1/vision/extract` | Extract structured JSON from image |

---

## GET /health

**Response:**
```json
{
  "status": "ok",
  "pipeline": "text_to_image"
}
```

---

## GET /v1/system/memory

**Response:**
```json
{
  "current_pipeline": "text_to_image",
  "rss_mb": 4872,
  "vms_mb": 14230,
  "system_percent": 34.5,
  "swap_used_mb": 128,
  "uptime_seconds": 1245.67
}
```

---

## GET /v1/models/status

**Response:**
```json
{
  "pipelines": {
    "text_to_image": true,
    "image_to_text": false
  },
  "current_pipeline": "text_to_image"
}
```

---

## POST /v1/images/generate

Generate an image from a text prompt. Only one pipeline stays loaded at a time — requesting T2I automatically unloads I2T first.

**Request (JSON):**
```json
{
  "prompt": "A futuristic robot working at a wooden desk, warm lighting",
  "negative_prompt": "blurry, deformed, low quality",
  "width": 512,
  "height": 512,
  "steps": 4,
  "guidance_scale": 1.0,
  "seed": 1337
}
```

**Fields:**
| Field | Type | Required | Default | Notes |
|-------|------|----------|---------|-------|
| `prompt` | string | ✅ | — | Main generation prompt |
| `negative_prompt` | string | ❌ | null | Things to avoid |
| `width` | int | ❌ | 512 | Multiple of 64, max 1024 |
| `height` | int | ❌ | 512 | Multiple of 64, max 1024 |
| `steps` | int | ❌ | 4 | FLUX.2 uses 4; SD uses 20 |
| `guidance_scale` | float | ❌ | 1.0 | FLUX.2: 1.0; SD: 7.0 |
| `seed` | int | ❌ | null | null = random seed |

**Response (JSON):**
```json
{
  "prompt": "A futuristic robot...",
  "negative_prompt": "blurry, deformed, low quality",
  "width": 512,
  "height": 512,
  "steps": 4,
  "guidance_scale": 1.0,
  "seed": 1337,
  "duration_seconds": 38.452,
  "model_id": "mlx-community/FLUX.2-Klein-4B-4bit",
  "output_path": "/path/to/project/outputs/text_to_image/flux2_1234567890_1337.png"
}
```

**Error Responses:**
| Status | Cause |
|--------|-------|
| 422 | Invalid dimensions (not multiple of 64, exceeds max) |
| 503 | Memory limit exceeded |
| 500 | Model error or generation failure |

---

## POST /v1/vision/analyze

Upload an image and analyze it with a prompt.

**Request (multipart/form-data):**
- `image`: File upload (PNG, JPG, WebP, max 10 MB, max 2 MP)
- `prompt`: Text prompt (e.g., "Describe this image")
- `response_format`: `"text"` or `"json"`
- `max_tokens`: int, default 512

**curl example:**
```bash
curl -X POST http://localhost:8000/v1/vision/analyze \
  -F "image=@photo.jpg" \
  -F "prompt=Describe this image in detail" \
  -F "response_format=text" \
  -F "max_tokens=512"
```

**Response (JSON):**
```json
{
  "image_path": "outputs/temp_uploads/1234567890_photo.jpg",
  "prompt": "Describe this image in detail",
  "content": "A black sensor with a blue LED indicator sits on a white desk...",
  "response_format": "text",
  "model_id": "mlx-community/SmolVLM-256M-Instruct-4bit",
  "duration_seconds": 2.847
}
```

---

## POST /v1/vision/extract

Upload an image and extract structured JSON.

**Request (multipart/form-data):**
- `image`: File upload
- `prompt`: Extraction instruction (e.g., "Extract objects and colors")
- `max_tokens`: int, default 512

**curl example:**
```bash
curl -X POST http://localhost:8000/v1/vision/extract \
  -F "image=@photo.jpg" \
  -F "prompt=Extract all objects and their colors as JSON" \
  -F "max_tokens=512"
```

**Response (JSON — when valid JSON produced):**
```json
{
  "image_path": "outputs/temp_uploads/1234567890_photo.jpg",
  "prompt": "Extract all objects and their colors as JSON",
  "content": "{\"objects\":[{\"name\":\"sensor\",\"color\":\"black\"},{\"name\":\"LED\",\"color\":\"blue\"}]}",
  "response_format": "json",
  "model_id": "mlx-community/SmolVLM-256M-Instruct-4bit",
  "duration_seconds": 3.12
}
```

**Notes:**
- The model may return non-JSON even when `response_format="json"` is requested.
- The backend attempts up to 3 retries with the same prompt.
- If all retries fail, the raw text is returned with `response_format="json"`.

---

## Error Response Format

All errors follow this structure:
```json
{
  "detail": "Error message here"
}
```

Common HTTP status codes:
| Code | Meaning |
|------|---------|
| 200 | Success |
| 422 | Validation error (bad dimensions, corrupt image, etc.) |
| 503 | Service unavailable (out of memory, model loading) |
| 500 | Internal server error |

---

## Model Switching Behavior

The API uses a single-model-at-a-time policy:

```
Request T2I  →  unload I2T  →  load T2I   →  generate
Request I2T  →  unload T2T  →  load I2T   →  analyze
```

This is enforced by the `MemoryManager`. Concurrent requests to different pipelines will serialize automatically.

---

## Rate Limiting / Concurrency

By default, only **one generation job** runs at a time. The second request will wait for the first to complete (and for model unload+load).

If you need concurrent processing, run multiple API instances with separate model copies (requires more RAM).

---

*Last updated: 2026-07-31*
