#!/usr/bin/env python3
"""Benchmark both pipelines and save results."""
import json, os, sys, time, psutil
from pathlib import Path
import gc

BASE = Path(__file__).resolve().parent.parent
os.chdir(BASE)
sys.path.insert(0, str(BASE))
results = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "platform": "macOS Apple Silicon",
    "benchmarks": []
}

def snapshot():
    p = psutil.Process()
    m = p.memory_info()
    s = psutil.swap_memory()
    return {"rss_mb": m.rss // (1024*1024), "vms_mb": m.vms // (1024*1024), "swap_mb": s.used // (1024*1024)}

# --- Text-to-Image Benchmark ---
try:
    import torch
    from services.text_to_image import TextToImageService
    t2i = TextToImageService()
    
    mem_before = snapshot()
    start = time.monotonic()
    t2i.load()
    load_time = time.monotonic() - start
    mem_after_load = snapshot()
    
    start = time.monotonic()
    img = t2i.generate("a cat sitting on a sofa", width=512, height=512, steps=20, seed=42)
    gen_time = time.monotonic() - start
    mem_after_gen = snapshot()
    
    results["benchmarks"].append({
        "pipeline": "text_to_image",
        "model": t2i.model_id,
        "load_time_s": round(load_time, 2),
        "first_inference_s": round(gen_time, 2),
        "mem_before_mb": mem_before["rss_mb"],
        "mem_after_load_mb": mem_after_load["rss_mb"],
        "mem_after_gen_mb": mem_after_gen["rss_mb"],
        "output_dims": f"{img.width}x{img.height}",
        "steps": img.steps,
        "seed": img.seed,
    })
    t2i.unload()
except Exception as e:
    results["benchmarks"].append({"pipeline": "text_to_image", "error": str(e)})

gc.collect()

# --- Image-to-Text Benchmark ---
try:
    from services.image_to_text import ImageToTextService
    i2t = ImageToTextService()
    test_img = BASE / "outputs" / "test_image.png"
    
    mem_before = snapshot()
    start = time.monotonic()
    i2t.load()
    load_time = time.monotonic() - start
    mem_after_load = snapshot()
    
    start = time.monotonic()
    result = i2t.analyze(str(test_img), "Describe this image briefly.")
    infer_time = time.monotonic() - start
    mem_after_inf = snapshot()
    
    results["benchmarks"].append({
        "pipeline": "image_to_text",
        "model": i2t.model_id,
        "load_time_s": round(load_time, 2),
        "first_inference_s": round(infer_time, 2),
        "mem_before_mb": mem_before["rss_mb"],
        "mem_after_load_mb": mem_after_load["rss_mb"],
        "mem_after_inf_mb": mem_after_inf["rss_mb"],
        "output_tokens_approx": len(result.response.split()),
    })
    i2t.unload()
except Exception as e:
    results["benchmarks"].append({"pipeline": "image_to_text", "error": str(e)})

# Save
(BASE / "benchmarks").mkdir(exist_ok=True)
with open(BASE / "benchmarks" / "results.json", "w") as f:
    json.dump(results, f, indent=2)
print(json.dumps(results, indent=2))
