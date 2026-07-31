#!/usr/bin/env python3
"""Download required model weights during setup.""""
import os, sys
from pathlib import Path

os.environ.setdefault("HF_HOME", str(Path("models/huggingface").resolve()))

def download_text_to_image():
    from diffusers import StableDiffusionPipeline
    model_id = os.getenv("T2I_MODEL_ID", "runwayml/stable-diffusion-v1-5")
    print(f"Downloading T2I model: {model_id}")
    StableDiffusionPipeline.from_pretrained(model_id, safety_checker=None)
    print("T2I model downloaded.")

def download_image_to_text():
    from mlx_vlm.utils import load
    model_id = os.getenv("I2T_MODEL_ID", "mlx-community/smolvlm-256m-8bit")
    print(f"Downloading I2T model: {model_id}")
    load(model_id)
    print("I2T model downloaded.")

if __name__ == "__main__":
    os.makedirs("models/huggingface", exist_ok=True)
    download_text_to_image()
    download_image_to_text()
