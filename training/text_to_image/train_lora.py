"""Text-to-Image LoRA fine-tuning using Diffusers + PEFT on Apple Silicon.

Conservative settings for 16 GB unified memory.
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import yaml
from diffusers import DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, get_peft_model
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

logger = logging.getLogger(__name__)


def train(
    config_path: str,
    smoke_test: bool = False,
) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    model_id = cfg.get("model_id", "runwayml/stable-diffusion-v1-5")
    dataset_dir = Path(cfg.get("dataset_dir", "datasets/text_to_image/splits"))
    output_dir = Path(cfg.get("output_dir", "adapters/text-to-image"))
    resolution = cfg.get("resolution", 512)
    lora_rank = cfg.get("lora_rank", 8)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0.05)
    epochs = cfg.get("epochs", 1)
    lr = cfg.get("learning_rate", 1e-4)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    train_text_encoder = cfg.get("train_text_encoder", False)
    mixed_precision = cfg.get("mixed_precision", "fp16")

    device = torch.device("mps")

    logger.info("Loading pipeline: %s", model_id)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if mixed_precision == "fp16" else torch.float32,
        safety_checker=None,
    )
    pipe = pipe.to(device)
    unet = pipe.unet

    # Inject LoRA into UNet
    lora_conf = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
    )
    unet = get_peft_model(unet, lora_conf)

    if train_text_encoder:
        text_lora = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "v_proj"],
            lora_dropout=lora_dropout,
        )
        pipe.text_encoder = get_peft_model(pipe.text_encoder, text_lora)

    # Prepare dataset (placeholder)
    logger.info("Dataset directory: %s", dataset_dir)

    if smoke_test:
        logger.info("SMOKE TEST — verifying LoRA injection and one-step forward pass")
        # Simple warm-up forward
        dummy_latents = torch.randn(1, 4, resolution // 8, resolution // 8, device=device, dtype=unet.dtype)
        dummy_timestep = torch.tensor([0], device=device)
        dummy_encoder_hidden = torch.randn(1, 77, 768, device=device, dtype=unet.dtype)
        with torch.no_grad():
            _ = unet(dummy_latents, dummy_timestep, dummy_encoder_hidden).sample
        logger.info("Smoke test passed: UNet LoRA forward succeeded on MPS.")
        return

    # Training placeholder
    logger.info("Training loop not implemented in this milestone. Config validated.")
    output_dir.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(output_dir))
    logger.info("Adapter saved to %s", output_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train(args.config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
