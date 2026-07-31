"""Image-to-text LoRA fine-tuning via Unsloth FastVisionModel.

Conservative settings for 16 GB unified memory:
- batch_size: 1
- lora_rank: 8
- lora_alpha: 16
- lora_dropout: 0.05
- gradient_accumulation_steps: 8
- epochs: 1 (initially)
- gradient_checkpointing: enabled
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import yaml
from unsloth import FastVisionModel

logger = logging.getLogger(__name__)


def train(
    config_path: str,
    smoke_test: bool = False,
) -> None:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    model_id = cfg.get("model_id", "unsloth/Qwen2.5-VL-3B-Instruct")
    lora_rank = cfg.get("lora_rank", 8)
    lora_alpha = cfg.get("lora_alpha", 16)
    lora_dropout = cfg.get("lora_dropout", 0.05)
    epochs = cfg.get("epochs", 1)
    lr = cfg.get("learning_rate", 2e-4)
    grad_accum = cfg.get("gradient_accumulation_steps", 8)
    max_seq_length = cfg.get("max_seq_length", 1024)
    output_dir = Path(cfg.get("output_dir", "adapters/image-to-text"))
    dataset_dir = Path(cfg.get("dataset_dir", "datasets/image_to_text/splits"))

    logger.info("Starting VLM LoRA training for %s", model_id)
    logger.info("Settings: rank=%s alpha=%s dropout=%s epochs=%s", lora_rank, lora_alpha, lora_dropout, epochs)

    # Load model with Unsloth
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=True,
    )
    model = FastVisionModel.get_peft_model(
        model,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    # Dataset prep placeholder — user should implement data collator
    logger.info("Dataset directory: %s", dataset_dir)

    if smoke_test:
        logger.info("SMOKE TEST — running one step only")
        # Perform a single forward/backward to verify the graph works
        # ... placeholder: real implementation requires a custom vision dataset collator
        logger.info("Smoke test placeholder complete. Implement dataset loading before full run.")
        return

    # Training loop placeholder
    logger.info("Training would run here with the conservative hyperparameters above.")
    logger.info("Saving adapter to %s", output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    train(args.config, smoke_test=args.smoke_test)


if __name__ == "__main__":
    main()
