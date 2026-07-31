"""Evaluate a fine-tuned image-to-text adapter (placeholder)."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def evaluate(adapter_dir: Path) -> None:
    logger.info("Evaluating adapter: %s", adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")
    logger.info("Evaluation placeholder complete.")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.adapter)


if __name__ == "__main__":
    main()
