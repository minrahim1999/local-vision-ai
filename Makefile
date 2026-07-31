# Local Vision AI — Makefile
.PHONY: setup audit download-models run test benchmark train-vlm-smoke train-vlm evaluate-vlm clean-cache generate-flux2

PYTHON := uv run python
VENV := .venv
HF_HOME := $(shell pwd)/models/huggingface

setup:
	uv venv --python 3.11
	uv pip install -e ".[text_to_image,image_to_text,unsloth,dev]"
	uv pip install mflux
	@echo "Setup complete. Copy .env.example to .env and edit."

audit:
	bash scripts/check_environment.sh
	bash scripts/check_cross_platform.sh

download-models:
	bash scripts/download_models.sh

run:
	HF_HOME=$(HF_HOME) $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000

test:
	$(PYTHON) -m pytest tests/ -v --timeout=120

benchmark:
	$(PYTHON) scripts/benchmark.py

# Convenience: generate an image with FLUX.2 directly via CLI
generate-flux2:
	HF_HOME=$(HF_HOME) .venv/bin/mflux-generate-flux2 \
		--model mlx-community/FLUX.2-Klein-4B-4bit \
		--prompt "$(PROMPT)" \
		--width $(or $(WIDTH),512) --height $(or $(HEIGHT),512) \
		--steps $(or $(STEPS),4) --seed $(or $(SEED),1337) \
		--output outputs/text_to_image/flux2_$(shell date +%s).png

train-vlm-smoke:
	$(PYTHON) -m training.image_to_text.train_lora --config config/image_to_text.yaml --smoke-test

train-vlm:
	$(PYTHON) -m training.image_to_text.train_lora --config config/image_to_text.yaml

evaluate-vlm:
	$(PYTHON) -m training.image_to_text.evaluate --adapter adapters/image-to-text

clean-cache:
	rm -rf outputs/cache/* outputs/text_to_image/* benchmarks/*.png benchmarks/*.jpg
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Cache cleaned."
