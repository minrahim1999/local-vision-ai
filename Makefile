# Local Vision AI — Makefile
.PHONY: setup audit download-models run test benchmark train-vlm-smoke train-vlm evaluate-vlm clean-cache

PYTHON := uv run python
VENV := .venv

setup:
	uv venv --python 3.11
	uv pip install -e ".[text_to_image,image_to_text,unsloth,dev]"
	@echo "Setup complete. Copy .env.example to .env and edit."

audit:
	bash scripts/check_environment.sh

download-models:
	$(PYTHON) scripts/download_models.py

run:
	$(PYTHON) -m uvicorn api.main:app --host $(shell echo $$API_HOST | sed 's/$$/0.0.0.0/') --port $(shell echo $$API_PORT | sed 's/$$/8000/')

test:
	$(PYTHON) -m pytest tests/ -v --timeout=120

benchmark:
	$(PYTHON) scripts/benchmark.py

train-vlm-smoke:
	$(PYTHON) -m training.image_to_text.train_lora --config config/image_to_text.yaml --smoke-test

train-vlm:
	$(PYTHON) -m training.image_to_text.train_lora --config config/image_to_text.yaml

evaluate-vlm:
	$(PYTHON) -m training.image_to_text.evaluate --adapter adapters/image-to-text

clean-cache:
	rm -rf outputs/cache/* benchmarks/*.png benchmarks/*.jpg
	find . -type d -name __pycache__ -exec rm -rf {} +
	@echo "Cache cleaned."
