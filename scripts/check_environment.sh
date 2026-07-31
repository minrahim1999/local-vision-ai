#!/usr/bin/env bash
set -euo pipefail

echo "================================"
echo "Local Vision AI Environment Audit"
echo "================================"

echo ""
echo "--- Hardware ---"
uname -m
sw_vers -productName -productVersion
echo "Model: $(sysctl -n hw.model 2>/dev/null || echo 'unknown')"
echo "CPU cores: $(sysctl -n hw.ncpu)"
echo "Memory pages: $(sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024) " GB"}')"

echo ""
echo "--- Disk ---"
df -h /System/Volumes/Data | tail -n 1

echo ""
echo "--- Python Interpreters ---"
python3 --version || true
python3.10 --version 2>/dev/null || true
python3.11 --version 2>/dev/null || true
python3.12 --version 2>/dev/null || true
python3.13 --version 2>/dev/null || true

echo ""
echo "--- uv ---"
uv --version || true

echo ""
echo "--- Unsloth CLI ---"
command -v unsloth && unsloth --version || echo "unsloth CLI not in PATH"

echo ""
echo "--- Hermes ---"
command -v hermes && hermes --version || true

echo ""
echo "--- Python acceleration packages ---"
python3 -c "
import platform, sys
print('Platform:', platform.platform())
try:
    import torch
    print('PyTorch:', torch.__version__)
    print('MPS built:', torch.backends.mps.is_built())
    print('MPS available:', torch.backends.mps.is_available())
except Exception as e:
    print('torch:', e)
try:
    import mlx.core as mx
    print('MLX:', getattr(mx, '__version__', 'unknown'))
except Exception as e:
    print('mlx:', e)
try:
    import mlx_vlm
    print('MLX-VLM: available')
except Exception as e:
    print('mlx-vlm:', e)
try:
    import diffusers
    print('Diffusers:', diffusers.__version__)
except Exception as e:
    print('diffusers:', e)
try:
    import transformers
    print('Transformers:', transformers.__version__)
except Exception as e:
    print('transformers:', e)
try:
    import unsloth
    print('Unsloth: available')
except Exception as e:
    print('unsloth:', e)
" || true

echo ""
echo "--- Done ---"
