#!/usr/bin/env bash
# Cross-platform environment detection and smoke test
# Usage: bash scripts/check_cross_platform.sh

set -euo pipefail

echo "=== Local Vision AI — Cross-Platform Detection ==="
echo ""

# OS detection
OS=$(uname -s)
ARCH=$(uname -m)
echo "OS: $OS"
echo "Architecture: $ARCH"

# Python
python3 --version 2>/dev/null || python --version

# PyTorch device
echo ""
echo "--- PyTorch Device Check ---"
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA device:', torch.cuda.get_device_name(0))
print('MPS available:', torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False)
" 2>/dev/null || echo "PyTorch not installed"

# MLX (Apple only)
echo ""
echo "--- MLX Check ---"
python3 -c "import mlx.core; print('MLX available')" 2>/dev/null || echo "MLX not available (expected on non-Apple)"

# mflux (Apple only)
echo "--- mflux Check ---"
command -v mflux-generate-flux2 >/dev/null 2>&1 && echo "mflux available" || echo "mflux not available (expected on non-Apple)"

# Diffusers
echo ""
echo "--- Diffusers Check ---"
python3 -c "import diffusers; print('Diffusers:', diffusers.__version__)" 2>/dev/null || echo "Diffusers not installed"

# Transformers
echo "--- Transformers Check ---"
python3 -c "import transformers; print('Transformers:', transformers.__version__)" 2>/dev/null || echo "Transformers not installed"

# Backend auto-detection
echo ""
echo "--- Backend Auto-Detection ---"
python3 -c "
from services.backends.factory import T2I_BACKENDS, I2T_BACKENDS
print('T2I backends:')
for cls in T2I_BACKENDS:
    avail = cls.is_available()
    print(f'  {cls.__name__}: {\"YES\" if avail else \"NO\"}')
print('I2T backends:')
for cls in I2T_BACKENDS:
    avail = cls.is_available()
    print(f'  {cls.__name__}: {\"YES\" if avail else \"NO\"}')
" 2>/dev/null || echo "Backend import failed — run from project root with .venv activated"

echo ""
echo "=== Detection Complete ==="
