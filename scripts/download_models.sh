#!/usr/bin/env bash
# Download FLUX.2 [klein] 4B weights via mflux (triggers caching on first run).
# Usage: bash scripts/download_models.sh
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export HF_HOME="${HF_HOME:-$PROJECT_ROOT/models/huggingface}"

VENV_BIN="$PROJECT_ROOT/.venv/bin"
MFLUX="$VENV_BIN/mflux-generate-flux2"

if [[ ! -x "$MFLUX" ]]; then
    echo "ERROR: mflux-generate-flux2 not found at $MFLUX"
    echo "Run: uv pip install --python $PROJECT_ROOT/.venv/bin/python mflux"
    exit 1
fi

echo "=== Pre-downloading FLUX.2 [klein] 4B weights ==="
echo "HF_HOME: $HF_HOME"
mkdir -p "$HF_HOME"

# Trigger a tiny generation so mflux downloads weights
echo "Running 1-step dummy generation to cache weights..."
"$MFLUX" \
    --model mlx-community/FLUX.2-Klein-4B-4bit \
    --prompt "test" \
    --width 256 --height 256 \
    --steps 1 \
    --output /tmp/flux2_dummy.png \
    2>&1 | tee /tmp/flux2_download.log

# Cleanup dummy
echo "Cleaning up dummy output..."
rm -f /tmp/flux2_dummy.png

echo "=== Download complete ==="
echo "Weights cached at: $HF_HOME/hub/models--mlx-community--FLUX.2-Klein-4B-4bit"
echo "Disk usage:"
du -sh "$HF_HOME/hub/models--mlx-community--FLUX.2-Klein-4B-4bit" 2>/dev/null || true
