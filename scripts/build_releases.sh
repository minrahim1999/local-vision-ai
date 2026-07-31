#!/usr/bin/env bash
# Build release binaries locally using PyInstaller.
# Usage: bash scripts/build_releases.sh [version_tag]

set -euo pipefail

VERSION="${1:-v0.2.0}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$PROJECT_DIR/dist"

mkdir -p "$OUTPUT_DIR"

echo "=== Building Local Vision AI $VERSION ==="
echo "Project: $PROJECT_DIR"
echo "Output:  $OUTPUT_DIR"
echo ""

# Check prerequisites
check_cmd() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "ERROR: '$1' not found. Install it first."
        exit 1
    fi
}

check_cmd python3
check_cmd uv

# Ensure venv exists and has deps
cd "$PROJECT_DIR"
if [[ ! -d ".venv" ]]; then
    echo "Creating virtual environment..."
    uv venv --python 3.11
fi

echo "Installing dependencies..."
uv pip install -e ".[dev,desktop]"
uv pip install pyinstaller

# Determine platform
PLATFORM=$(uname -s)
ARCH=$(uname -m)

echo ""
echo "Platform: $PLATFORM ($ARCH)"
echo ""

case "$PLATFORM" in
    Darwin)
        echo "=== Building macOS .app bundle ==="
        .venv/bin/pyinstaller app/desktop.spec --clean --noconfirm

        if [[ -d "dist/LocalVisionAI.app" ]]; then
            echo "✅ Built: dist/LocalVisionAI.app"

            # Try to create DMG
            if command -v create-dmg >/dev/null 2>&1; then
                echo "Creating DMG..."
                mkdir -p dist/dmg_staging
                cp -r "dist/LocalVisionAI.app" "dist/dmg_staging/"
                create-dmg \
                    --volname "Local Vision AI" \
                    --window-pos 200 120 \
                    --window-size 600 400 \
                    --icon-size 100 \
                    --app-drop-link 450 185 \
                    "dist/LocalVisionAI-${VERSION}-macOS.dmg" \
                    "dist/dmg_staging/"
                echo "✅ DMG: dist/LocalVisionAI-${VERSION}-macOS.dmg"
            else
                echo "⚠️ create-dmg not found. Creating zip instead."
                cd dist && zip -r "LocalVisionAI-${VERSION}-macOS.zip" LocalVisionAI.app
                echo "✅ ZIP: dist/LocalVisionAI-${VERSION}-macOS.zip"
            fi
        fi
        ;;

    Linux)
        echo "=== Building Linux executable ==="
        .venv/bin/pyinstaller app/desktop.spec --clean --noconfirm

        if [[ -d "dist/LocalVisionAI" ]]; then
            echo "✅ Built: dist/LocalVisionAI/"
            cd dist && tar -czf "LocalVisionAI-${VERSION}-Linux.tar.gz" LocalVisionAI/
            echo "✅ Archive: dist/LocalVisionAI-${VERSION}-Linux.tar.gz"
        fi
        ;;

    MINGW*|MSYS*|CYGWIN*)
        echo "=== Building Windows executable ==="
        .venv/Scripts/pyinstaller app/desktop.spec --clean --noconfirm

        if [[ -d "dist/LocalVisionAI" ]]; then
            echo "✅ Built: dist/LocalVisionAI/"
            cd dist && 7z a -tzip "LocalVisionAI-${VERSION}-Windows.zip" LocalVisionAI/
            echo "✅ ZIP: dist/LocalVisionAI-${VERSION}-Windows.zip"
        fi
        ;;

    *)
        echo "ERROR: Unsupported platform: $PLATFORM"
        exit 1
        ;;
esac

echo ""
echo "=== Build Complete ==="
ls -lh dist/*.{dmg,zip,tar.gz} 2>/dev/null || echo "Check dist/ directory"
