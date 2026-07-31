"""Standalone launcher — starts API server in background, then opens desktop GUI.

Usage:
    python -m app.standalone
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

API_PORT = int(os.getenv("LVAI_API_PORT", "8000"))
API_HOST = os.getenv("LVAI_API_HOST", "127.0.0.1")
API_URL = f"http://{API_HOST}:{API_PORT}"


def _wait_for_api(timeout: int = 60) -> bool:
    import urllib.request
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        try:
            with urllib.request.urlopen(f"{API_URL}/health", timeout=2) as resp:
                return resp.status == 200
        except Exception:
            time.sleep(0.5)
    return False


def main() -> None:
    # Check if API already running
    try:
        import urllib.request
        with urllib.request.urlopen(f"{API_URL}/health", timeout=2) as resp:
            if resp.status == 200:
                print("API already running. Launching desktop...")
                from app.desktop import main as desktop_main
                import flet as ft
                ft.app(target=desktop_main)
                return
    except Exception:
        pass

    # Start API in subprocess
    project_root = Path(__file__).resolve().parent.parent
    env = os.environ.copy()
    env["HF_HOME"] = str(project_root / "models" / "huggingface")
    env["LVAI_API_HOST"] = API_HOST
    env["LVAI_API_PORT"] = str(API_PORT)

    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--host", API_HOST, "--port", str(API_PORT)],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    print("Starting API server...")
    if not _wait_for_api(timeout=60):
        print("ERROR: API failed to start within 60s")
        api_proc.terminate()
        sys.exit(1)

    print(f"API ready at {API_URL}. Launching desktop...")
    try:
        from app.desktop import main as desktop_main
        import flet as ft
        ft.app(target=desktop_main)
    finally:
        print("Shutting down API...")
        api_proc.terminate()
        try:
            api_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api_proc.kill()


if __name__ == "__main__":
    main()
