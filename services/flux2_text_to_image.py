"""Text-to-Image inference service using FLUX.2 [klein] 4B via mflux (MLX native).

This service wraps mflux CLI calls because mflux does not expose a stable Python API
for Flux2 generation. Using a subprocess has the benefit of full memory isolation:
the heavy model is loaded inside the subprocess and memory is reclaimed on exit.
"""
from __future__ import annotations

import gc
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

from api.schemas import GeneratedImage

logger = logging.getLogger(__name__)


class Flux2TextToImageService:
    """Text-to-Image via FLUX.2 [klein] 4B int4 on Apple MLX.

    Memory footprint:
      - 512×512 @ int4 ~ 4.5–5.5 GB peak
      - 1024×1024 @ int4 ~ 11–13 GB peak

    The service does NOT keep the model resident in Python memory; instead it
    spawns mflux-generate-flux2 which loads weights inside a subprocess. This
    keeps the FastAPI process light and avoids OOM when switching pipelines.
    """

    MODEL_ID = "mlx-community/FLUX.2-Klein-4B-4bit"
    MFLUX_CMD = "mflux-generate-flux2"

    def __init__(
        self,
        outputs_dir: str = "outputs/text_to_image",
        hf_home: str | None = None,
        low_ram: bool = True,
        mlx_cache_limit_gb: float | None = None,
    ) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home = Path(hf_home) if hf_home else None
        self.low_ram = low_ram
        self.mlx_cache_limit_gb = mlx_cache_limit_gb or 8.0
        self._is_ready = False
        self._check_mflux()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def is_loaded(self) -> bool:
        """Return True when model weights are present locally (no RAM resident)."""
        return self._is_ready

    def load(self) -> None:
        """Verify the model is cached locally. Does NOT load weights into RAM."""
        if self._is_ready:
            return
        if not self._mflux_available():
            raise RuntimeError(
                f"'{self.MFLUX_CMD}' not found in PATH or venv. "
                "Install with: uv pip install mflux"
            )
        # Ensure weights have been downloaded at least once
        self._ensure_weights_cached()
        self._is_ready = True
        logger.info("Flux2T2I service ready (weights cached, not resident)")

    def unload(self) -> None:
        """Nothing to unload in-process; kill any stray subprocess if needed."""
        self._kill_stray_mflux()
        gc.collect()
        logger.info("Flux2T2I service unloaded")

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 4,
        guidance_scale: float = 1.0,  # FLUX.2 klein is guidance-distilled; ignore
        seed: int | None = None,
    ) -> GeneratedImage:
        if not self._is_ready:
            raise RuntimeError("Flux2TextToImageService is not ready. Call load() first.")

        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")

        # Safety limits for 16 GB unified memory
        if width > 1024 or height > 1024:
            raise ValueError(
                f"Dimensions {width}x{height} exceed safe limit of 1024x1024"
            )
        if width == 1024 or height == 1024:
            logger.warning(
                "1024x1024 generation uses ~12 GB peak memory. "
                "Ensure no other heavy pipeline is loaded."
            )

        timestamp = int(time.time())
        seed_val = seed if seed is not None else 1337
        filename = f"flux2_{timestamp}_{seed_val}.png"
        output_path = self.outputs_dir / filename

        cmd = [
            sys.executable,
            "-m",
            "mflux",
            "generate",
            "flux2",
            "--model",
            self.MODEL_ID,
            "--prompt",
            prompt,
            "--width",
            str(width),
            "--height",
            str(height),
            "--steps",
            str(steps),
            "--seed",
            str(seed_val),
            "--output",
            str(output_path),
        ]

        if self.low_ram:
            cmd.append("--low-ram")
        if self.mlx_cache_limit_gb:
            cmd.extend(["--mlx-cache-limit-gb", str(self.mlx_cache_limit_gb)])

        env = os.environ.copy()
        if self.hf_home:
            env["HF_HOME"] = str(self.hf_home.resolve())

        logger.info("Running mflux generate: %s", " ".join(cmd))
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self._kill_stray_mflux()
            raise RuntimeError("mflux generation timed out after 600s") from exc

        duration = time.monotonic() - start

        if result.returncode != 0:
            logger.error("mflux stderr: %s", result.stderr)
            raise RuntimeError(
                f"mflux generation failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        if not output_path.exists():
            raise RuntimeError("mflux reported success but output file was not created")

        # Add metadata to PNG
        metadata = {
            "model": self.MODEL_ID,
            "prompt": prompt,
            "negative_prompt": negative_prompt or "",
            "width": width,
            "height": height,
            "steps": steps,
            "guidance_scale": guidance_scale,
            "seed": seed_val,
            "duration_seconds": round(duration, 3),
        }
        self._embed_metadata(output_path, metadata)

        return GeneratedImage(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed_val,
            duration_seconds=round(duration, 3),
            model_id=self.MODEL_ID,
            output_path=str(output_path.resolve()),
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_mflux(self) -> None:
        if not self._mflux_available():
            logger.warning(
                "'%s' not found. Install with: uv pip install mflux", self.MFLUX_CMD
            )

    def _mflux_available(self) -> bool:
        venv_bin = Path(sys.executable).parent
        cmd_path = venv_bin / self.MFLUX_CMD
        if cmd_path.exists():
            return True
        return shutil.which(self.MFLUX_CMD) is not None

    def _ensure_weights_cached(self) -> None:
        """Trigger a tiny generation so mflux downloads weights if missing."""
        # We just let the first real generation handle caching; user should run
        # download_models.sh beforehand.
        logger.info(
            "Assuming weights are cached at %s. Run scripts/download_models.sh to pre-download.",
            self.hf_home or "$HF_HOME",
        )

    def _kill_stray_mflux(self) -> None:
        """Kill any leftover mflux processes."""
        try:
            subprocess.run(
                ["pkill", "-f", self.MFLUX_CMD],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            pass

    def _embed_metadata(self, image_path: Path, metadata: dict) -> None:
        from PIL.PngImagePlugin import PngInfo

        img = Image.open(image_path)
        png_info = PngInfo()
        png_info.add_text("Description", str(metadata))
        img.save(image_path, pnginfo=png_info)
