"""Text-to-Image backend: FLUX.2 via mflux on Apple Silicon (MLX)."""
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
from services.backends.base import TextToImageBackend

logger = logging.getLogger(__name__)


class AppleFlux2Backend(TextToImageBackend):
    """FLUX.2 [klein] 4B int4 via mflux + MLX.

    Requirements:
      - macOS
      - Apple Silicon (M1/M2/M3/M4+)
      - mflux package installed
    """

    MODEL_ID = "mlx-community/FLUX.2-Klein-4B-4bit"
    CMD = "mflux-generate-flux2"

    def __init__(
        self,
        outputs_dir: str = "outputs/text_to_image",
        hf_home: str | None = None,
        low_ram: bool = True,
        mlx_cache_limit_gb: float = 8.0,
    ) -> None:
        self.outputs_dir = Path(outputs_dir)
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        self.hf_home = Path(hf_home) if hf_home else None
        self.low_ram = low_ram
        self.mlx_cache_limit_gb = mlx_cache_limit_gb
        self._ready = False

    @staticmethod
    def is_available() -> bool:
        try:
            import platform as plat
            if plat.system() != "Darwin":
                return False
            import subprocess as sp
            result = sp.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return "Apple" in result.stdout
        except Exception:
            return False

    def load(self) -> None:
        if self._ready:
            return
        if not shutil.which(self.CMD):
            venv_bin = Path(sys.executable).parent / self.CMD
            if not venv_bin.exists():
                raise RuntimeError(
                    f"'{self.CMD}' not found. Install: uv pip install mflux"
                )
        self._ready = True
        logger.info("AppleFlux2Backend ready (weights cached on first generation)")

    def unload(self) -> None:
        self._ready = False
        try:
            subprocess.run(
                ["pkill", "-f", self.CMD],
                capture_output=True,
                timeout=5,
                check=False,
            )
        except Exception:
            pass
        gc.collect()
        logger.info("AppleFlux2Backend unloaded")

    def is_loaded(self) -> bool:
        return self._ready

    def generate(
        self,
        prompt: str,
        negative_prompt: str | None = None,
        width: int = 512,
        height: int = 512,
        steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int | None = None,
    ) -> GeneratedImage:
        if not self._ready:
            raise RuntimeError("Backend not loaded")
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError("Width and height must be multiples of 64")
        if width > 1024 or height > 1024:
            raise ValueError("Max safe resolution on 16 GB: 1024x1024")

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

        start = time.monotonic()
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, env=env
        )
        duration = time.monotonic() - start

        if result.returncode != 0:
            raise RuntimeError(f"mflux failed: {result.stderr[:500]}")
        if not output_path.exists():
            raise RuntimeError("mflux succeeded but no output file")

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
        from PIL.PngImagePlugin import PngInfo

        img = Image.open(output_path)
        png_info = PngInfo()
        png_info.add_text("Description", str(metadata))
        img.save(output_path, pnginfo=png_info)

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
