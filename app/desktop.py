"""Desktop GUI for Local Vision AI using Flet.

Usage:
    uv run python -m app.desktop
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import flet as ft
import requests

API_BASE = os.getenv("LVAI_API_URL", "http://localhost:8000")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


class LvaiDesktopApp:
    def __init__(self, page: ft.Page) -> None:
        self.page = page
        self.page.title = "Local Vision AI"
        self.page.theme_mode = ft.ThemeMode.SYSTEM
        self.page.window_width = 1200
        self.page.window_height = 800
        self.page.padding = 24
        # Set window icon (works on Windows/Linux; macOS uses .app icon)
        self.page.window_icon = str(Path(__file__).resolve().parent.parent / "assets" / "icons" / "icon_256.png")

        self._build_ui()
        self._check_api()

    def _build_ui(self) -> None:
        # --- Tabs ---
        self.tabs = ft.Tabs(
            tabs=[
                ft.Tab(text="🎨 Generate Image", content=self._build_generate_tab()),
                ft.Tab(text="🔍 Analyze Image", content=self._build_analyze_tab()),
                ft.Tab(text="📊 System Status", content=self._build_status_tab()),
            ],
            expand=True,
        )
        self.page.add(self.tabs)

    def _build_generate_tab(self) -> ft.Column:
        self.gen_prompt = ft.TextField(label="Prompt", multiline=True, min_lines=3, max_lines=6)
        self.gen_negative = ft.TextField(label="Negative Prompt (optional)", multiline=True, min_lines=2)
        self.gen_width = ft.TextField(label="Width", value="512", width=100)
        self.gen_height = ft.TextField(label="Height", value="512", width=100)
        self.gen_steps = ft.TextField(label="Steps", value="4", width=100)
        self.gen_seed = ft.TextField(label="Seed (optional)", value="", width=120)
        self.gen_status = ft.Text(value="Ready", color=ft.colors.GREEN)
        self.gen_image = ft.Image(src=None, width=512, height=512, fit=ft.ImageFit.CONTAIN)

        gen_btn = ft.ElevatedButton(
            "Generate Image",
            icon=ft.icons.BRUSH,
            on_click=self._on_generate,
            style=ft.ButtonStyle(bgcolor=ft.colors.BLUE, color=ft.colors.WHITE),
        )

        return ft.Column(
            [
                ft.Row([self.gen_prompt], expand=True),
                ft.Row([self.gen_negative], expand=True),
                ft.Row([self.gen_width, self.gen_height, self.gen_steps, self.gen_seed, gen_btn]),
                ft.Row([self.gen_status], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([self.gen_image], alignment=ft.MainAxisAlignment.CENTER, expand=True),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _build_analyze_tab(self) -> ft.Column:
        self.anal_file_path = ft.TextField(label="Image Path", read_only=True, expand=True)
        self.anal_browse = ft.ElevatedButton("Browse", icon=ft.icons.FOLDER_OPEN, on_click=self._on_browse)
        self.anal_prompt = ft.TextField(label="Prompt", value="Describe this image in detail", multiline=True)
        self.anal_format = ft.Dropdown(
            label="Output Format",
            options=[ft.dropdown.Option("text"), ft.dropdown.Option("json")],
            value="text",
            width=150,
        )
        self.anal_result = ft.TextField(label="Result", multiline=True, min_lines=10, read_only=True, expand=True)
        self.anal_status = ft.Text(value="Ready", color=ft.colors.GREEN)

        anal_btn = ft.ElevatedButton(
            "Analyze",
            icon=ft.icons.VISIBILITY,
            on_click=self._on_analyze,
            style=ft.ButtonStyle(bgcolor=ft.colors.BLUE, color=ft.colors.WHITE),
        )

        return ft.Column(
            [
                ft.Row([self.anal_file_path, self.anal_browse]),
                ft.Row([self.anal_prompt], expand=True),
                ft.Row([self.anal_format, anal_btn]),
                ft.Row([self.anal_status], alignment=ft.MainAxisAlignment.CENTER),
                ft.Row([self.anal_result], expand=True),
            ],
            scroll=ft.ScrollMode.AUTO,
            expand=True,
        )

    def _build_status_tab(self) -> ft.Column:
        self.status_text = ft.TextField(label="System Status", multiline=True, min_lines=20, read_only=True, expand=True)
        refresh_btn = ft.ElevatedButton("Refresh", icon=ft.icons.REFRESH, on_click=self._refresh_status)

        return ft.Column(
            [
                ft.Row([refresh_btn]),
                ft.Row([self.status_text], expand=True),
            ],
            expand=True,
        )

    def _check_api(self) -> None:
        def check():
            try:
                r = requests.get(f"{API_BASE}/health", timeout=5)
                if r.status_code == 200:
                    self._safe_snack("API connected ✅")
                else:
                    self._safe_snack("API returned error", alert=True)
            except Exception:
                self._safe_snack("API not running. Start with: make run", alert=True)
        threading.Thread(target=check, daemon=True).start()

    def _safe_snack(self, msg: str, alert: bool = False) -> None:
        def fn():
            self.page.snack_bar = ft.SnackBar(
                ft.Text(msg),
                bgcolor=ft.colors.RED if alert else ft.colors.GREEN,
            )
            self.page.snack_bar.open = True
            self.page.update()
        self.page.run_thread(fn)

    def _on_generate(self, e: ft.ControlEvent) -> None:
        self.gen_status.value = "Generating..."
        self.gen_status.color = ft.colors.ORANGE
        self.page.update()

        def generate():
            try:
                payload = {
                    "prompt": self.gen_prompt.value or "a cat",
                    "width": int(self.gen_width.value or 512),
                    "height": int(self.gen_height.value or 512),
                    "steps": int(self.gen_steps.value or 4),
                    "guidance_scale": 1.0,
                }
                seed_val = self.gen_seed.value
                if seed_val:
                    payload["seed"] = int(seed_val)
                if self.gen_negative.value:
                    payload["negative_prompt"] = self.gen_negative.value

                r = requests.post(f"{API_BASE}/v1/images/generate", json=payload, timeout=300)
                if r.status_code != 200:
                    self._safe_snack(f"Error {r.status_code}: {r.text[:200]}", alert=True)
                    self.gen_status.value = "Failed"
                    self.gen_status.color = ft.colors.RED
                    self.page.update()
                    return

                data = r.json()
                output_path = data["output_path"]
                self.gen_image.src = output_path
                self.gen_status.value = f"Done in {data['duration_seconds']:.1f}s"
                self.gen_status.color = ft.colors.GREEN
                self.page.update()
            except Exception as exc:
                self._safe_snack(f"Exception: {exc}", alert=True)
                self.gen_status.value = "Failed"
                self.gen_status.color = ft.colors.RED
                self.page.update()

        threading.Thread(target=generate, daemon=True).start()

    def _on_browse(self, e: ft.ControlEvent) -> None:
        def on_result(dialog: ft.FilePickerResultEvent):
            if dialog.files:
                self.anal_file_path.value = dialog.files[0].path
                self.page.update()

        picker = ft.FilePicker(on_result=on_result)
        self.page.overlay.append(picker)
        self.page.update()
        picker.pick_files(allowed_extensions=["png", "jpg", "jpeg", "webp"])

    def _on_analyze(self, e: ft.ControlEvent) -> None:
        self.anal_status.value = "Analyzing..."
        self.anal_status.color = ft.colors.ORANGE
        self.page.update()

        def analyze():
            try:
                path = self.anal_file_path.value
                if not path:
                    self._safe_snack("Select an image first", alert=True)
                    self.anal_status.value = "Failed"
                    self.anal_status.color = ft.colors.RED
                    self.page.update()
                    return

                with open(path, "rb") as f:
                    files = {"image": (os.path.basename(path), f)}
                    data = {
                        "prompt": self.anal_prompt.value or "Describe this image",
                        "response_format": self.anal_format.value,
                        "max_tokens": "512",
                    }
                    r = requests.post(f"{API_BASE}/v1/vision/analyze", files=files, data=data, timeout=120)

                if r.status_code != 200:
                    self._safe_snack(f"Error {r.status_code}: {r.text[:200]}", alert=True)
                    self.anal_status.value = "Failed"
                    self.anal_status.color = ft.colors.RED
                    self.page.update()
                    return

                result = r.json()
                self.anal_result.value = result.get("response", result.get("content", "No response"))
                self.anal_status.value = "Done"
                self.anal_status.color = ft.colors.GREEN
                self.page.update()
            except Exception as exc:
                self._safe_snack(f"Exception: {exc}", alert=True)
                self.anal_status.value = "Failed"
                self.anal_status.color = ft.colors.RED
                self.page.update()

        threading.Thread(target=analyze, daemon=True).start()

    def _refresh_status(self, e: ft.ControlEvent) -> None:
        self.status_text.value = "Refreshing..."
        self.page.update()

        def refresh():
            try:
                lines = []
                # Health
                try:
                    r = requests.get(f"{API_BASE}/health", timeout=10)
                    lines.append(f"Health: {r.json()}")
                except Exception as exc:
                    lines.append(f"Health: unreachable ({exc})")

                # Memory
                try:
                    r = requests.get(f"{API_BASE}/v1/system/memory", timeout=10)
                    d = r.json()
                    lines.append(f"\nMemory:")
                    lines.append(f"  RSS: {d.get('rss_mb', 0)} MB")
                    lines.append(f"  VMS: {d.get('vms_mb', 0)} MB")
                    lines.append(f"  System %: {d.get('system_percent', 0)}%")
                    lines.append(f"  Swap Used: {d.get('swap_used_mb', 0)} MB")
                except Exception as exc:
                    lines.append(f"\nMemory: unavailable ({exc})")

                # Models
                try:
                    r = requests.get(f"{API_BASE}/v1/models/status", timeout=10)
                    d = r.json()
                    lines.append(f"\nModels:")
                    for k, v in d.get("pipelines", {}).items():
                        lines.append(f"  {k}: {'Loaded' if v else 'Unloaded'}")
                except Exception as exc:
                    lines.append(f"\nModels: unavailable ({exc})")

                self.status_text.value = "\n".join(lines)
                self.page.update()
            except Exception as exc:
                self.status_text.value = f"Error: {exc}"
                self.page.update()

        threading.Thread(target=refresh, daemon=True).start()


def main(page: ft.Page) -> None:
    LvaiDesktopApp(page)


if __name__ == "__main__":
    ft.app(target=main)
