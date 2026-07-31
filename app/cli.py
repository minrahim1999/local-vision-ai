"""CLI for local-vision-ai.

Usage:
    lvai generate "A cat on a sofa" --width 512 --height 512
    lvai analyze photo.jpg --prompt "Describe this image"
    lvai extract photo.jpg --prompt "Extract objects as JSON"
    lvai health
    lvai backend
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
import requests

API_BASE = os.getenv("LVAI_API_URL", "http://localhost:8000")

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("lvai")


@click.group()
@click.option("--api-url", default=API_BASE, help="API base URL")
@click.pass_context
def cli(ctx: click.Context, api_url: str) -> None:
    """Local Vision AI — command-line interface."""
    ctx.ensure_object(dict)
    ctx.obj["api_url"] = api_url.rstrip("/")


@cli.command()
@click.argument("prompt")
@click.option("--width", default=512, help="Image width (multiple of 64, max 1024)")
@click.option("--height", default=512, help="Image height (multiple of 64, max 1024)")
@click.option("--steps", default=4, help="Number of diffusion steps")
@click.option("--guidance-scale", default=1.0, help="Guidance scale")
@click.option("--seed", default=None, type=int, help="Random seed")
@click.option("--negative-prompt", default=None, help="Negative prompt")
@click.option("--output", default=None, help="Output file path (PNG)")
@click.option("--no-server", is_flag=True, help="Run directly via backend instead of API")
@click.pass_context
def generate(
    ctx: click.Context,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    negative_prompt: str | None,
    output: str | None,
    no_server: bool,
) -> None:
    """Generate an image from a text prompt."""
    if no_server:
        _generate_direct(prompt, width, height, steps, guidance_scale, seed, negative_prompt, output)
    else:
        _generate_via_api(ctx.obj["api_url"], prompt, width, height, steps, guidance_scale, seed, negative_prompt, output)


def _generate_direct(
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    negative_prompt: str | None,
    output: str | None,
) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from services.backends.factory import create_t2i_backend

    backend = create_t2i_backend({"hf_home": "./models/huggingface", "low_ram": True})
    logger.info("Loading model...")
    backend.load()
    try:
        logger.info("Generating: %s", prompt)
        result = backend.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
        )
        click.echo(f"Generated: {result.output_path}")
        click.echo(f"Duration: {result.duration_seconds:.2f}s | Model: {result.model_id}")
        if output:
            import shutil
            shutil.copy(result.output_path, output)
            click.echo(f"Copied to: {output}")
    finally:
        backend.unload()


def _generate_via_api(
    api_url: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: int | None,
    negative_prompt: str | None,
    output: str | None,
) -> None:
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "negative_prompt": negative_prompt,
    }
    payload = {k: v for k, v in payload.items() if v is not None}

    click.echo(f"POST {api_url}/v1/images/generate")
    r = requests.post(f"{api_url}/v1/images/generate", json=payload, timeout=300)
    if r.status_code != 200:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)

    data = r.json()
    click.echo(f"Generated: {data['output_path']}")
    click.echo(f"Duration: {data['duration_seconds']:.2f}s | Model: {data['model_id']}")
    if output and "output_path" in data:
        import shutil
        shutil.copy(data["output_path"], output)
        click.echo(f"Copied to: {output}")


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--prompt", default="Describe this image in detail", help="Analysis prompt")
@click.option("--format", "fmt", default="text", type=click.Choice(["text", "json"]))
@click.option("--max-tokens", default=512, help="Max tokens to generate")
@click.option("--no-server", is_flag=True, help="Run directly via backend")
@click.pass_context
def analyze(
    ctx: click.Context,
    image_path: str,
    prompt: str,
    fmt: str,
    max_tokens: int,
    no_server: bool,
) -> None:
    """Analyze an image with a prompt."""
    if no_server:
        _analyze_direct(image_path, prompt, fmt, max_tokens)
    else:
        _analyze_via_api(ctx.obj["api_url"], image_path, prompt, fmt, max_tokens)


def _analyze_direct(image_path: str, prompt: str, fmt: str, max_tokens: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from services.backends.factory import create_i2t_backend

    backend = create_i2t_backend({"model_id": "mlx-community/SmolVLM-256M-Instruct-4bit"})
    logger.info("Loading model...")
    backend.load()
    try:
        logger.info("Analyzing: %s", image_path)
        result = backend.analyze(image_path=image_path, prompt=prompt, response_format=fmt, max_tokens=max_tokens)
        click.echo(result.content if hasattr(result, "content") else result.response)
    finally:
        backend.unload()


def _analyze_via_api(api_url: str, image_path: str, prompt: str, fmt: str, max_tokens: int) -> None:
    click.echo(f"POST {api_url}/v1/vision/analyze")
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f)}
        data = {"prompt": prompt, "response_format": fmt, "max_tokens": max_tokens}
        r = requests.post(f"{api_url}/v1/vision/analyze", files=files, data=data, timeout=120)
    if r.status_code != 200:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)
    click.echo(r.json()["response"])


@cli.command()
@click.argument("image_path", type=click.Path(exists=True))
@click.option("--prompt", default="Extract structured data as JSON", help="Extraction prompt")
@click.option("--max-tokens", default=512, help="Max tokens to generate")
@click.option("--no-server", is_flag=True, help="Run directly via backend")
@click.pass_context
def extract(
    ctx: click.Context,
    image_path: str,
    prompt: str,
    max_tokens: int,
    no_server: bool,
) -> None:
    """Extract structured JSON from an image."""
    if no_server:
        _extract_direct(image_path, prompt, max_tokens)
    else:
        _extract_via_api(ctx.obj["api_url"], image_path, prompt, max_tokens)


def _extract_direct(image_path: str, prompt: str, max_tokens: int) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from services.backends.factory import create_i2t_backend

    backend = create_i2t_backend({"model_id": "mlx-community/SmolVLM-256M-Instruct-4bit"})
    logger.info("Loading model...")
    backend.load()
    try:
        logger.info("Extracting from: %s", image_path)
        result = backend.extract(image_path=image_path, prompt=prompt, max_tokens=max_tokens)
        content = result.content if hasattr(result, "content") else result.response
        try:
            parsed = json.loads(content)
            click.echo(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            click.echo(content)
    finally:
        backend.unload()


def _extract_via_api(api_url: str, image_path: str, prompt: str, max_tokens: int) -> None:
    click.echo(f"POST {api_url}/v1/vision/extract")
    with open(image_path, "rb") as f:
        files = {"image": (os.path.basename(image_path), f)}
        data = {"prompt": prompt, "max_tokens": max_tokens}
        r = requests.post(f"{api_url}/v1/vision/extract", files=files, data=data, timeout=120)
    if r.status_code != 200:
        click.echo(f"Error {r.status_code}: {r.text}", err=True)
        sys.exit(1)
    content = r.json()["response"]
    try:
        parsed = json.loads(content)
        click.echo(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        click.echo(content)


@cli.command()
@click.pass_context
def health(ctx: click.Context) -> None:
    """Check API health."""
    api_url = ctx.obj["api_url"]
    try:
        r = requests.get(f"{api_url}/health", timeout=10)
        click.echo(json.dumps(r.json(), indent=2))
    except requests.ConnectionError:
        click.echo("API not reachable. Start with: make run", err=True)
        sys.exit(1)


@cli.command()
def backend() -> None:
    """Show which backend would be auto-selected."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from services.backends.factory import T2I_BACKENDS, I2T_BACKENDS
    click.echo("Text-to-Image backends:")
    for cls in T2I_BACKENDS:
        status = "✅ Available" if cls.is_available() else "❌ Not available"
        click.echo(f"  {cls.__name__:30s} {status}")
    click.echo("Image-to-Text backends:")
    for cls in I2T_BACKENDS:
        status = "✅ Available" if cls.is_available() else "❌ Not available"
        click.echo(f"  {cls.__name__:30s} {status}")


if __name__ == "__main__":
    cli()
