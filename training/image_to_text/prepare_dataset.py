"""Prepare and validate image-to-text datasets for fine-tuning."""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "webp"}
MAX_FILE_SIZE_MB = 10
MAX_IMAGE_PIXELS = 2_097_152


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _validate_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "File not found"
    ext = path.suffix.lstrip(".").lower()
    if ext not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {ext}"
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return False, f"Too large: {size_mb:.1f} MB"
    try:
        img = Image.open(path)
        img.verify()
    except Exception as exc:
        return False, f"Corrupt image: {exc}"
    try:
        img = Image.open(path)
        w, h = img.size
        if w * h > MAX_IMAGE_PIXELS:
            return False, f"Too many pixels: {w*h} > {MAX_IMAGE_PIXELS}"
    except Exception as exc:
        return False, f"Cannot read dimensions: {exc}"
    return True, ""


def _validate_record(record: dict, images_dir: Path) -> tuple[bool, str]:
    image_rel = record.get("image")
    if not image_rel:
        return False, "Missing 'image' field"
    image_path = images_dir / image_rel
    ok, msg = _validate_image(image_path)
    if not ok:
        return False, msg
    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) < 2:
        return False, "Missing or invalid 'messages' list"
    return True, ""


def _deduplicate(records: list[dict], images_dir: Path) -> list[dict]:
    seen_hashes: set[str] = set()
    unique: list[dict] = []
    for rec in records:
        img_path = images_dir / rec.get("image", "")
        if not img_path.exists():
            continue
        h = _file_hash(img_path)
        if h in seen_hashes:
            logger.warning("Duplicate image removed: %s", img_path)
            continue
        seen_hashes.add(h)
        unique.append(rec)
    return unique


def prepare_dataset(
    input_jsonl: Path,
    images_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> dict[str, str]:
    """Read, validate, deduplicate, and split an image-to-text dataset."""
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    invalid_count = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Line %s: JSON parse error: %s", line_num, exc)
                invalid_count += 1
                continue
            ok, msg = _validate_record(rec, images_dir)
            if not ok:
                logger.warning("Line %s: Invalid record: %s", line_num, msg)
                invalid_count += 1
                continue
            records.append(rec)

    records = _deduplicate(records, images_dir)
    total = len(records)
    if total == 0:
        raise ValueError("No valid records found after validation")

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    train_recs = records[:train_end]
    val_recs = records[train_end:val_end]
    test_recs = records[val_end:]

    def _write_split(name: str, data: list[dict]) -> Path:
        split_dir = output_dir / name
        split_dir.mkdir(parents=True, exist_ok=True)
        img_dir = split_dir / "images"
        img_dir.mkdir(exist_ok=True)
        out_jsonl = split_dir / "dataset.jsonl"
        with open(out_jsonl, "w", encoding="utf-8") as f:
            for rec in data:
                src = images_dir / rec["image"]
                dst = img_dir / Path(rec["image"]).name
                if not dst.exists():
                    shutil.copy2(src, dst)
                rec_copy = dict(rec)
                rec_copy["image"] = "images/" + dst.name
                f.write(json.dumps(rec_copy, ensure_ascii=False) + "\n")
        return out_jsonl

    train_path = _write_split("train", train_recs)
    val_path = _write_split("val", val_recs)
    test_path = _write_split("test", test_recs)

    logger.info(
        "Dataset prepared: %s total, %s train, %s val, %s test, %s invalid",
        total,
        len(train_recs),
        len(val_recs),
        len(test_recs),
        invalid_count,
    )
    return {
        "train": str(train_path),
        "val": str(val_path),
        "test": str(test_path),
    }


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("datasets/image_to_text/splits"))
    args = parser.parse_args()
    prepare_dataset(args.input_jsonl, args.images_dir, args.output_dir)


if __name__ == "__main__":
    main()
