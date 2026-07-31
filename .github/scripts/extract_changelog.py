#!/usr/bin/env python3
"""Extract changelog section for a specific version.

Usage:
    python extract_changelog.py v0.2.0 CHANGELOG.md

Outputs the section text (or empty string if not found).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


def extract_changelog_section(version: str, changelog_path: Path) -> str:
    """Extract the section for `version` from CHANGELOG.md.

    The version in CHANGELOG should be like `## [0.2.0] — YYYY-MM-DD`.
    We strip the leading "v" if present for matching.
    """
    version = version.lstrip("v")
    text = changelog_path.read_text(encoding="utf-8")

    # Find the section header for this version
    # Match: ## [0.2.0] — 2026-07-31
    header_pattern = rf"##\s*\[{re.escape(version)}\].*?\n"
    match = re.search(header_pattern, text)
    if not match:
        return ""

    start = match.start()
    # Find the next ## [X.Y.Z] or ## [Unreleased]
    remaining = text[start + len(match.group(0)):]
    next_section = re.search(r"##\s*\[", remaining)
    if next_section:
        section = remaining[:next_section.start()].strip()
    else:
        section = remaining.strip()

    return section


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: extract_changelog.py <version> <changelog.md>", file=sys.stderr)
        return 1

    version = sys.argv[1]
    changelog_path = Path(sys.argv[2])

    if not changelog_path.exists():
        print(f"File not found: {changelog_path}", file=sys.stderr)
        return 1

    section = extract_changelog_section(version, changelog_path)
    if section:
        print(section)
        return 0
    else:
        print(f"No section found for version {version}", file=sys.stderr)
        return 0  # Return 0 so workflow continues with fallback


if __name__ == "__main__":
    sys.exit(main())
