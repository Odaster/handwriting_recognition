#!/usr/bin/env python3
"""Remove superseded Hugging Face model caches from disk.

The app now defaults to SAGE 1.7B (``ai-forever/sage-v1.1.0`` + tokenizer
``ai-forever/FRED-T5-1.7B``). The previous default ``sage-fredt5-large``
(~3.3 GB) is unused and can be deleted.

Keeps distilled-95m (tests) and the 1.7B checkpoint/tokenizer.

Usage (from repo root):

    python scripts/cleanup_old_hf_cache.py
    python scripts/cleanup_old_hf_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

# Hugging Face hub folder names: models--org--name
REMOVE = (
    "models--ai-forever--sage-fredt5-large",
)
KEEP_HINTS = (
    "models--ai-forever--sage-v1.1.0",
    "models--ai-forever--FRED-T5-1.7B",
    "models--ai-forever--sage-fredt5-distilled-95m",
)


def _hub_dirs() -> list[Path]:
    dirs: list[Path] = []
    if os.environ.get("HF_HUB_CACHE"):
        dirs.append(Path(os.environ["HF_HUB_CACHE"]))
    hf_home = Path(os.environ.get("HF_HOME") or (Path.home() / ".cache" / "huggingface"))
    dirs.append(hf_home / "hub")
    old = os.environ.get("TRANSFORMERS_CACHE") or os.environ.get("HF_HOME")
    if old:
        dirs.append(Path(old))
    # Windows user cache + Linux-style under the same profile
    dirs.append(Path.home() / ".cache" / "huggingface" / "hub")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in dirs:
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def _size(path: Path) -> int:
    total = 0
    if path.is_file():
        return path.stat().st_size
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _fmt(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{num} B"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without removing anything",
    )
    args = parser.parse_args()

    found = []
    for hub in _hub_dirs():
        if not hub.is_dir():
            continue
        for name in REMOVE:
            target = hub / name
            if target.exists():
                found.append(target)

    if not found:
        print("Старый кэш sage-fredt5-large не найден — удалять нечего.")
        print("Проверенные каталоги:")
        for hub in _hub_dirs():
            print(f"  {hub}  ({'есть' if hub.is_dir() else 'нет'})")
        print("Оставлены (не трогаем): " + ", ".join(KEEP_HINTS))
        return 0

    for target in found:
        size = _size(target)
        action = "DRY-RUN, оставил бы" if args.dry_run else "Удаляю"
        print(f"{action}: {target}  ({_fmt(size)})")
        if not args.dry_run:
            shutil.rmtree(target, ignore_errors=False)
            print(f"  готово, освобождено ~{_fmt(size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
