"""Chandra OCR 2 engine — page-level document VLM (``datalab-to/chandra-ocr-2``).

Chandra 2 (Datalab) is a ~5B Qwen3.5-VL model that reads a whole page at once —
no line/word segmentation and no separate corrector needed. It handles cursive
handwriting, layout, tables and 90+ languages (incl. Russian) and tops the
olmOCR benchmark.

Hardware: local inference needs a **CUDA GPU with >=16 GB VRAM** (bf16 weights
~10-12 GB) and 16+ GB RAM. On CPU-only hosts the model will not load. Enable with:

    pip install -r requirements-chandra.txt   # installs chandra-ocr[hf]

Prompt mode via CHANDRA_PROMPT: "ocr" (plain text, default) or "ocr_layout"
(layout-preserving markdown).
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

from PIL import Image

from app.ocr import RecognitionResult

PROMPT_TYPE = os.environ.get("CHANDRA_PROMPT", "ocr")
_MISSING_DEPS_MSG = (
    "Движок Chandra требует зависимостей и GPU (>=16 ГБ VRAM). Установите: "
    "pip install -r requirements-chandra.txt"
)


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from chandra.model.hf import load_model
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    # Fail fast when torch cannot see a CUDA GPU, instead of downloading ~10 GB
    # and hitting OOM. The most common cause is a CPU-only torch build.
    if not torch.cuda.is_available():
        build = getattr(torch, "__version__", "?")
        is_cpu_build = "+cpu" in build or "+cu" not in build
        hint = (
            "Похоже, установлена CPU-сборка torch. Переустановите CUDA-версию:\n"
            "  python -m pip uninstall -y torch torchvision\n"
            "  python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121"
            if is_cpu_build
            else "Проверьте драйвер NVIDIA и совместимость CUDA."
        )
        raise RuntimeError(
            f"torch не видит CUDA-GPU (torch.cuda.is_available() == False; сборка torch: {build}). "
            f"{hint}\n"
            "Chandra OCR 2 (~5B) требует CUDA-GPU (для 5B в bf16 нужно ~10–12 ГБ VRAM). "
            "Альтернатива без GPU: движок 'trocr' (CPU) или хостируемый Datalab API (datalab.to)."
        )
    return load_model()


def recognize_image(image: Image.Image) -> RecognitionResult:
    """Recognize a full page with Chandra OCR 2 (page-level, layout-aware)."""
    try:
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    model = _load_model()
    item = BatchInputItem(image=image.convert("RGB"), prompt=None, prompt_type=PROMPT_TYPE)
    result = generate_hf([item], model)[0]

    if getattr(result, "error", None):
        raise RuntimeError(f"Chandra error: {result.error}")

    raw = result.raw or ""
    text = parse_markdown(raw) if PROMPT_TYPE == "ocr_layout" else raw
    return RecognitionResult(text=text.strip(), confidence=None, words=[], engine="chandra")


def recognize_bytes(data: bytes) -> RecognitionResult:
    return recognize_image(Image.open(io.BytesIO(data)))
