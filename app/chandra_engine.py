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


_NO_GPU_MSG = (
    "Chandra OCR 2 (~5B) требует CUDA-GPU с >=16 ГБ VRAM, а на этой машине GPU не найден. "
    "Запустите на GPU-хосте, либо используйте движок 'trocr' (работает на CPU), "
    "либо хостируемый Datalab API/playground (datalab.to)."
)


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        from chandra.model.hf import load_model
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    # Fail fast on CPU-only hosts instead of downloading ~10 GB and hitting OOM.
    if not torch.cuda.is_available():
        raise RuntimeError(_NO_GPU_MSG)
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
