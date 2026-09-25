"""Context-aware Russian text correction using a SAGE seq2seq model.

Unlike the dictionary corrector in :mod:`app.postprocess`, this model
(``ai-forever/sage-fredt5-distilled-95m``) understands context, so it fixes
OCR/spelling errors that a per-word dictionary cannot — e.g. "исли изти" ->
"Если идти" — while leaving valid words and short prepositions intact. It also
restores capitalization and punctuation.

Correction runs line-by-line (matching the recognizer's output structure) and
is fast on CPU (~0.1-0.2 s per line) because the distilled model is only ~95M
parameters. The model (~350 MB) downloads on first use.

Heavy dependencies (torch, transformers, sentencepiece) are optional and only
imported here. Install them with ``requirements-trocr.txt``.
"""

from __future__ import annotations

import os
from functools import lru_cache

# Quiet Hugging Face startup noise (progress bars, advisory token warning).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_NAME = "ai-forever/sage-fredt5-distilled-95m"
_MISSING_DEPS_MSG = (
    "Контекстный корректор требует дополнительных зависимостей "
    "(torch, transformers, sentencepiece). Установите их: "
    "pip install -r requirements-trocr.txt"
)


@lru_cache(maxsize=1)
def _load():
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from transformers.utils import logging as hf_logging
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    hf_logging.set_verbosity_error()  # silence advisory warnings during generate()
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - best-effort log quieting
        pass

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
    return tokenizer, model


def _correct_line(tokenizer, model, line: str) -> str:
    import torch

    inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=256)
    max_len = int(inputs["input_ids"].size(1) * 1.5) + 10
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=max_len, num_beams=4)
    # clean_up_tokenization_spaces=False: for BPE tokenizers the cleanup strips
    # spaces before punctuation and corrupts output.
    return tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def correct_text(text: str) -> str:
    """Return a context-corrected version of ``text`` (line by line)."""
    tokenizer, model = _load()
    corrected: list[str] = []
    for line in text.split("\n"):
        if not line.strip():
            corrected.append(line)
        else:
            corrected.append(_correct_line(tokenizer, model, line))
    return "\n".join(corrected)
