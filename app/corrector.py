"""Context-aware Russian text correction using a SAGE seq2seq model.

Unlike the dictionary corrector in :mod:`app.postprocess`, this model
understands context, so it fixes OCR/spelling errors that a per-word dictionary
cannot — e.g. "исли изти" -> "Если идти" — while leaving valid words and short
prepositions intact. It also restores capitalization and punctuation.

The default ``ai-forever/sage-fredt5-large`` (~820M params, ~3.3 GB) favors
quality; set ``SAGE_MODEL=ai-forever/sage-fredt5-distilled-95m`` for a much
smaller/faster model. Correction runs line-by-line (matching the recognizer's
output structure). The model downloads on first use.

Heavy dependencies (torch, transformers, sentencepiece) are optional and only
imported here. Install them with ``requirements-trocr.txt``.
"""

from __future__ import annotations

import os
import re
from functools import lru_cache

# Quiet Hugging Face advisory noise, but KEEP download progress bars so the
# multi-GB model download on first run is visible in the terminal.
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

# Large model favors quality; override with SAGE_MODEL for a faster/smaller one
# (e.g. ai-forever/sage-fredt5-distilled-95m).
MODEL_NAME = os.environ.get("SAGE_MODEL", "ai-forever/sage-fredt5-large")
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
    return tokenizer, model


# Text content between HTML tags (skips tags and their attributes, so layout
# markup like data-bbox coordinates from Chandra is never fed to the corrector).
_HTML_TEXT = re.compile(r"(?<=>)([^<>]+)(?=<)")
_MAX_WORDS = 50  # chunk long segments to avoid truncating the model's output


def _sage(tokenizer, model, text: str) -> str:
    import torch

    # No input truncation: we chunk beforehand so nothing is silently cut.
    inputs = tokenizer(text, return_tensors="pt")
    max_len = min(1024, int(inputs["input_ids"].size(1) * 1.6) + 16)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=max_len, num_beams=4)
    # clean_up_tokenization_spaces=False: for BPE tokenizers the cleanup strips
    # spaces before punctuation and corrupts output.
    return tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _correct_segment(tokenizer, model, segment: str) -> str:
    if not segment.strip():
        return segment
    words = segment.split(" ")
    if len(words) <= _MAX_WORDS:
        return _sage(tokenizer, model, segment)
    # Long segment: correct in word chunks and rejoin (prevents truncation).
    out = [
        _sage(tokenizer, model, " ".join(words[i : i + _MAX_WORDS]))
        for i in range(0, len(words), _MAX_WORDS)
    ]
    return " ".join(out)


def correct_text(text: str) -> str:
    """Return a context-corrected version of ``text``.

    For HTML/markup output (e.g. Chandra's layout HTML) only the visible text
    nodes are corrected — tags and attributes (``data-bbox`` coordinates) are
    preserved untouched. Plain text is corrected line by line. Long segments are
    chunked so the output is never truncated.
    """
    tokenizer, model = _load()

    if "<" in text and ">" in text:
        return _HTML_TEXT.sub(
            lambda m: _correct_segment(tokenizer, model, m.group(1)), text
        )

    return "\n".join(
        _correct_segment(tokenizer, model, line) for line in text.split("\n")
    )
