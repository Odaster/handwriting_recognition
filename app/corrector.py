"""Context-aware Russian text correction using a SAGE seq2seq model.

Unlike the dictionary corrector in :mod:`app.postprocess`, this model
understands context, so it fixes OCR/spelling errors that a per-word dictionary
cannot — e.g. "исли изти" -> "Если идти" — while leaving valid words and short
prepositions intact. It also restores capitalization and punctuation.

The default ``ai-forever/sage-v1.1.0`` is the 1.7B FRED-T5 SAGE corrector
(~7 GB ``model.safetensors``). There is no Hub repo named ``sage-fredt5-1.7b``:
the published 1.7B checkpoint is ``sage-v1.1.0``, with tokenizer
``ai-forever/FRED-T5-1.7B`` (tokenizer files only — do not snapshot the whole
FRED-T5 base, it is ~20 GB of unused weights).

SAGE and Chandra share a GPU exclusively: loading one evicts the other from
VRAM via :mod:`app.models`.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

DEFAULT_MODEL = "ai-forever/sage-v1.1.0"
_TOKENIZER_DEFAULTS = {
    "ai-forever/sage-v1.1.0": "ai-forever/FRED-T5-1.7B",
}
_LM_PREFIX_MODELS = {"ai-forever/sage-v1.1.0"}
_MIN_WEIGHT_BYTES = {
    "ai-forever/sage-v1.1.0": 4_000_000_000,  # ~7 GB fp32 safetensors
    "ai-forever/sage-fredt5-large": 1_000_000_000,
    "ai-forever/sage-fredt5-distilled-95m": 50_000_000,
}
_MISSING_DEPS_MSG = (
    "Контекстный корректор требует дополнительных зависимостей "
    "(torch, transformers, sentencepiece). Установите их: "
    "pip install -r requirements-trocr.txt"
)
log = logging.getLogger(__name__)


def _announce(message: str) -> None:
    """Print to stdout (uvicorn) and the app logger — progress must be visible."""
    print(message, flush=True)
    log.info(message)


def _model_ids() -> tuple[str, str]:
    model_name = os.environ.get("SAGE_MODEL", DEFAULT_MODEL)
    tokenizer_name = os.environ.get("SAGE_TOKENIZER") or _TOKENIZER_DEFAULTS.get(
        model_name, model_name
    )
    return model_name, tokenizer_name


# Resolved at import for docs/health; actual load always re-reads the env.
MODEL_NAME, TOKENIZER_NAME = _model_ids()


def _checkpoint_bytes(folder: Path) -> int:
    total = 0
    for pattern in ("model.safetensors", "pytorch_model.bin", "model-*.safetensors"):
        for path in folder.glob(pattern):
            if path.is_file():
                total += path.stat().st_size
    return total


def _download_weights(repo_id: str) -> str:
    """Download the seq2seq checkpoint to disk and refuse a config-only load."""
    from huggingface_hub import snapshot_download

    min_bytes = _MIN_WEIGHT_BYTES.get(repo_id, 50_000_000)
    _announce(
        f"SAGE: fetching weights {repo_id} "
        f"(need ≥ {min_bytes / 1e9:.1f} GB; Hugging Face progress bar follows)"
    )
    folder = Path(
        snapshot_download(
            repo_id=repo_id,
            # sage-v1.1.0 also ships a duplicate pytorch_model.bin (~7 GB extra).
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "model.safetensors.index.json",
                "model-*.safetensors",
                "*.json",
            ],
            ignore_patterns=["pytorch_model.bin", "pytorch_model*.bin", "images/*"],
        )
    )
    size = _checkpoint_bytes(folder)
    if size < min_bytes:
        raise RuntimeError(
            f"SAGE checkpoint {repo_id} on disk is only {size / 1e6:.1f} MB "
            f"(expected ≥ {min_bytes / 1e9:.1f} GB) at {folder}. "
            "Download incomplete or the wrong repo was resolved. "
            "Delete the broken cache folder and retry."
        )
    _announce(f"SAGE: {repo_id} weights {size / 1e9:.2f} GB at {folder}")
    return str(folder)


def _loader():
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    model_name, tokenizer_name = _model_ids()
    _announce(
        f"SAGE: SAGE_MODEL={model_name}  SAGE_TOKENIZER={tokenizer_name}  "
        f"(corrector.py={Path(__file__).resolve()})"
    )

    tok_kwargs = {}
    if "FRED-T5" in tokenizer_name:
        tok_kwargs["eos_token"] = "</s>"
    _announce(f"SAGE: loading tokenizer {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, **tok_kwargs)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    local = _download_weights(model_name)
    _announce(f"SAGE: loading {model_name} from local snapshot (not random init)")
    model = AutoModelForSeq2SeqLM.from_pretrained(local, local_files_only=True).eval()
    n_params = sum(p.numel() for p in model.parameters())
    min_params = 1_200_000_000 if model_name == DEFAULT_MODEL else 1
    if n_params < min_params:
        raise RuntimeError(
            f"SAGE {model_name} has {n_params / 1e6:.0f}M parameters, "
            f"expected ~1.7B. Refusing to run a truncated/random checkpoint."
        )
    _announce(f"SAGE: {n_params / 1e9:.2f}B parameters ready")
    try:
        if torch.cuda.is_available():
            model = model.to("cuda")
            _announce("SAGE: moved to CUDA")
        else:
            _announce("SAGE: CUDA unavailable, staying on CPU")
    except Exception:
        log.exception("Could not move SAGE to CUDA; staying on current device")
    return tokenizer, model


def _load():
    from app import models

    return models.load("sage", _loader)


_HTML_TEXT = re.compile(r"(?<=>)([^<>]+)(?=<)")
_MAX_WORDS = 50


def _sage(tokenizer, model, text: str) -> str:
    import torch

    model_name, _ = _model_ids()
    payload = ("<LM>" + text) if model_name in _LM_PREFIX_MODELS else text
    inputs = tokenizer(
        payload,
        return_tensors="pt",
        padding="longest",
        truncation=False,
    )
    device = getattr(model, "device", None)
    if device is not None:
        inputs = inputs.to(device)
    max_len = min(1024, int(inputs["input_ids"].size(1) * 1.5) + 8)
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=max_len, use_cache=True)
    return tokenizer.batch_decode(
        generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def _correct_segment(tokenizer, model, segment: str) -> str:
    if not segment.strip():
        return segment
    words = segment.split(" ")
    if len(words) <= _MAX_WORDS:
        return _sage(tokenizer, model, segment)
    out = [
        _sage(tokenizer, model, " ".join(words[i : i + _MAX_WORDS]))
        for i in range(0, len(words), _MAX_WORDS)
    ]
    return " ".join(out)


def correct_text(text: str) -> str:
    """Return a context-corrected version of ``text``.

    For HTML/markup output only visible text nodes are corrected. Plain text is
    corrected line by line. Long segments are chunked so the output is never
    truncated.
    """
    tokenizer, model = _load()

    if "<" in text and ">" in text:
        return _HTML_TEXT.sub(
            lambda m: _correct_segment(tokenizer, model, m.group(1)), text
        )

    return "\n".join(
        _correct_segment(tokenizer, model, line) for line in text.split("\n")
    )
