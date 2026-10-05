"""Chandra OCR 2 engine — page-level document VLM (``datalab-to/chandra-ocr-2``).

Chandra 2 (Datalab) is a ~5B Qwen3.5-VL model that reads a whole page at once.
On a 12 GB card Chandra and the corrector run **one at a time**.

Env:
    CHANDRA_PROMPT   ocr (Russian handwriting, default) | ocr_layout | generic
    CHANDRA_ENHANCE  1 (default) | 0
    CHANDRA_STRIPS   horizontal bands, default 3 (more pixels per letter)
    CHANDRA_REFINE   0 (default) | 1 — optional second pass; often invents fluent headings
    CHANDRA_MAX_TOKENS  per-band cap, default 1536 with strips else 2048
"""

from __future__ import annotations

import io
import logging
import os

from PIL import Image, ImageOps

import numpy as np

from app.ocr import RecognitionResult
from app.textout import html_to_text, line_similarity

log = logging.getLogger(__name__)

PROMPT_TYPE = os.environ.get("CHANDRA_PROMPT", "ocr")
_MISSING_DEPS_MSG = (
    "Движок Chandra требует зависимостей и GPU (>=16 ГБ VRAM). Установите: "
    "pip install -r requirements-chandra.txt"
)

# Chandra's stock prompt is English document-HTML. School cursive needs a
# tighter instruction or the model spends tokens on layout chrome and red marks.
RU_HANDWRITING_PROMPT = """
OCR this photograph of a Russian handwritten notebook page to HTML.

The page is Russian cursive (Cyrillic). Transcribe the student's handwriting.
Do not translate.

Guidelines:
* Join wrapped lines into paragraphs with <p>...</p>. Use <br> only for titles.
* Blue or black ink is the main text. Ignore faint show-through from the other side.
* Ignore red teacher marks, interlinear corrections, grades, and margin scores.
  Do not output LaTeX, \\frac, arrays, or equations for those marks.
* Keep original wording and names.
* If a word is unreadable, give a letter-level best effort. Do not replace it
  with a real dictionary word, a synonym, or a fluent invented phrase.
* Use only these tags: p, br, div, h1, h2, h3, i, b, u.
""".strip()


def _refine_prompt(draft: str) -> str:
    clip = draft.strip()[:1800]
    return f"""{RU_HANDWRITING_PROMPT}

A first OCR draft of this page (errors likely):

{clip}

Copy that draft. Change a word only when the handwriting on the image clearly shows a different word.
Do not rewrite sentences. Do not add adjectives. Do not make the prose more literary.
Do not turn an unreadable heading into a grammatical sentence.
Do not output LaTeX, arrays, or margin grades.
""".strip()


def _loader():
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
    log.info("Loading Chandra OCR 2 onto GPU")
    return load_model()


def _load_model():
    # Routed through the model manager so loading Chandra evicts the corrector from VRAM
    # (and vice versa) — they never occupy the GPU at the same time.
    from app import models

    return models.load("chandra", _loader)


def _env_on(name: str, default: str = "1") -> bool:
    return os.environ.get(name, default).strip().lower() not in {"0", "false", "no", "off"}


def _item_prompt() -> tuple[str | None, str]:
    mode = os.environ.get("CHANDRA_PROMPT", "ocr").strip().lower()
    if mode == "ocr_layout":
        return None, "ocr_layout"
    if mode in {"generic", "stock", "en"}:
        return None, "ocr"
    return RU_HANDWRITING_PROMPT, "ocr"


def _row_ink(image: Image.Image) -> np.ndarray:
    gray = np.array(image.convert("L"), dtype=np.float32)
    paper = float(np.percentile(gray, 90))
    profile = np.clip(paper - gray, 0, None).mean(axis=1)
    k = max(5, image.size[1] // 220)
    if k % 2 == 0:
        k += 1
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(profile, kernel, mode="same")


def _valley_ys(profile: np.ndarray) -> list[int]:
    peak = float(np.max(profile)) or 1.0
    limit = 0.32 * peak
    raw: list[int] = []
    height = len(profile)
    for y in range(1, height - 1):
        if profile[y] <= profile[y - 1] and profile[y] <= profile[y + 1] and profile[y] <= limit:
            raw.append(y)
    if not raw:
        return []
    groups = [[raw[0]]]
    for y in raw[1:]:
        if y - groups[-1][-1] <= 12:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [min(group, key=lambda yy: profile[yy]) for group in groups]


def _line_aware_cuts(image: Image.Image, n: int) -> list[int]:
    """n-1 y-cuts snapped to gaps between notebook lines, not through glyphs."""
    height = image.size[1]
    geometric = [int(height * i / n) for i in range(1, n)]
    profile = _row_ink(image)
    valleys = _valley_ys(profile)
    if not valleys:
        return geometric
    slack = max(80, height // 8)
    cuts: list[int] = []
    for target in geometric:
        nearby = [v for v in valleys if abs(v - target) <= slack]
        chosen = min(nearby, key=lambda v: abs(v - target)) if nearby else target
        if cuts and chosen <= cuts[-1] + 40:
            chosen = target
        cuts.append(int(chosen))
    return cuts


def page_bands(image: Image.Image, n: int, overlap: float = 0.28) -> list[Image.Image]:
    """Split a tall page into overlapping horizontal bands.

    Chandra downscales the *whole* page to ~3072×2048. Bands keep more pixels
    per glyph. Cuts snap to inter-line gaps so a word is not sliced in half.
    """
    n = max(1, int(n))
    width, height = image.size
    if n == 1 or height < 1400:
        return [image]
    cuts = _line_aware_cuts(image, n)
    pad = max(48, int(height * overlap / n))
    edges = [0, *cuts, height]
    bands: list[Image.Image] = []
    for i in range(n):
        y0 = max(0, edges[i] - (pad if i else 0))
        y1 = min(height, edges[i + 1] + (pad if i < n - 1 else 0))
        if y1 - y0 < 80:
            continue
        bands.append(image.crop((0, y0, width, y1)))
    return bands or [image]


def _overlap_score(prev: list[str], nxt: list[str], k: int) -> tuple[float, float]:
    """Return (min, mean) similarity of prev[-k:] aligned with nxt[:k]."""
    if k <= 0:
        return 0.0, 0.0
    scores = [line_similarity(prev[-k + i], nxt[i]) for i in range(k)]
    return min(scores), sum(scores) / len(scores)


def stitch_band_texts(parts: list[str]) -> str:
    """Concatenate strip OCR, dropping duplicated overlap lines (fuzzy)."""
    chunks: list[str] = []
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        if not chunks:
            chunks.append(part)
            continue
        prev = chunks[-1].splitlines()
        nxt = part.splitlines()
        overlap = 0
        max_k = min(6, len(prev), len(nxt))
        for k in range(max_k, 0, -1):
            min_s, mean_s = _overlap_score(prev, nxt, k)
            if min_s >= 0.5 and mean_s >= 0.62:
                overlap = k
                break
        rest = nxt[overlap:]
        while rest and not rest[0].strip():
            rest = rest[1:]
        if rest:
            chunks.append("\n".join(rest))
    return html_to_text("\n\n".join(chunks).strip())


class _ProgressStreamer:
    """Minimal HuggingFace streamer that reports generated-token progress."""

    def __init__(self, progress, estimate: int, offset: float, span: float, token_base: int = 0):
        self._progress = progress
        self._estimate = max(32, estimate)
        self._offset = offset
        self._span = span
        self._token_base = token_base
        self._count = 0
        self._skipped_prompt = False

    def put(self, value):  # noqa: ANN001 - HF streamer protocol
        if not self._skipped_prompt:
            self._skipped_prompt = True
            return
        self._count += 1
        fraction = min(0.98, self._offset + self._span * (self._count / self._estimate))
        self._progress(fraction, "recognize", token_count=self._token_base + self._count)

    def end(self):  # noqa: ANN001 - HF streamer protocol
        pass


def _prepare_image(image: Image.Image) -> Image.Image:
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    if not _env_on("CHANDRA_ENHANCE", "1"):
        return rgb
    from app.enhance import enhance_page

    return enhance_page(rgb)


def _default_max_tokens(n_bands: int) -> str:
    # Layout HTML (data-bbox on every field) burns tokens fast. 512 cut a
    # vacation form mid-tag: ``<div data-bbox="462 403 646 477``.
    return "1536" if n_bands > 1 else "2048"


def _prefer_refine(draft: str, refined: str) -> str:
    """Keep the draft if refine collapsed, and keep the draft heading if it drifted."""
    if not refined.strip():
        return draft
    draft_n = max(1, len(draft.split()))
    if len(refined.split()) < max(12, int(0.45 * draft_n)):
        log.warning(
            "chandra refine too short (%s vs %s words); keeping draft",
            len(refined.split()),
            draft_n,
        )
        return draft
    d_lines = draft.splitlines()
    r_lines = refined.splitlines()
    d0 = next((ln for ln in d_lines if ln.strip()), "")
    r0 = next((ln for ln in r_lines if ln.strip()), "")
    if not d0 or line_similarity(d0, r0) >= 0.62:
        return refined
    log.info("chandra refine diverged on heading; splicing draft header")
    cut_d = cut_r = None
    for i, dl in enumerate(d_lines):
        if i < 2 or not dl.strip():
            continue
        for j, rl in enumerate(r_lines):
            if j < 2 or not rl.strip():
                continue
            if line_similarity(dl, rl) >= 0.7:
                cut_d, cut_r = i, j
                break
        if cut_d is not None:
            break
    if cut_d is None:
        return draft
    return "\n".join(d_lines[:cut_d] + r_lines[cut_r:]).strip()


def recognize_image(image: Image.Image, progress=None) -> RecognitionResult:
    """Recognize a full page with Chandra OCR 2 (page-level, layout-aware)."""
    try:
        from chandra.model.hf import generate_hf
        from chandra.model.schema import BatchInputItem
        from chandra.output import parse_markdown
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    if progress:
        progress(0.03, "load")
    model = _load_model()
    prepared = _prepare_image(image)
    n_strips = max(1, int(os.environ.get("CHANDRA_STRIPS", "3")))
    bands = page_bands(prepared, n_strips)
    do_refine = _env_on("CHANDRA_REFINE", "0") and bool(bands)
    max_tokens = int(os.environ.get("CHANDRA_MAX_TOKENS", _default_max_tokens(len(bands))))
    prompt, prompt_type = _item_prompt()
    n_passes = len(bands) + (1 if do_refine else 0)
    log.info(
        "chandra: size=%sx%s enhance=%s strips=%s refine=%s tokens=%s prompt=%s",
        prepared.size[0],
        prepared.size[1],
        _env_on("CHANDRA_ENHANCE", "1"),
        len(bands),
        do_refine,
        max_tokens,
        "ru" if prompt else prompt_type,
    )
    if progress:
        progress(0.05, "recognize")

    orig_generate = getattr(model, "generate", None)
    texts: list[str] = []
    tokens_seen = 0

    def _one(pil_image, item_prompt, ptype, estimate, offset, span):
        nonlocal tokens_seen
        streamer = None
        if progress:
            streamer = _ProgressStreamer(
                progress,
                estimate=estimate,
                offset=offset,
                span=span,
                token_base=tokens_seen,
            )
        if orig_generate is not None:

            def _generate(*args, _orig=orig_generate, _streamer=streamer, **kwargs):
                kwargs.setdefault("no_repeat_ngram_size", 12)
                kwargs.setdefault("repetition_penalty", 1.12)
                if _streamer is not None:
                    kwargs.setdefault("streamer", _streamer)
                return _orig(*args, **kwargs)

            model.generate = _generate
        item = BatchInputItem(image=pil_image, prompt=item_prompt, prompt_type=ptype)
        result = generate_hf([item], model, max_output_tokens=estimate)[0]
        if getattr(result, "error", None):
            raise RuntimeError(f"Chandra error: {result.error}")
        raw = result.raw or ""
        if ptype == "ocr_layout":
            raw = parse_markdown(raw)
        tokens_seen += int(getattr(result, "token_count", 0) or 0)
        return html_to_text(raw)

    try:
        span = 0.68 / max(1, n_passes) if do_refine else 0.9 / max(1, len(bands))
        for index, band in enumerate(bands):
            texts.append(
                _one(band, prompt, prompt_type, max_tokens, 0.05 + span * index, span)
            )
        text = stitch_band_texts(texts)
        if do_refine and text.strip():
            refine_tokens = int(os.environ.get("CHANDRA_MAX_TOKENS", _default_max_tokens(1)))
            refined = _one(
                prepared,
                _refine_prompt(text),
                "ocr",
                refine_tokens,
                0.05 + span * len(bands),
                0.25,
            )
            text = _prefer_refine(text, refined)
    finally:
        if orig_generate is not None:
            model.generate = orig_generate

    return RecognitionResult(text=text, confidence=None, words=[], engine="chandra")


def recognize_bytes(data: bytes, progress=None) -> RecognitionResult:
    return recognize_image(Image.open(io.BytesIO(data)), progress=progress)
