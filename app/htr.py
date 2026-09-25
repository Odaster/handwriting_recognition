"""Handwriting text recognition (HTR) engine.

Tesseract cannot read cursive handwriting, so this module uses a Russian
handwriting TrOCR model (``kazars24/trocr-base-handwritten-ru``). The key trick
is segmentation: TrOCR resizes every input to 384x384, so feeding a full page —
or even a full line — squashes the text and destroys accuracy. Instead we split
the page into lines, then each line into words, and recognize word-by-word.

The heavy dependencies (torch, transformers, scipy) are optional and only
imported here, so the rest of the app runs without them. Install them with:

    pip install -r requirements-trocr.txt
"""

from __future__ import annotations

import io
import os
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from app.ocr import RecognitionResult, Word

# Quiet Hugging Face startup noise (progress bars, advisory token warning).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

MODEL_NAME = "kazars24/trocr-base-handwritten-ru"
_MISSING_DEPS_MSG = (
    "Рукописный движок требует дополнительных зависимостей "
    "(torch, transformers, scipy). Установите их: "
    "pip install -r requirements-trocr.txt"
)


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch  # noqa: F401
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        from transformers.utils import logging as hf_logging
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    hf_logging.set_verbosity_error()  # silence advisory warnings during generate()
    try:
        from huggingface_hub.utils import logging as hub_logging

        hub_logging.set_verbosity_error()
    except Exception:  # noqa: BLE001 - best-effort log quieting
        pass

    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME).eval()
    # Drive length purely via max_new_tokens; clearing max_length avoids the
    # "both max_new_tokens and max_length are set" warning on every batch.
    model.generation_config.max_length = None
    return processor, model


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Illumination-corrected binarization (text -> white on black)."""
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=35)
    normalized = cv2.divide(gray, background, scale=255)
    thr = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return cv2.medianBlur(thr, 3)


def segment_lines(thr: np.ndarray) -> list[tuple[int, int]]:
    """Split a binary page into horizontal line bands.

    Uses a smoothed horizontal ink projection; line centers are peaks and cuts
    are placed at the valleys between them. This tolerates cursive ascenders and
    descenders that would fool a simple "gap between lines" heuristic.
    """
    try:
        from scipy.signal import find_peaks
    except ImportError as exc:
        raise RuntimeError(_MISSING_DEPS_MSG) from exc

    height = thr.shape[0]
    projection = thr.sum(axis=1).astype(float) / 255.0
    smoothed = np.convolve(projection, np.ones(15) / 15, mode="same")
    if smoothed.max() <= 0:
        return [(0, height)]

    peaks, _ = find_peaks(smoothed, distance=38, prominence=smoothed.max() * 0.06)
    if len(peaks) == 0:
        return [(0, height)]

    cuts = [0]
    for i in range(len(peaks) - 1):
        a, b = peaks[i], peaks[i + 1]
        cuts.append(a + int(np.argmin(smoothed[a:b])))
    cuts.append(height)

    return [
        (cuts[i], cuts[i + 1])
        for i in range(len(cuts) - 1)
        if cuts[i + 1] - cuts[i] >= 20
    ]


def segment_words(line_thr: np.ndarray, min_gap: int | None = None, min_width: int | None = None) -> list[tuple[int, int]]:
    """Split one line band into word bounding ranges via vertical ink gaps.

    Thresholds scale with the line height so the same logic works across
    resolutions (a high-res photo and a small rendered sample alike).
    """
    line_height = line_thr.shape[0]
    if min_gap is None:
        min_gap = max(12, int(0.30 * line_height))
    if min_width is None:
        min_width = max(8, int(0.20 * line_height))

    column = line_thr.sum(axis=0).astype(float) / 255.0
    column = np.convolve(column, np.ones(5) / 5, mode="same")
    mask = column > 0.4

    words: list[list[int]] = []
    start = None
    for x in range(len(mask)):
        if mask[x] and start is None:
            start = x
        elif not mask[x] and start is not None:
            words.append([start, x])
            start = None
    if start is not None:
        words.append([start, len(mask)])

    merged: list[list[int]] = []
    for w in words:
        if merged and w[0] - merged[-1][1] < min_gap:
            merged[-1][1] = w[1]
        else:
            merged.append(w)

    return [(a, b) for a, b in merged if b - a >= min_width]


def recognize_image(image: Image.Image, batch_size: int = 16) -> RecognitionResult:
    """Recognize handwritten Russian text on a full page image."""
    import torch

    processor, model = _load_model()

    rgb = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thr = _binarize(gray)
    height, width = gray.shape

    crops: list[Image.Image] = []
    line_of: list[int] = []
    bands = segment_lines(thr)
    for line_index, (y0, y1) in enumerate(bands):
        top = max(0, y0 - 6)
        bottom = min(height, y1 + 6)
        for x0, x1 in segment_words(thr[y0:y1, :]):
            crop = rgb[top:bottom, max(0, x0 - 6):min(width, x1 + 6)]
            crops.append(Image.fromarray(crop))
            line_of.append(line_index)

    words: list[str] = []
    for i in range(0, len(crops), batch_size):
        pixel_values = processor(images=crops[i:i + batch_size], return_tensors="pt").pixel_values
        with torch.no_grad():
            generated = model.generate(pixel_values, max_new_tokens=48, num_beams=1)
        words.extend(
            processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        )

    lines_out: list[list[str]] = [[] for _ in bands]
    word_objs: list[Word] = []
    for text, line_index in zip(words, line_of):
        text = text.strip()
        if text:
            lines_out[line_index].append(text)
            word_objs.append(Word(text=text))

    line_strings = [" ".join(line) for line in lines_out if line]
    full_text = "\n".join(line_strings)
    return RecognitionResult(text=full_text, confidence=None, words=word_objs, engine="trocr")


def recognize_bytes(data: bytes, batch_size: int = 16) -> RecognitionResult:
    image = Image.open(io.BytesIO(data))
    return recognize_image(image, batch_size=batch_size)
