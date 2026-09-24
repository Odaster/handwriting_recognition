"""OCR core: preprocessing + Tesseract recognition for Russian text."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import pytesseract
from PIL import Image

DEFAULT_LANG = "rus"

# --oem 1 = LSTM engine, --psm 6 = assume a single uniform block of text.
DEFAULT_CONFIG = "--oem 1 --psm 6"


@dataclass
class Word:
    text: str
    confidence: Optional[float] = None


@dataclass
class RecognitionResult:
    text: str
    confidence: Optional[float] = None
    words: list[Word] = field(default_factory=list)
    engine: str = "tesseract"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "engine": self.engine,
            "confidence": round(self.confidence, 2) if self.confidence is not None else None,
            "words": [
                {
                    "text": w.text,
                    "confidence": round(w.confidence, 2) if w.confidence is not None else None,
                }
                for w in self.words
            ],
        }


def _preprocess(image: Image.Image) -> np.ndarray:
    """Convert to a clean binary image to give Tesseract the best chance.

    Small scans are upscaled because Tesseract works best around 300 DPI, and
    Otsu thresholding removes background noise from photographed pages.
    """
    arr = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    height, width = gray.shape[:2]
    scale = max(1.0, 1000.0 / max(height, width))
    if scale > 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    gray = cv2.medianBlur(gray, 3)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def recognize_image(
    image: Image.Image,
    lang: str = DEFAULT_LANG,
    config: str = DEFAULT_CONFIG,
) -> RecognitionResult:
    """Run Tesseract on a PIL image and return text + per-word confidence."""
    processed = _preprocess(image)

    data = pytesseract.image_to_data(
        processed,
        lang=lang,
        config=config,
        output_type=pytesseract.Output.DICT,
    )

    words: list[Word] = []
    confidences: list[float] = []
    for raw_text, raw_conf in zip(data["text"], data["conf"]):
        text = (raw_text or "").strip()
        try:
            conf = float(raw_conf)
        except (TypeError, ValueError):
            conf = -1.0
        if text and conf >= 0:
            words.append(Word(text=text, confidence=conf))
            confidences.append(conf)

    full_text = pytesseract.image_to_string(processed, lang=lang, config=config).strip()
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return RecognitionResult(text=full_text, confidence=avg_conf, words=words)


def recognize_bytes(data: bytes, lang: str = DEFAULT_LANG) -> RecognitionResult:
    """Decode raw image bytes and recognize them."""
    image = Image.open(io.BytesIO(data))
    return recognize_image(image, lang=lang)


def tesseract_info() -> dict:
    """Return the installed Tesseract version and available languages."""
    return {
        "tesseract_version": str(pytesseract.get_tesseract_version()),
        "languages": sorted(pytesseract.get_languages(config="")),
    }
