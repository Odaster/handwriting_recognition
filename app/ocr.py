"""Обёртка над OCR-движком."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from app.config import OCR_LANGUAGES, OCR_USE_GPU


@dataclass
class RecognizedLine:
    text: str
    confidence: float
    box: list[list[float]] | None = None


@dataclass
class RecognitionResult:
    text: str
    lines: list[RecognizedLine] = field(default_factory=list)
    engine: str = "unknown"
    languages: list[str] = field(default_factory=lambda: list(OCR_LANGUAGES))

    @property
    def average_confidence(self) -> float | None:
        if not self.lines:
            return None
        return sum(line.confidence for line in self.lines) / len(self.lines)


def sort_detections(detections: list[tuple]) -> list[tuple]:
    """Сортирует блоки EasyOCR сверху вниз, затем слева направо."""

    def key(item: tuple) -> tuple[float, float]:
        bbox = item[0]
        ys = [point[1] for point in bbox]
        xs = [point[0] for point in bbox]
        return (float(min(ys)), float(min(xs)))

    return sorted(detections, key=key)


def detections_to_result(
    detections: list[tuple],
    *,
    engine: str,
    languages: list[str],
) -> RecognitionResult:
    ordered = sort_detections(detections)
    lines: list[RecognizedLine] = []
    for bbox, text, confidence in ordered:
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        box = [[float(x), float(y)] for x, y in bbox] if bbox is not None else None
        lines.append(
            RecognizedLine(
                text=cleaned,
                confidence=float(max(0.0, min(1.0, confidence))),
                box=box,
            )
        )
    text = "\n".join(line.text for line in lines)
    return RecognitionResult(text=text, lines=lines, engine=engine, languages=languages)


class OCREngine(ABC):
    name: str = "base"

    @abstractmethod
    def recognize(self, image: np.ndarray) -> RecognitionResult:
        raise NotImplementedError

    def warmup(self) -> None:
        return None


class FakeOCREngine(OCREngine):
    """Движок для тестов: не грузит модели."""

    name = "fake"

    def __init__(self, text: str = "Пример распознанного текста", confidence: float = 0.99):
        self._text = text
        self._confidence = confidence

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        if image is None or image.size == 0:
            return RecognitionResult(text="", engine=self.name, languages=list(OCR_LANGUAGES))
        lines = [
            RecognizedLine(text=part, confidence=self._confidence)
            for part in self._text.split("\n")
            if part.strip()
        ]
        return RecognitionResult(
            text=self._text,
            lines=lines,
            engine=self.name,
            languages=list(OCR_LANGUAGES),
        )


class EasyOCREngine(OCREngine):
    name = "easyocr"

    def __init__(self, languages: list[str] | None = None, gpu: bool | None = None):
        self.languages = languages or list(OCR_LANGUAGES)
        self.gpu = OCR_USE_GPU if gpu is None else gpu
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self._reader

    def warmup(self) -> None:
        self._get_reader()

    def recognize(self, image: np.ndarray) -> RecognitionResult:
        reader = self._get_reader()
        detections = reader.readtext(image)
        return detections_to_result(
            detections,
            engine=self.name,
            languages=self.languages,
        )


def create_engine(backend: str | None = None) -> OCREngine:
    from app.config import OCR_BACKEND

    name = (backend or OCR_BACKEND).strip().lower()
    if name in {"fake", "mock", "test"}:
        return FakeOCREngine()
    if name in {"easyocr", "easy", "real", "default"}:
        return EasyOCREngine()
    raise ValueError(f"Неизвестный OCR-движок: {name}")
