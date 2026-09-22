from __future__ import annotations

import numpy as np
import pytest

from app.ocr import FakeOCREngine, create_engine, detections_to_result, sort_detections


def test_sort_detections_top_to_bottom():
    lower = ([[0, 40], [10, 40], [10, 50], [0, 50]], "низ", 0.8)
    upper = ([[5, 2], [20, 2], [20, 12], [5, 12]], "верх", 0.9)
    ordered = sort_detections([lower, upper])
    assert ordered[0][1] == "верх"
    assert ordered[1][1] == "низ"


def test_detections_to_result_joins_lines():
    detections = [
        ([[0, 30], [8, 30], [8, 40], [0, 40]], "мир", 0.7),
        ([[0, 0], [8, 0], [8, 10], [0, 10]], "Привет", 1.2),
        ([[0, 50], [8, 50], [8, 60], [0, 60]], "   ", 0.9),
    ]
    result = detections_to_result(detections, engine="easyocr", languages=["ru"])
    assert result.text == "Привет\nмир"
    assert result.lines[0].confidence == 1.0
    assert result.average_confidence == pytest.approx(0.85)


def test_fake_engine_splits_lines():
    engine = FakeOCREngine(text="одна\nдве")
    result = engine.recognize(np.zeros((10, 10, 3), dtype=np.uint8))
    assert result.engine == "fake"
    assert [line.text for line in result.lines] == ["одна", "две"]


def test_create_engine_fake(monkeypatch):
    monkeypatch.setenv("OCR_BACKEND", "easyocr")
    engine = create_engine("fake")
    assert engine.name == "fake"


def test_create_engine_unknown():
    with pytest.raises(ValueError, match="Неизвестный"):
        create_engine("tesseract-from-mars")
