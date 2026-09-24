"""End-to-end OCR tests: render Russian text, then recognize it back."""

from __future__ import annotations

from PIL import Image

from app.ocr import recognize_image, tesseract_info
from scripts.make_sample import make_image


def test_russian_language_available():
    info = tesseract_info()
    assert "rus" in info["languages"], f"Russian model missing: {info}"


def test_recognize_single_line(tmp_path):
    out = make_image("Привет мир", tmp_path / "line.png", font_size=52)
    result = recognize_image(Image.open(out))
    text = result.text.lower()
    assert "привет" in text
    assert "мир" in text
    assert result.confidence > 0


def test_recognize_multi_line(tmp_path):
    out = make_image("Москва\nроссия", tmp_path / "multi.png", font_size=52)
    result = recognize_image(Image.open(out))
    text = result.text.lower()
    assert "москва" in text
    assert "россия" in text
    assert len(result.words) >= 2
