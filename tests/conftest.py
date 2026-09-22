from __future__ import annotations

from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import create_app
from app.ocr import FakeOCREngine


def make_png_bytes(size: tuple[int, int] = (80, 40), color: tuple[int, int, int] = (255, 255, 255)) -> bytes:
    image = Image.new("RGB", size, color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def engine() -> FakeOCREngine:
    return FakeOCREngine(text="Привет, мир\nвторая строка", confidence=0.91)


@pytest.fixture
def client(engine: FakeOCREngine) -> TestClient:
    return TestClient(create_app(engine=engine))
