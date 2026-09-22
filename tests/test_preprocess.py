from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from app.config import MAX_UPLOAD_BYTES
from app.preprocess import (
    ImageValidationError,
    enhance_for_ocr,
    load_image,
    prepare_image,
    validate_upload,
)


def _png(size=(24, 16), color=(12, 80, 20)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, color).save(buffer, format="PNG")
    return buffer.getvalue()


def test_validate_upload_empty():
    with pytest.raises(ImageValidationError, match="Пустой"):
        validate_upload("a.png", "image/png", b"")


def test_validate_upload_too_large():
    data = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(ImageValidationError, match="большой"):
        validate_upload("a.png", "image/png", data)


def test_validate_upload_bad_extension():
    with pytest.raises(ImageValidationError, match="формат"):
        validate_upload("notes.pdf", "application/pdf", b"%PDF-fake")


def test_load_image_reads_png():
    array = load_image(_png())
    assert array.shape[2] == 3
    assert array.shape[0] == 16
    assert array.shape[1] == 24


def test_load_image_rejects_garbage():
    with pytest.raises(ImageValidationError, match="прочитать"):
        load_image(b"zzzz")


def test_enhance_for_ocr_returns_rgb():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :15] = 30
    image[:, 15:] = 200
    out = enhance_for_ocr(image)
    assert out.shape == image.shape
    assert out.dtype == np.uint8


def test_prepare_image_can_skip_enhance():
    raw = load_image(_png(color=(255, 0, 0)))
    prepared = prepare_image(_png(color=(255, 0, 0)), enhance=False)
    assert np.array_equal(raw, prepared)
