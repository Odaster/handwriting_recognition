"""Подготовка изображения перед OCR."""

from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.config import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    MAX_UPLOAD_BYTES,
    OCR_MAX_SIDE,
)


class ImageValidationError(ValueError):
    """Файл нельзя использовать для распознавания."""


def validate_upload(filename: str | None, content_type: str | None, data: bytes) -> None:
    if not data:
        raise ImageValidationError("Пустой файл.")
    if len(data) > MAX_UPLOAD_BYTES:
        max_mb = MAX_UPLOAD_BYTES / (1024 * 1024)
        raise ImageValidationError(f"Файл слишком большой. Максимум {max_mb:.0f} МБ.")

    suffix = ""
    if filename and "." in filename:
        suffix = "." + filename.rsplit(".", 1)[-1].lower()
    type_ok = (content_type or "").lower() in ALLOWED_CONTENT_TYPES
    ext_ok = suffix in ALLOWED_EXTENSIONS if suffix else False
    if suffix and not ext_ok and not type_ok:
        raise ImageValidationError(
            "Неподдерживаемый формат. Загрузите JPG, PNG, WEBP, BMP или TIFF."
        )


def load_image(data: bytes) -> np.ndarray:
    """Читает байты изображения в RGB-массив numpy."""
    try:
        with Image.open(BytesIO(data)) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("RGB")
            array = np.array(image)
    except Exception as exc:  # noqa: BLE001 — PIL кидает разные ошибки на битых файлах
        raise ImageValidationError("Не удалось прочитать изображение.") from exc

    if array.size == 0:
        raise ImageValidationError("Изображение пустое.")
    return downscale(array, OCR_MAX_SIDE)


def downscale(image: np.ndarray, max_side: int = OCR_MAX_SIDE) -> np.ndarray:
    """Уменьшает огромные сканы, чтобы не исчерпать память на CPU."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return image
    scale = max_side / longest
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
    """Повышает контраст и убирает шум, не превращая фото в жёсткую бинаризацию.

    EasyOCR лучше работает с серым/цветным входом, чем с агрессивным threshold.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    gray = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)


def prepare_image(data: bytes, enhance: bool = True) -> np.ndarray:
    image = load_image(data)
    if enhance:
        return enhance_for_ocr(image)
    return image
