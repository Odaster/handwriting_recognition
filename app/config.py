"""Настройки приложения."""

from __future__ import annotations

import os

APP_NAME = "Распознавание рукописного текста"
APP_VERSION = "0.1.0"


def _env(*names: str, default: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return default


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _max_upload_bytes() -> int:
    raw_bytes = os.getenv("MAX_UPLOAD_BYTES")
    if raw_bytes:
        return int(raw_bytes)
    return int(float(_env("MAX_UPLOAD_MB", default="10")) * 1024 * 1024)


MAX_UPLOAD_BYTES = _max_upload_bytes()
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/tiff",
    "application/octet-stream",
}

OCR_LANGUAGES = [
    part.strip()
    for part in _env("OCR_LANGUAGES", default="ru,en").split(",")
    if part.strip()
]
OCR_BACKEND = _env("OCR_BACKEND", "OCR_ENGINE", default="easyocr").strip().lower()
OCR_USE_GPU = _as_bool(_env("OCR_USE_GPU", "OCR_GPU", default="0"))
