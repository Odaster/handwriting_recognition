"""Prepare a photographed notebook page for a document VLM.

Generic denoise / HDR models smear thin ballpoint strokes. Chandra is a
vision-language model, so we also must not binarize. The useful cheap steps
for school-notebook photos are: upright the page, mute teacher's red ink,
drop ruled lines, lift ink-vs-paper contrast, mild unsharp.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger(__name__)


def enhance_page(image: Image.Image) -> Image.Image:
    """Return an RGB PIL image. On any OpenCV failure, return the original RGB."""
    rgb = ImageOps.exif_transpose(image).convert("RGB")
    bgr = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    try:
        bgr = _deskew(bgr)
        bgr = _mute_red(bgr)
        bgr = _drop_ruling(bgr)
        bgr = _lift_ink(bgr)
        bgr = _unsharp(bgr)
    except Exception:
        log.exception("enhance_page failed; using original photo")
        return rgb
    out = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(out)


def _hough_segments(lines: np.ndarray) -> np.ndarray:
    """Normalize HoughLinesP output: OpenCV may return (N, 1, 4) or (N, 4)."""
    return np.asarray(lines, dtype=np.float64).reshape(-1, 4)


def _deskew(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 0.5 if min(h, w) > 800 else 1.0
    small = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 50, 150)
    min_len = max(40, small.shape[1] // 4)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=80, minLineLength=min_len, maxLineGap=20
    )
    if lines is None:
        return bgr
    angles: list[float] = []
    for x1, y1, x2, y2 in _hough_segments(lines):
        if abs(int(x2) - int(x1)) < 8:
            continue
        ang = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(ang) < 12:
            angles.append(ang)
    if len(angles) < 8:
        return bgr
    angle = float(np.median(angles))
    if abs(angle) < 0.4:
        return bgr
    hh, ww = bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
    return cv2.warpAffine(
        bgr, matrix, (ww, hh), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def _mute_red(bgr: np.ndarray) -> np.ndarray:
    """Inpaint red/orange teacher marks instead of leaving faded digits for OCR."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    mask = ((hue < 18) | (hue > 162)) & (sat > 35) & (val > 40)
    mask_u8 = (mask.astype(np.uint8)) * 255
    if cv2.countNonZero(mask_u8) < 40:
        return bgr
    mask_u8 = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    return cv2.inpaint(bgr, mask_u8, 4, cv2.INPAINT_TELEA)


def _drop_ruling(bgr: np.ndarray) -> np.ndarray:
    """Inpaint long horizontal notebook lines; keep short letter strokes (д, у, щ)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    dark = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 35, 10
    )
    klen = max(48, width // 12)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (klen, 1))
    lines = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(lines) < 120:
        return bgr
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2)))
    return cv2.inpaint(bgr, lines, 3, cv2.INPAINT_TELEA)


def _lift_ink(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    light, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    light = clahe.apply(light)
    return cv2.cvtColor(cv2.merge([light, a_ch, b_ch]), cv2.COLOR_LAB2BGR)


def _unsharp(bgr: np.ndarray) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), 1.0)
    return cv2.addWeighted(bgr, 1.35, blur, -0.35, 0)
