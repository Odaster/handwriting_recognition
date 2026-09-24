"""Tests for handwriting-engine helpers (segmentation) and post-processing.

The neural TrOCR model itself is heavy and not exercised here; we test the
deterministic OpenCV segmentation and the spell-correction guardrails.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scripts.make_sample import make_image

pytest.importorskip("scipy")

from app.htr import _binarize, segment_lines, segment_words  # noqa: E402


def _binary_of(path):
    gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
    return _binarize(gray)


def test_segment_lines_counts_three(tmp_path):
    img = make_image("первая строка\nвторая строка\nтретья строка", tmp_path / "l.png", font_size=44)
    bands = segment_lines(_binary_of(img))
    assert len(bands) == 3
    # bands are ordered top-to-bottom and non-overlapping
    for (a0, a1), (b0, b1) in zip(bands, bands[1:]):
        assert a1 <= b0


def test_segment_words_counts(tmp_path):
    img = make_image("один    два    три    четыре", tmp_path / "w.png", font_size=48)
    thr = _binary_of(img)
    bands = segment_lines(thr)
    assert len(bands) == 1
    y0, y1 = bands[0]
    words = segment_words(thr[y0:y1, :])
    assert len(words) == 4


def test_spellcheck_preserves_short_words():
    from app.postprocess import correct_text

    # single-letter preposition "в" and the phrase must survive untouched
    assert correct_text("Ночь в лесу") == "Ночь в лесу"
