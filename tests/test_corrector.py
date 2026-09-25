"""Test the context-aware SAGE corrector (skipped if deps are absent)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("sentencepiece")

# Use the small model in tests to keep them fast and avoid a large download.
os.environ["SAGE_MODEL"] = "ai-forever/sage-fredt5-distilled-95m"

from app.corrector import correct_text  # noqa: E402


def test_context_corrector_fixes_spelling():
    # "исли изти" is unfixable by a dictionary but trivial with context.
    out = correct_text("исли изти в лесу").lower()
    assert "если" in out
    assert "идти" in out
    # a valid preposition must survive
    assert " в " in f" {out} "
