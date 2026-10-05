"""Corrector helpers (always) and SAGE-small integration (if deps exist)."""

from __future__ import annotations

import os

from types import SimpleNamespace

from app.corrector import (
    DEFAULT_MODEL,
    apply_to_paragraphs,
    _is_seq2seq,
    _packs,
    _sage_num_beams,
    _share_t5_input_embeddings,
    _t5_input_embeddings_shared,
    prefer_original,
    looks_like_form_fragment,
    strip_completion,
)


def test_default_corrector_is_sage_17b():
    assert DEFAULT_MODEL == "ai-forever/sage-v1.1.0"
    assert _is_seq2seq(DEFAULT_MODEL)


def test_sage_input_embeddings_are_shared():
    class Ptr:
        def __init__(self, n):
            self._n = n

        def data_ptr(self):
            return self._n

    shared = SimpleNamespace(weight=Ptr(11))
    model = SimpleNamespace(
        shared=shared,
        encoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=Ptr(99))),
        decoder=SimpleNamespace(embed_tokens=SimpleNamespace(weight=Ptr(98))),
    )
    assert not _t5_input_embeddings_shared(model)
    _share_t5_input_embeddings(model)
    assert _t5_input_embeddings_shared(model)
    assert model.encoder.embed_tokens is shared
    assert model.decoder.embed_tokens is shared


def test_sage_17b_defaults_to_greedy(monkeypatch):
    monkeypatch.delenv("SAGE_BEAMS", raising=False)
    assert _sage_num_beams("ai-forever/sage-v1.1.0") == 1
    assert _sage_num_beams("ai-forever/sage-fredt5-distilled-95m") == 4
    monkeypatch.setenv("SAGE_BEAMS", "4")
    assert _sage_num_beams("ai-forever/sage-v1.1.0") == 4


def test_strip_completion_drops_fences_and_labels():
    assert strip_completion("```\nПривет\n```") == "Привет"
    assert strip_completion("Исправленный текст:\nЕсли идти") == "Если идти"


def test_prefer_original_keeps_collapsed_sage():
    src = (
        "Деревня была где-то за лесом. Сами идти в ней по большой дороге, "
        "ну или отложить не один десяток километров."
    )
    junk = '.л..........\n.. ". ". ". ".7.N "NN "\n. " чуу  буББ.'
    assert prefer_original(src, junk) == src
    assert prefer_original("Дом в лесу", ". -.IC.ICO.ICABABУ ИДСа.") == "Дом в лесу"
    assert "лес" in prefer_original(src, "Деревня была где-то за лесом.")


def test_packs_groups_short_lines(monkeypatch):
    monkeypatch.setenv("CORRECTOR_MODEL", "t-tech/T-lite-it-2.1")
    monkeypatch.delenv("SAGE_MODEL", raising=False)
    packs = _packs("\n".join(["слово"] * 300))
    assert 1 <= len(packs) <= 3
    assert all(len(p.split()) <= 220 for p in packs)


def test_sage_packs_are_sentence_sized(monkeypatch):
    monkeypatch.setenv("CORRECTOR_MODEL", "ai-forever/sage-v1.1.0")
    monkeypatch.delenv("SAGE_MODEL", raising=False)
    monkeypatch.delenv("CORRECTOR_MAX_WORDS", raising=False)
    packs = _packs("Лес шумит, успокаивает. " * 30)
    assert len(packs) > 1
    assert all(len(p.split()) <= 24 for p in packs)
    assert all("\n" not in p for p in packs)


def test_seq2seq_keeps_paragraph_breaks(monkeypatch):
    monkeypatch.setenv("CORRECTOR_MODEL", "ai-forever/sage-v1.1.0")
    monkeypatch.delenv("SAGE_MODEL", raising=False)
    src = (
        "Дом в лесу\n\n"
        "Лес шумит, успокаивает. Толстые корни обхватили извилистую тропу."
    )
    out = apply_to_paragraphs(src, lambda pack: pack)
    assert "\n\n" in out
    assert "успокаивает. Толстые" in out
    assert "успокаивает.\nТолстые" not in out


def test_skips_form_dates_and_blanks():
    assert looks_like_form_fragment('с «  08  »           05             20 26   г.')
    assert looks_like_form_fragment("Прошу предоставить мне трудовой отпуск за _____ год")
    assert looks_like_form_fragment('<div data-bbox="462 403 646 477')
    assert not looks_like_form_fragment(
        "Согласен на выплату среднего заработка в связи с предоставлением трудового отпуска."
    )


def test_context_corrector_fixes_spelling():
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("sentencepiece")

    os.environ["CORRECTOR_MODEL"] = "ai-forever/sage-fredt5-distilled-95m"
    os.environ["SAGE_MODEL"] = "ai-forever/sage-fredt5-distilled-95m"
    from app.corrector import correct_text

    out = correct_text("исли изти в лесу").lower()
    assert "если" in out
    assert "идти" in out
    assert " в " in f" {out} "
