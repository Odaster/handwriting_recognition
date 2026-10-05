"""Cursive lookalike OOV fixes (no GPU)."""

from __future__ import annotations

from app.cursivefix import fix_cursive_oov, is_lexicon_word


def test_known_words_untouched():
    assert is_lexicon_word("никуда")
    assert is_lexicon_word("жутко")
    assert is_lexicon_word("чащобу")
    assert fix_cursive_oov("Лес шумит, Марина совсем одна.") == "Лес шумит, Марина совсем одна."


def test_fixes_chandra_nonwords():
    out = fix_cursive_oov("забиралась в лацубу осинышка, не выведет тебя кинуза")
    assert "лацубу" not in out
    assert "чащобу" in out
    assert "кинуза" not in out
    assert "никуда" in out


def test_leaves_real_lookalike_typos_for_sage():
    # both are dictionary verbs; this pass must not pick
    assert "крушиться" in fix_cursive_oov("начинают крушиться снежинки")
