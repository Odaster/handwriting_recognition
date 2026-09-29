"""Optional post-processing: conservative dictionary-based spell correction.

WARNING / trade-off: this is *not* context- or grammar-aware. It only replaces
tokens that are missing from a Russian frequency dictionary, using a small
edit distance. Russian is highly inflected and the dictionary does not cover
every valid word form, so aggressive correction tends to damage correct text
(e.g. "в" -> "я", "лесом" -> "летом"). We therefore keep it deliberately
conservative and disabled by default. For real orthography/grammar/meaning
correction, a context-aware model (e.g. an LLM or a Russian sequence-to-sequence
spelling corrector such as the SAGE models) is required.
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD_RE = re.compile(r"[А-Яа-яЁё-]+")

# Tokens no longer than this are never touched — protects prepositions and
# conjunctions ("в", "и", "на", "по", "не", ...) that a dictionary check would
# otherwise mangle.
_MIN_LEN = 4


@lru_cache(maxsize=1)
def _checker():
    from spellchecker import SpellChecker

    # distance=1 keeps corrections close to the OCR output and limits damage.
    return SpellChecker(language="ru", distance=1)


def _match_case(source: str, candidate: str) -> str:
    if source.isupper():
        return candidate.upper()
    if source[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def correct_text(text: str) -> str:
    """Return ``text`` with clearly-unknown long Russian words nudged toward the
    nearest dictionary word. Conservative by design (see module docstring)."""
    checker = _checker()

    def fix(match: re.Match) -> str:
        token = match.group()
        if len(token) < _MIN_LEN or "-" in token:
            return token
        lower = token.lower()
        if lower in checker:
            return token
        candidate = checker.correction(lower)
        if candidate and candidate != lower and len(candidate) >= _MIN_LEN:
            return _match_case(token, candidate)
        return token

    return _WORD_RE.sub(fix, text)
