"""Fix OCR tokens that are not real dictionary words using cursive lookalikes.

Chandra emits letter-soup that *looks* Russian (лацубу, кинуза) because the VLM
has no lexicon. SAGE also leaves many of those: it is trained on typos, not on
ч/л щ/ц substitutions. This pass only rewrites tokens that pymorphy3 does not
know as a full dictionary word, and only when a unique cheap lookalike exists.
"""

from __future__ import annotations

import re
from functools import lru_cache

_WORD_RE = re.compile(r"[А-Яа-яЁё]+")
_MIN_LEN = 5
_MAX_COST = 1.55
_SUB_COST = 0.5
_GAP = 0.35

# Symmetric pairs of letters that collapse in Russian school cursive.
_PAIRS = (
    "чл",
    "щц", "щш",
    "нк",
    "дз",
    "жи",
    "уо",
)


def _confusion_map() -> dict[str, tuple[str, ...]]:
    bucket: dict[str, set[str]] = {}
    for a, b in _PAIRS:
        bucket.setdefault(a, set()).add(b)
        bucket.setdefault(b, set()).add(a)
        bucket.setdefault(a.upper(), set()).add(b.upper() if b.isalpha() else b)
        bucket.setdefault(b.upper(), set()).add(a.upper() if a.isalpha() else a)
    return {ch: tuple(sorted(alts)) for ch, alts in bucket.items()}


_CONF = _confusion_map()


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def is_lexicon_word(word: str) -> bool:
    """True only if OpenCorpora knows the whole token, not a guessed prefix/suffix."""
    if len(word) < 2:
        return False
    parsed = _morph().parse(word)
    if not parsed:
        return False
    stack = parsed[0].methods_stack
    if not stack:
        return False
    names = [type(step[0]).__name__ for step in stack]
    if any(name in {"UnknownPrefixAnalyzer", "FakeDictionary", "KnownSuffixAnalyzer"} for name in names):
        return False
    first = stack[0]
    return type(first[0]).__name__ == "DictionaryAnalyzer" and first[1] == word.casefold()


def _neighbors(word: str) -> list[tuple[float, str]]:
    """Same-length cursive substitutions, cheapest first."""
    base = word.casefold()
    found: dict[str, float] = {}

    def walk(current: str, cost: float, start: int, depth: int) -> None:
        if depth == 3 and is_lexicon_word(current):
            prev = found.get(current)
            if prev is None or cost < prev:
                found[current] = cost
        if depth >= 3 or cost > _MAX_COST:
            return
        for i in range(start, len(current)):
            ch = current[i]
            for alt in _CONF.get(ch, ()):
                nxt = current[:i] + alt + current[i + 1 :]
                walk(nxt, cost + _SUB_COST, i + 1, depth + 1)

    walk(base, 0.0, 0, 0)
    return sorted((c, w) for w, c in found.items())


def _match_case(source: str, candidate: str) -> str:
    if source.isupper():
        return candidate.upper()
    if source[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def _pick(token: str) -> str | None:
    ranked = _neighbors(token)
    if not ranked:
        return None
    best_cost, best = ranked[0]
    if best_cost > _MAX_COST:
        return None
    if len(ranked) > 1 and ranked[1][0] <= best_cost + _GAP:
        return None
    if best.casefold() == token.casefold():
        return None
    return _match_case(token, best)


def fix_cursive_oov(text: str) -> str:
    """Replace unknown tokens with a unique cursive lookalike in the lexicon."""
    if not text:
        return text

    def repl(match: re.Match[str]) -> str:
        token = match.group()
        if len(token) < _MIN_LEN:
            return token
        if is_lexicon_word(token):
            return token
        chosen = _pick(token)
        return chosen if chosen else token

    return _WORD_RE.sub(repl, text)
