"""Convert layout HTML (e.g. Chandra output) into clean paragraph text.

The neural VLM returns HTML blocks with ``data-bbox`` layout info. Users want
plain text organized into paragraphs matching those blocks — no HTML markup.
"""

from __future__ import annotations

import html as _html
import re
from difflib import SequenceMatcher

_IMG = re.compile(r"<img[^>]*>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_END = re.compile(r"</(div|p|h[1-6]|li|ul|ol|tr|table|section|blockquote)>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_TRUNCATED_TAG = re.compile(r"<[a-zA-Z][^>\n]*$", re.MULTILINE)
_LOOP_PHRASE = re.compile(r"(.{12,160}?)(?:\s*\1){2,}", re.DOTALL)
_LATEX_ENV = re.compile(r"\\begin\{[a-zA-Z*]+\}.*?\\end\{[a-zA-Z*]+\}", re.DOTALL)
_LATEX_INLINE = re.compile(r"\$\$.*?\$\$|\$(?!\$)[^$\n]+\$")
_LATEX_CMD = re.compile(r"\\(?:frac|array|hline|cdot|left|right|mathrm|text|begin|end)(?:\s*\{[^{}]*\})*")
_LATEX_LEFTOVER = re.compile(r"\\[a-zA-Z]+\*?")


def collapse_loops(text: str) -> str:
    """Cut decoder collapse like 'масленников, а вот на самом низу ст' × N."""
    if not text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    prev = None
    streak = 0
    for ln in lines:
        if ln == prev:
            streak += 1
            if streak >= 3:
                continue
        else:
            prev = ln
            streak = 1
        out.append(ln)
    collapsed = "\n".join(out)
    collapsed = _LOOP_PHRASE.sub(r"\1", collapsed)
    return drop_repeated_tail(drop_teacher_chrome(collapse_near_duplicates(collapsed))).strip()


def _norm_line(text: str) -> str:
    return " ".join(text.casefold().split())


def line_similarity(left: str, right: str) -> float:
    """0..1 how likely two OCR lines are the same sentence from overlapping strips."""
    a, b = _norm_line(left), _norm_line(right)
    if not a or not b:
        return 0.0
    matcher = SequenceMatcher(None, a, b)
    seq = matcher.ratio()
    match = matcher.find_longest_match(0, len(a), 0, len(b))
    if match.size >= 22 and min(len(a), len(b)) >= 28:
        seq = max(seq, 0.72)
    ta, tb = set(a.split()), set(b.split())
    jac = len(ta & tb) / len(ta | tb) if ta and tb else 0.0
    return max(seq, 0.5 * seq + 0.5 * jac)


def collapse_near_duplicates(text: str, threshold: float = 0.68) -> str:
    """Keep the longer of two consecutive-ish OCR variants of the same line."""
    if not text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    for ln in lines:
        if not ln.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        replaced = False
        for i in range(len(out) - 1, max(-1, len(out) - 8), -1):
            if not out[i].strip():
                continue
            need = threshold if max(len(ln.strip()), len(out[i].strip())) >= 36 else 0.9
            if line_similarity(ln, out[i]) >= need:
                if len(ln.strip()) > len(out[i].strip()):
                    out[i] = ln
                replaced = True
                break
        if not replaced:
            out.append(ln)
    # trim trailing blanks
    while out and not out[-1].strip():
        out.pop()
    return "\n".join(out)


def drop_teacher_chrome(text: str) -> str:
    """Strip LaTeX/math that Chandra emits for red margin grades."""
    if not text:
        return text
    s = _LATEX_ENV.sub("", text)
    s = _LATEX_INLINE.sub("", s)
    s = _LATEX_CMD.sub("", s)
    s = _LATEX_LEFTOVER.sub("", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


_CONTENT = re.compile(r"[А-Яа-яЁё]{5,}")


def _is_repeat_tail(tail: str, head: str, min_jaccard: float) -> bool:
    tail_words = {w.casefold() for w in _CONTENT.findall(tail)}
    head_words = {w.casefold() for w in _CONTENT.findall(head)}
    if len(tail_words) < 4 or not head_words:
        return False
    shared = tail_words & head_words
    if len(shared) < 3:
        return False
    return len(shared) / len(tail_words) >= min_jaccard


def drop_repeated_tail(text: str, min_jaccard: float = 0.22) -> str:
    """Drop trailing paragraphs that repeat earlier OCR with worse glyphs."""
    if not text:
        return text
    parts = re.split(r"\n\s*\n+", text.strip())
    if len(parts) < 2:
        return text.strip()
    changed = True
    while changed and len(parts) >= 2:
        changed = False
        for n in range(min(3, len(parts) - 1), 0, -1):
            tail = "\n\n".join(parts[-n:])
            head = "\n\n".join(parts[:-n])
            if _is_repeat_tail(tail, head, min_jaccard):
                parts = parts[:-n]
                changed = True
                break
    return "\n\n".join(parts).strip()


def html_to_text(raw: str) -> str:
    """Return clean text with a blank line between blocks and line breaks for <br/>.

    If ``raw`` has no markup it is returned trimmed (no-op for plain OCR text).
    """
    if not raw:
        return ""
    if "<" not in raw:
        return collapse_loops(raw.strip())

    s = _IMG.sub("", raw)          # drop images/signatures
    s = _BR.sub("\n", s)           # line breaks inside a block
    s = _BLOCK_END.sub("\n\n", s)  # blank line between blocks
    s = _TAG.sub("", s)            # strip remaining tags
    s = _TRUNCATED_TAG.sub("", s)  # Chandra token cap stops mid-<div data-bbox=
    s = _html.unescape(s)

    # Normalize whitespace: trim lines, collapse runs of blank lines to one.
    lines = [ln.strip() for ln in s.split("\n")]
    out: list[str] = []
    blank = False
    for ln in lines:
        if ln:
            out.append(ln)
            blank = False
        elif out and not blank:
            out.append("")
            blank = True
    return collapse_loops("\n".join(out).strip())
