"""Convert layout HTML (e.g. Chandra output) into clean paragraph text.

The neural VLM returns HTML blocks with ``data-bbox`` layout info. Users want
plain text organized into paragraphs matching those blocks — no HTML markup.
"""

from __future__ import annotations

import html as _html
import re

_IMG = re.compile(r"<img[^>]*>", re.IGNORECASE)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_BLOCK_END = re.compile(r"</(div|p|h[1-6]|li|ul|ol|tr|table|section|blockquote)>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


def html_to_text(raw: str) -> str:
    """Return clean text with a blank line between blocks and line breaks for <br/>.

    If ``raw`` has no markup it is returned trimmed (no-op for plain OCR text).
    """
    if not raw:
        return ""
    if "<" not in raw or ">" not in raw:
        return raw.strip()

    s = _IMG.sub("", raw)          # drop images/signatures
    s = _BR.sub("\n", s)           # line breaks inside a block
    s = _BLOCK_END.sub("\n\n", s)  # blank line between blocks
    s = _TAG.sub("", s)            # strip remaining tags
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
    return "\n".join(out).strip()
