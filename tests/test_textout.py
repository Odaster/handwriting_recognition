"""Tests for HTML -> clean paragraph text conversion."""

from __future__ import annotations

from app.textout import html_to_text


def test_plain_text_passthrough():
    assert html_to_text("просто текст") == "просто текст"


def test_html_blocks_to_paragraphs():
    html = (
        '<div data-bbox="1 2 3 4"><h2>Заголовок</h2></div>'
        '<div data-bbox="0 8 9 3"><p>Первая строка.<br/>Вторая строка.</p></div>'
        '<div><p><img alt="sig"/></p></div>'
    )
    out = html_to_text(html)
    assert "data-bbox" not in out
    assert "<" not in out and ">" not in out
    assert "Заголовок" in out
    # <br/> becomes a newline inside the block
    assert "Первая строка.\nВторая строка." in out
    # blocks separated by a blank line
    assert "\n\n" in out
