"""Tests for HTML -> clean paragraph text conversion."""

from __future__ import annotations

from app.textout import collapse_loops, html_to_text


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


def test_collapse_inline_phrase_loop():
    phrase = "масленников, а вот на самом низу ст"
    raw = "ярых оттенках, а вот на самом низу ст " + " ".join([phrase] * 40)
    out = html_to_text(raw)
    assert out.count(phrase) <= 2
    assert "ярых оттенках" in out
    assert len(out) < 400


def test_collapse_repeated_lines():
    line = "масленников, а вот на самом низу ст"
    raw = "лес, который пересекается с небом.\n" + "\n".join([line] * 80)
    out = collapse_loops(raw)
    assert "лес, который пересекается с небом." in out
    assert out.count(line) <= 2


def test_drop_teacher_latex_array():
    raw = (
        "Для меня художник шикарно передал смысл картины. Какой?\n\n"
        r"\begin{array}{r} 1 - 0 \\ \hline 2 - 4 - 0 \end{array}"
        "\n\n9/4"
    )
    out = html_to_text(raw)
    assert "array" not in out
    assert "hline" not in out
    assert "Какой?" in out
    assert "9/4" in out


def test_html_truncated_tag_is_stripped():
    raw = (
        'Согласен на выплату среднего заработка.\n'
        '<div data-bbox="462 403 646 477\n'
        "«  » _____ 20   г. на _____ календарных дней, в том числе:\n"
        "1.2. дополнительные отпуска:"
    )
    out = html_to_text(raw)
    assert "data-bbox" not in out
    assert "<div" not in out
    assert "Согласен на выплату" in out
    assert "дополнительные отпуска" in out


def test_drop_garbled_strip_tail():
    raw = (
        "Идут дальше рисованью: осенью свертки леса страшные волнами. "
        "Марина забирается на дерево и решается перепрыгнуть длинную ночь в лесу.\n"
        "Мокрый снег напоял влагой пальто. Флангель в проглядном рассвете "
        "неописанно захрипели петухи. Деревья подвигались\n\n"
        "должно раскрываться: осинью сверкнув лесом страшными волнами.\n"
        "ночь в лесу.\n\n"
        "обшоропанных ног. Флаконцу в громогласном рассвете\n"
        "неожиданно закричали петухи. Деревья, охвачиваясь, бьют своим рудом."
    )
    out = html_to_text(raw)
    assert "Флаконцу" not in out
    assert "должно раскрываться" not in out
    assert "Марина" in out
    assert "петухи" in out
    assert "Флангель" in out
