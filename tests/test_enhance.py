"""Tests for notebook-photo enhance and Chandra strip stitching."""

from __future__ import annotations

import numpy as np
from PIL import Image

from app.chandra_engine import page_bands, stitch_band_texts
from app.enhance import _hough_segments, enhance_page


def test_hough_segments_both_opencv_shapes():
    a = np.array([[[0, 0, 100, 2]]], dtype=np.int32)
    b = np.array([[0, 0, 100, 2], [10, 1, 80, 3]], dtype=np.int32)
    assert _hough_segments(a).shape == (1, 4)
    assert _hough_segments(b).shape == (2, 4)
    x1, y1, x2, y2 = _hough_segments(b)[0]
    assert int(x1) == 0 and int(y2) == 2


def test_enhance_keeps_rgb_size():
    img = Image.new("RGB", (240, 320), (245, 242, 230))
    out = enhance_page(img)
    assert out.mode == "RGB"
    assert out.size == img.size


def test_enhance_lined_page_does_not_crash():
    arr = np.full((900, 700, 3), 245, dtype=np.uint8)
    for y in range(80, 820, 28):
        arr[y : y + 2, 40:660] = 80
    out = enhance_page(Image.fromarray(arr))
    assert out.size == (700, 900)


def test_enhance_mutes_red_marks():
    arr = np.full((80, 120, 3), 240, dtype=np.uint8)
    arr[20:50, 40:80] = (220, 20, 20)
    out = np.array(enhance_page(Image.fromarray(arr)))
    before = arr[30, 60]
    after = out[30, 60]
    # red channel should drop toward paper; not stay a saturated mark
    assert int(after[0]) - int(after[1]) < int(before[0]) - int(before[1])


def test_page_bands_overlap_tall_page():
    img = Image.new("RGB", (100, 1800), (255, 255, 255))
    bands = page_bands(img, 3)
    assert len(bands) == 3
    assert all(b.size[0] == 100 for b in bands)
    # first starts at top, last ends at bottom
    assert bands[0].size[1] > 500
    assert bands[-1].size[1] > 500


def test_page_bands_skip_short_page():
    img = Image.new("RGB", (100, 400), (255, 255, 255))
    assert page_bands(img, 3) == [img]


def test_chandra_default_tokens_cover_layout_html():
    from app.chandra_engine import _default_max_tokens

    assert int(_default_max_tokens(3)) >= 1536
    assert int(_default_max_tokens(1)) >= 2048


def test_page_bands_cut_in_gaps():
    arr = np.full((1800, 120, 3), 250, dtype=np.uint8)
    for y0, y1 in ((80, 160), (820, 900), (1560, 1640)):
        arr[y0:y1, 10:110] = 30
    bands = page_bands(Image.fromarray(arr), 3)
    assert len(bands) == 3
    top = np.array(bands[0])
    mid = np.array(bands[1])
    bot = np.array(bands[2])
    # first band contains the first ink bar
    assert top[:, :, 0].min() < 80
    # last band contains the last ink bar
    assert bot[:, :, 0].min() < 80
    # middle band is not a copy of the full page
    assert mid.shape[0] < 1800


def test_prefer_refine_keeps_draft_if_short():
    from app.chandra_engine import _prefer_refine

    draft = " ".join(["слово"] * 40)
    assert _prefer_refine(draft, "мало") == draft
    long = " ".join(["слово"] * 40)
    assert _prefer_refine(draft, long) == long


def test_prefer_refine_keeps_draft_heading():
    from app.chandra_engine import _prefer_refine

    draft = (
        "Рвадчать прешье сентимбры\n"
        "Обулаючее согинемче\n\n"
        "На картине художник изобразил остров с деревьями.\n"
        "От этого полотна веет свежкостью."
    )
    refined = (
        "Являющаяся прелюдией к сентимубрю\n"
        "Обустроенное сочинение\n\n"
        "На картине художник изобразил остров с деревьями.\n"
        "От этого полотна веет свежестью и лесом. 9/9"
    )
    out = _prefer_refine(draft, refined)
    assert "прелюдией" not in out
    assert "Рвадчать" in out
    assert "9/9" in out
    assert "свежестью и лесом" in out


def test_stitch_drops_overlap_line():
    top = "Перед нами картина.\nНа переднем плане остров."
    bottom = "На переднем плане остров.\nОт полотна веет свежестью."
    out = stitch_band_texts([top, bottom])
    assert out.count("На переднем плане остров.") == 1
    assert "Перед нами картина." in out
    assert "От полотна веет свежестью." in out


def test_stitch_fuzzy_maslennikov_variants():
    top = (
        'Серед наши картина замечательно\n'
        'художника Павла Мясленникова "Браславские озёра".'
    )
    bottom = (
        'художника Павла Сласенникова "Браславские озёра".\n'
        "На картине художник изобразил остров."
    )
    out = stitch_band_texts([top, bottom])
    assert out.count("Браславские озёра") == 1
    assert "На картине художник изобразил остров." in out


def test_stitch_fuzzy_bottom_paragraph():
    top = (
        "Автор деревья изобразил в\n"
        "ярких оттенках, а вот на самом низу от\n"
        "Сласенников маскирует едким более мешнее."
    )
    bottom = (
        "арших отменников, а вот на самом низу от\n"
        "классеников масками едем более дешнееае\n"
        "отмены.\n"
        "От этого поколения вест свежкостью и чесом."
    )
    out = stitch_band_texts([top, bottom])
    assert "От этого поколения" in out
    # one OCR variant of the "на самом низу" sentence, not both
    assert out.lower().count("на самом низу") == 1
