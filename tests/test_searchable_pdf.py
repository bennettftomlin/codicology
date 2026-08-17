"""The searchable PDF: page images with the OCR's text laid invisibly where
each block sits.

The contract has three parts, each tested against a reopened file rather than
the writer's own accounting: the text must be findable and positioned inside
its block's rectangle; the visible page must be the image, unchanged by the
text underneath; and a block whose text cannot fit its box must be left
unsearchable rather than spilled somewhere misleading."""
import os

import pytest
from PIL import Image

pdfium = pytest.importorskip("pypdfium2")
pytest.importorskip("pymupdf")


def _page_png(tmp_path, name="page.png", size=(600, 900)):
    img = Image.new("RGB", size, "white")
    px = img.load()
    for x in range(80, 520):            # a dark bar where the "text" is
        for y in range(100, 130):
            px[x, y] = (30, 30, 30)
    p = str(tmp_path / name)
    img.save(p, dpi=(150, 150))
    return p


def test_text_lands_inside_its_block_and_survives_reopening(vtb, tmp_path):
    page = _page_png(tmp_path)
    items = [[vtb.PageItem(html="<p>Fort la Baie stood here.</p>",
                           box=(0.10, 0.10, 0.90, 0.20))]]
    out = str(tmp_path / "book.pdf")
    placed, forced, _ = vtb.build_searchable_pdf([page], items, out)
    assert (placed, forced) == (1, 0)

    pdf = pdfium.PdfDocument(out)
    tp = pdf[0].get_textpage()
    text = tp.get_text_bounded()
    assert "Fort la Baie" in text
    # every character sits inside the block's rectangle (page: 288x432pt at
    # 150dpi; pdfium y runs bottom-up)
    w, h = pdf[0].get_size()
    idx = text.index("Fort")
    for k in range(idx, idx + 12):
        left, bottom, right, top = tp.get_charbox(k)
        assert 0.10 * w - 2 <= left and right <= 0.90 * w + 2
        assert (1 - 0.20) * h - 2 <= bottom and top <= (1 - 0.10) * h + 2


def test_the_visible_page_is_the_image_unchanged(vtb, tmp_path):
    page = _page_png(tmp_path)
    items = [[vtb.PageItem(html="<p>invisible words</p>",
                           box=(0.2, 0.5, 0.8, 0.6))]]
    out = str(tmp_path / "book.pdf")
    vtb.build_searchable_pdf([page], items, out)
    pdf = pdfium.PdfDocument(out)
    bmp = pdf[0].render(scale=2).to_pil().convert("L")
    # the text block region renders as pure page-white: nothing visibly drawn
    region = bmp.crop((int(0.2 * bmp.width), int(0.5 * bmp.height),
                       int(0.8 * bmp.width), int(0.6 * bmp.height)))
    assert min(region.getdata()) > 245


def test_a_block_too_dense_for_its_box_still_gets_its_words_in(vtb, tmp_path):
    """
    The layer has no reader, so nothing in it can mislead: an imperfectly
    laid-out block costs a loose highlight, while a dropped block costs the
    search itself and says nothing about the loss. Dense text must go in.
    """
    page = _page_png(tmp_path)
    words = "immensity " * 400
    items = [[vtb.PageItem(html=f"<p>{words}</p>", box=(0.48, 0.50, 0.52, 0.505))]]
    out = str(tmp_path / "book.pdf")
    placed, forced, _ = vtb.build_searchable_pdf([page], items, out)
    assert placed + forced == 1
    got = pdfium.PdfDocument(out)[0].get_textpage().get_text_bounded()
    assert "immensity" in got
    # not a token or two survived — effectively the whole block is present
    assert got.count("immensity") >= 380


def test_degenerate_geometry_does_not_lose_a_block(vtb, tmp_path):
    """A polygon that collapsed to a sliver still carries real words."""
    page = _page_png(tmp_path)
    items = [[vtb.PageItem(html="<p>Kaskaskia</p>", box=(0.5, 0.5, 0.5, 0.5))]]
    out = str(tmp_path / "book.pdf")
    placed, forced, _ = vtb.build_searchable_pdf([page], items, out)
    assert placed + forced == 1
    assert "Kaskaskia" in pdfium.PdfDocument(out)[0].get_textpage().get_text_bounded()


def test_no_page_of_a_multi_block_page_is_silently_dropped(vtb, tmp_path):
    """Every block's text is findable, across easy and hostile geometry."""
    page = _page_png(tmp_path)
    blocks = [
        ("Vincennes", (0.10, 0.05, 0.90, 0.12)),      # roomy
        ("Kekionga", (0.10, 0.20, 0.90, 0.205)),      # thin strip
        ("Piankeshaw " * 120, (0.10, 0.30, 0.30, 0.34)),  # dense
        ("Ouiatenon", (0.80, 0.90, 0.82, 0.91)),      # tiny corner
    ]
    items = [[vtb.PageItem(html=f"<p>{t}</p>", box=b) for t, b in blocks]]
    out = str(tmp_path / "book.pdf")
    placed, forced, _ = vtb.build_searchable_pdf([page], items, out)
    assert placed + forced == 4
    got = pdfium.PdfDocument(out)[0].get_textpage().get_text_bounded()
    for t, _ in blocks:
        assert t.split()[0] in got, f"lost {t.split()[0]}"


def test_boxes_round_trip_and_old_entries_go_back_to_the_engine(vtb, tmp_path):
    page = _page_png(tmp_path)
    cache_path = str(tmp_path / "c.ocr.gz")
    c = vtb.OCRCache(cache_path, "surya", ["en"])
    c.put(page, [vtb.PageItem(html="<p>x</p>", box=(0.1, 0.2, 0.3, 0.4)),
                 vtb.PageItem(html="<p>y</p>")])
    c.save()
    c2 = vtb.OCRCache(cache_path, "surya", ["en"])
    got = c2.get(page)
    assert got[0].box == (0.1, 0.2, 0.3, 0.4)
    assert got[1].box is None
    # a v2-era entry — written before boxes existed — is read again rather
    # than served geometry-less, and remains readable under the escape hatch
    key = c2._key(page)
    entry = c2.entries[key]
    for it in entry["items"]:
        it.pop("box", None)
    entry["v"] = 2
    assert c2.get(page) is None
    c3 = vtb.OCRCache(cache_path, "surya", ["en"], serve_stale=True)
    c3.entries[key] = entry
    got_old = c3.get(page)
    assert got_old[0].box is None and got_old[0].html == "<p>x</p>"


def test_words_take_tesseract_geometry_when_both_engines_agree(vtb, tmp_path, monkeypatch):
    """
    Surya's block says the text lives in the top half; tesseract knows the
    word "Altgeld" sits in a specific box inside it. The finished PDF's
    charboxes for that word must land in tesseract's box, not merely inside
    the block — and unmatched words must still be present via the block.
    """
    page = _page_png(tmp_path)          # 600x900 at 150dpi -> 288x432pt
    monkeypatch.setattr(vtb, "_tesseract_word_boxes", lambda _: [
        {"t": "altgeld", "x": 100, "y": 200, "w": 120, "h": 24, "used": False},
    ])
    items = [[vtb.PageItem(html="<p>Governor Altgeld pardoned them.</p>",
                           box=(0.05, 0.05, 0.95, 0.60))]]
    out = str(tmp_path / "book.pdf")
    placed, forced, wordp = vtb.build_searchable_pdf([page], items, out)
    assert wordp == 1
    pdf = pdfium.PdfDocument(out)
    tp = pdf[0].get_textpage()
    text = tp.get_text_bounded()
    for w in ("Governor", "Altgeld", "pardoned"):
        assert w in text
    i = text.index("Altgeld")
    boxes = [tp.get_charbox(k) for k in range(i, i + 7)]
    # tesseract box in pdf points: x 48-105.6, y(top-down) 96-107.5 ->
    # bottom-up y 324.5-336 on a 432pt page
    L = min(b[0] for b in boxes); R = max(b[2] for b in boxes)
    B = min(b[1] for b in boxes); T = max(b[3] for b in boxes)
    assert 44 <= L and R <= 110, (L, R)
    assert 320 <= B and T <= 340, (B, T)


def test_without_tesseract_the_block_path_carries_everything(vtb, tmp_path, monkeypatch):
    page = _page_png(tmp_path)
    monkeypatch.setattr(vtb, "_tesseract_word_boxes", lambda _: None)
    items = [[vtb.PageItem(html="<p>Governor Altgeld pardoned them.</p>",
                           box=(0.1, 0.1, 0.9, 0.3))]]
    out = str(tmp_path / "book.pdf")
    placed, forced, wordp = vtb.build_searchable_pdf([page], items, out)
    assert (placed, wordp) == (1, 0)
    assert "Altgeld" in pdfium.PdfDocument(out)[0].get_textpage().get_text_bounded()


def test_partial_matches_keep_reading_order_and_interpolate_between_anchors(vtb, tmp_path, monkeypatch):
    """
    The defect this replaces: matched words first, pooled leftovers after —
    which shuffled the layer's internal order and broke phrase search across
    every matched/unmatched seam. With anchors either side, the unmatched
    word must sit BETWEEN them, and extraction must read straight through.
    """
    page = _page_png(tmp_path)          # 600x900 @150dpi -> 288x432pt
    monkeypatch.setattr(vtb, "_tesseract_word_boxes", lambda _: [
        {"t": "governor", "x": 100, "y": 200, "w": 130, "h": 24, "used": False},
        {"t": "pardoned", "x": 330, "y": 200, "w": 130, "h": 24, "used": False},
    ])
    items = [[vtb.PageItem(html="<p>Governor Altgeld pardoned them.</p>",
                           box=(0.05, 0.10, 0.95, 0.60))]]
    out = str(tmp_path / "book.pdf")
    vtb.build_searchable_pdf([page], items, out)
    tp = pdfium.PdfDocument(out)[0].get_textpage()
    text = tp.get_text_bounded()
    flat = " ".join(text.split())
    assert "Governor Altgeld pardoned them." in flat, flat
    # Altgeld interpolated into the gap between its anchors, on their line:
    # anchors end at x=110.4pt and start at x=158.4pt (image px * 72/150)
    i = text.index("Altgeld")
    boxes = [tp.get_charbox(k) for k in range(i, i + 7)]
    L = min(b[0] for b in boxes); R = max(b[2] for b in boxes)
    B = min(b[1] for b in boxes); T = max(b[3] for b in boxes)
    assert 105 <= L and R <= 165, (L, R)          # inside the anchor gap
    assert 315 <= B and T <= 340, (B, T)          # on the anchors' line


def test_phrase_search_across_the_seam_survives(vtb, tmp_path, monkeypatch):
    """The user-visible consequence of order: a phrase spanning a matched and
    an unmatched word must be findable as a contiguous string."""
    page = _page_png(tmp_path)
    monkeypatch.setattr(vtb, "_tesseract_word_boxes", lambda _: [
        {"t": "pardoned", "x": 330, "y": 200, "w": 130, "h": 24, "used": False},
    ])
    items = [[vtb.PageItem(html="<p>Governor Altgeld pardoned them fully.</p>",
                           box=(0.05, 0.10, 0.95, 0.60))]]
    out = str(tmp_path / "book.pdf")
    vtb.build_searchable_pdf([page], items, out)
    text = " ".join(pdfium.PdfDocument(out)[0].get_textpage()
                    .get_text_bounded().split())
    assert "pardoned them fully" in text
    assert "Governor Altgeld pardoned" in text
