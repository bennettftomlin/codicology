"""Artwork the layout never boxed.

A figure is normally emitted because the recognition called that region a
picture. When it calls the region text instead, the artwork does not arrive
mis-cropped — it does not arrive at all. A first-aid manual lost its
pressure-point diagram off thirteen pages that way while gaining the prose
around it. The PDF still holds the picture and knows where it belongs.
"""
import json
import os
import pytest
from PIL import Image

pdfium = pytest.importorskip("pypdfium2")

FM = "/Users/Bennett1/claude_knowledge/assets/books/FM4-25x11.pdf"


def _slice(tmp_path, src_path, page_no):
    src = pdfium.PdfDocument(src_path)
    one = pdfium.PdfDocument.new()
    one.import_pages(src, [page_no])
    p = str(tmp_path / "slice.pdf")
    one.save(p)
    return p


@pytest.mark.skipif(not os.path.exists(FM), reason="source absent")
def test_a_diagram_the_layout_called_text_is_recovered(vtb, tmp_path):
    """FM 4-25.11 p46 holds Figure 2-31 as ONE embedded 1798x1796 image. The
    old build cut the rendered page into twelve fragments; the new one boxed
    nothing. Correct is one complete diagram."""
    p = _slice(tmp_path, FM, 46)
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    items = [vtb.PageItem(html="<p>2-20. Tourniquet. A tourniquet is a "
                               "constricting band placed around a limb.</p>",
                          box=(0.1, 0.75, 0.9, 0.95))]
    out = vtb.recover_orphan_artwork(items, pages[0])
    figs = [it for it in out if it.figure is not None]
    assert len(figs) == 1, f"expected one recovered diagram, got {len(figs)}"
    assert figs[0].figure.width > 800, figs[0].figure.size
    # and it is placed BEFORE the prose that follows it on the page
    assert out.index(figs[0]) < out.index(items[0])


@pytest.mark.skipif(not os.path.exists(FM), reason="source absent")
def test_artwork_already_emitted_is_not_duplicated(vtb, tmp_path):
    p = _slice(tmp_path, FM, 46)
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    src = pdfium.PdfDocument(p)[0]
    pw, ph = src.get_size()
    obj = next(o for o in src.get_objects() if isinstance(o, pdfium.PdfImage))
    b = obj.get_bounds()
    box = (b[0] / pw, 1 - b[3] / ph, b[2] / pw, 1 - b[1] / ph)
    already = [vtb.PageItem(figure=Image.new("RGB", (40, 40), "white"), box=box)]
    out = vtb.recover_orphan_artwork(already, pages[0])
    assert len(out) == 1, "the same picture must not be emitted twice"


def test_a_facsimile_page_has_nothing_to_orphan(vtb, tmp_path):
    """Citadel's pages ARE photographs; the whole page is the image."""
    p = str(tmp_path / "page.jpg")
    Image.new("RGB", (400, 500), "white").save(p)
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": str(tmp_path / "nope.pdf"), "page": 0,
                   "facsimile": True}, fh)
    items = [vtb.PageItem(html="<p>text</p>", box=(0.1, 0.1, 0.9, 0.2))]
    assert vtb.recover_orphan_artwork(items, p) is items


def test_no_sidecar_means_no_recovery(vtb, tmp_path):
    p = str(tmp_path / "page.jpg")
    Image.new("RGB", (400, 500), "white").save(p)
    items = [vtb.PageItem(html="<p>text</p>", box=(0.1, 0.1, 0.9, 0.2))]
    assert vtb.recover_orphan_artwork(items, p) is items


# ---- scanned pages: the scan is the page, not a figure on it -----------

from test_page_loading import (TYPE_AREA, _plate, _png, make_book,  # noqa: E402
                               scan_page, typeset_page)


def _load(vtb, tmp_path, pages):
    p = make_book(tmp_path / "book.pdf", pages)
    d = tmp_path / "pages"; d.mkdir()
    return vtb.load_pages_from_pdf(p, str(d))


def _read():
    from codicology import pipeline as vtb
    return [vtb.PageItem(html="<p>Lloyd's Coffee-house stood in Tower "
                              "Street.</p>", box=(0.1, 0.1, 0.9, 0.9))]


def test_the_scan_of_the_page_is_not_artwork_the_layout_missed(vtb, tmp_path):
    """
    A scan set inside its margins measures 0.72 of the leaf — inside the
    band where a picture counts as artwork — and no figure the recognition
    emitted covers it, because what the recognition emitted from it was
    the text. So every page of one book gained its own photograph: 438
    figures, 130MB.
    """
    pages = _load(vtb, tmp_path, [scan_page() for _ in range(4)])
    for page in pages:
        items = _read()
        out = vtb.recover_orphan_artwork(items, page)
        assert [it for it in out if it.figure is not None] == [], \
            "the page's own scan was inserted as a figure"


def test_a_plate_laid_over_its_scan_is_still_recovered(vtb, tmp_path):
    """Google stores a chapter's headpiece or a plate as its own layer over
    the page scan. That is artwork, and still the layout's to miss."""
    piece = ("image", (120, 300, 340, 440), _png((440, 280), _plate))
    pages = _load(vtb, tmp_path, [scan_page(extra=[piece])] +
                  [scan_page() for _ in range(3)])
    out = vtb.recover_orphan_artwork(_read(), pages[0])
    figs = [it for it in out if it.figure is not None]
    assert len(figs) == 1, f"expected the plate alone, got {len(figs)} figures"
    w, h = figs[0].figure.size
    assert 1.4 < w / h < 1.75, f"that is not the plate's shape: {w}x{h}"
    assert abs(figs[0].box[1] - 300 / 704) < 0.01


def test_a_strip_in_the_margin_of_a_scan_is_not_artwork(vtb, tmp_path):
    """
    Beside one scan's page image, out in the PDF's margin, sat a 33x634
    strip of scanner bed — 0.024 of the leaf, just over the floor for a
    figure — and eleven pages printed it as a black bar. A plate lies on
    its page; this lay off it.
    """
    strip = ("image", (8, 20, 36, 640), _png((28, 620), _plate))
    pages = _load(vtb, tmp_path,
                  [scan_page(extra=[strip]) for _ in range(4)])
    assert TYPE_AREA[0] > 36, "the strip must sit clear of the page image"
    for page in pages:
        out = vtb.recover_orphan_artwork(_read(), page)
        assert [it for it in out if it.figure is not None] == []


def test_a_lone_plate_in_a_typeset_book_is_still_recovered(vtb, tmp_path):
    """The rule is about scanned books. A typeset book's plate page is a
    picture of itself too, and its picture is exactly what orphan recovery
    exists for."""
    plate = [("image", (60, 80, 406, 600), _png((346, 520), _plate))]
    pages = _load(vtb, tmp_path, [plate] + [typeset_page() for _ in range(4)])
    out = vtb.recover_orphan_artwork([], pages[0])
    figs = [it for it in out if it.figure is not None]
    assert len(figs) == 1, "a typeset book lost its plate"
