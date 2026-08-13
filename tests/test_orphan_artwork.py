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
