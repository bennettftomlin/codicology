"""Drawings made of ink rather than pixels.

When a diagram is DRAWN — path objects, no bitmap anywhere — a layout pass
that reads the region as text leaves nothing for image-based recovery to
find. One urban-operations figure of 1,048 paths left only its caption.
The paths say where the drawing is; rendering that region from the vector
source is the best copy that can exist.
"""
import json
import os
import pytest

pdfium = pytest.importorskip("pypdfium2")

FM306 = "/Users/Bennett1/claude_knowledge/assets/books/fm_3-06_urban_operations.pdf"


def _slice(tmp_path, page_no):
    src = pdfium.PdfDocument(FM306)
    one = pdfium.PdfDocument.new()
    one.import_pages(src, [page_no])
    p = str(tmp_path / "s.pdf"); one.save(p)
    return p


@pytest.mark.skipif(not os.path.exists(FM306), reason="source absent")
def test_a_drawn_diagram_the_layout_called_text_is_recovered(vtb, tmp_path):
    """FM 3-06 p24: Figure 2-2, a 3D battlefield diagram of 1,048 paths.
    The build shipped its caption and none of it."""
    p = _slice(tmp_path, 24)
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    caption = vtb.PageItem(
        html="<p>Figure 2-2. The Multidimensional Urban Battlefield</p>",
        box=(0.2, 0.68, 0.8, 0.70))
    prose = vtb.PageItem(html="<p>2-11. Supersurface and subsurface areas "
                              "magnify the complexity.</p>",
                         box=(0.1, 0.72, 0.9, 0.9))
    out = vtb.recover_vector_artwork([caption, prose], pages[0])
    figs = [it for it in out if it.figure is not None]
    assert len(figs) == 1
    assert figs[0].figure.width >= 600
    # the drawing sits between y=0.40 and 0.70 of the page; its box must too
    assert 0.3 <= figs[0].box[1] <= 0.5, figs[0].box
    # caption and prose survive, in order, figure before them
    assert [it.html for it in out if it.html] == [caption.html, prose.html]
    assert out.index(figs[0]) < out.index(caption)


@pytest.mark.skipif(not os.path.exists(FM306), reason="source absent")
def test_a_region_already_emitted_as_a_figure_is_left_alone(vtb, tmp_path):
    from PIL import Image
    p = _slice(tmp_path, 24)
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    already = vtb.PageItem(figure=Image.new("RGB", (50, 50), "white"),
                           box=(0.15, 0.38, 0.85, 0.70))
    out = vtb.recover_vector_artwork([already], pages[0])
    assert sum(1 for it in out if it.figure is not None) == 1


@pytest.mark.skipif(not os.path.exists(FM306), reason="source absent")
def test_a_drawn_table_kept_as_a_table_is_not_turned_into_a_picture(vtb, tmp_path):
    p = _slice(tmp_path, 24)
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    table = vtb.PageItem(html="<table><tr><td>drawn with rules</td></tr></table>",
                         box=(0.15, 0.38, 0.85, 0.70))
    out = vtb.recover_vector_artwork([table], pages[0])
    assert sum(1 for it in out if it.figure is not None) == 0


def test_no_sidecar_or_facsimile_is_untouched(vtb, tmp_path):
    from PIL import Image
    p = str(tmp_path / "page.jpg")
    Image.new("RGB", (300, 400), "white").save(p)
    items = [vtb.PageItem(html="<p>x</p>", box=(0.1, 0.1, 0.9, 0.2))]
    assert vtb.recover_vector_artwork(items, p) is items
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": "nope.pdf", "page": 0, "facsimile": True}, fh)
    assert vtb.recover_vector_artwork(items, p) is items


@pytest.mark.skipif(not os.path.exists(
    "/Users/Bennett1/claude_knowledge/assets/books/FM-3-25-150-Combatives.pdf"),
    reason="source absent")
def test_a_frame_drawn_around_prose_does_not_swallow_the_prose(vtb, tmp_path):
    """
    Combatives p27: a chapter opener whose 278 words sit inside a drawn
    border. The cluster rule read the border as a diagram and its contents
    as labels, shipping the page with two words. Prose is not a label.
    """
    src = pdfium.PdfDocument(
        "/Users/Bennett1/claude_knowledge/assets/books/FM-3-25-150-Combatives.pdf")
    one = pdfium.PdfDocument.new()
    one.import_pages(src, [27])
    p = str(tmp_path / "s.pdf"); one.save(p)
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    prose = vtb.PageItem(
        html="<p>" + "Basic ground fighting techniques form the core of "
             "the training and every soldier must master them early. " * 6 + "</p>",
        box=(0.15, 0.3, 0.85, 0.7))
    out = vtb.recover_vector_artwork([prose], pages[0])
    assert prose in out, "the paragraph must survive"
