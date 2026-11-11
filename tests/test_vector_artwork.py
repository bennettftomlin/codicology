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


# ── a table's rules are the table's ──────────────────────────────────────────

def test_a_tables_rules_and_its_trim_are_not_a_drawing(vtb, tmp_path):
    """Principles of Microeconomics p. 205. A heading on a banner, a line of
    prose on a tint and a ruled table beneath: the paths touch, so they
    grouped as one drawing five times the size of the table, too large to
    be taken for it. The drawing went in as a picture of all three, and the
    heading and the table's one row went with it as its labels."""
    from test_doubled_boxes import _problem_blocks, _problem_page
    page = _problem_page(tmp_path)
    heading, asks, table, _ = _problem_blocks()
    items = [heading, asks, table]
    assert vtb.recover_vector_artwork(items, page) == items

    # the paths are a drawing where no table answers for them, which is what
    # a diagram the layout read as text looks like
    unclaimed = vtb.recover_vector_artwork([heading, asks], page)
    assert sum(1 for it in unclaimed if it.figure is not None) == 1


def _table_beside_a_drawing(tmp_path, gap):
    """A ruled table, and `gap` points to its right a fan of twenty strokes."""
    from test_doubled_boxes import LEAF, RULES
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=LEAF[0], height=LEAF[1])
    for k in range(14):
        page.insert_text((52, 60 + 14 * k), "The publisher set this line in type.",
                         fontsize=9)
    x0, y0, x1, y1 = RULES
    for k in range(11):
        y = y0 + k * (y1 - y0) / 10
        page.draw_line((x0, y), (x1, y), width=1)
    for x in (x0, (x0 + x1) / 2, x1):
        page.draw_line((x, y0), (x, y1), width=1)
    left = x1 + gap
    for k in range(20):
        page.draw_line((left, y0 + 20), (left + 200, y0 + 20 + 12 * k), width=1)
    pdf = tmp_path / "fan.pdf"
    doc.save(str(pdf))
    d = tmp_path / "pages"
    d.mkdir()
    from codicology import pipeline
    return (pipeline.load_pages_from_pdf(str(pdf), str(d))[0],
            (left / LEAF[0], (y0 + 20) / LEAF[1],
             (left + 200) / LEAF[0], (y0 + 20 + 12 * 19) / LEAF[1]))


@pytest.mark.parametrize("gap", [60, 4])
def test_a_drawing_beside_a_table_is_still_a_drawing(vtb, tmp_path, gap):
    """Set apart from the table it was always found. Set against it, the two
    grouped as one, which overlapped the table enough to be taken for the
    table, and the drawing was left out with the rules."""
    from test_doubled_boxes import _problem_blocks
    page, fan = _table_beside_a_drawing(tmp_path, gap)
    table = _problem_blocks()[2]
    out = vtb.recover_vector_artwork([table], page)
    figs = [it for it in out if it.figure is not None]
    assert len(figs) == 1 and table in out
    assert figs[0].box == pytest.approx(fan, abs=0.01), "the fan, and not the table"
