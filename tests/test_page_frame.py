"""A page's fractions are of what the page draws; the PDF's coordinates are
of its user space. Where the crop box does not start at the media box's
corner the two disagree, and on 8,844 pages of 42 books on the shelf it does
not: a publisher's trim set 21 to 60 points in. The table check asked the
PDF for the text of a region a tenth of the page from the table it was
checking, and boxes found in the PDF were placed as far off.
"""
import json

import pytest

pymupdf = pytest.importorskip("pymupdf")


def _trimmed(tmp_path, at=(46, 23), size=(383, 609), media=(475, 655)):
    """A page whose crop box starts `at` points in from its media box, with a
    table of figures set near its top and prose near its foot."""
    doc = pymupdf.open()
    page = doc.new_page(width=media[0], height=media[1])
    x0, y0 = at
    page.set_cropbox(pymupdf.Rect(x0, y0, x0 + size[0], y0 + size[1]))
    # pymupdf places text in the crop box's own frame, top-down
    for k, row in enumerate(["1990 4,210 3,880", "1991 4,455 3,912", "1992 4,630 4,003"]):
        page.insert_text((40, 60 + 14 * k), row, fontsize=9)
    page.insert_text((40, 560), "The prose at the foot of the page.", fontsize=9)
    page.draw_rect(pymupdf.Rect(36, 46, 200, 96), color=(0, 0, 0), width=0.6)
    pdf = str(tmp_path / "trimmed.pdf")
    doc.save(pdf)
    p = str(tmp_path / "page_0000.png")
    open(p, "w").close()
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": pdf, "page": 0, "facsimile": False, "scan": False}, fh)
    return pdf, p


def test_the_frame_is_what_the_page_draws(vtb, tmp_path):
    import pypdfium2 as pdfium
    pdf, _ = _trimmed(tmp_path)
    page = pdfium.PdfDocument(pdf)[0]
    x0, y0, w, h = vtb._page_frame(page)
    assert (round(w), round(h)) == (383, 609)
    # pymupdf's top-down y for the crop box is pdfium's bottom-up origin from the media box
    assert (round(x0), round(y0)) == (46, 655 - 23 - 609)


def test_a_box_goes_there_and_back(vtb, tmp_path):
    import pypdfium2 as pdfium
    frame = vtb._page_frame(pdfium.PdfDocument(_trimmed(tmp_path)[0])[0])
    box = (0.1, 0.07, 0.52, 0.16)
    pdf = vtb._normalized_to_pdf(box, frame)
    again = vtb._pdf_box_to_normalized((pdf["left"], pdf["bottom"], pdf["right"], pdf["top"]), frame)
    assert all(abs(a - b) < 1e-9 for a, b in zip(box, again))


def test_the_text_of_a_box_is_the_text_in_that_box(vtb, tmp_path):
    """Asked for the table's region, the PDF gives the table and not a strip
    of margin and half a row."""
    _, page_path = _trimmed(tmp_path)
    text_in = vtb._typeset_text_in(page_path)
    table = text_in((0.08, 0.06, 0.55, 0.17))
    assert "4,210" in table and "4,630" in table and "1992" in table
    assert "prose" not in table
    assert "prose" in text_in((0.05, 0.89, 0.9, 0.94))


def test_a_drawn_box_in_the_pdf_is_placed_where_it_is_drawn(vtb, tmp_path):
    import pypdfium2 as pdfium
    pdf, _ = _trimmed(tmp_path)
    page = pdfium.PdfDocument(pdf)[0]
    frame = vtb._page_frame(page)
    rects = [vtb._pdf_box_to_normalized(o.get_bounds(), frame)
             for o in page.get_objects() if o.type == 2]
    want = (36 / 383, 46 / 609, 200 / 383, 96 / 609)
    assert any(all(abs(a - b) < 0.01 for a, b in zip(r, want)) for r in rects), rects


def test_a_short_table_is_found_on_a_trimmed_page(vtb, tmp_path):
    _, page_path = _trimmed(tmp_path)
    box = (0.08, 0.06, 0.55, 0.17)
    full = vtb.PageItem(html="<table><tr><td>1990</td><td>4,210</td><td>3,880</td></tr>"
                             "<tr><td>1991</td><td>4,455</td><td>3,912</td></tr>"
                             "<tr><td>1992</td><td>4,630</td><td>4,003</td></tr></table>", box=box)
    short = full._replace(html="<table><tr><td>1990</td><td>4,210</td></tr></table>")
    vtb_min = vtb.TABLE_MIN_FIGURES
    try:
        vtb.TABLE_MIN_FIGURES = 6
        assert vtb.tables_short_of_figures([full], page_path) == []
        assert vtb.tables_short_of_figures([short], page_path) == [(9, 2)]
    finally:
        vtb.TABLE_MIN_FIGURES = vtb_min
