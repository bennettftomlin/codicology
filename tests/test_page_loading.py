"""The single-image shortcut may only fire on a page facsimile."""
import os
import pytest
pdfium = pytest.importorskip("pypdfium2")
from PIL import Image

SHADOW = "/Users/Bennett1/claude_knowledge/assets/books/Shadow Libraries - Joe Karaganis.pdf"


@pytest.mark.skipif(not os.path.exists(SHADOW), reason="source book absent")
def test_a_wide_chart_on_a_text_page_is_not_taken_as_the_page(vtb, tmp_path):
    """
    Shadow Libraries p259 holds one 4068x1457 chart beside 1,395 characters
    of vector prose. Extracting the chart as "the page" fed a textless image
    to the recogniser, which invented a table from it — four separate times.
    A page facsimile must be page-shaped; a chart is not.
    """
    src = pdfium.PdfDocument(SHADOW)
    one = pdfium.PdfDocument.new()
    one.import_pages(src, [259])
    slice_path = str(tmp_path / "slice.pdf")
    one.save(slice_path)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    pages = vtb.load_pages_from_pdf(slice_path, str(pages_dir))
    with Image.open(pages[0]) as im:
        w, h = im.size
    assert 0.7 <= (w / h) / (504 / 648) <= 1.3, \
        f"got {w}x{h}: the chart was extracted as the page again"
    # and the page text rode along as the witness sidecar
    assert os.path.exists(pages[0] + ".layer.txt")


DEMOCRACY = ("/Users/Bennett1/claude_knowledge/assets/books/"
             "Democracy-in-Brief_In-Brief-Series_English_Lo-Res.pdf")


@pytest.mark.skipif(not os.path.exists(DEMOCRACY), reason="source book absent")
def test_a_photo_with_a_caption_beside_it_is_not_taken_as_the_page(vtb, tmp_path):
    """
    Democracy in Brief sets a photograph over about 0.87 of each plate page
    with the caption below it. Extracting the photograph as "the page" left
    the caption unread on all eight plates — the same loss as Shadow
    Libraries' chart, at a coverage the aspect guard could not catch.
    A facsimile IS the page and measures 1.00.
    """
    src = pdfium.PdfDocument(DEMOCRACY)
    one = pdfium.PdfDocument.new()
    one.import_pages(src, [1])
    slice_path = str(tmp_path / "slice.pdf")
    one.save(slice_path)
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    pages = vtb.load_pages_from_pdf(slice_path, str(pages_dir))
    # rendered whole: the page keeps its own proportions and gains the
    # born-digital marker, so its caption text is available
    with Image.open(pages[0]) as im:
        w, h = im.size
    assert 0.9 <= (w / h) / (306 / 396) <= 1.1, f"got {w}x{h}"
    assert os.path.exists(pages[0] + ".native")
    layer = open(pages[0] + ".layer.txt").read()
    assert "Magna Carta" in layer


@pytest.mark.skipif(not os.path.exists(SHADOW), reason="source book absent")
def test_every_rendered_page_is_stored_losslessly(vtb, tmp_path):
    """
    The raster is what the recogniser reads and what a figure falls back to
    when it cannot be taken from the PDF, so loss here is permanent — and it
    would land hardest on photographic plates, which is exactly where a
    size-based choice would have picked JPEG. Prose and plates alike are
    kept lossless; the extra bytes live in a directory the run deletes.
    """
    dem = ("/Users/Bennett1/claude_knowledge/assets/books/"
           "Democracy-in-Brief_In-Brief-Series_English_Lo-Res.pdf")
    cases = [(SHADOW, 20, "prose")]
    if os.path.exists(dem):
        cases.append((dem, 1, "photographic plate"))
    for path, page_no, what in cases:
        src = pdfium.PdfDocument(path)
        one = pdfium.PdfDocument.new()
        one.import_pages(src, [page_no])
        p = str(tmp_path / f"{what.split()[0]}.pdf"); one.save(p)
        d = tmp_path / what.split()[0]; d.mkdir()
        got = vtb.load_pages_from_pdf(p, str(d))[0]
        assert got.endswith(".png"), f"{what} page stored lossily: {got}"


SCANS = {
    "indian_trails": "public-gdcmassbookdig-indiantrailscent00haub-indiantrailscent00haub.pdf",
    "conquest": "public-gdcmassbookdig-conquestofoldnor02bald-conquestofoldnor02bald.pdf",
    "eagle": "Eagle Forgotten_ The Life of John Peter Altgeld -- Harry Barnard.pdf",
}
BORN_DIGITAL = {
    "fm21-76": "FM21-76_SurvivalManual.pdf",
    "democracy": "Democracy-in-Brief_In-Brief-Series_English_Lo-Res.pdf",
}
ASSETS = "/Users/Bennett1/claude_knowledge/assets/books"


@pytest.mark.parametrize("name,fn", sorted(SCANS.items()))
def test_a_scan_layer_is_never_treated_as_the_publishers_text(vtb, tmp_path, name, fn):
    """
    A library scan carries somebody else's OCR, and treating it as
    authoritative substitutes a worse reading for ours — sixty words of one
    book, plus two blank endpapers filled with OCR of an embossed seal.
    A scan has a picture of the whole page behind its text; that is the tell,
    not whether the page happened to be rendered.
    """
    src = f"{ASSETS}/{fn}"
    if not os.path.exists(src):
        pytest.skip("source absent")
    doc = pdfium.PdfDocument(src)
    one = pdfium.PdfDocument.new()
    one.import_pages(doc, [min(20, len(doc) - 1)])
    p = str(tmp_path / "s.pdf"); one.save(p)
    d = tmp_path / "pages"; d.mkdir()
    page = vtb.load_pages_from_pdf(p, str(d))[0]
    assert not os.path.exists(page + ".native"), \
        f"{name}: a scan was marked as the publisher's own text"


@pytest.mark.parametrize("name,fn", sorted(BORN_DIGITAL.items()))
def test_a_born_digital_page_keeps_its_authority(vtb, tmp_path, name, fn):
    """The rule must not overreach: a photographic plate covers at most about
    nine tenths of its page, and its text is still the publisher's."""
    src = f"{ASSETS}/{fn}"
    if not os.path.exists(src):
        pytest.skip("source absent")
    doc = pdfium.PdfDocument(src)
    one = pdfium.PdfDocument.new()
    one.import_pages(doc, [1])
    p = str(tmp_path / "b.pdf"); one.save(p)
    d = tmp_path / "pages"; d.mkdir()
    page = vtb.load_pages_from_pdf(p, str(d))[0]
    assert os.path.exists(page + ".native"), \
        f"{name}: born-digital text lost its authority"
