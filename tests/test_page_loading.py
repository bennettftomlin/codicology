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


# ---- a scan that does not fill its page --------------------------------

def _png(size, draw):
    import io
    img = Image.new("RGB", size, "white")
    draw(img)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def _printed(img):
    """Dark bars where the lines of type would be."""
    px = img.load()
    for row in range(40, img.height - 40, 36):
        for x in range(30, img.width - 30):
            for y in range(row, row + 10):
                px[x, y] = (25, 25, 25)


def _plate(img):
    """Something no line of type looks like."""
    px = img.load()
    for x in range(img.width):
        for y in range(img.height):
            px[x, y] = ((x * 5) % 256, (y * 3) % 256, ((x + y) * 2) % 256)


def make_book(path, pages):
    """A PDF of `pages`, each a list of ("image", rect, bytes) and
    ("text", (x, y), words) drawn on a 466x704pt leaf — the size and the
    margins Google Books gives a scanned octavo."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    for parts in pages:
        page = doc.new_page(width=466, height=704)
        for kind, where, what in parts:
            if kind == "image":
                page.insert_image(pymupdf.Rect(*where), stream=what)
            else:
                page.insert_text(where, what, fontsize=9)
    doc.save(str(path))
    return str(path)


TYPE_AREA = (40, 45, 426, 659)          # 0.72 of the leaf, as Google sets it


def scan_page(layer=False, extra=()):
    parts = [("image", TYPE_AREA, _png((386, 614), _printed))]
    if layer:                           # a recogniser's text, over the ink
        parts += [("text", (60, 80 + 36 * k), "the scanner read this line")
                  for k in range(12)]
    return parts + list(extra)


def typeset_page():
    return [("text", (50, 60 + 14 * k), "The publisher set this line in type.")
            for k in range(40)]


def _sidecar(page):
    import json
    with open(page + ".source.json") as fh:
        return json.load(fh)


def test_a_scan_set_inside_its_margins_is_still_a_scan(vtb, tmp_path):
    """
    Google Books crops each page image to the type area and sets it inside
    the margins, at 0.72 of the leaf. Nothing about that page is typeset,
    but "a picture of the whole page" it is not, and a history of Lloyd's
    shipped with the scan of every one of its 438 pages inserted as a figure
    above that page's text.
    """
    p = make_book(tmp_path / "scan.pdf", [scan_page() for _ in range(4)])
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    for page in pages:
        src = _sidecar(page)
        assert src["scan"] is True
        assert src["facsimile"] is False, "the page was rendered, not extracted"
        assert page.endswith(".png")


def test_a_layer_inside_a_margin_set_scan_is_not_the_publishers(vtb, tmp_path):
    """
    The same shape with a recogniser's text over it: the picture covers
    under nine tenths of the leaf, so the layer passed for typesetting and
    Confusion de Confusiones had sixty-three pages of its scanner's reading
    handed to the reconciler as the author's.
    """
    p = make_book(tmp_path / "scan.pdf",
                  [scan_page(layer=True) for _ in range(4)])
    d = tmp_path / "pages"; d.mkdir()
    for page in vtb.load_pages_from_pdf(p, str(d)):
        assert os.path.exists(page + ".layer.txt"), "the layer still testifies"
        assert not os.path.exists(page + ".native"), \
            "a scanner's layer was marked as the publisher's own text"


def test_one_plate_does_not_make_a_typeset_book_a_scan(vtb, tmp_path):
    """A frontispiece is a picture and nothing else, in a book that is type
    from there on. The pages count only where they are most of the book."""
    plate = [("image", (60, 80, 406, 600), _png((346, 520), _plate))]
    p = make_book(tmp_path / "typeset.pdf",
                  [plate] + [typeset_page() for _ in range(4)])
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(p, str(d))
    assert [_sidecar(page)["scan"] for page in pages] == [False] * 5
    for page in pages[1:]:
        assert os.path.exists(page + ".native"), \
            "typeset text lost its authority"


def test_a_plate_with_its_caption_is_not_a_picture_of_itself(vtb, tmp_path):
    """What separates a plate from a scan is the type printed beside it."""
    plate = ("image", (60, 80, 406, 600), _png((346, 520), _plate))
    p = make_book(tmp_path / "three.pdf", [
        [plate, ("text", (60, 630), "PLATE IV. The Royal Exchange in 1810.")],
        [plate, ("text", (120, 300), "words the picture itself carries")],
        [("image", (150, 250, 300, 400), _png((150, 150), _plate))],
    ])
    doc = pdfium.PdfDocument(p)
    assert [vtb._is_picture_of_itself(page) for page in doc] \
        == [False, True, False]


# ── a long book is rendered by several processes ─────────────────────────────

def _everything_in(d):
    return {name: open(os.path.join(d, name), "rb").read()
            for name in sorted(os.listdir(d))}


def test_worker_processes_write_what_one_process_writes(vtb, tmp_path,
                                                        monkeypatch):
    """
    Rendering was one page at a time on a ten-core machine: 126 of a 202
    second warm rebuild. The cache is keyed on a page's bytes, so the
    workers may not differ from the loop they replace by a single byte —
    in the pages, their order, or any sidecar beside them.
    """
    plate = [("image", (60, 80, 400, 560), _png((340, 480), _plate))]
    book = make_book(tmp_path / "book.pdf",
                     [plate] + [typeset_page() for _ in range(4)]
                     + [scan_page(layer=True)] + [typeset_page()])
    alone, shared = tmp_path / "alone", tmp_path / "shared"
    alone.mkdir(); shared.mkdir()

    monkeypatch.setattr(vtb, "PDF_POOL_MIN_PAGES", 10 ** 9)
    one = vtb.load_pages_from_pdf(book, str(alone))
    monkeypatch.setattr(vtb, "PDF_POOL_MIN_PAGES", 2)
    monkeypatch.setattr(vtb, "PDF_POOL_WORKERS", 3)
    assert vtb._pdf_pool_workers(7) == min(3, os.cpu_count() or 1)
    # whoever runs several builds at once may say how many each starts
    monkeypatch.setenv(vtb.PDF_WORKERS_ENV, "1")
    assert vtb._pdf_pool_workers(7) == 1
    monkeypatch.setenv(vtb.PDF_WORKERS_ENV, "many")
    assert vtb._pdf_pool_workers(7) == min(3, os.cpu_count() or 1)
    monkeypatch.delenv(vtb.PDF_WORKERS_ENV)
    many = vtb.load_pages_from_pdf(book, str(shared))

    assert [os.path.basename(p) for p in many] == \
           [os.path.basename(p) for p in one] == \
           [f"page_{i:04d}.png" for i in range(7)]
    # the sidecar names the directory's own PDF, so the two trees can match
    assert _everything_in(shared) == _everything_in(alone)


def test_a_short_book_starts_no_workers(vtb):
    assert vtb._pdf_pool_workers(vtb.PDF_POOL_MIN_PAGES - 1) == 1


def test_a_failed_worker_costs_time_and_not_the_book(vtb, tmp_path,
                                                     monkeypatch, capsys):
    book = make_book(tmp_path / "book.pdf", [typeset_page() for _ in range(3)])
    d = tmp_path / "pages"; d.mkdir()
    monkeypatch.setattr(vtb, "PDF_POOL_MIN_PAGES", 2)
    monkeypatch.setattr(vtb, "PDF_POOL_WORKERS", 2)
    monkeypatch.setattr(vtb, "_PDF_WORKER", "import sys; sys.exit(3)")
    pages = vtb.load_pages_from_pdf(book, str(d))
    assert "a page-rendering worker failed" in capsys.readouterr().out
    assert len(pages) == 3 and all(os.path.exists(p) for p in pages)
