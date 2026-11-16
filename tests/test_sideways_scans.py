"""A page printed sideways in a book scanned two pages to a sheet has its
body turned upright before it is read; its head and page number stay as
printed.

ST 31-91B is scanned two pages to a sheet, and on two sheets a page is
drawings captioned in type that runs down the page: the recogniser read
the captions in a different order every time. Which way is up is
tesseract's orientation check, asked when the pages are loaded. The turn
waits for the page's first reading: the blocks it calls furniture are cut
out and set back upright above and below, and only the rest is turned. A
sheet is turned a page at a time, the left page set above the right. A
single page set on its side is left: the recogniser reads those well.
"""
import json
import os
import shutil
import subprocess
import tempfile

import pytest

pymupdf = pytest.importorskip("pymupdf")
from PIL import Image, ImageDraw, ImageFont

from codicology import pipeline as P

needs_tesseract = pytest.mark.skipif(shutil.which("tesseract") is None,
                                     reason="tesseract is not installed")

PROSE = ("The medic should keep the dressings dry and clean at all times, and "
         "wrap each bundle in a double sheet of muslin before it is sterilized. "
         "Fold the corners over the article one after another, tuck the last "
         "under the others, and label the bundle with its contents and the date. ")


def _typed_page(words: str, size=(612, 792)) -> "Image.Image":
    doc = pymupdf.open()
    page = doc.new_page(width=size[0], height=size[1])
    page.insert_textbox(pymupdf.Rect(54, 90, size[0] - 54, size[1] - 54),
                        words * 6, fontsize=11)
    pix = page.get_pixmap(dpi=150)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def _with_folio(page: "Image.Image", folio: str) -> "Image.Image":
    """A page whose folio is printed upright at its head, as the sideways
    pages of the medical handbook have theirs."""
    page = page.copy()
    d = ImageDraw.Draw(page)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Courier New.ttf", 40)
    except OSError:
        font = ImageFont.load_default()
    d.text((page.width // 2 - 60, 30), folio, fill="black", font=font)
    return page


GUTTER = 40


def _sheet(left, right, gap=GUTTER) -> "Image.Image":
    """Two pages scanned side by side, the gutter's shadow between them."""
    s = Image.new("RGB", (left.width + right.width + gap, max(left.height, right.height)), "white")
    s.paste(left, (0, 0))
    s.paste(right, (left.width + gap, 0))
    ImageDraw.Draw(s).rectangle((left.width + gap // 2 - 3, 0, left.width + gap // 2 + 3, s.height),
                                fill=(70, 70, 70))
    return s


def _scan_pdf(tmp_path, sheets) -> str:
    """A PDF whose every page is one picture filling it: a scan."""
    doc = pymupdf.open()
    for k, im in enumerate(sheets):
        f = str(tmp_path / f"sheet{k}.png")
        im.save(f)
        page = doc.new_page(width=im.width * 72 / 150, height=im.height * 72 / 150)
        page.insert_image(page.rect, filename=f)
    out = str(tmp_path / "scan.pdf")
    doc.save(out)
    return out


def _dir(tmp_path, name):
    d = tmp_path / name
    d.mkdir()
    return str(d)


def _source(path):
    with open(path + ".source.json") as fh:
        return json.load(fh)


def _reads_upright(im) -> bool:
    said = P._which_way_up(im)
    return said is not None and said[0] == 0


def _words_in(im, psm="3") -> str:
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "p.png")
        im.save(f)
        return subprocess.run(["tesseract", f, "-", "--psm", psm],
                              capture_output=True, text=True).stdout


def _on_side(page: "Image.Image") -> "Image.Image":
    """The body printed a quarter turn clockwise, the folio left upright at
    the head: the page as the medical handbook prints it."""
    body = page.rotate(-90, expand=True)
    return body


def _book(tmp_path):
    """A book of sheets: the first upright, the second with its left page's
    body printed on its side, the third with both."""
    left, right = _typed_page("LEFTHAND " + PROSE), _typed_page("RIGHTHAND " + PROSE)
    return _scan_pdf(tmp_path, [_sheet(left, right), _sheet(_on_side(left), right),
                                _sheet(_on_side(left), _on_side(right))])


class _Read:
    """A cache holding each page's first reading: its furniture, where the
    test says it is."""

    def __init__(self, furniture=()):
        self.furniture, self.asked = list(furniture), []

    def get(self, path):
        self.asked.append(os.path.basename(path))
        return [P.PageItem(html="<p>words</p>", box=(0.1, 0.2, 0.9, 0.9))] + [
            P.PageItem(html=f"<p>{t}</p>", is_furniture=True, box=b) for t, b in self.furniture]


@needs_tesseract
def test_a_sideways_page_is_noted_when_loaded_and_left_as_it_lies(tmp_path):
    turns = str(tmp_path / "book.ocr.gz") + P.TURNS_SUFFIX
    pdf = _book(tmp_path)
    paths = P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"), turns_path=turns)
    src = [_source(p) for p in paths]
    assert "sideways" not in src[0]
    assert src[1]["sideways"] == {"degrees": [270, 0], "spread": True}
    assert src[2]["sideways"] == {"degrees": [270, 270], "spread": True}
    assert not any("turned" in s for s in src)      # nothing turned until it is read
    # what the check said is remembered under each sheet's name
    assert len(P.load_turns(turns)) == 3


@needs_tesseract
def test_the_body_is_turned_and_the_head_kept_upright(tmp_path):
    left = _with_folio(_on_side(_typed_page("LEFTHAND " + PROSE)), "F-4")
    right = _with_folio(_on_side(_typed_page("RIGHTHAND " + PROSE)), "F-5")
    sheet = _sheet(left, right)
    pdf = _scan_pdf(tmp_path, [sheet] * 3)
    paths = P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"))
    assert _source(paths[0])["sideways"]["degrees"] == [270, 270]
    # the first reading found the folios at the head of each page
    W, H = sheet.size
    x_r = left.width + GUTTER
    folio = lambda x: ((x + left.width // 2 - 70) / W, 20 / H, (x + left.width // 2 + 80) / W, 80 / H)
    cache = _Read([("F-4", folio(0)), ("F-5", folio(x_r))])
    turned = P.turn_sideways_bodies(paths, cache, backend=None)
    assert turned == [0, 1, 2]
    src = _source(paths[0])
    assert src["turned"]["degrees"] == [270, 270] and src["turned"]["furniture"] == 2
    assert src["facsimile"]          # nothing may be asked of the PDF by position on it
    with Image.open(paths[0]) as im:
        top = im.crop((0, 0, im.width, im.height // 2))
        foot = im.crop((0, im.height // 2, im.width, im.height))
        assert _reads_upright(top) and _reads_upright(foot)
        assert "LEFTHAND" in _words_in(top) and "RIGHTHAND" in _words_in(foot)
        # each folio set upright at the head of its page, as printed
        line = lambda y: _words_in(im.crop((0, y, im.width, y + 100)), psm="7")
        assert "F-4" in line(0) and "F-5" in line(im.height // 2)


@needs_tesseract
def test_an_upright_page_beside_a_sideways_one_is_left_as_printed(tmp_path):
    right = _typed_page("RIGHTHAND " + PROSE)
    sheet = _sheet(_on_side(_typed_page("LEFTHAND " + PROSE)), right)
    pdf = _scan_pdf(tmp_path, [sheet] * 3)
    paths = P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"))
    with Image.open(paths[0]) as loaded:
        half, _ = P._halves(loaded.convert("RGB"))[1]
    P.turn_sideways_bodies(paths, _Read(), backend=None)
    with Image.open(paths[0]) as im:
        # the right-hand page, below, is the very picture it was as the sheet
        # was cut at its gutter
        below = im.crop((0, im.height - half.height, half.width, im.height))
        diff = sum(abs(a - b) for a, b in zip(below.convert("L").getdata(),
                                               half.convert("L").getdata()))
        assert diff / (half.width * half.height) < 2


def test_without_a_first_reading_and_reading_refused_nothing_is_turned(tmp_path, monkeypatch):
    """--cache-only: a page with no reading in the cache is left, and is
    refused with the rest."""
    p = str(tmp_path / "page_0000.png")
    Image.new("RGB", (300, 200), "white").save(p)
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": "x.pdf", "page": 0, "facsimile": True, "scan": True,
                   "sideways": {"degrees": [270, 270], "spread": True}}, fh)

    class Empty:
        def get(self, path):
            return None
    monkeypatch.setattr(P, "CACHE_ONLY", True)
    paths = [p]
    assert P.turn_sideways_bodies(paths, Empty(), backend=object()) == []
    assert "turned" not in _source(p)


@needs_tesseract
def test_what_the_check_said_is_not_asked_again(tmp_path, monkeypatch):
    pdf = _book(tmp_path)
    turns = str(tmp_path / "book.ocr.gz") + P.TURNS_SUFFIX
    first = P.load_pages_from_pdf(pdf, _dir(tmp_path, "one"), turns_path=turns)
    monkeypatch.setattr(P, "_which_way_up", lambda im: pytest.fail("asked again"))
    again = P.load_pages_from_pdf(pdf, _dir(tmp_path, "two"), turns_path=turns)
    assert [P.page_key(p) for p in first] == [P.page_key(p) for p in again]
    assert _source(again[2])["sideways"]["degrees"] == [270, 270]


def test_a_single_page_on_its_side_is_left_as_it_lies(tmp_path, monkeypatch):
    """A register set landscape in the history of Lloyd's is read well as it
    lies, its running head upright above it."""
    upright = _typed_page(PROSE)
    pdf = _scan_pdf(tmp_path, [upright, upright.rotate(-90, expand=True), upright])
    monkeypatch.setattr(P, "_which_way_up", lambda im: pytest.fail("a single page was checked"))
    paths = P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"))
    assert all("sideways" not in _source(p) for p in paths)


def test_without_tesseract_nothing_is_turned(tmp_path, monkeypatch):
    pdf = _book(tmp_path)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    paths = P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"))
    assert all("sideways" not in _source(p) for p in paths)


def test_an_unsure_check_turns_nothing(tmp_path, monkeypatch):
    pdf = _book(tmp_path)
    monkeypatch.setattr(P, "_which_way_up", lambda im: (270, P.TURN_MIN_CONFIDENCE - 0.5))
    paths = P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"))
    assert all("sideways" not in _source(p) for p in paths)


def test_a_typeset_page_is_never_asked(tmp_path, monkeypatch):
    doc = pymupdf.open()
    for _ in range(2):
        page = doc.new_page(width=1224, height=792)
        page.insert_textbox(pymupdf.Rect(54, 54, 1170, 738), PROSE * 4, fontsize=11)
    pdf = str(tmp_path / "typeset.pdf")
    doc.save(pdf)
    monkeypatch.setattr(P, "_which_way_up", lambda im: pytest.fail("a typeset page was checked"))
    P.load_pages_from_pdf(pdf, _dir(tmp_path, "pages"))


def test_a_turned_figure_is_cut_from_the_page_as_turned(tmp_path):
    """The PDF draws the scan the old way up; a figure on the turned page is
    the page's own crop."""
    p = str(tmp_path / "page_0000.png")
    Image.new("RGB", (100, 140), "white").save(p)
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": str(tmp_path / "none.pdf"), "page": 0, "facsimile": True,
                   "scan": True, "turned": {"degrees": [270, 0], "spread": True}}, fh)
    crop = Image.new("RGB", (40, 30), "white")
    assert P.better_figure(p, (0.0, 0.0, 1.0, 1.0), crop) is crop
