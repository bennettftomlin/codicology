"""A recogniser that stops answering stops the build.

Where There Is No Doctor was built on a day the server behind recognition
stopped answering after page 119 of 503. Nothing raised. The layout still
came back and the pictures were still cut out, so 64 pages were kept as
their drawings alone, cached as readings, and 19,866 words went missing;
318 more were laid down from the PDF's text as one paragraph each. The
build said so in a line at the end and the book was installed.

Now a page that comes back without its words, when words were sent to be
read, is not a reading; and the first such page of a run makes the backend
prove it can still read before the book goes on.
"""
import os
import re
import zipfile

import pytest
from PIL import Image, ImageDraw

from codicology import pipeline as vtb
from codicology.doctor import _TEST_LINES
from test_surya_labels import _Block, _Page, _backend

LAYER = ("Wash the wound well with soap and boiled water, then cover it "
         "with a clean cloth and keep the person lying down and warm until "
         "the bleeding has stopped; give plenty of liquids if they can "
         "drink, and send for help at once when the wound is deep.")


def _picture(i=0):
    """A drawing, different on every page: one that repeats from page to
    page is taken for a printer's ornament and dropped."""
    im = Image.new("RGB", (160, 120), "white")
    d = ImageDraw.Draw(im)
    d.ellipse((20, 20, 140 - 9 * i, 100), outline="black", width=4)
    for k in range(i + 1):
        d.line((20 + 12 * k, 30, 30 + 12 * k, 90 - 5 * i), fill="black", width=3)
    return im


@pytest.fixture
def pages(tmp_path):
    """Six inked pages, each with a text layer; told apart by their ink."""
    def make(n, born_digital=True, layer=LAYER):
        paths = []
        for i in range(n):
            im = Image.new("RGB", (300, 400), "white")
            d = ImageDraw.Draw(im)
            for row in range(40, 360, 24):
                d.rectangle((30, row, 250 + i, row + 9), fill=(30, 30, 30))
            p = tmp_path / f"page_{i:04d}.png"
            im.save(p)
            if layer:
                (tmp_path / f"page_{i:04d}.png.layer.txt").write_text(layer)
            if born_digital:
                (tmp_path / f"page_{i:04d}.png.native").write_text("")
            paths.append(str(p))
        return paths
    return make


class Scripted:
    """A Surya whose every page is scripted: "words", "pictures" (the
    layout and the pictures, every block of text missing), "plate" (a page
    that is all picture, read properly) or "nothing". `alive` is whether it
    can still read the proof page."""
    name = "surya"
    batch_size = 2
    langs = ["en"]

    def __init__(self, paths, script, alive=True):
        self.script = {Image.open(p).convert("RGB").tobytes(): (i, s)
                       for i, (p, s) in enumerate(zip(paths, script))}
        self.alive, self.read = alive, []

    def run_items(self, images):
        out, self.last_unread_blocks = [], []
        for im in images:
            if im.size == (1400, 500):                     # the proof page
                out.append([vtb.PageItem(html=f"<p>{' '.join(_TEST_LINES)}</p>")]
                           if self.alive else [])
                self.last_unread_blocks.append(0 if self.alive else 2)
                continue
            i, what = self.script[im.convert("RGB").tobytes()]
            self.read.append(i)
            if what == "words":
                out.append([vtb.PageItem(html=f"<h2>Page {i}</h2>", label="SectionHeader"),
                            vtb.PageItem(html=f"<p>Leaf {i}: {LAYER}</p>", label="Text")])
                self.last_unread_blocks.append(0)
            elif what in ("pictures", "plate"):
                out.append([vtb.PageItem(figure=_picture(i), label="Picture",
                                         box=(0.1, 0.1, 0.6, 0.4))])
                self.last_unread_blocks.append(3 if what == "pictures" else 0)
            else:
                out.append([])
                self.last_unread_blocks.append(0)
        return out


def _dying(backend, first_dead):
    """The server behind `backend` stops the moment page `first_dead` is
    asked for, and stays stopped: the proof page included."""
    real = backend.run_items

    def run_items(images):
        if any(im.size != (1400, 500) and
               backend.script[im.convert("RGB").tobytes()][0] >= first_dead
               for im in images):
            backend.alive = False
        return real(images)
    backend.run_items = run_items
    return backend


def _build(tmp_path, paths, backend, cache=None):
    out = tmp_path / "book.epub"
    vtb.build_epub(paths, str(out), backend, "Book", False, dedupe=False,
                   drop_blank=False, ocr_cache=cache)
    return zipfile.ZipFile(out)


def _page(z, i):
    return z.read(f"EPUB/page_{i:04d}.xhtml").decode()


# ── the backend says what came back unread ───────────────────────────────────

def test_surya_counts_the_blocks_that_came_back_with_nothing():
    page = Image.new("RGB", (800, 1000), "white")
    ImageDraw.Draw(page).ellipse((200, 300, 600, 700), fill="black")
    dead = [_Block("Text", (80, 80, 720, 200), html=""),
            _Block("Picture", (180, 280, 620, 720), skipped=True, order=1),
            _Block("Caption", (180, 740, 620, 780), html="", order=2)]
    dead[2].error = True
    well = [_Block("Text", (80, 80, 720, 200), "<p>As the picture shows,</p>"),
            _Block("Picture", (180, 280, 620, 720), skipped=True, order=1)]
    backend = _backend([_Page(dead, page.size), _Page(well, page.size)])
    items = backend.run_items([page, page])
    assert backend.last_unread_blocks == [2, 0]
    assert [it.figure is not None for it in items[0]] == [True]
    backend._rec = lambda images: [_Page([_Block("Picture", (0, 0, 800, 1000),
                                                 skipped=True)], page.size)]
    backend.run_items([page])
    assert backend.last_unread_blocks == [0], "a plate is not an unread page"


# ── the build ────────────────────────────────────────────────────────────────

def test_a_recogniser_that_stops_stops_the_build(tmp_path, pages, capsys):
    paths = pages(6)
    cache = str(tmp_path / "c.ocr.gz")
    backend = _dying(Scripted(paths, ["words", "words"] + ["pictures"] * 4), 2)

    with pytest.raises(SystemExit) as stop:
        _build(tmp_path, paths, backend, cache)
    said = str(stop.value)
    assert "the surya backend has stopped reading" in said
    assert "page_0002.png came back without its text" in said
    assert "read nothing at all" in said
    assert "carries on from page_0002.png" in said
    assert "No book was written" in said
    assert not os.path.exists(tmp_path / "book.epub")

    kept = vtb.OCRCache(cache, "surya", ["en"])
    assert [kept.get(p) is not None for p in paths] == [True, True] + [False] * 4


def test_the_next_run_carries_on_from_where_it_stopped(tmp_path, pages):
    paths = pages(6)
    cache = str(tmp_path / "c.ocr.gz")
    sick = _dying(Scripted(paths, ["words", "words"] + ["pictures"] * 4), 2)
    with pytest.raises(SystemExit):
        _build(tmp_path, paths, sick, cache)

    well = Scripted(paths, ["words"] * 6)
    z = _build(tmp_path, paths, well, cache)
    assert sorted(set(well.read)) == [2, 3, 4, 5]
    for i in range(6):
        assert f"Page {i}</h2>" in _page(z, i) and f"Leaf {i}:" in _page(z, i)


def test_hard_pages_do_not_stop_a_backend_that_still_reads(tmp_path, pages,
                                                           capsys):
    """Three pages running without their text, and the proof page reads:
    the pages are hard, the backend is not dead. Each keeps its pictures,
    takes its words from its own layer, and is not cached."""
    paths = pages(5)
    cache = str(tmp_path / "c.ocr.gz")
    backend = Scripted(paths, ["words", "pictures", "pictures", "pictures",
                               "words"])
    proofs = []
    backend.run_items = (lambda images, real=backend.run_items:
                         (proofs.append(1) if images[0].size == (1400, 500)
                          else None, real(images))[1])
    z = _build(tmp_path, paths, backend, cache)
    assert len(proofs) == 3, "before the build, at the first, at the third"
    log = capsys.readouterr().out
    assert log.count("came back with nothing, twice") == 3
    assert "restored from the page's own born-digital layer; not cached" in log
    for i in (1, 2, 3):
        page = _page(z, i)
        assert "<img" in page and "Wash the wound well" in page
    kept = vtb.OCRCache(cache, "surya", ["en"])
    assert [kept.get(p) is not None for p in paths] == [True, False, False,
                                                        False, True]


def test_a_scan_keeps_its_pictures_and_says_its_text_is_missing(tmp_path, pages,
                                                                capsys):
    paths = pages(3, born_digital=False, layer=None)
    cache = str(tmp_path / "c.ocr.gz")
    z = _build(tmp_path, paths, Scripted(paths, ["words", "pictures", "words"]),
               cache)
    assert "left without its text; not cached" in capsys.readouterr().out
    page = _page(z, 1)
    assert "<img" in page and "Wash the wound" not in page
    assert vtb.OCRCache(cache, "surya", ["en"]).get(paths[1]) is None


def test_a_page_that_is_all_picture_is_a_reading_and_is_kept(tmp_path, pages,
                                                             capsys):
    """A plate has no words to lose. Nothing was sent to be read, so nothing
    came back unread, however many words its caption's layer may hold."""
    paths = pages(4)
    cache = str(tmp_path / "c.ocr.gz")
    backend = Scripted(paths, ["words", "plate", "plate", "plate"])
    proofs = []
    backend.run_items = (lambda images, real=backend.run_items:
                         (proofs.append(1) if images[0].size == (1400, 500)
                          else None, real(images))[1])
    z = _build(tmp_path, paths, backend, cache)
    assert "came back with nothing" not in capsys.readouterr().out
    assert len(proofs) == 1, "and nobody asked it to prove a thing"
    assert "<img" in _page(z, 2) and "Wash the wound" not in _page(z, 2)
    kept = vtb.OCRCache(cache, "surya", ["en"])
    assert all(kept.get(p) is not None for p in paths)


def test_a_page_read_as_nothing_at_all_asks_the_same_question(tmp_path, pages):
    """The other shape of a dead server: not even the layout comes back."""
    paths = pages(5, born_digital=False)
    backend = _dying(Scripted(paths, ["words", "nothing", "nothing", "nothing",
                                      "nothing"]), 1)
    with pytest.raises(SystemExit) as stop:
        _build(tmp_path, paths, backend, str(tmp_path / "c.ocr.gz"))
    assert "page_0001.png came back without its text" in str(stop.value)


def test_one_unread_page_is_enough_when_it_is_all_there_is_to_read(tmp_path,
                                                                   pages):
    """The Knights of Industry has two pages its cache has never held. A
    rebuild met a server refusing connections, read those two as nothing,
    dropped one as blank, and so had one page fewer than the book's
    reviewed corrections were recorded against: all seven were refused.
    Neither page has a layer worth the name, and there were never going to
    be three of them."""
    paths = pages(4, born_digital=False, layer=None)
    cache = str(tmp_path / "c.ocr.gz")
    _build(tmp_path, paths[:3], Scripted(paths[:3], ["words"] * 3), cache)

    sick = _dying(Scripted(paths, ["words", "words", "words", "nothing"]), 3)
    (tmp_path / "book.epub").unlink()
    with pytest.raises(SystemExit) as stop:
        _build(tmp_path, paths, sick, cache)
    assert "page_0003.png came back without its text" in str(stop.value)
    assert not (tmp_path / "book.epub").exists()
    assert sick.read == [3, 3], "the page and its second look; the rest were cached"


def test_a_long_run_of_hard_pages_is_asked_about_again(tmp_path, pages):
    paths = pages(8)
    backend = Scripted(paths, ["words"] + ["pictures"] * 7)
    proofs = []
    backend.run_items = (lambda images, real=backend.run_items:
                         (proofs.append(1) if images[0].size == (1400, 500)
                          else None, real(images))[1])
    _build(tmp_path, paths, backend, str(tmp_path / "c.ocr.gz"))
    assert len(proofs) == 4, "before the build, then at the first, third, sixth"


def test_a_run_of_silence_ends_with_a_page_that_reads(tmp_path, pages):
    """Two, a good page, then two more: two runs, each asked about once."""
    paths = pages(6)
    backend = Scripted(paths, ["words", "pictures", "pictures", "words",
                               "pictures", "pictures"])
    proofs = []
    backend.run_items = (lambda images, real=backend.run_items:
                         (proofs.append(1) if images[0].size == (1400, 500)
                          else None, real(images))[1])
    _build(tmp_path, paths, backend, str(tmp_path / "c.ocr.gz"))
    assert len(proofs) == 3, "the one every build begins with, and one a run"
