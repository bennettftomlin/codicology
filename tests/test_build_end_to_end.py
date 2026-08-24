"""A whole build, driven: pages in, a valid EPUB out.

The test the codebase could not host until _read_pages was split out of
build_epub (S4): a scripted backend reads three synthetic pages, the full
assembly runs — furniture, dedupe, joins, decisions, nav — and the
shipped file is checked for the invariants every phase of the 2026-08-18
fix plan touched. Pass-ordering bugs like B1 lived precisely in the
stretch this exercises.
"""
import json
import re
import zipfile

import numpy as np
import cv2
import pytest


class ScriptedBackend:
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def __init__(self, vtb, pages):
        self.vtb = vtb
        self.pages = pages

    def run_items(self, images):
        out = []
        for _ in images:
            items = self.pages.pop(0) if self.pages else []
            out.append([self.vtb.PageItem(html=h) for h in items])
        return out


@pytest.fixture
def book_pages(tmp_path):
    paths = []
    for i in range(3):
        img = np.full((400, 300, 3), 255, np.uint8)
        cv2.rectangle(img, (40, 60 + i * 10), (260, 90 + i * 10), (30,) * 3, -1)
        p = tmp_path / f"page_{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def test_a_full_build_ships_a_valid_epub(vtb, tmp_path, book_pages,
                                         monkeypatch):
    # the witness guard re-reads pages that look inked but read empty;
    # scripted pages are inked rectangles, so give the guard its answer
    monkeypatch.setattr(vtb, "tesseract_words_on_page", lambda p: 12,
                        raising=False)
    backend = ScriptedBackend(vtb, [
        ["<p>he fled to Rus-</p>"],
        ["<p>sia at last, and the beligerents met him there.</p>"],
        ["<p>the end of the matter.</p>"],
    ])
    dec = tmp_path / "d.decisions.json"
    dec.write_text(json.dumps({"page_files": 3, "decisions": [
        {"page": 0, "occurrence": 0, "old": "Russia", "new": "RUSSIA",
         "source": "human", "rung": "human"}]}))
    out = tmp_path / "book.epub"
    vtb.build_epub(book_pages, str(out), backend, "E2E", False,
                   dedupe=False, drop_blank=False,
                   decisions_path=str(dec))
    z = zipfile.ZipFile(out)
    names = z.namelist()
    assert names[0] == "mimetype"
    pages = sorted(n for n in names if re.search(r"page_\d{4}\.xhtml$", n))
    assert len(pages) == 3
    p0 = z.read(pages[0]).decode()
    assert "RUSSIA" in p0, "join must run before the decision applies (B1)"
    p1 = z.read(pages[1]).decode()
    assert "sia at last" not in p1, "the fragment moved up with the join"
    nav = z.read(next(n for n in names if n.endswith("nav.xhtml"))).decode()
    ids = re.findall(r'id="([^"]+)"', nav)
    assert len(ids) == len(set(ids)), "nav ids must be unique (C1/C2)"


def test_the_calibre_flag_set_survives_a_blank_page(vtb, tmp_path,
                                                    book_pages, monkeypatch,
                                                    capsys):
    """The field failure of 2026-08-23: Calibre runs dedupe and drop_blank
    ON, the smoke test above ran them off, and both 'kept' report lines
    referenced a local the S4 extraction had moved into _read_pages. A
    blank fourth page drives the exact print that NameError'd on a real
    241-page book."""
    monkeypatch.setattr(vtb, "tesseract_words_on_page", lambda p: 12,
                        raising=False)
    monkeypatch.setattr(vtb, "_classical_word_count", lambda p, min_len=3:
                        None, raising=False)
    import numpy as np, cv2
    blank = tmp_path / "page_0003.png"
    cv2.imwrite(str(blank), np.full((400, 300, 3), 255, np.uint8))
    pages = book_pages + [str(blank)]
    backend = ScriptedBackend(vtb, [
        ["<p>he fled to Rus-</p>"],
        ["<p>sia at last, and the beligerents met him there.</p>"],
        ["<p>the end of the matter.</p>"],
        [],
    ])
    out = tmp_path / "book.epub"
    vtb.build_epub(pages, str(out), backend, "E2E", False,
                   dedupe=True, drop_blank=True)
    got = capsys.readouterr().out
    assert "dropped 1 blank page(s) (3 kept)" in got
    z = zipfile.ZipFile(out)
    shipped = [n for n in z.namelist() if re.search(r"page_\d{4}\.xhtml$", n)]
    assert len(shipped) == 3


class FurnishedBackend:
    """A book set the ordinary way: each chapter opens on a headless recto,
    then alternates a verso carrying the book's title with a recto carrying
    the chapter's."""
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def __init__(self, vtb, chapters):
        self.vtb = vtb
        self.pages = []
        for title, heading in chapters:
            self.pages.append([("<h2>%s</h2><p>It opens here.</p>" % heading,
                                False)])
            for k in range(5):
                self.pages.append([("<p>A Book of Tests</p>", True),
                                   ("<p>Verso prose %d.</p>" % k, False)])
                self.pages.append([("<p>%s</p>" % title, True),
                                   ("<p>Recto prose %d.</p>" % k, False)])

    def run_items(self, images):
        out = []
        for _ in images:
            page = self.pages.pop(0) if self.pages else []
            out.append([self.vtb.PageItem(html=h, is_furniture=f)
                        for h, f in page])
        return out


def test_a_chapter_title_reaches_the_shipped_page_as_h1(vtb, tmp_path,
                                                        monkeypatch):
    """The running heads name three chapters; the pages that open them are
    tagged h2 by the layout. The lift has to happen BEFORE the pages are
    written — it once ran forty lines too late, reported three promotions,
    and shipped none of them."""
    monkeypatch.setattr(vtb, "tesseract_words_on_page", lambda p: 12,
                        raising=False)
    backend = FurnishedBackend(vtb, [
        ("INTRODUCTION", "Introduction: Going Inside"),
        ("BEING THERE", "Being There"),
        ("DRAWING THE LINE", "Drawing the Line"),
    ])
    n_pages = len(backend.pages)
    paths = []
    for i in range(n_pages):
        img = np.full((400, 300, 3), 255, np.uint8)
        cv2.rectangle(img, (40, 60), (260, 90), (30,) * 3, -1)
        p = tmp_path / f"page_{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    out = tmp_path / "book.epub"
    vtb.build_epub(paths, str(out), backend, "Furnished", False,
                   dedupe=False, drop_blank=False)
    z = zipfile.ZipFile(out)
    shipped = "".join(
        z.read(n).decode() for n in sorted(z.namelist())
        if re.search(r"page_\d{4}\.xhtml$", n))
    for want in ("Introduction: Going Inside", "Being There",
                 "Drawing the Line"):
        assert f"<h1>{want}</h1>" in shipped, f"{want} did not ship as h1"


def test_a_build_stops_rather_than_shipping_pages_a_dead_server_blanked(
        vtb, tmp_path, book_pages, monkeypatch):
    """The whole point of the probe: a page not in the cache is read live,
    and an unserved backend answers with emptiness rather than an error, so
    the pages ship blank and the emptiness is cached. The run must stop
    before writing any of that."""
    monkeypatch.setattr(vtb, "tesseract_words_on_page", lambda p: 12,
                        raising=False)

    class DeadServer:
        batch_size = 4
        name = "surya"
        langs = ["en"]

        def run_items(self, images):
            return [[] for _ in images]

    out = tmp_path / "book.epub"
    with pytest.raises(SystemExit) as e:
        vtb.build_epub(book_pages, str(out), DeadServer(), "E2E", False,
                       dedupe=False, drop_blank=False)
    msg = str(e.value)
    assert "not reading" in msg and "doctor --full" in msg
    assert not out.exists(), "nothing may be written on a dead backend"
