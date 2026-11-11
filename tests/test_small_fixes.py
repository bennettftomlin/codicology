"""Small defects from the audit of 2026-11-10, each with the case that showed it."""
import gzip
import os
import re
import sys
import zipfile

import numpy as np
import pytest
from PIL import Image

from test_counting_tables import PageBackend


# ── a chapter's number ends where it ends ────────────────────────────────────

@pytest.mark.parametrize("head", [
    "<h1>Chapter Introduction</h1>",        # was chapter I, "ntroduction"
    "<h2>Chapter Vital Signs</h2>",         # was chapter V, "ital Signs"
    "<h1>Chapter Limits</h1>",
    "<h1>Chapter 1984</h1>",                # not chapter 198, title "4"
    "<h1>Chapter Onerous Duties</h1>",
])
def test_a_word_that_begins_like_a_numeral_is_not_a_chapter_number(vtb, head):
    assert vtb.CHAPTER_HEAD.search(head) is None


@pytest.mark.parametrize("head, number, title", [
    ("<h1>Chapter I</h1>", "I", ""),
    ("<h1>CHAPTER IV. The Yards</h1>", "IV", "The Yards"),
    ("<h2>Chapter 12: Scarcity</h2>", "12", "Scarcity"),
    ("<h1>Chapter V—Vital Signs</h1>", "V", "Vital Signs"),
    ("<h1>Chapter Twenty-One</h1>", "Twenty-One", ""),
])
def test_a_real_chapter_head_still_reads(vtb, head, number, title):
    m = vtb.CHAPTER_HEAD.search(head)
    assert m and (m.group(3), m.group(4)) == (number, title)


def test_chapter_introduction_keeps_its_title(vtb):
    bodies = ["<h1>Chapter Introduction</h1><p>Prose.</p>"]
    vtb.normalize_chapter_heads(bodies)
    assert "Chapter Introduction" in bodies[0] and "ntroduction</h1>" in bodies[0]
    assert "Chapter I." not in bodies[0]


# ── a page's language is the book's ──────────────────────────────────────────

def test_every_page_carries_the_books_language(vtb, tmp_path):
    """Each page was stamped lang="en" whatever --lang said."""
    paths = []
    for i in range(2):
        p = tmp_path / f"page_{i:04d}.png"
        im = Image.new("RGB", (300, 400), "white")
        for x in range(40, 260):
            for y in range(60 + 10 * i, 70 + 10 * i):
                im.putpixel((x, y), (30, 30, 30))
        im.save(p)
        paths.append(str(p))
    backend = PageBackend(vtb, paths, [[("<p>Erste Seite, deutsch.</p>", "Text")],
                                       [("<p>Zweite Seite.</p>", "Text")]])
    backend.langs = ["de", "en"]
    out = tmp_path / "buch.epub"
    vtb.build_epub(paths, str(out), backend, "Buch", False, dedupe=False,
                   drop_blank=False)
    z = zipfile.ZipFile(out)
    assert "<dc:language>de</dc:language>" in z.read("EPUB/content.opf").decode()
    for name in ("EPUB/page_0000.xhtml", "EPUB/page_0001.xhtml"):
        page = z.read(name).decode()
        assert re.search(r'<html[^>]*\blang="de"', page), page[:200]
        assert 'lang="en"' not in page


# ── the output may not be the book being read ────────────────────────────────

def _pdf(path):
    Image.new("RGB", (300, 400), "white").save(path, "PDF")
    return str(path)


@pytest.mark.parametrize("spelling", ["same", "relative", "link"])
def test_writing_over_the_source_is_refused(vtb, tmp_path, monkeypatch,
                                            spelling):
    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    src = _pdf(tmp_path / "book.pdf")
    before = open(src, "rb").read()
    out = src
    if spelling == "relative":
        monkeypatch.chdir(tmp_path)
        out = "./book.pdf"
    elif spelling == "link":
        out = str(tmp_path / "also-book.pdf")
        os.symlink(src, out)
    with pytest.raises(SystemExit) as stop:
        vtb.main(["--pages-from", src, out])
    assert "is the --pages-from source itself" in str(stop.value)
    assert open(src, "rb").read() == before


def test_an_epub_named_as_the_source_is_refused_too(vtb, tmp_path,
                                                    monkeypatch):
    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    src = _pdf(tmp_path / "book.pdf")
    with pytest.raises(SystemExit) as stop:
        vtb.main(["--pages-from", src, "--epub", src])
    assert "is the --pages-from source itself" in str(stop.value)


# ── a spread without scikit-image ────────────────────────────────────────────

def test_a_spread_is_cut_straight_when_the_seam_finder_is_absent(
        vtb, make_spread, monkeypatch, capsys):
    """scikit-image is in the `geometry` extra and was imported here without
    asking: a photographed spread ended the capture in an ImportError."""
    spread = make_spread()
    with_it = vtb._gutter_path(spread)

    monkeypatch.setitem(sys.modules, "skimage", None)
    monkeypatch.setitem(sys.modules, "skimage.graph", None)
    monkeypatch.setattr(vtb, "_SEAM_UNAVAILABLE_SAID", False)
    without = vtb._gutter_path(spread)
    assert without.shape == with_it.shape
    assert float(without.max()) == float(without.min()), "a straight cut"
    assert abs(float(without[0]) - float(np.median(with_it))) < 0.05 * spread.shape[1]
    vtb._gutter_path(spread)
    said = capsys.readouterr().out
    assert said.count("scikit-image is not installed") == 1, "said once"


# ── a cache cut short ────────────────────────────────────────────────────────

def test_a_truncated_cache_is_a_cold_cache(vtb, tmp_path):
    path = str(tmp_path / "c.ocr.gz")
    page = tmp_path / "p.png"
    page.write_bytes(b"\x89PNG a")
    whole = vtb.OCRCache(path, "surya", ["en"])
    for k in range(40):
        other = tmp_path / f"q{k}.png"
        other.write_bytes(b"\x89PNG %d" % k)
        whole.put(str(other), [vtb.PageItem(html=f"<p>{os.urandom(400).hex()}</p>")])
    whole.save()
    data = open(path, "rb").read()
    with open(path, "wb") as fh:
        fh.write(data[:len(data) // 2])
    with pytest.raises(EOFError):               # what loading it used to raise
        gzip.open(path, "rt").read()

    cold = vtb.OCRCache(path, "surya", ["en"])
    assert cold.entries == {} and cold.get(str(page)) is None
    assert vtb.OCRCache.filed_by_bytes(path) is False
    cold.put(str(page), [vtb.PageItem(html="<p>read again</p>")])
    cold.save()
    assert vtb.OCRCache(path, "surya", ["en"]).get(str(page)) is not None
