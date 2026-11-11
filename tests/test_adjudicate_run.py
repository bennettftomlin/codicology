"""An adjudication that checked nothing must not say it found nothing.

`codicology adjudicate` printed "disputes: 0" and exited 0 for a book no
witness had read: tesseract's output was taken whatever its exit status,
and a missing tesseract was a page with no words. Around that: the pages
are read several at a time, the page images are removed afterwards, and a
word a second look disagreed about now reaches the ladder — it was set on
the page's entry one line before the entry was replaced.
"""
import json
import os
import threading
import time

import pytest
from PIL import Image

from codicology import adjudicate as adj
from test_decision_sites import _epub, _tsv

PAGES = {
    0: "<p>the quick brown fox jumps over the lazy dog</p>",
    1: "<p>pack my box with five dozen liquor jugs</p>",
    2: "<p>sphinx of black quartz judge my vow</p>",
}


def _pdf(tmp_path, n):
    path = str(tmp_path / "book.pdf")
    sheets = [Image.new("RGB", (600, 200), "white") for _ in range(n)]
    sheets[0].save(path, "PDF", save_all=True, append_images=sheets[1:])
    return path


def _words(n):
    import re
    return re.sub(r"<[^>]+>", "", PAGES[n]).split()


def _page_of(png):
    return int(os.path.basename(png)[1:-4])        # "p12.png"


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)
    return (_epub(tmp_path, PAGES), _pdf(tmp_path, len(PAGES)),
            str(tmp_path / "book-disputes.json"))


# ── a witness that is not there ──────────────────────────────────────────────

def test_no_tesseract_is_a_failure_and_writes_no_record(book, monkeypatch,
                                                        capsys):
    epub, pdf, report = book
    monkeypatch.setattr(adj, "_tesseract_present", lambda: False)
    assert adj.main(epub, pdf, report=report) == 2
    assert not os.path.exists(report)
    out = capsys.readouterr().out
    assert "tesseract is not installed" in out and "disputes:" not in out


def test_a_tesseract_that_reads_nothing_is_a_failure(book, monkeypatch, capsys):
    epub, pdf, report = book
    monkeypatch.setattr(adj, "read_tesseract_tsv", lambda png: None)
    assert adj.main(epub, pdf, report=report) == 2
    assert not os.path.exists(report)
    out = capsys.readouterr().out
    assert "could not read 3 page(s)" in out
    assert "nothing was checked" in out and "disputes:" not in out


def test_one_unread_page_is_named_and_the_rest_is_still_checked(
        book, monkeypatch, capsys):
    epub, pdf, report = book

    def read(png):
        n = _page_of(png)
        if n == 1:
            return None
        return _tsv(["sphinx", "of", "black", "quarts", "judge", "my", "vow"]
                    if n == 2 else _words(n))
    monkeypatch.setattr(adj, "read_tesseract_tsv", read)
    assert adj.main(epub, pdf, report=report) == 0
    record = json.load(open(report))
    assert record["unread_pdf_pages"] == [1]
    (row,) = record["disputes"]
    assert (row["page"], row["surya"], row["tesseract"]) == (2, "quartz",
                                                             "quarts")
    assert "could not read 1 page(s) of the PDF (1)" in capsys.readouterr().out


def test_a_clean_run_says_nothing_about_unread_pages(book, monkeypatch):
    epub, pdf, report = book
    monkeypatch.setattr(adj, "read_tesseract_tsv",
                        lambda png: _tsv(_words(_page_of(png))))
    assert adj.main(epub, pdf, report=report) == 0
    record = json.load(open(report))
    assert record["disputes"] == [] and "unread_pdf_pages" not in record


def test_a_tesseract_that_exits_badly_has_not_read_the_page(monkeypatch):
    class Died:
        returncode, stdout = 1, "level\tpage_num\n"
    monkeypatch.setattr(adj.subprocess, "run", lambda *a, **k: Died)
    assert adj.read_tesseract_tsv("p.png") is None
    assert adj.read_tesseract("p.png") is None

    class Read:
        returncode, stdout = 0, "a page"
    monkeypatch.setattr(adj.subprocess, "run", lambda *a, **k: Read)
    assert adj.read_tesseract("p.png") == "a page"


# ── several pages at a time ──────────────────────────────────────────────────

def test_pages_are_read_together_and_each_reading_stays_with_its_page(
        book, monkeypatch):
    epub, pdf, report = book
    monkeypatch.setattr(adj, "TESS_WORKERS", 3)
    at_once, peak, lock = [0], [0], threading.Lock()

    def read(png):
        n = _page_of(png)
        with lock:
            at_once[0] += 1
            peak[0] = max(peak[0], at_once[0])
        time.sleep(0.15 * (3 - n))          # the first page comes back last
        with lock:
            at_once[0] -= 1
        words = _words(n)
        words[2] = words[2] + "x"           # one disagreement on every page
        return _tsv(words)
    monkeypatch.setattr(adj, "read_tesseract_tsv", read)
    assert adj.main(epub, pdf, report=report) == 0
    got = {(r["page"], r["surya"], r["tesseract"])
           for r in json.load(open(report))["disputes"]}
    assert got == {(0, "brown", "brownx"), (1, "box", "boxx"),
                   (2, "black", "blackx")}
    assert peak[0] > 1, "the pages were read one at a time"


def test_a_limited_run_reads_no_further_than_it_looks(book, monkeypatch):
    epub, pdf, report = book
    seen = []
    monkeypatch.setattr(adj, "read_tesseract_tsv",
                        lambda png: seen.append(_page_of(png))
                        or _tsv(_words(_page_of(png))))
    assert adj.main(epub, pdf, report=report, limit=1) == 0
    assert seen == [0]


def test_the_page_images_are_removed(book, monkeypatch):
    epub, pdf, report = book
    made = []
    real = adj.tempfile.mkdtemp
    monkeypatch.setattr(adj.tempfile, "mkdtemp",
                        lambda **kw: made.append(real(**kw)) or made[-1])
    monkeypatch.setattr(adj, "read_tesseract_tsv",
                        lambda png: _tsv(_words(_page_of(png))))
    adj.main(epub, pdf, report=report)
    assert made and not os.path.exists(made[0])

    # and when the run dies halfway
    def dies(png):
        raise RuntimeError("the reader fell over")
    monkeypatch.setattr(adj, "read_tesseract_tsv", dies)
    with pytest.raises(RuntimeError):
        adj.main(epub, pdf, report=report)
    assert len(made) == 2 and not os.path.exists(made[1])


# ── what a second look disagreed about ───────────────────────────────────────

def test_where_each_run_begins():
    page = "a b one two of three c one two of three d".split()
    runs = [("one two of three".split(), 3), ("one two of three".split(), 3)]
    assert adj._run_starts(page, runs) == [2, 7]
    assert adj._run_starts(page, [(["not", "there"], 2)]) == [None]


def test_a_word_the_second_look_disputes_reaches_the_ladder(tmp_path,
                                                            monkeypatch):
    """The page read misses a line that we have. A read of the crop where
    the line sits finds the place — most of its words — and differs in two
    of them, each a real word. Those differences are disputes like any
    other, addressed to the words on the page."""
    body = ("<p>the committee met on tuesday and resolved nothing</p>"
            "<p>whereupon the chairman adjourned proceedings indefinitely</p>")
    epub = _epub(tmp_path, {0: body})
    pdf = _pdf(tmp_path, 1)
    report = str(tmp_path / "book-disputes.json")
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)
    # the page read sees only the first line…
    monkeypatch.setattr(adj, "read_tesseract_tsv", lambda png: _tsv(
        "the committee met on tuesday and resolved nothing".split()))
    # …the layer says where the second is, and a read of that crop has it
    monkeypatch.setattr(adj, "_run_box", lambda page, run: [0.1, 0.5, 0.9, 0.6])
    crop = "whereupon the chairman adjoined proceedings immediately".split()
    boxes = [[0.1 + 0.1 * k, 0.5, 0.2 + 0.1 * k, 0.6] for k in range(len(crop))]
    monkeypatch.setattr(adj, "_crop_read_tsv", lambda png, box: (crop, boxes))
    assert adj._run_verdict(body.split("<p>")[2].split("<")[0].split(),
                            crop) == "located"

    assert adj.main(epub, pdf, report=report) == 0
    rows = {r["surya"]: r for r in json.load(open(report))["disputes"]
            if r.get("via") == "crop"}
    assert set(rows) == {"adjourned", "indefinitely"}
    row = rows["adjourned"]
    assert row["tesseract"] == "adjoined" and row["box"] == boxes[3]
    assert row["occurrence"] == 0
    assert row["before"][-2:] == ["the", "chairman"]
    assert row["after"][:1] == ["proceedings"]
