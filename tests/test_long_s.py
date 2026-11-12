"""The long s, which a witness that has never met it reads as an f.

Lloyd's register of 1764 prints "Swainſon", "Guernſy", "Amſt.", "Briſt.".
The first reader writes s. Apple Vision wrote f, and the ladder took its
word five times out of five, all wrong. One such pair in a book may be a
real misreading; a book with three prints a long s.
"""
import json

import pytest

from codicology import adjudicate as adj
from test_adjudicate_run import _page_of, _pdf
from test_decision_sites import _epub, _tsv


@pytest.mark.parametrize("ours, theirs, yes", [
    ("Swainson", "Swainfon", True),
    ("Brist", "Brift", True),
    ("Mississippi", "Miffiffippi", True),
    ("Mississippi", "Mifsissippi", True),       # some of them
    ("sat", "fat", True),                       # which is why one alone is not enough
    ("fat", "sat", False),                      # the other way is a misreading
    ("Swainson", "Swainfan", False),            # something else differs too
    ("Brist", "Brifts", False),
    ("same", "same", False),
])
def test_what_a_long_s_looks_like_to_a_witness(ours, theirs, yes):
    assert adj.long_s(ours, theirs) is yes


def _run(tmp_path, monkeypatch, pages, misread):
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)
    import re
    words = {n: re.sub(r"<[^>]+>", "", p).split() for n, p in pages.items()}

    def read(png):
        return _tsv([misread.get(w, w) for w in words[_page_of(png)]])
    monkeypatch.setattr(adj, "read_tesseract_tsv", read)
    report = str(tmp_path / "book-disputes.json")
    assert adj.main(_epub(tmp_path, pages), _pdf(tmp_path, len(pages)), report=report) == 0
    return json.load(open(report))["disputes"]


REGISTER = {0: "<p>the ship of Swainson cleared for Guernsy with wine and salt aboard</p>",
            1: "<p>bound from Bristol to Amsterdam the master Velasco made his entry</p>"}


def test_in_a_book_that_prints_a_long_s_the_witnesss_f_is_no_dispute(tmp_path, monkeypatch, capsys):
    rows = _run(tmp_path, monkeypatch, REGISTER,
                {"Swainson": "Swainfon", "Guernsy": "Guernfy", "Bristol": "Briftol",
                 "master": "mafter", "wine": "vine"})
    assert [(r["surya"], r["tesseract"]) for r in rows] == [("wine", "vine")]
    assert "long s: 4 word(s)" in capsys.readouterr().out


def test_one_such_word_in_a_book_is_still_a_dispute(tmp_path, monkeypatch):
    rows = _run(tmp_path, monkeypatch, REGISTER, {"salt": "falt", "wine": "vine"})
    assert sorted((r["surya"], r["tesseract"]) for r in rows) == [("salt", "falt"), ("wine", "vine")]
