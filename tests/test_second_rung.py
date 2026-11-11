"""A second reader's vote on a word the ladder left open.

The first reader and tesseract are one against one, and the book's own
vocabulary and the dictionary settle most of what they dispute. What is left
went to Apple Vision alone. Where a second reader is installed and asked
for, it and Vision are two more readers of the same ink and settle a word as
a count of readers: one of them with a view, or both agreeing, and never the
two of them on opposite sides.
"""
import gzip
import json

import pytest
from PIL import Image

from codicology import adjudicate as adj, cli, second
from test_adjudicate_run import PAGES, _page_of, _pdf, _words
from test_decision_sites import _epub, _tsv


# ── the count of readers ─────────────────────────────────────────────────────

@pytest.mark.parametrize("vision,theirs,rung,winner", [
    ({"quartz"}, None, "vision", "quartz"),         # as it was before
    (None, {"quarts"}, "second", "quarts"),
    ({"quartz"}, {"quartz"}, "readers", "quartz"),  # three against one
    ({"quartz"}, set(), "vision", "quartz"),        # the other has no view
    (set(), {"quarts"}, "second", "quarts"),
])
def test_a_reader_with_a_view_settles_it(vision, theirs, rung, winner):
    said = adj.readers_verdict("quartz", "quarts", vision, theirs)
    assert (said["rung"], said["winner"]) == (rung, winner)


def test_two_readers_apiece_is_recorded_and_not_guessed():
    said = adj.readers_verdict("quartz", "quarts", {"quartz"}, {"quarts"})
    assert said == {"rung": "abstain", "winner": None, "note": "readers split"}


@pytest.mark.parametrize("vision,theirs", [
    (None, None),
    (set(), set()),
    ({"quartz", "quarts"}, {"quartz", "quarts"}),   # both words are on the page
    ({"quartz", "quarts"}, None),
])
def test_a_reader_with_both_words_or_neither_has_no_view(vision, theirs):
    assert adj.readers_verdict("quartz", "quarts", vision, theirs) is None


def test_the_words_are_compared_as_the_ladder_folds_them():
    said = adj.readers_verdict("Quartz", "quarts", None, {"quartz"})
    assert (said["rung"], said["winner"]) == ("second", "Quartz")


# ── in a run ─────────────────────────────────────────────────────────────────

class _Reader:
    """Stands where the model does: a page's words, however it was asked."""

    def __init__(self, pages):
        self.pages, self.read_calls, self.closed = pages, 0, False

    def read(self, image):
        self.read_calls += 1
        # which page: the test's PDF pages are blank, so the order of asking
        return self.pages[self.read_calls - 1]

    def close(self):
        self.closed = True


@pytest.fixture
def book(tmp_path, monkeypatch):
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)

    def read(png):
        words = _words(_page_of(png))
        if _page_of(png) == 2:
            words[3] = "quarts"         # two words of the language: left open
        return _tsv(words)
    monkeypatch.setattr(adj, "read_tesseract_tsv", read)
    return (_epub(tmp_path, PAGES), _pdf(tmp_path, len(PAGES)),
            str(tmp_path / "book-disputes.json"))


def _row(report):
    (row,) = json.load(open(report))["disputes"]
    return row


def test_without_a_second_reader_the_word_stays_open(book):
    epub, pdf, report = book
    assert adj.main(epub, pdf, report=report) == 0
    row = _row(report)
    assert (row["surya"], row["tesseract"]) == ("quartz", "quarts")
    assert (row["rung"], row["winner"]) == ("abstain", None)


def test_the_second_reader_settles_a_word_the_ladder_left_open(book, capsys):
    epub, pdf, report = book
    reader = _Reader(["sphinx of black **quartz** judge my vow"])
    assert adj.main(epub, pdf, report=report,
                    second=adj.SecondWords(reader)) == 0
    row = _row(report)
    assert (row["rung"], row["winner"]) == ("second", "quartz")
    # only the page with an open dispute was put to it, and it was let go
    assert reader.read_calls == 1 and reader.closed
    assert "1 page(s) with a dispute were put to it" in capsys.readouterr().out


def test_it_can_side_with_the_witness_against_what_shipped(book):
    epub, pdf, report = book
    reader = _Reader(["sphinx of black quarts judge my vow"])
    adj.main(epub, pdf, report=report, second=adj.SecondWords(reader))
    row = _row(report)
    assert (row["rung"], row["winner"], row["shipped"]) == ("second", "quarts",
                                                            "quartz")


def test_vision_and_the_second_reader_on_opposite_sides_leave_it_open(
        book, monkeypatch):
    epub, pdf, report = book
    monkeypatch.setattr(adj, "_vision_present", lambda: True)

    def vision(png):
        """Apple Vision, as the run finds it."""
        return "sphinx of black quartz judge my vow"
    monkeypatch.setattr(adj, "read_vision", vision)
    reader = _Reader(["sphinx of black quarts judge my vow"])
    adj.main(epub, pdf, report=report, second=adj.SecondWords(reader))
    row = _row(report)
    assert (row["rung"], row["winner"]) == ("abstain", None)


def test_a_reader_that_gave_no_reading_has_no_view(book):
    epub, pdf, report = book
    adj.main(epub, pdf, report=report, second=adj.SecondWords(_Reader([None])))
    assert (_row(report)["rung"], _row(report)["winner"]) == ("abstain", None)


def test_a_reader_that_could_not_be_asked_is_asked_again_by_the_next_run(
        book, tmp_path, capsys):
    epub, pdf, report = book
    kept = str(tmp_path / "book-disputes.second.gz")

    class Down:
        def read(self, image):
            raise RuntimeError("the second reader's server did not start")
        def close(self):
            pass
    adj.main(epub, pdf, report=report,
             second=adj.SecondWords(Down(), second.SecondCache(kept)))
    assert (_row(report)["rung"], _row(report)["winner"]) == ("abstain", None)
    assert "could not be asked about 1 of those pages" in capsys.readouterr().out
    # nothing was filed about the page, so the next run reads it
    after = _Reader(["sphinx of black quartz judge my vow"])
    adj.main(epub, pdf, report=report,
             second=adj.SecondWords(after, second.SecondCache(kept)))
    assert after.read_calls == 1
    assert (_row(report)["rung"], _row(report)["winner"]) == ("second", "quartz")


def test_a_page_is_read_once_and_remembered_for_the_next_run(book, tmp_path):
    epub, pdf, report = book
    kept = str(tmp_path / "book-disputes.second.gz")
    first = _Reader(["sphinx of black quartz judge my vow"])
    adj.main(epub, pdf, report=report,
             second=adj.SecondWords(first, second.SecondCache(kept)))
    assert first.read_calls == 1
    with gzip.open(kept, "rt") as fh:
        (key,) = json.load(fh)["pages"]
    assert key.startswith("adjudicate:")

    again = _Reader([])                 # it would fail if it were asked
    adj.main(epub, pdf, report=report,
             second=adj.SecondWords(again, second.SecondCache(kept)))
    assert again.read_calls == 0
    assert (_row(report)["rung"], _row(report)["winner"]) == ("second", "quartz")


# ── a verdict the second reader reads the other way ──────────────────────────

LEXICON = __import__("collections").Counter({"appear": 12, "the": 400})


def _heard(verdict, a, b, vision=None, theirs=None):
    return adj.heard_by_the_readers(verdict, a, b, lambda: vision,
                                    lambda: theirs)


def test_a_lexicon_verdict_the_second_reader_contradicts_is_withdrawn():
    """One of the born-digital pages this was measured on prints
    "appertain". The first reader had "appear", which the book says often;
    tesseract had the page's word, and the lexicon ruled for the common
    one."""
    ruled = adj.adjudicate_pair("appear", "appertain", LEXICON)
    assert (ruled["rung"], ruled["winner"]) == ("lexicon", "appear")
    said = _heard(ruled, "appear", "appertain", theirs={"appertain", "the"})
    assert (said["rung"], said["winner"], said["overruled"]) == (
        "abstain", None, "lexicon")
    assert "second reader" in said["note"]


def test_it_is_withdrawn_and_not_reversed():
    ruled = adj.adjudicate_pair("zzyx", "quartz", LEXICON)
    assert (ruled["rung"], ruled["winner"]) == ("dictionary", "quartz")
    said = _heard(ruled, "zzyx", "quartz", vision={"zzyx"}, theirs={"zzyx"})
    assert said["winner"] is None and said["overruled"] == "dictionary"


@pytest.mark.parametrize("theirs", [
    {"appear"},                 # it reads what the lexicon ruled
    {"appear", "appertain"},    # both are on the page: no view
    set(), None,                # no view, or not asked
])
def test_a_verdict_it_does_not_contradict_stands(theirs):
    ruled = adj.adjudicate_pair("appear", "appertain", LEXICON)
    assert _heard(ruled, "appear", "appertain", theirs=theirs) == ruled


def test_vision_alone_does_not_withdraw_a_verdict():
    """It never had that say, and over the same pages a lexicon verdict it
    contradicted was right one time in three."""
    ruled = adj.adjudicate_pair("appear", "appertain", LEXICON)
    assert _heard(ruled, "appear", "appertain", vision={"appertain"}) == ruled


def test_a_dispute_only_a_reviewer_may_settle_is_not_put_to_the_readers():
    ruled = adj.adjudicate_pair("a-or", "or", LEXICON)
    assert ruled.get("reviewer")
    asked = []
    said = adj.heard_by_the_readers(ruled, "a-or", "or",
                                    lambda: asked.append("v") or {"or"},
                                    lambda: asked.append("s") or {"or"})
    assert said == ruled and asked == []


def test_a_fold_is_not_a_dispute_and_nobody_is_asked():
    ruled = adj.adjudicate_pair("naïve", "naive", LEXICON)
    assert ruled["rung"] == "fold"
    asked = []
    assert adj.heard_by_the_readers(ruled, "naïve", "naive",
                                    lambda: asked.append("v"),
                                    lambda: asked.append("s")) == ruled
    assert asked == []


def test_in_a_run_the_withdrawn_verdict_is_recorded_as_one(tmp_path, monkeypatch,
                                                           capsys):
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)
    pages = {0: "<p>the quick brown fox jumps over the lazy zzyx</p>"}
    monkeypatch.setattr(adj, "read_tesseract_tsv", lambda png: _tsv(
        "the quick brown fox jumps over the lazy quartz".split()))
    epub, pdf = _epub(tmp_path, pages), _pdf(tmp_path, 1)
    report = str(tmp_path / "book-disputes.json")
    reader = _Reader(["the quick brown fox jumps over the lazy zzyx"])
    adj.main(epub, pdf, report=report, second=adj.SecondWords(reader))
    row = _row(report)
    assert (row["rung"], row["winner"], row["overruled"]) == (
        "abstain", None, "dictionary")
    assert "1 verdict(s) of the lexicon or the dictionary withdrawn" \
        in capsys.readouterr().out
    # without it the dictionary's word stands, as it always has
    adj.main(epub, pdf, report=report)
    assert (_row(report)["rung"], _row(report)["winner"]) == ("dictionary",
                                                              "quartz")


# ── asked for where there is none ────────────────────────────────────────────

def test_asking_for_a_second_reader_that_is_not_installed_is_refused(
        book, monkeypatch, capsys):
    epub, pdf, report = book
    monkeypatch.setattr(second, "available", lambda: False)
    assert cli.main(["adjudicate", epub, pdf, "--report", report,
                     "--second-reader"]) == 2
    assert "there is none to ask" in capsys.readouterr().out


def test_its_readings_are_kept_beside_the_dispute_record(tmp_path, monkeypatch):
    monkeypatch.setattr(second, "available", lambda: True)
    monkeypatch.setattr(second, "SecondReader", lambda: _Reader([]))
    got = adj.second_words_for(str(tmp_path / "7-abc-disputes.json"))
    assert got.cache.path == str(tmp_path / "7-abc-disputes.second.gz")
    assert adj.second_words_for(None).cache is None
