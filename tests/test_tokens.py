"""A word is letters in any script, and old addresses are still read the old way.

The word pattern was written out by hand as far as Latin-1, so a word was
cut at its first macron, háček or Greek letter and the sheet disputed the
fragment: "Kenky" for Kenkyū, "techn" for technē. 1,406 of the shelf's
38,528 pages tokenise differently under a Unicode pattern.

Dispute records and decisions files are addressed in tokens, so each says
which generation of the pattern made it, and one made under the old
pattern is read with the old pattern.
"""
import json
import re
import zipfile

import pytest
from PIL import Image

from codicology import adjudicate as adj
from codicology import pipeline as vtb
from codicology import review
from test_decision_sites import _epub, _fix, _read, _tsv


# ── the pattern ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text, words", [
    ("the journal Kenkyū printed it", ["the", "journal", "Kenkyū", "printed", "it"]),
    ("technē and diabolē", ["technē", "diabolē"]),   # a mark of its own
    ("the πόλις of Ἀθῆναι", ["the", "πόλις", "of", "Ἀθῆναι"]),
    ("a dose of 5 μgs daily", ["a", "dose", "of", "μgs", "daily"]),
    ("β-phenethylamine, 2C-B", ["β-phenethylamine", "C-B"]),
    ("Čapek, Łódź, İstanbul, Erdoğan", ["Čapek", "Łódź", "İstanbul", "Erdoğan"]),
    ("война и мир", ["война", "и", "мир"]),
])
def test_a_word_is_letters_of_any_script(text, words):
    got = adj.tokens(text)
    for w in words:
        assert w in got, (w, got)
    if text.startswith("a dose"):
        assert got == words, "digits are still not words"


LATIN1 = ("L'État, c'est moi — naïve café-society; Œuvres complètes de "
          "Molière. It's well-nigh over, don’t you think? Straße 7, año 1917.")


def test_latin1_text_tokenises_as_it_always_did():
    now = adj.tokens(LATIN1)
    with adj.tokens_as(None):
        before = adj.tokens(LATIN1)
    assert now == before and "café-society" in now and "don't" in now


def test_the_old_pattern_is_put_back_whatever_happens():
    with pytest.raises(RuntimeError):
        with adj.tokens_as(None):
            assert adj.tokens("Kenkyū") == ["Kenky"]
            with adj.tokens_as(adj.TOKENS):
                assert adj.tokens("Kenkyū") == ["Kenkyū"]
            assert adj.tokens("Kenkyū") == ["Kenky"]
            raise RuntimeError("mid-apply")
    assert adj.tokens("Kenkyū") == ["Kenkyū"]
    # the other module's handle on it is the same object
    assert review.WORD is adj.WORD


# ── decisions, old and new ───────────────────────────────────────────────────

BODY = "<p>the Greek diabolē, which is slender, as Kenkyū has it</p>"


def _file(tmp_path, rows, **meta):
    path = tmp_path / "d.decisions.json"
    path.write_text(json.dumps({"page_files": 1, "decisions": rows, **meta}))
    return str(path)


def _through_both(tmp_path, body, dec):
    bodies = [body]
    st = vtb.apply_reviewer_decisions(bodies, dec)
    epub = _epub(tmp_path, {0: body})
    r = review.apply_decisions(epub, dec)
    page = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert bodies[0] in page, "rebuild and apply must agree"
    assert st == {"applied": len(r["applied"]), "stale": len(r["stale"])}
    return bodies[0], st


def test_a_whole_word_can_now_be_corrected(tmp_path):
    """Under the old pattern this row could only be refused: it named a
    piece of the word."""
    assert _read(BODY)[2] == "diabolē"
    dec = _file(tmp_path, [_fix(BODY, 2, "diabolé")], tokens=adj.TOKENS)
    got, st = _through_both(tmp_path, BODY, dec)
    assert got == BODY.replace("diabolē", "diabolé")
    assert st == {"applied": 1, "stale": 0}


def test_a_decision_made_under_the_old_pattern_lands_as_it_did(tmp_path):
    """Its neighbours were recorded as the old pattern cut them: "Kenky",
    not "Kenkyū". Read with today's pattern those are not this page's
    words; read with its own, the row is placed exactly as before."""
    with adj.tokens_as(None):
        toks = _read(BODY)
        k = toks.index("slender")
        row = _fix(BODY, k, "slander")
    assert "Kenky" in row["after"] and "diabol" in row["before"]
    got, st = _through_both(tmp_path, BODY, _file(tmp_path, [row]))
    assert got == BODY.replace("slender", "slander")
    assert st == {"applied": 1, "stale": 0}


def test_a_fragment_from_an_old_sheet_is_still_refused(tmp_path):
    with adj.tokens_as(None):
        row = _fix(BODY, 2, "diabolé")
    assert row["old"] == "diabol"
    got, st = _through_both(tmp_path, BODY, _file(tmp_path, [row]))
    assert (got, st) == (BODY, {"applied": 0, "stale": 1})


# ── the generation travels with the record ───────────────────────────────────

def test_a_record_says_its_generation_and_the_sheet_passes_it_on(tmp_path,
                                                                 monkeypatch):
    body = "<p>Aristotle's technē is not epistēmē, he wrote plainly</p>"
    epub = _epub(tmp_path, {0: body})
    pdf = str(tmp_path / "book.pdf")
    Image.new("RGB", (600, 200), "white").save(pdf, "PDF")
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)
    # a witness without the macrons, and with one real difference
    monkeypatch.setattr(adj, "read_tesseract_tsv", lambda png: _tsv(
        "Aristotle's techne is not episteme, he wrote plainty".split()))
    path = str(tmp_path / "book-disputes.json")
    assert adj.main(epub, pdf, report=path) == 0
    record = json.load(open(path))
    assert record["tokens"] == adj.TOKENS
    # technē against techne is a fold, not a dispute; under the old pattern
    # it was "techn" against "techne", and on the sheet
    assert [(r["surya"], r["tesseract"]) for r in record["disputes"]] == \
           [("plainly", "plainty")]

    sheet = review.render_sheet(record)
    meta = json.loads(re.search(
        r"<script type='application/json' id='meta'>(.*?)</script>",
        sheet, re.S).group(1))
    assert meta["tokens"] == adj.TOKENS

    old = dict(record)
    del old["tokens"]
    meta = json.loads(re.search(
        r"<script type='application/json' id='meta'>(.*?)</script>",
        review.render_sheet(old), re.S).group(1))
    assert "tokens" not in meta, "an old record's sheet exports old addresses"


def test_filing_a_round_keeps_its_generation(tmp_path):
    src = _file(tmp_path, [_fix(BODY, 2, "diabolé")], tokens=adj.TOKENS,
                sheet="abc")
    target = str(tmp_path / "cache" / "book-disputes.decisions.json")
    review.file_decisions(src, target)
    assert json.load(open(target))["tokens"] == adj.TOKENS


# ── a page in another script is not an empty page ────────────────────────────

@pytest.mark.parametrize("text, empty", [
    ("Ἐν ἀρχῇ ἦν ὁ λόγος", False),
    ("Все счастливые семьи похожи друг на друга", False),
    ("論語", False),
    ("A page of English.", False),
    ("", True),
    ("  \n ", True),
    ("17", True),                               # a folio leaked into the body
    ("[Faded text block, likely bleed-through]", True),
    ("[Faded text] — · …", True),
])
def test_reads_as_empty_knows_letters_of_any_script(text, empty):
    assert vtb._reads_as_empty(text) is empty


# ── what the witness cannot testify to ───────────────────────────────────────

@pytest.mark.parametrize("ours, theirs", [
    ("chōra", "chéra"),                         # the nearest letter it has
    ("THβC", "THBC"),
    ("μέρος", "pépos"),                         # a Greek word, as Latin shapes
    ("tetrahydro-β-carboline", "tetrahydroB-carboline"),
    ("α-DMT", "DMT"),                           # or the letter simply dropped
    ("Biežuńska-Małowist", "Biezuriska-Mafowist"),
    ("Rhōmaioi", "Rhémaioi'"),
])
def test_a_difference_only_at_letters_the_witness_cannot_read_is_no_dispute(
        ours, theirs):
    """With whole words these all reach the ladder, and its rungs ruled on
    them: Vision crowned "THBC" over "THβC" 223 times in one book."""
    from collections import Counter
    for lexicon in (Counter(), Counter({adj.fold_word(theirs): 50})):
        got = adj.adjudicate_pair(ours, theirs, lexicon)
        assert got == {"rung": "fold", "winner": ours,
                       "note": "beyond the witness"}


@pytest.mark.parametrize("ours, theirs", [
    ("lēistai", "éistai"),          # it lost a letter it can read, too
    ("THβC", "THCB"),               # and here it disagrees about the C
    ("plainly", "plainty"),         # nothing beyond it at all
    ("DMT", "α-DMT"),               # the witness has more than we do
    ("oikétai", "oixérau"),
])
def test_a_difference_at_a_letter_it_can_read_is_still_a_dispute(ours, theirs):
    from collections import Counter
    got = adj.adjudicate_pair(ours, theirs, Counter())
    assert got["rung"] != "fold", got


def test_a_dropped_diacritic_was_already_no_dispute():
    from collections import Counter
    assert adj.adjudicate_pair("Kenkyū", "Kenkyu", Counter()) == {
        "rung": "fold", "winner": "Kenkyū"}


def test_a_greek_quotation_is_not_a_run_of_words_nobody_else_saw():
    """Words the witness could not have read say nothing by their absence
    from its page."""
    ours = adj.tokens("as Aristotle says, ἡ πόλις τῶν ἐλευθέρων ἐστίν, "
                      "and he meant it; then gibberish wholly invented here")
    theirs = adj.tokens("as Aristotle says, and he meant it; then")
    frac, only, runs = adj.surya_only_runs(ours, theirs)
    assert [" ".join(words) for words, _n in runs] == [
        "gibberish wholly invented here"]
    assert only == 4
