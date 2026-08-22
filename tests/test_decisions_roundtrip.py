"""The decisions loop, round-tripped: what a reviewer stages must land.

The review found the loop's seams (B1, I2, I1, I3, T4 in the 2026-08-18
ledger); this net encodes each as a strict xfail so the suite stays green
today and phase 2 of the fix plan is FORCED to flip the markers — a
strict xfail that starts passing fails the suite until its marker is
removed. Cases that already hold are pinned green alongside, so the
working half of the contract cannot regress while the broken half is
being fixed.
"""
import json

import pytest

from codicology import review
from codicology import pipeline as pl


def _decisions(tmp_path, rows, **meta):
    p = tmp_path / "d.decisions.json"
    p.write_text(json.dumps({**meta, "decisions": rows}))
    return str(p)


def _epub(tmp_path, pages):
    import zipfile
    p = tmp_path / "book.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        for n, body in pages.items():
            z.writestr(f"EPUB/page_{n:04d}.xhtml",
                       f"<html><body>{body}</body></html>")
    return str(p)


def _row(page, occ, old, new):
    return {"page": page, "occurrence": occ, "old": old, "new": new,
            "source": "human", "rung": "human"}


# ── the working half, pinned ─────────────────────────────────────────────

def test_plain_word_lands_through_both_consumers(vtb, tmp_path):
    dec = _decisions(tmp_path, [_row(0, 0, "beligerents", "belligerents")])
    bodies = ["<p>the beligerents met</p>"]
    assert pl.apply_reviewer_decisions(bodies, dec) == {"applied": 1,
                                                        "stale": 0}
    assert bodies[0] == "<p>the belligerents met</p>"

    epub = _epub(tmp_path, {0: "<p>the beligerents met</p>"})
    r = review.apply_decisions(epub, dec)
    assert len(r["applied"]) == 1 and not r["stale"]


def test_apostrophe_decision_survives_the_rebuild_path(vtb, tmp_path):
    """adjudicate records straight apostrophes; a --typography book carries
    curly ones. The rebuild consumer's variant retry bridges that today."""
    dec = _decisions(tmp_path, [_row(0, 0, "don't", "won't")])
    bodies = ["<p>they don’t know</p>"]
    st = pl.apply_reviewer_decisions(bodies, dec)
    assert st["applied"] == 1
    assert "won't" in bodies[0]


# ── the formerly broken half, fixed in phase 2 and held green ───────────

def test_apostrophe_decision_survives_codicology_apply(vtb, tmp_path):
    dec = _decisions(tmp_path, [_row(0, 0, "don't", "won't")])
    epub = _epub(tmp_path, {0: "<p>they don’t know</p>"})
    r = review.apply_decisions(epub, dec)
    assert len(r["applied"]) == 1, "the two consumers must be equivalent"


def test_pageturn_joined_word_survives_the_rebuild(vtb, tmp_path):
    """The reviewer read 'Russia' whole (the joiner shipped it that way);
    the rebuild must reproduce the join before decisions replay."""
    dec = _decisions(tmp_path, [_row(0, 0, "Russia", "RUSSIA")])
    bodies = ["<p>he fled to Rus-</p>", "<p>sia at last</p>"]
    # build_epub's order since the B1 fix: join first, decisions second
    pl.join_page_break_hyphens(bodies, set())
    st = pl.apply_reviewer_decisions(bodies, dec)
    assert st["stale"] == 0, "a decision on a joined word must not go stale"
    assert "RUSSIA" in bodies[0]


def test_case_variant_pair_lands_on_its_own_words(vtb, tmp_path):
    """Two disputes fold to one token; the sheet numbers them on one
    ordinal stream; apply must land each on its own literal word."""
    rows = [{"page": 0, "surya": "The", "tesseract": "Thc",
             "rung": "abstain", "winner": None, "shipped": "The"},
            {"page": 0, "surya": "the", "tesseract": "thc",
             "rung": "abstain", "winner": None, "shipped": "the"}]
    occs = review.assign_occurrences(rows)
    dec = _decisions(tmp_path, [_row(0, occs[0], "The", "A"),
                                _row(0, occs[1], "the", "a")])
    epub = _epub(tmp_path, {0: "<p>The cat. the dog. the end</p>"})
    review.apply_decisions(epub, dec)
    import re, zipfile
    got = re.sub(r"<[^>]+>", "", zipfile.ZipFile(epub).read(
        "EPUB/page_0000.xhtml").decode())
    assert "A cat" in got and "a dog" in got and "the end" in got, \
        f"second decision must hit the FIRST lowercase 'the', got: {got!r}"


def test_string_page_numbers_are_coerced_not_silently_staled(vtb, tmp_path):
    dec = _decisions(tmp_path, [dict(_row(0, 0, "word", "world"),
                                     page="0")])
    bodies = ["<p>a word here</p>"]
    st = pl.apply_reviewer_decisions(bodies, dec)
    assert st["applied"] == 1, "a numeric-string page is a page"


def test_renumbered_build_refuses_the_whole_decisions_file(vtb, tmp_path):
    """Recorded against a 5-page build, replayed onto 4 pages: every
    decision must be refused loudly, not applied coincidentally."""
    dec = _decisions(tmp_path, [_row(1, 0, "word", "world")],
                     page_files=5)
    bodies = ["<p>x</p>", "<p>a word here</p>", "<p>y</p>", "<p>z</p>"]
    st = pl.apply_reviewer_decisions(bodies, dec)
    assert st["applied"] == 0, \
        "page-count mismatch means the indices cannot be trusted"
