"""The four properties, and proof that they catch what counting cannot.

The shelf gate counts links. The worst regression this pipeline has had was
190 links that all resolved, all counted, and every one of which pointed at
the wrong chapter's note — a count cannot see it, and did not.

So the last test here does not argue that these checks would have caught it.
It performs the injury: takes a healthy book, rebinds a later chapter's
markers to the first chapter's notes number for number, confirms the link
count is untouched, and fails if the checks stay quiet.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import pytest

import linkcheck


def book(chapters):
    """A book as pages: prose pages of markers, then the notes section.

    chapters is [(n_notes, first_body_page)] — one group per chapter, each
    numbering its notes from 1, exactly as a real book does.
    """
    pages, notes = {}, []
    for g, (count, first) in enumerate(chapters):
        for k in range(count):
            pg = first + k // 3
            pages.setdefault(pg, "")
            pages[pg] += (f'<p>text<sup><a epub:type="noteref" id="ref-g{g}-{k+1}" '
                          f'href="page_{900 + g:04d}.xhtml#note-g{g}-{k+1}">'
                          f'{k+1}</a></sup></p>')
        notes.append((g, count))
    for g, count in notes:
        pages[900 + g] = "<h1>NOTES</h1>" + "".join(
            f'<p id="note-g{g}-{k+1}"><sup>{k+1}</sup>Source {k+1}.</p>'
            for k in range(count))
    return pages


def test_a_healthy_book_violates_nothing():
    r = linkcheck.properties(book([(6, 10), (6, 20), (6, 30)]))
    assert r["links"] == 18
    assert not r["digits"] and not r["span"] and not r["order"] and not r["unreached"]


def test_a_marker_reaching_an_entry_with_another_number_is_caught():
    pages = book([(4, 10), (4, 20)])
    pages[10] = pages[10].replace("#note-g0-1", "#note-g0-3")
    r = linkcheck.properties(pages)
    assert [(t, shown, found) for t, shown, found in r["digits"]] == \
        [("note-g0-3", "1", "3")]


def test_a_note_nothing_points_to_is_caught():
    pages = book([(4, 10)])
    pages[10] = pages[10].replace(
        '<a epub:type="noteref" id="ref-g0-1" '
        'href="page_0900.xhtml#note-g0-1">1</a>', "1")
    assert linkcheck.properties(pages)["unreached"] == ["note-g0-1"]


def test_chapters_whose_notes_run_backwards_are_caught():
    pages = book([(3, 10), (3, 20)])
    pages[900], pages[901] = pages[901], pages[900]      # notes sections swapped
    assert linkcheck.properties(pages)["order"]


# ── the injury ───────────────────────────────────────────────────────────────

def test_the_regression_a_count_cannot_see():
    pages = book([(6, 10), (6, 20), (6, 30)])
    clean = linkcheck.properties(pages)
    assert not clean["span"]

    # Chapter THREE's markers rebound to chapter one's notes, number for
    # number, with chapter two left correct -- the shape the real regression
    # took, where some chapters bound rightly and some did not. Each link
    # still resolves; each shows the number its entry shows.
    hurt = {i: t.replace('href="page_0902.xhtml#note-g2-',
                         'href="page_0900.xhtml#note-g0-')
            for i, t in pages.items()}
    bad = linkcheck.properties(hurt)
    assert bad["links"] == clean["links"]          # the count is untouched
    assert not bad["digits"]                       # and every number agrees
    assert bad["span"], "mis-grouping must show as one chapter's markers " \
                        "spanning another's"
    assert bad["unreached"], "and as notes no marker reaches"
