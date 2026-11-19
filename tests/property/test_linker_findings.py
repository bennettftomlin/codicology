"""Faults the property search found (19 November 2026), each shrunk to a
page or two and pinned here as an expected failure. When one is mended its
test starts passing, strict xfail turns that into a failure, and the marker
comes off — and the matching knob in the property generators
(test_endnote_linker_props.LOSE_CHAPTER_ENTRIES, RUN_ON_HEAD_PAGE,
GUESS_CHAPTERS, test_footnote_linker_props' symbol sets and tolerance) can
be widened so the search covers that shape again.
"""
from __future__ import annotations

import re

import pytest

from codicology.notes import link_chapter_notes, link_footnotes, link_notes
from codicology.typography import normalize_typography

from .notes_oracle import check_links, parses

PROSE = "<p>" + "The river ran north past the old mills. " * 5 + "</p>"


# ── same-page footnotes ─────────────────────────────────────────────────────

# fixed 2026-11-19: the * pass reads the second star of a ** call
def test_an_asterisk_note_is_not_bound_to_half_a_double_asterisk():
    """A page whose call is "**" and whose only foot is "*": the * pass
    finds one * above the rule (the second star of the **) and one below,
    and links them, splitting the printed ** into * and a linked *."""
    bodies = ["<p>Some prose here word** and more.</p><hr/><p>* first note.</p>"]
    before = list(bodies)
    link_footnotes(bodies)
    assert bodies == before


# fixed 2026-11-19: the * pass nests its link inside the ** link
def test_asterisk_and_double_asterisk_feet_do_not_nest_links():
    """With both feet present the ** call is linked first, and the * pass
    then links the second star inside it: an anchor inside an anchor."""
    bodies = ["<p>Some prose here word** and more.</p><hr/>"
              "<p>* first note.</p><p>** second note.</p>"]
    link_footnotes(bodies)
    assert not re.search(r"<a\b[^>]*>[^<]*<sup><a\b", bodies[0])


# fixed 2026-11-19: _scattered_feet takes a table's call as a prose call
def test_a_table_s_call_is_not_taken_by_the_note_of_another_table():
    """Page 0 ends with a table whose note is carried to the head of page 1;
    page 1 ends with another table whose call 1 is answered overleaf on page
    2. The scattered-feet pass pairs page 1's carried note with the second
    table's call, because each is the only one of its number on the page."""
    bodies = [PROSE + "<p>Table 1: Grain</p><table><tr><td>a</td><td>941</td></tr></table>",
              "<p><sup>1</sup> Note of table one.</p><p>Prose of the page.</p>"
              "<p>Table 2: Sold</p><table><tr><td>b</td><td>8,300<sup>1</sup></td></tr></table>",
              "<p><sup>1</sup> Note of table two.</p><p>More prose.</p>"]
    link_footnotes(bodies, allow_numbered=True)
    assert "8,300<sup>1</sup>" in bodies[1]


# fixed 2026-11-19: any title and table opening the next page is taken for a continuation
def test_a_new_table_overleaf_is_not_the_last_one_continued():
    """A page ends with a table whose cell carries a raised 1 and no notes
    under it; the next page opens with a different table, with its own
    notes 1 and 2. The overleaf pass reads the second table as the first
    one's continuation ("a title line and a table") and binds the first
    table's call to the second table's note. Table passes run whatever
    allow_numbered says, so in a book with endnotes the raised 1 may well
    be an endnote call."""
    bodies = [PROSE + "<p>Table 1: Grain sold</p>"
              "<table><tr><td>Kaiser</td><td>8,300<sup>1</sup></td></tr></table>",
              "<p>Table 2: Land held</p><table><tr><td>Getty</td><td>40<sup>1</sup></td></tr>"
              "<tr><td>Mellon</td><td>52<sup>2</sup></td></tr></table>"
              "<p><sup>1</sup> Note one of table two.</p><p><sup>2</sup> Note two of table two.</p>"
              "<p>Prose goes on.</p>"]
    link_footnotes(bodies)
    assert "8,300<sup>1</sup>" in bodies[0]


_OVERLEAF = [
    PROSE + "<p>Table 1: Grain sold</p><table><tr><td>row 0</td><td>a<sup>2</sup></td></tr>"
    "<tr><td>row 1</td><td>b<sup>1</sup></td></tr></table>",
    "<p><sup>1</sup> First note of the table.</p><p><sup>2</sup> Second note of the table.</p>"
    "<p>Prose of the next page raising its own<sup>2</sup> number.</p>"]


@pytest.mark.xfail(strict=True, reason="_scattered_feet links what the overleaf pass refused")
def test_a_carried_table_note_the_overleaf_pass_refused_stays_plain():
    """The overleaf pass leaves the table's carried notes plain: 2 is
    raised in page 1's prose too, so one of two notes is too few to settle
    the scope. The scattered-feet pass then binds that prose call to the
    table's note 2 all the same."""
    bodies = list(_OVERLEAF)
    link_footnotes(bodies, allow_numbered=True)
    assert "own<sup>2</sup>" in bodies[1]


@pytest.mark.xfail(strict=True, reason="a second run links note 1 alone overleaf")
def test_a_second_run_links_nothing_new():
    """Once note 2 is linked (above), the overleaf pass on a second run sees
    one carried note, called once, and links it: 'at least two must be (or
    the only one)' is read against what the first run left."""
    bodies = list(_OVERLEAF)
    link_footnotes(bodies, allow_numbered=True)
    assert link_footnotes(bodies, allow_numbered=True)["linked"] == 0


# ── chapter-end notes ───────────────────────────────────────────────────────

# fixed 2026-11-19: markers after a head on its own page go to that section
def test_a_chapter_begun_on_its_predecessor_s_notes_page_keeps_its_calls():
    """'Markers between one section and the next belong to the next.' A
    chapter that begins on the page where the last one's notes end has its
    calls bound to the last chapter's notes, and its own left plain."""
    bodies = ["<p>One begins.<sup>1</sup> More.<sup>2</sup></p>",
              "<p>End of one.</p><h2>Notes</h2><p>1. Alpha source.</p><p>2. Beta source.</p>"
              "<h1>Chapter 2</h1><p>Two begins.<sup>1</sup> And<sup>2</sup></p>",
              "<p>More of two.</p><h2>Notes</h2><p>1. Gamma source.</p><p>2. Delta.</p>"]
    link_chapter_notes(bodies, set())
    assert "note-c0-1" not in bodies[1].split("<h1>Chapter 2</h1>")[1]


_INTO_THE_NEXT = [
    "<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
    "<h2>Notes</h2><p>1. Alpha.</p><p>2. Beta.</p>",         # note 3 lost
    "<p>A short chapter.<sup>1</sup> On.<sup>2</sup> And on.<sup>3</sup></p>"
    "<h2>Notes</h2><p>1. Gamma.</p><p>2. Delta.</p><p>3. Epsilon.</p>"]


# fixed 2026-11-19: a section's walk reads past the next head on its page
def test_a_section_s_entries_stop_at_the_next_section_s_head():
    """Chapter one's walk goes on to the next page while it finds entries,
    and stops only past the page of the next head — so it reads that page
    whole, past the head, and takes chapter two's note 3 for its own."""
    bodies = list(_INTO_THE_NEXT)
    link_chapter_notes(bodies, set())
    assert "Three.<sup>3</sup>" in bodies[0]


# fixed 2026-11-19: two sections anchor the same entry; the edits collide
def test_an_entry_claimed_twice_does_not_garble_the_page():
    """Both sections then anchor 'Epsilon': two edits over the same span,
    applied one over the other. The page reads '3.="note-c1-3">3. Epsilon'
    and chapter two's call 3 points at an id that is no longer there."""
    bodies = list(_INTO_THE_NEXT)
    link_chapter_notes(bodies, set())
    assert '="note-' not in re.sub(r"<[^>]+>", "", bodies[2])
    check_links(bodies)


# fixed 2026-11-19: a lost entry 1 lets a list in the prose be read as notes
def test_a_numbered_list_after_the_notes_is_not_taken_for_them():
    """'A numbered list in the following chapter's prose ("1. First, …")
    cannot be swallowed as more notes.' It is, where the section's own
    entry 1 was lost: its 2 is refused (a 1 turns up later on the page),
    and the list's 1 and 2 are then exactly what the walk expects."""
    bodies = ["<p>One.<sup>1</sup> Two.<sup>2</sup></p>",
              "<h2>Notes</h2><p>1. Alpha.</p><p>2. Beta.</p>",
              "<p>Chapter two.<sup>1</sup> On.<sup>2</sup></p>",
              "<h2>Notes</h2>",
              "<p>2. Delta.</p><h1>Chapter 3</h1><p>Three begins.</p>"
              "<p>1. First, the mills.</p><p>2. Second, the grain.</p>",
              "<p>Three goes on.<sup>1</sup></p><h2>Notes</h2><p>1. Zeta.</p>"]
    link_chapter_notes(bodies, set())
    assert "<p>1. First, the mills.</p>" in bodies[4]


# ── one back-of-book section, chapters placed by their numbers alone ───────

def _book(calls_by_chapter, notes_by_chapter, heads):
    pages = ["<p>Contents.</p>"]
    for c, calls in enumerate(calls_by_chapter):
        pages.append(f"<h1>Chapter {c + 1}</h1><p>Text "
                     + " ".join(f"c{c}w<sup>{n}</sup>" for n in calls) + ".</p>")
    sec = "<h1>Notes</h1>"
    for c, ns in enumerate(notes_by_chapter):
        sec += heads[c] + "".join(f"<p>{n}. Source ch{c + 1} n{n}.</p>" for n in ns)
    return pages + [sec]


def _wrong_chapter_links(pages):
    """Links from chapter c's prose to another chapter's source."""
    wrong = 0
    for c, page in enumerate(pages[1:-1]):
        for pg, ident in re.findall(r'href="page_(\d+)\.xhtml#(note-g\d+-\d+)"', page):
            target = pages[int(pg)]
            at = target.index(f'id="{ident}"')
            if f"Source ch{c + 1} " not in target[at:target.index("</p>", at)]:
                wrong += 1
    return wrong


@pytest.mark.xfail(strict=True, reason="a lost first group head shifts every chapter by one")
def test_a_lost_first_group_head_does_not_shift_the_chapters():
    """The notes section's first group lost its head, so its entries are
    read past and the section has three groups to the body's four. The fit
    keeps the unshifted pairing 'if it passes', and with chapters of like
    length the tops agree: every call of chapters 1-3 is bound to the next
    chapter's sources."""
    ks = [10, 11, 10, 9]
    pages = _book([range(1, k + 1) for k in ks], [range(1, k + 1) for k in ks],
                  [""] + [f"<h2>Chapter {c + 1}</h2>" for c in range(1, 4)])
    link_notes(pages, set())
    assert _wrong_chapter_links(pages) == 0


@pytest.mark.xfail(strict=True, reason="a chapter citing note 1 twice is split in two")
def test_a_note_cited_twice_at_once_stays_in_its_chapter():
    """Chapter one cites note 1 twice; chapter two cites 1 and 2. No reading
    the body splitter offers has two groups ([1] [1] [1, 2] or one group),
    and the three-group reading fits the two note groups at offset 0: the
    second citation of chapter one's note 1 is bound to chapter two's."""
    pages = _book([[1, 1], [1, 2]], [[1], [1, 2]],
                  ["<h2>Chapter 1</h2>", "<h2>Chapter 2</h2>"])
    link_notes(pages, set())
    assert _wrong_chapter_links(pages) == 0


# ── typography ──────────────────────────────────────────────────────────────

def _typo(page):
    out = [page]
    normalize_typography(out)
    return out[0]


# fixed 2026-11-19: a quote opening a text run is read as opening
@pytest.mark.parametrize("page, want", [
    ('<p>"<i>Ulysses</i>" is long.</p>', "<p>“<i>Ulysses</i>” is long.</p>"),
    ("<p>the <i>Guardian</i>'s view</p>", "<p>the <i>Guardian</i>’s view</p>"),
    ('<p>as he said.<sup>3</sup>" Then</p>', "<p>as he said.<sup>3</sup>” Then</p>"),
])
def test_a_quote_after_inline_markup_follows_the_letter_before_it(page, want):
    """'A quote after space or an opening bracket opens, any other closes.'
    Each text run is requoted alone, so a quote just after </i> or </sup>
    has no character before it and opens."""
    assert _typo(page) == want


# fixed 2026-11-19: the opener test reads the straight mark, not the turned one
@pytest.mark.parametrize("page, want", [
    ("<p>\"'Free trade,' he wrote.\"</p>", "<p>“‘Free trade,’ he wrote.”</p>"),
    ("<p>'\"No,\" she said.'</p>", "<p>‘“No,” she said.’</p>"),
])
def test_a_quote_inside_a_quote_opens(page, want):
    """_requote lists ‘ and “ among the marks a quote opens after, but
    compares against the text as it came in, where they are still straight."""
    assert _typo(page) == want


# fixed 2026-11-19: a tag is split at the first >, even inside a value
def test_quotes_inside_an_attribute_holding_a_greater_than_are_untouched():
    """'Text inside tags is never touched.' A tag is found as <[^>]+>, so an
    attribute value holding ">" ends it early and the rest of the tag is
    requoted as text: title="a>b” href=”x”, which no longer parses."""
    page = '<p><a title="a>b" href="x">it\'s</a></p>'
    out = _typo(page)
    assert parses(out)
