"""Prose markers bind to their endnotes only where the binding is certain.

The asymmetry that shapes every rule here: an unlinked superscript is the page
as printed; a wrong link sends the reader to someone else's citation and looks
authoritative doing it. So markers link only when the chapter scope is settled
— by the markers' own numbering resets, aligned to the endnote section's own
chapter groups — and exactly one note answers the number in that scope.
"""
import pytest


def book(vtb):
    """A tiny two-chapter book with an endnotes section, as surya renders one.

    Each chapter carries at least three markers: the reset detector refuses to
    call "1 after 2" a new chapter — an OCR misread of a double-digit marker
    looks exactly like that — and only trusts a restart after a run has
    reached three. Real chapters carry ten to thirty notes.
    """
    return [
        "<p>Chapter one prose.<sup>1</sup> More.<sup>2</sup> Again.<sup>3</sup></p>",
        "<p>Second chapter begins.<sup>1</sup> And continues.<sup>2</sup> "
        "Then more.<sup>3</sup></p>",
        "<h1>NOTES</h1><h2>CHAPTER ONE</h2>"
        "<p><sup>1</sup>First source.</p><p><sup>2</sup>Second source.</p>"
        "<p><sup>3</sup>Third source.</p>",
        "<p>CHAPTER TWO</p>"
        "<p><sup>1</sup>Fourth source.</p><p><sup>2</sup>Fifth source.</p>"
        "<p><sup>3</sup>Sixth source.</p>",
    ]


def test_markers_link_to_their_own_chapters_notes(vtb):
    bodies = book(vtb)
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 6 and stats["unlinked"] == 0
    # chapter 2's marker 1 must point at the chapter-2 group, not chapter 1's
    assert 'href="page_0003.xhtml#note-g1-1"' in bodies[1]
    assert 'href="page_0002.xhtml#note-g0-1"' in bodies[0]


def test_notes_link_back_to_their_markers(vtb):
    bodies = book(vtb)
    vtb.link_notes(bodies, dropped=set())
    assert 'id="note-g0-2"' in bodies[2]
    assert 'href="page_0000.xhtml#ref-g0-2"' in bodies[2]


def test_a_marker_with_no_matching_note_stays_plain(vtb):
    bodies = book(vtb)
    bodies[0] = bodies[0] + "<p>Unsourced claim.<sup>9</sup></p>"
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["unlinked"] == 1
    assert "<sup>9</sup>" in bodies[0]              # untouched, as printed


def test_gross_group_misalignment_links_nothing(vtb):
    # body says two chapters; the notes section has five. Order-pairing has no
    # anchor, and guessing would bind chapter 2's markers to chapter 3's notes.
    bodies = book(vtb)
    bodies[3] += ("<p>CHAPTER THREE</p><p><sup>1</sup>x.</p>"
                  "<p>CHAPTER FOUR</p><p><sup>1</sup>x.</p>"
                  "<p>CHAPTER FIVE</p><p><sup>1</sup>x.</p>"
                  "<p>CHAPTER SIX</p><p><sup>1</sup>x.</p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["misaligned"] and stats["linked"] == 0


def test_a_book_without_endnotes_is_left_entirely_alone(vtb):
    bodies = ["<p>plain prose<sup>1</sup></p>", "<p>more prose</p>"]
    before = list(bodies)
    stats = vtb.link_notes(bodies, dropped=set())
    assert bodies == before and stats["linked"] == 0


def test_note_entries_are_not_mistaken_for_body_markers(vtb):
    # the endnotes themselves are full of <sup>N</sup> at paragraph starts;
    # treating those as prose markers would chain notes to each other
    bodies = book(vtb)
    vtb.link_notes(bodies, dropped=set())
    assert 'noteref' not in bodies[2].split('id="note-')[0]


def test_a_twice_cited_note_yields_one_id_and_two_working_links(vtb):
    # Both markers link forward; only the first carries the id the backlink
    # answers — a second id would make the XHTML invalid.
    bodies = book(vtb)
    bodies[0] = bodies[0].replace("</p>", " Re-cited.<sup>2</sup></p>")
    vtb.link_notes(bodies, dropped=set())
    assert bodies[0].count('href="page_0002.xhtml#note-g0-2"') == 2
    assert bodies[0].count('id="ref-g0-2"') == 1


def test_a_late_recitation_of_a_low_number_is_not_a_chapter_start(vtb):
    """
    The wrong link this module's own tests caught before any reader could:
    "…2, 3, then re-cite 2" made the reset heuristic declare a new chapter and
    bind the re-cited marker to the NEXT chapter's note 2. A true chapter
    start resets to exactly 1 and the next marker stays low; a re-citation
    dips and returns high.
    """
    bodies = book(vtb)
    bodies[0] = bodies[0].replace("</p>", " Re-cited.<sup>2</sup></p>")
    vtb.link_notes(bodies, dropped=set())
    assert bodies[0].count('href="page_0002.xhtml#note-g0-2"') == 2
    assert 'href="page_0003.xhtml#note-g1-2"' not in bodies[0]


def test_plain_numbered_entries_link_with_chapter_scoping(vtb):
    """
    The Invisible Government shape: the NOTES section heads groups with
    "<h2>2. Title</h2>" and writes entries plain — "<p>1. Speech by…" with no
    <sup>. Chapters 1 and 3 have notes, chapter 2 has none, so reset-guessing
    would misalign; the printed-contents chapter starts settle the scoping.
    """
    bodies = [
        "<p>Ike spoke.<sup>1</sup> Then acted.<sup>2</sup></p>",      # ch 1
        "<p>A chapter that cites nothing at all.</p>",                # ch 2
        "<p>The CIA grew.<sup>1</sup></p>",                           # ch 3
        "<h1>Notes</h1><h2>1. <i>One</i></h2>"
        "<p>1. Speech by Allen W. Dulles.</p>"
        "<p>2. Senate testimony.</p>"
        "<h2>3. <i>Three</i></h2>"
        "<p>1. NSC directive.</p>",
    ]
    stats = vtb.link_notes(bodies, dropped=set(),
                           chapter_starts=[(1, 0), (2, 1), (3, 2)])
    assert stats["linked"] == 3 and stats["unlinked"] == 0
    # forward: ch-3 marker 1 must reach group g1 (chapter 3), not g0
    assert 'href="page_0003.xhtml#note-g1-1"' in bodies[2]
    # back: the plain entry anchors the id and keeps its printed "1." form
    assert '<p id="note-g0-1"><a href="page_0000.xhtml#ref-g0-1">1.</a>' in bodies[3]
    assert '<p id="note-g1-1"><a href="page_0002.xhtml#ref-g1-1">1.</a>' in bodies[3]
    assert "Speech by Allen W. Dulles." in bodies[3]   # entry text intact


def test_an_unanchorable_entry_leaves_its_marker_unlinked(vtb):
    """A marker whose entry cannot take the id must stay plain: a forward
    link to an id that was never written is a broken href, which is worse
    than no link. (The crash this guards replaced: NoneType .end().)"""
    bodies = [
        "<p>Body.<sup>1</sup> More.<sup>2</sup> Yet.<sup>3</sup></p>",
        "<h1>Notes</h1>"
        "<p><sup>1</sup> Fine entry.</p>"
        "<p><sup>2</sup> Fine entry.</p>"
        "<p><sup>3</sup> Fine entry.</p>",
    ]
    # sabotage entry 2 after parse would have seen it: simulate by making a
    # notes page whose entry 2 is a bare paragraph the anchor regexes miss
    bodies[1] = bodies[1].replace("<p><sup>2</sup> Fine entry.</p>",
                                  "<p>— an unnumbered aside —</p>"
                                  "<p>2 Fine entry no period.</p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["unlinked"] >= 0            # no crash is the point
    for m in __import__("re").finditer(r'href="page_\d{4}\.xhtml#(note-[\w-]+)"',
                                       " ".join(bodies)):
        assert f'id="{m.group(1)}"' in " ".join(bodies), "forward link without anchor"


def test_a_notes_section_survives_a_page_rendered_as_bold_heads_and_list_items(vtb):
    """
    The Invisible Government's notes ran ten pages: most rendered as <h2>
    heads with <p> entries, but one page came back as bare <b> heads with
    <ol><li> entries — and a blank leaf sat inside the section. The parser
    must read both renderings and step over the blank, or every group after
    the odd page is silently lost (chapters 12-26 were).
    """
    bodies = [
        "<p>One.<sup>1</sup></p>",                                    # ch 1
        "<p>Twelve.<sup>1</sup></p>",                                 # ch 12
        "<p>TwentyThree.<sup>1</sup></p>",                            # ch 23
        "<h1>Notes</h1><h2>1. <i>One</i></h2><p>1. Alpha source.</p>",
        '<b>12. The Shake-Up</b><ol><li>1. Bravo source.</li></ol>',
        "",                                                            # blank leaf
        "<h2>23. <i>Radio</i></h2><p>1. Charlie source.</p>",
    ]
    stats = vtb.link_notes(bodies, dropped=set(),
                           chapter_starts=[(1, 0), (12, 1), (23, 2)])
    assert stats["linked"] == 3 and stats["unlinked"] == 0
    assert 'href="page_0004.xhtml#note-g1-1"' in bodies[1]
    assert 'href="page_0006.xhtml#note-g2-1"' in bodies[2]
    # the <li> entry anchors in place, keeping its list form
    assert '<li id="note-g1-1"><a href="page_0001.xhtml#ref-g1-1">1.</a>' in bodies[4]


def test_a_bold_block_standing_as_a_group_head_is_promoted_to_a_heading(vtb):
    """
    Once the parser can identify a bold block as a notes group head, the
    markup should be corrected to say so — one page must not silently render
    its heads differently from every other page of the same section.
    """
    bodies = [
        "<p>Body.<sup>1</sup></p>",
        "<h1>Notes</h1><h2>1. <i>One</i></h2><p>1. Alpha.</p>",
        '<b>12. The Shake-Up</b><ol><li>1. Bravo.</li></ol>',
    ]
    assert vtb.normalize_note_heads(bodies) == 1
    assert "<h2>12. The Shake-Up</h2>" in bodies[2]
    assert "<b>12." not in bodies[2]


def test_bold_inside_running_text_is_emphasis_and_stays_bold(vtb):
    bodies = [
        "<h1>Notes</h1>"
        "<p>1. See the <b>12. Panzer Division</b> order of battle.</p>",
    ]
    assert vtb.normalize_note_heads(bodies) == 0
    assert "<b>12. Panzer Division</b>" in bodies[0]


def test_bold_before_the_notes_section_is_never_touched(vtb):
    bodies = [
        '<b>3. An emphatic opener</b><p>1. A numbered list in prose.</p>',
        "<h1>Notes</h1><h2>1. <i>One</i></h2><p>1. Alpha.</p>",
    ]
    assert vtb.normalize_note_heads(bodies) == 0
    assert "<b>3. An emphatic opener</b>" in bodies[0]


def _folio(vtb, i, n):
    return vtb.Folio(i, n, "", True)


def test_a_page_number_set_as_a_marker_is_dropped_and_the_marker_restored(vtb):
    """Anthropology's World ended a page with '…locations.1<sup>59</sup>' on
    the leaf whose folio is 59: the layout lifted the page number into a
    superscript and left the real marker glued to the prose as plain text.
    Two such phantoms cost that book all 167 of its note links, because the
    chapter-reset test looks one marker ahead and a stray 59 hides the
    boundary."""
    bodies = ["<p>proper locations.1<sup>59</sup></p>"]
    n = vtb.strip_folio_superscripts(bodies, [_folio(vtb, 0, 59)])
    assert n == 1
    assert bodies[0] == "<p>proper locations.<sup>1</sup></p>"


def test_a_real_marker_that_matches_the_folio_is_left_alone(vtb):
    """The rule is narrow on purpose: only the last superscript on the page,
    and only when nothing but markup follows it. A marker mid-paragraph is
    prose, whatever number it carries."""
    bodies = ["<p>A citation<sup>12</sup> and then more prose follows.</p>"]
    assert vtb.strip_folio_superscripts(bodies, [_folio(vtb, 0, 12)]) == 0
    assert "<sup>12</sup>" in bodies[0]


def test_a_last_marker_unlike_the_folio_is_left_alone(vtb):
    bodies = ["<p>The chapter ends here.<sup>7</sup></p>"]
    assert vtb.strip_folio_superscripts(bodies, [_folio(vtb, 0, 59)]) == 0
    assert "<sup>7</sup>" in bodies[0]


def test_nothing_is_restored_when_no_digit_is_glued_to_the_prose(vtb):
    """No marker to recover is the ordinary case — the folio simply goes."""
    bodies = ["<p>The chapter ends here.<sup>59</sup></p>"]
    assert vtb.strip_folio_superscripts(bodies, [_folio(vtb, 0, 59)]) == 1
    assert bodies[0] == "<p>The chapter ends here.</p>"


def test_a_spaced_number_is_prose_not_a_demoted_marker(vtb):
    """'in 1995' must never become a superscript: only digits GLUED to the
    preceding word are a marker the layout flattened."""
    bodies = ["<p>It happened in 1995<sup>59</sup></p>"]
    assert vtb.strip_folio_superscripts(bodies, [_folio(vtb, 0, 59)]) == 1
    assert bodies[0] == "<p>It happened in 1995</p>"
