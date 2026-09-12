"""Chapter-endnotes linking: a "Notes" section closes each chapter.

The scoping rule is position and nothing else — markers since the previous
section belong to this one. What needs testing is the honesty at the edges:
entries that continue onto the next page are accepted only while their
numbers keep ascending, numbers reused across chapters bind locally, and a
marker with no entry stays plain rather than pointing at nothing.
"""
import re


def _book():
    return [
        "<p>Front matter.</p>",                                      # 0
        "<p>One begins.<sup>1</sup> More.<sup>2</sup></p>",          # 1
        '<p>End of one.</p><h2>Notes</h2><p>1. Alpha source.</p>',   # 2
        "<p>2. Beta source.</p>",                                    # 3 (continues)
        "<p>Two begins.<sup>1</sup></p>",                            # 4
        "<p>1. Not a note, a list the prose happens to number.</p>", # 5
        '<h2>Notes</h2><p>1. Gamma source.</p>',                     # 6
    ]


def test_markers_bind_to_their_own_chapters_section(vtb):
    bodies = _book()
    stats = vtb.link_chapter_notes(bodies, dropped=set())
    assert stats == {"linked": 3, "unlinked": 0, "sections": 2}
    # chapter 1's markers reach section 0's entries, including the page-3 continuation
    assert 'href="page_0002.xhtml#note-c0-1"' in bodies[1]
    assert 'href="page_0003.xhtml#note-c0-2"' in bodies[1]
    # chapter 2's marker 1 reaches section 1, NOT section 0's note 1
    assert 'href="page_0006.xhtml#note-c1-1"' in bodies[4]
    assert 'note-c0-1"' not in bodies[4]
    # backlinks anchor in place, plain entries keeping their printed form
    assert '<p id="note-c0-1"><a href="page_0001.xhtml#ref-c0-1">1.</a>' in bodies[2]


def test_a_numbered_list_in_the_next_chapter_is_not_swallowed_as_notes(vtb):
    bodies = _book()
    vtb.link_chapter_notes(bodies, dropped=set())
    # the page-5 list paragraph gained no anchor: section 0's entries ended
    # at 2, and a fresh "1." does not continue that sequence
    assert 'id="note-' not in bodies[5]


def test_a_marker_without_an_entry_stays_plain(vtb):
    bodies = [
        "<p>Text.<sup>1</sup> And.<sup>7</sup></p>",
        '<h2>Notes</h2><p>1. Only source.</p>',
        "<p>Two.<sup>1</sup></p>",
        '<h2>Notes</h2><p>1. Other source.</p>',
    ]
    stats = vtb.link_chapter_notes(bodies, dropped=set())
    assert stats["linked"] == 2 and stats["unlinked"] == 1
    assert "<sup>7</sup>" in bodies[0]          # untouched, as printed


def test_one_notes_head_is_not_this_layout(vtb):
    bodies = ["<p>Body.<sup>1</sup></p>",
              '<h1>Notes</h1><p>1. Source.</p>']
    stats = vtb.link_chapter_notes(bodies, dropped=set())
    assert stats == {"linked": 0, "unlinked": 0, "sections": 0}


def test_twice_cited_note_gets_one_id_and_two_links(vtb):
    bodies = [
        "<p>Cite.<sup>1</sup> Again.<sup>1</sup></p>",
        '<h2>Notes</h2><p>1. The source.</p>',
        "<p>x.<sup>1</sup></p>",
        '<h2>Notes</h2><p>1. Other.</p>',
    ]
    vtb.link_chapter_notes(bodies, dropped=set())
    assert bodies[0].count('href="page_0001.xhtml#note-c0-1"') == 2
    assert bodies[0].count('id="ref-c0-1"') == 1


def test_a_back_of_book_layout_falls_back_when_chapter_sections_find_nothing(vtb):
    """
    Counting "Notes" headings guesses the layout. A back-of-book section
    whose title appears twice — in the contents and again at the section —
    looks like chapter endnotes, parses to nothing, and used to give up:
    one book lost all 893 of its links that way, silently.
    """
    bodies = [
        "<p>Contents</p><p>Notes</p>",                  # the word, in the contents
        "<p>One.<sup>1</sup> Two.<sup>2</sup></p>",
        "<p>Three.<sup>1</sup></p>",
        "<h1>Notes</h1><p>CHAPTER ONE</p>"
        "<p><sup>1</sup> Alpha.</p><p><sup>2</sup> Beta.</p>"
        "<p>CHAPTER TWO</p><p><sup>1</sup> Gamma.</p>",
    ]
    heads = sum(1 for b in bodies if vtb.NOTES_HEAD.search(b))
    chapter = vtb.link_chapter_notes([b for b in bodies], dropped=set())
    assert chapter["sections"] == 0, "fixture must defeat the chapter path"
    stats = vtb.link_notes(bodies, dropped=set())
    # All three bind, each to its own chapter. Chapter two has a single
    # note and therefore a single marker, and a one-marker group stands
    # because the notes section says that chapter exists — without that it
    # was either dropped (the marker left plain) or folded into chapter one
    # (the marker sent to Alpha, which is worse).
    assert stats["linked"] == 3 and not stats["misaligned"], stats
    assert stats["groups"] == 2, stats
    assert 'epub:type="noteref"' in bodies[1]
    assert 'href="page_0003.xhtml#note-g1-1"' in bodies[2], "must reach Gamma"
    assert '<p id="note-g1-1">' in bodies[3]


def test_entries_whose_forms_alternate_all_link(vtb):
    """sup 1, plain 2, sup 3 on one page: three sequential per-pattern
    passes shared one ascending counter, so the sup pass consumed 1, met
    3 while expecting 2, and dropped it forever — entry 3's marker stayed
    unlinked though the entry was on the page. One position-ordered scan
    takes them as printed."""
    bodies = [
        "<p>Alpha.<sup>1</sup> Beta.<sup>2</sup> Gamma.<sup>3</sup></p>",
        '<h2>Notes</h2><p><sup>1</sup> First source.</p>'
        '<p>2. Second source.</p>'
        '<p><sup>3</sup> Third source.</p>',
        '<h2>Notes</h2><p>1. Unrelated next chapter.</p>',
    ]
    stats = vtb.link_chapter_notes(bodies, dropped=set())
    assert stats["linked"] == 3, "the alternating-form entry must link"
    assert 'href="page_0001.xhtml#note-c0-3"' in bodies[0]


# ── which layout a book actually uses is settled by the book ────────────────

def _chapter_layout(n_chapters=2, notes_each=3):
    """Prose chapters each followed by their own Notes section."""
    bodies = []
    for c in range(n_chapters):
        bodies.append("<p>text" + "".join(f"<sup>{k}</sup>"
                      for k in range(1, notes_each + 1)) + "</p>")
        bodies.append("<h2>NOTES</h2><ol>" + "".join(
            f"<li>{k}. c{c} source {k}</li>"
            for k in range(1, notes_each + 1)) + "</ol>")
    return bodies


def test_a_true_chapter_layout_still_wins(vtb):
    """Both layouts are read now, so the ordinary case must not change."""
    bodies = _chapter_layout()
    stats = vtb.link_chapter_notes(list(bodies), set())
    assert stats["sections"] == 2 and stats["linked"] == 6
    # and through the real dispatch the same links come out
    out = list(bodies)
    vtb.link_chapter_notes(out, set())
    assert out[0].count("noteref") == 3 and out[2].count("noteref") == 3


def test_a_back_of_book_section_that_says_notes_twice_is_not_two_chapters(vtb):
    """One continuous back-of-book section, its title set again on the page
    where the entries begin. Read as chapter-endnotes it scopes by position
    and accounts for almost nothing; read as one grouped section it binds
    every marker. The book decides, by how many of its own markers each
    layout can account for."""
    prose = [f"<p>ch{c}." + "".join(f"<sup>{k}</sup>" for k in range(1, 6))
             + "</p>" for c in range(4)]
    notes = ["<h1>NOTES</h1>",
             "<h1>NOTES</h1>" + "".join(
                 f"<h2>Chapter {w}</h2><ol>" + "".join(
                     f"<li>{k}. c{c} src {k}</li>" for k in range(1, 6))
                 + "</ol>"
                 for c, w in enumerate(("One", "Two", "Three", "Four")))]
    bodies = prose + notes
    chapter = vtb.link_chapter_notes(list(bodies), set())
    section = vtb.link_notes(list(bodies), set())
    # The narrow test — did the chapter layout find ANY section — says yes,
    # and it binds every marker, so a link count alone cannot separate them.
    assert chapter["sections"] == 1 and chapter["linked"] == 20
    # but all four chapters were sent to chapter one's notes
    bound = list(bodies)
    vtb.link_chapter_notes(bound, set())
    assert all("note-c0-1" in bound[c] for c in range(4)), "not the shape meant"
    # the section layout tells the four apart, and pairs them with the body
    assert section["groups"] == 4 and not section["misaligned"]
    assert section["linked"] == 20
    read = list(bodies)
    vtb.link_notes(read, set())
    assert [re.search(r"#note-g(\d+)-1", read[c]).group(1) for c in range(4)] \
        == ["0", "1", "2", "3"], "each chapter must reach its own notes"
