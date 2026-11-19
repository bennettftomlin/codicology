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
    # Read as chapter-endnotes, the first NOTES head has no entries of its
    # own: they are under the second head, and a section's notes never run
    # past the next section's head. It once read on into them and sent all
    # four chapters to chapter one's notes; now it binds nothing.
    assert chapter["sections"] == 0 and chapter["linked"] == 0
    # the section layout tells the four apart, and pairs them with the body
    assert section["groups"] == 4 and not section["misaligned"]
    assert section["linked"] == 20
    read = list(bodies)
    vtb.link_notes(read, set())
    assert [re.search(r"#note-g(\d+)-1", read[c]).group(1) for c in range(4)] \
        == ["0", "1", "2", "3"], "each chapter must reach its own notes"


# ── a missing entry does not end its group ───────────────────────────────────

def _missing(n_notes=10, gone=(7,), extra=""):
    """Two chapters of n_notes calls each; in the first chapter's notes the
    entries numbered in `gone` were not read (a misread, a dropped line)."""
    bodies = []
    for c in range(2):
        bodies.append("<p>text" + "".join(f" word<sup>{k}</sup>"
                      for k in range(1, n_notes + 1)) + "</p>")
        bodies.append("<h2>NOTES</h2>" + "".join(
            f"<p>{k}. c{c} source {k}.</p>" for k in range(1, n_notes + 1)
            if not (c == 0 and k in gone)) + (extra if c == 0 else ""))
    return bodies


def test_the_entries_after_a_missing_one_still_link(vtb):
    """2083 and 2106: one entry the reading lost stopped its whole group,
    and every call after it went unlinked."""
    bodies = _missing()
    stats = vtb.link_chapter_notes(bodies, set())
    assert stats["sections"] == 2
    for k in (6, 8, 9, 10):
        assert f'href="page_0001.xhtml#note-c0-{k}"' in bodies[0], k
    # the call whose entry is gone stays as printed
    assert 'note-c0-7"' not in bodies[0] and "<sup>7</sup>" in bodies[0]
    assert stats["linked"] == 19 and stats["unlinked"] == 1


def test_two_missing_together_are_stepped_over_and_three_are_not(vtb):
    bodies = _missing(gone=(4, 5))
    vtb.link_chapter_notes(bodies, set())
    assert 'note-c0-6"' in bodies[0] and 'note-c0-10"' in bodies[0]
    bodies = _missing(gone=(4, 5, 6))
    vtb.link_chapter_notes(bodies, set())
    assert 'note-c0-3"' in bodies[0] and 'note-c0-7"' not in bodies[0]


def test_a_stray_number_does_not_carry_the_walk_off(vtb):
    """A number that jumps but is not followed by its own next is not an
    entry: a year, a page reference, a figure in a note's own text."""
    bodies = _missing(n_notes=5, gone=(), extra="<p>8. A line that is not a note.</p>")
    vtb.link_chapter_notes(bodies, set())
    assert 'id="note-c0-8"' not in bodies[1]
    assert all(f'note-c0-{k}"' in bodies[0] for k in range(1, 6))


# ── an entry printed without its full stop ───────────────────────────────────

def _bare(entry_for_7, n_notes=10, extra=""):
    bodies = _missing(n_notes=n_notes, gone=(7,), extra=extra)
    bodies[1] = bodies[1].replace("<p>8. ", f"{entry_for_7}<p>8. ")
    return bodies


def test_an_entry_without_its_full_stop_links_between_its_neighbours(vtb):
    """2083: "<li>12 The newspaper…" between 11. and 13. left call 12 plain."""
    bodies = _bare("<p>7 The newspaper said so.</p>")
    stats = vtb.link_chapter_notes(bodies, set())
    assert 'href="page_0001.xhtml#note-c0-7"' in bodies[0]
    # the book printed no full stop and none is written
    assert '<p id="note-c0-7"><a href="page_0000.xhtml#ref-c0-7">7</a> The' in bodies[1]
    assert stats["linked"] == 20 and stats["unlinked"] == 0


def test_a_bare_number_out_of_sequence_is_not_an_entry(vtb):
    # "75 per cent…" opening a paragraph is wanted by nothing
    bodies = _bare("<p>75 per cent of the losses.</p>")
    vtb.link_chapter_notes(bodies, set())
    assert 'note-c0-75"' not in bodies[1] and 'note-c0-7"' not in bodies[0]
    # the right number, but what follows is not its next
    bodies = _missing(gone=(7, 8))
    bodies[1] = bodies[1].replace("<p>9. ", "<p>7 Ibid.</p><p>9. ")
    vtb.link_chapter_notes(bodies, set())
    assert 'id="note-c0-7"' not in bodies[1]
    assert 'note-c0-9"' in bodies[0]          # the gap step still carries on


# ── notes heads and entries the reader set unusually ────────────────────────

def _two_chapters(head1, entries1):
    return ["<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
            head1 + entries1,
            "<p>Next.<sup>1</sup> b.<sup>2</sup> c.<sup>3</sup></p>",
            "<h2>Notes</h2><ol><li>1. x</li><li>2. y</li><li>3. z</li></ol>"]


def test_a_notes_head_read_as_a_paragraph_still_opens_the_section(vtb):
    """Decolonising the University: "<p>Notes</p>", and the chapter's calls
    were bound to the next chapter's notes."""
    bodies = _two_chapters("<p>Notes</p>", "<ol><li>1. a</li><li>2. b</li><li>3. c</li></ol>")
    stats = vtb.link_chapter_notes(bodies, set())
    assert stats["sections"] == 2 and stats["linked"] == 6
    assert 'href="page_0001.xhtml#note-c0-1"' in bodies[0]
    assert 'href="page_0003.xhtml#note-c1-1"' in bodies[2]


def test_a_paragraph_saying_notes_without_entries_from_one_is_not_a_head(vtb):
    bodies = _two_chapters("<p>Notes</p>", "<p>On the sources used here.</p>")
    assert vtb.count_notes_heads(bodies) == 1


def test_entries_a_list_numbers_itself_are_counted(vtb):
    """Revolutionary Movements in Africa ch. 10: notes 1-4 numbered by the
    list, "5." onwards in text."""
    bodies = ["<p>a<sup>1</sup> b<sup>2</sup> c<sup>3</sup> d<sup>4</sup> e<sup>5</sup> f<sup>6</sup></p>",
              "<h2>NOTES</h2><ol><li>Kurita.</li><li>Leftist.</li><li>Sikainga.</li>"
              "<li>Johnson.</li></ol><ol style=\"list-style-type: none;\">"
              "<li>5. Vezzadini.</li><li>6. Young.</li></ol>",
              "<p>z<sup>1</sup></p>", "<h2>NOTES</h2><ol><li>1. q</li><li>2. r</li><li>3. s</li></ol>"]
    stats = vtb.link_chapter_notes(bodies, set())
    assert all(f'#note-c0-{k}"' in bodies[0] for k in range(1, 7))
    assert '<li id="note-c0-1">Kurita.' in bodies[1]


def test_a_list_the_page_broke_does_not_count_from_one_again(vtb):
    """Decolonising the University ch. 8: note 1 under the head, notes 2 on
    as a fresh list on the next page. Counted from 1 they were each one low."""
    bodies = ["<p>a<sup>1</sup> b<sup>2</sup> c<sup>3</sup> d<sup>4</sup></p>",
              "<h2>Notes</h2><ol style=\"list-style-type: none;\"><li>1. Tomaselli.</li></ol>",
              "<p>(eds) Handbook, p. 352.</p><ol><li>Kovach.</li><li>Clandinin.</li></ol>",
              "<p>z<sup>1</sup></p>", "<h2>Notes</h2><ol><li>1. q</li><li>2. r</li><li>3. s</li></ol>"]
    vtb.link_chapter_notes(bodies, set())
    # four calls and three notes: an entry was lost somewhere
    assert 'id="note-c0-2"' not in bodies[2]       # no numbers it cannot know


def test_a_list_the_page_broke_counts_on_where_the_totals_meet(vtb):
    """The same chapter as the book prints it: three calls, note 1 under the
    head, notes 2 and 3 as a fresh list on the next page. Counted on from 2,
    the section ends on the chapter's last call, so nothing was lost."""
    bodies = ["<p>a<sup>1</sup> b<sup>2</sup> c<sup>3</sup></p>",
              "<h2>Notes</h2><ol style=\"list-style-type: none;\"><li>1. Tomaselli.</li></ol>",
              "<p>(eds) Handbook, p. 352.</p><ol><li>Kovach.</li><li>Clandinin.</li></ol>",
              "<p>z<sup>1</sup></p>", "<h2>Notes</h2><ol><li>1. q</li><li>2. r</li><li>3. s</li></ol>"]
    vtb.link_chapter_notes(bodies, set())
    assert 'id="note-c0-2"' in bodies[2] and 'id="note-c0-3"' in bodies[2]
    assert 'href="page_0002.xhtml#note-c0-3"' in bodies[0]


def test_a_paragraph_notes_head_whose_first_note_is_long_counts(vtb):
    """Decolonising the University p92: "Notes" set as a paragraph, and note 1
    runs to 800 characters before note 2."""
    long_note = "On the decolonial turn see " + "Maldonado-Torres, The Decolonial Turn; " * 25
    bodies = ["<p>a<sup>1</sup> b<sup>2</sup> c<sup>3</sup></p>",
              f'<p>Notes</p><ol style="list-style-type: none;"><li>1. {long_note}</li>'
              "<li>2. Ibid.</li><li>3. Ibid. p. 4.</li></ol>"]
    assert vtb.notes_head_on(bodies, 1) is not None
