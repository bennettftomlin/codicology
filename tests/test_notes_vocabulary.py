"""What a notes section is called, what its groups are called, and what is
not a notes section for all that it says the word.

Every heading on the shelf that says "notes" was listed on 12 November 2026
(151 books), and every unlinked note call traced to the condition that left
it plain. Five books had lost 1,472 calls between them to words, not to
structure:

  Reconstruction           its notes are headed REFERENCE NOTES            191
  The War Correspondent    its groups are headed "CHAPTER 3. THE TITLE"    188
  Other People's Money     its groups are headed "Notes to Chapter 1"       89
  KGB                      one "1." inside a note opened a group           643
  Small is Necessary       a page headed NOTE, in the front matter         361

and Theories of Social Capital 14 more to a numbered list inside a note.
Each rule here is the general form of one of those, and each is tested on
the thing it must not take as well.
"""
import pytest


def chapter(n):
    return "<p>Prose.<sup>1</sup> More.<sup>2</sup> Again.<sup>3</sup> And.<sup>4</sup></p>"


def entries(first=1, last=4):
    return "".join(f"<p>{k}. Source {k}.</p>" for k in range(first, last + 1))


# ── what a notes section is called ───────────────────────────────────────────

@pytest.mark.parametrize("head", [
    "NOTES", "Notes", "FOOTNOTES", "Endnotes", "End Notes", "REFERENCE NOTES",
    "Foot-notes", "Note"])
def test_a_notes_head_by_any_of_its_names(vtb, head):
    assert vtb.NOTES_HEAD.search(f"<h1>{head}</h1>")
    assert vtb.NOTES_SECTION_HEAD.search(f"<h2>{head}</h2>")


@pytest.mark.parametrize("head", [
    "Notes on Contributors", "A Note on the Text", "Notes payable to order",
    "Author's Note", "Notes to Chapter 1", "Concluding note", "Promissory Notes",
    "NOTES · BIBLIOGRAPHY", "Technical Notes", "Note to the Reader",
    "Source Notes"])       # in both books that have it, a list keyed by page
def test_a_heading_about_something_is_not_a_notes_head(vtb, head):
    assert not vtb.NOTES_HEAD.search(f"<h1>{head}</h1>")
    assert not vtb.NOTES_SECTION_HEAD.search(f"<h1>{head}</h1>")


def test_notes_under_another_name_are_linked(vtb):
    """Allen's Reconstruction: REFERENCE NOTES, then a group to a chapter."""
    bodies = [chapter(1), chapter(2),
              "<h1>REFERENCE NOTES</h1><h2>INTRODUCTION</h2>" + entries()
              + "<h2>CHAPTER I</h2>" + entries()]
    assert vtb.count_notes_heads(bodies) == 1
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 8 and stats["groups"] == 2


# ── what its groups are called ───────────────────────────────────────────────

@pytest.mark.parametrize("head, number", [
    ("CHAPTER 3. JOURNALISM, OBJECTIVITY AND WAR", 3),
    ("Chapter 12: The Press", 12),
    ("CHAPTER IV — THE ROOTS OF COERCION", 4),
    ("Chapter Seven. Aftermath", 7),
    ("<i>Chapter 2.</i> Beginnings", 2)])
def test_a_group_head_may_carry_its_chapters_title(vtb, head, number):
    bodies = [chapter(1), "<h1>Notes</h1>" + f"<h2>{head}</h2>" + entries()]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1 and len(groups[0]) == 4
    assert vtb.parse_notes_section.last_group_numbers == [number]


@pytest.mark.parametrize("not_a_head", [
    "<p>Chapter 3: see the discussion in Smith.</p>",        # a paragraph is somebody's note
    "<h2>Chapter 3 was written last</h2>",                   # no mark between number and words
    "<h2>Chapters 3 and 4: sources</h2>",
    "<h2>CHAPTER MMXX. Nothing</h2>"])                       # not a number the book counts in
def test_what_merely_begins_with_the_word_opens_no_group(vtb, not_a_head):
    bodies = [chapter(1), "<h1>Notes</h1>" + not_a_head + entries()]
    _, groups = vtb.parse_notes_section(bodies)
    # with no group opened the entries are one ungrouped run, not a chapter's
    assert vtb.parse_notes_section.last_group_numbers == [None]


def test_titled_heads_link_a_book_chapter_by_chapter(vtb):
    """The War Correspondent."""
    bodies = [chapter(1), chapter(2),
              "<h1>Notes</h1><h2>CHAPTER 1. INTRODUCTION</h2>" + entries()
              + "<h2>CHAPTER 2. THE WAR CORRESPONDENT: RISK AND TRADITION</h2>" + entries()]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 8 and stats["groups"] == 2
    assert 'href="page_0002.xhtml#note-g1-1"' in bodies[1]


@pytest.mark.parametrize("head", [
    "Notes to Chapter 1", "NOTES TO CHAPTER ONE", "Notes to the Introduction",
    "Notes for Chapter 4", "<i>Notes to Preface</i>"])
def test_a_group_may_say_whose_notes_it_holds(vtb, head):
    bodies = [chapter(1), "<h1>Notes</h1>" + f"<h2>{head}</h2>" + entries()]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1 and len(groups[0]) == 4
    assert not vtb.NOTES_HEAD.search(f"<h2>{head}</h2>")     # and is not itself a notes head


def test_notes_to_something_else_is_not_a_group(vtb):
    assert not vtb.GROUP_HEAD.search("<h2>Notes to the Bibliography</h2>")
    assert not vtb.GROUP_HEAD.search("<h2>Notes to Chapter 1 and after</h2>")


# ── a numbered list inside a note ────────────────────────────────────────────

def stream(*numbers):
    return [(0, k, "group" if n == "head" else "note", None if n == "head" else n)
            for k, n in enumerate(numbers)]


def test_a_list_inside_a_note_is_told_by_what_follows_it(vtb):
    # note 15 holds "1. …", and note 16 follows
    assert vtb._lists_inside_notes(stream("head", 13, 14, 15, 1, 16, 17)) == {4}
    # a longer list: 1, 2, 3 inside note 15
    assert vtb._lists_inside_notes(stream("head", 14, 15, 1, 2, 3, 16)) == {3, 4, 5}


def test_a_chapters_notes_beginning_again_are_not_a_list(vtb):
    # the next chapter's notes, its head unread: they go on from 1
    assert vtb._lists_inside_notes(stream("head", 13, 14, 15, 1, 2, 3, 4)) == set()
    # and end the section
    assert vtb._lists_inside_notes(stream("head", 14, 15, 1, 2, 3)) == set()
    # a head between says so itself
    assert vtb._lists_inside_notes(stream("head", 14, 15, "head", 1, 16)) == set()
    # what follows is not the next of the notes it interrupted
    assert vtb._lists_inside_notes(stream("head", 14, 15, 1, 2, 17)) == set()
    # the numbers had not passed 1: a group whose first entries read 1, 1
    assert vtb._lists_inside_notes(stream("head", 1, 1, 2, 3)) == set()


def test_one_stray_entry_does_not_put_every_later_chapter_one_out(vtb):
    """KGB: Masters of the Soviet Union lost all 643 of its calls to one."""
    one = "".join(f"<p>{k}. Source {k}.</p>" for k in (1, 2, 3))
    one += "<p>4. Four points are made:</p><p>1. That the first is so.</p><p>5. Source 5.</p>"
    body = "<p>" + "".join(f"Prose.<sup>{k}</sup> " for k in range(1, 6)) + "</p>"
    bodies = [body, chapter(2), chapter(3),
              "<h1>Notes</h1><h2>Chapter 1</h2>" + one
              + "<h2>Chapter 2</h2>" + entries() + "<h2>Chapter 3</h2>" + entries()]
    _, groups = vtb.parse_notes_section(bodies)
    assert [len(g) for g in groups] == [5, 4, 4]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 13 and not stats["misaligned"]
    assert "<p>1. That the first is so.</p>" in bodies[3]       # left as it was read


def test_a_return_to_one_still_opens_the_group_whose_head_went_unread(vtb):
    bodies = [chapter(1), chapter(2),
              "<h1>Notes</h1><h2>Chapter 1</h2>" + entries() + "<p><i>Chapter</i> 2</p>" + entries()]
    _, groups = vtb.parse_notes_section(bodies)
    assert [len(g) for g in groups] == [4, 4]


# ── a heading that says NOTE and is not a notes section ──────────────────────

def test_a_notes_head_is_one_that_notes_follow(vtb):
    front = "<h2>NOTE</h2><p>All sums are given in the currency of the day.</p>"
    caution = "<h3>NOTE</h3><p>Do not move a casualty whose neck may be broken.</p>"
    real = "<h1>Notes</h1><h2>Chapter 1</h2>" + entries()
    foot_of_page = "<p>The chapter ends here.</p><h2>Notes</h2>"
    bodies = [front, chapter(1), caution, "<p>Prose with nothing numbered in it.</p>",
              foot_of_page, entries(), real]
    assert [bool(vtb.notes_head_on(bodies, i)) for i in range(len(bodies))] == [
        False, False, False, False, True, False, True]
    assert vtb.count_notes_heads(bodies) == 2
    assert vtb.notes_section_start(bodies) == 4


def test_the_section_begins_at_the_first_head_that_notes_follow(vtb):
    """Small is Necessary: a NOTE on its terms, then nine chapters, then the
    Notes. The section was taken to begin on the first of the two."""
    bodies = ["<h2>NOTE</h2><p>All sums are given in the currency of the day.</p>",
              chapter(1), chapter(2),
              "<h1>Notes</h1><h2>Chapter 1</h2>" + entries() + "<h2>Chapter 2</h2>" + entries()]
    assert vtb.notes_section_start(bodies) == 3
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 8 and stats["groups"] == 2


def test_a_book_none_of_whose_heads_is_followed_by_notes_keeps_its_first(vtb):
    bodies = ["<p>Prose.</p>", "<h1>References</h1><p>Smith, J. 1990. A Book.</p>", "<p>More.</p>"]
    assert vtb.notes_section_start(bodies) == 1
    assert vtb.count_notes_heads(bodies) == 0
    assert vtb.notes_section_start(["<p>No heads at all.</p>"]) is None


def test_cautions_headed_note_do_not_make_a_book_of_chapter_notes(vtb):
    """Two dozen NOTE headings in a field manual were two dozen notes
    sections, and the count is what chooses a book's linker."""
    caution = "<h3>NOTE</h3><p>Do not move a casualty whose neck may be broken.</p>"
    bodies = ["<p>Step one.</p>" + caution, "<p>Step two.</p>" + caution, "<p>Step three.</p>"]
    assert vtb.count_notes_heads(bodies) == 0
    assert sum(1 for b in bodies if vtb.NOTES_HEAD.search(b)) == 2


# ── a power is not a note call ───────────────────────────────────────────────

@pytest.mark.parametrize("read, set_as", [
    ("it controlled over 60,000 km<sup>2</sup> of land", "it controlled over 60,000 km² of land"),
    ("smaller than 35ft<sup>2</sup> [32.6m<sup>2</sup>]", "smaller than 35ft² [32.6m²]"),
    ("<td>ft<sup>2</sup></td><td>square feet</td>", "<td>ft²</td><td>square feet</td>"),
    ("kWh/(m<sup>2</sup>a)", "kWh/(m²a)"),
    ("about 4 m<sup> 3 </sup> of timber", "about 4 m³ of timber")])
def test_a_raised_two_or_three_on_a_unit_of_length_is_a_power(vtb, read, set_as):
    assert vtb.set_unit_powers(read) == set_as


@pytest.mark.parametrize("call", [
    "the ‘non-debates’ of the late 1970s<sup>75</sup>",      # a note call after a decade
    "Colin Turnbull’s<sup>32</sup> account",                 # after a possessive
    "the film<sup>2</sup> was shown",                        # a word that ends in m
    "as shown in<sup>2</sup> the survey",                    # "in" is a word
    "Dimension, ft<sup>1</sup>",                             # a note mark in a table
    "over 60,000 km <sup>2</sup>",                           # not standing on the unit
    "ten km<sup>12</sup>"])
def test_a_note_call_is_left_a_note_call(vtb, call):
    assert vtb.set_unit_powers(call) == call


def test_a_unit_set_as_a_power_is_nothing_a_linker_binds(vtb):
    bodies = ["<p>Prose.<sup>1</sup> It held 60,000 km<sup>2</sup> by then.<sup>2</sup> "
              "More.<sup>3</sup> And.<sup>4</sup></p>", chapter(2),
              "<h1>Notes</h1><h2>Chapter 1</h2>" + entries() + "<h2>Chapter 2</h2>" + entries()]
    bodies = [vtb.set_unit_powers(b) for b in bodies]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 8 and "60,000 km² by then" in bodies[0]


# ── nor is an edition ────────────────────────────────────────────────────────

def test_a_section_designation_alone_heads_a_group(vtb):
    """Ste. Croix: "[I.ii]", "[IV.vi]", and one read as "[III. vi]"."""
    for head in ("[I.ii]", "[IV.vi]", "[III. vi]", "I.ii", "(2.3)", "<i>[VIII.iv]</i>"):
        bodies = [chapter(1), "<h1>Notes</h1>" + f"<h2>{head}</h2>" + entries()]
        _, groups = vtb.parse_notes_section(bodies)
        assert len(groups) == 1 and len(groups[0]) == 4, head
    for not_one in ("[I.ii] and after", "Part I", "1945", "IV", "[see II.iv]"):
        assert not vtb.GROUP_HEAD_SECTION.search(f"<h2>{not_one}</h2>"), not_one


def edition_book(body_one):
    return [body_one, chapter(2),
            "<h1>Notes</h1><h2>[I.i]</h2>" + entries(1, 5) + "<h2>[I.ii]</h2>" + entries()]


def test_a_raised_figure_on_a_title_that_has_not_climbed_is_its_edition(vtb):
    bodies = edition_book(
        "<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup> As Rostovtzeff has it "
        "(<i>SEHRE</i><sup>2</sup> 344) and the stone says (<i>IG</i> II<sup>2</sup> 1672), "
        "and the dictionary (OCD<sup>2</sup>). Four.<sup>4</sup> Five.<sup>5</sup></p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 9 and stats["unlinked"] == 3
    assert "<i>SEHRE</i><sup>2</sup>" in bodies[0] and "II<sup>2</sup>" in bodies[0]
    assert "OCD<sup>2</sup>" in bodies[0]


def test_a_call_on_a_word_in_italics_that_goes_on_climbing_is_a_call(vtb):
    bodies = edition_book(
        "<p>One.<sup>1</sup> Two.<sup>2</sup> The <i>dēmos</i><sup>3</sup> met; its "
        "<i>auctoritas</i><sup>4</sup> grew. Five.<sup>5</sup></p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 9 and stats["unlinked"] == 0


def test_a_figure_once_shown_to_be_a_titles_edition_is_its_edition_everywhere(vtb):
    """Met just after call 1, an edition's 2 has climbed, and looks like
    call 2. Met again on the same title after call 3, it has said what it
    is, there and here."""
    bodies = edition_book(
        "<p>One.<sup>1</sup> So Woodhouse (<i>PL</i><sup>2</sup> 53). Two.<sup>2</sup> "
        "Three.<sup>3</sup> Four.<sup>4</sup> Five.<sup>5</sup></p>")
    bodies[1] = bodies[1].replace("Again.<sup>3</sup>", "Again.<sup>3</sup> (<i>PL</i><sup>2</sup> 71)")
    stats = vtb.link_notes(bodies, dropped=set())
    assert bodies[0].count("<i>PL</i><sup>2</sup>") == 1 and "<i>PL</i><sup>2</sup>" in bodies[1]
    assert stats["linked"] == 9 and stats["unlinked"] == 2


def test_the_same_call_on_the_same_abbreviation_in_two_chapters_is_two_calls(vtb):
    """Met twice is not shown: "in the EU¹" may open two chapters."""
    bodies = edition_book(
        "<p>The largest in the EU<sup>1</sup> by then. Two.<sup>2</sup> Three.<sup>3</sup> "
        "Four.<sup>4</sup> Five.<sup>5</sup></p>")
    bodies[1] = bodies[1].replace("Prose.<sup>1</sup>", "And again in the EU<sup>1</sup> later.")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 9 and stats["unlinked"] == 0


def test_a_re_citation_after_ordinary_words_is_still_linked(vtb):
    """Only a figure on a title is doubted. A chapter that cites its
    second note again is read as before."""
    bodies = edition_book(
        "<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup> As was said above.<sup>2</sup> "
        "Four.<sup>4</sup> Five.<sup>5</sup></p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 10 and stats["unlinked"] == 0
