"""A running head the layout pass called content, caught by the book itself.

The layout model is not consistent about its own furniture. On The Rebel
Passion it labelled "<folio> THE REBEL PASSION" a PageHeader on 209 pages and
a SectionHeader on 49 -- same position on the page, same 0.99 confidence -- so
47 running heads shipped inside the prose, 35 of them as HEADINGS wedged into
sentences that ran across the page break. Nothing about the 49 differs from
the 209, so no test of shape or position can tell them apart.

What can is the book's own repetition: a running head is printed on every page
of a section, so a shape that recurs under a furniture label is the book saying
what that line is. That is the same evidence find_image_furniture uses for a
picture, at the same threshold, and these tests are mostly about the two
conditions that keep it off real structure --

  * parse_folio must read a page number out of the line, which refuses a bare
    chapter number standing over its title; and
  * the block beside it must not itself be a heading, because a chapter's
    OPENING page prints no running head.

Both were measured against every cached book before being written down: the
rule takes 57 lines in 8 books, and these two conditions are what spare 129
repeated section titles and 3 chapter openings from going with them.
"""
import pytest


def census(shape_counts):
    """page_furniture as read_pages builds it: the text of each page's heads."""
    pages = []
    for text, n in shape_counts.items():
        pages.extend([[text.format(i)] for i in range(1, n + 1)])
    return pages


# --------------------------------------------------------------------------
# the defect itself
# --------------------------------------------------------------------------

def test_a_head_the_layout_called_a_heading_goes_when_the_book_prints_it_elsewhere(vtb):
    bodies = ["<h2>10 THE REBEL PASSION</h2><p>Mass and sometimes in the very "
              "early morning when the birds sing.</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 9}))

    assert removed == [(0, "10 THE REBEL PASSION")]
    assert bodies == ["<p>Mass and sometimes in the very early morning when "
                      "the birds sing.</p>"]


def test_the_sentence_the_head_interrupted_survives_it_untouched(vtb):
    prose = ("<p>go into a convent, neither allow any man&#8217;s hand to come "
             "upon her.</p><p>So my father sent me to Glastonbury.</p>")
    bodies = ["<h2>12 THE REBEL PASSION</h2>" + prose]
    vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 9}))
    assert bodies[0] == prose


def test_a_running_foot_at_the_far_end_of_the_page_goes_the_same_way(vtb):
    bodies = ["<p>the text of the page</p><p>Saylor.org 872</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"Saylor.org {0}00": 4}))
    assert removed == [(0, "Saylor.org 872")]
    assert bodies == ["<p>the text of the page</p>"]


def test_a_page_carrying_a_head_and_a_foot_loses_both(vtb):
    bodies = ["<h3>46 THE REBEL PASSION</h3><p>prose</p><p>Saylor.org 46</p>"]
    removed = vtb.strip_missed_running_heads(
        bodies, census({"{0} THE REBEL PASSION": 5, "Saylor.org {0}": 5}))
    assert [t for _, t in removed] == ["46 THE REBEL PASSION", "Saylor.org 46"]
    assert bodies == ["<p>prose</p>"]


# --------------------------------------------------------------------------
# a head with no block of its own
# --------------------------------------------------------------------------
#
# The layout pass does not always give the running head a block. On 21 pages
# of one book it put the head and the first line of prose in ONE paragraph
# with a <br/> between them. The first version of this pass read a block's
# WHOLE text, which for a fused head is the head plus the paragraph it
# interrupts -- far past any length a running head has -- so all 21 survived.
#
# The check written to confirm that fix asked whether a block's entire text
# WAS a running head. It shared the assumption, so it reported the book clean
# and the heads shipped. The test below is the one that had to exist.


def both_dresses(n=9):
    """A book numbers versos at the front of the head and rectos at the back."""
    return ([[f"{2 * k} THE REBEL PASSION"] for k in range(5, 5 + n)]
            + [[f"THE REBEL PASSION {2 * k + 1}"] for k in range(5, 5 + n)])


def test_a_head_fused_to_the_paragraph_it_interrupts_is_still_found(vtb):
    bodies = ["<p>34 THE REBEL PASSION<br/>cruelly ill-used her. "
              "His strength was whole.</p>"]
    removed = vtb.strip_missed_running_heads(bodies, both_dresses())
    assert removed == [(0, "34 THE REBEL PASSION")]
    assert bodies == ["<p>cruelly ill-used her. His strength was whole.</p>"]


def test_the_other_dress_of_the_same_head_too(vtb):
    bodies = ["<p>THE REBEL PASSION 29<br/>whole arch as it had been.</p>"]
    removed = vtb.strip_missed_running_heads(bodies, both_dresses())
    assert removed == [(0, "THE REBEL PASSION 29")]
    assert bodies == ["<p>whole arch as it had been.</p>"]


def test_a_fused_foot_at_the_end_of_the_page_goes_as_well(vtb):
    bodies = ["<p>prose that ends the page<br/>52 THE REBEL PASSION</p>"]
    removed = vtb.strip_missed_running_heads(bodies, both_dresses())
    assert removed == [(0, "52 THE REBEL PASSION")]
    assert bodies == ["<p>prose that ends the page</p>"]


def test_a_line_break_in_ordinary_prose_is_not_a_head(vtb):
    bodies = ["<p>ordinary prose<br/>with a line break and no head at all</p>"]
    assert vtb.strip_missed_running_heads(bodies, both_dresses()) == []


def test_a_fused_line_the_book_never_prints_as_furniture_stays(vtb):
    bodies = ["<p>34 SOME OTHER TITLE<br/>and the prose beneath it.</p>"]
    assert vtb.strip_missed_running_heads(bodies, both_dresses()) == []


def test_the_fused_head_is_judged_on_its_own_text_not_the_block(vtb):
    """The blind spot, stated as a property.

    The block's whole text runs to hundreds of characters; only the segment
    before the <br/> is the head, and only that segment is measured.
    """
    prose = ("cruelly ill-used her. " * 12).strip()
    bodies = [f"<p>34 THE REBEL PASSION<br/>{prose}</p>"]
    assert len(vtb._strip_tags(bodies[0])) > vtb.RUNNING_HEAD_MAX_CHARS * 3
    removed = vtb.strip_missed_running_heads(bodies, both_dresses())
    assert removed == [(0, "34 THE REBEL PASSION")]
    assert bodies == [f"<p>{prose}</p>"]


# --------------------------------------------------------------------------
# the threshold: the book must say it more than once by accident
# --------------------------------------------------------------------------

def test_two_furniture_sightings_are_a_coincidence_and_the_line_stays(vtb):
    bodies = ["<h2>10 THE REBEL PASSION</h2><p>prose</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 2}))
    assert removed == []
    assert bodies == ["<h2>10 THE REBEL PASSION</h2><p>prose</p>"]


def test_a_book_whose_layout_pass_found_no_furniture_at_all_is_left_alone(vtb):
    bodies = ["<h2>10 THE REBEL PASSION</h2><p>prose</p>"]
    assert vtb.strip_missed_running_heads(bodies, [[], [], []]) == []
    assert bodies == ["<h2>10 THE REBEL PASSION</h2><p>prose</p>"]


# --------------------------------------------------------------------------
# what the folio condition protects: a chapter number is not a folio
# --------------------------------------------------------------------------

def test_a_bare_chapter_number_over_its_title_is_not_a_folio_and_stays(vtb):
    # 'the largest class of lookalikes on the shelf': books whose running foot
    # is a bare folio make the shape '#' recur hundreds of times, and every
    # chapter number in the book matches it.
    bodies = ["<h1>3</h1><h2>The Rise of the Third Estate</h2><p>prose</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"{0}": 40}))
    assert removed == []
    assert bodies[0].startswith("<h1>3</h1>")


def test_a_repeated_section_title_with_no_folio_in_it_stays(vtb):
    # Working the Phones prints its own chapter names as running heads; the
    # copy on the chapter's first page is the heading, not the furniture.
    bodies = ["<h2>Working the Phones</h2><p>prose</p>"]
    removed = vtb.strip_missed_running_heads(
        bodies, [["Working the Phones"]] * 6)
    assert removed == []


def test_a_year_in_a_head_is_not_a_folio(vtb):
    bodies = ["<h2>The 2000s: acceleration renewed</h2><p>prose</p>"]
    removed = vtb.strip_missed_running_heads(
        bodies, [["The 2000s: acceleration renewed"]] * 5)
    assert removed == []


# --------------------------------------------------------------------------
# what the neighbour condition protects: a chapter opening prints no head
# --------------------------------------------------------------------------

def test_a_chapter_number_above_the_chapters_name_is_an_opening_not_a_head(vtb):
    # FM 3-24: 'Chapter 9' over 'Direct Methods for Countering Insurgencies'.
    # parse_folio accepts 'Chapter 9' -- a number at the end of a line that
    # carries letters is exactly the recto convention -- so this page is
    # spared by the heading beside it and by nothing else.
    assert vtb.parse_folio("Chapter 9") == 9
    bodies = ["<h2>Chapter 9</h2><h2>Direct Methods for Countering "
              "Insurgencies</h2><p>prose</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"Chapter {0}": 8}))
    assert removed == []


def test_the_same_line_over_prose_instead_of_a_title_is_furniture(vtb):
    # The other half of that pair: identical line, no heading beside it.
    bodies = ["<h2>Chapter 9</h2><p>the insurgency had by then split.</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"Chapter {0}": 8}))
    assert removed == [(0, "Chapter 9")]


def test_a_foot_directly_under_a_heading_is_spared_too(vtb):
    bodies = ["<p>prose</p><h2>A Closing Section</h2><p>Saylor.org 872</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"Saylor.org {0}": 5}))
    assert removed == []


# --------------------------------------------------------------------------
# what it will not reach into
# --------------------------------------------------------------------------

def test_a_head_shaped_line_in_the_middle_of_the_page_is_not_touched(vtb):
    bodies = ["<p>opening prose</p><p>10 THE REBEL PASSION</p><p>closing</p>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 9}))
    assert removed == []


def test_a_page_that_opens_with_a_figure_is_left_alone(vtb):
    bodies = ['<figure><img src="fig_0001.jpg"/><figcaption>10 THE REBEL '
              'PASSION</figcaption></figure><p>prose</p>']
    removed = vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 9}))
    assert removed == []
    assert "<figure>" in bodies[0]


def test_a_long_line_is_prose_however_often_the_shape_recurs(vtb):
    long = ("10 THE REBEL PASSION and a great deal more text besides, well "
            "past anything a running head would ever carry")
    bodies = [f"<p>{long}</p><p>more</p>"]
    removed = vtb.strip_missed_running_heads(
        bodies, census({"{0} THE REBEL PASSION": 9}))
    assert removed == []


def test_a_head_alone_on_an_otherwise_empty_leaf_is_left_where_it_is(vtb):
    # FM 5-103 has such a page. Emptying it would hand it to the blank pass,
    # which deletes it -- a page and its page-list anchor spent to tidy one
    # line that was interrupting nothing.
    bodies = ["<h2>110 THE REBEL PASSION</h2>"]
    removed = vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 9}))
    assert removed == []
    assert bodies == ["<h2>110 THE REBEL PASSION</h2>"]


def test_a_head_on_a_page_that_is_otherwise_only_a_picture_still_goes(vtb):
    bodies = ['<h2>110 THE REBEL PASSION</h2><figure><img src="f.jpg"/></figure>']
    removed = vtb.strip_missed_running_heads(bodies, census({"{0} THE REBEL PASSION": 9}))
    assert removed == [(0, "110 THE REBEL PASSION")]
    assert bodies == ['<figure><img src="f.jpg"/></figure>']


def test_every_page_is_judged_on_the_whole_books_evidence_not_its_own(vtb):
    # The head leaks on page 0; the furniture that convicts it is printed on
    # pages the pass has already walked past. This is why the census is built
    # once over the finished book rather than page by page.
    bodies = ["<h2>10 THE REBEL PASSION</h2><p>alpha</p>",
              "<p>beta</p>",
              "<p>gamma</p>"]
    furn = [[], ["12 THE REBEL PASSION"], ["14 THE REBEL PASSION"]]
    furn += [["16 THE REBEL PASSION"]]
    removed = vtb.strip_missed_running_heads(bodies, furn)
    assert removed == [(0, "10 THE REBEL PASSION")]


# --------------------------------------------------------------------------
# a head with no folio in it: known by its rhythm
# --------------------------------------------------------------------------
#
# A book that numbers its pages at the foot gives its running heads nothing
# for parse_folio to read. China's Engine of Environmental Collapse set its
# recto head REFERENCES as a heading on all 36 notes pages it heads, and
# never once as furniture, so neither condition above could ever see it.


def _notes_pages(n=8):
    """The notes section as China's Engine prints it: the opening page's own
    heading, then REFERENCES atop every recto and the book's title as
    furniture atop every verso, folios at the foot."""
    notes = "<ol><li>22. Data sourced from the IMF's outlook database.</li></ol>"
    bodies, furn = [], []
    for p in range(n):
        if p == 0:
            bodies.append("<h1>References</h1><p>All hyperlinks accessed "
                          "before December 31, 2019.</p>" + notes)
            furn.append(["201"])
        elif p % 2 == 0:
            bodies.append("<h2>REFERENCES</h2>" + notes)
            furn.append([str(201 + p)])
        else:
            bodies.append(notes)
            furn.append(["CHINA'S ENGINE OF ENVIRONMENTAL COLLAPSE",
                         str(201 + p)])
    return bodies, furn


def test_a_head_with_no_folio_goes_on_its_rhythm(vtb):
    bodies, furn = _notes_pages()
    removed = vtb.strip_missed_running_heads(bodies, furn)
    assert removed == [(2, "REFERENCES"), (4, "REFERENCES"), (6, "REFERENCES")]
    assert not any(b.startswith("<h2>REFERENCES") for b in bodies)


def test_the_first_page_of_the_run_is_the_sections_own_heading(vtb):
    # An opening page prints no running head, so the run's first sighting is
    # the heading the section opens with — the one the notes linker and the
    # contents both look for.
    bodies, furn = _notes_pages()
    vtb.strip_missed_running_heads(bodies, furn)
    assert bodies[0].startswith("<h1>References</h1>")


def test_a_verso_head_the_layout_calls_a_heading_once_goes_too(vtb):
    # The same book's title, furniture on every other verso and a heading on
    # this one: the furniture sightings carry the rhythm.
    bodies, furn = _notes_pages()
    bodies[3] = ("<h2>CHINA'S ENGINE OF ENVIRONMENTAL COLLAPSE</h2>"
                 "<p>formula see the recall notice.</p>")
    furn[3] = ["204"]
    removed = vtb.strip_missed_running_heads(bodies, furn)
    assert (3, "CHINA'S ENGINE OF ENVIRONMENTAL COLLAPSE") in removed
    assert bodies[3] == "<p>formula see the recall notice.</p>"


def test_a_section_title_that_returns_once_a_chapter_stays(vtb):
    # "Key Terms" opens a page in chapter after chapter, but chapters are far
    # apart: no rhythm, and every copy is a real heading.
    bodies = ["<p>prose</p>"] * 90
    for p in (10, 40, 70):
        bodies[p] = "<h2>Key Terms</h2><p>absolute advantage</p>"
    furn = [[str(p + 1)] for p in range(90)]
    assert vtb.strip_missed_running_heads(bodies, furn) == []


def test_an_opening_and_two_more_sightings_are_still_a_coincidence(vtb):
    # The run's first page is the heading itself, so the head must be seen
    # three times besides it — the threshold every repetition here meets.
    # A cover, half-title and title page printing one title are a run of
    # three, and none of them is a running head.
    bodies, furn = _notes_pages(6)          # References, then pages 2 and 4
    assert vtb.strip_missed_running_heads(bodies, furn) == []


def test_a_callout_under_the_pages_own_running_head_stays(vtb):
    # FM 4-25 opens page after page with a NOTE callout, under the manual's
    # designation set as furniture. A page with its running head already in
    # hand has no second one: the rhythm is the manual's, not a head's.
    bodies, furn = [], []
    for p in range(8):
        bodies.append("<h2>NOTE</h2><p>Do not remove the helmet.</p>"
                      if p % 2 else "<p>Check the airway first.</p>")
        furn.append(["FM 4-25.11/NTRP 4-02.1/AFMAN 44-163(I)", f"2-{p + 1}"])
    assert vtb.strip_missed_running_heads(bodies, furn) == []


def test_a_line_repeated_at_the_foot_of_the_page_stays(vtb):
    # Burning Up closes four pages running with the same source line under
    # its charts. The rhythm is read at the head only: the foot is where a
    # page repeats its own apparatus.
    firsts = ["Coal led the mix.", "Oil followed it.", "Gas rose fastest.",
              "Hydro held steady.", "Wind barely showed.", "Solar was last."]
    bodies = [f"<p>{t}</p><p>Source: IEA Energy Balances</p>" for t in firsts]
    furn = [[str(70 + p)] for p in range(6)]
    assert vtb.strip_missed_running_heads(bodies, furn) == []
