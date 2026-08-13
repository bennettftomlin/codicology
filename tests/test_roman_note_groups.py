"""A book that numbers its chapters in Roman must have its endnotes bind.

Schwartz's Chinese Communism heads each endnote group "I.", "II.", "III." and
carried 697 markers. Not one linked: the parser knew "CHAPTER TWO" and knew a
heading opening "2.", but a heading opening "II." matched nothing, so no group
ever opened and the whole apparatus fell through as unlinkable.

The risk the fix has to carry is the paragraph form. Twelve of that book's
thirteen heads are headings and one is an ordinary paragraph, so paragraphs
must parse — but a note continuing across a page break also opens a paragraph,
and in this literature notes open "I. V. Stalin, Works…". Binding that as a
chapter head would silently rescope every note after it, which is the one
failure mode worth more than the links themselves.
"""


def test_roman_numeral_value_is_read_strictly(vtb):
    assert vtb._roman_to_int("I") == 1
    assert vtb._roman_to_int("IV") == 4
    assert vtb._roman_to_int("IX") == 9
    assert vtb._roman_to_int("XIII") == 13
    assert vtb._roman_to_int("XL") == 40
    # things that merely use the letters are refused, not guessed at: a
    # misread group number rescopes a whole chapter's notes
    assert vtb._roman_to_int("IIII") is None
    assert vtb._roman_to_int("VX") is None
    assert vtb._roman_to_int("IL") is None
    assert vtb._roman_to_int("") is None
    assert vtb._roman_to_int("CHINA") is None


def roman_book():
    """Two chapters headed in Roman, notes grouped the same way."""
    return [
        "<p>Chapter one prose.<sup>1</sup> More.<sup>2</sup> Again.<sup>3</sup></p>",
        "<p>Second chapter begins.<sup>1</sup> And continues.<sup>2</sup> "
        "Then more.<sup>3</sup></p>",
        "<h1>NOTES</h1><h3>I. <i>The Origins of Marxism-Leninism</i></h3>"
        "<p>1. First source.</p><p>2. Second source.</p><p>3. Third source.</p>",
        "<h3>II. The Founding of the Party</h3>"
        "<p>1. Fourth source.</p><p>2. Fifth source.</p><p>3. Sixth source.</p>",
    ]


def test_roman_headed_groups_link(vtb):
    bodies = roman_book()
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["groups"] == 2
    assert stats["linked"] == 6 and stats["unlinked"] == 0
    # chapter two's marker 1 must reach chapter two's note, not chapter one's
    assert 'href="page_0003.xhtml#note-g1-1"' in bodies[1]
    assert 'href="page_0002.xhtml#note-g0-1"' in bodies[0]


def test_a_head_set_as_a_paragraph_still_opens_its_group(vtb):
    """Twelve heads as headings and one as a paragraph is what the book did."""
    bodies = roman_book()
    bodies[3] = bodies[3].replace("<h3>II. The Founding of the Party</h3>",
                                  "<p>II. The Founding of the Party</p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["groups"] == 2
    assert stats["linked"] == 6
    assert 'href="page_0003.xhtml#note-g1-1"' in bodies[1]


def test_an_authors_initials_do_not_open_a_group(vtb):
    """"I. V. Stalin, Works" opens a note, not a chapter."""
    bodies = roman_book()
    bodies[2] = bodies[2].replace(
        "<p>2. Second source.</p>",
        "<p>2. Second source.</p><p>I. V. Stalin, <i>Works</i>, vol. VI.</p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["groups"] == 2, "an initial was mistaken for a chapter head"
    assert stats["linked"] == 6


def test_a_long_paragraph_opening_in_roman_is_not_a_head(vtb):
    """A head is short. A note running on is not, however it begins."""
    bodies = roman_book()
    bodies[2] += ("<p>V. the discussion of this point in the sources cited "
                  "above, where the argument is developed at considerably "
                  "greater length than can be reproduced here in a note.</p>")
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["groups"] == 2
    assert stats["linked"] == 6


def test_arabic_headed_books_are_untouched(vtb):
    """The existing renderings must parse exactly as they did before."""
    bodies = [
        "<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
        "<p>Next.<sup>1</sup> More.<sup>2</sup> Again.<sup>3</sup></p>",
        "<h1>NOTES</h1><h2>1. Beginnings</h2>"
        "<p>1. a</p><p>2. b</p><p>3. c</p>",
        "<h2>2. Consequences</h2><p>1. d</p><p>2. e</p><p>3. f</p>",
    ]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["groups"] == 2 and stats["linked"] == 6


def test_chapter_word_headed_books_are_untouched(vtb):
    bodies = [
        "<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
        "<p>Next.<sup>1</sup> More.<sup>2</sup> Again.<sup>3</sup></p>",
        "<h1>NOTES</h1><h2>CHAPTER ONE</h2>"
        "<p><sup>1</sup>a</p><p><sup>2</sup>b</p><p><sup>3</sup>c</p>",
        "<p>CHAPTER TWO</p>"
        "<p><sup>1</sup>d</p><p><sup>2</sup>e</p><p><sup>3</sup>f</p>",
    ]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["groups"] == 2 and stats["linked"] == 6
