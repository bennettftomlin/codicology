"""Chapter openings: one rank for all of them, and a contents page when the
book never printed one.

The layout pass reads each page alone, so it cannot notice that it called
chapter five an h1 and chapter six an h2 — Principles of Macroeconomics came
out seven of one and fourteen of the other for the same job. It also splits
the opening in two, the number in one heading and the title in the next,
leaving neither able to say what chapter it is.

Rank goes to the SHALLOWEST level the book used, not the commonest. That
distinction is the whole of test_rank_is_the_shallowest_not_the_commonest:
taking the mode would have pushed macroeconomics' chapters to h2, level with
the very titles they own, and the fix would have read as working.

The contents fallback is the other half. That book has no printed contents —
copyright, preface, chapter one — so the reader was handed 826 navigation
entries reading "Page 1", "Page 2", "Page 3".
"""


def test_split_openings_are_made_whole(vtb):
    bodies = ["<h1>Chapter 1</h1><h2>Economics: The Study of Choice</h2>"
              "<h3>Start Up</h3><p>Prose.</p>",
              "<h1>Chapter 2</h1><h2>Confronting Scarcity</h2><p>More.</p>"]
    n = vtb.normalize_chapter_heads(bodies)
    assert n == 2
    assert "<h1>Chapter 1. Economics: The Study of Choice</h1>" in bodies[0]
    assert "<h1>Chapter 2. Confronting Scarcity</h1>" in bodies[1]
    # the section head below it is untouched
    assert "<h3>Start Up</h3>" in bodies[0]


def test_rank_is_the_shallowest_not_the_commonest(vtb):
    """Macroeconomics in miniature: two chapters as h1, four as h2. The mode
    is h2 — and h2 is what the chapter TITLES use, so ranking chapters there
    puts a chapter level with its own subtitle. The shallowest is correct."""
    bodies = ["<h1>Chapter 1</h1><h2>First</h2>",
              "<h1>Chapter 2</h1><h2>Second</h2>",
              "<h2>Chapter 3</h2><h2>Third</h2>",
              "<h2>Chapter 4</h2><h2>Fourth</h2>",
              "<h2>Chapter 5</h2><h2>Fifth</h2>",
              "<h2>Chapter 6</h2><h2>Sixth</h2>"]
    vtb.normalize_chapter_heads(bodies)
    for k, word in enumerate(["First", "Second", "Third", "Fourth", "Fifth",
                              "Sixth"], start=1):
        assert f"<h1>Chapter {k}. {word}</h1>" in bodies[k - 1], bodies[k - 1]
    assert not any("<h2>Chapter" in b for b in bodies)


def test_a_book_that_already_got_it_right_is_left_alone(vtb):
    """TiHKAL's openings carry number and title together already."""
    bodies = ["<h2>CHAPTER 1. INVASION</h2><p>(Alice's voice)</p>",
              "<h2>CHAPTER 2. LOURDES</h2><p>One evening.</p>"]
    before = list(bodies)
    assert vtb.normalize_chapter_heads(bodies) == 0
    assert bodies == before


def test_roman_numbered_chapters_are_handled(vtb):
    bodies = ["<h2>CHAPTER I</h2><h3>INTRODUCTION</h3>",
              "<h2>CHAPTER II</h2><h3>CLOTHING</h3>",
              "<h2>CHAPTER III</h2><h3>SHELTER</h3>"]
    vtb.normalize_chapter_heads(bodies)
    assert "<h2>CHAPTER I. INTRODUCTION</h2>" in bodies[0]
    assert "<h2>CHAPTER III. SHELTER</h2>" in bodies[2]


def test_letters_that_merely_look_roman_are_refused(vtb):
    """The strict parser is what keeps "CHAPTER IIII" from becoming a number."""
    bodies = ["<h1>Chapter 1</h1><h2>Real</h2>",
              "<h1>CHAPTER IIII</h1><h2>Not a numeral</h2>",
              "<h1>Chapter 2</h1><h2>Also real</h2>"]
    vtb.normalize_chapter_heads(bodies)
    assert "<h1>CHAPTER IIII</h1>" in bodies[1], "a non-numeral was accepted"


def test_prose_is_never_folded_into_a_heading(vtb):
    long_line = "The argument of this chapter is developed at some length " \
                "and continues well past the point where a title would stop."
    bodies = [f"<h1>Chapter 1</h1><h2>{long_line}</h2>",
              "<h1>Chapter 2</h1><h2>Short Title</h2>"]
    vtb.normalize_chapter_heads(bodies)
    assert "<h1>Chapter 1</h1>" in bodies[0], "prose was folded in as a title"
    assert long_line in bodies[0], "the paragraph was destroyed"
    assert "<h1>Chapter 2. Short Title</h1>" in bodies[1]


def test_a_following_chapter_head_is_not_eaten_as_a_title(vtb):
    bodies = ["<h1>Chapter 1</h1><h1>Chapter 2</h1><p>Prose.</p>"]
    vtb.normalize_chapter_heads(bodies)
    assert "Chapter 1. Chapter 2" not in bodies[0]


def test_no_words_are_lost_when_an_opening_is_merged(vtb):
    bodies = ["<h1>Chapter 4</h1><h2>Applications of Supply and Demand</h2>"
              "<p>Body text here.</p>",
              "<h1>Chapter 5</h1><h2>Elasticity</h2><p>More body.</p>"]
    before = sorted(w for b in bodies for w in vtb._strip_tags(b).split())
    vtb.normalize_chapter_heads(bodies)
    after = sorted(w for b in bodies for w in vtb._strip_tags(b).split())
    assert before == after or \
        [w.rstrip(".") for w in after] == [w.rstrip(".") for w in before]


def test_contents_are_built_from_the_headings_when_none_was_printed(vtb):
    bodies = ["<p>Copyright.</p>", "<h2>Preface</h2><p>Words.</p>",
              "<h1>Chapter 1</h1><h2>The Study of Choice</h2>",
              "<p>prose</p>",
              "<h1>Chapter 2</h1><h2>Confronting Scarcity</h2>",
              "<p>prose</p>",
              "<h1>Chapter 3</h1><h2>Demand and Supply</h2>"]
    vtb.normalize_chapter_heads(bodies)
    toc = vtb.chapter_head_contents(bodies)
    assert [i for i, _ in toc] == [2, 4, 6]
    assert toc[0][1] == "Chapter 1. The Study of Choice"
    assert toc[2][1] == "Chapter 3. Demand and Supply"


def test_two_chapters_are_too_few_to_call_a_contents(vtb):
    bodies = ["<h1>Chapter 1</h1><h2>One</h2>", "<h1>Chapter 2</h1><h2>Two</h2>"]
    vtb.normalize_chapter_heads(bodies)
    assert vtb.chapter_head_contents(bodies) == []


def test_numbers_that_do_not_climb_are_refused(vtb):
    """A running head read as a chapter opening, or a back-reference in the
    prose, breaks the sequence — and a contents that jumps backwards is
    worse than none."""
    bodies = ["<h1>Chapter 1</h1><h2>One</h2>",
              "<h1>Chapter 7</h1><h2>Seven</h2>",
              "<h1>Chapter 2</h1><h2>Two</h2>",
              "<h1>Chapter 3</h1><h2>Three</h2>"]
    vtb.normalize_chapter_heads(bodies)
    assert vtb.chapter_head_contents(bodies) == []


def test_a_book_with_no_chapter_headings_yields_nothing(vtb):
    bodies = ["<p>Just prose.</p>", "<h2>A Section</h2><p>More prose.</p>"]
    assert vtb.normalize_chapter_heads(bodies) == 0
    assert vtb.chapter_head_contents(bodies) == []


# ── openings the layout ran into the prose ───────────────────────────────────

def _book_with_a_buried_opening():
    """Chapters 1-3 open cleanly; chapter 2's opening is buried mid-page."""
    return [
        "<h1>Chapter 1</h1><h2>First</h2><p>Prose of one.</p>",
        "<p>The last exercise of chapter one ends here. Chapter 2 The Nature "
        "of Money Start Up: A Story. Larry helped a client.</p>",
        "<p>more of two</p>",
        "<h1>Chapter 3</h1><h2>Third</h2><p>Prose of three.</p>",
    ]


def test_a_buried_opening_is_recovered_as_a_break(vtb):
    bodies = _book_with_a_buried_opening()
    vtb.normalize_chapter_heads(bodies)
    assert vtb.promote_missing_chapter_heads(bodies) == 1
    assert "<h1>Chapter 2</h1>" in bodies[1]
    # the prose either side survives, split into its own paragraphs
    assert "The last exercise of chapter one ends here." in bodies[1]
    assert "The Nature of Money" in bodies[1]


def test_the_recovered_break_completes_the_contents(vtb):
    bodies = _book_with_a_buried_opening()
    vtb.normalize_chapter_heads(bodies)
    vtb.promote_missing_chapter_heads(bodies)
    toc = vtb.chapter_head_contents(bodies)
    assert [i for i, _ in toc] == [0, 1, 3]
    assert toc[1][1] == "Chapter 2"


def test_a_cross_reference_is_never_promoted(vtb):
    """A textbook points at its own chapters constantly. "see Chapter 2" in
    the middle of a sentence must not become a chapter break."""
    bodies = [
        "<h1>Chapter 1</h1><h2>First</h2>"
        "<p>As we will see in Chapter 2 The Nature of Money is subtle.</p>",
        "<p>Filler.</p>",
        "<h1>Chapter 3</h1><h2>Third</h2>",
        "<h1>Chapter 4</h1><h2>Fourth</h2>",
    ]
    vtb.normalize_chapter_heads(bodies)
    assert vtb.promote_missing_chapter_heads(bodies) == 0
    assert "<h1>Chapter 2</h1>" not in bodies[0]


def test_a_number_mentioned_twice_in_the_gap_is_left_alone(vtb):
    """Two candidates and no way to choose: refuse rather than guess."""
    bodies = [
        "<h1>Chapter 1</h1><h2>First</h2>",
        "<p>End of a sentence. Chapter 2 The Nature of Money follows. "
        "Another sentence ends. Chapter 2 Reprised Again here.</p>",
        "<h1>Chapter 3</h1><h2>Third</h2>",
        "<h1>Chapter 4</h1><h2>Fourth</h2>",
    ]
    vtb.normalize_chapter_heads(bodies)
    assert vtb.promote_missing_chapter_heads(bodies) == 0


def test_only_a_hole_in_the_sequence_is_filled(vtb):
    """A chapter that already has its heading is never promoted a second
    time, however often its number appears in the prose."""
    bodies = [
        "<h1>Chapter 1</h1><h2>First</h2>",
        "<p>A sentence ends. Chapter 2 Mentioned In Passing here.</p>",
        "<h1>Chapter 2</h1><h2>Second</h2>",
        "<h1>Chapter 3</h1><h2>Third</h2>",
    ]
    vtb.normalize_chapter_heads(bodies)
    assert vtb.promote_missing_chapter_heads(bodies) == 0
    assert bodies[1].count("<h1>") == 0


def test_a_mention_outside_the_gap_pages_is_ignored(vtb):
    """The opening must lie between the chapters either side of it."""
    bodies = [
        "<h1>Chapter 1</h1><h2>First</h2>",
        "<h1>Chapter 3</h1><h2>Third</h2>",
        "<h1>Chapter 4</h1><h2>Fourth</h2>",
        "<p>Long after the fact. Chapter 2 The Nature of Money was covered.</p>",
    ]
    vtb.normalize_chapter_heads(bodies)
    assert vtb.promote_missing_chapter_heads(bodies) == 0


def test_no_words_are_lost_when_a_break_is_recovered(vtb):
    """Measured across the promotion alone — the merge before it deliberately
    adds the separator, and that is tested where it belongs."""
    bodies = _book_with_a_buried_opening()
    vtb.normalize_chapter_heads(bodies)
    before = sorted(w for b in bodies for w in vtb._strip_tags(b).split())
    vtb.promote_missing_chapter_heads(bodies)
    after = sorted(w for b in bodies for w in vtb._strip_tags(b).split())
    assert before == after
