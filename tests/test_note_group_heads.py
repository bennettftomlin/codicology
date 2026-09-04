"""A notes section's group heads, however the book dresses them — and the
guard that stops a half-recognised list linking everything to the wrong
chapter.

Work, Sex & Power sets all eighteen of its heads as <h2><i>Chapter One</i></h2>.
The inline <i> is invisible to a reader and fatal to a pattern that does not
expect it, so not one head was recognised and all 359 markers went unlinked.

The trap sits underneath. Teaching the pattern about <i> alone would find
seventeen "Chapter" heads and miss the eighteenth, "Introduction" — and
groups pair by POSITION, so the body's introduction would pair with the
notes for chapter one and every group after it would shift by one. Every
marker still finds a note, because every chapter numbers from 1, so the run
reports a full set of links and nothing downstream notices. That failure is
worse than the one it replaces, and the last tests here are what make it
impossible.
"""


def notes_page(head_html):
    return ("<h1>NOTES</h1>" + head_html
            + "<ol><li>1. First source.</li><li>2. Second source.</li>"
              "<li>3. Third source.</li></ol>")


def test_a_head_wrapped_in_italics_is_recognised(vtb):
    bodies = ["<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
              notes_page("<h2>\n  <i>Chapter One</i>\n</h2>")]
    start, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1 and len(groups[0]) == 3


def test_a_bare_head_still_works(vtb):
    bodies = ["<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
              notes_page("<h2>CHAPTER ONE</h2>")]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1


def test_front_matter_sections_open_groups_too(vtb):
    for name in ("Introduction", "Preface", "Prologue", "Foreword",
                 "Conclusion", "Epilogue", "Afterword"):
        bodies = ["<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
                  notes_page(f"<h2><i>{name}</i></h2>")]
        _, groups = vtb.parse_notes_section(bodies)
        assert len(groups) == 1, f"{name} did not open a group"


def test_the_notes_head_itself_is_not_a_group(vtb):
    bodies = ["<p>One.<sup>1</sup></p>",
              "<h1>Notes</h1><ol><li>1. a</li><li>2. b</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert groups == []


def test_an_index_heading_does_not_become_a_group(vtb):
    """It opens no notes, and an empty group would shift the pairing."""
    bodies = ["<p>One.<sup>1</sup> Two.<sup>2</sup> Three.<sup>3</sup></p>",
              notes_page("<h2><i>Chapter One</i></h2>")
              + "<h1>Index</h1><p>Aboriginal peoples, 45, 67</p>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1


def test_a_heading_that_merely_opens_with_the_word_is_not_a_head(vtb):
    """"Chapter One was drafted twice" is prose about a chapter, and must not
    open a group. The section may still parse as an ungrouped one — three
    notes numbered 1, 2, 3 under no head is exactly that — so what is asserted
    here is the head, not the outcome: no group is opened BY this heading.
    The companion below is the case where that distinction bites."""
    frag = notes_page("<h2>Chapter One was drafted twice</h2>")
    assert vtb.GROUP_HEAD.search(frag) is None


def test_an_unreadable_head_over_restarting_numbers_still_refuses(vtb):
    """The distinction that matters: a grouped book whose heads we cannot
    read must not be flattened into one sequence. Its numbers restart, and
    that is what refuses — not the heading."""
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup> C.<sup>3</sup></p>",
              "<h1>NOTES</h1><h2>Chapter One was drafted twice</h2>"
              "<ol><li>1. a</li><li>2. b</li><li>3. c</li></ol>"
              "<h2>Chapter Two was also drafted twice</h2>"
              "<ol><li>1. d</li><li>2. e</li><li>3. f</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert groups == []
    assert vtb.link_notes(bodies, dropped=set())["linked"] == 0


# ── the alignment guard ──────────────────────────────────────────────────────

def group(page, count):
    return [(page, n, n * 10) for n in range(1, count + 1)]


def test_matching_pairs_agree(vtb):
    body = [group(0, 4), group(1, 25), group(2, 13), group(3, 21)]
    notes = [group(9, 4), group(9, 25), group(9, 13), group(9, 21)]
    assert vtb._paired_groups_agree(body, notes) is True


def test_a_list_shifted_by_one_is_caught(vtb):
    """The exact shape a missed Introduction produces."""
    body = [group(0, 4), group(1, 25), group(2, 13), group(3, 21)]
    notes = [group(9, 25), group(9, 13), group(9, 21), group(9, 25)]
    assert vtb._paired_groups_agree(body, notes) is False


def test_one_chapter_disagreeing_does_not_condemn_the_book(vtb):
    """A note the recogniser missed, or a marker read as a footnote, moves
    one pair. A shift moves nearly all of them."""
    body = [group(0, 20), group(1, 25), group(2, 13), group(3, 21)]
    notes = [group(9, 20), group(9, 25), group(9, 4), group(9, 21)]
    assert vtb._paired_groups_agree(body, notes) is True


def test_too_few_groups_to_judge_defers(vtb):
    """Across two groups an honest disagreement and a shift look the
    same, so the rule keeps quiet — unless the counts already failed to
    match, which is when the pairing was a guess to begin with."""
    body = [group(0, 12), group(1, 30)]
    notes = [group(9, 30), group(9, 12)]
    assert vtb._paired_groups_agree(body, notes) is True
    assert vtb._paired_groups_agree(body, notes, min_judged=2) is False


def test_a_shifted_book_links_nothing_rather_than_everything_wrongly(vtb):
    """End to end: three chapters of prose, and a notes section whose first
    group head the parser cannot see. Refusing is the only right answer."""
    bodies = [
        "<p>Intro.<sup>1</sup> More.<sup>2</sup> Again.<sup>3</sup></p>",
        "<p>Ch1.<sup>1</sup> b.<sup>2</sup> c.<sup>3</sup> d.<sup>4</sup> "
        "e.<sup>5</sup> f.<sup>6</sup> g.<sup>7</sup> h.<sup>8</sup></p>",
        "<p>Ch2.<sup>1</sup> b.<sup>2</sup> c.<sup>3</sup></p>",
        "<h1>NOTES</h1><h2>An Unrecognisable Head</h2>"
        "<ol><li>1. i</li><li>2. ii</li><li>3. iii</li></ol>"
        "<h2>Chapter One</h2>"
        "<ol><li>1. a</li><li>2. b</li><li>3. c</li><li>4. d</li>"
        "<li>5. e</li><li>6. f</li><li>7. g</li><li>8. h</li></ol>"
        "<h2>Chapter Two</h2><ol><li>1. x</li><li>2. y</li><li>3. z</li></ol>",
    ]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 0, "markers were bound across a shifted pairing"
    assert stats["misaligned"] is True


# ── books that do not group their notes at all ───────────────────────────────

def ungrouped(head="<h1>Notes</h1>", n=6, tail=""):
    items = "".join(f"<li>{k}. Source {k}.</li>" for k in range(1, n + 1))
    return head + "<ol>" + items + "</ol>" + tail


def test_a_single_continuous_sequence_binds(vtb):
    """A Critical History of Poverty Finance cites author-date in the prose
    and keeps numbered notes for archival sources, so twenty-one notes run
    straight through under one heading with no chapter heads at all."""
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup> C.<sup>3</sup></p>",
              "<p>D.<sup>4</sup> E.<sup>5</sup> F.<sup>6</sup></p>",
              ungrouped()]
    start, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1 and len(groups[0]) == 6
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 6 and stats["unlinked"] == 0


def test_a_note_cited_twice_still_binds_once_backwards(vtb):
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup> C.<sup>3</sup> "
              "again.<sup>2</sup> and.<sup>4</sup></p>",
              ungrouped(n=4)]
    stats = vtb.link_notes(bodies, dropped=set())
    assert stats["linked"] == 5
    assert bodies[0].count('epub:type="noteref"') == 5
    assert bodies[1].count("#ref-") == 4, "one backlink per note, not per marker"


def test_numbers_that_restart_without_a_heading_still_refuse(vtb):
    """Two chapters' notes run together with nothing to say where one ends.
    Which "1" a marker means is unknowable, so nothing binds."""
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup> C.<sup>3</sup></p>",
              "<h1>Notes</h1><ol><li>1. a</li><li>2. b</li><li>3. c</li>"
              "<li>1. d</li><li>2. e</li><li>3. f</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert groups == []
    assert vtb.link_notes(bodies, dropped=set())["linked"] == 0


def test_the_bibliography_after_it_is_not_swallowed(vtb):
    """A numbered bibliography beyond the section boundary belongs to
    somebody else; taking it would inflate the group and mis-bind."""
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup> C.<sup>3</sup></p>",
              ungrouped(n=3),
              "<h1>Bibliography</h1><ol><li>4. Adams, A. Some Book.</li>"
              "<li>5. Brown, B. Another Book.</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1
    assert [n for _, n, _ in groups[0]] == [1, 2, 3]


def test_a_grouped_book_never_reaches_the_fallback(vtb):
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup> C.<sup>3</sup></p>",
              "<p>D.<sup>1</sup> E.<sup>2</sup> F.<sup>3</sup></p>",
              "<h1>NOTES</h1><h2>Chapter One</h2>"
              "<ol><li>1. a</li><li>2. b</li><li>3. c</li></ol>"
              "<h2>Chapter Two</h2>"
              "<ol><li>1. d</li><li>2. e</li><li>3. f</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 2, "the grouped path was bypassed"


def test_too_few_entries_to_be_a_notes_section(vtb):
    bodies = ["<p>A.<sup>1</sup></p>",
              "<h1>Notes</h1><ol><li>1. Only one.</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert groups == []


def test_a_group_head_may_number_without_a_period(vtb):
    """Beyond Money heads its note groups "1 capital and crises" where other
    books write "1. Capital and crises". Requiring the dot cost that book
    every one of its 410 links: no head parsed, so no group ever opened and
    all its entries were discarded before pairing."""
    bodies = ["<p>Prose with a marker.<sup>1</sup> And another.<sup>2</sup></p>",
              "<h1>Notes</h1><h2>1 capital and crises</h2>"
              "<ol><li>1. Schor and Jorgenson, 2019.</li>"
              "<li>2. Piketty, 2014.</li></ol>"]
    start, groups = vtb.parse_notes_section(bodies)
    assert start == 1
    assert [len(g) for g in groups] == [2]


def test_the_numbered_head_still_parses_with_its_period(vtb):
    bodies = ["<p>A marker.<sup>1</sup> Another.<sup>2</sup></p>",
              "<h1>Notes</h1><h2>1. Capital and crises</h2>"
              "<p><sup>1</sup> First.</p><p><sup>2</sup> Second.</p>"]
    start, groups = vtb.parse_notes_section(bodies)
    assert [len(g) for g in groups] == [2]


# ── the boundary the section's own numbering declares ────────────────────────
#
# Gendzier's Development Against Democracy heads all eight of its note groups
# and four of them are invisible to the patterns above: chapters one, two and
# three wrap the number in <i>, and chapter four's head is a centred paragraph
# carrying a <br/>. Eight groups parsed as four, four is too far from the
# body's eight to pair, and all 705 markers went unlinked. Teaching the
# numbered pattern about <i> alone is the trap this file already warns of: it
# finds three of the four, leaves chapter four merged into chapter three, and
# shifts every pairing after it. The numbering itself is the evidence that
# does not depend on how a head is dressed.


def _run(start, count):
    return "".join(f"<li>{n}. Source {n}.</li>"
                   for n in range(start, start + count))


def test_a_return_to_one_opens_a_group(vtb):
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup>c<sup>3</sup></p>",
              "<p>d<sup>1</sup>e<sup>2</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2><ol>" + _run(1, 3)
              + '</ol><p style="text-align:center"><i>2. Dressed As Prose</i></p>'
                "<ol>" + _run(1, 2) + "</ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [len(g) for g in groups] == [3, 2]
    assert [n for _, n, _ in groups[1]] == [1, 2]


def test_the_head_the_patterns_miss_needs_no_head_at_all(vtb):
    """The same section with the second head removed entirely parses the
    same way: the boundary is the numbering, not the markup."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup>c<sup>3</sup></p>",
              "<p>d<sup>1</sup>e<sup>2</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2><ol>" + _run(1, 3)
              + "</ol><ol>" + _run(1, 2) + "</ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [len(g) for g in groups] == [3, 2]


def test_a_repeated_number_is_damage_not_a_boundary(vtb):
    """Splitting on any number that fails to climb would invent a group out
    of one misread entry and shift every pairing after it. Only a return to
    1 is a restart."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup>c<sup>3</sup>d<sup>4</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2>"
              "<ol><li>1. One.</li><li>2. Two.</li>"
              "<li>2. Two again.</li><li>4. Four.</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1 and len(groups[0]) == 4


def test_a_group_whose_own_notes_open_one_one_stays_whole(vtb):
    """The numbering must climb above 1 before a return to it means
    anything; two 1s at the head of a group are a damaged entry."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2>"
              "<ol><li>1. One.</li><li>1. One again.</li>"
              "<li>2. Two.</li></ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert len(groups) == 1 and len(groups[0]) == 3


def test_a_restart_at_a_head_does_not_open_two_groups(vtb):
    """The ordinary case — a recognised head followed by note 1 — must open
    exactly one group, or every well-formed book gains an empty group and
    the pairing shifts."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup></p>", "<p>c<sup>1</sup></p>",
              "<h1>NOTES</h1><h2>CHAPTER ONE</h2><ol>" + _run(1, 2)
              + "</ol><h2>CHAPTER TWO</h2><ol>" + _run(1, 1) + "</ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [len(g) for g in groups] == [2, 1]


def test_restart_split_groups_link_to_their_own_chapter(vtb):
    """The whole point: chapter two's marker must reach chapter two's note,
    not the note of the chapter its head was merged into."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup>c<sup>3</sup></p>",
              "<p>d<sup>1</sup>e<sup>2</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2><ol>" + _run(1, 3)
              + "</ol><ol>" + _run(1, 2) + "</ol>"]
    stats = vtb.link_notes(bodies, set())
    assert stats["groups"] == 2 and not stats["misaligned"]
    assert stats["linked"] == 5
    assert 'href="page_0002.xhtml#note-g1-1"' in bodies[1]
    assert 'href="page_0002.xhtml#note-g0-1"' in bodies[0]


# ── a run of notes divided by <br/> rather than by blocks ────────────────────
#
# One book sets the tail of two chapters' notes as a single paragraph, the
# entries separated by <br/>. Every entry pattern anchors the number to a
# block's opening, so only the run's first line was seen: each group's highest
# note stopped there and the 34 markers past it had nothing to point at.


def _br_run(first, count):
    return ("<p>" + f"{first}. Ibid., p. {first}."
            + "".join(f"<br/>\n  {n}. Ibid., p. {n}."
                      for n in range(first + 1, first + count))
            + "</p>")


def test_a_br_divided_run_yields_every_entry(vtb):
    bodies = ["<p>" + "".join(f"x<sup>{n}</sup>" for n in range(1, 6)) + "</p>",
              "<h1>NOTES</h1><h2>Introduction</h2><ol>" + _run(1, 2) + "</ol>"
              + _br_run(3, 3)]
    _, groups = vtb.parse_notes_section(bodies)
    assert [n for _, n, _ in groups[0]] == [1, 2, 3, 4, 5]


def test_a_br_run_links_and_anchors_each_entry(vtb):
    bodies = ["<p>" + "".join(f"x<sup>{n}</sup>" for n in range(1, 6)) + "</p>",
              "<h1>NOTES</h1><h2>Introduction</h2><ol>" + _run(1, 2) + "</ol>"
              + _br_run(3, 3)]
    stats = vtb.link_notes(bodies, set())
    assert stats["linked"] == 5 and stats["unlinked"] == 0
    # the run's first entry opens the paragraph, so it anchors on the block
    # as any other paragraph entry does; only the <br/> lines need the
    # number itself to carry the id
    assert '<p id="note-g0-3"><a href="page_0000.xhtml#ref-g0-3">3.</a>' in bodies[1]
    for n in (4, 5):
        assert f'<a id="note-g0-{n}" href="page_0000.xhtml#ref-g0-{n}">{n}.</a>' \
            in bodies[1]
    for n in (3, 4, 5):
        assert f'href="page_0001.xhtml#note-g0-{n}"' in bodies[0]
    # the break and its space are the run's typesetting: without them the
    # anchored notes run together on one line
    assert "<br/>\n  <a id=\"note-g0-4\"" in bodies[1]


def test_numbers_that_do_not_climb_are_not_a_run(vtb):
    """A note may carry a <br/> before a numeral of its own — a date, an
    address, a line of verse. Only an ascending sequence is a run."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup>c<sup>3</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2>"
              "<ol><li>1. One.</li><li>2. Two.</li></ol>"
              "<p>3. Written at:<br/>14. Rue de la Paix<br/>2. Floor</p>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [n for _, n, _ in groups[0]] == [1, 2, 3]


def test_a_run_is_taken_whole_or_not_at_all(vtb):
    """Half a run ends its group early, which is the failure this repairs."""
    bodies = ["<p>a<sup>1</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2>"
              "<p>1. One.<br/>2. Two.<br/>2. Two again.<br/>4. Four.</p>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [n for _, n, _ in groups[0]] == [1]


def test_a_br_run_inside_running_prose_is_never_examined(vtb):
    """The paragraph must already be a note entry, so prose that happens to
    break before a numeral is not a run."""
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2>"
              "<ol><li>1. One.</li><li>2. Two.</li></ol>"
              "<p>The vote stood as follows:<br/>3. For<br/>4. Against</p>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [n for _, n, _ in groups[0]] == [1, 2]


def test_a_br_run_at_a_chapter_tail_keeps_its_own_group(vtb):
    """The shape that lost the links: a chapter's notes end in a <br/> run,
    the next chapter's head follows in the same page, and its notes restart."""
    bodies = ["<p>" + "".join(f"x<sup>{n}</sup>" for n in range(1, 5)) + "</p>",
              "<p>y<sup>1</sup>y<sup>2</sup></p>",
              "<h1>NOTES</h1><h2>Introduction</h2><ol>" + _run(1, 1) + "</ol>"
              + _br_run(2, 3)
              + "<h2>CHAPTER ONE</h2><ol>" + _run(1, 2) + "</ol>"]
    _, groups = vtb.parse_notes_section(bodies)
    assert [[n for _, n, _ in g] for g in groups] == [[1, 2, 3, 4], [1, 2]]
    stats = vtb.link_notes(bodies, set())
    assert stats["linked"] == 6 and not stats["misaligned"]
