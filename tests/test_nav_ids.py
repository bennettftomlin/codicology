"""The nav the book ships is valid and forward-marching.

nav_from_placed was extracted from build_epub precisely so these
invariants — found broken by the 2026-08-18 review — stay pinned:
every id unique however entries nest or collide on a target (C1/C2),
the title hunt never reaching backward past anything already placed
(C3), and groupings degrading to bare headings rather than vanishing.
"""
from codicology import pipeline as pl


def _link(href, title, uid):
    return ("link", href, title, uid)


def _section(title, href):
    return ("section", title, href)


def _entry(title, folio, depth=2):
    return pl.TocEntry(title, folio, str(folio or ""), depth)


def _run(placed, bodies=None, toc_pages=frozenset()):
    bodies = bodies or ["<p>x</p>"] * 40
    kept = list(range(len(bodies)))
    pos_of = {i: i for i in kept}
    return pl.nav_from_placed(placed, pos_of, kept, bodies, set(toc_pages),
                              make_link=_link, make_section=_section)


def _ids(tree):
    out = []
    for node in tree:
        if isinstance(node, tuple) and node and node[0] == "link":
            out.append(node[3])
        elif isinstance(node, tuple) and len(node) == 2:
            head, kids = node
            out.extend(_ids(kids))
    return out


def test_two_entries_in_one_section_sharing_a_page_get_distinct_ids(vtb):
    """C2: two short chapters under one BOOK open on the same physical
    page; the frozen len(links) once gave them identical ids."""
    placed = [
        (_entry("BOOK ONE", None, depth=0), None, False),
        (_entry("I ALPHA", 5), 5, True),
        (_entry("II BETA", 5), 5, True),
        (_entry("BOOK TWO", None, depth=0), None, False),
        (_entry("III GAMMA", 9), 9, True),
        (_entry("IV DELTA", 9), 9, True),
    ]
    links, verified, missed, _ = _run(placed)
    ids = _ids(links)
    assert len(ids) == 4 and len(set(ids)) == 4, f"ids collided: {ids}"
    assert missed == 0


def test_the_hunt_never_reaches_backward_past_a_folio_placement(vtb):
    """C3: an entry with no folio, listed AFTER folio-resolved chapters,
    must be hunted forward of them — its title also appearing in an
    earlier chapter's prose must not win."""
    bodies = ["<p>x</p>"] * 40
    bodies[3] = "<p>The Reckoning was discussed in passing here.</p>"
    bodies[20] = "<p>THE RECKONING</p><p>The chapter opens.</p>"
    placed = [
        (_entry("I OPENING", 2), 2, True),
        (_entry("II MIDDLE", 15), 15, True),
        (_entry("The Reckoning", None), None, False),
    ]
    links, verified, missed, _ = _run(placed, bodies=bodies)
    hunted = [n for n in links if n[0] == "link" and "Reckoning" in n[2]]
    assert hunted and hunted[0][1] == "page_0020.xhtml", \
        f"hunted backward: {hunted}"


def test_an_unresolvable_grouping_stays_a_bare_heading(vtb):
    placed = [
        (_entry("BOOK NOWHERE", None, depth=0), None, False),
        (_entry("I ALPHA", 5), 5, True),
    ]
    links, verified, missed, _ = _run(placed)
    head, kids = links[0]
    assert head == ("section", "BOOK NOWHERE", None)
    assert [k[3] for k in kids] and missed == 0


def test_a_childless_grouping_ships_as_a_link_not_an_empty_list(vtb):
    """ebooklib writes <ol/> for a section with no children, and an <ol>
    with no <li> is invalid in a nav document — five installed books were
    shipping one. A grouping that ends up childless is the plain link it
    always was."""
    placed = [(_entry("Chapter One", 10, 1), 10, True),
              (_entry("Chapter Two", 20, 1), 20, True),
              (_entry("A Section", 21, 2), 21, True)]
    links, _, _, _ = pl.nav_from_placed(
        placed, {10: 0, 20: 1, 21: 2}, [10, 20, 21], ["", "", ""], set(),
        make_link=_link, make_section=_section)

    def grouping(x):
        return isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], list)

    def walk(seq):
        for item in seq:
            if grouping(item):
                assert item[1], f"empty grouping shipped: {item[0]}"
                walk(item[1])
    walk(links)
    assert any(not grouping(x) and x[2] == "Chapter One" for x in links), \
        "the childless chapter should ship as a link"
    assert any(grouping(x) and x[0][1] == "Chapter Two" for x in links), \
        "the chapter that has a section keeps its grouping"
