"""A running head that is the book's own title, labelled a heading.

Paul Robeson prints its title as the verso head of its notes and index, and the
layout called it a heading on fifteen of those pages; the reading order set it
inside the page's text, where the rules that look at a page's first line could
not see it.
"""
from codicology.furniture import strip_title_heads
from codicology.model import Block, Page


def _page(i, blocks, furniture=()):
    return Page(index=i, path=f"p{i}", blocks=list(blocks), furniture=list(furniture))


def _head(text, y=(0.04, 0.059), label="SectionHeader"):
    # as wide as Paul Robeson's verso head is set: 0.144 of the page
    return Block("heading", f"<h2>{text}</h2>", label=label, box=(0.41, y[0], 0.554, y[1]))


def _furn(text):
    return Block("furniture", f"<p>{text}</p>", label="PageHeader", box=(0.3, 0.04, 0.7, 0.06))


def _text(text):
    return Block("text", f"<p>{text}</p>", label="Text", box=(0.1, 0.1, 0.9, 0.9))


def _book(page3):
    pages = [_page(0, [_text("Prose.")], [_furn("Paul Robeson"), _furn("208")]),
             _page(1, [_text("Prose.")], [_furn("Chapter 3")]),
             _page(2, [_text("Prose.")], [_furn("Paul Robeson")]),
             _page(3, page3)]
    return pages, [p.render() for p in pages]


def test_the_books_title_at_the_top_of_a_page_is_its_running_head():
    pages, bodies = _book([_text("12. Seton, Paul Robeson, 96."), _head("Paul Robeson"),
                           _text("13. Ibid.")])
    assert strip_title_heads(pages, bodies, "Paul Robeson: A Watched Man") == [(3, "Paul Robeson")]
    assert "<h2>" not in bodies[3] and "13. Ibid." in bodies[3]


def test_a_chapters_title_at_the_top_of_its_opening_page_stays():
    pages, bodies = _book([_head("Chapter 3"), _text("It began in Princeton.")])
    assert strip_title_heads(pages, bodies, "Paul Robeson") == []


def test_the_title_lower_on_the_page_is_not_a_head():
    pages, bodies = _book([_text("Prose."), _head("Paul Robeson", y=(0.3, 0.33))])
    assert strip_title_heads(pages, bodies, "Paul Robeson") == []


def test_the_title_the_book_prints_as_furniture_only_once_stays():
    pages, bodies = _book([_head("Paul Robeson"), _text("Prose.")])
    pages[2].furniture = []
    assert strip_title_heads(pages, bodies, "Paul Robeson") == []


def test_a_page_that_is_only_the_title_keeps_it():
    pages, bodies = _book([_head("Paul Robeson")])
    assert strip_title_heads(pages, bodies, "Paul Robeson") == []


def test_an_edition_line_after_the_title_is_the_same_head():
    """Where There Is No Doctor's head reads "Where There Is No Doctor 2010"."""
    pages = [_page(0, [_text("a")], [_furn("Where There Is No Doctor 2010")]),
             _page(1, [_text("b")], [_furn("Where There Is No Doctor 2010")]),
             _page(2, [_head("Where There Is No Doctor 2010", y=(0.02, 0.04)), _text("Dear Village Health Worker")])]
    bodies = [p.render() for p in pages]
    assert strip_title_heads(pages, bodies, "Where there is no doctor") == [(2, "Where There Is No Doctor 2010")]


def test_a_title_page_keeps_its_title():
    """Reconstruction: The Battle for Democracy, its title page: the title in
    large type at the top, the subtitle and a picture under it."""
    from codicology.furniture import strip_title_heads
    from codicology.model import Block, Page
    title = Page(0, "p0", [Block("heading", "<h1>RECONSTRUCTION</h1>", "SectionHeader",
                                 box=(0.102, 0.054, 0.608, 0.08)),
                           Block("text", "<p>THE BATTLE FOR<br/>DEMOCRACY</p>", "Text",
                                 box=(0.1, 0.1, 0.5, 0.15))])
    others = [Page(i, f"p{i}", [Block("text", "<p>Prose.</p>", "Text", box=(0.1, 0.1, 0.9, 0.9))],
                   furniture=[Block("text", "<p>Reconstruction</p>", "PageHeader")]) for i in (1, 2, 3)]
    pages = [title] + others
    bodies = [p.render() for p in pages]
    assert strip_title_heads(pages, bodies, "Reconstruction: The Battle for Democracy") == []
    assert "RECONSTRUCTION" in bodies[0]
