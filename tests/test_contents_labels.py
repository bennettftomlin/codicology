"""What the layout labelled a heading or a printed contents, read from each
page's blocks (contents.labelled_heads, contents.labelled_contents_pages):
what the contents falls back on where no printed contents parses."""
from codicology.contents import labelled_contents_pages, labelled_heads
from codicology.model import Block, Page


def _pages():
    return [
        Page(0, "a", [Block("toc", "<p>Contents ... 1</p>", label="TableOfContents")]),
        Page(1, "b", [Block("heading", "<h1>Chapter One</h1>", label="SectionHeader"),
                      Block("text", "<p>It began.</p>", label="Text"),
                      Block("heading", "<h3>  The   Firm </h3>", label="SectionHeader"),
                      Block("heading", "<h2>12</h2>", label="SectionHeader")]),
        Page(2, "c", [Block("heading", "<p>A heading set as a paragraph</p>",
                            label="SectionHeader"),
                      Block("heading", "<h2>" + "Long " * 30 + "</h2>", label="SectionHeader")]),
    ]


def test_the_layout_s_headings_with_their_rank_and_page():
    heads = labelled_heads(_pages())
    assert heads[:2] == [(1, 1, "Chapter One"), (1, 3, "The Firm")]
    # a bare number is a note marker the layout took for a heading
    assert all(t != "12" for _, _, t in heads)
    # no <hN>: rank 3; the text is cut at ninety characters
    assert heads[2] == (2, 3, "A heading set as a paragraph")
    assert len(heads[3][2]) == 90 and heads[3][1] == 2


def test_the_pages_labelled_a_contents():
    assert labelled_contents_pages(_pages()) == {0}


def test_how_much_of_a_page_s_foot_the_layout_called_footnotes():
    from codicology.notes import footnote_share
    pages = [
        Page(0, "a", [Block("text", "<p>one two three</p>", label="Text")]),
        Page(1, "b", [Block("text", "<p>body words here</p>", label="Text"),
                      Block("rule", "<hr/>"),
                      Block("footnote", "<p>1 note one two</p>", label="Footnote"),
                      Block("text", "<p>stray four</p>", label="Text"),
                      Block("footnote", "<p>2 note</p>", label="Footnote")]),
    ]
    assert footnote_share(pages) == [0.0, 6 / 8]
