"""What the property tests of model.sync_page found, shrunk to named cases.

Each is marked xfail (strict) until the carrying is mended: when one starts
to pass, the mark comes off and it stays as an ordinary test.
"""
import pytest

from codicology.model import Block, Page, sync_page


def _carried(htmls, new):
    page = Page(0, "p", [Block("text", h, label=f"L{i}") for i, h in enumerate(htmls)])
    n = sync_page(page, new)
    assert page.render() == new
    return n, [(b.label, b.html) for b in page.blocks]


# fixed 2026-11-19
@pytest.mark.parametrize("head, para", [
    ("<p>12</p>", "<p><i>Ibid.</i> goes on</p>"),
    ("<p>10 THE REBEL PASSION</p>", "<p><b>the</b> sentence goes on.</p>"),
    ("<ul><li>one</li></ul>", "<ul><li>two</li></ul>"),
    ("<table><tr><td>a</td></tr></table>", "<table><tr><td>b</td></tr></table>"),
    ("<blockquote><p>HEAD</p></blockquote>", "<blockquote><p>the text</p></blockquote>"),
])
def test_a_block_cut_whole_from_the_top_keeps_the_next_its_label_whatever_it_opens_with(head, para):
    """The case fixed for "<p>…" over "<p>the sentence…" (test_page_blocks),
    where the next block opens with two tags instead of one."""
    assert _carried([head, para], para) == (0, [("L0", ""), ("L1", para)])


# fixed 2026-11-19
def test_a_heading_retagged_beside_a_paragraph_given_a_class():
    assert _carried(["<p>a</p>", "<p>b</p>"], '<h2>a</h2><p class="y">b</p>') == \
        (0, [("L0", "<h2>a</h2>"), ("L1", '<p class="y">b</p>')])


# fixed 2026-11-19
def test_a_block_cut_beside_one_whose_opening_tag_changed():
    para = "<p>It kept its books in a fair hand.</p>"
    assert _carried(["<p>The firm was old.</p>", para],
                    para.replace("<p>", '<p class="noindent">')) == \
        (0, [("L0", ""), ("L1", para.replace("<p>", '<p class="noindent">'))])


@pytest.mark.xfail(strict=True, reason=(
    "a boundary inside a run of tokens that was only deleted is counted as "
    "straddled, though where it goes is no judgement"))
def test_two_whole_blocks_cut_together_straddle_nothing():
    assert _carried(["<p>a</p>", "<p>b</p>", "<p>c</p>", "<p>d</p>"], "<p>a</p><p>d</p>") == \
        (0, [("L0", "<p>a</p>"), ("L1", ""), ("L2", ""), ("L3", "<p>d</p>")])


@pytest.mark.xfail(strict=True, reason=(
    "a block cut while the blocks beside it are edited: the boundaries "
    "fall in one replaced run, and the last paragraph lands in the cut "
    "block (counted as straddled)"))
def test_a_block_cut_while_its_neighbours_are_edited():
    olds = [f"<p>u{k} the the</p>" for k in range(4)]
    assert _carried(olds, "<p>u0 the</p><p>u1 the the</p><p>u3 the</p>") == \
        (0, [("L0", "<p>u0 the</p>"), ("L1", "<p>u1 the the</p>"), ("L2", ""),
             ("L3", "<p>u3 the</p>")])
