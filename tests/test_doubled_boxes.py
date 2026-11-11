"""A region the page already has a block for is not printed a second time.

A ruled box is looked for on the page so that a drawing the layout never
made a block of is not lost. It was told which regions were already
figures and nothing else, so a ruled table the recogniser read as a table
was cut out as a picture too. Across the shelf's caches that is 360 pictures
of things the page already says (332 tables, 28 boxed notes and lists), and
57 more where the block was only a picture waiting to be cut, which then
got cut twice: Where There Is No Dentist printed its diagnostic charts
twice over.
"""
import re
import zipfile

import pytest
from PIL import Image, ImageDraw

from codicology import pipeline as vtb
from test_orphan_artwork import _load
from test_page_loading import _plate, _png, typeset_page

TABLE = ("<table><tr><th>Price per bushel</th><th>Quantity demanded</th></tr>"
         "<tr><td>$0</td><td>6</td></tr><tr><td>1</td><td>5</td></tr></table>")
BOX = (0.12, 0.30, 0.88, 0.62)
NEARLY = (0.125, 0.303, 0.883, 0.624)       # the same box, found by its border


def _art():
    im = Image.new("RGB", (300, 200), "white")
    ImageDraw.Draw(im).rectangle((4, 4, 295, 195), outline="black", width=3)
    return im


def _ruled(box=NEARLY):
    return vtb.PageItem(figure=_art(), box=box)        # no label: found by its rule


def test_a_table_that_was_read_is_not_also_a_picture():
    items = [vtb.PageItem(html="<p>Consider the market for wheat.</p>",
                          label="Text", box=(0.1, 0.1, 0.9, 0.28)),
             vtb.PageItem(html=TABLE, label="Table", box=BOX),
             _ruled()]
    kept, n = vtb.drop_doubled_boxes(items)
    assert n == 1 and kept == items[:2]


@pytest.mark.parametrize("label, html", [
    ("Text", "<p><b>WARNING</b> When the dressing is applied, do not put "
             "pressure on the wound.</p>"),
    ("ListGroup", "<ul><li>Intelligence</li><li>Maneuver</li></ul>"),
    ("TableOfContents", "<p>A Complex Environment ..... 2-2</p>"),
])
def test_a_boxed_note_that_was_read_is_not_also_a_picture(label, html):
    items = [vtb.PageItem(html=html, label=label, box=BOX), _ruled()]
    kept, n = vtb.drop_doubled_boxes(items)
    assert n == 1 and kept == items[:1]


def test_a_block_that_is_only_a_picture_waiting_to_be_cut_gives_way():
    """The recogniser returned the chart as an empty <img>, and the ruled
    box found the same chart. One picture: the one already cut."""
    waiting = vtb.PageItem(html="<table><tr><td><img/></td></tr></table>",
                           label="Table", box=BOX)
    cut = _ruled()
    kept, n = vtb.drop_doubled_boxes([waiting, cut])
    assert n == 1 and kept == [cut]
    assert vtb.recover_placeholder_figures(kept, "no-such-page.png") == [cut]


def test_a_picture_the_recogniser_named_is_never_dropped():
    cover = vtb.PageItem(figure=_art(), label="Picture", box=NEARLY)
    title = vtb.PageItem(html="<h1>A People's History</h1>",
                         label="SectionHeader", box=BOX)
    kept, n = vtb.drop_doubled_boxes([title, cover])
    assert n == 0 and kept == [title, cover]


@pytest.mark.parametrize("box", [
    (0.12, 0.30, 0.88, 0.46),       # half the table's height
    (0.12, 0.64, 0.88, 0.90),       # a different place on the page
    (0.05, 0.05, 0.95, 0.95),       # a frame round the whole page
])
def test_a_ruled_box_that_is_not_the_blocks_box_is_kept(box):
    items = [vtb.PageItem(html=TABLE, label="Table", box=BOX), _ruled(box)]
    kept, n = vtb.drop_doubled_boxes(items)
    assert n == 0 and kept == items


def test_a_drawing_no_block_covers_is_still_a_picture():
    """What the ruled box is looked for in the first place."""
    items = [vtb.PageItem(html="<p>The structure is shown here.</p>",
                          label="Text", box=(0.1, 0.1, 0.9, 0.28)),
             _ruled()]
    assert vtb.drop_doubled_boxes(items) == (items, 0)


def test_nothing_with_words_is_ever_dropped():
    items = [vtb.PageItem(html=TABLE, label="Table", box=BOX), _ruled(),
             vtb.PageItem(html="<p>Table 3.1 shows the schedule.</p>",
                          label="Caption", box=(0.12, 0.63, 0.88, 0.67))]
    kept, _ = vtb.drop_doubled_boxes(items)
    said = lambda its: " ".join(vtb._strip_tags(i.html) for i in its if i.html)
    assert said(kept) == said(items)


# ── a picture inside a picture ───────────────────────────────────────────────

def test_a_drawing_inside_a_chart_already_shipped_is_not_added_again(vtb,
                                                                     tmp_path):
    """The chart is one picture. A drawing set inside it overlaps the
    chart's whole box too little to count as covered, so it was added a
    second time, on its own, beneath the chart."""
    drawing = ("image", (160, 330, 300, 420), _png((280, 180), _plate))
    (page,) = _load(vtb, tmp_path, [typeset_page() + [drawing]])
    chart = vtb.PageItem(figure=_art(), label="Picture",
                         box=(0.10, 0.30, 0.90, 0.75))
    out = vtb.recover_orphan_artwork([chart], page)
    assert out == [chart]

    alone = vtb.recover_orphan_artwork(
        [vtb.PageItem(html="<p>Prose.</p>", box=(0.1, 0.05, 0.9, 0.2))], page)
    assert sum(1 for it in alone if it.figure is not None) == 1, \
        "with no chart round it, the drawing is recovered as before"


# ── through a build ──────────────────────────────────────────────────────────

class _Reads:
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def run_items(self, images):
        return [[vtb.PageItem(html="<p>Consider the market for wheat.</p>",
                              label="Text", box=(0.1, 0.1, 0.9, 0.28)),
                 vtb.PageItem(html=TABLE, label="Table", box=BOX),
                 _ruled()] for _ in images]


def test_a_built_page_carries_its_table_once(tmp_path, capsys):
    page = tmp_path / "page_0000.png"
    im = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(im)
    for row in range(80, 700, 40):
        d.rectangle((60, row, 540, row + 14), fill=(30, 30, 30))
    im.save(page)
    out = tmp_path / "book.epub"
    vtb.build_epub([str(page)], str(out), _Reads(), "Wheat", False,
                   dedupe=False, drop_blank=False)
    body = zipfile.ZipFile(out).read("EPUB/page_0000.xhtml").decode()
    assert "Price per bushel" in body and "<img" not in body
    assert "left out 1 picture(s) of a ruled box" in capsys.readouterr().out
