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
    kept, n, held = vtb.drop_doubled_boxes(items)
    assert n == 1 and kept == items[:2] and held == [NEARLY]


@pytest.mark.parametrize("label, html", [
    ("Text", "<p><b>WARNING</b> When the dressing is applied, do not put "
             "pressure on the wound.</p>"),
    ("ListGroup", "<ul><li>Intelligence</li><li>Maneuver</li></ul>"),
    ("TableOfContents", "<p>A Complex Environment ..... 2-2</p>"),
])
def test_a_boxed_note_that_was_read_is_not_also_a_picture(label, html):
    items = [vtb.PageItem(html=html, label=label, box=BOX), _ruled()]
    kept, n, _ = vtb.drop_doubled_boxes(items)
    assert n == 1 and kept == items[:1]


def test_a_block_that_is_only_a_picture_waiting_to_be_cut_gives_way():
    """The recogniser returned the chart as an empty <img>, and the ruled
    box found the same chart. One picture: the one already cut."""
    waiting = vtb.PageItem(html="<table><tr><td><img/></td></tr></table>",
                           label="Table", box=BOX)
    cut = _ruled()
    kept, n, held = vtb.drop_doubled_boxes([waiting, cut])
    assert n == 1 and kept == [cut] and held == [BOX]
    assert vtb.recover_placeholder_figures(kept, "no-such-page.png") == [cut]


def test_a_picture_the_recogniser_named_is_never_dropped():
    cover = vtb.PageItem(figure=_art(), label="Picture", box=NEARLY)
    title = vtb.PageItem(html="<h1>A People's History</h1>",
                         label="SectionHeader", box=BOX)
    kept, n, _ = vtb.drop_doubled_boxes([title, cover])
    assert n == 0 and kept == [title, cover]


@pytest.mark.parametrize("box", [
    (0.12, 0.30, 0.88, 0.46),       # half the table's height
    (0.12, 0.64, 0.88, 0.90),       # a different place on the page
    (0.05, 0.05, 0.95, 0.95),       # a frame round the whole page
])
def test_a_ruled_box_that_is_not_the_blocks_box_is_kept(box):
    items = [vtb.PageItem(html=TABLE, label="Table", box=BOX), _ruled(box)]
    kept, n, held = vtb.drop_doubled_boxes(items)
    assert n == 0 and kept == items and held == []


def test_a_drawing_no_block_covers_is_still_a_picture():
    """What the ruled box is looked for in the first place."""
    items = [vtb.PageItem(html="<p>The structure is shown here.</p>",
                          label="Text", box=(0.1, 0.1, 0.9, 0.28)),
             _ruled()]
    assert vtb.drop_doubled_boxes(items) == (items, 0, [])


def test_nothing_with_words_is_ever_dropped():
    items = [vtb.PageItem(html=TABLE, label="Table", box=BOX), _ruled(),
             vtb.PageItem(html="<p>Table 3.1 shows the schedule.</p>",
                          label="Caption", box=(0.12, 0.63, 0.88, 0.67))]
    kept, _, _ = vtb.drop_doubled_boxes(items)
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


# ── a box that is dropped is still accounted for ─────────────────────────────

LEAF = (466, 704)
RULES = (52, 314, 180, 612)             # the table as it is ruled, in points
HEAD = "NUMERICAL PROBLEMS"
ASKS = "Suppose a firm faces the following supply schedule for labor by unskilled workers:"
WAGES = ("<table><tr><th>Wage per day</th><th>Number of workers</th></tr>"
         + "".join(f"<tr><td>{8 * k}</td><td>{k}</td></tr>" for k in range(9))
         + "</table>")


def _problem_page(tmp_path):
    """A banner with a heading on it, a tinted line of prose under that, and
    a ruled table under that: sixteen drawn paths that touch one another,
    and so one drawing to anything that groups paths by where they are."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=LEAF[0], height=LEAF[1])
    for k in range(14):
        page.insert_text((52, 60 + 14 * k), "The publisher set this line in type.",
                         fontsize=9)
    page.draw_rect(pymupdf.Rect(52, 272, 412, 288), color=None, fill=(0.03, 0.39, 0.61))
    page.insert_text((180, 284), HEAD, fontsize=10, color=(1, 1, 1))
    page.draw_rect(pymupdf.Rect(64, 294, 392, 314), color=None, fill=(0.88, 0.93, 0.97))
    page.insert_text((66, 308), ASKS, fontsize=8)
    x0, y0, x1, y1 = RULES
    rows = [y0 + k * (y1 - y0) / 10 for k in range(11)]
    for y in rows:
        page.draw_line((x0, y), (x1, y), width=1)
    for x in (x0, (x0 + x1) / 2, x1):
        page.draw_line((x, y0), (x, y1), width=1)
    page.insert_text((56, rows[0] + 18), "Wage per day", fontsize=7)
    page.insert_text((120, rows[0] + 18), "Number of workers", fontsize=6)
    for k in range(9):
        page.insert_text((80, rows[k + 1] + 18), str(8 * k), fontsize=8)
        page.insert_text((146, rows[k + 1] + 18), str(k), fontsize=8)
    pdf = tmp_path / "problems.pdf"
    doc.save(str(pdf))
    d = tmp_path / "pages"
    d.mkdir()
    return vtb.load_pages_from_pdf(str(pdf), str(d))[0]


def _problem_blocks():
    """The page as read: every part of it a block, and the table's rule
    found as a box as well. The recogniser's box for the table sits a
    little inside the rule, as it does on the page this is drawn from."""
    x0, y0, x1, y1 = (RULES[0] / LEAF[0], RULES[1] / LEAF[1],
                      RULES[2] / LEAF[0], RULES[3] / LEAF[1])
    return [vtb.PageItem(html=f"<h2>{HEAD}</h2>", label="SectionHeader",
                         box=(0.35, 0.388, 0.65, 0.408)),
            vtb.PageItem(html=f"<p>{ASKS}</p>", label="Text",
                         box=(0.137, 0.420, 0.80, 0.445)),
            vtb.PageItem(html=WAGES, label="Table",
                         box=(x0 + 0.008, y0 + 0.009, x1 - 0.006, y1 - 0.009)),
            vtb.PageItem(figure=_art(), box=(x0, y0, x1, y1))]


def test_a_table_whose_picture_was_dropped_is_not_then_found_as_a_drawing(tmp_path):
    """Principles of Microeconomics, pages 372 and 716. The picture of the
    table had been what told the artwork passes that the rules were spoken
    for. Dropped, it left them looking like a drawing nobody had boxed:
    banner, tinted line and table went in as one picture, and the heading
    and the sentence went with it as the picture's labels."""
    page = _problem_page(tmp_path)
    items = _problem_blocks()
    kept, n, held = vtb.drop_doubled_boxes(items)
    assert n == 1 and kept == items[:3]

    out = vtb.recover_orphan_artwork(kept, page, held)
    out = vtb.recover_vector_artwork(out, page, held)
    assert out == kept, "nothing cut, and nothing taken as a label"


KIT = ["Compass", "Matches", "Knife", "Whistle", "Map", "Torch", "Cord", "Tarp",
       "Water", "Salt", "Needle", "Thread", "Mirror", "Candle", "Flint", "Hooks"]
KIT_BOX = (120, 300, 346, 430)          # the rule round the list, in points


def _kit_page(tmp_path):
    """A ruled box holding a list of sixteen things, each with a drawn square
    to tick: seventeen paths in one place, and few words."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=LEAF[0], height=LEAF[1])
    for k in range(14):
        page.insert_text((52, 60 + 14 * k), "The publisher set this line in type.",
                         fontsize=9)
    page.draw_rect(pymupdf.Rect(*KIT_BOX), width=1)
    for k, thing in enumerate(KIT):
        x, y = 132 + 110 * (k % 2), 310 + 14 * (k // 2)
        page.draw_rect(pymupdf.Rect(x, y, x + 8, y + 8), width=0.7)
        page.insert_text((x + 14, y + 8), thing, fontsize=9)
    pdf = tmp_path / "kit.pdf"
    doc.save(str(pdf))
    d = tmp_path / "pages"
    d.mkdir()
    return vtb.load_pages_from_pdf(str(pdf), str(d))[0]


def test_a_boxed_list_whose_picture_was_dropped_is_not_then_found_as_a_drawing(tmp_path):
    """The squares are drawn, so the list is a drawing to anything that
    groups paths, and its picture had been what said the region was spoken
    for. A table's rules are known for what they are whoever asks; a list's
    tick boxes are known only by the box that was dropped."""
    page = _kit_page(tmp_path)
    box = tuple(v / LEAF[k % 2] for k, v in enumerate(KIT_BOX))
    listed = vtb.PageItem(
        html="<ul>" + "".join(f"<li>{t}</li>" for t in KIT) + "</ul>",
        label="ListGroup",
        box=(box[0] + 0.01, box[1] + 0.008, box[2] - 0.01, box[3] - 0.008))
    kept, n, held = vtb.drop_doubled_boxes([listed, _ruled(box)])
    assert n == 1 and kept == [listed] and held == [box]

    assert vtb.recover_vector_artwork(kept, page, held) == kept
    untold = vtb.recover_vector_artwork(kept, page)
    assert sum(1 for it in untold if it.figure is not None) == 1, \
        "not told of the box, the pass draws the list a second time"


def test_a_stored_picture_under_a_dropped_box_is_not_recovered_either(tmp_path):
    """The same, for a region the PDF holds as pixels."""
    drawing = ("image", (160, 330, 300, 420), _png((280, 180), _plate))
    (page,) = _load(vtb, tmp_path, [typeset_page() + [drawing]])
    box = (160 / 466, 330 / 704, 300 / 466, 420 / 704)
    table = vtb.PageItem(html=TABLE, label="Table", box=box)
    kept, n, held = vtb.drop_doubled_boxes([table, _ruled(box)])
    assert n == 1 and kept == [table]
    assert vtb.recover_orphan_artwork(kept, page, held) == kept
    assert sum(1 for it in vtb.recover_orphan_artwork(kept, page)
               if it.figure is not None) == 1, "untold, it cuts the picture"


# ── a box no single block matches ────────────────────────────────────────────

WARN_BOX = (60, 300, 400, 360)          # in points, on the 466 x 704 leaf
WARN = "DO NOT elevate a suspected fractured limb unless it has been properly splinted."


def _warning_page(tmp_path, drawing=False, name="warn"):
    """A boxed WARNING: the word, and a sentence under it, inside one rule."""
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=LEAF[0], height=LEAF[1])
    for k in range(14):
        page.insert_text((52, 60 + 14 * k), "The publisher set this line in type.",
                         fontsize=9)
    page.draw_rect(pymupdf.Rect(*WARN_BOX), width=1.5)
    page.insert_text((200, 318), "WARNING", fontsize=10)
    page.insert_text((70, 340), WARN, fontsize=8)
    if drawing:                          # a sketch beside the words
        for k in range(6):
            page.draw_line((330 + 8 * k, 306), (360, 352), width=1)
    pdf = tmp_path / f"{name}.pdf"
    doc.save(str(pdf))
    d = tmp_path / f"{name}-pages"
    d.mkdir()
    return vtb.load_pages_from_pdf(str(pdf), str(d))[0]


def _warning_blocks():
    box = tuple(v / LEAF[k % 2] for k, v in enumerate(WARN_BOX))
    return [vtb.PageItem(html="<p>The publisher set this line in type.</p>",
                         label="Text", box=(0.1, 0.07, 0.9, 0.36)),
            vtb.PageItem(html="<p><b>WARNING</b></p>", label="SectionHeader",
                         box=(0.40, 0.435, 0.60, 0.455)),
            vtb.PageItem(html=f"<p><b>{WARN}</b></p>", label="Text",
                         box=(0.14, 0.465, 0.85, 0.49)),
            vtb.PageItem(figure=_art(), box=box)]


def test_a_boxed_warning_read_as_two_blocks_is_not_also_a_picture(tmp_path):
    """FM 4-25, page 43: the box as a picture, then WARNING, then the
    sentence. No one block has the box's outline, so the rule that asks for
    one let the picture through on 62 pages of that book."""
    items = _warning_blocks()
    kept, n, held = vtb.drop_doubled_boxes(items, _warning_page(tmp_path))
    assert n == 1 and kept == items[:3] and held == [items[3].box]


def test_without_the_page_to_ask_the_box_is_kept(tmp_path):
    items = _warning_blocks()
    assert vtb.drop_doubled_boxes(items) == (items, 0, [])


def test_a_box_with_a_drawing_in_it_is_kept(tmp_path):
    """The picture is the only copy of the drawing."""
    items = _warning_blocks()
    kept, n, _ = vtb.drop_doubled_boxes(
        items, _warning_page(tmp_path, drawing=True))
    assert n == 0 and kept == items


def test_a_box_whose_words_were_not_read_is_kept(tmp_path):
    """And here it is the only copy of the words."""
    items = _warning_blocks()
    del items[2]                         # the sentence never became a block
    kept, n, _ = vtb.drop_doubled_boxes(items, _warning_page(tmp_path))
    assert n == 0 and kept == items


def test_a_cell_of_a_table_that_was_read_is_not_also_a_picture(tmp_path):
    """FM 4-25, page 127: Table 5-2 rebuilt as a table, and three of its
    cells again as pictures of their words."""
    page = _problem_page(tmp_path)
    heading, asks, table, _ = _problem_blocks()
    x0, y0, x1, y1 = RULES
    rows = [y0 + k * (y1 - y0) / 10 for k in range(11)]
    cell = (x0 / LEAF[0], rows[1] / LEAF[1], x1 / LEAF[0], rows[4] / LEAF[1])
    items = [heading, asks, table, vtb.PageItem(figure=_art(), box=cell)]
    kept, n, held = vtb.drop_doubled_boxes(items, page)
    assert n == 1 and kept == items[:3] and held == [cell]


def test_a_scan_is_left_to_the_rule_it_had(tmp_path):
    """A scan's PDF holds one picture of the page and says nothing about
    what is in a box; TiHKAL's structures are ruled boxes inside text."""
    import json
    page = _warning_page(tmp_path)
    with open(page + ".source.json") as fh:
        src = json.load(fh)
    src["scan"] = True
    with open(page + ".source.json", "w") as fh:
        json.dump(src, fh)
    items = _warning_blocks()
    assert vtb.drop_doubled_boxes(items, page) == (items, 0, [])


# ── a table that lost its figures ────────────────────────────────────────────

def test_a_table_that_holds_too_few_of_its_figures_is_named(tmp_path):
    """Burning Up, Table 12: four of six columns. The page is typeset, so
    the PDF can be asked how many figures stand where the table stands."""
    page = _problem_page(tmp_path)
    heading, asks, table, _ = _problem_blocks()
    assert vtb.tables_short_of_figures([heading, asks, table], page) == []
    thin = table._replace(html=(
        "<table><tr><th>Wage per day</th><th>Number of workers</th></tr>"
        "<tr><td>0</td><td>0</td></tr><tr><td>8</td><td>1</td></tr></table>"))
    short = vtb.tables_short_of_figures([heading, asks, thin], page)
    assert len(short) == 1 and short[0][0] == 18 and short[0][1] < 0.7 * 18


def test_a_table_on_a_scan_is_not_judged_this_way(tmp_path):
    import json
    page = _problem_page(tmp_path)
    with open(page + ".source.json") as fh:
        src = json.load(fh)
    src["scan"] = True
    with open(page + ".source.json", "w") as fh:
        json.dump(src, fh)
    thin = _problem_blocks()[2]._replace(html="<table><tr><td>0</td></tr></table>")
    assert vtb.tables_short_of_figures([thin], page) == []


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


class _ReadsProblems(_Reads):
    def run_items(self, images):
        return [_problem_blocks() for _ in images]


def test_a_built_page_keeps_the_heading_over_its_table(tmp_path, capsys):
    page = _problem_page(tmp_path)
    out = tmp_path / "book.epub"
    vtb.build_epub([page], str(out), _ReadsProblems(), "Problems", False,
                   dedupe=False, drop_blank=False)
    body = zipfile.ZipFile(out).read("EPUB/page_0000.xhtml").decode()
    assert f"<h2>{HEAD}</h2>" in body and "following supply schedule" in body
    assert "Number of workers" in body and "<img" not in body
    assert "left out 1 picture(s) of a ruled box" in capsys.readouterr().out
