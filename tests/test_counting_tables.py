"""The guard against tables that only count.

A generative recogniser handed a page it cannot read does not always loop.
Sometimes it counts: three tables on the shelf were nothing but integers
climbing by one — 674 of them in 2,029 cells on a page of The History of
Lloyd's, 539 beside the same three words on a part title of TiHKAL, and
1 to 18 twice around an empty grid on a facsimile plate. The loop guard
cannot see a count (every six-word run is new) and none of the three pages
had layer enough for the layer guard. The shapes below are those three and
the real tables measured beside them; the witness cases are the pages that
genuinely print a count, which no shape can tell from an invention.
"""
import re
import zipfile

import cv2
import numpy as np
import pytest
from PIL import Image


def _table(rows):
    return "<table><tbody>" + "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>"
        for row in rows) + "</tbody></table>"


# The three inventions, at their measured shapes.
LLOYDS = _table([["Ref.", "", ""]] + [[n, "", ""] if n % 2 else ["", n, n]
                                      for n in range(1, 451)])
TIHKAL = _table([["The Prophetic Mother", 2 * k, 2 * k + 1]
                 for k in range(1, 271)])
GRID = _table([[""] + list(range(1, 19))]
              + [[k] + [""] * 18 for k in range(1, 19)])

# A page that really prints a count: the numbers one to a hundred, in rows.
HUNDRED = _table([list(range(r, r + 10)) for r in range(1, 101, 10)])


def _item(vtb, html, **kw):
    return vtb.PageItem(html=html, **kw)


def _page(tmp_path, layer=None, name="page_0007.png"):
    p = str(tmp_path / name)
    Image.new("RGB", (300, 400), "white").save(p)
    if layer is not None:
        with open(p + ".layer.txt", "w", encoding="utf-8") as fh:
            fh.write(layer)
    return p


@pytest.fixture
def silent(vtb, monkeypatch):
    """A classical reader that finds nothing on the page."""
    monkeypatch.setattr(vtb, "_classical_text", lambda path: "")


@pytest.mark.parametrize("table, integers", [(LLOYDS, 675), (TIHKAL, 540),
                                             (GRID, 36)])
def test_the_three_shapes_that_occurred_are_counts(vtb, table, integers):
    assert len(vtb._count_in_table(table)) == integers


def test_a_table_that_only_counts_is_dropped_and_the_page_kept(vtb, tmp_path,
                                                               silent):
    """TiHKAL's part title: the heading is really printed, the table is
    not, and the layout pass called the table a contents page."""
    items = [_item(vtb, "<h2>Epilogue</h2>", label="SectionHeader"),
             _item(vtb, TIHKAL, label="TableOfContents")]
    out, dropped = vtb.drop_counting_tables(items, _page(tmp_path, "Epilogue"))
    assert [it.html for it in out] == ["<h2>Epilogue</h2>"]
    assert dropped == [540]


def test_a_page_that_was_nothing_but_the_count_is_left_empty(vtb, tmp_path,
                                                             silent):
    """Lloyd's page 49: one item, 2,029 cells. An empty page is something
    the passes after this one already know how to look at."""
    out, dropped = vtb.drop_counting_tables([_item(vtb, LLOYDS, label="Table")],
                                            _page(tmp_path))
    assert out == [] and dropped == [675]


def test_an_empty_grid_numbered_round_its_edge_is_dropped(vtb, tmp_path,
                                                          silent):
    out, dropped = vtb.drop_counting_tables([_item(vtb, GRID)],
                                            _page(tmp_path, "Journal 1546"))
    assert out == [] and dropped == [36]


def test_a_contents_page_is_not_a_count(vtb):
    """Page numbers climb, and several chapters open on one page — but a
    contents page says a different thing beside every number. The least
    varied real one measured had 33 distinct lines beside 36 numbers."""
    toc = _table([[f"Chapter {k}. On the matter of part {k}", 10 + 7 * (k // 2)]
                  for k in range(1, 41)])
    assert vtb._count_in_table(toc) is None
    items = [_item(vtb, toc, label="TableOfContents")]
    assert vtb.drop_counting_tables(items) == (items, [])


def test_numbered_rows_are_not_a_count(vtb):
    ranks = _table([[k, f"Holding company no. {k}", f"{k * 3.1:.1f}%"]
                    for k in range(1, 36)])
    assert vtb._count_in_table(ranks) is None


def test_a_grid_whose_rows_run_back_and_forth_is_not_a_count(vtb):
    """A surveyed township numbers its thirty-six sections boustrophedon:
    half the steps go down, so it never looked like a counter."""
    rows = [list(range(r, r + 6)) for r in range(1, 37, 6)]
    township = _table([row if i % 2 else row[::-1]
                       for i, row in enumerate(rows)])
    assert vtb._count_in_table(township) is None


def test_a_short_run_of_numbers_is_not_a_count(vtb):
    """Fourteen sizes in a row and two labels: the shape, at a size no
    invention has come in and plenty of real tables do."""
    shoes = _table([["Left shoes"] + list(range(1, 8)),
                    ["Right shoes"] + list(range(1, 8))])
    assert vtb._count_in_table(shoes) is None


def test_figures_that_do_not_climb_are_not_a_count(vtb):
    ledger = _table([[(n * 37) % 91, (n * 53) % 67] for n in range(1, 40)])
    assert vtb._count_in_table(ledger) is None


def test_a_count_the_pages_own_layer_prints_is_kept(vtb, tmp_path,
                                                    monkeypatch):
    """A hundred chart has the shape exactly. The layer cannot invent, so
    what it prints is on the page — however few words it has, and without
    troubling the classical reader."""
    def boom(path):
        raise AssertionError("the layer had already answered")
    monkeypatch.setattr(vtb, "_classical_text", boom)
    layer = "\n".join(" ".join(map(str, range(r, r + 10)))
                      for r in range(1, 101, 10))
    items = [_item(vtb, HUNDRED)]
    assert vtb.drop_counting_tables(items, _page(tmp_path, layer)) \
        == (items, [])


def test_a_count_only_the_classical_reader_sees_is_kept(vtb, tmp_path,
                                                        monkeypatch):
    """A scan: the layer is prose from somewhere else or nothing at all
    (four real tables measured 0.00 against theirs), and tesseract reads
    the numbers. Either witness is enough."""
    monkeypatch.setattr(vtb, "_classical_text", lambda path: " ".join(
        str(n) for n in range(1, 101) if n % 7))      # and it misses a few
    prose = "the answers to the exercises of this chapter follow " * 9
    items = [_item(vtb, HUNDRED)]
    for layer in (None, "", prose):
        page = _page(tmp_path, layer)
        assert vtb.drop_counting_tables(items, page) == (items, [])


def test_a_count_neither_witness_holds_is_dropped(vtb, tmp_path, monkeypatch):
    """The grid on the facsimile plate: both witnesses read the plate's own
    figures, which happen to include a few small numbers — 0.22 and 0.28 of
    the count. Half is the bar."""
    monkeypatch.setattr(vtb, "_classical_text",
                        lambda path: "Anno 1546 the 26 day of April 3 5 9 li")
    page = _page(tmp_path, "Journal of Sir Thomas Gresham 1546 2 7 12 18")
    assert vtb.drop_counting_tables([_item(vtb, GRID)], page) == ([], [36])


def test_digits_inside_longer_numbers_do_not_vouch(vtb, tmp_path, silent):
    page = _page(tmp_path, "".join(str(n) for n in range(1, 101)) + " 1900")
    assert vtb.drop_counting_tables([_item(vtb, HUNDRED)], page) == ([], [100])


def test_without_a_witness_a_count_stays_refused(vtb, tmp_path, monkeypatch):
    """No tesseract and no layer: the loop guard's rule, for the same
    reason — the shape accuses and nothing can answer for it."""
    monkeypatch.setattr(vtb, "_classical_text", lambda path: None)
    assert vtb.drop_counting_tables([_item(vtb, HUNDRED)],
                                    _page(tmp_path)) == ([], [100])
    assert vtb.drop_counting_tables([_item(vtb, HUNDRED)]) == ([], [100])


def test_a_count_inside_a_longer_block_is_cut_out_of_it(vtb, tmp_path,
                                                        silent):
    block = "<p>Schedule of the rates.</p>" + TIHKAL + "<p>Here it ends.</p>"
    out, dropped = vtb.drop_counting_tables(
        [_item(vtb, block, label="Text")], _page(tmp_path))
    assert [it.html for it in out] == [
        "<p>Schedule of the rates.</p><p>Here it ends.</p>"]
    assert out[0].label == "Text" and dropped == [540]


def test_a_real_table_beside_a_count_stays(vtb, tmp_path, silent):
    real = _table([["Year", "Tonnage"], ["1771", "79,000"], ["1810", "2.1m"]])
    out, dropped = vtb.drop_counting_tables(
        [_item(vtb, real + GRID)], _page(tmp_path))
    assert [it.html for it in out] == [real] and dropped == [36]


def test_running_heads_are_left_to_their_own_pass(vtb, tmp_path, silent):
    items = [_item(vtb, GRID, is_furniture=True)]
    assert vtb.drop_counting_tables(items, _page(tmp_path)) == (items, [])


# --- through a whole build -------------------------------------------------

class PageBackend:
    """Reads each page as scripted for that page, however often it is asked
    — an invention that comes back the same on a second look, as these did."""
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def __init__(self, vtb, paths, readings):
        self.vtb, self.calls = vtb, 0
        self.by_page = {Image.open(p).convert("RGB").tobytes(): r
                        for p, r in zip(paths, readings)}

    def run_items(self, images):
        self.calls += len(images)
        return [[self.vtb.PageItem(html=h, label=lab)
                 for h, lab in self.by_page[im.tobytes()]] for im in images]


@pytest.fixture
def three_pages(tmp_path):
    paths = []
    for i in range(3):
        img = np.full((400, 300, 3), 255, np.uint8)
        cv2.rectangle(img, (40, 60 + i * 10), (260, 90 + i * 10), (30,) * 3, -1)
        p = tmp_path / f"page_{i:04d}.png"
        cv2.imwrite(str(p), img)
        paths.append(str(p))
    return paths


def test_a_build_leaves_the_count_out_fresh_and_from_the_cache(
        vtb, tmp_path, three_pages, monkeypatch, capsys):
    """The three that prompted this were all sitting in caches, so the guard
    stands where the book is assembled: the cache keeps what the model said,
    and no build prints it."""
    monkeypatch.setattr(vtb, "tesseract_words_on_page", lambda p: 12,
                        raising=False)
    monkeypatch.setattr(vtb, "_classical_word_count",
                        lambda p, min_len=2: 200)
    monkeypatch.setattr(vtb, "_classical_text", lambda path: "Epilogue")
    readings = [
        [("<p>The first page of the book, which reads as it should.</p>",
          "Text")],
        [("<h2>Epilogue</h2>", "SectionHeader"), (TIHKAL, "TableOfContents")],
        [("<p>The last page of the book, and the end of the matter.</p>",
          "Text")],
    ]
    cache = str(tmp_path / "book.ocr.gz")
    for build in ("fresh", "cached"):
        backend = PageBackend(vtb, three_pages, readings)
        out = tmp_path / f"{build}.epub"
        vtb.build_epub(three_pages, str(out), backend, "Counted", False,
                       dedupe=False, drop_blank=False, ocr_cache=cache)
        log = capsys.readouterr().out
        assert (backend.calls == 0) == (build == "cached")
        assert re.search(r"\[!\] page_0001\.png: a table of 540 numbers that "
                         r"only counts.*left out", log), log
        z = zipfile.ZipFile(out)
        page = z.read(next(n for n in z.namelist()
                           if n.endswith("page_0001.xhtml"))).decode()
        assert "Epilogue" in page
        assert "<table" not in page and "Prophetic" not in page
        book = "".join(z.read(n).decode() for n in z.namelist()
                       if n.endswith(".xhtml"))
        assert "reads as it should" in book and "<table" not in book
