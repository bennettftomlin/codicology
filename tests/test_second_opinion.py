"""What is done with a second reading.

The second reader is never taken on its own word. It is asked only where
something that cannot invent has found the first reading wanting, and its
answer is printed only where that same witness prefers it.
"""
import json

import pytest

from codicology import pipeline as vtb
from test_doubled_boxes import _problem_blocks, _problem_page

THIN = ("<table><tr><th>Wage per day</th><th>Number of workers</th></tr>"
        "<tr><td>0</td><td>0</td></tr><tr><td>8</td><td>1</td></tr></table>")
WHOLE = ("<table><thead><tr><th>Wage per day</th><th>Number of workers</th></tr></thead>"
         + "".join(f"<tr><td>{8 * k}</td><td>{k}</td></tr>" for k in range(9))
         + "</table>")


class _Second:
    """A second reader that says what it is told to, and counts being asked."""
    def __init__(self, reading):
        self._reading, self.asked = reading, 0

    def reading(self, page_path):
        self.asked += 1
        return self._reading


def _page_with(tmp_path, table_html):
    page = _problem_page(tmp_path)
    heading, asks, table, _ = _problem_blocks()
    return page, [heading, asks, table._replace(html=table_html)]


def test_a_table_that_lost_its_rows_takes_the_second_readers(tmp_path):
    """Burning Up, Table 12, kept four of its six columns. The page is
    typeset, the PDF says which figures stand where the table stands, and
    the second reader's table holds them all."""
    page, items = _page_with(tmp_path, THIN)
    second = _Second("Suppose a firm faces this schedule:\n\n" + WHOLE)
    out, mended = vtb.mend_tables(items, page, second)
    assert mended == [(18, 4, 18, [])]
    assert "<td>64</td>" in out[2].html and out[2].html.startswith('<table border="1"')
    assert out[2].box == items[2].box and out[:2] == items[:2]


def test_a_table_that_holds_its_figures_is_not_asked_about(tmp_path):
    page, items = _page_with(tmp_path, WHOLE)
    second = _Second(WHOLE)
    assert vtb.mend_tables(items, page, second) == (items, [])
    assert second.asked == 0, "the second reader costs a page's reading time"


def test_a_second_table_no_better_than_the_first_is_not_taken(tmp_path):
    page, items = _page_with(tmp_path, THIN)
    assert vtb.mend_tables(items, page, _Second(THIN)) == (items, [])
    assert vtb.mend_tables(items, page, _Second("no table on this page")) == (items, [])
    assert vtb.mend_tables(items, page, _Second(None)) == (items, [])


def test_a_larger_table_that_merely_contains_this_one_is_not_taken(tmp_path):
    """The second reader may have run two tables into one. What it holds
    beyond the figures printed in this table's place counts against it."""
    page, items = _page_with(tmp_path, THIN)
    bigger = WHOLE.replace("</table>", "".join(
        f"<tr><td>{100 + k}</td><td>{200 + k}</td></tr>" for k in range(6)) + "</table>")
    assert vtb.mend_tables(items, page, _Second(bigger)) == (items, [])


def test_without_a_second_reader_nothing_changes(tmp_path):
    page, items = _page_with(tmp_path, THIN)
    assert vtb.mend_tables(items, page, None) == (items, [])


def test_a_scan_has_no_witness_to_prefer_either_table(tmp_path):
    page, items = _page_with(tmp_path, THIN)
    with open(page + ".source.json") as fh:
        src = json.load(fh)
    src["scan"] = True
    with open(page + ".source.json", "w") as fh:
        json.dump(src, fh)
    second = _Second(WHOLE)
    assert vtb.mend_tables(items, page, second) == (items, [])
    assert second.asked == 0


def test_a_figure_the_pdf_does_not_print_there_is_said(tmp_path):
    """The table is still the better one and is taken. What it holds that
    the PDF does not is named in the build's log beside the change."""
    page, items = _page_with(tmp_path, THIN)
    odd = WHOLE.replace("<th>Number of workers</th>", "<th>Number of workers, 1975</th>")
    out, mended = vtb.mend_tables(items, page, _Second(odd))
    assert mended == [(18, 4, 18, ["1975"])]


def test_a_box_that_stops_short_of_the_tables_last_column_is_looked_round(tmp_path):
    """The first reader lost a column and drew its box without it. What the
    second reader holds beyond the box is still printed beside it."""
    page, items = _page_with(tmp_path, THIN)
    x0, y0, x1, y1 = items[2].box
    narrow = items[2]._replace(box=(x0, y0, x0 + (x1 - x0) * 0.62, y1))
    out, mended = vtb.mend_tables(items[:2] + [narrow], page, _Second(WHOLE))
    assert len(mended) == 1 and mended[0][3] == [] and "<td>64</td>" in out[2].html


# ── the readings themselves ──────────────────────────────────────────────────

class _Reader:
    def __init__(self):
        self.read_count = 0
    def read(self, image):
        self.read_count += 1
        return "what the second reader said"
    def close(self):
        self.closed = True


def test_a_page_is_read_once_and_remembered(tmp_path):
    from codicology import second as second_
    page = _problem_page(tmp_path)
    cache_path = str(tmp_path / "book.second.gz")
    reader = _Reader()
    opinion = vtb.SecondOpinion(reader, second_.SecondCache(cache_path))
    assert opinion.reading(page) == "what the second reader said"
    assert opinion.reading(page) == "what the second reader said"
    assert reader.read_count == 1 and opinion.asked == 1
    opinion.close()
    again = vtb.SecondOpinion(_Reader(), second_.SecondCache(cache_path))
    assert again.reading(page) == "what the second reader said"
    assert again.reader.read_count == 0, "a rebuild does not read it again"


def test_a_build_says_what_the_second_reader_was_asked_and_what_came_of_it(tmp_path):
    from codicology import second as second_
    page = _problem_page(tmp_path)
    opinion = vtb.SecondOpinion(_Reader(), second_.SecondCache(
        str(tmp_path / "book.second.gz")))
    assert opinion.summary() is None, "never asked, nothing to say"
    opinion.reading(page)
    assert opinion.summary() == ("second reader: asked about 1 page(s); "
                                 "nothing was changed on its word")
    opinion.tables, opinion.equations, opinion.differ = 1, 2, 3
    assert opinion.summary() == (
        "second reader: asked about 1 page(s); 1 table(s) and 2 equation(s) "
        "set as it reads them; 3 equation(s) on which the readers differ are "
        "listed above for a look")
    opinion.close()
    again = vtb.SecondOpinion(_Reader(), second_.SecondCache(
        str(tmp_path / "book.second.gz")))
    again.reading(page)
    assert again.summary().startswith(
        "second reader: asked about 1 page(s), 1 of them remembered;")


class _Down:
    """A reader that cannot be asked."""
    def __init__(self):
        self.tried = 0
    def read(self, image):
        self.tried += 1
        raise RuntimeError("the second reader's server did not start (exit 1)")
    def close(self):
        pass


def test_a_page_it_could_not_be_asked_about_is_asked_about_next_time(tmp_path):
    """A server that failed to start once left every page it was tried on
    filed as having nothing to say, and no later build asked again."""
    from codicology import second as second_
    page = _problem_page(tmp_path)
    cache_path = str(tmp_path / "book.second.gz")
    opinion = vtb.SecondOpinion(_Down(), second_.SecondCache(cache_path))
    assert opinion.reading(page) is None
    assert opinion.reading(page) is None
    assert opinion.reader.tried == 1, "and not tried twice in one build"
    said = opinion.summary()
    assert "could not be asked about 1 of those pages" in said
    assert "did not start" in said and "remembered" not in said
    opinion.close()
    again = vtb.SecondOpinion(_Reader(), second_.SecondCache(cache_path))
    assert again.reading(page) == "what the second reader said"
    assert again.reader.read_count == 1


def test_without_a_cache_a_page_is_still_read_once_in_a_build(tmp_path):
    """A page with a short table and an equation is put to it twice."""
    page = _problem_page(tmp_path)
    opinion = vtb.SecondOpinion(_Reader())
    assert opinion.reading(page) == opinion.reading(page)
    assert opinion.reader.read_count == 1 and opinion.asked == 1
    assert opinion.summary() == ("second reader: asked about 1 page(s); "
                                 "nothing was changed on its word")


# ── equations ────────────────────────────────────────────────────────────────

IRON_PRINTED = "(90+30+30)(1+20/100) = 108 t. iron"
IRON_FIRST = r"(90 + 30 + 30)(1 + \frac{2}{100}) = 108 \text{ t. iron}"
IRON_SECOND = r"(90 + 30 + 30)(1 + \frac{20}{100}) = 108 \text{ t. iron}"
EQ_BOX = (0.10, 0.44, 0.80, 0.48)


def _equation_page(tmp_path, scan=False, printed=None):
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=466, height=704)
    for k in range(14):
        page.insert_text((52, 60 + 14 * k), "The publisher set this line in type.",
                         fontsize=9)
    page.insert_text((60, 326), printed or IRON_PRINTED, fontsize=10)
    pdf = tmp_path / "iron.pdf"
    doc.save(str(pdf))
    d = tmp_path / "eq-pages"
    d.mkdir()
    path = vtb.load_pages_from_pdf(str(pdf), str(d))[0]
    if scan:
        with open(path + ".source.json") as fh:
            src = json.load(fh)
        src["scan"] = True
        with open(path + ".source.json", "w") as fh:
            json.dump(src, fh)
    return path


def _equation(tex, box=EQ_BOX):
    return vtb.PageItem(html=f'<math display="block">{tex}</math>',
                        label="Equation", box=box)


def test_an_equation_the_pdf_bears_out_is_not_asked_about(tmp_path):
    page = _equation_page(tmp_path)
    second = _Second("$$ " + IRON_FIRST + " $$")
    items = [_equation(IRON_SECOND)]
    assert vtb.mend_equations(items, page, second) == (items, [], [])
    assert second.asked == 0


def test_on_a_typeset_page_the_pdfs_symbols_choose_between_two_readings(tmp_path):
    """The first reader dropped a digit from a fraction. The PDF's text has
    no fraction bar to show, but it has the digits, and the second reading
    is nearer them."""
    page = _equation_page(tmp_path)
    out, changed, _ = vtb.mend_equations(
        [_equation(IRON_FIRST)], page, _Second("so that\n\n$$ " + IRON_SECOND + " $$\n"))
    assert changed == [("(90+30+30)(1+2/100)=108t.iron",
                        "(90+30+30)(1+20/100)=108t.iron")]
    assert out[0].html == f'<math display="block">{IRON_SECOND}</math>'


def test_a_second_reading_further_from_the_pdf_is_not_taken(tmp_path):
    page = _equation_page(tmp_path)
    slipped = IRON_FIRST.replace("108", "103")
    worse = IRON_FIRST.replace("108", "1O3").replace("90", "9O")
    items = [_equation(slipped)]
    assert vtb.mend_equations(items, page, _Second("$$ " + worse + " $$")) \
        == (items, [], [])


def test_on_a_scan_a_difference_in_figures_alone_goes_to_the_second_reader(tmp_path):
    """Sraffa, section 27: (1 + 2/100) three times where the page prints
    20/100. There is no PDF text on a scan to say so. On figures the second
    reader has a record: it held 93% of the figures PDFs print where the
    first held 75%."""
    page = _equation_page(tmp_path, scan=True)
    out, changed, differ = vtb.mend_equations(
        [_equation(IRON_FIRST)], page, _Second("$$ " + IRON_SECOND + " $$"))
    assert len(changed) == 1 and differ == [] and r"\frac{20}{100}" in out[0].html


def test_a_second_reading_with_a_digit_fewer_is_listed_and_not_taken(tmp_path):
    """The same page the other way round: had the first reader the 20 and
    the second the 2, the second would be the one that dropped a figure."""
    page = _equation_page(tmp_path, scan=True)
    items = [_equation(IRON_SECOND)]
    out, changed, differ = vtb.mend_equations(items, page,
                                              _Second("$$ " + IRON_FIRST + " $$"))
    assert out == items and changed == [] and len(differ) == 1


@pytest.mark.parametrize("first, second, alone", [
    ("(1+2/100)", "(1+20/100)", True),      # a digit more
    ("r=R'(1-1/2)", "r=R'(1-1/3)", True),   # another digit
    ("(1+r)25", "(1+r)2", False),           # a digit fewer
    ("(1+r)25", "(1+r)", False),            # the figure gone
    ("pa+pb", "pa+pk", False),              # not a figure at all
    ("10,000", "10000", False),             # punctuation is not a figure
])
def test_what_counts_as_differing_in_figures_alone(first, second, alone):
    assert vtb._differ_in_figures_alone(first, second) is alone


def test_a_system_set_as_one_block_is_matched_to_its_lines(tmp_path):
    """The first reader sets a system of equations as one aligned block and
    the second as a line each. They are the same mathematics, and differ in
    one subscript."""
    page = _equation_page(tmp_path, scan=True)
    block = (r"\begin{aligned} (A_a p_a + B_a p_b)(1 + r) &amp;= A p_a \\ "
             r"(A_{bp_c} + B_b p_b)(1 + r) &amp;= B p_k \end{aligned}")
    lines = ("$$ (A_a p_a + B_a p_b)(1 + r) = A p_a $$\n\n"
             "$$ (A_b p_a + B_b p_b)(1 + r) = B p_b $$")
    out, changed, differ = vtb.mend_equations([_equation(block)], page, _Second(lines))
    assert changed == [] and out == [_equation(block)], \
        "letters are not figures: two readers cannot outvote each other"
    assert differ == [("(Aapa+Bapb)(1+r)=Apa(Abpc+Bbpb)(1+r)=Bpk",
                       "(Aapa+Bapb)(1+r)=Apa(Abpa+Bbpb)(1+r)=Bpb")]


def test_a_system_that_differs_in_a_figure_is_set_a_line_each(tmp_path):
    page = _equation_page(tmp_path, scan=True)
    block = (r"\begin{aligned} (90 + 30)(1 + \frac{2}{100}) &amp;= 108 \\ "
             r"(120 + 75)(1 + \frac{2}{100}) &amp;= 270 \end{aligned}")
    lines = ("$$ (90 + 30)(1 + \\frac{20}{100}) = 108 $$\n\n"
             "$$ (120 + 75)(1 + \\frac{20}{100}) = 270 $$")
    out, changed, differ = vtb.mend_equations([_equation(block)], page, _Second(lines))
    assert len(changed) == 1 and differ == []
    assert out[0].html.count('<math display="block">') == 2
    assert r"(120 + 75)(1 + \frac{20}{100}) = 270" in out[0].html


@pytest.mark.parametrize("theirs", [
    "$$ " + IRON_FIRST + " $$",                             # the same reading
    "$$ " + IRON_FIRST.replace("= 108", ". . . = 108") + " $$",  # but for dots
    "$$ E = m c^2 $$",                                      # another equation
    "no mathematics on this page at all",
    None,                                                   # it did not answer
])
def test_what_does_not_differ_or_does_not_match_is_left(tmp_path, theirs):
    page = _equation_page(tmp_path, scan=True)
    items = [_equation(IRON_FIRST)]
    assert vtb.mend_equations(items, page, _Second(theirs)) == (items, [], [])


def test_a_prime_before_or_after_its_subscript_is_one_symbol(vtb_unused=None):
    assert vtb._flat_tex(r"p'_a (A_a q''_{k})") == vtb._flat_tex(r"p_a' (A_a q_{k}'')")


def test_a_reading_that_runs_the_symbols_together_is_not_taken(tmp_path):
    """Principles of Microeconomics, page 403: seven short formulas on one
    line. The first reader dropped a letter (PL for APL). The second has
    every letter and runs each formula into the next ("LAPL", "LTVC",
    "TCATC"), which the PDF's text, having no spaces, cannot object to."""
    page = _equation_page(tmp_path, scan=True)
    first = r"MPL = \Delta Q / \Delta L PL = Q / L TVC + TFC = TC ATC = TC / Q"
    theirs = (r"\text{MPL} = \Delta Q / \Delta \text{LAPL} = Q / \text{LTVC} + "
              r"\text{TFC} = \text{TCATC} = \text{TC} / Q")
    items = [_equation(first)]
    out, changed, differ = vtb.mend_equations(items, page, _Second("$$ " + theirs + " $$"))
    assert out == items and changed == [] and len(differ) == 1


# ── an account that outlasts the build's log ─────────────────────────────────

def test_changes_and_differences_are_written_out_beside_the_readings(tmp_path):
    from codicology import second as second_
    cache = second_.SecondCache(str(tmp_path / "7-abc.second.gz"))
    opinion = vtb.SecondOpinion(_Reader(), cache)
    assert opinion.sheet_path() == str(tmp_path / "7-abc.second.txt")
    opinion.changed.append(("page_0034", "equation", "(1+2/100)", "(1+20/100)"))
    opinion.listed.append(("page_0017", "(Abpc+Bbpb)(1+r)=Bpk",
                           "(Abpa+Bbpb)(1+r)=Bpb"))
    path = opinion.write_sheet()
    said = open(path, encoding="utf-8").read()
    assert "Printed as the second reader has it (1):" in said
    assert "page_0034  equation" in said and "now: (1+20/100)" in said
    assert "The readers differ, and the first is printed (1):" in said
    assert "second: (Abpa+Bbpb)(1+r)=Bpb" in said


def test_a_build_with_nothing_to_say_takes_the_old_account_away(tmp_path):
    from codicology import second as second_
    stale = tmp_path / "7-abc.second.txt"
    stale.write_text("an earlier build's account")
    opinion = vtb.SecondOpinion(_Reader(), second_.SecondCache(
        str(tmp_path / "7-abc.second.gz")))
    assert opinion.write_sheet() is None and not stale.exists()


def test_without_a_cache_there_is_nowhere_to_write_it(tmp_path):
    opinion = vtb.SecondOpinion(_Reader())
    opinion.listed.append(("page_0001", "a", "b"))
    assert opinion.sheet_path() is None and opinion.write_sheet() is None


# ── found on review ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("first, printed, whole", [
    ("x=10", "x=10", True),
    ("x=10", "x=10(3)", True),              # the equation's number follows
    ("x=10", "Thusx=10,where", True),
    ("x=10", "x=100", False),               # it stops a digit short
    ("5x=10", "25x=10", False),             # or starts a digit late
    ("x=10", "x=100andx=10", True),         # the whole of it is there as well
    ("", "x=10", False),
])
def test_a_reading_that_is_a_piece_of_a_printed_number_is_not_borne_out(
        first, printed, whole):
    assert vtb._borne_out(first, printed) is whole


def test_an_equation_that_lost_its_last_digit_is_put_to_the_second_reader(tmp_path):
    """The gate asked only whether the reading was somewhere in the PDF's
    symbols, and one that stops a digit short is."""
    page = _equation_page(tmp_path, printed="r = R(1 - 1/3) = 100")
    short, whole = r"r = R(1 - \frac{1}{3}) = 10", r"r = R(1 - \frac{1}{3}) = 100"
    assert vtb._flat_tex(short) in "r=R(1-1/3)=100", "the old gate let it by"
    second = _Second("$$ " + whole + " $$")
    out, changed, _ = vtb.mend_equations([_equation(short)], page, second)
    assert second.asked == 1 and changed == [("r=R(1-1/3)=10", "r=R(1-1/3)=100")]
    assert out[0].html == f'<math display="block">{whole}</math>'


def test_one_equation_of_the_second_readers_answers_one_block(tmp_path):
    """Two blocks the first reader read alike, and one equation offered:
    it is printed once."""
    page = _equation_page(tmp_path, scan=True)
    items = [_equation(IRON_FIRST), _equation(IRON_FIRST, box=(0.10, 0.52, 0.80, 0.56))]
    out, changed, _ = vtb.mend_equations(items, page,
                                         _Second("$$ " + IRON_SECOND + " $$"))
    assert len(changed) == 1
    assert [IRON_SECOND in it.html for it in out] == [True, False]


def test_a_build_that_asked_nothing_takes_an_old_account_away(tmp_path, monkeypatch):
    """The account was only written, or removed, when the second reader had
    been asked something: a rebuild that no longer needed it left the last
    build's changes described beside a book that no longer has them."""
    from codicology import second as second_
    from test_unvouched import PRINTED, _blocks, _leaf

    class Reads:
        batch_size, name, langs = 4, "fake", ["en"]

        def run(self, images):
            return ["" for _ in images]

        def run_items(self, images):
            return [_blocks(PRINTED, 0.995) for _ in images]

    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    monkeypatch.setattr(vtb, "_classical_text", lambda path: PRINTED)
    monkeypatch.setattr(vtb, "_classical_word_count",
                        lambda path, min_len=2: len(PRINTED.split()))
    stale = tmp_path / "7-abc.second.txt"
    stale.write_text("an earlier build's account")
    opinion = vtb.SecondOpinion(_Reader(), second_.SecondCache(
        str(tmp_path / "7-abc.second.gz")))
    monkeypatch.setattr(vtb, "SECOND_OPINION", opinion)
    vtb.build_epub([_leaf(tmp_path)], str(tmp_path / "book.epub"), Reads(),
                   "Print", False, dedupe=False, drop_blank=False)
    assert opinion.reader.read_count == 0, "a page of prose is not put to it"
    assert not stale.exists()

