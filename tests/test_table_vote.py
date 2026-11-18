"""A scanned table's figures that two readers read otherwise, put to a third.

On a scan nothing printed says which reading of a cell is the page's. The
first two readers agree on most scanned tables; where they read a figure
otherwise, the third reader is shown the table's block, and the figure two
of the three hold is printed.
"""
import json

from codicology import pipeline as vtb
from codicology.opinions import (_page_reference, _read_otherwise, _swap_figure,
                                 vote_scanned_tables)
from collections import Counter

YEARS = ("<table><tr><th>Years</th><th>Exports</th></tr>"
         "<tr><td>1712</td><td>£11,384,427</td></tr>"
         "<tr><td>1743</td><td>14,623,653</td></tr>"
         "<tr><td>1744</td><td>11,429,628</td></tr></table>")


def _table(rows):
    return ("<table><tr><th>Years</th><th>Exports</th></tr>"
            + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows)
            + "</table>")


SECOND = _table([("1742", "£11,384,427"), ("1743", "14,623,653"),
                 ("1744", "11,429,628")])


class _Second:
    def __init__(self, reading):
        self._reading = reading

    def reading(self, page_path):
        return self._reading


class _Third:
    def __init__(self, table):
        self._table, self.asked = table, 0

    def table(self, page_path, box):
        self.asked += 1
        return self._table


def _scan(tmp_path, scan=True):
    page = tmp_path / "page_0192.png"
    page.write_bytes(b"")
    (tmp_path / "page_0192.png.source.json").write_text(
        json.dumps({"pdf": "book.pdf", "page": 191, "scan": scan}))
    return str(page)


def _items(table_html=YEARS):
    return [vtb.PageItem(html="<p>LLOYD'S DURING THE WAR.</p>", label="Text",
                         box=(0.2, 0.05, 0.8, 0.08)),
            vtb.PageItem(html=table_html, label="Table", box=(0.1, 0.1, 0.9, 0.5))]


def test_a_figure_the_second_and_third_read_alike_is_printed(tmp_path):
    """The History of Lloyd's p192: a smeared year column, read 1712 for the
    printed 1742."""
    third = _Third(SECOND)
    out, changed, ties = vote_scanned_tables(_items(), _scan(tmp_path),
                                             _Second(SECOND), third)
    assert changed == [("1712", "1742")] and ties == []
    assert "<td>1742</td>" in out[1].html and "1712" not in out[1].html
    assert out[1].box == _items()[1].box and out[0] == _items()[0]


def test_a_figure_the_third_reads_as_the_first_stays(tmp_path):
    out, changed, ties = vote_scanned_tables(_items(), _scan(tmp_path),
                                             _Second(SECOND), _Third(YEARS))
    assert changed == [] and out == _items()
    assert ties == [("1712", "1742", "the third reader holds the first's")]


def test_with_both_or_neither_the_first_stays(tmp_path):
    both = _table([("1712", "1742"), ("1743", "1744")])
    _, changed, ties = vote_scanned_tables(_items(), _scan(tmp_path),
                                           _Second(SECOND), _Third(both))
    assert changed == [] and ties[0][2] == "the third reader holds both"
    _, changed, ties = vote_scanned_tables(_items(), _scan(tmp_path),
                                           _Second(SECOND), _Third(""))
    assert changed == [] and ties[0][2] == "the third reader holds neither"


def test_a_typeset_page_is_left_to_its_own_pdf(tmp_path):
    third = _Third(SECOND)
    out, changed, _ = vote_scanned_tables(_items(), _scan(tmp_path, scan=False),
                                          _Second(SECOND), third)
    assert changed == [] and third.asked == 0


def test_a_page_not_described_is_not_taken_for_a_scan(tmp_path):
    page = tmp_path / "page_0001.png"
    page.write_bytes(b"")
    third = _Third(SECOND)
    assert vote_scanned_tables(_items(), str(page), _Second(SECOND), third)[1] == []
    assert third.asked == 0


def test_the_third_reader_is_asked_only_where_the_two_differ(tmp_path):
    third = _Third(SECOND)
    vote_scanned_tables(_items(SECOND), _scan(tmp_path), _Second(SECOND), third)
    assert third.asked == 0


def test_a_figure_printed_twice_is_no_one_cell(tmp_path):
    first = _table([("0830", "1000"), ("0830", "1000"), ("1100", "1200")])
    second = _table([("0930", "1000"), ("0930", "1000"), ("1100", "1200")])
    _, changed, ties = vote_scanned_tables(_items(first), _scan(tmp_path),
                                           _Second(second), _Third(second))
    assert changed == [] and ties == [("0830", "0930", "printed more than once: no one cell")] * 2


def test_another_table_of_the_page_is_not_a_reading_of_this_one(tmp_path):
    other = _table([("21", "34"), ("55", "89"), ("144", "233")])
    third = _Third(SECOND)
    _, changed, _ = vote_scanned_tables(_items(), _scan(tmp_path), _Second(other), third)
    assert changed == [] and third.asked == 0


def test_cells_read_otherwise_are_one_digit_apart_and_of_a_length():
    first = Counter({"1712": 1, "3.35": 1, "1928": 1, "7": 1, "100": 1})
    second = Counter({"1742": 1, "3.85": 1, "19282": 1, "2": 1, "100": 1})
    assert _read_otherwise(first, second) == [("1712", "1742"), ("3.35", "3.85")]


def test_a_swapped_figure_keeps_its_commas():
    assert _swap_figure("<td>6,909,090</td>", "6909090", "6969090") == "<td>6,969,090</td>"
    assert _swap_figure('<td class="n1712">1712</td>', "1712", "1742") == \
        '<td class="n1712">1742</td>'
    assert _swap_figure("<td>1712</td><td>1712</td>", "1712", "1742") is None


def test_a_contents_lines_page_number_is_left_to_the_contents(tmp_path):
    """Kill or Get Killed p17: the old-style "219" read "229" by two of the
    three readers. The contents finds each title on the page it names."""
    first = _table([("8. COMMAND AND CONTROL ..... 219", ""),
                    ("9. RAIDS AND AMBUSHES ..... 244", ""),
                    ("10. PATROLS ..... 261", "")])
    second = first.replace("219", "229")
    third = _Third(second)
    out, changed, ties = vote_scanned_tables(_items(first), _scan(tmp_path),
                                             _Second(second), third)
    assert changed == [] and out[1].html == first and third.asked == 0
    assert ties == [("219", "229", "a contents line's page number")]


def test_a_figure_after_a_single_stop_is_not_a_page_number():
    assert not _page_reference("Canteen, w/cup & cover. 3.35", "3.35")
    assert _page_reference("GLOSSARY ............ 102", "102")
    assert _page_reference("Index . . . . 194", "194")
