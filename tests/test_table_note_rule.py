"""A table's own notes are not the page's feet.

The Modern Corporation and Private Property sets a table, its notes straight
under it, and the page's prose going on below them, with the page's own
footnotes at the foot. The layout calls both kinds of note Footnote. The rule
the build draws above a page's labelled feet went above the table's notes, so
the table lost its notes and the prose under them was taken for the foot:
neither the table's calls nor the page's linked.
"""
from codicology.model import PageItem
from codicology.notes import link_footnotes
from codicology.reading import _table_note

TABLE = PageItem(html="<table><tr><td>Business wealth</td><td>38.0% <sup>1</sup></td></tr>"
                      "</table>", label="Table")
TABLE_NOTE = PageItem(html="<p><sup>1</sup> Unadjusted for unconsolidated income tax "
                           "returns.</p>", label="Footnote")
PROSE = PageItem(html="<p>It is apparent from these figures that a considerable "
                      "portion of the wealth is concentrated.<sup>16</sup></p>", label="Text")
FOOT = PageItem(html="<p><sup>16</sup> The Conference Board Bulletin, No. 38.</p>",
                label="Footnote")
HEAD = PageItem(html="<p>THE CONCENTRATION OF ECONOMIC POWER 33</p>", label="PageHeader",
                is_furniture=True)


def test_a_footnote_under_a_table_with_prose_after_it_is_the_tables():
    items = [HEAD, PROSE, TABLE, TABLE_NOTE, PROSE, FOOT]
    assert _table_note(items, 3)
    assert not _table_note(items, 5)


def test_notes_at_the_foot_under_a_table_are_the_pages():
    """With nothing after them they may be either; they stay the page's feet,
    as they always were."""
    items = [PROSE, TABLE, TABLE_NOTE, FOOT]
    assert not _table_note(items, 2) and not _table_note(items, 3)


def test_a_footnote_not_under_a_table_is_not_a_tables():
    items = [PROSE, TABLE_NOTE, PROSE, FOOT]
    assert not _table_note(items, 1)


def test_the_run_under_a_table_is_the_tables_whole():
    second = PageItem(html="<p><sup>2</sup> Estimated.</p>", label="Footnote")
    items = [TABLE, TABLE_NOTE, second, PROSE, FOOT]
    assert _table_note(items, 1) and _table_note(items, 2) and not _table_note(items, 4)


def test_with_the_rule_above_the_pages_feet_both_kinds_link():
    """The page as it is now rendered: table, its note, prose, the rule, the
    page's foot."""
    body = (PROSE.html.replace("16", "15") + TABLE.html + TABLE_NOTE.html + PROSE.html
            + "<hr/>" + FOOT.html.replace("16", "15") + FOOT.html)
    bodies = [body]
    stats = link_footnotes(bodies, allow_numbered=True)
    assert stats["table"] == 1
    assert 'href="#tn-p0-t0-1"' in bodies[0]
    assert stats["linked"] == 3


def test_a_tables_note_marked_with_a_symbol_links():
    """The Horn of Africa p46: the asterisk after a cell, and its note straight
    under the table, with the page's prose going on below."""
    body = ("<table><tr><td>Eritrea</td><td>The Democratic Movement for the Liberation "
            "of Kunama (DMLK).*</td></tr></table>"
            "<p>*On the basis of the casualty threshold, these are not civil wars.</p>"
            "<p>Nevertheless, the conflict areas are among the most deprived.</p>")
    bodies = [body]
    stats = link_footnotes(bodies, allow_numbered=True)
    assert stats["table"] == 1
    assert '(DMLK).<sup><a epub:type="noteref" id="tnref-p0-t0-s1" href="#tn-p0-t0-s1">*</a></sup>' \
        in bodies[0]
    assert '<p id="tn-p0-t0-s1"><a href="#tnref-p0-t0-s1">*</a>On the basis' in bodies[0]


def test_a_symbol_used_elsewhere_on_the_page_is_left_alone():
    body = ("<p>Asterisks* mark officers detailed only in some situations.</p>"
            "<table><tr><td>Journal*</td></tr></table>"
            "<p>* Kept by the adjutant.</p><p>More prose.</p>")
    bodies = [body]
    assert link_footnotes(bodies, allow_numbered=True)["table"] == 0
