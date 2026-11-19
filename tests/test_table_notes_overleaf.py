"""A table's notes carried to the next page, and feet the page answers one
for one (The Modern Corporation and Private Property)."""
from codicology.notes import _answered_feet, _table_notes_overleaf, link_footnotes


def _table(rows):
    return "<table>" + "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in rows) + "</table>"


def test_notes_carried_to_the_head_of_the_next_page_link_to_the_table():
    """Table VII: its note 1 under it at the foot of the page, notes 2 and 3
    opening the next."""
    first = ("<p>TABLE VII: Stockholders</p>"
             + _table([("1901", "8,300 <sup>1</sup>"), ("1910", "9,100 <sup>2</sup>"),
                       ("1929", "40,153 <sup>3</sup>")])
             + "<p><sup>1</sup> The Growth of the Pennsylvania Railroad.</p>")
    second = ("<p><sup>2</sup> Ibid., p. 186.</p><p><sup>3</sup> Ibid., p. 303.</p><hr/>"
              "<p>agement. More important is the indication.</p>")
    bodies = [first, second]
    assert _table_notes_overleaf(bodies) == 2
    assert 'href="page_0001.xhtml#tn-p1-o-2"' in bodies[0]
    assert '<p id="tn-p1-o-3"><sup><a href="page_0000.xhtml#tnref-p0-o-3">3</a></sup>' in bodies[1]


def test_a_table_continued_overleaf_with_its_notes_under_the_second_half():
    first = ("<p>Prose before.</p><p>TABLE II: Adjusted Assets</p>"
             + _table([("Kaiser <sup>1</sup>", "729.6"), ("Richfield <sup>2</sup>", "491.0")]))
    second = ("<p>TABLE II: Adjusted Assets (Continued)</p>" + _table([("Ratio", "49.0")])
              + "<p><sup>1</sup> Compiled from Moody’s Manuals.</p><p><sup>2</sup> Estimate.</p>"
              + "<p>Thus, if concentration is measured by total assets.</p>")
    bodies = [first, second]
    assert link_footnotes(bodies, allow_numbered=True)["table"] == 2
    assert 'id="tn-p1-o-1"' in bodies[1] and 'href="page_0001.xhtml#tn-p1-o-2"' in bodies[0]


def test_a_table_the_page_goes_on_past_keeps_its_own_scope():
    first = _table([("Kaiser <sup>1</sup>", "729.6")]) + "<p>The page goes on.</p>"
    second = "<p><sup>1</sup> Compiled from Moody’s Manuals.</p>"
    assert _table_notes_overleaf([first, second]) == 0


def test_a_number_raised_elsewhere_on_either_page_is_left_alone():
    first = _table([("Kaiser <sup>1</sup>", "1"), ("Richfield <sup>2</sup>", "2"),
                    ("Getty <sup>3</sup>", "3")])
    second = ("<p><sup>1</sup> Moody’s.</p><p><sup>2</sup> Estimate.</p><p><sup>3</sup> Revised.</p>"
              "<p>Prose raising its own note.<sup>2</sup></p>")
    bodies = [first, second]
    assert _table_notes_overleaf(bodies) == 2
    assert 'href="page_0001.xhtml#tn-p1-o-1"' in bodies[0] and "<sup>2</sup></td>" in bodies[0]
    lone = [_table([("Kaiser <sup>1</sup>", "1"), ("Richfield <sup>2</sup>", "2")]),
            "<p><sup>1</sup> Moody’s.</p><p><sup>2</sup> Estimate.</p><p>Again.<sup>2</sup></p>"]
    assert _table_notes_overleaf(lone) == 0      # one of two is too few to settle the scope


def test_feet_holding_a_table_or_a_long_carried_note_are_answered_one_for_one():
    above = "<p>Thus the dividend.<sup>6</sup> In fact.<sup>7</sup></p>"
    below = ("<p>in excess of cost, a long note carried over the page and running on.</p>"
             "<p><sup>6</sup> In its simplest form:</p><p>In 1931 it declares $70,000.</p>"
             "<p><sup>7</sup> For instance:</p><table><tr><td>Continental Baking</td></tr></table>")
    assert _answered_feet(above, below)
    assert not _answered_feet(above + "<p>Again.<sup>6</sup></p>", below)
    assert not _answered_feet(above, "<h2>Part Two</h2>" + below)
    bodies = [above + "<hr/>" + below]
    assert link_footnotes(bodies, allow_numbered=True)["linked"] == 2
