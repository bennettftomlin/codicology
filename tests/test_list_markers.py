"""A list must not draw a marker the item has already printed.

The recogniser hands back a real <ol> whose items still carry the number the
page showed — "<li>1. Suppose the multiplier…" — so the reader adds its own
and the item reads "1. 1. Suppose". Principles of Economics shipped 813 of
those and 450 doubled bullets. The recogniser already writes
list-style-type: none on some lists, which is exactly why other books came
out clean; it is simply not consistent about it.

What is suppressed is the marker the READER would draw. The one in the text
is what the page printed and it stays — a list may begin at five, or number
by citation rather than position, and stripping it would delete something
the book actually says.
"""


def test_a_self_numbering_list_stops_drawing_its_own(vtb):
    bodies = ["<ol><li>1. Suppose the multiplier is 1.5.</li>"
              "<li>2. Explain the concept.</li>"
              "<li>3. Understand the three questions.</li></ol>"]
    assert vtb.normalize_list_markers(bodies) == 1
    assert 'list-style-type: none' in bodies[0]
    assert "1. Suppose the multiplier" in bodies[0], "the printed number was lost"


def test_a_self_bulleting_list_stops_too(vtb):
    bodies = ["<ul><li>• Keep the poncho dry.</li><li>• Check the seams.</li></ul>"]
    assert vtb.normalize_list_markers(bodies) == 1
    assert 'list-style-type: none' in bodies[0]
    assert "•" in bodies[0]


def test_a_plain_list_is_left_to_draw_its_markers(vtb):
    bodies = ["<ol><li>Suppose the multiplier is 1.5.</li>"
              "<li>Explain the concept.</li></ol>"]
    assert vtb.normalize_list_markers(bodies) == 0
    assert "list-style" not in bodies[0]


def test_a_list_the_recogniser_already_styled_is_untouched(vtb):
    body = ('<ol style="list-style-type: none;"><li>1. First.</li>'
            '<li>2. Second.</li></ol>')
    bodies = [body]
    assert vtb.normalize_list_markers(bodies) == 0
    assert bodies[0] == body


def test_one_item_opening_with_a_number_is_not_enough(vtb):
    """A single item beginning "1980 the treaty was signed" — or one genuine
    numbered reference among prose items — must not restyle the list."""
    bodies = ["<ol><li>3. A numbered citation.</li>"
              "<li>Ordinary prose item.</li>"
              "<li>Another ordinary item.</li>"
              "<li>And a fourth.</li></ol>"]
    assert vtb.normalize_list_markers(bodies) == 0


def test_a_four_digit_year_is_not_a_marker(vtb):
    bodies = ["<ol><li>1980 was a year of change.</li>"
              "<li>1991 brought the collapse.</li></ol>"]
    assert vtb.normalize_list_markers(bodies) == 0


def test_no_text_is_ever_removed(vtb):
    bodies = ["<ol><li>1. First point.</li><li>2. Second point.</li></ol>",
              "<ul><li>• A bullet.</li><li>• Another.</li></ul>"]
    before = [vtb._strip_tags(b) for b in bodies]
    vtb.normalize_list_markers(bodies)
    assert [vtb._strip_tags(b) for b in bodies] == before


def test_an_existing_style_attribute_survives(vtb):
    bodies = ['<ol class="tight"><li>1. First.</li><li>2. Second.</li></ol>']
    vtb.normalize_list_markers(bodies)
    assert 'class="tight"' in bodies[0]
    assert 'list-style-type: none' in bodies[0]


def test_nested_lists_are_handled_separately(vtb):
    """The inner list marks itself, the outer does not; only the inner is
    restyled, and the outer's structure survives intact."""
    bodies = ["<ul><li>Plain outer item"
              "<ul><li>• inner one</li><li>• inner two</li></ul>"
              "</li><li>Another plain outer</li></ul>"]
    vtb.normalize_list_markers(bodies)
    assert bodies[0].count("<ul") == 2
    assert bodies[0].count("</ul>") == 2
    assert bodies[0].count("list-style-type: none") == 1


def test_an_unclosed_list_does_not_break_the_pass(vtb):
    bodies = ["<ol><li>1. First.</li><li>2. Second.</li>"]
    vtb.normalize_list_markers(bodies)          # must not raise
    assert "1. First." in bodies[0]


def test_lists_across_many_pages_are_each_judged_on_their_own(vtb):
    bodies = ["<ol><li>1. a</li><li>2. b</li></ol>",
              "<ol><li>plain</li><li>also plain</li></ol>",
              "<ul><li>• x</li><li>• y</li></ul>"]
    assert vtb.normalize_list_markers(bodies) == 2
    assert "list-style" in bodies[0]
    assert "list-style" not in bodies[1]
    assert "list-style" in bodies[2]


def test_a_stray_bullet_on_one_item_is_removed(vtb):
    """The shape that actually shipped: the recogniser caught the glyph on
    the first item and missed it on the rest, so that item alone rendered
    "• •". Suppressing the list's markers would strip the other three of
    theirs, so the stray goes instead."""
    bodies = ["<ul><li>• Economics is a social science.</li>"
              "<li> Scarcity implies we must give up one alternative.</li>"
              "<li> A good that is not scarce is a free good.</li>"
              "<li> The opportunity cost is the value forgone.</li></ul>"]
    vtb.normalize_list_markers(bodies)
    assert "•" not in bodies[0], "the stray bullet survived"
    assert "list-style" not in bodies[0], "the other items lost their markers"
    assert "Economics is a social science." in bodies[0]


def test_a_stray_number_is_left_alone(vtb):
    """A lone number may mean something a bullet cannot — a numbered item
    among unnumbered ones — so it is reported by the sweep, never removed."""
    bodies = ["<ol><li>3. A numbered citation.</li>"
              "<li>Ordinary item.</li><li>Another ordinary item.</li>"
              "<li>And a fourth.</li></ol>"]
    vtb.normalize_list_markers(bodies)
    assert "3. A numbered citation." in bodies[0]


def test_removing_a_stray_bullet_loses_no_word(vtb):
    bodies = ["<ul><li>• First point here.</li><li> Second point.</li>"
              "<li> Third point.</li></ul>"]
    before = vtb._strip_tags(bodies[0]).replace("•", "").split()
    vtb.normalize_list_markers(bodies)
    assert vtb._strip_tags(bodies[0]).split() == before


def test_a_fully_bulleted_list_still_takes_suppression_not_stripping(vtb):
    bodies = ["<ul><li>• One.</li><li>• Two.</li><li>• Three.</li></ul>"]
    vtb.normalize_list_markers(bodies)
    assert "list-style-type: none" in bodies[0]
    assert bodies[0].count("•") == 3, "the printed bullets were stripped"
