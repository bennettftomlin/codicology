"""What a printer set at the edge of a page and the layout took for text.

Bound into chapters, a page's edge is the middle of a chapter, and what was
left there stands between two halves of a sentence: a running head under a
manual's paragraph designation, a signature mark at the foot of a
gathering's first page, a page number run into the last line. Each rule is
as narrow as the census of 152 books said it had to be: a loose rule for
running heads was wrong nine times in ten.
"""
import pytest

from codicology import pipeline as vtb


# ── a designation is not a running head ──────────────────────────────────────

@pytest.mark.parametrize("line, worded", [
    ("SWM 6-77", False),                    # the Small Wars Manual's paragraph
    ("AGO 8641A", False),                   # a printer's job number
    ("DOI: 10.21983/P3.0250.1.07", False),
    ("FM 3-06", False),                     # no word of three letters at all
    ("217", False),
    ("NOTES 217", True),                    # a running head of one word, and the page
    ("INTRODUCTION · 5", True),
    ("THE KGB&#x2019;S USE OF THE MVD", True),
    ("… One and All: 1794–", True),
    ("10 THE REBEL PASSION", True),
    ("Index", True),
])
def test_which_furniture_is_a_running_head(line, worded):
    assert vtb._worded_head(line) is worded


def _manual(head_furniture):
    """Six pages of a manual: a designation at the top of each, and under
    it a running head the layout took for a heading."""
    bodies, furniture = [], []
    for k in range(6):
        opening = k == 0
        bodies.append(("<h2>COMPOSITION OF THE FORCE</h2>" if True else "")
                      + f"<p>Paragraph {k} of the section runs on for a line "
                        f"or two of ordinary prose, as a manual's does.</p>")
        furniture.append([head_furniture, str(40 + k)])
    return bodies, furniture


def test_a_running_head_under_a_designation_is_taken_out():
    bodies, furniture = _manual("SWM 6-77")
    removed = vtb.strip_missed_running_heads(bodies, furniture)
    # the section's own heading, on its first page, stays
    assert [i for i, _ in removed] == [1, 2, 3, 4, 5]
    assert "COMPOSITION OF THE FORCE" in bodies[0]
    assert all("COMPOSITION" not in b and "ordinary prose" in b for b in bodies[1:])


def test_a_page_that_has_its_running_head_keeps_the_line_under_it():
    bodies, furniture = _manual("SMALL WARS MANUAL")
    assert vtb.strip_missed_running_heads(bodies, furniture) == []


# ── signature marks ──────────────────────────────────────────────────────────

def _book(feet, every=4):
    """Pages of prose; `feet` are set as the last paragraph of every
    fourth page, in order."""
    bodies = [f"<p>Page {k} of the trial: the evidence was heard and the "
              f"court rose for the day.</p>" for k in range(len(feet) * every)]
    for n, mark in enumerate(feet):
        if mark:
            bodies[n * every] += f"<p>{mark}</p>"
    return bodies


def test_a_printers_signature_marks_are_taken_out():
    bodies = _book(["B", "D", "F", "H", "L"])
    removed = vtb.strip_signature_marks(bodies)
    assert removed == [(0, "B"), (4, "D"), (8, "F"), (12, "H"), (16, "L")]
    assert all(b.endswith("for the day.</p>") for b in bodies)


def test_doubled_letters_follow_the_single_ones():
    bodies = _book(["F", "N", "x", "A A", "B B", "D D"])
    assert [m for _, m in vtb.strip_signature_marks(bodies)] == ["F", "N", "x", "A A", "B B", "D D"]


def test_one_letter_at_the_foot_of_a_page_is_a_paragraph():
    for feet in (["H"], ["K", ""], ["B", "D"], ["B", "B", "B"]):
        bodies = _book(feet)
        was = list(bodies)
        assert vtb.strip_signature_marks(bodies) == [] and bodies == was


def test_what_does_not_climb_is_not_one_of_them():
    bodies = _book(["B", "D", "A", "F", "H"])          # an "A" out of its place
    removed = vtb.strip_signature_marks(bodies)
    assert [m for _, m in removed] == ["B", "D", "F", "H"]
    assert bodies[8].endswith("<p>A</p>")


def test_a_page_that_is_only_a_letter_is_left_alone():
    bodies = ["<p>B</p>", "<p>D</p>", "<p>F</p>", "<p>H</p>"]
    assert vtb.strip_signature_marks(bodies) == []


# ── a page's number run into its last line ───────────────────────────────────

@pytest.mark.parametrize("last, number, left", [
    ("greet Marang and his valuable bag-89", 89, "greet Marang and his valuable bag-"),
    ("go into the family business. The oldest 296", 296, "go into the family business. The oldest"),
    ("the staff had developed a system, 105", 105, "the staff had developed a system,"),
    ("How else could the 188", 188, "How else could the"),
    ("selling at least 9,000,000 francs (about 288", 288, "selling at least 9,000,000 francs (about"),
])
def test_the_pages_own_number_comes_off_the_end_of_its_last_line(last, number, left):
    bodies = [f"<p>An earlier paragraph of the page.</p><p>{last}</p>"]
    assert vtb.strip_fused_folios(bodies, {0: number}) == [(0, number)]
    assert bodies[0] == f"<p>An earlier paragraph of the page.</p><p>{left}</p>"


@pytest.mark.parametrize("last, number", [
    ("The firm was founded in 1893.", 1893),            # a sentence's last figure
    ("as is shown in Table 12", 12),                     # a word that takes a number
    ("see page 296", 296),
    ("allies, conducted 29,200", 200),                  # the end of a larger figure
    ("a rate of 3.200", 200),
    ("go into the family business. The oldest 296", 297),    # another page's number
    ("go into the family business. The oldest 296 son, James", 296),
])
def test_a_figure_of_the_text_is_left(last, number):
    bodies = [f"<p>{last}</p>"]
    assert vtb.strip_fused_folios(bodies, {0: number}) == []
    assert bodies[0] == f"<p>{last}</p>"


def test_a_number_inside_markup_at_the_end_is_still_the_end():
    bodies = ["<p>greet Marang and <i>his valuable bag-89</i></p>"]
    assert vtb.strip_fused_folios(bodies, {0: 89}) == [(0, 89)]
    assert bodies[0] == "<p>greet Marang and <i>his valuable bag-</i></p>"
