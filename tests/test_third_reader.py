"""A third reader's vote on a scanned equation.

Where two readers differ on an equation of a scan, nothing printed can say
which is the page. A third reader is shown the block and asked for the
formula, and the reading that has another reader on its side at every point
where it differs is the one printed: the second reader's, the third's own,
or the first's as it stood. Tried on the fourteen equations of Sraffa's
Production of Commodities on which the first two differ, that is the
printed equation in twelve, and in none is it worse than what stood.
"""
import gzip
import json

import pytest
from PIL import Image

from codicology import doctor, pipeline as vtb, second, third
from test_second_opinion import EQ_BOX, _Second, _equation, _equation_page

# Sraffa, page 100: the first reader has an A for the Λ, the second a p for
# the ρ, and the third has both symbols
RENT_FIRST = r"(A_{c_1}p_a + \dots + K_{c_1}p_k)(1+r) + L_{c_1}w + A_1\rho_1 = C_{(1)}p_c"
RENT_SECOND = r"(A_{c_1}p_a + \dots + K_{c_1}p_k)(1+r) + L_{c_1}w + \Lambda_1 p_1 = C_{(1)}p_c"
RENT_THIRD = (r"\left(A _ {c _ {1}} p _ {a} +. + K _ {c _ {1}} p _ {k}\right) (1 + r) "
              r"+ L _ {c _ {1}} w + \Lambda_ {1} \rho_ {1} = C _ {(1)} p _ {c}")
# page 17: a subscript and a price the first reader had wrong
SYSTEM_FIRST = r"(A_b p_c + B_b p_b + \dots + K_b p_k)(1+r) = B p_k"
SYSTEM_SECOND = r"(A_b p_a + B_b p_b + \dots + K_b p_k)(1+r) = B p_b"
SYSTEM_THIRD = (r"\left(A _ {b} p _ {a} + B _ {b} p _ {b} + \dots + K _ {b} p _ {k}"
                r"\right) (1 + r) = B p _ {b}")


class _Third:
    """A third reader that answers the same about any block."""
    def __init__(self, answer):
        self.answer, self.asked = answer, 0

    def formula(self, page_path, box):
        self.asked += 1
        return third.formula(self.answer)


def _vote(tmp_path, first, theirs, said, page=None):
    page = page or _equation_page(tmp_path, scan=True)
    voter = _Third(said)
    out, changed, differ = vtb.mend_equations(
        [_equation(first)], page, _Second("$$ " + theirs + " $$"), voter)
    return out[0].html, changed, differ, voter


# ── the reading two of three have at every point ─────────────────────────────

@pytest.mark.parametrize("flats, which", [
    (["Bpk", "Bpb", "Bpb"], 1),                 # the second and the third agree
    (["A1ρ1", "Λ1p1", "Λ1ρ1"], 2),              # each of the others half right
    (["pm0=r", "pm0r", "pmor"], 1),             # the third slips somewhere else
    (["G(g)pg", "G(x)pg", "G(g)pg"], 0),        # the first, with the third
    (["abc", "abd", "abe"], None),              # three readings of one symbol
    (["x=1", "x=1", "x=7"], 0),                 # a third that stands alone
    (["same", "same", "same"], 0),
])
def test_the_majority_reading(flats, which):
    assert vtb.majority_reading(flats) == which


def test_a_reading_is_the_majority_only_if_backed_at_every_point():
    """The second differs from the first in two places and the third backs
    it in one: there is no reading two readers have throughout."""
    assert vtb.majority_reading(["a+b=c", "x+b=z", "x+b=q"]) is None


# ── three ways of writing one equation ───────────────────────────────────────

def test_three_readers_writing_one_equation_come_out_the_same():
    a = vtb._flat_tex(r"p'_a(A_a q''_a) \frac{(1+r)^n - 1}{r} = \{L_g w\}")
    b = vtb._flat_tex(r"p_a'(A_a q_a'') \frac{(1+r)^{n} - 1}{r} = \left\{L_g w\right\}")
    c = vtb._flat_tex(r"p _ {a} ^ {\prime} \left(A _ {a} q _ {a} ^ {\prime \prime}\right) "
                      r"\frac {(1 + r) ^ {n} - 1}{r} = \left\{L _ {g} w\right\}")
    assert a == b == c == "pa'(Aaqa'')(1+r)n-1/r=(Lgw)"


def test_a_fraction_with_a_power_in_it_is_still_a_fraction():
    assert vtb._flat_tex(r"\frac{r(1+r)^{n}}{(1+r)^{n}-1}") == "r(1+r)n/(1+r)n-1"
    assert vtb._flat_tex(r"\frac{\frac{a}{b}}{c}") == "a/b/c"


def test_the_third_readers_tex_is_set_as_the_first_readers_is():
    assert vtb._tidy_tex(RENT_THIRD) == (
        r"(A_{c_1} p_a + \dots + K_{c_1} p_k) (1 + r) + L_{c_1} w + "
        r"\Lambda_1 \rho_1 = C_{(1)} p_c")
    assert vtb._tidy_tex(r"(9 0 + 3 0) \left(1 + \frac {2 0}{1 0 0}\right) = "
                         r"1 0 8 \mathrm {t}. \mathrm {i r o n}") == (
        r"(90 + 30) (1 + \frac{20}{100}) = 108 \text{t}. \text{iron}")
    assert vtb._tidy_tex(r"p _ {a} ^ {\prime \prime}") == "p_a''"


def test_the_lines_of_an_array_are_its_equations():
    assert vtb._tex_rows(r"\begin{array}{l} a = b \\ c = d \\ \end{array}") == [
        "a = b", "c = d"]
    assert vtb._tex_rows("a = b") == ["a = b"]


# ── the vote, in the rule ────────────────────────────────────────────────────

def test_where_the_third_sides_with_the_second_the_second_is_printed(tmp_path):
    html_, changed, differ, voter = _vote(tmp_path, SYSTEM_FIRST, SYSTEM_SECOND,
                                          "$$ " + SYSTEM_THIRD + " $$")
    assert html_ == f'<math display="block">{SYSTEM_SECOND}</math>'
    assert differ == [] and voter.asked == 1
    (was, now, why), = changed
    assert (was, now) == ("(Abpc+Bbpb+...+Kbpk)(1+r)=Bpk",
                          "(Abpa+Bbpb+...+Kbpk)(1+r)=Bpb")
    assert why == "the third reader sides with the second"


def test_where_each_of_the_others_is_half_right_the_thirds_own_is_printed(tmp_path):
    html_, changed, differ, _ = _vote(tmp_path, RENT_FIRST, RENT_SECOND,
                                      "\n$$\n" + RENT_THIRD + "\n$$")
    assert html_ == ('<math display="block">(A_{c_1} p_a + \\dots + K_{c_1} p_k) '
                     '(1 + r) + L_{c_1} w + \\Lambda_1 \\rho_1 = C_{(1)} p_c</math>')
    assert differ == [] and changed[0][2].startswith("the third reader's own")


def test_where_the_third_sides_with_the_first_it_stands_and_is_still_listed(tmp_path):
    """Listed, because two readers can share a mistake: of the three such
    equations in Sraffa, two are still wrong."""
    html_, changed, differ, _ = _vote(tmp_path, SYSTEM_SECOND, SYSTEM_FIRST,
                                      "$$ " + SYSTEM_THIRD + " $$")
    assert html_ == f'<math display="block">{SYSTEM_SECOND}</math>'
    assert changed == []
    assert differ[0][2] == "the third reader sides with the first"


def test_three_readings_of_one_symbol_settle_nothing(tmp_path):
    html_, changed, differ, _ = _vote(
        tmp_path, SYSTEM_FIRST, SYSTEM_SECOND,
        "$$ " + SYSTEM_THIRD.replace("B p _ {b}", "B p _ {x}") + " $$")
    assert html_ == f'<math display="block">{SYSTEM_FIRST}</math>'
    assert changed == [] and "the third reads" in differ[0][2]


def test_a_majority_with_fewer_figures_than_the_first_is_not_printed(tmp_path):
    """Principles of Microeconomics, page 634: the 1 of "1 + r" stood in a
    different place in two readings, so the one reading that lacked it had
    a reader with it at each point. The PDF had the answer there. On a scan
    nothing would."""
    first = r"NPV_0 = R_0 - C_0 + \frac{R_1 - C_1}{1 + r}"
    html_, changed, differ, _ = _vote(tmp_path, first,
                                      r"NPV_0 = R_0 - C_0 + \frac{R_1 - C_1}{r}",
                                      r"$$ NPV_0 = R_0 - C_0 + \frac{R_1 - C_1}{r} $$")
    assert html_ == f'<math display="block">{first}</math>' and changed == []
    assert "fewer figures" in differ[0][2]


def test_a_third_reader_with_nothing_to_say_leaves_the_rule_as_it_was(tmp_path):
    """No answer, or an answer about something else: a difference in
    figures alone still goes to the second reader, and any other is listed."""
    page = _equation_page(tmp_path, scan=True)
    for said in (None, "$$ E = mc^2 $$"):
        html_, changed, differ, _ = _vote(tmp_path, SYSTEM_FIRST, SYSTEM_SECOND,
                                          said, page)
        assert changed == [] and len(differ) == 1 and len(differ[0]) == 2
    iron_first = r"(90 + 30 + 30)(1 + \frac{2}{100}) = 108 \text{ t. iron}"
    iron_second = r"(90 + 30 + 30)(1 + \frac{20}{100}) = 108 \text{ t. iron}"
    html_, changed, differ, _ = _vote(tmp_path, iron_first, iron_second, None, page)
    assert html_ == f'<math display="block">{iron_second}</math>' and len(changed[0]) == 2


def test_it_is_not_asked_where_the_first_two_agree(tmp_path):
    html_, changed, differ, voter = _vote(tmp_path, SYSTEM_SECOND, SYSTEM_SECOND,
                                          "$$ " + SYSTEM_THIRD + " $$")
    assert voter.asked == 0 and changed == [] and differ == []


def test_on_a_typeset_page_the_pdf_is_asked_and_the_third_reader_is_not(tmp_path):
    page = _equation_page(tmp_path)
    voter = _Third("$$ x $$")
    vtb.mend_equations(
        [_equation(r"(90 + 30 + 30)(1 + \frac{2}{100}) = 108 \text{ t. iron}")], page,
        _Second(r"$$ (90 + 30 + 30)(1 + \frac{20}{100}) = 108 \text{ t. iron} $$"), voter)
    assert voter.asked == 0


def test_a_block_of_several_lines_is_matched_line_by_line(tmp_path):
    """One block, two equations in it, and the third reader's array of both:
    each equation is voted on with its own line."""
    page = _equation_page(tmp_path, scan=True)
    other = r"(A_k p_a + B_k p_b + \dots + K_k p_k)(1+r) = K p_k"
    block = vtb.PageItem(
        html=f'<math display="block">{SYSTEM_FIRST}</math>'
             f'<math display="block">{other}</math>', label="Formula", box=EQ_BOX)
    said = (r"$$ \begin{array}{l} " + SYSTEM_THIRD + r" \\ " + other + r" \\ \end{array} $$")
    out, changed, differ = vtb.mend_equations(
        [block], page, _Second(f"$$ {SYSTEM_SECOND} $$\n\n$$ {other} $$"), _Third(said))
    assert out[0].html == (f'<math display="block">{SYSTEM_SECOND}</math>'
                           f'<math display="block">{other}</math>')
    assert len(changed) == 1 and differ == []


# ── found on review ──────────────────────────────────────────────────────────

def test_cases_and_matrices_are_not_cut_at_their_rows():
    """Their rows are parts of one thing. Cut there, the first row carried
    the \\begin and the last the \\end, and one of them was printed."""
    whole = r"\begin{cases} L_1 = 2 x + 3 y \\ z = 4 \end{cases}"
    assert vtb._tex_rows(whole) == [whole]
    assert vtb._tex_rows(r"\begin{aligned}{x} &= 1 \end{aligned}") == ["{x} &= 1"]


@pytest.mark.parametrize("tex, alone", [
    (r"a_{b} + \{c\}", True),
    (r"\begin{cases} a \\ b \end{cases}", True),
    (r"\begin{cases} L_1 = 2x", False),
    (r"\left( a + b", False),
    (r"a } b {", False),
    (r"\frac{a}{b", False),
])
def test_what_is_printed_from_the_third_reader_closes_what_it_opens(tex, alone):
    assert vtb._stands_alone(tex) is alone


def test_a_third_reading_not_fit_to_print_is_not_printed(tmp_path):
    """It is the majority, and it is half an environment: the equation is
    listed and the first reader's stands."""
    broken = RENT_THIRD.replace(r"\left(A", r"\begin{cases} \left(A")
    html_, changed, differ, _ = _vote(tmp_path, RENT_FIRST, RENT_SECOND,
                                      "$$ " + broken + " $$")
    assert html_ == f'<math display="block">{RENT_FIRST}</math>'
    assert changed == [] and len(differ) == 1


def test_the_bracket_that_is_not_there_leaves_no_full_stop():
    assert vtb._tidy_tex(r"\left. L_1 = 2 x + 3 y \right.") == "L_1 = 2 x + 3 y"
    assert vtb._tidy_tex(r"\left\{ a + b \right.") == r"\{ a + b"
    assert vtb._tidy_tex(r"\left\langle x \right\rangle") == r"\langle x \rangle"


def test_upright_letters_are_text_and_upright_mathematics_is_not():
    assert vtb._tidy_tex(r"\mathrm {i r o n}") == r"\text{iron}"
    assert vtb._tidy_tex(r"\mathrm {t}.") == r"\text{t}."
    assert vtb._tidy_tex(r"\mathrm{\Delta} t") == r"\mathrm{\Delta} t"
    assert vtb._tidy_tex(r"\mathrm{x^2}") == r"\mathrm{x^2}"
    assert vtb._tidy_tex(r"\text{a b}") == r"\text{a b}", "words keep their space"


def test_a_full_stop_that_is_a_full_stop_stays_one():
    assert vtb._tidy_tex("x = 3. + y") == "x = 3. + y"
    assert vtb._tidy_tex("1.5 + 2") == "1.5 + 2"
    assert vtb._tidy_tex("a +. + b") == r"a + \dots + b"


def test_a_rows_end_inside_a_fraction_does_not_hide_its_brace():
    assert vtb._flat_tex(r"\frac{a \\}{b}") == "a/b"
    assert vtb._flat_tex(r"\frac{\{a\}}{b}") == "(a)/b"


def test_a_vote_that_settles_nothing_leaves_a_dropped_figure_to_the_second(tmp_path):
    """Three readings, no two alike. Before there was a third reader the
    second's figure was taken where figures were all that differed, and a
    third that has only muddied it does not take that away."""
    first = r"(90 + 30 + 30)(1 + \frac{2}{100}) = 108 \text{ t. iron}"
    theirs = r"(90 + 30 + 30)(1 + \frac{20}{100}) = 108 \text{ t. iron}"
    third_ = r"$$ (90 + 30 + 30)(1 + \frac{26}{100}) = 108 \text{ t. iron} $$"
    html_, changed, differ, voter = _vote(tmp_path, first, theirs, third_)
    assert voter.asked == 1 and differ == []
    assert html_ == f'<math display="block">{theirs}</math>' and len(changed[0]) == 2


def test_where_the_third_sides_with_the_first_on_a_figure_it_is_not_taken(tmp_path):
    """Two readers have 2/100. The second's 20 may be the one that was
    supplied."""
    first = r"(90 + 30 + 30)(1 + \frac{2}{100}) = 108 \text{ t. iron}"
    theirs = r"(90 + 30 + 30)(1 + \frac{20}{100}) = 108 \text{ t. iron}"
    html_, changed, differ, _ = _vote(tmp_path, first, theirs, "$$ " + first + " $$")
    assert html_ == f'<math display="block">{first}</math>' and changed == []
    assert differ[0][2] == "the third reader sides with the first"


def test_one_line_of_the_third_readers_is_printed_once(tmp_path):
    """Two equations of one block that the first reader read alike, and one
    line from the third: it answers for the first of them only."""
    page = _equation_page(tmp_path, scan=True)
    block = vtb.PageItem(
        html=f'<math display="block">{RENT_FIRST}</math>'
             f'<math display="block">{RENT_FIRST}</math>', label="Formula", box=EQ_BOX)
    out, changed, differ = vtb.mend_equations(
        [block], page, _Second(f"$$ {RENT_SECOND} $$\n\n$$ {RENT_SECOND} $$"),
        _Third("$$ " + RENT_THIRD + " $$"))
    assert out[0].html.count(r"\Lambda_1 \rho_1") == 1 and len(changed) == 1


def test_a_build_says_how_many_equations_each_reader_set(tmp_path):
    opinion = vtb.SecondOpinion(_Reader())
    opinion.consulted.add("a page")
    opinion.asked = 1
    opinion.equations, opinion.by_third = 9, 2
    assert opinion.summary() == (
        "second reader: asked about 1 page(s); 9 equation(s) set as it reads "
        "them, and 2 equation(s) as the third reader reads them")


def test_the_account_says_how_each_vote_went(tmp_path):
    opinion = vtb.SecondOpinion(_Reader(), second.SecondCache(
        str(tmp_path / "7-abc.second.gz")))
    opinion.changed.append(("page_0017", "equation", "Bpk", "Bpb",
                            "the third reader sides with the second"))
    opinion.listed.append(("page_0100", "Anρn", "Λnpn",
                           "the third reader sides with the first"))
    said = open(opinion.write_sheet(), encoding="utf-8").read()
    assert "      (the third reader sides with the second)" in said
    assert "      (the third reader sides with the first)" in said


# ── the asking ───────────────────────────────────────────────────────────────

class _Reader:
    def __init__(self, answer="\n$$\nE = m c ^ {2}\n$$"):
        self.answer, self.sizes = answer, []

    def read(self, image):
        self.sizes.append(image.size)
        return self.answer

    def close(self):
        pass


def _page(tmp_path):
    path = tmp_path / "page_0000.png"
    Image.new("RGB", (1000, 2000), "white").save(path)
    return str(path)


def test_the_block_is_what_the_third_reader_is_shown(tmp_path):
    reader = _Reader()
    opinion = vtb.ThirdOpinion(reader)
    assert opinion.formula(_page(tmp_path), (0.1, 0.5, 0.9, 0.52)) == "E = m c ^ {2}"
    # the box and a little round it: 800 + 24 wide, 40 + 16 high
    assert reader.sizes == [(824, 56)]


def test_a_block_is_asked_about_once_and_remembered(tmp_path):
    page, box = _page(tmp_path), (0.1, 0.5, 0.9, 0.52)
    kept = str(tmp_path / "book.third.gz")
    reader = _Reader()
    opinion = vtb.ThirdOpinion(reader, second.SecondCache(kept, tag=third.NAME))
    opinion.formula(page, box)
    opinion.formula(page, box)
    opinion.formula(page, (0.1, 0.6, 0.9, 0.62))        # another block of the page
    assert len(reader.sizes) == 2
    assert opinion.summary() == ("third reader: asked about 2 equation(s) on "
                                 "which the first two differ")
    opinion.close()
    with gzip.open(kept, "rt") as fh:
        filed = json.load(fh)
    assert filed["tag"] == third.NAME and len(filed["pages"]) == 2
    again = vtb.ThirdOpinion(_Reader("unused"), second.SecondCache(kept, tag=third.NAME))
    assert again.formula(page, box) == "E = m c ^ {2}"
    assert again.reader.sizes == []


def test_a_third_reader_that_cannot_be_asked_is_asked_again_next_time(tmp_path):
    class Down:
        def read(self, image):
            raise RuntimeError("the server did not start")
        def close(self):
            pass
    page, box = _page(tmp_path), (0.1, 0.5, 0.9, 0.52)
    kept = str(tmp_path / "book.third.gz")
    opinion = vtb.ThirdOpinion(Down(), second.SecondCache(kept, tag=third.NAME))
    assert opinion.formula(page, box) is None
    assert "could not be asked about 1 of them" in opinion.summary()
    opinion.close()
    again = vtb.ThirdOpinion(_Reader(), second.SecondCache(kept, tag=third.NAME))
    assert again.formula(page, box) == "E = m c ^ {2}"


def test_it_is_asked_for_a_formula_and_sent_the_crop_as_it_is(monkeypatch):
    reader = third.ThirdReader(model="m.gguf", mmproj="p.gguf", server="llama-server")
    assert (reader.prompt, reader.long_side, reader.name) == (
        "Formula Recognition:", None, "glm-ocr")


def test_what_comes_back_is_the_tex_without_its_marks():
    assert third.formula("\n$$\na = b\n$$") == "a = b"
    assert third.formula("") is None and third.formula(None) is None


# ── installed or not ─────────────────────────────────────────────────────────

def test_no_third_reader_is_neither_a_problem_nor_a_warning(monkeypatch, tmp_path,
                                                           capsys):
    monkeypatch.setattr(third, "DEFAULT_DIR", str(tmp_path / "none"))
    monkeypatch.delenv(third.MODEL_ENV, raising=False)
    monkeypatch.delenv(third.MMPROJ_ENV, raising=False)
    assert not third.available()
    with pytest.raises(RuntimeError, match="no third reader is installed"):
        third.ThirdReader()
    rep = doctor.report()
    assert rep["third_reader"]["present"] is False
    assert not any("third" in s for s in rep["problems"] + rep["warnings"])
    doctor._human(rep)
    assert "third reader: none" in capsys.readouterr().out


def test_a_third_reader_is_found_in_its_own_place(monkeypatch, tmp_path, capsys):
    for name in ("GLM-OCR-Q8_0.gguf", "mmproj-GLM-OCR-Q8_0.gguf"):
        (tmp_path / name).write_bytes(b"gguf")
    monkeypatch.setattr(third, "DEFAULT_DIR", str(tmp_path))
    monkeypatch.delenv(third.MODEL_ENV, raising=False)
    monkeypatch.delenv(third.MMPROJ_ENV, raising=False)
    monkeypatch.setattr(second, "server_binary", lambda: "/usr/bin/llama-server")
    assert third.model_files() == (str(tmp_path / "GLM-OCR-Q8_0.gguf"),
                                   str(tmp_path / "mmproj-GLM-OCR-Q8_0.gguf"))
    rep = doctor.report()
    assert rep["third_reader"]["present"] and rep["third_reader"]["name"] == "glm-ocr"
    doctor._human(rep)
    assert "votes on a scan's equations" in capsys.readouterr().out
