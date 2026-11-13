r"""Inline math the recogniser wrapped, set as the characters it stands for.

Surya writes a printed "7½ inches" as 7<math>\frac{1}{2}</math> inches, and
no EPUB reader renders LaTeX. Stripping the backslash — all normalize_math
did with a name it did not know — put "7frac{1}{2} inches" on the page: 240
fractions in 25 of the 151 books measured, with "14^{circ}F" and "4 times
7" beside them. Every input below is a span one of those books' readings
holds.
"""
import pytest


def flat(vtb, tex, scripts=True):
    return vtb.normalize_math(f"<math>{tex}</math>", scripts=scripts)


@pytest.mark.parametrize("tex, set_as", [
    (r"\frac{1}{2}", "½"),
    (r"7\frac{1}{2}", "7½"),
    (r"1,482\frac{1}{2}", "1,482½"),
    (r"33\frac{1}{3}", "33⅓"),
    (r"3\frac{1}{8}", "3⅛"),
    (r"\frac{1}{10}", "⅒"),
])
def test_a_fraction_with_its_own_character_takes_it(vtb, tex, set_as):
    assert flat(vtb, tex) == set_as


@pytest.mark.parametrize("tex, set_as", [
    (r"\frac{3}{16}", "³⁄₁₆"),
    (r"46\frac{8}{10}", "46⁸⁄₁₀"),
    (r"5\frac{1}{12}", "5¹⁄₁₂"),
])
def test_any_other_fraction_is_raised_over_lowered(vtb, tex, set_as):
    # 46 8/10 would read as forty-six, then eight tenths of something else
    assert flat(vtb, tex) == set_as


@pytest.mark.parametrize("tex, set_as", [
    (r"\frac{\$10}{\text{passenger}}", "$10/passenger"),
    (r"\frac{100 \text{ cc./kg.}}{10} + 50", "(100  cc./kg.)/10 + 50"),
    (r"\frac{.006 \times 50}{20} + 1", "(.006 × 50)/20 + 1"),
    (r"\frac{(1+r)^n - 1}{r}", "((1+r)ⁿ - 1)/r"),
    (r"\frac{\$400}{(40 \text{ passengers})}", "$400/(40  passengers)"),
    (r"n \leq \frac{1}{R}", "n ≤ 1/R"),
    (r"\frac{1}{\frac{1}{2}}", "1/½"),
])
def test_a_fraction_of_words_is_written_on_one_line(vtb, tex, set_as):
    assert flat(vtb, tex) == set_as


@pytest.mark.parametrize("tex, set_as", [
    (r"14^{\circ}\text{F}", "14°F"),
    (r"45^\circ", "45°"),
    (r"-273^\circ\text{ C}", "-273° C"),
    (r"36^{\circ}30'", "36°30'"),
    (r"^{\circ}\text{C}", "°C"),
])
def test_a_raised_ring_is_a_degree_sign(vtb, tex, set_as):
    assert flat(vtb, tex) == set_as


@pytest.mark.parametrize("tex, set_as", [
    (r'68\frac{1}{2}" \times 11\frac{1}{4}"', '68½" × 11¼"'),
    (r"\pm 1\frac{1}{4}", "± 1¼"),
    (r"1 \div 0.1292", "1 ÷ 0.1292"),
    (r"M \rightarrow M'", "M → M'"),
    (r"A, B, \dots, K", "A, B, …, K"),
    (r"\geq 0", "≥ 0"),
])
def test_an_operator_is_its_sign_not_its_name(vtb, tex, set_as):
    assert flat(vtb, tex) == set_as


@pytest.mark.parametrize("tex, set_as", [
    (r"\mu\text{g}", "μg"),
    (r"\text{TH}\beta\text{C-3-COOH}", "THβC-3-COOH"),
    (r"\Delta\text{EKA}", "ΔEKA"),
    (r"\rho_1, \rho_2", "ρ<sub>1</sub>, ρ<sub>2</sub>"),
])
def test_a_name_ends_where_the_recogniser_ended_it(vtb, tex, set_as):
    # unwrapping \text first fuses \mu and g into a name nobody knows
    assert flat(vtb, tex) == set_as


@pytest.mark.parametrize("tex, set_as", [
    ("P_1", "P<sub>1</sub>"),
    (r"\text{CO}_2", "CO<sub>2</sub>"),
    ("x^2", "x²"),
    ("10^{-9}", "10⁻⁹"),
    (r"\$1,000 \times (1 + r)^n", "$1,000 × (1 + r)ⁿ"),
    (r"\text{C}_4\text{H}_{10}\text{N}^+", "C<sub>4</sub>H<sub>10</sub>N⁺"),     # TiHKAL
    (r"^{18}\text{FCH}_2\text{CH}_2\text{I}", "¹⁸FCH<sub>2</sub>CH<sub>2</sub>I"),
    ("Q^*", 'Q<sup class="power">*</sup>'),           # no superscript character for it
    ("P_1^b", 'P<sub>1</sub><sup class="power">b</sup>'),
    (r"[\alpha]_D + 41.8^\circ", "[α]<sub>D</sub> + 41.8°"),
])
def test_subscripts_and_powers_are_set(vtb, tex, set_as):
    assert flat(vtb, tex) == set_as


def test_a_power_is_never_taken_for_a_note_call(vtb):
    """A bare <sup> holding a figure is a call to every linker here; a power
    is written so that none of them can take it for one."""
    for tex in ("x^2", "10^{12}", "a^{2b}", "Q^{24}"):
        set_as = flat(vtb, tex)
        assert not vtb.BODY_MARKER.search(set_as), set_as


def test_an_underscore_outside_the_recognisers_math_is_left_alone(vtb):
    """Most underscores on the shelf are in web addresses."""
    body = '<p>See https://example.org/report_2013_final.pdf and x^2 written loose.</p>'
    assert vtb.normalize_math(body) == body


def test_the_prose_around_the_fraction_is_untouched(vtb):
    body = ("<p>about 12 inches long and 7<math>\\frac{1}{2}</math> inches "
            "broad, of 5 3/16 by 11 inches</p>")
    assert vtb.normalize_math(body) == (
        "<p>about 12 inches long and 7½ inches broad, of 5 3/16 by 11 "
        "inches</p>")


def test_a_displayed_equation_keeps_its_fraction(vtb):
    body = '<math display="block">w = \\frac{1}{2}</math>'
    assert vtb.normalize_math(body) == body


def test_on_a_born_digital_page_scripts_are_left_for_the_layer(vtb):
    """There the PDF's own text corrects the letters a subscript was read
    as, and a tag in the word would stop it being asked."""
    assert flat(vtb, "Y_L", scripts=False) == "Y_L"
    assert flat(vtb, "x^2", scripts=False) == "x^2"


def test_a_bold_letter_is_its_letter(vtb):
    assert flat(vtb, r"\mathbf{R}_1") == "R<sub>1</sub>"
