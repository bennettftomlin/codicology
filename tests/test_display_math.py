"""An equation set on its own line, written as MathML.

The recogniser writes it as TeX inside <math display="block">, which has no
MathML namespace and so is shown by every reading system as the TeX itself:
429 times in 14 books on the shelf.
"""
import re
import zipfile

import pytest

from codicology import pipeline as P

EQUATION = r"\frac{\Delta Q / Q}{\Delta P / P} = -1.5"
needs_pandoc = pytest.mark.skipif(P._pandoc() is None, reason="pandoc is not installed")


def _page(tex=EQUATION):
    return f'<p>The elasticity is</p><math display="block">{tex}</math><p>which is elastic.</p>'


@needs_pandoc
def test_an_equation_becomes_mathml(monkeypatch):
    bodies = [_page()]
    assert P.write_display_math(bodies) == {"mathml": 1, "text": 0}
    m = re.search(r"<math\b[^>]*>.*?</math>", bodies[0], re.S).group(0)
    assert m.startswith('<math xmlns="http://www.w3.org/1998/Math/MathML" display="block" '
                        'alttext="(Δ Q / Q)/(Δ P / P) = -1.5">')
    assert "<mfrac>" in m and "<mi>Δ</mi>" in m
    # pandoc's TeX annotation is not kept: every check counts a page's words
    assert "annotation" not in m and "frac" not in re.sub(r"<[^>]+>", " ", m)
    assert bodies[0].startswith("<p>The elasticity is</p>") and bodies[0].endswith("<p>which is elastic.</p>")


@needs_pandoc
def test_what_pandoc_cannot_parse_is_set_as_its_text():
    bodies = [_page("P } d } r = P"), _page()]
    assert P.write_display_math(bodies) == {"mathml": 1, "text": 1}
    assert "<mtext>P } d } r = P</mtext></math>" in bodies[0]
    assert "<mfrac>" in bodies[1]           # and its neighbour is not paired with another's


def test_without_pandoc_every_equation_is_set_as_its_text(monkeypatch, tmp_path):
    monkeypatch.setenv(P.PANDOC_ENV, str(tmp_path / "no-such-pandoc"))
    assert P._pandoc() is None
    bodies = [_page(r"WBGT = 0.7 \times \text{wet bulb temperature}")]
    assert P.write_display_math(bodies) == {"mathml": 0, "text": 1}
    assert (f'<math xmlns="{P.MATHML_NS}" display="block" alttext="WBGT = 0.7 × wet bulb temperature">'
            "<mtext>WBGT = 0.7 × wet bulb temperature</mtext></math>") in bodies[0]


def test_written_once_it_is_not_written_again(monkeypatch, tmp_path):
    monkeypatch.setenv(P.PANDOC_ENV, str(tmp_path / "no-such-pandoc"))
    bodies = [_page()]
    P.write_display_math(bodies)
    once = bodies[0]
    assert P.write_display_math(bodies) == {"mathml": 0, "text": 0} and bodies[0] == once


def test_an_equation_in_a_line_of_text_is_not_touched():
    body = "<p>where <math>\\alpha</math> is the rate</p>"
    bodies = [body]
    assert P.write_display_math(bodies) == {"mathml": 0, "text": 0} and bodies == [body]


def test_a_page_with_mathml_says_so_in_the_package(tmp_path, monkeypatch):
    """EPUB 3.3: a content document holding MathML has the "mathml" property,
    and a page bound into a chapter file gives it to the file."""
    from PIL import Image, ImageDraw
    from codicology import binding as B
    monkeypatch.setenv(P.PANDOC_ENV, str(tmp_path / "no-such-pandoc"))

    class Reader:
        batch_size = 4
        name = "fake"
        langs = ["en"]

        def run_items(self, images):
            out = []
            for k, _ in enumerate(images):
                html = "<p>The firm has stood for some years, and its books were kept in a fair hand.</p>"
                if k == 1:
                    html += '<math display="block">A = 2(10) + 11 + 5 = 36 \\text{ feet}</math>'
                out.append([P.PageItem(html=html, label="Text")])
            return out
    paths = []
    for k in range(3):
        im = Image.new("RGB", (600, 800), "white")
        d = ImageDraw.Draw(im)
        for row in range(80, 480 + 40 * k, 40):
            d.rectangle((60, row, 540, row + 14), fill=(30, 30, 30))
        im.save(tmp_path / f"page_{k:04d}.png")
        paths.append(str(tmp_path / f"page_{k:04d}.png"))
    out = str(tmp_path / "book.epub")
    P.build_epub(paths, out, Reader(), "A Book", False, dedupe=False, drop_blank=False)
    with zipfile.ZipFile(out) as z:
        opf = z.read("EPUB/content.opf").decode()
        pages = B.page_bodies(z)
    assert "36 feet" in pages[1] and P.MATHML_NS in pages[1]
    items = re.findall(r"<item\b[^>]*>", opf)
    holding = [i for i in items if 'properties="mathml"' in i or "mathml" in (re.search(r'properties="([^"]*)"', i) or [None, ""])[1]]
    assert len(holding) == 1 and re.search(r'href="(?:ch|page)_\d{4}\.xhtml"', holding[0])


@needs_pandoc
def test_a_dollar_sign_is_money_and_not_the_end_of_the_equation():
    bodies = [_page("AE = $1,400 + 0.8($6,000) = $6,200"), _page(r"\text{Real wage} = \$5 / 1.630")]
    assert P.write_display_math(bodies) == {"mathml": 2, "text": 0}
    assert "<mi>$</mi>" in bodies[0] and "6,200" in bodies[0]
