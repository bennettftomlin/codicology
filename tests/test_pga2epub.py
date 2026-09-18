"""The two ways a Project Gutenberg Australia page lies about its text.

The first is the one the converter was written for: the page declares
windows-1252 and stores UTF-8 bytes that were read through it, so an
e-acute arrives as `&Atilde;&copy;`. Reversing that is arithmetic, and
the tests for it are arithmetic.

The second is the one that matters more, because it is silent. **Not
every book on that site is mojibake.** Measured across eight of them,
exactly one was; the other seven carried real em dashes, a real e-acute,
a real pound sign. Reversing a page that was already correct turns every
one of those into an invalid byte, and the build would have called the
wreckage "damage the source did not survive" and asked for a repair.
So the detector gets a test with a clean page in it, and that test is
the point of this file.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

import pga2epub as P


PAGE = """<html><body>
<table><tr><td>Project Gutenberg Australia a treasure-trove of literature
</td></tr></table>
<pre>Title: A Book
* A Project Gutenberg of Australia eBook *
eBook No.: 0000000h.html</pre>
{body}
<p>This site is full of FREE ebooks - Project Gutenberg Australia</p>
</body></html>"""


def page(body: str) -> str:
    return PAGE.format(body=body)


# --------------------------------------------------------------- encoding

def test_a_pair_that_survived_comes_back_as_its_letter():
    text, moji = P.decode_text("Nov&Atilde;&copy; a &Aring;&iexcl; a &Atilde;&shy; a &Auml;&rsaquo;")
    assert moji is True
    assert "Nové a š a í a ě" in text


def test_a_character_the_source_lost_comes_back_as_its_own_escape():
    # &Atilde; with nothing after it: the second half never made it into
    # the source, so the byte is left standing and nameable in JSON.
    text, _ = P.decode_text("NOV&Atilde; STR&Aring;AC&Atilde; "
                               "&Atilde;&copy;&Aring;&iexcl;&Auml;&rsaquo;")
    assert "NOV\\xc3 STR\\xc5AC\\xc3" in text


def test_a_clean_page_is_left_exactly_as_it_is():
    """The seven books in eight that this protects."""
    clean = ("<p>A dash &mdash; an e-acute &eacute; &mdash; a pound "
             "&pound; &mdash; and a caf&eacute;.</p>")
    text, moji = P.decode_text(clean)
    assert moji is False
    assert "—" in text and "café" in text
    assert "\\x" not in text


def test_the_detector_calls_a_mojibake_page_mojibake():
    moji = "<p>Nov&Atilde;&copy; &Aring;&iexcl; &Atilde;&shy; "\
           "&Auml;&rsaquo; &Aring;&frac34;</p>"
    assert P.is_mojibake(P.unescape_keeping_markup(moji)) is True


def test_the_detector_leaves_a_page_of_real_accents_alone():
    plain = "<p>caf&eacute; na&iuml;ve r&ocirc;le &mdash; &pound;5 &mdash; "\
            "&Uuml;ber &frac12;</p>"
    assert P.is_mojibake(P.unescape_keeping_markup(plain)) is False


def test_a_book_file_can_overrule_the_detector():
    plain = "<p>caf&eacute;</p>"
    _, moji = P.decode_text(plain, force=True)
    assert moji is True


# ---------------------------------------------------------------- repairs

def test_a_repair_fires_the_number_of_times_it_claims():
    text, log = P.apply_repairs("a \\xc3coles b \\xc3coles",
                                [{"bad": "\\xc3coles", "good": "Écoles",
                                  "count": 2, "why": ""}])
    assert text == "a Écoles b Écoles"
    assert len(log) == 1


def test_a_repair_that_matches_a_different_number_stops_the_build():
    with pytest.raises(SystemExit) as e:
        P.apply_repairs("one \\xc3coles only",
                        [{"bad": "\\xc3coles", "good": "Écoles",
                          "count": 2, "why": ""}])
    assert "matches 1 places" in str(e.value)


def test_a_byte_no_repair_names_stops_the_build():
    with pytest.raises(SystemExit) as e:
        P.apply_repairs("Ha\\xc4ti", [])
    assert "\\xc4" in str(e.value)


# -------------------------------------------------------------- structure

def test_a_chapter_closed_early_gets_its_orphans_back():
    """One book closes a chapter one </div> too soon, which puts a
    paragraph and a half outside it. Counting words could not see it:
    the lost words were ordinary ones the book uses elsewhere."""
    body = P.parse(page(
        '<div id="book1"><h3>BOOK ONE</h3><h4>PART</h4>'
        '<div id="c1" class="chapter"><h3>Chapter 1</h3><h4>ONE</h4>'
        "<p>inside the chapter</p></div>"
        "<p>orphaned by a stray close tag</p>"
        '<div id="c2" class="chapter"><h3>Chapter 2</h3><h4>TWO</h4>'
        "<p>second chapter</p></div></div>"))
    assert P.absorb_strays(body) == 1
    sections = P.read_divs(body, {})
    chapters = [c for k, c in sections if k == "chapter"]
    assert "orphaned by a stray close tag" in chapters[0].source_text
    assert "orphaned" not in chapters[1].source_text


def test_the_site_banner_and_footer_go_and_the_book_stays():
    body = P.parse(page("<p>" + "the text of the book. " * 60 + "</p>"))
    gone = P.strip_site_furniture(body)
    assert len(gone) == 3
    left = body.text_content()
    assert "treasure-trove" not in left and "FREE ebooks" not in left
    assert "the text of the book." in left


def test_a_block_too_big_to_be_furniture_is_kept_and_said_so():
    body = P.parse("<html><body><div>Google Site Search "
                   + "the whole book lives in here. " * 200
                   + "</div></body></html>")
    gone = P.strip_site_furniture(body)
    assert gone and "too much of the book" in gone[0]
    assert "the whole book lives in here." in body.text_content()


def test_coverage_names_a_block_that_reaches_no_chapter():
    body = P.parse(page("<p>a paragraph that no chapter ever sees</p>"))
    missed = P.uncovered(body, "some chapter text", floor=20)
    assert any("aparagraphthatnochapterever" in m for m in missed)


def test_coverage_is_quiet_when_every_block_reaches_a_chapter():
    body = P.parse("<html><body><p>all of this is in a chapter</p>"
                   "</body></html>")
    kept = P.squash("all of this is in a chapter")
    assert P.uncovered(body, kept, floor=5) == []


# ------------------------------------------------------------ serialising

def test_an_empty_element_is_not_left_self_closed():
    """`<em/>` is legal XML and a trap: a reader that falls back to an
    HTML parser reads it as an opening tag and italicises the rest."""
    assert P.expand_empty("<p><em/></p>") == "<p><em></em></p>"
    assert P.expand_empty('<a href="x"/>') == '<a href="x"></a>'


def test_the_void_elements_stay_self_closed():
    assert P.expand_empty("<p>a<br/>b</p>") == "<p>a<br/>b</p>"
    assert P.expand_empty('<img src="x"/>') == '<img src="x"/>'


# -------------------------------------------------------------- book file

def test_the_shipped_book_file_still_parses_and_names_its_repairs():
    here = os.path.join(os.path.dirname(__file__), "..", "tools", "pga")
    for name in os.listdir(here):
        if not name.endswith(".json"):
            continue
        cfg = json.loads(open(os.path.join(here, name),
                              encoding="utf-8").read())
        assert cfg["source"].startswith("http")
        assert cfg["reader"] in ("divs", "flow")
        for r in cfg.get("repairs", []):
            assert r["bad"] and r["good"] and int(r["count"]) >= 1
            assert r["why"], f"{name}: a repair with no reason given"
