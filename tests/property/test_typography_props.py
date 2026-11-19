"""Typography restoration (typography.normalize_typography) on generated
pages.

What the function promises (its docstring and _requote's): quote direction
restored and the double spaces no printed page contains collapsed; spaced
ellipses, hyphens and "anything involving letters" left alone; "text inside
tags is never touched"; a born-digital page (a ".native" witness beside its
page file) not touched at all. The rule for a quote is "the typesetter's
own: a quote after space or an opening bracket opens, any other closes; a
straight apostrophe between letters, after a letter, or eliding is ’."
"""
from __future__ import annotations

import re

from hypothesis import given, strategies as st

from codicology.typography import normalize_typography

from .notes_oracle import parses

QUOTES = "\"'‘’“”"
OPENING = {'"': "“", "'": "‘"}
CLOSING = {'"': "”", "'": "’"}

# text: letters (some outside ASCII), digits, punctuation, both kinds of
# quote straight and curly, single and repeated spaces, spaced ellipses
_CHARS = st.sampled_from(list("abcdefXYZéßøñ0123456789.,;:!?()[]—–-…&") + [
    '"', '"', "'", "'", "“", "”", "‘", "’", " ", " ", " ", "  ", "   ",
    ". . .", "&amp;", "&quot;", "&#39;", "'90s", "'tis", "\n"])
TEXT = st.lists(_CHARS, max_size=30).map("".join).map(
    lambda t: t.replace("&", "&amp;").replace("&amp;amp;", "&amp;")
               .replace("&amp;quot;", "&quot;").replace("&amp;#39;", "&#39;"))

INLINE = ("i", "b", "em", "sup", "sub", "span")
ATTRS = ('', ' class="it\'s"', ' title="say &quot;hi&quot;"',
         ' href="http://x.org/a\'b?q=1&amp;r=\'2\'"')


@st.composite
def inline(draw, depth=0):
    parts = []
    for _ in range(draw(st.integers(0, 4))):
        kind = draw(st.integers(0, 5))
        if kind == 0 and depth < 2:
            tag = draw(st.sampled_from(INLINE + ("a",)))
            attrs = draw(st.sampled_from(ATTRS)) if tag in ("a", "span") else ""
            parts.append(f"<{tag}{attrs}>{draw(inline(depth + 1))}</{tag}>")
        elif kind == 1:
            parts.append(draw(st.sampled_from(
                ['<br/>', '<img alt="it\'s &quot;x&quot;" src="a.png"/>'])))
        else:
            parts.append(draw(TEXT))
    return "".join(parts)


@st.composite
def pages(draw):
    blocks = []
    for _ in range(draw(st.integers(1, 4))):
        tag = draw(st.sampled_from(["p", "p", "h2", "li", "blockquote"]))
        body = draw(inline())
        blocks.append(f"<ol><li>{body}</li></ol>" if tag == "li"
                      else f"<{tag}>{body}</{tag}>")
    return "".join(blocks)


def _run(bodies, page_paths=None):
    out = list(bodies)
    normalize_typography(out, page_paths)
    return out


def _tags(page):
    return re.findall(r"<[^>]+>", page)


def _texts(page):
    return re.split(r"<[^>]+>", page)


@given(st.lists(pages(), min_size=1, max_size=3))
def test_markup_is_never_touched(bodies):
    for a, b in zip(bodies, _run(bodies)):
        assert _tags(b) == _tags(a)


@given(st.lists(pages(), min_size=1, max_size=3))
def test_only_quotes_turn_and_only_spaces_go(bodies):
    """Every character but a straight quote is kept, in order; a straight
    quote becomes a curly one of its own kind; spaces only collapse."""
    for a, b in zip(bodies, _run(bodies)):
        for ta, tb in zip(_texts(a), _texts(b)):
            sa, sb = re.sub(" +", " ", ta), re.sub(" +", " ", tb)
            assert len(sa) == len(sb), (ta, tb)
            for x, y in zip(sa, sb):
                if x == '"':
                    assert y in "“”", (ta, tb)
                elif x == "'":
                    assert y in "‘’", (ta, tb)
                else:
                    assert x == y, (ta, tb)


@given(st.lists(pages(), min_size=1, max_size=3))
def test_letters_and_digits_are_never_touched(bodies):
    def letters(t):
        return [c for c in re.sub(r"<[^>]+>|&\w+;|&#\d+;", "", t) if c.isalnum()]
    for a, b in zip(bodies, _run(bodies)):
        assert letters(b) == letters(a)


@given(st.lists(pages(), min_size=1, max_size=3))
def test_twice_is_once(bodies):
    once = _run(bodies)
    assert _run(once) == once


@given(st.lists(pages(), min_size=1, max_size=3))
def test_no_double_space_is_left_between_words(bodies):
    for b in _run(bodies):
        for t in _texts(b):
            assert not re.search(r"\S {2,}\S", t), t


@given(st.lists(pages(), min_size=1, max_size=3))
def test_spaced_ellipses_and_hyphens_stay_as_set(bodies):
    for a, b in zip(bodies, _run(bodies)):
        assert b.count(". . .") >= a.count(". . .")   # (a ". .  ." may close up)
        assert b.count("-") == a.count("-") and b.count("—") == a.count("—")


@given(st.lists(pages(), min_size=1, max_size=3))
def test_a_page_that_parsed_still_parses(bodies):
    for a, b in zip(bodies, _run(bodies)):
        if parses(a):
            assert parses(b), b


@given(st.lists(_CHARS, min_size=1, max_size=40).map("".join))
def test_the_typesetters_rule_in_plain_text(text):
    """On one run of text (no markup), a straight double quote after
    whitespace, the start or an opening bracket opens; after a letter, a
    digit or closing punctuation it closes. A straight apostrophe after a
    letter or digit is ’. (After a dash or another quote mark the rule as
    written does not say, and nothing is asserted.)"""
    text = text.replace("&", "")
    page = f"<p>{text}</p>"
    out = _run([page])[0][3:-4]
    # line the two up: only runs of spaces change length
    src = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)
    assert len(src) == len(out)
    for k, ch in enumerate(src):
        prev = src[k - 1] if k else " "
        if ch == '"':
            if prev in " \n([{":
                assert out[k] == "“", (text, out)
            elif prev.isalnum() or prev in ".,;:!?)]}…":
                assert out[k] == "”", (text, out)
        elif ch == "'" and prev.isalnum():
            assert out[k] == "’", (text, out)


def test_a_born_digital_page_is_left_as_set(tmp_path):
    page = tmp_path / "page_0000.xhtml"
    page.write_text("x")
    (tmp_path / "page_0000.xhtml.native").write_text("")
    bodies = ['<p>"Set  by the publisher," it\'s said.</p>',
              '<p>"OCR  page," it\'s said.</p>']
    out = list(bodies)
    stats = normalize_typography(out, [str(page), str(tmp_path / "page_0001.xhtml")])
    assert out[0] == bodies[0] and out[1] != bodies[1]
    assert stats["native_skipped"] == 1
