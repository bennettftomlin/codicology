"""Where a word stands on its page, for any page.

The contract (adjudicate.py, "where a word stands on its page"): a dispute
is addressed in the token stream tokens() makes of the page's reader text,
and the appliers find it again through page_words(), which makes the SAME
stream from the xhtml and keeps, for each token, the span of markup it was
read from. One tokeniser on both sides is the point.

Checked on generated pages, under both generations of the word pattern
(tokens_as(None): the Latin-1 pattern of decisions files made before
TOKENS=2; tokens_as(2): any script).
"""
import html
import re
import unicodedata

import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

from codicology import adjudicate as adj

from .pages import page_body, reader_text, VOCAB

GENERATIONS = [None, adj.TOKENS]


def _cat(ch):
    return unicodedata.category(ch)[0]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(body=page_body())
def test_text_view_is_what_the_readers_see(gen, body):
    text, starts, ends, markup = adj.text_view(body)
    assert text == reader_text(body)
    assert len(starts) == len(ends) == len(markup) == len(text)
    # every character's source is a real, ordered stretch of the body
    for k in range(len(text)):
        assert 0 <= starts[k] < ends[k] <= len(body)
        if k:
            assert starts[k] >= starts[k - 1]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(body=page_body())
def test_page_words_makes_the_same_stream_as_tokens(gen, body):
    with adj.tokens_as(gen):
        folded, words = adj.page_words(body)
        assert folded == adj.fold_text(reader_text(body))
        assert [w[0] for w in words] == adj.tokens(reader_text(body))
        assert [t for t, _, _ in adj.page_tokens(body)] == \
            [w[0] for w in words]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(body=page_body())
def test_each_word_points_at_its_own_markup(gen, body):
    """A word's offset in the folded text reads the word; its span of the
    body, where it has one, holds no markup, reads back as exactly that
    word, and spans and offsets stand in page order without overlapping."""
    with adj.tokens_as(gen):
        folded, words = adj.page_words(body)
        last_at, last_hi = -1, 0
        for tok, lo, hi, at in words:
            assert folded[at:at + len(tok)] == tok
            assert at > last_at
            last_at = at + len(tok) - 1
            assert "<" not in tok and ">" not in tok and "&" not in tok
            if lo is None:
                assert hi is None
                continue
            assert lo >= last_hi, "spans overlap or run backwards"
            last_hi = hi
            src = body[lo:hi]
            assert "<" not in src and ">" not in src, "a span holds a tag"
            assert adj.tokens(html.unescape(src)) == [tok]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(body=page_body(), data=st.data())
def test_rewriting_one_span_changes_that_word_alone(gen, body, data):
    with adj.tokens_as(gen):
        words = adj.page_words(body)[1]
        editable = [k for k, w in enumerate(words) if w[1] is not None]
        assume(editable)
        k = data.draw(st.sampled_from(editable))
        _, lo, hi, _ = words[k]
        after = adj.tokens(reader_text(body[:lo] + "Zq" + body[hi:]))
        toks = [w[0] for w in words]
        assert after == toks[:k] + ["Zq"] + toks[k + 1:]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(body=page_body())
def test_hard_view_lines_up_with_the_folded_text(gen, body):
    with adj.tokens_as(gen):
        folded, _ = adj.page_words(body)
        hard = adj.hard_view(body)
        assert len(hard) == len(folded)
        for a, b in zip(hard, folded):
            assert a == b or (a == adj.STRUCK and b == " ")


@given(body=page_body())
def test_the_old_pattern_only_cuts_words_the_new_one_reads_whole(body):
    """The documented difference between the generations: the Latin-1
    pattern stops at a letter outside Latin-1, so each of its tokens is a
    piece of one any-script token (or that whole token), and on a page
    whose letters are all Latin-1 the two agree."""
    folded = adj.fold_text(reader_text(body))
    with adj.tokens_as(None):
        old = [(m.start(), m.end()) for m in adj.WORD.finditer(folded)]
    with adj.tokens_as(adj.TOKENS):
        new = [(m.start(), m.end()) for m in adj.WORD.finditer(folded)]
    for a, b in old:
        assert any(lo <= a and b <= hi for lo, hi in new), \
            (folded[a:b], [folded[lo:hi] for lo, hi in new])
    latin1 = all(ord(c) < 0x100 or c in "Œœ’" or _cat(c) not in "LM"
                 for c in folded)
    if latin1:
        assert old == new


def test_what_the_old_pattern_makes_of_greek_and_macrons():
    page = "plēpos plēpos οἰκόσιτοι technē ſuch café"
    with adj.tokens_as(None):
        assert adj.tokens(page) == ["pl", "pos", "ple", "pos", "techn",
                                    "uch", "café"]
    with adj.tokens_as(2):
        assert adj.tokens(page) == ["plēpos", "plēpos", "οἰκόσιτοι",
                                    "technē", "ſuch", "café"]


@given(body=page_body(breaks=False))
def test_nfc_and_nfd_read_as_the_same_words_under_any_script(body):
    """fold_text promises nothing about normalisation ("nothing else
    touches the words"), so an NFD page's tokens are NFD — but there must
    be the same words, one for one, and they fold the same."""
    text = reader_text(body)
    nfc = unicodedata.normalize("NFC", text)
    nfd = unicodedata.normalize("NFD", text)
    with adj.tokens_as(adj.TOKENS):
        a, b = adj.tokens(nfc), adj.tokens(nfd)
    assert [unicodedata.normalize("NFC", t) for t in b] == a
    assert [adj.fold_word(t) for t in b] == [adj.fold_word(t) for t in a]


# fixed 2026-11-19
@given(word=st.sampled_from(["plēthos", "technēma", "Kenkyūjo", "μέτρον"]),
       data=st.data())
@example(word="plēthos", data=None)
def test_a_line_break_hyphen_joins_nfd_as_it_joins_nfc(word, data):
    if data is None:
        k = 3                                   # "plē-\nthos"
    else:
        k = data.draw(st.integers(1, len(word) - 1))
    text = word[:k] + "-\n" + word[k:]
    nfc = adj.tokens(unicodedata.normalize("NFC", text))
    nfd = adj.tokens(unicodedata.normalize("NFD", text))
    assert [unicodedata.normalize("NFC", t) for t in nfd] == nfc


@given(body=page_body())
def test_site_address_names_one_place(body):
    toks = adj.tokens(reader_text(body))
    for i, t in enumerate(toks):
        a = adj.site_address(toks, i)
        # the address read back finds toks[i] and nothing earlier
        matches = [j for j, u in enumerate(toks) if u == t]
        assert matches[a["occurrence"]] == i
        assert len(a["before"]) == min(i, adj.SITE_CONTEXT)
        assert toks[i - len(a["before"]):i] == a["before"]
        assert toks[i + 1:i + 1 + len(a["after"])] == a["after"]


@given(body=page_body())
def test_body_span_of_a_page_file_is_the_body(body):
    xhtml = (f"<?xml version='1.0'?><html><head><title>be and be"
             f"</title></head><body class='x'>{body}</body></html>")
    lo, hi = adj.body_span(xhtml)
    assert xhtml[lo:hi] == body
    assert adj.body_span(body) == (0, len(body))


# fixed 2026-11-19
@pytest.mark.parametrize("lig,plain", [("ﬀ", "ff"), ("ﬃ", "ffi"),
                                       ("ﬄ", "ffl"), ("ﬆ", "st")])
def test_every_ligature_folds_as_fi_does(lig, plain):
    word = "e" + lig + "ort"
    assert adj.fold_word(word) == adj.fold_word("e" + plain + "ort")
    with adj.tokens_as(None):
        assert adj.tokens(word) == ["e" + plain + "ort"]
