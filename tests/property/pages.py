"""Generated pages for the property tests: structured xhtml bodies with a
small vocabulary, so the same word stands many times on a page, in every
script and spelling the shelf prints it in.

The vocabulary is deliberately awkward: Greek with breathings and accents
(polytonic), macrons written both precomposed (ē) and as e + U+0304,
combining acute, the fi ligature, long s, curly and straight apostrophes,
a Latin-Greek compound, and a few plain English words that recur.
"""
import html
import re
import unicodedata

from hypothesis import strategies as st

COMBINING_MACRON = "̄"

# words a page prints; each is one token under the any-script pattern
LATIN = ["the", "be", "and", "of", "navy", "to"]
UNICODE = [
    "plēthos",                              # precomposed macron
    "ple" + COMBINING_MACRON + "thos",      # the same word, decomposed
    "οἰκόσιτοι",                            # polytonic Greek
    "λόγος",
    "ﬁnal",                                 # ligature
    "ſuch",                                 # long s
    "café",
    "café",                           # decomposed acute
    "don’t", "don't",
    "technē",
    "Kenkyū",
    "logosλόγος",                           # mixed Latin + Greek
    "naïve",
]
VOCAB = LATIN + UNICODE

# words a reviewer types that no generated page prints, so where one lands
# is never in doubt
FRESH = ["navvy", "Zq", "ēxtra", "Wörd", "ἄλλος", "ﬂow"]

INLINE = ["i", "b", "em", "sup", "small", "span"]


def entity_form(word: str, draw) -> str:
    """The word as xhtml may store it: an entity for an accented letter or
    a curly apostrophe, sometimes."""
    out = []
    for ch in word:
        if ch == "’" and draw(st.booleans()):
            out.append("&#8217;")
        elif ch == "é" and draw(st.booleans()):
            out.append("&eacute;")
        elif ch == "ﬁ" and draw(st.booleans()):
            out.append("&#xFB01;")
        else:
            out.append(html.escape(ch, quote=False))
    return "".join(out)


@st.composite
def word_markup(draw, vocab=VOCAB, breaks=True):
    """One printed word: maybe set in another type, maybe broken across a
    line with a hyphen, maybe stored with entities."""
    w = draw(st.sampled_from(vocab))
    text = entity_form(w, draw)
    if breaks and len(w) >= 4 and draw(st.integers(0, 9)) == 0:
        # a line-break hyphen inside a word (not inside an entity, and not
        # after a combining mark: see the NFD finding)
        k = draw(st.integers(2, len(w) - 2))
        if (w[k - 1].isalnum() and w[k].isalnum()
                and unicodedata.category(w[k]) != "Mn"):
            # sometimes the break crosses markup: the joined word then
            # straddles a tag and cannot be rewritten in place
            brk = draw(st.sampled_from(["-\n", "-\n", "-<br/>\n"]))
            text = (html.escape(w[:k], quote=False) + brk
                    + html.escape(w[k:], quote=False))
    if draw(st.integers(0, 4)) == 0:
        tag = draw(st.sampled_from(INLINE))
        text = f"<{tag}>{text}</{tag}>"
    elif draw(st.integers(0, 12)) == 0:
        text = f'<a href="#n{draw(st.integers(1, 9))}">{text}</a>'
    return text


SEPARATORS = [" ", " ", " ", " ", ", ", ". ", "; ", " &amp; ", "<br/>",
              " (", ") ", " “", "” ", "\n", " &lt;", "&gt; ", " &nbsp;"]


@st.composite
def block(draw, vocab=VOCAB, breaks=True, max_words=14):
    n = draw(st.integers(1, max_words))
    parts = []
    for j in range(n):
        if j:
            parts.append(draw(st.sampled_from(SEPARATORS)))
        parts.append(draw(word_markup(vocab, breaks)))
        if draw(st.integers(0, 15)) == 0:
            parts.append(f"<sup>{draw(st.integers(1, 9))}</sup>")
    tag = draw(st.sampled_from(["p", "p", "p", "h2", "li", "div"]))
    return f"<{tag}>{''.join(parts)}</{tag}>"


@st.composite
def page_body(draw, vocab=VOCAB, breaks=True, max_blocks=4, max_words=14):
    blocks = draw(st.lists(block(vocab, breaks, max_words),
                           min_size=1, max_size=max_blocks))
    return "\n".join(blocks)


def reader_text(body: str) -> str:
    """A page as epub_page_texts hands it to the readers: every tag one
    space, then entities unescaped (written out here, not imported, so the
    property does not take the code's word for it)."""
    return html.unescape(re.sub(r"<[^>]+>", " ", body))
