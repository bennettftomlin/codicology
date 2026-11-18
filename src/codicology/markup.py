"""Small helpers for the markup and text a page is held in, shared by the
modules that work on it: tags stripped, the raised figure of a note call,
the ways one word can be set, Roman numerals. Nothing here imports the rest
of codicology.
"""
from __future__ import annotations

import html
import re


def _strip_tags(fragment: str) -> str:
    """The words of a fragment, with markup removed.

    Removing a tag leaves a word boundary behind, not nothing: a page whose
    heading and subheading are adjacent blocks flattened to "Chapter
    5Elasticity", which is why a chapter opener could not be recognised by
    its own printed title. Inline tags inside a word (<i>, <b>) would suffer
    the opposite fault, so only tags that open or close a block introduce
    the space.

    Only what is shaped like a tag is removed. A bare "<" in the text (a
    reading that left "a < b and c > d" unescaped) is text, and so is
    everything up to the next ">"; matching any "<...>" deleted it.
    """
    spaced = re.sub(r"</(?:p|div|h[1-6]|li|tr|td|th|table|section|figure|"
                    r"figcaption|blockquote|ol|ul|br)\s*>|<(?:br|hr)\s*/?>",
                    " ", fragment, flags=re.I)
    return html.unescape(_TAG.sub("", spaced)).strip()


# A tag: a name, then attributes that each carry a value. Every one of the
# 998,117 tags in the shelf's 151 installed books has this shape; "<b and c>"
# does not. Comments, declarations and processing instructions also go.
_TAG = re.compile(
    r"<!--.*?-->|<[!?][A-Za-z][^<>]*>"
    r"|</?[A-Za-z][\w:.-]*"
    r"(?:\s+[\w:.-]+\s*=\s*(?:\"[^\"]*\"|'[^']*'|[^\s\"'<>]+))*\s*/?>",
    re.S)


RUNNING_HEAD_MAX_CHARS = 60


BODY_MARKER = re.compile(r"<sup>\s*(\d{1,3})\s*</sup>")


def _roman_to_int(s: str) -> "int | None":
    """The value of a Roman numeral, or None if it is not one.

    Strict: it must re-render to exactly what was read, so "IIII", "VX" and
    other things that merely use the letters are refused rather than guessed
    at. A misread group number would scope a whole chapter's notes wrongly.
    """
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = prev = 0
    for ch in reversed(s.upper()):
        if ch not in values:
            return None
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    if not 1 <= total <= 99:
        return None
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return total if tens[total // 10] + ones[total % 10] == s.upper() else None


# How one word can be set two ways: the publisher's curly quotes and primes
# where the recogniser read straight ones, a true minus for a hyphen, a
# ligature, an ellipsis as one character. None of them changes which word
# it is, so none of them needs the pairing proven.
_SETTING = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": "''",
    "–": "-", "—": "-", "‒": "-", "−": "-", "…": "...",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
})
