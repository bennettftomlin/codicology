"""Mathematics as the reader will see it: vulgar fractions and scripts set as
characters or <sub>/<sup>, a power on a unit set as one, TeX left over from
the recogniser set as text, and an equation set on its own line written
as MathML (through pandoc, where it is installed).
"""
from __future__ import annotations

import html
import os
import re
from .tex import GREEK, MATH_SIGNS, _DISPLAY_MATH


# The fractions Unicode sets as one character. Any other is set the way
# a compositor without the sort would: raised numerator, fraction slash,
# lowered denominator.
VULGAR = {"1/2": "½", "1/3": "⅓", "2/3": "⅔", "1/4": "¼", "3/4": "¾",
          "1/5": "⅕", "2/5": "⅖", "3/5": "⅗", "4/5": "⅘", "1/6": "⅙",
          "5/6": "⅚", "1/7": "⅐", "1/8": "⅛", "3/8": "⅜", "5/8": "⅝",
          "7/8": "⅞", "1/9": "⅑", "1/10": "⅒"}


_RAISED = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


_LOWERED = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


_FRACTION = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


_MATH_FORMS = {"text", "mathrm", "mathit", "mathbf", "frac", "dfrac", "tfrac",
               "circ"}


def _set_fraction(m) -> str:
    over, under = m.group(1).strip(), m.group(2).strip()
    if re.fullmatch(r"[0-9]+", over) and re.fullmatch(r"[0-9]+", under):
        return (VULGAR.get(f"{over}/{under}")
                or over.translate(_RAISED) + "⁄" + under.translate(_LOWERED))

    def term(t):        # "100 cc./kg." over 10 is (100 cc./kg.)/10
        if not re.search(r"[\s+\-−×·/]", t):
            return t
        depth = 0           # already one bracketed group: leave it
        for k, ch in enumerate(t):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and k < len(t) - 1:
                return f"({t})"
        return t if t.startswith("(") else f"({t})"
    return f"{term(over)}/{term(under)}"


def _math_as_text(t: str) -> str:
    """What the recogniser wrote as TeX, as the letters and signs it
    stands for: Greek by name, a fraction as one, a degree sign, and every
    other command and brace taken away."""
    def name(m):
        word = m.group(1)
        if word in _MATH_FORMS:
            return m.group(0)       # has arguments: set below
        return GREEK.get(word) or MATH_SIGNS.get(word) or word

    # Names first, while each still ends where the recogniser ended it:
    # \mu\text{g} is μ and then g, and unwrapped first it is \mug.
    t = re.sub(r"\\([A-Za-z]+)", name, t)
    t = re.sub(r"\\(?:text|mathrm|mathit|mathbf)\{([^}]*)\}", r"\1", t)
    while True:     # innermost first, so a fraction over a fraction sets
        t, n = _FRACTION.subn(_set_fraction, t)
        if not n:
            break
    t = re.sub(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)", "°", t)
    t = t.replace("\\,", ",").replace("\\", "")
    return t


_SUPERSCRIPT = str.maketrans("0123456789+-−=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿⁱ")


_SCRIPT = re.compile(r"([_^])\s*(?:\{([^{}]*)\}|([^\s{}_^]))")


def _scripts_as_html(t: str) -> str:
    """TeX's subscripts and powers, in what is otherwise text, as HTML."""
    def one(m):
        inner = m.group(2) if m.group(2) is not None else m.group(3)
        inner = inner.strip()
        if not inner:
            return ""
        if m.group(1) == "_":
            return f"<sub>{inner}</sub>"
        if all(ch in "0123456789+-−=()ni" for ch in inner):
            return inner.translate(_SUPERSCRIPT)
        return f'<sup class="power">{inner}</sup>'
    return _SCRIPT.sub(one, t)


def normalize_math(body: str, scripts: bool = False) -> str:
    """
    LaTeX out, letters in.

    The recogniser wraps Greek in <math>\\alpha</math>, and no EPUB reader
    renders LaTeX: the reading experience is a raw backslash-word in the middle
    of a chemical name, over a thousand times across one book's chemistry half.
    The content is only ever letters and simple wrappers, so it translates
    exactly; anything unrecognised keeps its text with the markup stripped,
    which is never worse than what the reader would have shown.

    It wraps a printed fraction the same way — 7<math>\\frac{1}{2}</math>
    inches — in 25 of 151 books, and stripping that left "7frac{1}{2} inches"
    on the page. A fraction, a degree sign and the common operators are set
    as the characters they are.

    With `scripts`, a subscript or a power too, which are otherwise left as
    written ("P_1", "C_4H_{10}N^+"). A subscript is set as <sub>. A power is
    set as superscript characters where each of its characters has one (n²,
    C₄H₁₀N⁺), and otherwise as <sup class="power">: a bare <sup> holding a
    figure is what every note linker here takes for a call, and a power is
    not one. The caller asks for it only on a page with no born-digital
    text (see build_epub).
    """
    def clean(m):
        t = _math_as_text(m.group(1))
        return _scripts_as_html(t) if scripts else t
    # Display math is exempt: an equation set on its own line is not a Greek
    # letter in a chemical name, and its structure is all it has. The
    # recogniser marks it display="block" itself, on 410 of the 419 blocks
    # the layout labels Equation in the shelf's caches. The label is not
    # consulted: the eight it would add are expressions run into a sentence
    # ("In perfect competition, MRP = MP \times P"), which read better set
    # as text. (The assembly loop used to stamp blocks labelled "Formula",
    # a label Surya does not have, so it never stamped one.)
    return re.sub(r"<math(?![^>]*display=\"block\")[^>]*>(.*?)</math>",
                  clean, body, flags=re.S)


# A raised 2 or 3 standing directly on a unit of length is a power. The
# recogniser writes "60,000 km<sup>2</sup>" exactly as it writes a note
# call, and where the chapter had a note 2 the linker bound the unit to
# it: ten square kilometres on the shelf pointed at somebody's citation,
# and Small is Necessary's list of abbreviations ("ft² square feet")
# would have. These five are not words, so nothing else can be meant; "in"
# is one, and a raised 1 on "ft" in a table was a note mark, so neither is
# taken. Set as the character, it is what the page prints and nothing a
# linker looks at. (It is set once the page's own text layer has been
# deferred to: see where build_epub calls this.)
_UNIT_POWER = re.compile(
    r"(?<![A-Za-z])(mm|cm|km|m|ft)<sup>\s*([23])\s*</sup>")


def set_unit_powers(body: str) -> str:
    return _UNIT_POWER.sub(
        lambda m: m.group(1) + ("²" if m.group(2) == "2" else "³"), body)


# An equation set on its own line, written as MathML. The recogniser writes
# it as TeX inside <math display="block">, which is no MathML at all: it
# has no namespace, so every reading system showed the TeX, backslashes and
# braces, 429 times in 14 books (two economics textbooks hold 288 of them).
# MathML is part of EPUB 3 itself, and a reading system that draws pages
# must draw it; pandoc, where it is installed, turns the TeX into MathML.
# What it cannot parse (a recogniser's misreading, "P } d } r = P") and
# every equation when pandoc is not there is set as its plain text inside
# the same element, which draws as text everywhere. Either way the element
# carries that text as its alttext, for a reader that speaks it.
#
# pandoc also keeps the TeX inside, as an annotation. It is left out here:
# no reader shows it, and every check this pipeline makes of a built book
# counts the words of its pages, where "frac" and "Delta" would be words.
#
# This is the last thing done to a page's text, after every decision has
# been applied, so a decisions file made against a book built before it
# still finds its words: the token stream it addresses is the TeX one.
MATHML_NS = "http://www.w3.org/1998/Math/MathML"


PANDOC_ENV = "CODICOLOGY_PANDOC"


_SEPARATOR = "CODICOLOGYEQUATIONSEPARATOR"


def _pandoc() -> "str | None":
    import shutil
    said = os.environ.get(PANDOC_ENV)
    if said:
        return said if os.access(said, os.X_OK) else None
    return shutil.which("pandoc")


def _mathml_from_pandoc(texs: "list[str]") -> "list[str | None]":
    """Each TeX equation as the inside of a MathML <math> element, or None
    where pandoc is not there or could not parse it. One pandoc for all."""
    exe = _pandoc()
    if not exe or not texs:
        return [None] * len(texs)
    import subprocess
    # A bare $ is money ("AE = $300 + 0.8Y") to the recogniser and the end
    # of the equation to pandoc, which then gives up on it: 35 of the 57
    # equations it could not parse on the shelf. As \$ it is a dollar sign.
    doc = f"\n\n{_SEPARATOR}\n\n".join(
        "$$" + re.sub(r"(?<!\\)\$", r"\\$", re.sub(r"\s+", " ", t).strip())
        + "$$" for t in texs)
    try:
        done = subprocess.run([exe, "-f", "markdown", "-t", "html5",
                               "--mathml"], input=doc, capture_output=True,
                              text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return [None] * len(texs)
    if done.returncode != 0:
        return [None] * len(texs)
    parts = re.split(rf"<p>\s*{_SEPARATOR}\s*</p>", done.stdout)
    if len(parts) != len(texs):
        return [None] * len(texs)       # never pair one with another's
    out = []
    for part in parts:
        m = re.search(r"<math\b[^>]*>\s*<semantics>(.*?)"
                      r"<annotation\b.*?</annotation>\s*</semantics>\s*</math>",
                      part, re.S)
        out.append(m.group(1).strip() if m else None)
    return out


def write_display_math(bodies: list) -> dict:
    """Every display equation of the book as MathML (see MATHML_NS).
    Returns how many were set by pandoc and how many as text."""
    found = [(i, m) for i, b in enumerate(bodies)
             for m in _DISPLAY_MATH.finditer(b)]
    if not found:
        return {"mathml": 0, "text": 0}
    texs = [html.unescape(m.group(1)) for _, m in found]
    made = _mathml_from_pandoc(texs)
    stats = {"mathml": 0, "text": 0}
    new = {}
    for (i, m), tex, inner in zip(found, texs, made):
        plain = re.sub(r"\s+", " ", _math_as_text(tex)).strip()
        if inner is None:
            inner = f"<mtext>{html.escape(plain, quote=False)}</mtext>"
            stats["text"] += 1
        else:
            stats["mathml"] += 1
        new.setdefault(i, []).append((m.start(), m.end(), (
            f'<math xmlns="{MATHML_NS}" display="block" '
            f'alttext="{html.escape(plain)}">{inner}</math>')))
    for i, edits in new.items():
        b = bodies[i]
        for lo, hi, text in sorted(edits, reverse=True):
            b = b[:lo] + text + b[hi:]
        bodies[i] = b
    return stats
