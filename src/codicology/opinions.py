"""The second and third readers, and what is taken from them: a table where
the first reader's holds fewer of the figures the PDF prints, an equation
where the readings differ in figures alone or two of three agree.
"""
from __future__ import annotations

import difflib
import html
import json
import os
import re
from PIL import Image
from collections import Counter
from .backends import _to_xhtml
from .cache import page_key
from .markup import _strip_tags
from .pdf_frame import _normalized_to_pdf, _page_frame
from .tex import _DISPLAY_MATH, _tex_name, _unfrac


# Mathematics is what the first reader reads worst and what nothing
# classical can check. In Sraffa's Production of Commodities it set
# (1 + 2/100) three times where the page prints 20/100, gave a system of
# equations subscripts that belong to no commodity, read a third as a half
# and put an equals sign into a formula that has none. The second reader
# read the same blocks, and where the two differ somebody has to be
# believed.
#
# On a typeset page the PDF's own text says which symbols are printed, flat
# as it is, and the reading nearer to it is kept.
#
# On a scan nothing says. The fourteen blocks of that book on which the two
# readers differ in more than a row of dots were each looked at: the second
# reader is the one as printed on nine, the first on two (a prime dropped
# three times, a subscript g read as x), and on three each has one letter
# right that the other has wrong. Asking the second reader again about the
# equation alone, cut out and enlarged, was tried and reads worse than
# either. Two readers cannot outvote each other, so only the difference
# they have a record on is acted on: figures. Where the two readings of an
# equation differ in nothing but digits the second reader's is taken (it
# held 93% of the figures the PDFs print where the first held 75%, and has
# both such equations in Sraffa as printed); any other difference is said,
# with both readings, and the first reader's stands.
#
# What is compared is the symbols, not the TeX: the two write the same
# equation differently (a system as one aligned block or a line each,
# p'_a or p_a'), and neither way of writing is the page's.
EQUATION_SAME_BLOCK = 0.75   # likeness of two readings of one block


# A reading is preferred when it is at most this far from the PDF's symbols,
# as a share of how far the other is. One wrong digit in a long equation is
# a small distance, so the test is which is nearer and by what proportion,
# not by how much.
EQUATION_NEARER = 0.5


def _flat_tex(tex: str) -> str:
    """The symbols of an equation in the order they are printed, with what
    says how to set them taken off. Three readers write TeX three ways
    ("p_a'", "p'_a", "p _ {a} ^ {\\prime}"), and this is what they are
    compared in."""
    t = html.unescape(tex)
    t = re.sub(r"\s*([_^])\s*", r"\1", t)             # "p _ {a}" is "p_{a}"
    t = re.sub(r"\^\{((?:\s*\\prime)+)\s*\}",
               lambda m: "'" * m.group(1).count("prime"), t)
    t = t.replace("\\prime", "'")
    # p'_a and p_a' are one symbol written two ways
    t = re.sub(r"('+)_(\{[^{}]*\}|[A-Za-z0-9])", r"_\2\1", t)
    t = re.sub(r"\\tag\s*\{([^{}]*)\}", r"(\1)", t)
    t = _unfrac(t)
    for _ in range(4):
        t = re.sub(r"\\(?:text|mathrm|mathit|mathbf|operatorname|textbf|"
                   r"textit)\s*\{([^{}]*)\}", r"\1", t)
        t = re.sub(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"\1/\2", t)
        t = re.sub(r"\\(?:bar|overline|hat|tilde)\s*\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\\\(?:\[[^\]]*\])?", "", t)                 # a row's end
    t = re.sub(r"\\(?:begin|end)\{[a-z*]+\}(?:\{[^{}]*\})?", "", t)
    t = re.sub(r"\\(?:left|right|displaystyle|quad|qquad|[,;!: ])", "", t)
    t = t.replace("\\%", "%").replace("\\$", "$").replace("\\{", "(") \
         .replace("\\}", ")")
    t = re.sub(r"\\([A-Za-z]+)", lambda m: _tex_name(m.group(1)), t)
    t = re.sub(r"[{}_^\\&\s]", "", t)
    return t.replace("−", "-").replace("–", "-").replace("’", "'") \
            .replace("′", "'").replace("·", ".").replace("⋅", ".")


def _tex_words(tex: str) -> int:
    """How many separate runs of letters an equation is written in. The
    flat form cannot see two symbols run into one: the PDF's text has no
    spaces to compare with, and a reading that sets "ΔL APL = Q/L TVC" as
    "ΔLAPL = Q/LTVC" has every symbol of the page and is not the page."""
    t = html.unescape(tex)
    t = re.sub(r"\\(?:begin|end)\{[a-z*]+\}(?:\{[^{}]*\})?", " ", t)
    t = re.sub(r"\\(?:text|mathrm|mathit|mathbf|operatorname|textbf|textit|"
               r"left|right|frac|dfrac|tfrac|bar|overline|hat|tilde|"
               r"displaystyle|quad|qquad)\b", " ", t)
    t = re.sub(r"\\([A-Za-z]+)", lambda m: " " + _tex_name(m.group(1)) + " ", t)
    return len(re.findall(r"[^\W\d_]+", t))


def _flat_printed(text: str) -> str:
    return re.sub(r"\s", "", text).replace("−", "-").replace("–", "-") \
        .replace("’", "'").replace("′", "'").replace("·", ".").replace("⋅", ".")


def _undotted(flat: str) -> str:
    """Without the rows of dots that stand for "and so on", which each
    reader writes its own way."""
    return re.sub(r"vdots|ddots|\.+|…", "", flat)


def _borne_out(first: str, printed: str) -> bool:
    """Whether the PDF's symbols hold this reading whole. The box's text
    may run on past the equation (its number, the line above), so it is
    looked for inside it; but a reading that ends where a number goes on
    ("x=10" in "x=100") or begins inside one has found a piece of what is
    printed, which is the very thing it is being checked for."""
    at = printed.find(first) if first else -1
    while at >= 0:
        end = at + len(first)
        cut_short = (first[-1].isdigit() and end < len(printed)
                     and printed[end].isdigit()) \
            or (first[0].isdigit() and at > 0 and printed[at - 1].isdigit())
        if not cut_short:
            return True
        at = printed.find(first, at + 1)
    return False


def _likeness(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


_FIGURES_ONLY = re.compile(r"[0-9]*")


def _differ_in_figures_alone(a: str, b: str) -> bool:
    """Whether the second flat reading is the first but for digits, and has
    no fewer of them anywhere. What the first reader does to figures is
    drop them (the 0 of 20/100, the 1 of a dose of 1 1/2, two columns of a
    table), and a second reading that has a digit more, or another digit,
    is the one worth taking. One that has a digit fewer is as likely to
    have dropped it."""
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    return all(tag == "equal" or (_FIGURES_ONLY.fullmatch(a[a0:a1])
                                  and _FIGURES_ONLY.fullmatch(b[b0:b1])
                                  and b1 - b0 >= a1 - a0)
               for tag, a0, a1, b0, b1 in ops)


def _differences(x: str, y: str) -> list:
    return [(tag, i1, i2, j2 - j1) for tag, i1, i2, j1, j2 in
            difflib.SequenceMatcher(None, x, y, autojunk=False).get_opcodes()
            if tag != "equal"]


def _stands(lo: int, hi: int, ops: list) -> bool:
    """Whether a reader whose differences from this reading are `ops` has
    this reading's own symbols from lo to hi (and, where lo is hi, has put
    nothing in at that point either)."""
    for tag, i1, i2, n in ops:
        if lo == hi:
            if i1 <= lo <= i2 and (tag == "insert"
                                   or (tag == "replace" and n != i2 - i1)
                                   or (tag == "delete" and i1 < lo < i2)):
                return False
        elif (i1 < hi and i2 > lo) or (i1 == i2 and lo < i1 < hi):
            return False
    return True


def majority_reading(flats: "list[str]") -> "int | None":
    """Which of three readings has, wherever it differs from one of the
    others, the third on its side; None where none has.

    That is a vote taken at each point of difference and not on the
    equation as a whole, which is what lets it find the page where every
    reader is wrong somewhere: Sraffa's Λ₁ρ₁ was A₁ρ₁ to the first reader
    and Λ₁p₁ to the second, and the third had both symbols. Two readers
    that share a mistake outvote the one that is right, and nothing here
    can know it."""
    for k, x in enumerate(flats):
        ops = [_differences(x, y) for j, y in enumerate(flats) if j != k]
        if all(_stands(i1, i2, ops[1 - n])
               for n in (0, 1) for _, i1, i2, _ in ops[n]):
            return k
    return None


_LINES_OF = r"array|aligned|align\*?|gathered"


def _tex_rows(tex: str) -> "list[str]":
    """The lines of a reading that sets several in one array. Only an array
    whose rows are equations is taken apart: the rows of a matrix or of a
    set of cases are parts of one thing, and cut at a row's end they are
    half an environment each."""
    if re.search(r"\\begin\{(?!(?:" + _LINES_OF + r")\})", tex):
        return [tex.strip()] if tex.strip() else []
    t = re.sub(r"\\begin\{array\}\s*\{[^{}]*\}"
               r"|\\(?:begin|end)\{(?:" + _LINES_OF + r")\}", " ", tex)
    return [r.strip() for r in re.split(r"\\\\(?:\[[^\]]*\])?", t)
            if r.strip()]


def _stands_alone(tex: str) -> bool:
    """Whether a piece of TeX opens nothing it does not close. What is
    printed from the third reader is printed as it wrote it, and half an
    environment or a brace left open would be printed as that."""
    depth = 0
    for k, ch in enumerate(tex):
        j = k
        while j > 0 and tex[j - 1] == "\\":
            j -= 1
        if (k - j) % 2:
            continue                    # \{ and \} are characters
        depth += (ch == "{") - (ch == "}")
        if depth < 0:
            return False
    return depth == 0 \
        and len(re.findall(r"\\begin\{", tex)) == len(re.findall(r"\\end\{", tex)) \
        and len(re.findall(r"\\left\b", tex)) == len(re.findall(r"\\right\b", tex))


def _tidy_tex(tex: str) -> str:
    """A third reader's TeX set the way the first reader's is. It writes
    every token apart ("p _ {a}", "2 0", "\\mathrm {i r o n}") and wraps
    its brackets in \\left and \\right; the book prints the TeX as it
    stands, so the habits come off and the symbols stay."""
    t = re.sub(r"\s+", " ", tex).strip()
    # \left. and \right. are the bracket that is not there; any other
    # \left or \right only sizes the bracket after it
    t = re.sub(r"\\(?:left|right)\s*\.", "", t)
    t = re.sub(r"\\(?:left|right)(?![A-Za-z])\s*", "", t)
    t = re.sub(r"\s*([_^])\s*", r"\1", t)
    t = re.sub(r"\\([A-Za-z]+) (?=\{)", r"\\\1", t)

    def upright(m: "re.Match") -> str:
        """\\mathrm {i r o n} is the word iron, and \\mathrm{\\Delta}
        is still mathematics."""
        inside = m.group(1)
        if re.fullmatch(r"[A-Za-z](?: [A-Za-z])*", inside):
            return "\\text{" + inside.replace(" ", "") + "}"
        if re.fullmatch(r"[A-Za-z0-9 .,]+", inside):
            return "\\text{" + inside + "}"
        return m.group(0)
    t = re.sub(r"\\mathrm\{([^{}]*)\}", upright, t)
    t = re.sub(r"(?<=\d) (?=\d)", "", t)
    for _ in range(3):                  # c_{1} as c_1, inside a group or out
        t = re.sub(r"([_^])\{([A-Za-z0-9])\}", r"\1\2", t)
    t = re.sub(r"\^\{((?:\\prime ?)+)\}",
               lambda m: "'" * m.group(1).count("prime"), t)
    # a row of dots it saw one of
    t = re.sub(r"(?<=[+\-=,]) ?\. ?(?=[+\-=,])", r" \\dots ", t)
    return re.sub(r" +", " ", t).strip()


def mend_equations(items: list, page_path: "str | None",
                   second: "SecondOpinion | None",
                   third: "ThirdOpinion | None" = None
                   ) -> "tuple[list, list[tuple[str, str]], list[tuple[str, str]]]":
    """Items with a display equation the second reader reads otherwise set
    as the second reader has it, where the PDF's own text prefers that
    reading or, on a scan, where the two differ only in figures; then (as it
    was, as it is) in symbols for each one changed, and (first, second) for
    each on which the readers differ and the first was kept.

    With a third reader, a scan's equation on which the first two differ is
    put to it, and the reading that has another reader with it at every
    point of difference is printed: the second's, the third's own, or the
    first's as it stood. Such an entry carries a third member, saying how
    the vote went. A reading with fewer figures than the first is never
    printed on a vote: on a typeset page, where the PDF could be asked, the
    one majority that was wrong had lost a digit."""
    if second is None or not any(
            it.html and not it.is_furniture and 'display="block"' in it.html
            for it in items):
        return items, [], []
    from . import second as _second
    text_in = _typeset_text_in(page_path)
    offered = None
    taken: set = set()      # the second reader's equations already printed
    rows_of: dict = {}      # the third reader's lines for a block, by block
    rows_used: dict = {}    # and those of them already printed
    out, changed, differ = [], [], []

    def best_run(first: str, lines: list, used=()) -> tuple:
        best = (0.0, 0, 0)
        for i in range(len(lines)):
            for j in range(i, min(len(lines), i + 12)):
                if used and used.intersection(range(i, j + 1)):
                    break               # printed once already, for a block before
                like = _likeness(first, "".join(f for _, f in lines[i:j + 1]))
                if like > best[0]:
                    best = (like, i, j)
        return best

    def third_run(it, first: str) -> "list | None":
        """The third reader's lines for this equation of the block, or None
        where it was not asked, gave nothing, or has nothing like it."""
        if third is None or not it.box:
            return None
        if id(it) not in rows_of:
            said = third.formula(page_path, it.box)
            rows_of[id(it)] = [(t, _flat_tex(t)) for t in
                               map(_tidy_tex, _tex_rows(said))] if said else []
        lines = rows_of[id(it)]
        used = rows_used.setdefault(id(it), set())
        like, i, j = best_run(first, lines, used)
        if not lines or like < EQUATION_SAME_BLOCK:
            return None
        return [(*lines[n], n) for n in range(i, j + 1)]
    for it in items:
        if not (it.html and not it.is_furniture and 'display="block"' in it.html):
            out.append(it)
            continue

        def judge(m: "re.Match") -> str:
            nonlocal offered
            first = _flat_tex(m.group(1))
            printed = _flat_printed(text_in(it.box)) \
                if text_in is not None and it.box else ""
            if text_in is not None:
                # typeset: only a reading the PDF's symbols do not bear out
                # is worth the second reader's time
                if len(printed) < 4 or _borne_out(_undotted(first),
                                                  _undotted(printed)):
                    return m.group(0)
            if offered is None:
                theirs = _second.display_equations(second.reading(page_path) or "")
                offered = [(t, _flat_tex(t)) for t in theirs]
            best = best_run(first, offered, taken)
            run = offered[best[1]:best[2] + 1]
            theirs_flat = "".join(f for _, f in run)
            if best[0] < EQUATION_SAME_BLOCK \
                    or _undotted(theirs_flat) == _undotted(first):
                return m.group(0)
            if _tex_words(" ".join(t for t, _ in run)) < _tex_words(m.group(1)) - 1:
                differ.append((first, theirs_flat))
                return m.group(0)       # it runs the symbols together
            if text_in is not None:
                off_first = 1 - _likeness(_undotted(first), _undotted(printed))
                off_theirs = 1 - _likeness(_undotted(theirs_flat),
                                           _undotted(printed))
                if off_theirs > EQUATION_NEARER * off_first:
                    return m.group(0)
            else:
                voted = third_run(it, first)
                if voted is not None:
                    third_flat = "".join(f for _, f, _ in voted)
                    flats = [_undotted(first), _undotted(theirs_flat),
                             _undotted(third_flat)]
                    k = majority_reading(flats)
                    figures = lambda f: sum(ch.isdigit() for ch in f)
                    if k == 2 and not all(_stands_alone(t) for t, _, _ in voted):
                        k = None        # its own, and not fit to print
                    if k is None and _differ_in_figures_alone(flats[0], flats[1]):
                        # three readings and no two alike: the third has
                        # settled nothing, and what the second reader was
                        # trusted with before it came is still its to give
                        changed.append((first, theirs_flat))
                        taken.update(range(best[1], best[2] + 1))
                        return "".join(
                            f'<math display="block">{html.escape(t, quote=False)}</math>'
                            for t, _ in run)
                    if k is None:
                        differ.append((first, theirs_flat,
                                       f"the third reads {third_flat}"))
                        return m.group(0)
                    if k == 0:
                        differ.append((first, theirs_flat,
                                       "the third reader sides with the first"))
                        return m.group(0)
                    if figures(flats[k]) < figures(flats[0]):
                        differ.append((first, theirs_flat,
                                       "two readers have it otherwise, with "
                                       "fewer figures than the first"))
                        return m.group(0)
                    if k == 2:
                        changed.append((first, third_flat,
                                        "the third reader's own: the first and "
                                        "the second each have part of it"))
                        rows_used[id(it)].update(n for _, _, n in voted)
                        body = voted[0][0] if len(voted) == 1 else (
                            "\\begin{aligned} "
                            + " \\\\ ".join(t for t, _, _ in voted)
                            + " \\end{aligned}")
                        return ('<math display="block">'
                                + html.escape(body, quote=False) + "</math>")
                    changed.append((first, theirs_flat,
                                    "the third reader sides with the second"))
                    taken.update(range(best[1], best[2] + 1))
                    return "".join(
                        f'<math display="block">{html.escape(t, quote=False)}</math>'
                        for t, _ in run)
                if not _differ_in_figures_alone(_undotted(first),
                                                _undotted(theirs_flat)):
                    differ.append((first, theirs_flat))
                    return m.group(0)
            changed.append((first, theirs_flat))
            taken.update(range(best[1], best[2] + 1))
            return "".join(f'<math display="block">{html.escape(t, quote=False)}</math>'
                           for t, _ in run)

        out.append(it._replace(html=_DISPLAY_MATH.sub(judge, it.html)))
    return out, changed, differ


_FIGURE = re.compile(r"\d[\d,.]*\d|\d")


def _figures(text: str) -> "Counter[str]":
    return Counter(n.replace(",", "").rstrip(".") for n in _FIGURE.findall(text))


def _typeset_text_in(page_path: "str | None"):
    """A function giving the PDF's own text inside a box (in fractions of
    the page), for a typeset page; None for a scan, or with no page to ask."""
    if not page_path:
        return None
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile") or src.get("scan"):
            return None
        import pypdfium2 as pdfium
        page = pdfium.PdfDocument(src["pdf"])[src["page"]]
        frame = _page_frame(page)
        textpage = page.get_textpage()
    except Exception:
        return None

    def text_in(box) -> str:
        try:
            return textpage.get_text_bounded(**_normalized_to_pdf(box, frame))
        except Exception:
            return ""
    return text_in


def typeset_body_size_text_in(page_path: "str | None"):
    """A function giving the PDF's own text inside a box, counting only the
    glyphs set at the page's body size, for a typeset page; None for a scan.

    A glyph is at body size when its loose box (the font's ascent to its
    descent, the same for every glyph of one size) stands at least
    BODY_SIZE_SHARE of the page's median. A raised note call is set smaller
    and is left out; a figure in a table column is not."""
    if not page_path:
        return None
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile") or src.get("scan"):
            return None
        import pypdfium2 as pdfium
        page = pdfium.PdfDocument(src["pdf"])[src["page"]]
        frame = _page_frame(page)
        textpage = page.get_textpage()
        glyphs = []
        for i in range(textpage.count_chars()):
            ch = textpage.get_text_range(i, 1)
            if not ch.strip():
                continue
            l, b, r, t = textpage.get_charbox(i, loose=True)
            glyphs.append((ch, (l + r) / 2, (b + t) / 2, t - b))
    except Exception:
        return None
    if not glyphs:
        return None
    heights = sorted(g[3] for g in glyphs)
    body = heights[len(heights) // 2]
    if body <= 0:
        return None

    def text_in(box) -> str:
        e = _normalized_to_pdf(box, frame)
        return "".join(ch for ch, x, y, h in glyphs
                       if e["left"] <= x <= e["right"]
                       and e["bottom"] <= y <= e["top"]
                       and h >= BODY_SIZE_SHARE * body)
    return text_in


BODY_SIZE_SHARE = 0.85


def _share_held(printed: "Counter[str]", held: "Counter[str]") -> float:
    total = sum(printed.values())
    return sum(min(n, held.get(w, 0)) for w, n in printed.items()) / total \
        if total else 1.0


# ── a second opinion ─────────────────────────────────────────────────────────
#
# What the second reader (second.py) is asked, and when its answer is taken.
# It is never taken on its own word. It reads a page only where the first
# reading has been found wanting by something that cannot invent, and its
# answer replaces the first reader's only where that same witness prefers
# it. On a typeset page the witness is the PDF's own text.

class SecondOpinion:
    """Second readings of a book's pages, read the first time each is asked
    for and kept beside the first reader's cache."""

    def __init__(self, reader, cache=None):
        self.reader, self.cache = reader, cache
        self.asked = 0              # pages read in this run
        self.consulted: set = set() # pages it was asked about, read or remembered
        self.tables = self.equations = self.differ = 0
        self.by_third = 0           # equations set as a third reader has them
        self.changed: list = []     # (page, what, as it was, as it is)
        self.listed: list = []      # (page, the first reading, the second)
        self.figures = 0            # scanned-table figures set by a vote
        self.ties: list = []        # (page, first, second, why) kept as first
        self.unanswered: set = set()    # pages it could not be asked about
        self.why_unanswered: "str | None" = None
        self._read: dict = {}       # this run's readings, with or without a cache

    def summary(self) -> "str | None":
        """One line on what the second reader was asked and what came of
        it, or None where it was never asked."""
        if not self.consulted:
            return None
        n = len(self.consulted)
        took = []
        if self.tables:
            took.append(f"{self.tables} table(s)")
        if self.equations:
            took.append(f"{self.equations} equation(s)")
        if self.figures:
            took.append(f"{self.figures} scanned-table figure(s) by a vote "
                        f"of three readers")
        remembered = n - self.asked - len(self.unanswered)
        return (f"second reader: asked about {n} page(s)"
                + (f", {remembered} of them remembered" if remembered > 0 else "")
                + (f"; {' and '.join(took)} set as it reads them"
                   if took else "; nothing was changed on its word")
                + (f", and {self.by_third} equation(s) as the third reader "
                   f"reads them" if self.by_third else "")
                + (f"; {self.differ} equation(s) on which the readers "
                   f"differ are listed above for a look" if self.differ else "")
                + (f"; {len(self.ties)} scanned-table figure(s) the first two "
                   f"read otherwise, without a majority, are as the first "
                   f"reads them" if self.ties else "")
                + (f"\n    [!] the second reader could not be asked about "
                   f"{len(self.unanswered)} of those pages "
                   f"({self.why_unanswered}); they are as the first reader "
                   f"read them, and the next build will ask again"
                   if self.unanswered else ""))

    def reading(self, page_path: str) -> "str | None":
        """The second reader's reading of the page: its text, or None where
        it gave none or could not be asked. Only a reading it gave (or
        declined to give, by running on without stopping) is remembered. A
        page it could not be asked about is asked about again by the next
        build: a server that failed to start once must not leave every page
        of the book filed as having nothing to say."""
        return self._ask(page_key(page_path), lambda: Image.open(page_path))

    def _ask(self, key: str, picture, **task) -> "str | None":
        """The reader's answer filed under `key`: remembered from this run
        or an earlier one, or asked for now with the picture `picture()`
        opens (and `task`, a prompt and token limit other than the
        reader's own)."""
        self.consulted.add(key)
        if key in self._read:
            return self._read[key]
        if key in self.unanswered:
            return None
        if self.cache is not None:
            try:
                self._read[key] = self.cache.get(key)
                return self._read[key]
            except KeyError:
                pass
        try:
            with picture() as im:
                text = self.reader.read(im, **task)
        except Exception as exc:
            self.unanswered.add(key)
            self.why_unanswered = str(exc) or type(exc).__name__
            return None
        self.asked += 1
        self._read[key] = text
        if self.cache is not None:
            self.cache.put(key, text)
        return text

    def sheet_path(self) -> "str | None":
        """Where this build's account of the second reader goes: beside its
        readings, which are beside the first reader's cache."""
        path = getattr(self.cache, "path", None)
        if not path:
            return None
        stem = path[:-len(".second.gz")] if path.endswith(".second.gz") else path
        return stem + ".second.txt"

    def write_sheet(self) -> "str | None":
        """What was changed on the second reader's word and what was only
        listed, in full, where it outlasts the build's log. A build with
        nothing to say takes an earlier build's sheet away: it would
        describe a book that is no longer the one on the shelf."""
        path = self.sheet_path()
        if path is None:
            return None
        if not (self.changed or self.listed or self.ties):
            try:
                os.remove(path)
            except OSError:
                pass
            return None
        lines = ["What the second reader changed, and what it reads otherwise.",
                 "Pages are named as the book's files are: page_0034 is "
                 "page_0034.xhtml.", ""]
        if self.changed:
            lines.append(f"Printed otherwise than the first reader read it "
                         f"({len(self.changed)}):")
            for page, what, was, now, *why in self.changed:
                lines += ["", f"  {page}  {what}", f"      was: {was}",
                          f"      now: {now}"]
                lines += [f"      ({w})" for w in why]
            lines.append("")
        if self.listed:
            lines.append(f"The readers differ, and the first is printed "
                         f"({len(self.listed)}):")
            for page, first, theirs, *why in self.listed:
                lines += ["", f"  {page}  equation", f"      first:  {first}",
                          f"      second: {theirs}"]
                lines += [f"      ({w})" for w in why]
            lines.append("")
        if self.ties:
            lines.append(f"Scanned-table figures the first two read "
                         f"otherwise, with no majority, as the first reads "
                         f"them ({len(self.ties)}):")
            for page, first, theirs, why in self.ties:
                lines += ["", f"  {page}  table figure", f"      first:  {first}",
                          f"      second: {theirs}", f"      ({why})"]
            lines.append("")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines))
        except OSError:
            return None
        return path

    def close(self) -> None:
        if self.cache is not None:
            self.cache.save()
        try:
            self.reader.close()
        except Exception:
            pass


class ThirdOpinion(SecondOpinion):
    """A third reader's formula for one block of a page, asked for the
    first time two readers differ on it and kept beside the first reader's
    cache under the page's name and the block's box."""

    def formula(self, page_path: str, box) -> "str | None":
        """The block's equation as TeX, or None where the reader gave none
        or could not be asked."""
        from . import third as _third

        def crop():
            with Image.open(page_path) as im:
                return im.convert("RGB").crop(_third.crop_box(im.size, box))
        key = page_key(page_path) + "@" + ",".join(f"{v:.4f}" for v in box)
        return _third.formula(self._ask(key, crop))

    def table(self, page_path: str, box) -> "str | None":
        """The block's table as the reader writes it, or None where it gave
        none or could not be asked. Filed apart from a formula of the same
        box."""
        from . import third as _third

        def crop():
            with Image.open(page_path) as im:
                return im.convert("RGB").crop(_third.crop_box(im.size, box))
        key = (page_key(page_path) + "#table@"
               + ",".join(f"{v:.4f}" for v in box))
        return self._ask(key, crop, prompt=_third.TABLE,
                         max_tokens=_third.TABLE_MAX_TOKENS)

    def summary(self) -> "str | None":
        if not self.consulted:
            return None
        n = len(self.consulted)
        remembered = n - self.asked - len(self.unanswered)
        return (f"third reader: asked about {n} equation(s) or table(s) on "
                f"which the first two differ"
                + (f", {remembered} of them remembered" if remembered > 0 else "")
                + (f"\n    [!] the third reader could not be asked about "
                   f"{len(self.unanswered)} of them ({self.why_unanswered}); "
                   f"those were settled without it, and the next build will "
                   f"ask again" if self.unanswered else ""))


# A rebuilt table is questioned where it holds less than this share of the
# figures the PDF prints in its place, and the second reader's table is
# taken where it holds nearly all of them, holds clearly more than the first,
# and prints next to nothing the PDF does not print there. "There" is the
# first reader's box and a little round it: a table that lost its last
# column has a box that stops short of it too (Table 12 of Burning Up: the
# box held 55 of the table's 69 figures, and its header row sat above it).
#
# Measured on the 26 typeset pages of the shelf whose tables had lost most:
# the first reader's page held 78% of the 1,662 figures the PDF prints and
# the second's 91%. The second held more on seventeen of those pages and
# fewer on seven, which is why its table is taken only where the PDF's
# figures say it is the better one: six tables there, none of them with a
# figure the PDF does not print.
TABLE_QUESTIONED = 0.98


TABLE_TAKEN = 0.95


TABLE_BETTER_BY = 0.1


TABLE_REACH = 0.05          # round the first reader's box, of the page


TABLE_UNPRINTED = 0.05      # of the figures the second reader's table holds


def mend_tables(items: list, page_path: "str | None",
                second: "SecondOpinion | None"
                ) -> "tuple[list, list[tuple[int, int, int, list]]]":
    """Items with a table that lost figures replaced by the second reader's
    table of the same place, where the PDF's own text prefers it; and
    (figures printed, held before, held now, figures it holds that the PDF
    does not print) for each one replaced."""
    if second is None:
        return items, []
    tables = [it for it in items if it.html and it.box and it.figure is None
              and "<table" in it.html and not it.is_furniture]
    text_in = _typeset_text_in(page_path) if tables else None
    if text_in is None:
        return items, []
    out, mended, offered = list(items), [], None
    for it in tables:
        printed = _figures(text_in(it.box))
        total = sum(printed.values())
        if total < 6:
            continue
        first = _share_held(printed, _figures(_strip_tags(it.html)))
        if first >= TABLE_QUESTIONED:
            continue
        if offered is None:
            from . import second as _second
            offered = _second.tables(second.reading(page_path) or "")
        x0, y0, x1, y1 = it.box
        near = _figures(text_in((x0 - TABLE_REACH, y0 - TABLE_REACH,
                                 x1 + TABLE_REACH, y1 + TABLE_REACH)))
        best = None
        for cand in offered:
            if _share_held(printed, _figures(_strip_tags(cand))) \
                    < TABLE_TAKEN - TABLE_BETTER_BY:
                continue                # another table of the page
            held = _figures(_strip_tags(cand))
            unprinted = sorted((held - near).elements())
            share = _share_held(printed, held)
            if share >= TABLE_TAKEN and share >= first + TABLE_BETTER_BY \
                    and len(unprinted) <= max(1, TABLE_UNPRINTED
                                              * sum(held.values())) \
                    and (best is None or share > best[0]):
                best = (share, cand, unprinted)
        if best is None:
            continue
        html_ = _to_xhtml(best[1])
        if "<table" not in html_:
            continue
        html_ = re.sub(r"<table(?![^>]*\bborder=)", '<table border="1"', html_,
                       count=1)
        k = next(j for j, x in enumerate(out) if x is it)
        out[k] = it._replace(html=html_)
        mended.append((total, round(first * total), round(best[0] * total),
                       best[2]))
    return out, mended


# On a scan nothing printed referees between two readings of a table, and
# neither reader is right often enough alone: of the shelf's 370 scanned
# table pages both read as tables, 256 agree to within a figure in fifty, and
# on 29 the two read 81 cells otherwise, the second right more often than the
# first but not always (it made 1117's old-style "219" a "229"). A third
# reader shown the table's block breaks the tie: it holds one of the two
# figures and not the other for 46 of the 81, and judged against the pages
# the majority is the printed figure on seven of eight looked at, the one
# miss a figure all three readers misread alike (5498's year column, read
# 1712-39 for 1742-59, is the largest it mends: nineteen cells).
#
# Only a figure read otherwise is voted on: a pair of the same length, one
# digit apart, each at least two digits long. A figure the first prints more
# than once, or that both hold, is not a cell the vote can find, so it stays.
# So does a page number after a contents line's leader dots: the contents is
# placed by finding each title on the page it names, which checks the
# number better than a vote, and the vote's one wrong turn was such a number
# (1117's "219"), which moved the chapter's link ten pages on.
VOTE_SAME_TABLE = 0.6       # share of the first's figures the second's table
                            # must hold to be a reading of the same table
VOTE_MIN_FIGURES = 6


def _scanned(page_path: "str | None") -> bool:
    """Whether the page is known to be a scan: a page with no PDF of its own
    to ask, or one not yet described, is not."""
    try:
        with open(page_path + ".source.json") as fh:
            return bool(json.load(fh).get("scan"))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


_LEADER = re.compile(r"(?:[.…·_]\s?){3,}\s*$")


def _page_reference(text: str, figure: str) -> bool:
    """Whether `figure` stands after a run of leader dots in `text`: a
    contents line's page number."""
    return any(f.group().replace(",", "").rstrip(".") == figure
               and _LEADER.search(text[max(0, f.start() - 24):f.start()])
               for f in _FIGURE.finditer(text))


def _read_otherwise(first: "Counter[str]", second: "Counter[str]"
                    ) -> "list[tuple[str, str]]":
    """(first's figure, second's) for each cell the two read otherwise: the
    same length, one digit apart, neither a single digit."""
    mine = sorted((first - second).elements())
    theirs = sorted((second - first).elements())
    pairs = []
    for a in mine:
        for b in theirs:
            if len(a) == len(b) >= 2 and sum(x != y for x, y in zip(a, b)) == 1 \
                    and all(x == y or (x.isdigit() and y.isdigit())
                            for x, y in zip(a, b)):
                pairs.append((a, b))
                theirs.remove(b)
                break
    return pairs


def _swap_figure(fragment: str, was: str, now: str) -> "str | None":
    """The fragment with its one figure `was` (as _figures writes it) set as
    `now`, keeping the printed figure's commas; None unless the fragment
    prints `was` exactly once outside its tags."""
    at = next(i for i, (x, y) in enumerate(zip(was, now)) if x != y)
    spots = []
    for m in re.finditer(r"<[^>]*>|([^<]+)", fragment):
        if m.group(1) is None:
            continue
        for f in _FIGURE.finditer(m.group(1)):
            if f.group().replace(",", "").rstrip(".") == was:
                spots.append((m.start(1) + f.start(), f.group()))
    if len(spots) != 1:
        return None
    start, token = spots[0]
    k = -1
    for j, ch in enumerate(token):
        if ch == ",":
            continue
        k += 1
        if k == at:
            return fragment[:start + j] + now[at] + fragment[start + j + 1:]
    return None


def vote_scanned_tables(items: list, page_path: "str | None",
                        second: "SecondOpinion | None",
                        third: "ThirdOpinion | None"
                        ) -> "tuple[list, list[tuple[str, str]], list[tuple[str, str, str]]]":
    """Items with each scanned table's figures that the first two readers
    read otherwise set by a vote of three: the second reader's figure where
    the third reader, shown the table's block, holds it and not the first's.
    Returns the items, (as it was, as it is) for each figure set, and (first,
    second, why) for each kept as the first reads it."""
    if second is None or third is None or not _scanned(page_path):
        return items, [], []
    tables = [it for it in items if it.html and it.box and it.figure is None
              and "<table" in it.html and not it.is_furniture]
    if not tables:
        return items, [], []
    from . import second as _second
    offered = None
    out, changed, ties = list(items), [], []
    for it in tables:
        mine = _figures(_strip_tags(it.html))
        if sum(mine.values()) < VOTE_MIN_FIGURES:
            continue
        if offered is None:
            offered = [_figures(_strip_tags(t))
                       for t in _second.tables(second.reading(page_path) or "")]
        if not offered:
            break
        held = max(offered, key=lambda c: sum((mine & c).values()))
        if sum((mine & held).values()) < VOTE_SAME_TABLE * sum(mine.values()):
            continue
        pairs = _read_otherwise(mine, held)
        if not pairs:
            continue
        fragment, words = it.html, _strip_tags(it.html)
        pairs = [(a, b) for a, b in pairs if not _page_reference(words, a)
                 or ties.append((a, b, "a contents line's page number"))]
        if not pairs:
            continue
        voted = _figures(_strip_tags(third.table(page_path, it.box) or ""))
        for a, b in pairs:
            if mine[a] > 1 or held[b] > 1:
                ties.append((a, b, "printed more than once: no one cell"))
                continue
            if not voted[b] or voted[a]:
                ties.append((a, b, "the third reader holds "
                             + ("both" if voted[a] and voted[b] else
                                "the first's" if voted[a] else "neither")))
                continue
            swapped = _swap_figure(fragment, a, b)
            if swapped is None:
                ties.append((a, b, "not found once in the table's markup"))
                continue
            fragment = swapped
            changed.append((a, b))
        if fragment != it.html:
            k = next(j for j, x in enumerate(out) if x is it)
            out[k] = it._replace(html=fragment)
    return out, changed, ties
