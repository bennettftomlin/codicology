"""The book's contents: the printed contents parsed and each entry placed
on the page it names, chapter openings found and their heads set as one
rank, the running heads' chapters merged in where the printed contents
missed them, and, where nothing parses, what the layout labelled as
headings and as a contents page, read from each page's blocks.
"""
from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING, NamedTuple

from .folios import ROMAN_FOLIO_BASE
from .furniture import _same_head
from .markup import _roman_to_int, _strip_tags
from .model import Page

if TYPE_CHECKING:
    from .folios import Folio
    from .furniture import Chapter


def labelled_heads(pages: "list[Page]") -> "list[tuple[int, int, str]]":
    """Every heading the layout labelled (SectionHeader): the page it stands
    on, its rank from its <hN>, and its text, cut at ninety characters. A
    bare number is a note marker the layout took for a heading, and is left
    out."""
    out = []
    for p in pages:
        for b in p.blocks:
            if b.label != "SectionHeader" or not b.html:
                continue
            t = re.sub(r"\s+", " ", _strip_tags(b.html)).strip()
            rank = re.match(r"\s*<h([1-6])", b.html)
            if t and not re.fullmatch(r"\d{1,3}", t):
                out.append((p.index, int(rank.group(1)) if rank else 3, t[:90]))
    return out


def labelled_contents_pages(pages: "list[Page]") -> "set[int]":
    """The pages on which the layout labelled a printed contents."""
    return {p.index for p in pages for b in p.blocks if b.label == "TableOfContents"}


# How a book designates a division before naming it: "Chapter 1", "PART
# ONE", "III.", "7". A contents line usually carries the designation and
# the page it points at usually does not, so both are stripped before the
# titles are compared. Four digits are left alone — a chapter called "1984"
# is a title, not a number.
_DESIGNATION = r"(?:chapter|part|section|book|appendix)"


# A number a book spells out. Only ever read where a designation introduces
# it: bare, "one" opens too many real titles to strip — "One Hundred Years
# of Solitude" is not chapter one of anything.
#
# Longest spelling first, and a boundary at the end. Alternation takes the
# first branch that fits, not the longest: with the small numbers leading,
# "FOURTEEN" matched "four" and left "TEEN" standing as the chapter's title.
# That renumbered fourteen as four in the body AND in the endnotes, so the
# two could no longer be aligned and every note link in the book was
# withheld. Either guard alone would do; both are cheap.
_NUMBER_WORD = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|ten|eleven|twelve"
    r"|one|two|three|four|five|six|seven|eight|nine)\b")


_CHAPTER_MARK = re.compile(
    r"^\s*(?:"
    rf"{_DESIGNATION}\s+{_NUMBER_WORD}"
    rf"|{_DESIGNATION}?\s*(?:\d{{1,3}}|[ivxlc]{{1,7}})"
    r")\s*[:.\-—]?\s+", re.I)


def _bare_title(t: str) -> str:
    """A division's title with its designation taken off."""
    return " ".join(_CHAPTER_MARK.sub("", t).split())


def _names_chapter(heading: str, head: str) -> bool:
    """Whether a heading on the page is the chapter this running head names.

    A running head ABBREVIATES, and it abbreviates in one particular way: it
    drops the SUBTITLE. The head reads "INTRODUCTION" where the page is
    titled "Introduction: Going Inside", "BEING THERE" where the page says
    "Being There: Social Life in the Centre" — scoring 0.59 and 0.48, so
    similarity alone matches neither. What licenses the match is the colon.
    A plain space does not: "Border" must not claim the section called
    "Border security", and on real books every abbreviation observed was cut
    at punctuation, never mid-phrase.
    """
    a = " ".join(heading.lower().split())
    b = " ".join(head.lower().split())
    if _same_head(a, b):
        return True
    # A contents line and the page it points at rarely agree about the
    # chapter's number: the contents says "Chapter 1 Understanding the
    # Strategic Context" and the page is headed by the title alone, or the
    # reverse. Strip the designation from both and compare what is left.
    ba, bb = _bare_title(a), _bare_title(b)
    if ba and bb and _same_head(ba, bb):
        return True
    a, b = ba or a, bb or b
    if len(b) < 6 or len(a) < 6:
        return False
    lo, hi = (b, a) if len(b) <= len(a) else (a, b)
    if hi.startswith(lo) and hi[len(lo):len(lo) + 1] in (":", ".", ",",
                                                         "—", "–", "-"):
        return True
    # The mirror image: a head that abbreviates to the SUBTITLE. One book's
    # margins run THE COLLAPSE OF THE MIDDLE AGES where the contents line is
    # "CHAPTER ONE. “The King's in His Castle …”: The Collapse of the
    # Middle Ages" — the head is the post-colon half, and matching only the
    # front half called five of its chapters unaccounted for. The same
    # guards apply: cut at punctuation, never mid-phrase.
    rest = hi[:len(hi) - len(lo)].rstrip()
    return hi.endswith(lo) and bool(rest) and rest[-1] in (":", ".", ",",
                                                           "—", "–", "-")


def promote_chapter_headings(bodies: list[str],
                             chapters: "list[Chapter]") -> int:
    """Put each chapter's own title at the top of the document outline.

    Measured across the shelf, h1 is mostly occupied by the title page —
    often in pieces, and often misread ("Inthropology'sWorld", "SEX &
    POLYNER") — while the chapters that actually structure the book sit at
    h2 or h3, and one manual reached the reader with no h1 at all. A
    reading system builds its outline from these tags, so the book's
    skeleton was the one thing they did not describe.

    Only a heading whose text is the chapter's own name, on the page where
    that chapter opens or the one after, is promoted. Nothing is demoted:
    a title page's h1 is not wrong, it is just not a chapter, and moving it
    would trade one arguable outline for another.
    """
    n = 0
    floor = 0
    for c in chapters:
        # The head names a chapter from the page AFTER its opening, and on a
        # verso-title book the opening can be two leaves back. Search a small
        # window, never past where the previous chapter ended.
        window = [p for p in range(c.start - 2, c.start + 2)
                  if floor <= p < len(bodies)]
        floor = c.end + 1
        for pi in window:
            if not 0 <= pi < len(bodies):
                continue
            hit = False
            for m in re.finditer(r"<h([1-6])([^>]*)>(.*?)</h\1>",
                                 bodies[pi], re.S):
                rank, attrs, inner = m.group(1), m.group(2), m.group(3)
                text = " ".join(_strip_tags(inner).split())
                if rank == "1" or not _names_chapter(text, c.title):
                    continue
                bodies[pi] = (bodies[pi][:m.start()]
                              + f"<h1{attrs}>{inner}</h1>"
                              + bodies[pi][m.end():])
                n += 1
                hit = True
                break
            if hit:
                break
    return n


# What a division of a book calls itself, in the contents and on its own
# title page: "BOOK SEVEN", "Part III", "SECTION 2". The row branch has
# always required this before grouping a folio-less line; the heading
# branch now does too, so a book's own title reprinted atop a second
# contents page cannot adopt the chapters beneath it.
# The book's apparatus, as a contents page names it. These lines belong to
# no chapter: they are the front and back matter, and a chapter that
# appears to subordinate them is an artefact of their being the only plain
# rows on a page of bold ones.
_APPARATUS_WORD = re.compile(
    r"\b(notes?|references?|index|bibliography|works cited|"
    r"acknowledge?ments?|preface|foreword|afterword|"
    r"glossary|chronolog(?:y|ies)|appendix|appendices|abbreviations|contributors|"
    r"further reading|about the author|credits|permissions)\b", re.I)


def APPARATUS_ROW(title: str):
    """Whether a contents line names the book's apparatus rather than a
    part of its argument.

    Usually the word opens the line. But a book may qualify it — "PERSONAL
    ACKNOWLEDGMENTS" — and matching only at the start stopped the
    back-matter walk dead there, stranding the NOTES and BIBLIOGRAPHY
    printed above it inside the last chapter. So a short line counts
    wherever the word falls. Short is the whole guard: at four words the
    test starts claiming real sections ("The PCE Price Index"), while
    every apparatus line a book actually prints is one or two.
    """
    t = " ".join((title or "").split())
    return _APPARATUS_WORD.match(t) or (
        _APPARATUS_WORD.search(t) if len(t.split()) <= 3 else None)


# What a contents row can name that is not a division of the book. A
# figure caption set in the same weight as a chapter title reads as a
# chapter to any typographic test — one book promoted "Exhibit 3.2",
# "Exhibit 3.5" and "Exhibit 3.6" to chapter rank and left Chapter One and
# Chapter Two as leaves beneath them.
NOT_A_DIVISION = re.compile(
    r"\s*(exhibit|figure|fig\.|table|plate|map|chart|diagram|box|"
    r"illustration|appendix table)\b", re.I)


PART_HEAD = re.compile(
    r"\s*(BOOK|PART|VOLUME|SECTION|APPENDI(X|CES))\b", re.I)


class TocEntry(NamedTuple):
    """One line of a book's own table of contents, as printed."""
    title: str
    folio: int | None        # None: an unnumbered line. Roman is carried
                             # as i - ROMAN_FOLIO_BASE, as the audit carries it
    folio_text: str          # the number as printed, roman numerals included
    depth: int               # 0 = top grouping, 1 = subgroup, 2 = leaf
    # How the page itself declared this line's rank. A typesetter marks a
    # chapter by setting it bold and indents what belongs under it; both
    # survive recognition and both were being discarded at parse time.
    bold: bool = False
    indent: int = 0
    src_page: int = -1


# A contents line that is nothing but the designation — the book puts
# "CHAPTER ONE" on one line and names the chapter on the next.
_DESIGNATION_ONLY = re.compile(
    rf"^\s*{_DESIGNATION}\s+(?:\d{{1,3}}|[ivxlcdm]+|{_NUMBER_WORD})"
    r"\s*[.:—–-]?\s*$", re.I)


# A line carrying no title at all: a folio, roman or arabic, or the date
# range that finishes the title above it.
_FOLIO_ONLY = re.compile(r"^\s*(?:[ivxlcdm]+|[\d\s.,:–—-]+)\s*$", re.I)


def _line_text(el) -> str:
    """An element's text with its line breaks kept as spaces.

    A contents entry set over two lines carries a <br/>, and text_content()
    closes it up: one book's chapter one arrived as "All's Right with the
    World":The Collapse of the Middle Ages, which then matched no heading
    anywhere in the book.
    """
    for br in el.iter("br"):
        br.tail = " " + (br.tail or "")
    return " ".join((el.text_content() or "").split())


def _fold_contents_lines(lines: "list[str]") -> "list[str]":
    """Put a designation back together with the title it introduces.

    A contents page with no table gives each of these its own line, and read
    separately neither says what chapter it is: sixteen chapters arrived as
    thirty-two entries, half of them bare numbers. The same page also breaks
    a long title across lines, and where it does the first half ends in a
    comma — which is the only evidence offered that the line is unfinished.
    """
    out: "list[str]" = []
    for line in lines:
        starts_one = bool(_DESIGNATION_ONLY.match(line))
        folio = bool(_FOLIO_ONLY.match(line))
        unfinished = bool(out) and out[-1].rstrip().endswith((",", ":"))
        if out and _DESIGNATION_ONLY.match(out[-1]) and not starts_one \
                and not folio:
            sep = "" if out[-1].rstrip().endswith((":", ".", "-", "—", "–")) else "."
            out[-1] = f"{out[-1].rstrip()}{sep} {line}"
        elif out and unfinished and not starts_one:
            # An unfinished line. A colon or a comma ending a contents line
            # is the book saying the title continues, and on a page with no
            # rows to align it is the only such evidence offered. This is
            # also the one thing that keeps a year range: "…and Socialism,"
            # is finished by "1871–1914" and by nothing else.
            out[-1] = f"{out[-1].rstrip()} {line}"
        elif folio:
            continue                 # a page number, standing on its own
        else:
            out.append(line)
    return out


# A page calls itself a contents page, as against merely mentioning the word.
# "The top should be opened and the contents allowed to melt slowly" is a
# manual talking about a thermos, and reading it as a contents page harvested
# a page of prose about canteens into the book's structure. What separates
# them is the article: a label stands on its own, a common noun is preceded
# by one. "TABLE OF CONTENTS" keeps its "of" and is unaffected.
_NAMES_ITSELF_CONTENTS = re.compile(
    r"(?<!\bthe )(?<!\bits )(?<!\btheir )(?<!\bthis )(?<!\bwhose )contents\b")


# The table's own column headings. They sit in a row like any other and,
# where the leading cell is blank, look exactly like a subordinate entry.
_COLUMN_HEADING = re.compile(
    r"^(?:page|pages|contents|chapter|section|paragraphs?|title|no\.?)$", re.I)


def _blank_lead_runs(body: str) -> bool:
    """Whether this page's blank-leading-cell rows ever come two in a row.

    Both a subsection and a chapter's byline are set the same way — the
    number cell left empty, the title cell holding text — so the row alone
    cannot say which it is. What tells them apart is company. Sections come
    in runs; a byline is one line naming who wrote the chapter above it, and
    a book that gives every chapter an author gives every one of them
    exactly one. One edited collection listed nine contributors that way and
    each arrived as a subsection of the chapter they had written.
    """
    run = 0
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        cells = re.findall(r"<td[^>]*/>|<td[^>]*>(.*?)</td>", row, re.S)
        vals = [" ".join(re.sub(r"<[^>]+>", "", c or "").split()) for c in cells]
        if len(vals) >= 2 and not vals[0] and vals[1]:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def _leads_indented_rows(el, indent: int, need: int = 2) -> bool:
    """Whether the contents rows after this one are set further in than it.

    A part banner carries no folio, and so does a stray line of prose that
    happened to begin with the word Section. What separates them is what
    follows: a grouping has its members set under it. Two are required, so
    that one indented straggler cannot promote a line on its own.
    """
    deeper = 0
    for sib in el.itersiblings("tr"):
        raw = [td.text_content() or "" for td in sib.iter("td", "th")]
        cells = [" ".join(r.split()) for r in raw]
        if not any(cells):
            continue
        if next((len(r) - len(r.lstrip())
                 for r, c in zip(raw, cells) if c), 0) <= indent:
            return False
        deeper += 1
        if deeper >= need:
            return True
    return False


def _peer_banners(el, word: str) -> bool:
    """Whether the page prints other folio-less lines of the same designation.

    A book that sets its chapters flush and indents the part banners instead
    — one here does exactly that — offers no positional evidence at all.
    What it does offer is a series: Part I, Part II, Part III, each printed
    the same way and each without a page number. A line of prose that merely
    opens with the word Section keeps no such company.
    """
    for sib in (list(el.itersiblings("tr", preceding=True))
                + list(el.itersiblings("tr"))):
        cells = [c for c in (" ".join((td.text_content() or "").split())
                             for td in sib.iter("td", "th")) if c]
        if len(cells) != 1:
            continue
        m = PART_HEAD.match(cells[0])
        if m and m.group(1).lower() == word:
            return True
    return False


def parse_printed_toc(bodies: list[str], limit: int = 25,
                      book_title: "str | None" = None
                      ) -> tuple[list[TocEntry], set[int]]:
    """
    The book's own contents pages, read back as structure.

    The layout pass renders a printed contents page as a real table — one row
    per line, folio in its own right-aligned cell, group headings as <h3>/<h4>
    — so this is a parse, not a reconstruction. Books put their own names on
    their chapters; deriving a contents list any other way guesses at what the
    book already states.

    A page qualifies as a contents page if it says so in its opening text, or
    if it directly continues one and still looks like one (folio-terminated
    rows). TiHKAL's contents run two pages and only the first is titled.
    """
    try:
        from lxml import html as lhtml
    except ImportError:
        return [], set()
    entries: list[TocEntry] = []
    toc_pages: set[int] = set()
    prev_was_toc = False
    for page_i, body in enumerate(bodies[:limit]):
        text_open = _strip_tags(body)[:1000].lower()
        rows_here = body.count("</tr>")
        # A contents page need not announce itself. This one opens straight
        # into "Preface . . . . . vii" and was skipped for want of the word
        # "contents", costing the book its whole structure. What makes a
        # printed contents page recognisable is the dot leader — a run of
        # dots carrying the eye to a folio — which no ordinary table has.
        leaders = len(re.findall(r"[.·…]\s*[.·…]\s*[.·…]", body))
        # Rows or list items: the layout renders a block it has labelled a
        # contents as <li> items rather than a table.
        lines_here = rows_here + body.count("</li>")
        looks_like_contents = lines_here >= 5 and leaders >= max(3, lines_here // 3)
        # A bare list of titles runs past its first page as bare paragraphs,
        # with no rows to recognise it by — the Saylor contents names its
        # first 23 chapters on one page and its last 11 on the next, and
        # stopping at the page break loses a third of the book's structure.
        short_lines = len([m for m in re.finditer(r"<p>([^<]{3,120})</p>", body)
                           if re.search(r"[A-Za-z]{3}", m.group(1))])
        is_toc = (bool(_NAMES_ITSELF_CONTENTS.search(text_open))
                  or looks_like_contents
                  or (prev_was_toc and (rows_here >= 5 or short_lines >= 5)))
        prev_was_toc = is_toc
        if not is_toc:
            continue
        toc_pages.add(page_i)
        try:
            root = lhtml.fragment_fromstring(body, create_parent="div")
        except Exception:
            continue
        subordinate_runs = _blank_lead_runs(body)
        # A contents page that lists titles and no page numbers at all — the
        # Saylor economics volumes name 34 chapters as bare paragraphs. There
        # is nothing to resolve against the folio map, so these carry no folio
        # and are placed later by hunting the title through the book. Only
        # taken when the page yields no rows, so a real contents table is
        # never competing with the prose around it.
        if not body.count("</tr>"):
            # Headings as well as paragraphs. A page with no table still
            # sets its designations as headings — one book puts CHAPTER ONE
            # through SEVEN in <h2> and the rest in <p>, and reading only
            # the paragraphs lost the first half of the book. They are not
            # part divisions here: nothing on a table-less page is ranked,
            # so everything arrives at one level, as before.
            # List items as well as paragraphs: the layout renders a block
            # it has labelled a contents as <li> items rather than a table
            # — one book's seventeen chapters arrived that way and the nav
            # was built from nothing. And a line that ends in a dot leader
            # and a folio is a complete entry, folio and all; it is read
            # here as the table branch reads a row, so the nav is not left
            # to hunt titles it was handed the pages for.
            pending: list = []

            def flush():
                for t in _fold_contents_lines(pending):
                    entries.append(TocEntry(t, None, "", 2))
                pending.clear()

            for el in root.iter("p", "h1", "h2", "h3", "h4", "li"):
                raw_t = " ".join(_line_text(el).split())
                led = re.match(r"^(.*?\S)\s*[.·…]{2,}[\s.·…]*(\d{1,3}|[ivxlcdm]{1,7})$",
                               raw_t, re.I)
                t = re.sub(r"[\s.·…]+$", "", raw_t)
                # A folio is carried this far, and no further: it is the only
                # thing that can finish a title broken before its year range,
                # and the fold drops it when it finishes nothing.
                if not (t and len(t) <= 120):
                    continue
                if led and re.search(r"[A-Za-z]{3}", led.group(1)) \
                        and "contents" not in led.group(1).lower():
                    flush()
                    title = re.sub(r"[\s.·…]+$", "", led.group(1))
                    folio_text = led.group(2)
                    folio = _contents_folio(folio_text)
                    entries.append(TocEntry(title, folio, folio_text, 2, False, 0, page_i))
                    continue
                if not _FOLIO_ONLY.match(t) and not (
                        3 <= len(t) and re.search(r"[A-Za-z]{3}", t)):
                    continue
                if "contents" in t.lower():
                    continue
                if book_title and _same_head(t, book_title):
                    continue
                pending.append(t)
            flush()
            continue
        for el in root.iter():
            if el.tag in ("h1", "h2", "h3", "h4"):
                t = " ".join((el.text_content() or "").split())
                # A heading on a contents page is a part division — or it
                # is the book announcing itself again. A contents running
                # onto a second page reprints its own title at the head of
                # it, and taken as a division that title adopted every
                # chapter below: chapters 5 and 6, the notes and the index
                # all hung under "AFTER QUEER THEORY" in one book's nav.
                # The test is that identity, not a vocabulary of division
                # words — one book's parts are headed "HOW THE NORTHWEST
                # WAS LOST TO FRANCE", which names no division and is one.
                if t and "contents" not in t.lower() \
                        and not (book_title and _same_head(t, book_title)):
                    entries.append(TocEntry(t, None, "", 0 if el.tag in ("h1", "h2", "h3") else 1))
            elif el.tag == "tr":
                tds = list(el.iter("td", "th"))
                # A cell holding a title set over two lines carries a <br/>,
                # and text_content() closes it up: "Migrant Communities" and
                # "in the United Kingdom" arrived as "Communitiesin", which
                # then matched no heading in the book. The tail carries the
                # space so the leading whitespace an indent is read from is
                # left alone.
                for br in el.iter("br"):
                    br.tail = " " + (br.tail or "")
                raw = [td.text_content() or "" for td in tds]
                cells = [" ".join(r.split()) for r in raw]
                # the title is every cell but the folio; its own typography
                # is the book's statement about rank
                bold = any(len(td.findall(".//b")) or len(td.findall(".//strong"))
                           for td in tds[:-1] or tds)
                indent = next((len(r) - len(r.lstrip())
                               for r, c in zip(raw, cells) if c), 0)
                cells = [c for c in cells if c]
                if not cells:
                    continue
                folio_text = cells[-1]
                title = " ".join(cells[:-1]) if len(cells) > 1 else ""
                # printed dot leaders survive OCR as trailing runs of dots
                title = re.sub(r"[\s.·…]+$", "", title)
                m = re.fullmatch(r"(\d{1,3})", folio_text)
                cm = re.fullmatch(r"(\d{1,2})-(\d{1,3})", folio_text)
                # the appendix form the furniture reader already parses:
                # "A-12" encodes exactly as folios_from_furniture encodes
                # it, so folio_resolver can match the two (B4)
                lm = re.fullmatch(r"([A-Z])-([1-9]\d{0,2})", folio_text)
                roman = re.fullmatch(r"[ivxlc]+", folio_text.lower())
                if lm and title:
                    entries.append(TocEntry(
                        title,
                        (100 + ord(lm.group(1)) - ord("A")) * 1000
                        + int(lm.group(2)), folio_text, 2, bold, indent, page_i))
                elif cm and title:
                    entries.append(TocEntry(title, int(cm.group(1)) * 1000
                                            + int(cm.group(2)), folio_text, 2,
                                            bold, indent, page_i))
                elif m and title:
                    entries.append(TocEntry(title, int(m.group(1)), folio_text,
                                            2, bold, indent, page_i))
                elif roman and title:
                    entries.append(TocEntry(title, _contents_folio(folio_text),
                                            folio_text, 2, bold, indent, page_i))
                else:
                    # A row with no folio at all is usually noise — but a
                    # part heading is printed exactly this way: eagle's
                    # contents groups its chapters under "BOOK SEVEN: The
                    # Revolt" lines that carry no page number, and dropping
                    # them flattened the book's own declared structure.
                    # Length was standing in for structure. It fits a banner
                    # named "BOOK SEVEN: The Revolt" and not one named for
                    # everything it contains — "Part I: YUGOSLAVIA, GREECE,
                    # POLAND AND LATVIA – Between the blocs" runs eleven
                    # words and was thrown away, taking the book's four
                    # parts with it. A line that the rows beneath it are
                    # indented under is a grouping however long its name,
                    # and that is the same evidence the depth rule reads.
                    # A chapter that puts its number in a cell of its own
                    # and prints no page against it. The number is not a
                    # folio and the title is not a page, so the row parsed
                    # as neither and was dropped whole — three chapters of
                    # one book, whose other two put the number and the title
                    # in a single cell and came through. Rejoining them
                    # gives every chapter the shape the survivors had.
                    if (len(cells) == 2
                            and re.fullmatch(r"\d{1,3}|[ivxlcdm]{1,7}",
                                             cells[0], re.I)
                            and re.search(r"[A-Za-z]{3}", cells[1])):
                        entries.append(TocEntry(f"{cells[0]} {cells[1]}", None,
                                                "", 2, bold, indent, page_i))
                        continue
                    full = re.sub(r"[\s.·…]+$", "", " ".join(cells))
                    words = len(full.split())
                    mark = PART_HEAD.match(full)
                    if mark and words >= 2 and (
                            words <= 8
                            or _leads_indented_rows(el, indent)
                            or _peer_banners(el, mark.group(1).lower())):
                        entries.append(TocEntry(full, None, "", 0))
                    elif (len(cells) == 1 and len(raw) >= 2
                            and subordinate_runs
                            and not raw[0].strip()
                            and 3 <= len(full) <= 120
                            and re.search(r"[A-Za-z]{3}", full)
                            and not _FOLIO_ONLY.match(full)
                            and not _COLUMN_HEADING.match(full)):
                        # A subsection printed with no page against it. What
                        # says so is the EMPTY cell in front of it: this book
                        # sets a chapter as number-cell plus title-cell, and
                        # everything under that chapter with the number cell
                        # left blank. The row parsed as neither a title nor a
                        # folio and was dropped, taking every subsection of
                        # three chapters with it.
                        #
                        # The empty leading cell is the whole guard. Keeping
                        # any lone folio-less cell instead pulled in column
                        # headings ("Page", "CONTENTS") and, from one book,
                        # its copyright page — LIBRARY OF CONGRESS, Two
                        # Copies Received, JAN 27 1908.
                        entries.append(TocEntry(full, None, "", 2, bold,
                                                indent, page_i))
    # The prefix rule first: where a book numbers its subsections (1)…(6)
    # that is an explicit declaration and outranks an inference from type.
    # Where it does not, the page's own setting is the declaration.
    # Indentation is relative to the page it was set on. One manual sets
    # its early chapters at four spaces with their sections at eight, and
    # then, pages later, sets chapters flush left — so a baseline taken
    # across the whole contents made the flush chapters heads and demoted
    # the indented ones to sit beside their own sections. Each page's
    # setting is measured against itself.
    # Only rows carry a measured indent. A heading on the page — a part
    # division, or the book's own name — is not set in the table and
    # arrives at zero by default, which would make it the page's baseline
    # and every real row look indented beneath it.
    by_page: dict = {}
    for e in entries:
        if e.depth == 2:
            by_page.setdefault(e.src_page, []).append(e.indent)
    # Per page, and by the same test: the floor is the shallowest indent
    # that several rows share. A single flush-left row — the manual's own
    # name above its chapters — is an outlier, not the page's baseline.
    from collections import Counter as _C
    floor = {}
    for pg, v in by_page.items():
        tally = _C(v)
        real = [ind for ind, n in tally.items() if n >= 2]
        floor[pg] = min(real) if real else min(v)
    entries = [e._replace(indent=e.indent - floor.get(e.src_page, 0))
               for e in entries]

    ranked = _lift_trailing_apparatus(_infer_row_depths(entries))
    if ranked == entries:            # the prefix rule found nothing to read
        ranked = _lift_trailing_apparatus(_depths_from_typography(ranked))
    return _close_group_on_outdent(ranked), toc_pages


def _close_group_on_outdent(entries: "list[TocEntry]") -> "list[TocEntry]":
    """End a part when the contents stops setting lines inside it.

    A row below a part banner belongs to that part, which is right until the
    book finishes with its parts and goes back to speaking for itself. One
    book's Conclusion is set flush with its Introduction while every chapter
    inside a part is indented under it — so once the parts were read, the
    Conclusion, being merely the next line after the last of them, was filed
    inside Part IV.

    A line cannot belong to a group whose members are set further in than it
    is. That is the whole test: if anything between the banner and this row
    is indented deeper than this row, the group has closed.
    """
    out = list(entries)
    open_group = False
    inside = None                # how far in this group's members are set
    for i, e in enumerate(entries):
        if e.depth == 0:
            open_group, inside = True, None
            continue
        if not open_group:
            continue
        if inside is None:
            inside = e.indent
        elif e.indent < inside:
            out[i] = e._replace(depth=0)
            inside = None
    return out


def _lift_trailing_apparatus(entries: "list[TocEntry]") -> "list[TocEntry]":
    """The book's back matter belongs to the book, not to its last chapter.

    A contents ends with its apparatus — notes, bibliography, index,
    glossary, the author's note — set at the same indent as a chapter's
    subsections. Nested by that indent alone, all of it disappeared inside
    whichever chapter happened to come last: fourteen such entries across
    seven books, including one where the index sat under a chapter it has
    nothing to do with.

    Position is what separates the two cases, not the words. A References
    line BETWEEN two chapters is that chapter's, and stays where it is;
    only the run at the very END is the book's own, and it is lifted to
    the top level where the book prints it.
    """
    # Only where the contents has a hierarchy to be buried in. A flat
    # list has no last chapter to fall into, and lifting its tail would
    # invent the structure this exists to correct.
    if not any(e.depth < 2 for e in entries):
        return entries
    # Lift to whatever rank this book's top level IS. A book divided into
    # parts puts them at 0 and its chapters at 1, so lifting the back
    # matter to 1 made it a sibling of the CHAPTERS — which the nav then
    # nests inside the last part, exactly the burial this is meant to
    # undo. Four books came out unchanged for that reason.
    top = min(e.depth for e in entries)
    out = list(entries)
    for i in range(len(out) - 1, -1, -1):
        if out[i].depth != 2 or not APPARATUS_ROW(out[i].title):
            break
        out[i] = out[i]._replace(depth=top)
    return out


def _infer_row_depths(entries: "list[TocEntry]") -> "list[TocEntry]":
    """The printed page declares its own hierarchy in the entries' prefixes:
    Russian Purge lists roman-numeral chapters with runs of (1)…(6)
    subsections beneath them, and a nav that flattens all forty-nine lines
    is our failure, not the book's. Rows parse flat; this pass reads the
    declaration back. Only parenthesized prefixes are believed — (1), (a) —
    and only when at least three rows carry one, so a flat contents stays
    flat. A row is promoted to section rank only when a subsection actually
    follows it: a chapter with none stays a plain leaf, because an empty
    grouping in the nav is noise."""
    sub = re.compile(r"^\([0-9a-z]{1,3}\)\s")
    rows = [i for i, e in enumerate(entries) if e.depth == 2]
    n_sub = sum(1 for i in rows if sub.match(entries[i].title))
    if n_sub < 3 or n_sub == len(rows):
        return entries
    out = list(entries)
    for k, i in enumerate(rows):
        if sub.match(entries[i].title):
            continue
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        if nxt is None or not sub.match(entries[nxt].title):
            continue
        # the subsection must FOLLOW this chapter, not open the next BOOK:
        # an intervening lower-depth heading between the two rows means
        # the (1) belongs to someone else, and promoting across it makes
        # exactly the empty grouping this pass exists to avoid (B3)
        if any(e.depth < 2 for e in entries[i + 1:nxt]):
            continue
        if NOT_A_DIVISION.match(entries[i].title):
            continue
        out[i] = entries[i]._replace(depth=1)
    return out


def _depths_from_typography(entries: "list[TocEntry]") -> "list[TocEntry]":
    """The other way a contents page declares its hierarchy: by setting it.

    Prefixes are one convention. The commoner one is typographic — the
    chapter is bold, and what belongs under it is indented — and both cues
    survive recognition into the markup, where the parser was discarding
    them. One book listed forty entries this way and shipped every one of
    them flat, chapters level with their own subsections.

    Bold leads, because it is the sturdier of the two: a contents that runs
    across two pages can lose its indentation on the second (measured: cells
    at indent 0 throughout) while the bold still marks every chapter.
    Indentation is read only where nothing is bold.

    The guards are the prefix rule's, for the same reasons: three of each
    kind at least, so a uniformly bold contents stays flat; and a row is
    promoted only where a subsection actually follows it, so no chapter
    gains an empty grouping.
    """
    rows = [i for i, e in enumerate(entries) if e.depth == 2]
    if len(rows) < 6:
        return entries
    heads = [i for i in rows if entries[i].bold]
    if len(heads) < 3 or len(heads) == len(rows):
        # Nothing bold, or everything: fall back to the indentation, which
        # says the same thing when the compositor used it instead.
        # The baseline is the shallowest indent that is a LEVEL rather than
        # an outlier. One manual prints its own name as a single flush-left
        # row above chapters set at four spaces; taking the plain minimum
        # made that one row the baseline, left every chapter looking
        # indented beneath it, and wrapped the book's nav in its own title.
        # Taking the commonest indent instead fails the other way, since
        # subsections usually outnumber chapters. A level needs members.
        from collections import Counter
        tally = Counter(entries[i].indent for i in rows)
        levels = [ind for ind, n in tally.items() if n >= 3]
        if not levels:
            return entries
        base = min(levels)
        heads = [i for i in rows if entries[i].indent == base]
        if len(heads) < 3 or len(heads) == len(rows):
            return entries
        subs = {i for i in rows if i not in heads}
    else:
        subs = {i for i in rows if not entries[i].bold}
    # The apparatus is not a subsection. Where a book lists chapters and no
    # subsections at all, its contents is bold chapters and a plain tail of
    # Notes, References and Index — and the last chapter, being the only one
    # a plain row follows, adopted all three. Three books flattened that way
    # into "7 Conclusion" parenting the back matter. A row that names the
    # apparatus is evidence of nothing.
    subs = {i for i in subs if not APPARATUS_ROW(entries[i].title)}
    if len(subs) < 3:
        return entries
    # Every row at the head level is a head. Requiring a subsection to
    # follow each one seemed safe — it stops an empty grouping — but rank
    # is a property of the level, not of what happens to come next: one
    # book's Chapter Five is set exactly like its Chapter Four and was
    # left inside it, because the only row following Five was its
    # References, which the apparatus test had already set aside. A
    # chapter with nothing under it is a top-level entry, which is what
    # the book prints. The guard that matters is the one above: unless
    # this contents really has subsections, none of this runs at all.
    out = list(entries)
    for i in rows:
        if i not in heads or NOT_A_DIVISION.match(entries[i].title):
            continue
        out[i] = entries[i]._replace(depth=1)
    return out


def _contents_folio(text: str) -> "int | None":
    """A contents line's printed folio, in the encoding the folio audit uses.

    Roman included. Front matter paginates in Roman and the audit already
    reads Roman off a running head, carrying it as i - ROMAN_FOLIO_BASE so
    the whole numbering space stays ordered. The contents parser recognised
    the same numerals and threw them away — "FIGURES ..... iv" arrived with
    no address at all — and a line with no address falls through to the
    title hunt, which only moves FORWARD. One manual's list of figures was
    placed 233 pages past the page its own contents named, and because the
    hunt's floor moves with every placement, every line after it inherited
    that floor: four front-matter entries and a notes section lost.

    _roman_to_int is strict and stops at 99, which is what front matter is.
    """
    t = text.strip()
    if t.isdigit():
        # Three digits at most, the same bar parse_folio holds a running
        # head to: a book prints no page 1997, and a contents line that
        # ends in a year would otherwise be given one as an address.
        return int(t) if 1 <= len(t) <= 3 and int(t) else None
    if re.fullmatch(r"[ivxlc]+", t, re.I):
        r = _roman_to_int(t)
        if r is not None:
            return r - ROMAN_FOLIO_BASE
    return None


def _folio_series(n: int) -> int:
    """Which numbering a folio belongs to: Roman front matter, plain Arabic,
    or one of the compound forms. Interpolating across two of them is
    meaningless arithmetic — the distance from folio iv to folio 7 is not a
    number of pages — so the resolver only ever reasons within one."""
    return -1 if n < 0 else n // 1000


def folio_resolver(folios: "list[Folio]") -> "callable":
    """
    Printed folio → position in the book, exact where read, pinned where not.

    Chapter openings print no running head, so the very pages a contents line
    points at are the ones with no direct reading. Their neighbours hold them
    in place: with hundreds of confident anchors, the page carrying folio F
    sits a known distance from the nearest anchor on each side, and both sides
    must agree before the answer is trusted.
    """
    by_series: dict = {}
    for f in folios:
        if f.confident and f.number is not None:
            by_series.setdefault(_folio_series(f.number), []).append(
                (f.number, f.index))
    for v in by_series.values():
        v.sort()

    def resolve(folio: int) -> int | None:
        anchors = by_series.get(_folio_series(folio), [])
        if not anchors:
            return None
        import bisect
        i = bisect.bisect_left(anchors, (folio, -1))
        if i < len(anchors) and anchors[i][0] == folio:
            return anchors[i][1]
        left = anchors[i - 1] if i > 0 else None
        right = anchors[i] if i < len(anchors) else None
        cand_l = left[1] + (folio - left[0]) if left else None
        cand_r = right[1] - (right[0] - folio) if right else None
        if cand_l is not None and cand_r is not None:
            return cand_l if cand_l == cand_r else None   # neighbours disagree: refuse
        return cand_l if cand_l is not None else cand_r

    return resolve


def _title_on_page(title: str, text: str) -> bool:
    """Whether enough of a contents line's words appear in a page's text."""
    words = [w for w in re.findall(r"[a-z0-9']+", title.lower()) if len(w) > 2]
    if not words:
        return False
    hay = text.lower()
    hits = sum(1 for w in words if w in hay)
    # Rounded UP: for a two-word title, 60% rounded down is one word, and one
    # word is how "Comments on the Title" matched the cover art. Every word
    # short of the fraction has to be there.
    need = -(-len(words) * 3 // 5)
    return hits >= max(1, need)


def _title_names_this_page(title: str, text: str) -> bool:
    """
    A stricter title test, for lines with no folio to corroborate them.

    The ordinary test asks whether enough of a title's words appear, matching
    substrings so plurals still count. Hunting a chapter across a whole book
    that is far too easy to satisfy: "Chapter 7: The Analysis of Consumer
    Choice" matched a section called "The Analysis of Maximizing Behavior"
    because the prose beneath it mentioned consumers. Here every LONG word of
    the title must be present — the distinctive ones, which prose in the
    neighbourhood does not supply by accident.
    """
    # A chapter announces itself by printing its title, whole and unbroken.
    # Bag-of-words cannot tell that apart from a page that merely discusses
    # the same subject: "Chapter 10: Monopoly" matched a page inside chapter
    # 9 that mentioned monopoly while saying "in this chapter", and "The
    # Analysis of Consumer Choice" matched a section called "The Analysis of
    # Maximizing Behavior" whose prose spoke of consumers. A contiguous
    # phrase is not supplied by accident.
    flat = lambda t: " " + re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() + " "
    hay = flat(text)
    needle = flat(title)
    if len(needle) > 6 and needle in hay:
        return True
    # A contents line carries the chapter's number; the opening page often
    # prints only its name. "I THE INVISIBLE GOVERNMENT" — where the OCR read
    # a printed 1 as a roman I — never matched the page that says just "The
    # Invisible Government", and chapter one lost its link. Try again without
    # the leading numeral, roman or arabic, which the page is entitled to
    # omit and the contents is entitled to carry.
    bare = re.sub(r"^\s*(?:[0-9]{1,3}|[ivxlcIVXLC]{1,5})[\s.:)-]+", "", title)
    if bare != title:
        needle = flat(bare)
        # Without its numeral a chapter title is just the book's own words,
        # and a half-title page says exactly that and nothing else. A chapter
        # OPENS: it carries its title and then it starts talking.
        if len(needle) > 6 and needle in hay and len(text.split()) >= 40:
            return True
    return False


def _place_toc_entries(bodies: list[str], folios,
                       dropped: "set[int]" = frozenset(),
                       book_title: "str | None" = None):
    """Resolve each printed-contents line to a page, refusing placements that
    break the contents' own order.

    Resolution goes wrong in one specific, nasty way: a chapter opener prints
    its chapter NUMBER where a folio normally sits, the furniture pass reads
    it as a folio, and some other chapter's contents line then resolves onto
    that page exactly. Title checking cannot be relied on to catch it —
    "2 48 HOURS" verified happily on the page whose prose says "48 hours".
    What does catch it is the contents page itself: its lines run in reading
    order, so the pages they resolve to must ascend. Placements off the
    longest non-decreasing run are refused rather than emitted wrong.
    """
    printed, toc_pages = parse_printed_toc(bodies, book_title=book_title)
    resolve = folio_resolver(folios)
    placed: list[tuple] = []          # (entry, target | None, verified)
    for e in printed:
        # groupings without a printed folio have nothing to resolve; a
        # promoted chapter row still carries its folio and resolves like
        # any leaf
        if e.folio is None and e.depth < 2:
            placed.append((e, None, False))
            continue
        target = resolve(e.folio) if e.folio is not None else None
        verified = False
        if target is not None:
            # a contents line may point a page or two off after drops and
            # unread folios; the title's own words settle it
            cands = [t for t in (target, target + 1, target - 1)
                     if 0 <= t < len(bodies) and t not in dropped]
            hit = next((t for t in cands
                        if _title_on_page(e.title, _strip_tags(bodies[t]))),
                       None)
            if hit is not None:
                target, verified = hit, True
        placed.append((e, target, verified))
    leaf = [(k, t) for k, (e, t, _) in enumerate(placed) if t is not None]

    def lis(seq):
        n = len(seq)
        best, prev = [1] * n, [-1] * n
        for i in range(n):
            for j in range(i):
                if seq[j][1] <= seq[i][1] and best[j] + 1 > best[i]:
                    best[i], prev[i] = best[j] + 1, j
        i = max(range(n), key=lambda k: best[k])
        out = set()
        while i != -1:
            out.add(seq[i][0])
            i = prev[i]
        return out

    if len(leaf) >= 3:
        # A printed contents is not always ONE ascending sequence: a manual
        # follows its chapters with a list of figures whose folios start
        # over, and a single longest-run pass kept the figure list while
        # refusing every chapter. So long runs are extracted repeatedly —
        # each is a real sequence the book printed — and only the short
        # residue is refused, which is still exactly what a poisoned anchor
        # leaves behind: a spike belongs to no long run.
        # The first run is the book's principal sequence and is kept at any
        # length, as the single-pass guard always did. A FURTHER run must be
        # long enough to be a second printed list rather than salvage of
        # scattered spikes.
        keep: set = lis(leaf)
        remaining = [x for x in leaf if x[0] not in keep]
        while len(remaining) >= 5:
            run = lis(remaining)
            if len(run) < 5:
                break
            keep |= run
            remaining = [x for x in remaining if x[0] not in run]
        for k, t in leaf:
            if k not in keep:
                e, _, _ = placed[k]
                placed[k] = (e, None, False)
    return placed, toc_pages


# A bold block standing on its own where a numbered group head belongs — at
# the fragment start or hard against another block's close, never bold inside
# running text, which is emphasis and must not be promoted.
BOLD_HEAD_BLOCK = re.compile(
    r"(^|</(?:p|h\d|ol|ul|table|figure)>)(\s*)<b>\s*(\d{1,2})\.\s+([^<]+?)\s*</b>")


# A heading that opens a chapter: the word, then the number the book counts
# in. Roman is admitted because half the books here number that way, and the
# strict parser refuses anything that is merely made of those letters. Books
# that spell the number out count too — two here do, and leaving them out
# left sixty openings split between a heading holding the number and a
# heading holding the title, neither able to say what chapter it was.
# The number must end where it ends. Without the boundary, "Chapter
# Introduction" was chapter I with the title "ntroduction", and "Chapter
# Vital Signs" chapter V, "ital Signs".
CHAPTER_HEAD = re.compile(
    r"<(h[1-6])[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    rf"(CHAPTER|Chapter)\s+(\d{{1,3}}|[IVXL]{{1,7}}|(?i:{_NUMBER_WORD}))"
    r"(?!\w)\s*[.:—-]?\s*(.*?)</\1>", re.S)


_WORD_VALUE = {w: v for v, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split(), start=1)}


_WORD_VALUE.update({w: v for v, w in zip(
    range(20, 100, 10),
    "twenty thirty forty fifty sixty seventy eighty ninety".split())})


def _chapter_head_number(token: str) -> "int | None":
    if token.isdigit():
        return int(token)
    parts = [p for p in re.split(r"[-\s]+", token.lower()) if p]
    if parts and all(p in _WORD_VALUE for p in parts):
        return sum(_WORD_VALUE[p] for p in parts)
    return _roman_to_int(token)


def normalize_chapter_heads(bodies: list[str]) -> int:
    """
    Give every chapter opening the same rank, and its title back.

    The layout pass reads each page alone, so it has no way to notice that it
    called chapter five an h1 and chapter six an h2 — Principles of
    Macroeconomics came out seven of one and fourteen of the other for the
    same job. It also splits the opening in two, the number in one heading
    and the title in the next, which leaves neither able to say on its own
    what chapter this is.

    Both are repaired from the same recognition. Rank goes to the SHALLOWEST
    level the book itself used for a chapter: a chapter is the outermost
    division a book names, so where the layout ever managed to see one as
    top-level, that is what it is — taking the commonest instead would have
    pushed macroeconomics' chapters down to h2, level with the very titles
    they own. The title is then folded back into the heading, giving the
    shape the books that got it right already have ("CHAPTER 1. INVASION").

    Only an exactly-numbered heading is touched, so a book whose openings are
    already whole is left alone, and a heading carrying prose is never
    merged into.
    """
    found = []                                   # (page, match, level)
    for i, body in enumerate(bodies):
        for m in CHAPTER_HEAD.finditer(body):
            if _chapter_head_number(m.group(3)) is None:
                continue
            found.append((i, m, int(m.group(1)[1])))
    if len(found) < 2:
        return 0
    rank = min(level for _, _, level in found)
    changed = 0
    for i in sorted({p for p, _, _ in found}, reverse=True):
        body = bodies[i]
        edits = []
        for m in CHAPTER_HEAD.finditer(body):
            if _chapter_head_number(m.group(3)) is None:
                continue
            word, num = m.group(2), m.group(3)
            title = re.sub(r"\s+", " ", _strip_tags(m.group(4))).strip()
            end = m.end()
            if not title:
                # the title is the next heading along, if that is what it is:
                # adjacent, no deeper than a title should be, and short enough
                # to be one rather than a paragraph that got promoted
                nxt = re.match(r"\s*<(h[1-6])[^>]*>(.*?)</\1>", body[m.end():], re.S)
                if nxt and int(nxt.group(1)[1]) >= rank:
                    cand = re.sub(r"\s+", " ", _strip_tags(nxt.group(2))).strip()
                    if 0 < len(cand) <= 90 and not re.match(
                            r"^(CHAPTER|Chapter)\s", cand):
                        title, end = cand, m.end() + nxt.end()
            label = f"{word} {num}" + (f". {title}" if title else "")
            new = f"<h{rank}>{html.escape(label)}</h{rank}>"
            if body[m.start():end] != new:
                edits.append((m.start(), end, new))
        for a, z, new in sorted(edits, reverse=True):
            body = body[:a] + new + body[z:]
            changed += 1
        bodies[i] = body
    return changed


def furniture_absent_from_contents(placed, furn_chapters, pos_of):
    """The chapters the running heads describe that the contents never gave
    the reader: no placed entry lands on the chapter's opening pages, and no
    entry's title names it.

    Title cover counts even for an entry that failed to place — the contents
    still declares that chapter, and adding a second copy from the margins
    would duplicate it rather than rescue it.
    """
    targets = {t for _, t, _ in placed if t is not None}
    # This expects the RESOLVED placements — the list nav_from_placed hands
    # back after every title hunt has settled. Judged before the hunt, a
    # wholly folio-less contents has no targets at all, and one book merged
    # all nineteen of its chapters as "missing" while its nav named every
    # one. Only entries the reader actually gets can cover: resolved
    # leaves, and groupings, which emit as headings even unresolved. An
    # unresolved leaf is skipped from the nav entirely, and counting it as
    # cover blocked the rescue in exactly the case rescue helps.
    emitted = [e for e, t, _ in placed if t is not None or e.depth < 2]

    def covered(c):
        # A head names its chapter from the page after the opening — two
        # leaves back on a verso-title book — so the window is the one the
        # heading promoter already uses, not a tighter one.
        if any(abs(t - c.start) <= 2 for t in targets):
            return True
        return any(_names_chapter(e.title, c.title)
                   or _names_chapter(c.title, e.title)
                   for e in emitted)

    return [c for c in furn_chapters
            if c.start in pos_of and not covered(c)]


def merge_missing_furniture(placed, missing, furn_total):
    """Give the nav the chapters the margins describe, where the printed
    contents plainly failed to.

    The printed contents is the book's own declaration and outranks an
    inference from its running heads — which is enforced per chapter, not
    per book: a chapter counts as missing only when no placed entry lands
    within the verso window of its opening and no entry the reader will
    actually see names it, under the same fuzzy matching the heads
    themselves are grouped by. A chapter that survives all of that is
    evidenced by a sustained run of its own margins and absent from the
    nav, and there is no number of its peers that makes leaving it out
    better for the reader. An aggregate gate stood here once — at least
    three missing and half the book's — and its only effect was to keep
    proven chapters out of thin navs.

    Merged, never replaced: everything the contents did place is kept, and
    the absent chapters are added in page order, each anchored where its
    own running heads begin. The one refusal is a title with no run of
    three letters — one manual's appendix heads are the bare letters A and
    B, real destinations but meaningless as labels.
    """
    depth = min((e.depth for e, _, _ in placed), default=2)
    out = list(placed)
    merged = 0
    # Resolution before insertion. Where an UNRESOLVED entry — leaf or
    # grouping — names the chapter the margins located, the entry is given
    # the margins' page rather than a copy of the margins' words: the
    # contents keeps its own title and its own position, the furniture
    # supplies the destination, and nothing duplicates. Inserted alongside
    # instead, one book's chapter three arrived twice — an abbreviated
    # head-link above the real grouping, with the neighbouring chapter's
    # section caught between them. The head may be cut mid-word (MEDIA
    # CONTENT, 2 is the front of a title ending 2006), so the naming test
    # here is looser than coverage's, and safely so: one direction only,
    # the head abbreviating the entry, and its worst failure is a link on
    # a heading that already stands where the book put it.
    lent: set = set()
    unresolved = [(k, e) for k, (e, t, _) in enumerate(placed) if t is None]
    for c in sorted(missing, key=lambda c: c.start):
        if not re.search(r"[A-Za-z]{3}", c.title):
            continue
        head_core = re.sub(r"[^a-z]", "", _bare_title(c.title.lower()))
        home = None
        for k, e in unresolved:
            if k in lent:
                continue
            entry_core = re.sub(r"[^a-z]", "", _bare_title(e.title.lower()))
            if (_names_chapter(e.title, c.title)
                    or _names_chapter(c.title, e.title)
                    or (len(head_core) >= 8
                        and entry_core.startswith(head_core))):
                home = k
                lent.add(k)
                break
        if home is not None:
            e, _t, ok = out[home]
            out[home] = (e, c.start, ok)
            merged += 1
            continue
        at = next((k for k, (_, t, _) in enumerate(out)
                   if t is not None and t > c.start), len(out))
        out.insert(at, (TocEntry(c.title, None, "", depth), c.start, False))
        merged += 1
    return out, merged


def contents_confidence(placed, n_pages, missed, furn_unmerged):
    """Warnings for a contents parse that deserves a human eye.

    Every bad parse found on this shelf had its signature sitting in the
    build log, printed and never compared: seven entries for a 288-page
    book, chapters four and five printed with one to three nowhere, the
    margins describing chapters the nav never mentioned. Each check here is
    a comparison the log was already in a position to make. Warnings only —
    nothing downstream reads them.
    """
    warns = []
    n = len(placed)
    if n and n_pages >= 60 and n * 40 < n_pages:
        warns.append(f"contents parse looks sparse: {n} entr"
                     f"{'y' if n == 1 else 'ies'} for {n_pages} pages")
    if n and missed >= max(3, n // 4):
        warns.append(f"{missed} of {n} contents entries never placed "
                     f"in the book")
    if furn_unmerged >= 3:
        warns.append(f"running heads describe {furn_unmerged} chapter(s) "
                     f"the contents never placed")
    nums = set()
    for e, _, _ in placed:
        t = e.title.strip()
        m = re.match(r"chapter\s+([A-Za-z0-9-]+)", t, re.I)
        v = _chapter_head_number(m.group(1)) if m else None
        if v is None:
            m = re.match(r"(\d{1,3})\b(?!\s*[-\u2013])", t)
            v = int(m.group(1)) if m else None
        if v is not None and v < 200:
            nums.add(v)
    if len(nums) >= 2:
        gaps = sorted(set(range(1, max(nums))) - nums)
        if len(gaps) >= 2:
            shown = ", ".join(map(str, gaps[:6]))
            warns.append(f"chapters {min(nums)}\u2013{max(nums)} are printed "
                         f"but {len(gaps)} number(s) never parsed: {shown}"
                         + (", \u2026" if len(gaps) > 6 else ""))
    return warns


def nav_from_placed(placed, pos_of, kept, bodies, toc_pages,
                    make_link, make_section) -> tuple:
    """The printed contents as a nav tree: links, nested sections, and the
    hunt for folio-less lines. Extracted from build_epub so the invariants
    the 2026-08-18 review found broken here are testable: every id unique
    (C1/C2 — one monotonic counter, no frozen list lengths), the title
    hunt always moving forward from the last thing placed, folio-resolved
    or hunted alike (C3), and an unresolvable grouping degrading to a bare
    heading.

    make_link(href, title, uid) / make_section(title, href|None) keep
    ebooklib out of the function's signature — tests pass plain tuples.
    Returns (links, verified, missed, resolved), where resolved is the
    placed list with every title hunt settled — the after-the-fact truth
    of what emitted and where, which is what the furniture check needs:
    measured against the pre-hunt list, a wholly folio-less contents
    looks entirely unplaced and every chapter looks missing."""
    links: list = []
    resolved: list = []
    verified = missed = 0
    next_id = iter(range(10 ** 9))
    empties: list = []
    # Titles are hunted in reading order, starting past the contents —
    # a book's half-title repeats the chapter's words before the book
    # has begun, and would otherwise win chapter one every time.
    hunt_from = max(toc_pages) if toc_pages else -1
    stack: list = [(-1, links)]
    stripped_cache: dict = {}

    def stripped(t):
        if t not in stripped_cache:
            stripped_cache[t] = _strip_tags(bodies[t])
        return stripped_cache[t]

    for e, target, ok in placed:
        if ok:
            verified += 1
        elif target is not None and target not in pos_of:
            target = None
        if target is None:
            # No folio to resolve: the title text is the only address the
            # line has. Front matter sits in the opening pages, but a
            # chapter list without page numbers points anywhere in the
            # book, so the search runs forward from the last thing placed
            # — which keeps the contents in their own order for free — and
            # demands the title near the TOP of a page, where a chapter
            # announces itself, so a passing mention in prose cannot win.
            # The contents pages are excluded: every title appears there
            # by definition, which is how "Foreword" once pointed at the
            # table that listed it.
            search = [t for t in kept if t > hunt_from and t not in toc_pages]
            target = next((t for t in search
                           if _title_names_this_page(
                               e.title, stripped(t)[:300])),
                          None)
            if target is None and hunt_from < 0:
                # Front matter — a Preface, a Foreword — sits in the
                # opening pages and carries roman folios the audit never
                # sees. Only before anything else has been placed, and
                # only on the same strict terms: this fallback once put
                # "Chapter 3: Demand and Supply" on chapter 2's opening
                # page, which shared two of its three words.
                target = next((t for t in kept[:30] if t not in toc_pages
                               and _title_names_this_page(
                                   e.title, stripped(t)[:400])),
                              None)
        if target is not None and target in pos_of:
            # forward from the last thing placed means EVERY placement,
            # folio-resolved included — leaving hunt_from behind on
            # resolved entries let a later hunt match backward into an
            # already-passed chapter (C3)
            hunt_from = max(hunt_from, target)
        resolved.append((e, target if target in pos_of else None, ok))
        if e.depth < 2:
            # a grouping: BOOK SEVEN as printed, or a chapter promoted
            # because subsections follow it. A chapter that nests
            # children is still a destination, so it keeps its link
            # when it resolved; a grouping that never resolves is a
            # heading and no more.
            sect: list = []
            while stack and stack[-1][0] >= e.depth:
                stack.pop()
            href = (f"page_{target:04d}.xhtml"
                    if target is not None and target in pos_of else None)
            head = make_section(e.title, href)
            slot = [(head, sect)]
            stack[-1][1].append(slot)
            # A grouping that ends up with no children must not ship as
            # one: ebooklib writes an empty <ol/> for it, and an <ol> with
            # no <li> is invalid in a nav document. Recorded here so the
            # pass below can turn a childless grouping back into the plain
            # link it always was.
            empties.append((slot, e.title, href))
            stack.append((e.depth, sect))
            continue
        if target is None or target not in pos_of:
            missed += 1
            continue
        # one monotonic counter for every id: the old ids froze len(links)
        # inside sections and reused page numbers across same-rank heads,
        # shipping duplicate ids in nav.xhtml (C1/C2)
        stack[-1][1].append(make_link(f"page_{target:04d}.xhtml", e.title,
                                      f"toc{next(next_id)}-{pos_of[target]}"))
    # Childless groupings become links again, or vanish if they never
    # resolved to a page — either way no empty <ol> reaches the reader.
    for slot, title, href in empties:
        if slot[0][1]:
            continue
        slot[0] = (("link", make_link(href, title, f"toc{next(next_id)}-g"))
                   if href else None)
    def _flatten(seq):
        out = []
        for item in seq:
            if isinstance(item, list):          # a grouping's slot
                item = item[0]
                if item is None:                # never resolved: drop it
                    continue
                if item[0] == "link":           # pruned to a plain link
                    out.append(item[1])
                else:
                    out.append((item[0], _flatten(item[1])))
            else:
                out.append(item)
        return out
    return _flatten(links), verified, missed, resolved


def promote_missing_chapter_heads(bodies: list[str]) -> int:
    """
    Recover a chapter opening the layout pass ran into the prose.

    A chapter that begins on a fresh page is easy: its number and title stand
    alone and come back as headings. One that begins halfway down a page,
    under the previous chapter's exercises, does not — Principles of
    Macroeconomics welded the whole of chapter nine's opening onto the end of
    chapter eight's last problem, "…over the two decades. Chapter 9 The
    Nature and Creation of Money Start Up: How Many Macks…". The reader lost
    the chapter break itself, not merely a line in the contents.

    "Chapter 9" also appears in ordinary prose — a textbook cross-references
    its own chapters constantly — so recognition alone would be reckless.
    What makes this safe is that only a HOLE is filled: the number must be
    one the sequence is missing, it must lie between the pages of the
    chapters either side of it, it must open a sentence, and it must be the
    only such mention in that range. A cross-reference fails all of these at
    once; the genuine opening passes them by construction.

    The title is deliberately left where it is. Where it ends in run-together
    prose cannot be known — "…Creation of Money Start Up: How Many Macks…"
    offers no honest boundary — and a heading that swallows half the next
    one would be invention. The break is what was lost; the break is what is
    restored.
    """
    found = []
    for i, body in enumerate(bodies):
        for m in CHAPTER_HEAD.finditer(body):
            no = _chapter_head_number(m.group(3))
            if no is not None:
                found.append((no, i, int(m.group(1)[1])))
    if len(found) < 2:
        return 0                              # nothing to be a gap between
    found.sort()
    rank = min(level for _, _, level in found)
    seen = {no: page for no, page, _ in found}
    pages_in_order = [seen[no] for no in sorted(seen)]
    if any(b < a for a, b in zip(pages_in_order, pages_in_order[1:])):
        return 0                              # sequence disagrees with the book
    filled = 0
    for missing in range(min(seen) + 1, max(seen)):
        if missing in seen:
            continue
        lo, hi = seen.get(missing - 1), seen.get(missing + 1)
        if lo is None or hi is None or hi <= lo:
            continue
        word = "CHAPTER" if any(
            CHAPTER_HEAD.search(bodies[p]) and
            CHAPTER_HEAD.search(bodies[p]).group(2) == "CHAPTER"
            for p in (lo, hi)) else "Chapter"
        pat = re.compile(
            r"(?<=[.!?])(\s+)(" + word + r"\s+" + str(missing) + r")\s+(?=[A-Z])")
        hits = [(p, m) for p in range(lo, hi + 1)
                for m in pat.finditer(bodies[p])]
        if len(hits) != 1:
            continue
        p, m = hits[0]
        body = bodies[p]
        # Split the paragraph: what came before stays in it, the opening
        # becomes a heading of the rank its siblings use, the remainder
        # resumes in a fresh paragraph.
        head = f"</p><h{rank}>{word} {missing}</h{rank}><p>"
        bodies[p] = body[:m.start()] + head + body[m.end():]
        filled += 1
    return filled


def chapter_head_contents(bodies: list[str]) -> "list[tuple[int, str]]":
    """
    (page, label) for each chapter opening, as a contents list of last resort.

    Some books simply have no printed contents page — Principles of
    Macroeconomics runs copyright, preface, chapter one — and until now that
    meant the reader got one navigation entry per page, eight hundred and
    twenty-six of them reading "Page 1", "Page 2". The book still knows where
    its chapters begin; nothing was asking it. This is only ever consulted
    when the printed page yielded nothing, so a real contents always wins.
    """
    out, seen = [], set()
    for i, body in enumerate(bodies):
        for m in CHAPTER_HEAD.finditer(body):
            no = _chapter_head_number(m.group(3))
            if no is None or no in seen:
                continue
            seen.add(no)
            label = re.sub(r"\s+", " ", _strip_tags(m.group(0))).strip()
            if label:
                out.append((i, label))
    # numbers must climb with the pages, or this is not a chapter sequence
    if len(out) < 3:
        return []
    nums = [_chapter_head_number(CHAPTER_HEAD.search(bodies[i]).group(3))
            for i, _ in out]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        return []
    return out
