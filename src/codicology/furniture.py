"""Running heads and feet: the lines a book repeats at the edge of its pages,
set aside from the text wherever the layout or the book's own repetition
shows them for what they are, and read for the chapters they name.
"""
from __future__ import annotations

import difflib
import html
import re
from typing import NamedTuple
from .folios import parse_folio
from .markup import RUNNING_HEAD_MAX_CHARS, _SETTING, _strip_tags
from .model import Page, PageItem


# A running head that the layout pass missed, recognised by its shape instead:
# a short line at the very top or bottom of a page, carrying a page number at
# one end. Kept deliberately narrow — it only ever examines the first and last
# fragment of a page, so a numeral opening a sentence mid-page is never at risk.
RUNNING_HEAD = re.compile(r"""
    ^\s*(?: \d{1,3}\s*[•·|:]          # "28 • Citadel of Sin"
         |  \d{1,3}\s*$               # a bare folio on its own line
    )""", re.X)


RUNNING_FOOT = re.compile(r"[•·|:]\s*\d{1,3}\s*$")   # "Chapter 13 • 199"


# How many separate pages must carry the same picture before it stops being a
# figure and becomes decoration. Two is a book showing one thing twice, which
# happens and is content — a chart repeated for reference, a plate reproduced
# in a later chapter. Three is a pattern.
FURNITURE_PAGES = 3


def strip_running_heads(items: list[PageItem]) -> list[PageItem]:
    """
    Drop the running head and folio from the ends of a page.

    The layout pass labels most of them and they never reach here. It misses
    some, and those are worth catching by shape, because a running head is the
    one line on the page guaranteed to be wrong in an EPUB: the text reflows,
    the page it numbered no longer exists, and the title interrupts a sentence
    that ran across the page break.
    """
    if not items:
        return items
    keep = list(items)
    for end in (0, -1):
        while keep:
            item = keep[end]
            if item.figure is not None or item.is_caption:
                break
            text = _strip_tags(item.html).strip()
            if not text or len(text) > RUNNING_HEAD_MAX_CHARS:
                break
            # Either shape can appear at either end: this book numbers versos at
            # the front of the line ("28 • Citadel of Sin") and rectos at the
            # back ("Chapter 13 • 199"), and both sit at the top of the page.
            if not (RUNNING_HEAD.match(text) or RUNNING_FOOT.search(text)):
                break
            keep.pop(end)
    return keep


def head_shape(text: str) -> str:
    """A running head with its folio masked, so every page's is one shape."""
    masked = re.sub(r"\d+", "#", " ".join(text.split()))
    return re.sub(r"[^A-Za-z#]+", " ", masked).strip().lower()


# A block with nothing block-level inside it, which is what a running head is.
# Written so it can never swallow a figure, a list or a table by accident.
_LEAF_BLOCK = (r"<(p|h[1-6])\b[^>]*>"
               r"(?:(?!</?(?:p|h[1-6]|figure|div|table|ul|ol|blockquote)\b).)*?"
               r"</\1>")


_FIRST_LEAF = re.compile(r"\A\s*" + _LEAF_BLOCK, re.S)


_LAST_LEAF = re.compile(_LEAF_BLOCK + r"\s*\Z", re.S)


# The same head, not given a block of its own: the layout pass put the
# running head and the first line of prose in ONE paragraph with a <br/>
# between them. One book carries 21 of these, and they survived the first
# version of this pass untouched — it read a block's WHOLE text, which for
# a fused head is the head plus the paragraph it interrupts, far past any
# length a running head has. The check written to confirm that fix shared
# the same assumption and reported the book clean.
_INLINE = r"(?:(?!<br\s*/?>|</?(?:p|h[1-6]|figure|div|table|ul|ol|blockquote)\b).)*?"


_FIRST_FUSED = re.compile(r"\A\s*<(?:p|h[1-6])\b[^>]*>(" + _INLINE
                          + r")(<br\s*/?>\s*)", re.S)


_LAST_FUSED = re.compile(r"(\s*<br\s*/?>)(" + _INLINE
                         + r")</(?:p|h[1-6])>\s*\Z", re.S)


_OPENS_HEADING = re.compile(r"\A\s*<h[1-6]\b")


_ENDS_HEADING = re.compile(r"</h[1-6]>\s*\Z")


# Four places a head can sit: alone in the page's first or last block, or
# fused by a <br/> into the front of the first block or the back of the
# last. Whole blocks first, so a page that has both is not read through the
# wrong lens.
_HEAD_SITES = ((_FIRST_LEAF, True, False), (_LAST_LEAF, False, False),
               (_FIRST_FUSED, True, True), (_LAST_FUSED, False, True))


# A running head that carries no folio is known by its rhythm instead: the
# same line opening page after page, never more than this many pages between
# one sighting and the next — every page, or every recto.
HEAD_CADENCE = 2


def _head_site(body: str, pattern, at_top: bool, fused: bool):
    """The span a running head at this site would be cut from, and its text,
    or None where the page has no such block."""
    m = pattern.search(body)
    if m is None:
        return None
    if fused:
        # the head is one group and the break the other; the span to cut
        # covers both, so the paragraph closes up behind it
        return ((m.start(1), m.end(2)),
                " ".join(_strip_tags(m.group(1 if at_top else 2)).split()))
    return (m.start(), m.end()), " ".join(_strip_tags(m.group(0)).split())


def _head_key(text: str) -> str:
    """One line however each page happens to set it: case, the setting of
    its quotes and dashes, and its spacing aside."""
    return " ".join(text.translate(_SETTING).split()).casefold()


def _head_runs(bodies: list[str], page_furniture: list[list[str]]) -> dict:
    """{(line, page): (sightings in its run, the run's first page)} for every
    short line that opens a page, or that the layout called furniture there,
    split into runs wherever more than HEAD_CADENCE pages pass without it."""
    seen: dict[str, set] = {}
    for i, texts in enumerate(page_furniture):
        for t in texts:
            seen.setdefault(_head_key(t), set()).add(i)
    for i, body in enumerate(bodies):
        for site in _HEAD_SITES:
            if not site[1]:
                continue
            hit = _head_site(body, *site)
            if hit and hit[1] and len(hit[1]) <= RUNNING_HEAD_MAX_CHARS:
                seen.setdefault(_head_key(hit[1]), set()).add(i)
    runs = {}
    for key, pages in seen.items():
        pages = sorted(pages)
        start = 0
        for k in range(1, len(pages) + 1):
            if k < len(pages) and pages[k] - pages[k - 1] <= HEAD_CADENCE:
                continue
            for p in pages[start:k]:
                runs[(key, p)] = (k - start, pages[start])
            start = k
    return runs


def _interrupts_a_sentence(before: str, rest: str) -> bool:
    """Whether a line at the head of a page stands inside a sentence: the
    page before ends its last paragraph without finishing it, and the
    paragraph after the line goes on in lower case. A chapter or section
    never opens there, so such a line is never its own heading."""
    last = _LAST_LEAF.search(before)
    first = _FIRST_LEAF.search(rest)
    if last is None or first is None or last.group(1) != "p" \
            or first.group(1) != "p":
        return False
    tail = _strip_tags(last.group(0)).split()
    head = _strip_tags(first.group(0)).strip()
    # The page must stop on a word: a page that ends on an address or a
    # figure has stopped on a reference, and a lower-case line can follow
    # it with nothing interrupted (Working the Phones ends its notes on a
    # web address and opens its index, "absence 21", under INDEX).
    return bool(tail and head) and head[0].islower() \
        and re.fullmatch(r"[(\[“‘\"']?[^\W\d_][\w’'-]*[,-]?", tail[-1]) is not None


def _worded_head(text: str) -> bool:
    """Whether a line of page furniture carries words, as a running head
    does. A designation does not: "SWM 6-77" names a paragraph of the
    manual as a page number names a page, "AGO 8641A" is the printer's
    and "DOI: 10.21983/…" the publisher's. One short word in capitals
    beside a figure is that, and a page that carries it may still have a
    running head the layout took for a heading: the Small Wars Manual sets
    one under its designation on 198 pages, 115 of them in the middle of a
    sentence that runs over the page."""
    text = html.unescape(text)
    words = [w for w in re.findall(r"[A-Za-z]+", text)
             if not re.fullmatch(r"[ivxlcdm]+", w, re.I)]
    if not any(len(w) >= 3 for w in words):
        return False
    longer = [w for w in words if len(w) >= 2]
    return not (len(longer) == 1 and len(longer[0]) <= 3
                and longer[0].isupper() and re.search(r"\d", text))


def strip_missed_running_heads(bodies: list[str],
                               page_furniture: list[list[str]],
                               pages: "list | None" = None
                               ) -> list[tuple[int, str]]:
    """
    Drop the running heads the layout pass called content, and say which.

    The layout model is not consistent about its own furniture. On one book
    it labelled the line "10 THE REBEL PASSION" a PageHeader on 209 pages
    and a SectionHeader on 49 — the same place on the page, the same 0.99
    confidence — so 47 running heads shipped inside the prose, and 35 of
    them as HEADINGS, dropped into sentences that ran across the page
    break. No test of shape or position can separate the two groups,
    because there is no difference between them to find. Only the book's
    own testimony can, and the book gives it freely: a running head is
    printed on every page of a section, so a shape recurring under a
    furniture label is the book naming that line.

    Three sightings, the threshold FURNITURE_PAGES already sets for
    repeated decoration and for the same reason: two is a coincidence,
    three is a habit. Recurrence alone is not enough, though, and the two
    conditions that guard it were measured on the shelf rather than
    supposed. parse_folio must read a page number out of the line, which
    refuses the largest class of lookalikes — a bare chapter number
    standing over its title, 20-odd of each digit across the books. And
    the block beside it must not itself be a heading, because a chapter's
    OPENING page prints no running head: a head-shaped line there is the
    chapter number above the chapter's name.

    Across every cached book the rule takes 57 lines in 8 books, while the
    two conditions spare 129 repeated section titles and 3 chapter
    openings.

    A book that numbers its pages at the foot gives its running heads no
    folio to read, and there the folio condition refuses every leak: China's
    Engine of Environmental Collapse set its recto head REFERENCES as a
    heading on all 36 notes pages it heads, and never once as furniture, so
    the census never saw it either. What such a head has instead is rhythm —
    the identical line, case and typesetting aside, opening page after page
    with never more than HEAD_CADENCE pages between sightings — where a
    repeated section title returns once a chapter and a chapter number
    changes every time. The run's first page is the section's own heading,
    since an opening page prints no running head, so it stays, and the head
    must be seen FURNITURE_PAGES times besides it. The rhythm is read at the
    head of the page only, because the foot is where a page repeats its own
    apparatus, and not on a page whose furniture already carries words: that
    page has its running head. Across the shelf this takes 217 more lines in
    13 books — recto heads through notes, conclusions and indexes, leaked
    book titles, a manual's designation — while those conditions spare a
    chart's source line under four charts running, a field manual's NOTE
    callouts, and a title printed on a cover, a half-title and a title page.

    Given the pages, the head is cut from their blocks (Page.cut) and the
    string rendered from them, so the block that held it goes with its
    label rather than lending that label to the paragraph after it.
    """
    census: dict[str, int] = {}
    for texts in page_furniture:
        for t in texts:
            s = head_shape(t)
            if s:
                census[s] = census.get(s, 0) + 1
    removed: list[tuple[int, str]] = []
    if not census:
        return removed
    runs = _head_runs(bodies, page_furniture)
    # A page whose furniture already carries words, not just a number, has
    # its running head; a second line the rhythm points at is its own.
    has_head = [any(_worded_head(t) for t in texts)
                for texts in page_furniture]
    has_head += [False] * (len(bodies) - len(has_head))
    for i, body in enumerate(bodies):
        for pattern, at_top, fused in _HEAD_SITES:
            site = _head_site(body, pattern, at_top, fused)
            if site is None:
                continue
            cut, text = site
            if not text or len(text) > RUNNING_HEAD_MAX_CHARS:
                continue
            if parse_folio(text) is not None:
                if census.get(head_shape(text), 0) < FURNITURE_PAGES:
                    continue
            else:
                n, first = runs.get((_head_key(text), i), (0, i))
                if not at_top:
                    continue
                if first == i or n - 1 < FURNITURE_PAGES or has_head[i]:
                    # Seen in rhythm but not often enough, or as the run's
                    # first sighting: still a running head where it stands
                    # inside a sentence that runs over the page, which no
                    # opening page does (Sraffa's chapters are short enough
                    # to show their verso head three times, two of them in
                    # mid-sentence).
                    if n < FURNITURE_PAGES or fused or i == 0 or \
                            not _interrupts_a_sentence(bodies[i - 1],
                                                       body[cut[1]:]):
                        continue
            rest = body[cut[1]:] if at_top else body[:cut[0]]
            if not fused and \
                    (_OPENS_HEADING if at_top else _ENDS_HEADING).search(rest):
                continue
            left = body[:cut[0]] + body[cut[1]:]
            # A head printed on an otherwise empty leaf interrupts nothing,
            # and taking it leaves a body that reads as blank — which the
            # blank pass then deletes, costing a page and its page-list
            # anchor to tidy one line. Losing a page is the larger harm.
            if "<img" not in left and not _strip_tags(left).strip():
                continue
            if pages is not None and i < len(pages) \
                    and pages[i].render() == body:
                # the head's block goes from the page, with its label, and
                # a head fused into a paragraph leaves that block the rest
                pages[i].cut(*cut)
            body = left
            bodies[i] = body
            removed.append((i, text))
    return removed


class Chapter(NamedTuple):
    """A chapter as the running heads describe it: a title and the span of
    pages that carried it."""
    title: str
    start: int           # first page index whose head named this chapter
    end: int             # last such page, inclusive


def head_title(line: str) -> str:
    """A running head with its folio, and whatever separated them, taken off.

    parse_folio reads the number out of the same line; this reads the rest.
    Books set the pair a dozen ways — "60 · Citadel of Sin", "THE NEW CAREER
    51", "Chapter 7 — 61" — so both ends are tried and the separator goes
    with the number.
    """
    t = html.unescape(" ".join((line or "").split()))
    t = re.sub(r"^\d{1,3}\s*[•·|:.\-–—]*\s*", "", t)
    t = re.sub(r"\s*[•·|:.\-–—]*\s*\d{1,3}$", "", t)
    # Front matter paginates in roman, and a lone "vi" left standing becomes
    # a two-page chapter called vi. Only a bare numeral is taken — a chapter
    # actually titled "I" keeps company with words and survives.
    if re.fullmatch(r"(?i)[ivxlcdm]{1,7}", t):
        return ""
    return " ".join(t.strip(" •·|:.-–—").split())


def _same_head(a: str, b: str) -> bool:
    """Whether two lines are the same running head read twice.

    A head is re-recognised on every page it appears, and the readings
    differ: one page's TIHKAL is another's THIKAL, one escapes an
    apostrophe where the next prints it. Exact matching splits a single
    chapter into several — measured, one stray misread turned an 18-page
    chapter into a 12 and a 6 — so heads are compared for closeness.
    """
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    # A single substituted letter is the commonest misreading of a head, and
    # on a short title it does not survive a ratio test: "Invasiom" against
    # "Invasion" scores 0.875 and would open a second chapter halfway
    # through the first. Same length, one letter apart, is that misreading.
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.90


# A stream of heads that changes on nearly every page is not chapters — it is
# marginal notes, or an edited volume naming a different contributor each
# leaf. Measured across the shelf: real chapter partitions run 8-12 pages to
# the run, degenerate ones run 1.0-1.2, and nothing legitimate falls between.
CHAPTER_RUN_MIN_MEAN = 4.0


# What a running head can say that is not a chapter's name. A manual prints
# its issue date and the bare word "Chapter" where a book prints a title,
# and those passed every structural guard — the runs were long and evenly
# spread — while naming nothing. They matched no heading on any page, and
# by existing at all they kept the printed contents from being consulted.
_NOT_A_CHAPTER_NAME = re.compile(
    r"^(chapter|part|section|book|volume|appendix|figure|table|page|"
    r"continued|cont|contents|table of contents|index of contents)$"
    r"|^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,4}$"
    r"|^[\d\s./–—-]+$"
    # a scan's margins can carry the archive's own furniture: one book's
    # every page bottom reads http://www.archive.org/details/… and the run
    # is as sustained as any chapter's
    r"|.*(https?://|www\.)", re.I)


def chapters_from_furniture(page_furniture: list[list[str]]
                            ) -> "tuple[list[Chapter], dict]":
    """The book's chapters, as its own running heads report them.

    A running head is the one piece of structure a book repeats on every
    page, and it names the chapter the reader is inside. Two streams share
    the position — the verso usually carries the book's title and the recto
    the chapter's — so the constant one is identified and set aside, and
    what varies is the chapter.

    This is evidence the printed contents cannot give: a span asserted
    twenty times over rather than a single line whose page must be inferred
    from a folio. Where a contents page parses it still outranks this, being
    the book's own declaration of hierarchy; where none does, or where it
    parses badly, this is the better skeleton.

    Returns (chapters, stats). An empty list means the heads did not
    describe chapters, which is a real answer for a book whose head is only
    its own title, and for a field manual whose head is a document number.
    """
    n = len(page_furniture)
    stats = {"pages": n, "with_head": 0, "runs": 0, "mean_run": 0.0,
             "refused": ""}
    if n < 8:
        stats["refused"] = "too few pages"
        return [], stats
    titles = [[head_title(t) for t in (page or [])] for page in page_furniture]
    flat = [t for page in titles for t in page if t]
    if not flat:
        stats["refused"] = "no running heads carry text"
        return [], stats
    # Which stream is the book's own title? Not the most frequent one —
    # frequency alone mistakes a long chapter for the title and eats it.
    # The distinguishing property is SPREAD: a title recurs from the first
    # leaf to the last, while a chapter head owns a band and no more. Where
    # no head spans the book, there is no title stream to set aside, and
    # every head is a chapter — some books print the chapter on both leaves.
    from collections import Counter
    seed = Counter(flat).most_common(1)[0][0]
    seen_at = [i for i, page in enumerate(titles)
               if any(_same_head(t, seed) for t in page if t)]
    spans_book = (seen_at and
                  (seen_at[-1] - seen_at[0] + 1) >= 0.6 * n)
    varying: list[str | None] = []
    for page in titles:
        rest = [t for t in page
                if t and not (spans_book and _same_head(t, seed))]
        varying.append(rest[0] if rest else None)
    stats["with_head"] = sum(1 for v in varying if v)
    if stats["with_head"] < 4:
        stats["refused"] = "no second head stream (the head is the title)"
        return [], stats
    # Walk the book. A page with no chapter head does not end a chapter — a
    # verso carrying only the book's title sits INSIDE one, and so does the
    # chapter's own opening page, which by convention prints no head at all.
    runs: list[list] = []
    for i, v in enumerate(varying):
        if v is None:
            continue
        if runs and _same_head(runs[-1][0], v):
            runs[-1][2] = i
        else:
            runs.append([v, i, i])
    # A single page bearing a head unlike its neighbours, between two that
    # agree, is a misread — not a one-page chapter between two halves of the
    # same one.
    merged: list[list] = []
    for k, r in enumerate(runs):
        if (r[1] == r[2] and merged and k + 1 < len(runs)
                and _same_head(merged[-1][0], runs[k + 1][0])):
            continue
        if merged and _same_head(merged[-1][0], r[0]):
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    # A run named by something that is not a name is not a chapter.
    merged = [r for r in merged if not _NOT_A_CHAPTER_NAME.match(r[0].strip())]
    stats["runs"] = len(merged)
    if len(merged) < 3:
        stats["refused"] = "fewer than three chapters"
        return [], stats
    stats["mean_run"] = round(stats["with_head"] / len(merged), 2)
    if stats["mean_run"] < CHAPTER_RUN_MIN_MEAN:
        stats["refused"] = (f"heads change every {stats['mean_run']} pages — "
                            f"marginalia, not chapters")
        return [], stats
    # If one head owns most of the book it is not a chapter, it is the
    # constant stream wearing the varying one's clothes — a manual whose
    # every page names the manual, where the head set aside as "the title"
    # was something rarer. Measured: a legitimate widest chapter takes a
    # quarter of its book, this failure took 96% of one.
    widest = max(b - a + 1 for _, a, b in merged)
    if widest > 0.6 * n:
        stats["refused"] = (f"one head spans {widest} of {n} pages — the "
                            f"book's own title, not a chapter")
        return [], stats
    # A partition that is mostly one-page runs is a list of section names
    # repeating, not a book divided into chapters.
    if sum(1 for _, a, b in merged if b - a + 1 >= 3) < 3:
        stats["refused"] = "fewer than three chapters span three pages"
        return [], stats
    # A chapter's opening page prints no running head — that is the
    # convention, the drop cap stands where the head would be — so the first
    # page NAMING a chapter is usually its second. Step back onto the
    # opening where no other chapter has claimed it.
    claimed = {p for _, a, b in merged for p in range(a, b + 1)}
    out = []
    for t, a, b in merged:
        if a - 1 >= 0 and (a - 1) not in claimed:
            a -= 1
        out.append(Chapter(t, a, b))
    return out, stats


# A line of furniture that carries this much is a sentence or a title, not a
# folio: the same measure the reading reports such lines by.
LONE_LINE_CHARS = 60
LONE_LINE_WORDS = 8
# Two lines this alike are one running head however a page varies it ("The
# Class Struggle on the Ideological Plane (ii)" and "… (iv) 427").
LONE_ALIKE = 0.8


def _full_line(text: str) -> bool:
    return len(text) > LONE_LINE_CHARS \
        or len(re.findall(r"[A-Za-z]{2,}", text)) > LONE_LINE_WORDS


_HEADING = re.compile(r"<h[1-6][^>]*>(.*?)</h[1-6]>", re.S)


def _titled(key: str) -> str:
    """A head's words without the section's number before them or the
    page's number after: "7. reasons for ..." and "reasons for ... 23" are
    one title."""
    return re.sub(r"^(?:\d+|[ivxlc]+)[.)]?\s+|\s+\d+$", "", key)


def restore_lone_furniture(pages: "list[Page]", bodies: "list[str]"
                           ) -> "list[tuple[int, str]]":
    """
    Put back into its page a line the layout called furniture that the book
    never repeats, and say which.

    The layout pass labels page furniture, and the reading sets every such
    block aside, trusting the label. What makes a line furniture, though, is
    that it recurs: a running head or foot is printed on page after page,
    with or without the page's number. The label also falls on lines that
    sit at a page's edge only once: a plate's printed caption (the Gresham
    ledger in Studies in the History of Accounting), an author's or a
    source note at the foot of an article ("This is an updated version of an
    article printed in Green Left Weekly"), a manual's issuing line on its
    title page, a printer's imprint. Across the shelf the reading reported
    702 furniture blocks that carry a full line of words; 682 recur in their
    book and are running heads, and of the 20 that do not, all but a few are
    such lines, lost from the book.

    The few are running heads of articles short enough to show theirs once,
    on their second page ("Reasons for Popular Scepticism Towards Politics &
    Politicians 23" in Ideas for the Struggle). Such a head repeats the
    article's title, printed as a heading a page or two before, so a line
    that matches a heading anywhere in the book (its numbering and folio
    aside) stays out, as does one near enough to another page's furniture
    to be the same head varied.

    Nothing is put back on a page with nothing else on it, neither words nor
    a picture. That would make a page the book did not have: on a blank
    leaf the line is as often the layout's invention (a journal citation
    "read" off the blank page 433 of The History of Lloyd's) as a printer's
    imprint standing alone, and a new page moves every later page's index,
    which the reviewer's decisions are keyed by.

    What is put back goes where it stood: a foot at the end of the page, a
    head at the start. It leaves the page's furniture, so the folio audit
    and the chapter reading no longer see it.
    """
    restored: "list[tuple[int, str]]" = []
    if not all(p.furniture_known for p in pages):
        # a page cached before furniture was kept would hide the repetition
        # that is the whole test
        return restored
    texts = [[" ".join(_strip_tags(b.html).split()) for b in p.furniture
              if b.html] for p in pages]
    headings = [_titled(_head_key(_strip_tags(h)))
                for body in bodies for h in _HEADING.findall(body)]
    for i, page in enumerate(pages):
        if i >= len(bodies):
            break
        if "<img" not in bodies[i] and not re.search(
                r"\w", _strip_tags(bodies[i])):
            continue
        keep = []
        for b in page.furniture:
            t = " ".join(_strip_tags(b.html or "").split())
            if not t or not _full_line(t) or parse_folio(t) is not None:
                keep.append(b)
                continue
            key = _head_key(t)
            elsewhere = any(
                difflib.SequenceMatcher(None, key, _head_key(u)).ratio()
                >= LONE_ALIKE
                for j, us in enumerate(texts) if j != i for u in us)
            titled = _titled(key)
            if elsewhere or any(
                    difflib.SequenceMatcher(None, titled, h).ratio()
                    >= LONE_ALIKE for h in headings):
                keep.append(b)
                continue
            at_top = b.label == "PageHeader" or (
                b.label != "PageFooter" and b.box is not None
                and (b.box[1] + b.box[3]) / 2 < 0.5)
            bodies[i] = b.html + bodies[i] if at_top else bodies[i] + b.html
            restored.append((i, t))
        page.furniture = keep
    return restored


TITLE_HEAD_BAND = 0.09      # a block ending this near the top of its page
TITLE_HEAD_SIGHTINGS = 2

# A running head is set small; a title page sets the title large. Measured
# as the width its block takes per character, of the page's width: the 21
# heads taken on the shelf run 0.010 to 0.016, Reconstruction's title page
# "RECONSTRUCTION" 0.036.
TITLE_HEAD_MAX_CHAR = 0.025    # pages besides its own that print it as furniture


def _letters(text: str) -> str:
    return " ".join(re.sub(r"[^a-z]+", " ", text.translate(_SETTING).casefold()).split())


def strip_title_heads(pages: "list[Page]", bodies: "list[str]", title: str
                      ) -> "list[tuple[int, str]]":
    """
    Take out of the body a running head that is the book's own title, where
    the layout called it a heading: (page, text) for each taken.

    The verso head of many books is their title, and the layout labels it a
    heading as often as page furniture: Paul Robeson on fifteen pages of its
    notes and index, The European Radical Left on seven, Where There Is No
    Doctor's "2010" edition line. Where the reading order put it inside the
    page, as two-column notes and indexes do, the rules that look for a head
    at the start of a page's text could not see it. Where it stands is
    known: in the top band of its page. And a book's title never opens one
    of its chapters, where a chapter's or a section's title (Notes,
    Bibliography, Chapter 7) may stand at the top of the page that opens it,
    so only the title is taken, only where the book prints it as furniture on
    other pages too, and never where it is all the page holds.
    """
    taken: "list[tuple[int, str]]" = []
    names = {_letters(title), _letters(title.split(":")[0])} - {""}
    if not names:
        return taken
    seen: "dict[str, set]" = {}
    for i, p in enumerate(pages):
        for b in p.furniture:
            seen.setdefault(_letters(_strip_tags(b.html or "")), set()).add(i)
    for i, page in enumerate(pages):
        if i >= len(bodies):
            break
        for b in list(page.blocks):
            if b.box is None or b.box[3] > TITLE_HEAD_BAND \
                    or b.label not in ("SectionHeader", "Text"):
                continue
            text = " ".join(_strip_tags(b.html).split())
            key = _letters(text)
            if key not in names or len(seen.get(key, set()) - {i}) < TITLE_HEAD_SIGHTINGS:
                continue
            if not any(o is not b and (o.kind == "figure" or _letters(_strip_tags(o.html)))
                       for o in page.blocks):
                continue
            if (b.box[2] - b.box[0]) / max(len(text), 1) > TITLE_HEAD_MAX_CHAR:
                continue                # the title, set as a title page sets it
            page.blocks.remove(b)
            bodies[i] = page.render()
            taken.append((i, text))
    return taken


def furniture_texts(pages: "list[Page]") -> "list[list[str]]":
    """The text of each page's running heads and feet, as the folio audit and
    the chapter reading take it."""
    return [[_strip_tags(b.html) for b in p.furniture if b.html] for p in pages]
