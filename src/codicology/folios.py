"""Printed page numbers: read from a page's running head or foot, checked
against reading order, the gaps between them filled where only one way of
filling them fits, and published as the book's page list.
"""
from __future__ import annotations

import re
from PIL import Image
from typing import TYPE_CHECKING, NamedTuple
from . import progress
from .backends import FURNITURE_LABELS
from .markup import BODY_MARKER, RUNNING_HEAD_MAX_CHARS, _roman_to_int, _strip_tags

if TYPE_CHECKING:
    from .backends import OCRBackend


class Folio(NamedTuple):
    """The printed page number read off one page, and how much to trust it."""
    index: int
    number: int | None
    text: str            # the running head as recognised, for a human to check
    confident: bool


# Front matter paginates in Roman, and those pages must sort before Arabic 1
# while staying integers, because the audit, the gap filler and the resolver
# all compare and subtract folios. Roman i…xcix is carried as i - 1000, so
# the whole numbering space reads in order: Roman below zero, then plain
# Arabic 1…999, then the manual compound forms above.
ROMAN_FOLIO_BASE = 1000


_ROMAN_TENS = ["", "x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc"]


_ROMAN_ONES = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]


def folio_label(n: int) -> str:
    """A folio as the page prints it, from the integer it is carried as.

    The encodings exist so that every mechanism built on plain folios keeps
    working, and they were never meant to be read by anyone. They were:
    three manuals shipped a page-list labelled 1001, 1002, 2001 for pages
    printed 1-1, 1-2, 2-1, so "go to page 2-1" could not find them and the
    numbers a reader saw belonged to no page in the book.

    Roman comes back lowercase, which is what front matter prints; a book
    that sets its front matter in capitals is rendered against its own
    house style, and nothing but the label is affected.
    """
    if n < 0:                                  # front matter, Roman
        k = n + ROMAN_FOLIO_BASE
        return _ROMAN_TENS[k // 10] + _ROMAN_ONES[k % 10] if 1 <= k <= 99 else str(n)
    if n >= 100 * 1000:                        # appendix, "B-14"
        return f"{chr(ord('A') + n // 1000 - 100)}-{n % 1000}"
    if n >= 1000:                              # chapter-page, "4-9"
        return f"{n // 1000}-{n % 1000}"
    return str(n)


def parse_folio(text: str) -> int | None:
    """
    The page number out of a running head, or None if the line is not one.

    Books put it at either end — "60 - Citadel of Sin" on a verso, "Chapter 7 - 61"
    on a recto — so both are read, but only from a line short enough to be
    furniture and only when the rest of the line carries letters. A bare number
    is rejected: a stray numeral from the body text would otherwise sail through
    as a folio and quietly corrupt the ordering it was meant to check.
    """
    line = " ".join(text.split())
    if not line or len(line) > RUNNING_HEAD_MAX_CHARS:
        return None
    # Number at either end, set off by a separator or by nothing but space:
    # "60 - Citadel of Sin" and "THE NEW CAREER IN THE NEW ERA 51" are the same
    # convention in different dress. Three digits at most, so a year in a head
    # ("CHICAGO IN 1893") can never be read as a folio — four digits simply do
    # not match, and \d{1,3} anchored to a boundary cannot take the tail of one.
    # Manuals number pages by chapter — "4-9" is chapter four, page nine —
    # and that convention carries the same information a plain folio does.
    # Encoded as chapter*1000+page it stays an integer, so every mechanism
    # built on plain folios (the resolver, the audit, the contents mapper)
    # works unchanged; only the audit's idea of a gap needs to know that the
    # jump from 1-27 to 2-1 is a chapter boundary and not 974 missing pages.
    # A reader's own pagination — "Page 303 of 496" — is a folio too: proof of
    # position from the screen rather than the print. It outranks the generic
    # patterns because the same line often ends with a percentage whose digits
    # would otherwise be read as the number.
    pm = re.search(r"\bPage (\d{1,3}) of \d{1,4}\b", line)
    if pm:
        return int(pm.group(1))
    cm = (re.match(r"^(\d{1,2})-(\d{1,3})(?:\s|$)", line)
          or re.search(r"\s(\d{1,2})-(\d{1,3})$", line))
    # "FM 4-25.11" clears a two-letter bar where prose needs three; a lone
    # "1-2" is the whole line and vouches for itself. "60 - 61" matches
    # neither — the compound is written tight, no spaces round the dash.
    # A zero-padded page component is a DESIGNATION, not pagination: no book
    # numbers a page "3-06", but a manual calls ITSELF that in every running
    # head, and one seeded sixty-three false anchors before this test.
    if cm and (re.search(r"[A-Za-z]{2}", line) or line == cm.group(0).strip()):
        if not cm.group(2).startswith("0"):
            ch, pg = int(cm.group(1)), int(cm.group(2))
            if 1 <= ch <= 99 and 1 <= pg <= 999:
                return ch * 1000 + pg
    # Appendix pages paginate by letter — "A-2", "B-14" — the same
    # chapter-page convention with a letter for the chapter. Encoded past
    # every numeric chapter so appendix order sorts after the body.
    am = re.fullmatch(r"([A-Z])-([1-9]\d{0,2})", line.strip())
    if am:
        return (100 + ord(am.group(1)) - ord("A")) * 1000 + int(am.group(2))
    if not re.search(r"[A-Za-z]{3}", line):
        return None
    m = (re.match(r"^(\d{1,3})(?:\s*[•·|:\-]|\s)\s*\S", line)
         or re.search(r"(?:[•·|:\-]\s*|\s)(\d{1,3})$", line))
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 999 else None


def folios_from_furniture(page_furniture: list[list[str]]) -> list[Folio]:
    """Folio records straight from already-recognised running heads and feet."""
    out: list[Folio] = []
    for i, texts in enumerate(page_furniture):
        number, shown = None, ""
        # Rank the candidates instead of taking the first that parses. A page
        # offers its head and its foot together, and the head is sometimes a
        # chapter title that opens with a number — "48 Hours" — which the
        # verso pattern ("60 Citadel of Sin") reads as folio 48 while the real
        # folio sits beside it as "((9". Taking the first match seeded three
        # false anchors into one book and left chapter one unplaceable.
        # A line that is nothing but a numeral outranks a number quarried
        # out of a title.
        best_rank = 99
        for t in texts:
            t = " ".join(t.split())
            if t and not shown:
                shown = t
            n = parse_folio(t)
            rank = 1 if n is not None else 99
            # Folios arrive dressed: "(56)" in one book's parentheses, "((54"
            # where the scan double-struck the paren. The layout pass already
            # vouched this line is furniture, so stripping decoration before
            # the number test cannot promote prose — only undress a folio.
            bare = re.sub(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$", "", t)
            if re.fullmatch(r"[1-9]\d?-[1-9]\d{0,2}", bare):
                ch, pg = bare.split("-")
                n, rank = int(ch) * 1000 + int(pg), 0
            elif re.fullmatch(r"[A-Z]-[1-9]\d{0,2}", bare):
                ch, pg = bare.split("-")
                n, rank = (100 + ord(ch) - ord("A")) * 1000 + int(pg), 0
            elif re.fullmatch(r"[1-9]\d{0,2}", bare):
                n, rank = int(bare), 0
            if n is None and re.fullmatch(r"[1-9]\d{0,2}", t):
                # A bare numeral in running text is shrapnel, but these lines
                # were judged furniture by the layout pass, and a chapter
                # opening's foot is often nothing but the number. The context
                # that makes bare numbers dangerous is exactly what is absent.
                n, rank = int(t), 0
            if n is None and _roman_to_int(bare) is not None:
                # Front matter's own pagination, on the same terms: the whole
                # furniture line and nothing else. Roman numerals are letters,
                # so the guards are what keep prose out — _roman_to_int
                # re-renders what it read and refuses anything that does not
                # come back identical, which turns away every English word
                # spelled from the same letters, and it stops at xcix, past
                # any front matter a book actually has. A stray reading that
                # survives all that still has to join the strictly increasing
                # run in fill_folio_gaps, where a Roman numeral sitting among
                # Arabic pages cannot.
                n, rank = _roman_to_int(bare) - ROMAN_FOLIO_BASE, 0
            if n is not None and rank < best_rank:
                number, shown, best_rank = n, t, rank
                if rank == 0:
                    break            # a naked folio needs no second opinion
        out.append(Folio(i, number, shown, number is not None))
    # A folio ascends; a designation is constant. "FM 5-103" parses as
    # chapter 5, page 103 — no zero-padding to give it away — and seeded
    # eighty-one identical anchors. No real page number recurs like that: a
    # scan that shot a leaf twice repeats it twice, not eighty times. Any
    # value this frequent is the book naming itself, whatever its shape.
    from collections import Counter
    freq = Counter(f.number for f in out if f.number is not None)
    constant = {n for n, c in freq.items() if c > 5}
    if constant:
        out = [f if f.number not in constant else
               Folio(f.index, f.number, f.text, False) for f in out]
    return out


# A page's own number, run into its last line. Where the number is printed
# close under the text the layout draws one box round both, and the page
# ends "…his valuable bag-89": bound, "bag-89 gage". Only the page's own
# number, only as the last thing on the page and a whole figure there (not
# the 200 of "29,200"), only after a letter, a hyphen, or a comma and a
# space (after a full stop it would be a sentence's last figure), and not
# after a word that names what the number counts. That a page's text should
# end on a figure that happens to be the page's own number is rare enough
# to leave to chance: on 152 books the rule takes seven, six of them in The
# Man Who Stole Portugal, and each is the number under the text.
_TAKES_A_NUMBER = re.compile(
    r"(?:pages?|pp?|chapters?|ch|chap|tables?|figures?|figs?|sections?|sec|"
    r"vols?|volumes?|nos?|numbers?|parts?|lines?|notes?|items?|"
    r"paragraphs?|paras?)\.?$", re.I)


def strip_fused_folios(bodies: list[str], numbers: dict) -> list:
    """Take a page's own printed number off the end of its last line;
    return (page, number) for each."""
    removed = []
    for i, number in numbers.items():
        if not isinstance(number, int) or number <= 0 or i >= len(bodies):
            continue
        m = re.search(r"(?:(?<=[A-Za-z\-–])|(?<=[A-Za-z,(]\s)|(?<=[A-Za-z]\s\s))"
                      r"(\s*)%d(\s*(?:</[^>]+>\s*)*)$" % number, bodies[i])
        if not m:
            continue
        kept = bodies[i][:m.start(1)].rstrip()
        said = _strip_tags(kept).split()
        if not said or _TAKES_A_NUMBER.match(said[-1].strip("([\"“‘'")):
            continue
        bodies[i] = kept + m.group(2)
        removed.append((i, number))
    return removed


def strip_folio_superscripts(bodies: list[str],
                             folios: "list[Folio] | None") -> int:
    """A page's own printed number, mistaken for a note marker.

    Anthropology's World ended a page with `…Anthropology?”2<sup>38</sup>` on
    the leaf whose folio is 38: the layout lifted the page number into a
    superscript and left the real marker beside it as plain text. Two such
    phantoms cost that book all 167 of its note links — the chapter-reset
    test looks one marker ahead, so a stray 38 sitting just after a chapter's
    opening 1 hides the boundary, merges three chapters into one run, and the
    body's group count no longer matches the notes'; the misalignment guard
    then correctly refuses to link anything at all.

    Deliberately narrow, and the narrowness is measured: across 3,253 books
    on the shelf this fires on exactly those two pages. A superscript
    qualifies only as the last one on its page, only when nothing but markup
    follows it, and only when it reads as that page's own folio. The demoted
    marker beside it is left alone — recovering it would be a guess, and one
    unlinked note is a smaller loss than a wrong link.
    """
    if not folios:
        return 0
    n = 0
    for f in folios:
        if f is None or f.number is None or not 0 <= f.index < len(bodies):
            continue
        b = bodies[f.index]
        marks = list(BODY_MARKER.finditer(b))
        if not marks:
            continue
        m = marks[-1]
        if int(m.group(1)) != f.number:
            continue
        if _strip_tags(b[m.end():]).strip():
            continue                      # prose follows: a real marker
        # The layout merged marker and folio into one run and superscripted
        # only its tail, so the marker is sitting in the prose as digits
        # GLUED to the preceding word — "locations.1" is note 1 on the leaf
        # whose folio is 59. Restoring it is what puts the chapter boundary
        # back, and it cannot ship a wrong link even if it is wrong: the
        # group-alignment check downstream refuses to pair what it cannot
        # reconcile, exactly as it did before this ran.
        head, rest = b[:m.start()], b[m.end():]
        dm = re.search(r"(?<=[^\s\d>])(\d{1,3})\Z", head)
        if dm:
            head = head[:dm.start(1)] + f"<sup>{dm.group(1)}</sup>"
        bodies[f.index] = head + rest
        n += 1
    return n


def read_folios(page_paths: list[str], backend: "OCRBackend",
                locate_width: int = 1100, batch: int = 8) -> list[Folio]:
    """
    Read each page's printed number from its running head or foot.

    One full-page pass per page, downscaled and batched. Everything narrower was
    tried and failed: reading just the head's strip through the OCR stack comes
    back empty or hallucinated more often than not — the layout stage has no
    page around the ribbon to orient by — and running full pages at full
    resolution once doubled a build. With the whole (small) page in view the
    same models read the same heads correctly, and what the head says arrives
    in the layout blocks the pass produces anyway.

    Each page is read once. The defence against a misread is not a second
    reading — a dropped repeated digit misleads the same way at any size — but
    the sequence itself: audit_folios discards a number its neighbours disown.
    """
    pages: list[Image.Image | None] = []
    for path in page_paths:
        try:
            with Image.open(path) as im:
                page = im.convert("RGB").copy()
        except OSError:
            pages.append(None)
            continue
        if page.width > locate_width:
            f = locate_width / page.width
            page = page.resize((locate_width, max(8, int(page.height * f))),
                               Image.LANCZOS)
        pages.append(page)

    out: list[Folio] = []
    live = [(i, p) for i, p in enumerate(pages) if p is not None]
    results: dict[int, Folio] = {}
    for c0 in range(0, len(live), batch):
        chunk = live[c0:c0 + batch]
        try:
            preds = list(backend._predict([p for _, p in chunk]))
        except Exception:
            preds = [None] * len(chunk)
        for (i, _), pred in zip(chunk, preds):
            texts = [_strip_tags(getattr(b, "html", "") or "").strip()
                     for b in (getattr(pred, "blocks", None) or [])
                     if getattr(b, "label", "") in FURNITURE_LABELS]
            f = folios_from_furniture([texts])[0]
            results[i] = Folio(i, f.number, f.text, f.confident)
        done = min(c0 + batch, len(live))
        print(f"    read {done}/{len(live)} pages…")
        progress.emit(event="progress", phase="folios",
                      done=done, total=len(live))
    for i in range(len(page_paths)):
        out.append(results.get(i, Folio(i, None, "", False)))
    return out


class FolioAudit(NamedTuple):
    """What the printed numbers say about the order of the book."""
    read: int
    total: int
    inversions: list[tuple[int, int]]          # (page index, page index) out of order
    gaps: list[tuple[int, list[int]]]          # (page index after which, numbers absent)
    duplicates: list[tuple[int, list[int]]]    # (folio number, page indices)
    misreads: list[tuple[int, int]]            # (page index, number read but disbelieved)


def audit_folios(folios: list[Folio]) -> FolioAudit:
    """
    Compare the printed numbers against the order the pages are actually in.

    Only confident readings take part. Pages with no running head at all are not
    evidence of anything — a chapter's first page prints none, and neither does
    front matter — so they are stepped over rather than counted as a break. A
    gap is only reported where there are fewer pages between two folios than
    there are missing numbers, since a blank verso the pipeline dropped is a
    legitimate reason for a number to be absent.
    """
    good = [f for f in folios if f.confident and f.number is not None]

    # Numbering domains do not mix. A manual's folios are chapter-page
    # compounds ("1-13"); its chapter openings also carry a large bare digit —
    # the chapter's own number — and its appendices number their pages afresh.
    # Read together they fake an inversion at every chapter break: 1-13, then
    # "2", then 2-1. When compound folios carry the book, a bare number in the
    # furniture is a heading, not a page.
    compound = sum(1 for f in good if f.number >= 1000)
    if compound > len(good) // 2:
        good = [f for f in good if f.number >= 1000]

    # A reading that fits NEITHER neighbour, while its neighbours fit each
    # other, is a misread and not an ordering fault. Both defences upstream can
    # miss the same error: reading "444" as "44" drops a repeated digit the same
    # way at every magnification, so the two readings agree and the confidence
    # gate waves it through. Left in, one such number detonates into an
    # inversion, a duplicate AND a hundreds-long gap — which is exactly what a
    # real book produced. The sequence itself is the third, decisive check.
    misreads: list[tuple[int, int]] = []
    kept: list[Folio] = []
    for k, f in enumerate(good):
        # the previous neighbour is the last reading that SURVIVED — a
        # just-discarded misread must not vouch for the one after it, or
        # two consecutive misreads slip a false anchor into kept (B5)
        prev = kept[-1] if kept else None
        nxt = good[k + 1] if k + 1 < len(good) else None
        if prev is not None and nxt is not None:
            span = nxt.index - prev.index
            neighbours_fit = 0 < nxt.number - prev.number <= span * 2 + 2
            fits_between = prev.number < f.number < nxt.number
            # A transposition is not a misread: two pages bound out of order
            # still carry real numbers, and exchanging them repairs the
            # sequence. A misread repairs nothing. Only the irreparable is
            # discarded; the repairable is exactly what the inversion report
            # below exists to say.
            swap_left = prev.number > f.number and f.number > (
                good[k - 2].number if k >= 2 else f.number - 1)
            swap_right = nxt.number < f.number and f.number < (
                good[k + 2].number if k + 2 < len(good) else f.number + 1)
            # Equality with a neighbour is its own finding: the same page
            # captured twice reads the same number twice, and the duplicate
            # report below is where that belongs.
            repeats_neighbour = f.number in (prev.number, nxt.number)
            if (neighbours_fit and not fits_between
                    and not (swap_left or swap_right) and not repeats_neighbour):
                misreads.append((f.index, f.number))
                continue
        kept.append(f)
    good = kept
    inversions: list[tuple[int, int]] = []
    gaps: list[tuple[int, list[int]]] = []
    for a, b in zip(good, good[1:]):
        if b.number <= a.number:
            inversions.append((a.index, b.index))
        elif b.number > a.number + 1:
            if b.number // 1000 != a.number // 1000:
                continue          # a new chapter restarts the count; no gap
            missing = list(range(a.number + 1, b.number))
            between = b.index - a.index - 1
            if len(missing) > between:
                gaps.append((a.index, missing))
    seen: dict[int, list[int]] = {}
    for f in good:
        seen.setdefault(f.number, []).append(f.index)
    duplicates = [(n, idxs) for n, idxs in sorted(seen.items()) if len(idxs) > 1]
    return FolioAudit(len(good), len(folios), inversions, gaps, duplicates, misreads)


def fill_folio_gaps(folios: list[Folio]) -> tuple[dict[int, int], list[tuple]]:
    """
    The numbers the printer never inked, restored where the book's own
    arithmetic supplies them.

    Most pages that read no folio never carried one: chapter openings set
    their number as display type, part-titles and blank versos print nothing
    at all. The page still occupies a number in the publisher's pagination —
    it simply is not on the leaf. Measured on Schwartz's Chinese Communism,
    every one of the twenty-one fillable pages was of that kind, and not one
    was a page where the reading merely failed.

    So a run is filled only when the arithmetic closes: bounded by confident
    anchors on BOTH sides, and the difference between their numbers exactly
    equal to the number of pages between them. Where a plate is bound in or a
    leaf is missing from the capture, the difference exceeds the page count,
    nothing is known about which page holds which number, and the run is
    refused and reported. Nothing is ever extrapolated past the outermost
    anchor: front matter is commonly numbered in Roman, and running Arabic
    backwards off the first anchor would invent a numbering the book does not
    use.

    Confidence is judged again here, against the one thing a book guarantees:
    folios strictly increase in page order. The lone-misread demotion upstream
    cannot catch misreads that SUPPORT each other — two adjacent pages both
    read as folio 16 between 215 and 218 look mutually plausible, and shipped
    a page-list that ran 215, 16, 16, 218. So the anchors are cut to their
    longest strictly-increasing run first; a reading outside it is distrusted
    however confidently it was read. The dividend: once the false 16s are
    gone, the pages they sat on interpolate to the 216 and 217 the printer
    actually bound there.

    Returns (number by page index,
             runs refused as (lo, hi, lo_no, hi_no),
             page indices whose readings were distrusted for breaking order).
    """
    known = {f.index: f.number for f in folios
             if f.confident and f.number is not None}
    idx = sorted(known)
    # longest strictly-increasing subsequence of the numbers in page order
    best_len = [1] * len(idx)
    prev = [-1] * len(idx)
    for i in range(len(idx)):
        for j in range(i):
            if known[idx[j]] < known[idx[i]] and best_len[j] + 1 > best_len[i]:
                best_len[i], prev[i] = best_len[j] + 1, j
    keep: set[int] = set()
    if idx:
        i = max(range(len(idx)), key=lambda k: best_len[k])
        while i != -1:
            keep.add(idx[i])
            i = prev[i]
    distrusted = [i for i in idx if i not in keep]
    known = {i: known[i] for i in keep}
    filled, refused = dict(known), []
    for lo, hi in zip(sorted(known), sorted(known)[1:]):
        if hi - lo < 2:
            continue                       # adjacent anchors leave no gap
        if known[hi] // 1000 != known[lo] // 1000:
            # Front matter giving way to Arabic 1, or a manual's chapter
            # turning over: the count restarts, so the arithmetic between
            # these two anchors means nothing and there is no gap to report.
            # The same test the audit uses to tell a restart from a gap.
            continue
        if hi - lo != known[hi] - known[lo]:
            refused.append((lo, hi, known[lo], known[hi]))
            continue
        for i in range(lo + 1, hi):
            filled[i] = known[lo] + (i - lo)
    return filled, refused, distrusted
