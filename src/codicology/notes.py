"""The book's notes, linked both ways where the binding is certain: a notes
section known by its name and by the numbered entries that follow it, its
groups by the ways a book names them, calls in the body aligned with
entries chapter by chapter, footnotes at a page's foot (by its rule, or by
the layout's own word for them), and a raised figure that is an edition's
number and not a call.
"""
from __future__ import annotations

import re

from .contents import BOLD_HEAD_BLOCK, _NUMBER_WORD, _chapter_head_number
from .markup import BODY_MARKER, _roman_to_int, _strip_tags
from .model import Page


def footnote_share(pages: "list[Page]") -> "list[float]":
    """For each page, how much of its foot the layout itself called
    footnotes: of the words from its first Footnote block on, the share in
    Footnote blocks. 0 on a page with no Footnote block. The foot linker
    takes a page the layout has named this way at its word (see
    link_footnotes)."""
    out = []
    for p in pages:
        first = next((k for k, b in enumerate(p.blocks) if b.label == "Footnote"), None)
        if first is None:
            out.append(0.0)
            continue
        words = lambda b: len(_strip_tags(b.html).split()) if b.html else 0
        after = sum(words(b) for b in p.blocks[first:])
        out.append(sum(words(b) for b in p.blocks[first:] if b.label == "Footnote") / after
                   if after else 0.0)
    return out


NOTE_ENTRY = re.compile(r"<p>\s*<sup>\s*(\d{1,3})\s*</sup>")


# FOOTNOTES and ENDNOTES included: one book heads every chapter-end notes
# section "FOOTNOTES", and the bare word blocked all of them — the linker,
# the group walk, and the rank probe all gate on this one pattern. The word
# must still be the whole heading.
#
# And the word a publisher puts before it. Every heading on the shelf that
# says "notes" was listed on 12 November 2026, 151 books: besides the bare
# word there are "REFERENCE NOTES" (Allen's Reconstruction, whose 192
# markers all stayed plain for it) and "End Notes" written as two words.
# Everything else that says the word is a heading ABOUT something ("A Note
# on the Text", "Notes on Contributors", "Notes payable to order") and is
# still refused, because the designation must be the whole heading.
#
# "Source Notes" is refused too, though it reads like one of these. Both
# books that have it are field manuals whose source notes are a list keyed
# by page, not numbered notes, and admitted, the heading made a notes
# section of FM 3-06's back matter: a stray raised 2 in the body was bound
# to the second sense of a word in the glossary, thirty-eight pages on.
# The first whole-shelf run found it; the link probe, which skips a book
# with so few markers, had not.
_NOTES_WORDS = r"(?:(?:FOOT|END|REFERENCE)[\s-]*)?NOTES?"


NOTES_HEAD = re.compile(rf"<h\d[^>]*>\s*{_NOTES_WORDS}\s*</h\d>", re.I)


# A publisher may call the same thing by the other name: one book heads its
# endnotes "References" and then groups them PREFACE, INTRODUCTION, CHAPTER
# 1…, numbered from 1 in each — a notes section in every respect but the
# word. Nothing was found, so nothing was grouped, and all 810 of its
# markers stayed plain.
#
# It is a SEPARATE pattern because the word carries different weight in the
# two questions asked of it. "Where does the back-of-book section start?"
# may read References; "how many notes sections does this book have?" may
# not, because a repeated NOTES head means chapter endnotes while a
# repeated REFERENCES head is nearly always the running head over one
# back-of-book section. Answering the second with this pattern sent that
# same book down the chapter-endnotes path, where position alone scopes
# the markers: 190 links, and chapters five, six and eight all pointed at
# the preface's only citation.
#
# The word decides nothing beyond the start, which is what makes it safe
# there: an author-date bibliography under the same heading opens no group,
# because a group needs a chapter head, and its entries carry no numbers,
# so the ungrouped fallback refuses it for want of an ascending run. A
# section has to be shaped like notes to be read as notes.
NOTES_SECTION_HEAD = re.compile(
    rf"<h\d[^>]*>\s*(?:{_NOTES_WORDS}|REFERENCES?)\s*</h\d>", re.I)


# The names a notes section gives its groups. Chapter forms, spelled out or
# numbered — and the sections a book's front and back matter use, because a
# book that annotates its introduction gives those notes a group of their
# own. Missing that group does not merely lose it: groups pair by position,
# so an unrecognised first group shifts every one after it by one, and the
# markers then link confidently to the wrong chapter's citations.
#
# A notes section may also say whose notes they are in so many words:
# Cressey's Other People's Money heads its groups "Notes to Chapter 1",
# and none of the five was a head, so its 102 markers had five groups too
# few to pair with.
GROUP_NAMES = (r"(?:NOTES\s+(?:TO|FOR)\s+(?:THE\s+)?)?"
               r"(?:CHAPTER\s+[A-Z\-]+|CHAPTER\s+\d{1,3}"
               r"|INTRODUCTION|PREFACE|PROLOGUE|FOREWORD|FORWARD"
               r"|CONCLUSION|EPILOGUE|AFTERWORD|POSTSCRIPT)")


# Inline markup between the heading and its words is invisible to a reader
# and fatal to a pattern that does not expect it: Work, Sex & Power sets
# every one of its eighteen group heads as <h2><i>Chapter One</i></h2>, and
# not one was recognised, so all 359 of its markers went unlinked. The
# designation must still be the WHOLE heading — a heading that merely opens
# with the word is prose, not a head.
GROUP_HEAD = re.compile(
    r"<(h[1-6]|p)\b[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    r"(" + GROUP_NAMES + r")"
    r"\s*(?:</[a-zA-Z]+>\s*)*</\1>", re.I)


# A group's head may carry the chapter's title after its number: The War
# Correspondent heads its eight "CHAPTER 3. JOURNALISM, OBJECTIVITY AND
# WAR". GROUP_HEAD wants the designation to be the whole heading, for the
# reason given above, so not one of the eight opened a group and all 188
# markers stayed plain. Here the heading must be a HEADING (a paragraph
# that opens "Chapter 3: see…" is somebody's note), the number must be
# one the strict parsers accept, and a mark must stand between it and the
# title, which is what tells "CHAPTER 3. THE PRESS" from a heading that
# merely begins with the words.
GROUP_HEAD_TITLED = re.compile(
    r"<(h[1-6])\b[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    rf"CHAPTER\s+(\d{{1,3}}|[IVXL]{{1,7}}|{_NUMBER_WORD})(?!\w)"
    r"\s*[.:—–-]\s*\S", re.I)


# And a group may be named by where in the book its notes belong and by
# nothing else. Ste. Croix's Class Struggle in the Ancient Greek World
# numbers its notes afresh in every section of every chapter and heads each
# group with the section's designation alone: "[I.ii]", "[IV.vi]". Thirty-
# eight heads, not one of them a head. The whole heading must be the
# designation: numerals joined by full stops, in brackets or not.
_SECTION_NUMERAL = r"(?:[IVXL]{1,7}|[ivxl]{1,7}|\d{1,3})"


GROUP_HEAD_SECTION = re.compile(
    r"<(h[1-6])\b[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    rf"[\[(]?\s*{_SECTION_NUMERAL}(?:\s*\.\s*{_SECTION_NUMERAL})+\s*[\])]?"
    r"\s*(?:</[a-zA-Z]+>\s*)*</\1>")


_HEAD_THEN_FIRST = re.compile(
    r"<(h[1-6])\b[^>]*>(?:(?!</\1>).){1,300}</\1>\s*"
    r"(?:<(?:ol|ul)\b[^>]*>\s*)?<(?:p|li)\b[^>]*>\s*(?:<sup>\s*)?1"
    r"(?:\.|\s*</sup>|\s+(?=[^\W\d_]|<a\b))",
    re.S)


def _notes_follow(bodies: list[str], page: int, pos: int) -> bool:
    """Whether what stands under a NOTES head is notes: a numbered entry,
    in any dress the parsers read, before the page after next begins.

    The word alone is not the thing. A field manual sets a caution under
    the heading NOTE two dozen times, and Small is Necessary opens with a
    one-page NOTE on its terms; each was counted as a notes section. The
    count decides which linker a book gets, and the first such heading is
    where the back-of-book section is taken to begin, so that one page cost
    Small is Necessary 361 links: its real Notes, two hundred pages on,
    were read as the tail of a section that had begun in the front matter.
    """
    under = bodies[page][pos:]
    if page + 1 < len(bodies):
        under += bodies[page + 1]
    return any(pat.search(under) for pat in (
        NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI, NOTE_ENTRY_BARE))


# The head may come back as a plain paragraph that says "Notes": Decolonising
# the University sets three of its chapters' notes under one, and the
# calls of each were bound to the next chapter's notes that had a heading
# — notes 1 to 12 of another chapter, confidently and wrongly — and the
# rest left plain. A paragraph is weaker evidence than a heading, so it
# stands for one only where the entries under it count 1, 2, 3, the floor
# the headless sections are held to.
PARA_NOTES_HEAD = re.compile(
    rf"<p[^>]*>\s*(?:<(b|i|strong|em)>\s*)?{_NOTES_WORDS}\s*(?:</\1>\s*)?</p>",
    re.I)


def _counts_from_one(bodies: list[str], page: int, pos: int) -> bool:
    # the rest of the page: a first note can run to a paragraph
    # (Decolonising the University p92: 800 characters before note 2)
    tail = bodies[page][pos:]
    ns = [n for _, n in sorted(
        (m.start(), int(m.group(1)))
        for pat in (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
        for m in pat.finditer(tail))]
    return ns[:3] == [1, 2, 3]


def notes_head_on(bodies: list[str], page: int, pattern=None):
    """The first NOTES head on this page that notes follow, or None."""
    for m in (pattern or NOTES_HEAD).finditer(bodies[page]):
        if _notes_follow(bodies, page, m.end()):
            return m
    if pattern is None:
        for m in PARA_NOTES_HEAD.finditer(bodies[page]):
            if _counts_from_one(bodies, page, m.end()):
                return m
    return None


def count_notes_heads(bodies: list[str]) -> int:
    """How many pages open a notes section."""
    return sum(1 for i in range(len(bodies)) if notes_head_on(bodies, i))


def notes_section_start(bodies: list[str]) -> "int | None":
    """The page the back-of-book notes begin on: the first with a notes
    head (or the other name for one) that notes follow. A book none of
    whose heads is followed by anything this can read keeps its first, as
    before: the parser may still find what this glance did not."""
    first = None
    for i, b in enumerate(bodies):
        if not NOTES_SECTION_HEAD.search(b):
            continue
        if notes_head_on(bodies, i, NOTES_SECTION_HEAD):
            return i
        if first is None:
            first = i
    return first


def _bare_numbered(bodies: list[str], start: int) -> bool:
    """Whether the notes section opening at START numbers its entries bare,
    "1 Source…" — at least ten such entries, and not one in any other dress
    before the section ends. A single "1." or superscript entry says the
    book uses that form, and then a bare numeral is somebody's citation
    running on from the page before. Ten, because a notes section set this
    way runs to dozens of entries, while three bare numerals were the panel
    labels of a figure in FM 31-70's appendix, under a REFERENCES head."""
    rank = re.search(rf"<(h[1-6])[^>]*>\s*{_NOTES_WORDS}\s*</\1>",
                     bodies[start], re.I)
    stop = rf"<{rank.group(1)}[^>]*>" if rank else r"<h1[^>]*>"
    bare = 0
    for i in range(start, len(bodies)):
        if i > start and re.search(stop, bodies[i], re.I):
            break
        if any(p.search(bodies[i]) for p in (NOTE_ENTRY, NOTE_ENTRY_PLAIN,
                                              NOTE_ENTRY_LI, NOTE_ENTRY_BR)):
            return False
        bare += len(NOTE_ENTRY_BARE.findall(bodies[i]))
    return bare >= 10


def parse_notes_section(bodies: list[str], title_heads: bool = False
                        ) -> tuple[int, list[list[tuple[int, int, int]]]]:
    """
    The book's endnotes, grouped as the book groups them.

    Returns (first_notes_page, groups) where each group is the notes of one
    chapter, in order, as (page_index, number, match_start). A note is a
    paragraph that OPENS with a superscript number — the position is what
    tells a citation apart from a marker in running prose, which never leads
    its paragraph.
    """
    start = notes_section_start(bodies)
    if start is None:
        return -1, []
    bare = _bare_numbered(bodies, start)
    entry_pats = (NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI) + (
        (NOTE_ENTRY_BARE,) if bare else ())
    groups: list[list[tuple[int, int, int]]] = []
    group_nos: list[int | None] = []
    current: list[tuple[int, int, int]] | None = None
    empty_streak = 0
    # The section is read first and grouped after, because one question
    # asked while grouping needs what comes next (_lists_inside_notes).
    stream: list[tuple[int, int, str, "int | None"]] = []
    seen_group = open_notes = False       # a group is open and has a note
    for i in range(start, len(bodies)):
        b = bodies[i]
        sup_notes = [(m.start(), "note", int(m.group(1))) for m in NOTE_ENTRY.finditer(b)]
        if sup_notes:
            sup_notes += _br_run_notes(b, sup=True)
        # a section that sets its notes as "1. Source…" rather than with a
        # superscript; only consulted where the superscript form is absent,
        # since "1. " also opens ordinary numbered lists in prose
        plain_notes = [] if sup_notes else \
            [(m.start(), "note", int(m.group(1)))
             for pat in entry_pats
             for m in pat.finditer(b)] + _br_run_notes(b, bare=bare)
        heads = []
        for m in GROUP_HEAD.finditer(b):
            heads.append((m.start(), "group", None))
        for pat in (GROUP_HEAD_NUMBERED, GROUP_HEAD_NUMBERED_B):
            for m in pat.finditer(b):
                heads.append((m.start(), "group", int(m.group(1))))
        for pat in (GROUP_HEAD_ROMAN, GROUP_HEAD_ROMAN_P):
            for m in pat.finditer(b):
                no = _roman_to_int(m.group(1))
                if no is not None:
                    heads.append((m.start(), "group", no))
        for m in GROUP_HEAD_TITLED.finditer(b):
            no = _chapter_head_number(m.group(2))
            if no is not None:
                heads.append((m.start(), "group", no))
        for m in GROUP_HEAD_SECTION.finditer(b):
            heads.append((m.start(), "group", None))
        # A heading with nothing to say it is a chapter's, whose first
        # entry follows at once: How America Became Capitalist heads the
        # notes to its epigraph and its introduction with their titles
        # alone, before "1. The Origins of Colonial Society", and entries
        # are kept only once a group is open, so 56 notes were read past
        # and the 56 calls in front of chapter one had nothing to pair with.
        # The entry that follows must be the first, as a group's is, and
        # such a heading counts only AHEAD of the first group a pattern
        # does recognise — a preface's, an epigraph's, an introduction's.
        # A section none of whose heads can be read must still be refused
        # rather than guessed at (see the test of that name).
        taken = {pos for pos, _, _ in heads}
        for m in _HEAD_THEN_FIRST.finditer(b):
            if m.start() not in taken \
                    and not NOTES_SECTION_HEAD.match(b, m.start()):
                heads.append((m.start(), "front", None))
        events = sorted(heads + sup_notes + plain_notes, key=lambda e: e[0])
        if i > start and not events and open_notes:
            # one event-less page may be a blank leaf inside the notes; two
            # in a row mean the notes have ended and this is the index
            empty_streak += 1
            if empty_streak >= 2:
                break
            continue
        empty_streak = 0
        for pos, kind, n in events:
            if kind == "group":
                seen_group, open_notes = True, False
            elif seen_group and kind != "front":
                open_notes = True
            stream.append((i, pos, kind, n))
    first_group = next((at for at, e in enumerate(stream) if e[2] == "group"),
                       None)
    stream = [(i, pos, "group" if title_heads or at < (first_group or 0)
               else None, n)
              if kind == "front" else (i, pos, kind, n)
              for at, (i, pos, kind, n) in enumerate(stream)]
    stream = [e for e in stream if e[2] is not None]
    skip = _lists_inside_notes(stream)
    for at, (i, pos, kind, n) in enumerate(stream):
        if at in skip:
            continue
        if kind == "group":
            current = []
            groups.append(current)
            group_nos.append(n)
        elif current is not None:
            if n == 1 and current and current[-1][1] > 1:
                # The numbering came back to 1. A book restarts its
                # notes at 1 only where a new chapter's notes begin, so
                # the section's own sequence declares the boundary — a
                # structural fact no typographic pattern can supply, and
                # the only evidence available when a head is dressed in
                # a way the patterns miss. Gendzier's Development
                # Against Democracy heads all eight of its groups and
                # four are invisible here: three wrap the number in
                # <i>, and chapter four's is a centred paragraph with a
                # <br/> inside it. Eight groups parsed as four, too far
                # from the body's eight to pair, and all 705 markers
                # went unlinked.
                #
                # A RETURN TO 1 specifically, not any number that fails
                # to climb: a repeated or misread number is a damaged
                # entry, and splitting on one would invent a group and
                # shift every pairing after it. The numbering must also
                # have climbed above 1 first, so a group whose own
                # first entries read 1, 1 stays whole.
                current = []
                groups.append(current)
                group_nos.append(None)
            current.append((i, n, pos))
    keep = [(no, g) for no, g in zip(group_nos, groups) if g]
    if not keep:
        solo = _ungrouped_notes(bodies, start)
        if solo:
            parse_notes_section.last_group_numbers = [None]
            return start, [solo]
    parse_notes_section.last_group_numbers = [no for no, _ in keep]
    return start, [g for _, g in keep]


def _lists_inside_notes(stream: list) -> set:
    """Which entries of a notes section are a numbered list inside a note,
    as positions in `stream` ((page, pos, kind, number), in reading order).

    A return to 1 is how a notes section says a new chapter's notes begin
    where its head went unread (see parse_notes_section). But a note may
    itself hold a numbered list, and that returns to 1 as well: KGB: Masters
    of the Soviet Union has one "1." inside note 15 of a chapter, and the
    phantom group it opened put every later group one out against the
    body's, so that 643 markers, the whole book's, were refused.

    What tells the two apart is what comes after. A chapter's notes go on
    from 1; a list inside note 15 is followed by note 16. So a run that
    climbs from 1 by ones, begun where the numbers had passed 1 and with
    no head before it, is a list inside a note exactly when the entry
    after it is the next number of the notes it interrupted.
    """
    skip: set = set()
    prev = None                 # the last note of the group being read
    at = 0
    while at < len(stream):
        _, _, kind, n = stream[at]
        if kind == "group":
            prev = None
            at += 1
            continue
        if n == 1 and prev is not None and prev > 1:
            end = at + 1
            while (end < len(stream) and stream[end][2] == "note"
                   and stream[end][3] == stream[end - 1][3] + 1):
                end += 1
            after = stream[end] if end < len(stream) else None
            if after and after[2] == "note" and after[3] == prev + 1:
                skip.update(range(at, end))
                at = end
                continue
        prev = n
        at += 1
    return skip


def _ungrouped_notes(bodies: list[str], start: int) -> list:
    """
    The whole notes section as one group, for a book that does not group.

    Not every book divides its endnotes by chapter. A Critical History of
    Poverty Finance cites author-date in the prose and keeps numbered notes
    for archival sources only, so its twenty-one notes run 1 to 21 straight
    through under a single "Notes" heading with no chapter heads at all. Every
    entry was recognised and then discarded, because a group must be opened
    before an entry has anywhere to go.

    The absence of grouping is what makes this safe rather than risky: with
    one sequence the numbers are unique across the whole book, so the scope
    question the group machinery exists to answer does not arise — note 7 is
    note 7 wherever it is cited. That is also the condition. The numbers must
    climb without restarting; a section that starts over without a heading to
    say so is genuinely ambiguous and keeps its refusal.

    The section ends at the next part of the book. A bibliography or an index
    opens with a heading of the notes head's own rank, and numbered entries
    beyond that boundary are somebody else's.
    """
    rank = re.search(rf"<(h[1-6])[^>]*>\s*{_NOTES_WORDS}\s*</\1>",
                     bodies[start], re.I)
    stop = rf"<{rank.group(1)}[^>]*>" if rank else r"<h1[^>]*>"
    bare = _bare_numbered(bodies, start)
    entry_pats = (NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI) + (
        (NOTE_ENTRY_BARE,) if bare else ())
    entries: list[tuple[int, int, int]] = []
    for i in range(start, len(bodies)):
        if i > start and re.search(stop, bodies[i], re.I):
            break
        sup = [(m.start(), int(m.group(1))) for m in NOTE_ENTRY.finditer(bodies[i])]
        if sup:
            sup += [(pos, n) for pos, _kind, n in
                    _br_run_notes(bodies[i], sup=True)]
        plain = [] if sup else [(m.start(), int(m.group(1)))
                                for pat in entry_pats
                                for m in pat.finditer(bodies[i])] \
            + [(pos, n) for pos, _kind, n in _br_run_notes(bodies[i],
                                                            bare=bare)]
        entries += [(i, n, pos) for pos, n in sorted(sup + plain)]
    if len(entries) < 3:
        return []
    nums = [n for _, n, _ in entries]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        return []
    return entries


def _entry_anchor(entry: str, note_id: str, back_href: str, n: int):
    """
    (span, replacement) anchoring a note entry's opening with its id and
    backlink, in whichever of the three renderings the entry uses — or None
    when the entry's opening matches none of them, in which case the caller
    must leave the marker unlinked: a forward href to an id that was never
    written is a broken link.
    """
    nm = NOTE_ENTRY.match(entry)
    if nm is not None:
        return nm.end(), (f'<p id="{note_id}"><sup><a href="{back_href}">'
                          f'{n}</a></sup>')
    pm = re.match(r"<(p|li)[^>]*>\s*(\d{1,3})\.", entry)
    if pm is not None:
        return pm.end(), (f'<{pm.group(1)} id="{note_id}">'
                          f'<a href="{back_href}">{n}.</a>')
    # A bare-numbered entry (NOTE_ENTRY_BARE) keeps its own setting: no full
    # stop is written that the book never printed.
    bare = re.match(r"<(p|li)[^>]*>\s*(\d{1,3})(?=\s)", entry)
    if bare is not None and int(bare.group(2)) == n:
        return bare.end(), (f'<{bare.group(1)} id="{note_id}">'
                            f'<a href="{back_href}">{n}</a>')
    # An entry inside a <br/>-divided run has no block of its own to carry
    # the id, so the number itself becomes the anchor. The break and the
    # space after it are reproduced exactly: they are the run's typesetting,
    # and swallowing them would run two notes together on one line.
    bm = re.match(r"(<br\s*/?>)(\s*)(\d{1,3})\.", entry)
    if bm is not None:
        return bm.end(), (f'{bm.group(1)}{bm.group(2)}'
                          f'<a id="{note_id}" href="{back_href}">{n}.</a>')
    bb = re.match(r"(<br\s*/?>)(\s*)(\d{1,3})(?=\s)", entry)
    if bb is not None and int(bb.group(3)) == n:
        return bb.end(), (f'{bb.group(1)}{bb.group(2)}'
                          f'<a id="{note_id}" href="{back_href}">{n}</a>')
    # An item its list numbers: the list draws the figure, so there is none
    # to make a link of; the item takes the id, and the call reaches it.
    lm = re.match(r"<li\b([^>]*)>", entry)
    if lm is not None and "id=" not in lm.group(1) \
            and not re.match(r"<li\b[^>]*>\s*(?:<[^>]+>\s*)*\d", entry):
        return lm.end(), f'<li{lm.group(1)} id="{note_id}">'
    sm = re.match(r"(<br\s*/?>)(\s*)<sup>\s*(\d{1,3})\s*</sup>", entry)
    if sm is not None:
        # the superscript is the book's own typesetting and stays; only the
        # number inside it becomes the anchor, exactly as a first entry's does
        return sm.end(), (f'{sm.group(1)}{sm.group(2)}<sup>'
                          f'<a id="{note_id}" href="{back_href}">{n}</a></sup>')
    return None


# An entry the reading lost (a misread number, a dropped line) used to end
# its group: entries were taken while each was the next number, so after a
# missing 7 the 8, 9 and 10 were never looked for, and every call after 6
# stayed plain. 2083 and 2106 lost some ninety links that way. The walk now
# steps over up to ENTRY_GAP numbers, where what it steps to is borne out:
# the entry after the gap is followed by its own next number (on its page or
# the page after), and none of the numbers stepped over turns up later on the
# page out of order. A lone higher number, a year or a page reference in a
# note's own text, is followed by nothing and is not taken. The calls whose
# entries were lost stay as printed.
ENTRY_GAP = 2


def _steps_over_a_lost_entry(n: int, expect: int, later: "list[int]",
                             next_page) -> bool:
    """Whether entry number n, met where `expect` was wanted, is the group
    going on past lost entries. `later` holds the entry numbers after it on
    its page; `next_page` gives those of the page after, asked only if
    needed."""
    if not (expect < n <= expect + ENTRY_GAP):
        return False
    if any(expect <= m < n for m in later):
        return False                    # a skipped number is there after all
    return (n + 1) in later or (n + 1) in next_page()


# An entry printed without the full stop its neighbours have: "<li>12 The
# newspaper…" between "11." and "13.". The book's own setting is the dotted
# one, so a bare numeral is not read as an entry in general (it is a page
# number running on, or "75 per cent" opening a paragraph) — only where it is
# the very next number after an entry just taken, and the entry straight
# after it is its own next. 2083 lost calls 12, 18 and 34 to it.
def _bare_in_sequence(n: int, expect: int, taken: dict, after) -> bool:
    """Whether a bare-numbered entry n sits in sequence: n is wanted, n - 1
    was taken, and `after()` (the number of the next entry, dotted or
    superscript, on its page or the page after) is n + 1."""
    return n == expect and (n - 1) in taken and after() == n + 1


# A notes section may open with entries the list numbers itself: an <ol>
# whose items carry no figure, the reader having left the numbering to the
# list. Revolutionary Movements in Africa's chapter 10 sets notes 1 to 4
# that way and "5." onwards in text, so its walk found no 1 and all 57
# calls stayed plain. Such an item is entry (start + its place).
_COUNTING_LIST = re.compile(
    r"<ol(?![^>]*list-style-type:\s*none)([^>]*)>((?:(?!</ol>).)*)</ol>", re.S)


CARRIED_WORDS = 60      # the end of a note carried over, before the list


def _counted_list_entries(body: str, carry_from: "int | None" = None
                          ) -> "list[tuple[int, int]]":
    """(position, number) of each item a numbering list numbers itself.

    `carry_from` is the number the notes being read have reached, where this
    page goes on with a section begun on an earlier one."""
    out = []
    for lm in _COUNTING_LIST.finditer(body):
        # Only a list standing directly under the notes head counts from
        # its own start. Further on it is a list the page broke: Decolonising
        # the University's chapter 8 sets note 1 under its head and notes 2
        # to 21 on the next page as a fresh list, which counted from 1 put
        # every one of them a number low. Such a list counts on from the
        # notes before it instead, where it opens the page: nothing before
        # it but the end of the note the page turned in.
        heads = [m.end() for pat in (NOTES_HEAD, PARA_NOTES_HEAD)
                 for m in pat.finditer(body, 0, lm.start())]
        carried = None
        if not heads or len(_strip_tags(body[max(heads):lm.start()]).split()) > 12 \
                or re.search(r"<li\b", body[max(heads):lm.start()]):
            before = body[:lm.start()]
            if carry_from is None or heads or re.search(r"<li\b", before) \
                    or any(p.search(before) for p in
                           (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)) \
                    or len(_strip_tags(before).split()) > CARRIED_WORDS:
                continue
            carried = carry_from
        items = list(re.finditer(r"<li\b[^>]*>", lm.group(2)))
        texts = [lm.group(2)[it.end():it.end() + 12] for it in items]
        if len(items) < 2 or any(re.match(r"\s*(?:<[^>]+>\s*)*\d", t)
                                 for t in texts):
            continue
        start = re.search(r'start="(\d+)"', lm.group(1))
        first = carried if carried is not None else \
            int(start.group(1)) if start else 1
        base = lm.start(2)
        out += [(base + it.start(), first + k) for k, it in enumerate(items)]
    return out


def link_chapter_notes(bodies: list[str], dropped: set) -> dict:
    """
    Chapter-endnotes layout: a "Notes" section closes each chapter, entries
    numbered from 1, markers scattered through the chapter body before it.

    The scoping needs no guessing of any kind — the sections partition the
    book by position. Markers between one section and the next belong to the
    next, and that is the whole rule; no reset detection, no chapter numbers,
    no contents page. Entries may run past the section's first page, accepted
    only while their numbers keep ascending by one, so a numbered list in the
    following chapter's prose ("1. First, …") cannot be swallowed as more
    notes.
    """
    stats = {"linked": 0, "unlinked": 0, "sections": 0}
    heads = [(i, m.start()) for i in range(len(bodies))
             for m in [notes_head_on(bodies, i)] if m]
    # A section whose display heading the OCR dropped still announces
    # itself: a bare chapter designation followed at once by entries
    # numbered from one. One book lost a single FOOTNOTES head that way,
    # and the cost was double — that chapter's high markers went unlinked,
    # and its low markers bound BY POSITION to the next chapter's entries,
    # confidently and wrongly. Three ascending entries are required, the
    # same floor the single-section walk uses, so a chapter OPENING page
    # ("Chapter III" over prose) can never qualify.
    head_pages = {i for i, _ in heads}
    for i, b in enumerate(bodies):
        if i in head_pages:
            continue
        for gm in GROUP_HEAD.finditer(b):
            tail = b[gm.end():gm.end() + 400]
            ns = [n for _, n in sorted(
                (m.start(), int(m.group(1)))
                for pat in (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
                for m in pat.finditer(tail))]
            if ns[:3] == [1, 2, 3]:
                heads.append((i, gm.start()))
                break
    heads.sort()
    if len(heads) < 2:
        return stats
    edits: dict[int, list[tuple[int, int, str]]] = {}
    prev_head_page = -1
    for k, (hp, hpos) in enumerate(heads):
        # entries: from just past the head, across pages while numbers ascend
        entries: dict[int, tuple[int, int]] = {}
        carried: dict[tuple[int, int], int] = {}   # counted on from earlier notes
        expect = 1
        pi, from_pos = hp, hpos
        def entries_on(page: int) -> "list[tuple[int, int]]":
            # all three entry forms gathered FIRST, then walked in page
            # order with one ascending counter — three sequential passes
            # shared the counter and a page whose entries alternate forms
            # (sup 1, plain 2, sup 3) permanently skipped the later sup
            return sorted(
                [(m.start(), int(m.group(1)))
                 for pat in (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
                 for m in pat.finditer(bodies[page])
                 if not (page == hp and m.start() < from_pos)]
                + [(pos, n) for pos, n in counted(page)
                   if not (page == hp and pos < from_pos)],
                key=lambda t: t[0])

        def counted(page: int) -> "list[tuple[int, int]]":
            got = _counted_list_entries(bodies[page])
            if page > hp:
                own = {pos for pos, _ in got}
                carried_on = [(pos, n) for pos, n in _counted_list_entries(
                    bodies[page], carry_from=expect) if pos not in own]
                carried.update(((page, pos), n) for pos, n in carried_on)
                got += carried_on
            return got

        def bare_on(page: int) -> "list[tuple[int, int]]":
            return [(m.start(), int(m.group(1)))
                    for m in NOTE_ENTRY_BARE.finditer(bodies[page])
                    if not (page == hp and m.start() < from_pos)]

        while pi < len(bodies):
            found_here = False
            matches = entries_on(pi)
            following = (lambda: [m for _, m in entries_on(pi + 1)]
                         if pi + 1 < len(bodies)
                         and not (k + 1 < len(heads) and pi + 1 > heads[k + 1][0])
                         else [])
            walk = sorted([(pos, n, j) for j, (pos, n) in enumerate(matches)]
                          + [(pos, n, None) for pos, n in bare_on(pi)])
            for pos, n, j in walk:
                if j is None:
                    def after(pos=pos):
                        nxt = [m for p, m in matches if p > pos] or following()
                        return nxt[0] if nxt else None
                    taken = _bare_in_sequence(n, expect, entries, after)
                else:
                    taken = n == expect or _steps_over_a_lost_entry(
                        n, expect, [m for _, m in matches[j + 1:]], following)
                if taken:
                    entries.setdefault(n, (pi, pos))
                    expect = n + 1
                    found_here = True
            if not found_here and pi > hp:
                break
            pi += 1
            if k + 1 < len(heads) and pi > heads[k + 1][0]:
                break                       # never read into the next section
        # markers: strictly the body span since the previous section's head
        markers = []
        for bpi in range(prev_head_page + 1, hp + 1):
            for m in BODY_MARKER.finditer(bodies[bpi]):
                if bodies[bpi][:m.start()].rstrip().endswith("<p>"):
                    continue                # an entry, not a marker
                markers.append((bpi, int(m.group(1)), m.start()))
        prev_head_page = hp
        # A list counted on from the notes before it is taken only where
        # the section then ends on the chapter's last call: a lost entry
        # anywhere would put every note after it one out, and the totals
        # would not meet.
        on = sorted(n for n, at in entries.items() if carried.get(at) == n)
        if on and markers and max(entries) != max(n for _, n, _ in markers):
            entries = {n: at for n, at in entries.items() if n < on[0]}
        if not entries or not markers:
            continue
        stats["sections"] += 1
        seen: set[int] = set()
        for (bpi, n, pos) in markers:
            target = entries.get(n)
            if target is None or bpi in dropped:
                stats["unlinked"] += 1
                continue
            tpi, tpos = target
            ref_id, note_id = f"ref-c{k}-{n}", f"note-c{k}-{n}"
            first = n not in seen
            if first:
                anchored = _entry_anchor(bodies[tpi][tpos:], note_id,
                                         f"page_{bpi:04d}.xhtml#{ref_id}", n)
                if anchored is None:
                    stats["unlinked"] += 1
                    continue
                span, back = anchored
                edits.setdefault(tpi, []).append((tpos, tpos + span, back))
                seen.add(n)
            mm = BODY_MARKER.match(bodies[bpi][pos:])
            id_attr = f' id="{ref_id}"' if first else ""
            new_html = (f'<sup><a epub:type="noteref"{id_attr} '
                        f'href="page_{tpi:04d}.xhtml#{note_id}">{n}</a></sup>')
            edits.setdefault(bpi, []).append((pos, pos + mm.end(), new_html))
            stats["linked"] += 1
    for pi, page_edits in edits.items():
        b = bodies[pi]
        for a, z, new_html in sorted(page_edits, reverse=True):
            b = b[:a] + new_html + b[z:]
        bodies[pi] = b
    return stats


def find_body_marker_groups(bodies: list[str], stop: int,
                            climbed: int = 1, headless: bool = False
                            ) -> list[list[tuple[int, int, int]]]:
    """
    Superscript markers in the prose, split into chapters by their own resets.

    Note numbering restarts at 1 every chapter, so the numbers themselves mark
    the chapter boundaries: a marker at or below its predecessor after an
    ascending run means a new chapter has begun. No table of contents is
    consulted — the markers carry their own structure.

    `climbed` is how far the run before a 1 must have got for that 1 to open
    a chapter, and no single value of it is right for every book. Two 1s in
    a row are a chapter with one note followed by the next chapter, or one
    chapter citing its first note twice, and nothing in the body tells them
    apart — the marker after both readings is 2. So the caller tries both
    and lets the notes section, which knows how many chapters it has,
    decide; see `_best_body_groups`.

    `headless` also opens a chapter where the numbers fall to 2 or 3 and
    climb from there, with no 1 before them: a chapter whose first call was
    not read as one. Burning Up's tenth chapter lost its 1, so the run went
    "…44, 45, 2, 3, 4" and was read as chapter nine citing its early notes
    again: forty-three calls opened the ninth chapter's notes, and every
    chapter after it was paired one group out, 104 calls in all. A
    re-citation falls and comes straight back ("…16, 17, 2, 18"); this
    falls and stays down for two more calls. It is one more reading for the
    notes section to choose between, not a rule.
    """
    # flatten first: the reset test needs one-marker lookahead
    stream: list[tuple[int, int, int]] = []
    for i in range(max(0, stop)):
        for m in BODY_MARKER.finditer(bodies[i]):
            if bodies[i][:m.start()].rstrip().endswith("<p>"):
                continue               # a note entry, not a marker
            stream.append((i, int(m.group(1)), m.start()))
    groups: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    prev = 0
    for k, (i, n, pos) in enumerate(stream):
        # A chapter start resets the count to exactly 1 and stays low; a
        # re-citation dips and then returns high. "…16, 17, 2, 18…" is the
        # author citing note 2 again, and calling it a chapter would bind that
        # marker to the NEXT chapter's note 2 — a wrong link found by this
        # module's own regression test before any reader could find it.
        nxt = stream[k + 1][1] if k + 1 < len(stream) else None
        # What tells a chapter boundary from an author citing note 1 again
        # is the marker AFTER it: a new chapter goes on to 2, a re-citation
        # returns to wherever the chapter had got to. Requiring the run
        # before it to have reached 3 as well was a second guard on the
        # same thing, and it cost more than it caught — a chapter with one
        # or two notes never reaches 3, so its markers were absorbed by the
        # next chapter and the body came up short of the notes. Measured
        # over every built book, dropping it links 365 markers that went
        # plain — one book from none to 296 — puts not one link in the
        # wrong chapter, and costs a single marker in a single book.
        if n == 1 and prev >= climbed and (nxt is None or nxt <= 3):
            if current:
                groups.append(current)
            current = []
        elif headless and 1 < n <= 3 and prev >= n + 5 \
                and k + 2 < len(stream) \
                and n < nxt < stream[k + 2][1] <= n + 4:
            if current:
                groups.append(current)
            current = []
        current.append((i, n, pos))
        prev = n
    if current:
        groups.append(current)
    return groups


NOTE_ENTRY_PLAIN = re.compile(r"<p>\s*(\d{1,3})\.\s+\S")


# The layout pass does not render a notes section consistently: The Invisible
# Government's ran ten pages and came back as <h2> heads with <p> entries on
# most, but one page as bare <b> heads with <ol><li> entries. Same printed
# convention, two renderings — both must parse or every group after the odd
# page is silently lost.
NOTE_ENTRY_LI = re.compile(r"<li[^>]*>\s*(\d{1,3})\.\s+\S")


# A book may hang its note numbers in a column with no full stop at all:
# "1  James Joyce, Ulysses…". How the West Came to Rule does, as a list on
# its first notes page and as paragraphs after that, and every pattern here
# wanted "1." or a superscript — its notes section parsed empty and all
# 1,667 markers went plain. A bare numeral opening a paragraph is too
# ordinary to trust everywhere (a citation continued from the page before can
# open with a page number), so this form is read only in a notes section
# that shows no other: see _bare_numbered.
NOTE_ENTRY_BARE = re.compile(
    r"<(?:p|li)[^>]*>\s*(\d{1,3})\s+(?=[^\W\d_]|[‘“'\"(\[<])")


# A run of notes may be set as ONE paragraph, its entries divided by <br/>
# rather than by blocks. Every pattern above anchors the number to a block's
# opening, so only the run's first line is seen and the group's highest note
# stops there — every marker past it has nothing to point at. Two such runs,
# both at a chapter's tail, cost one book 34 links.
NOTE_ENTRY_BR = re.compile(r"<br\s*/?>\s*(\d{1,3})\.\s+\S")


# The same run set the other way: the number in a superscript and no period
# after it, which is how a book that superscripts its entry numbers renders a
# column the layout pass failed to split. Eagle Forgotten hides 67 entries in
# four such paragraphs, and 69 of its markers went plain for want of them.
NOTE_ENTRY_BR_SUP = re.compile(r"<br\s*/?>\s*<sup>\s*(\d{1,3})\s*</sup>")


# And the bare dress: How the West Came to Rule lost notes 60 to 87 into one
# paragraph, "<p>59 Shaw,<br/>60 ĪnalcIk,<br/>61 Coles,…".
NOTE_ENTRY_BR_BARE = re.compile(
    r"<br\s*/?>\s*(\d{1,3})\s+(?=[^\W\d_]|[‘“'\"(\[<])")


def _br_run_notes(body: str, sup: bool = False, bare: bool = False) -> list:
    """The note entries a paragraph hides behind its <br/>s.

    Returned as (position, "note", number), the position being the <br/>
    that opens each — which is where the anchor step rewrites.

    `sup` picks the rendering. A book that numbers its entries "1. Source…"
    continues a run as "<br/>2. Source…"; a book that superscripts them
    writes "<sup>2</sup>Source…" with no period. `bare` is the third dress,
    "<br/>2 Source…", read only in a section _bare_numbered says is set that
    way. All are the same column of notes with the same break between them,
    and the opening of the paragraph says which one this book uses — so they
    are never mixed, and a run is only ever read in the dress its own first
    entry wears.

    The guard against reading an ordinary line break this way is the
    numbering. The paragraph must already BE a note entry, so a block of
    running prose is never examined, and the numbers must climb from the
    one the paragraph opens with. A note that merely carries a <br/> before
    a numeral — a date, an address, a line of verse, a table of figures —
    does not climb, and then none of its lines are taken. Refusing the
    whole paragraph rather than its first few lines is deliberate: a run
    read half-way would end its group early, which is the very failure
    this exists to repair.
    """
    head, tail = ((NOTE_ENTRY, NOTE_ENTRY_BR_SUP) if sup
                  else (NOTE_ENTRY_BARE, NOTE_ENTRY_BR_BARE) if bare
                  else (NOTE_ENTRY_PLAIN, NOTE_ENTRY_BR))
    out = []
    for block in re.finditer(r"<p\b[^>]*>.*?</p>", body, re.S):
        opening = head.match(block.group(0))
        if opening is None:
            continue
        run = [(block.start() + m.start(), int(m.group(1)))
               for m in tail.finditer(block.group(0))]
        if not run:
            continue
        nums = [int(opening.group(1))] + [n for _, n in run]
        if any(b <= a for a, b in zip(nums, nums[1:])):
            continue
        out += [(pos, "note", n) for pos, n in run]
    return out


# The period is optional: Beyond Money heads its note groups "1 capital and
# crises" where another book writes "1. Capital and crises", and requiring
# the dot cost it every one of its 410 links — no head parsed, so no group
# ever opened and every entry was discarded before it could be paired.
GROUP_HEAD_NUMBERED = re.compile(r"<h\d[^>]*>\s*(\d{1,2})\.?\s+\S")


GROUP_HEAD_NUMBERED_B = re.compile(r"<b>\s*(\d{1,2})\.\s+\S")


# A book may number its chapters in Roman, and then its endnote groups are
# headed "I.", "II.", "III." where another book heads them "1.". Schwartz's
# Chinese Communism does, and without this every one of its 697 markers went
# unlinked: no head was recognized, so no group ever opened.
GROUP_HEAD_ROMAN = re.compile(r"<h\d[^>]*>\s*([IVXL]{1,6})\.\s")


# The same book set twelve of its thirteen group heads as headings and one as
# an ordinary paragraph, so the paragraph form has to parse too — but it is
# the risky one, because a note continuing across a page break also opens a
# paragraph. Three things keep it honest: the block must stand alone and end
# right there, it must be short the way a title is and not a citation, and it
# must not be an author's initials, which is how "I. V. Stalin" opens a note
# in exactly the literature this pattern reads.
GROUP_HEAD_ROMAN_P = re.compile(
    r"<p[^>]*>\s*([IVXL]{1,6})\.\s+(?![A-Z]\.)(?:<[^>]+>|[^<]){1,70}</p>")


def normalize_note_heads(bodies: list[str]) -> int:
    """
    Promote bold blocks that the notes parser recognizes as group heads into
    the heading form their siblings use.

    The layout pass rendered one notes page's heads as bare <b> where every
    other page used <h2>. The parser learned to read both — but once a bold
    block is identified as a head, the markup itself should say so: the
    reader gets one consistent rendering, not a page that silently differs.
    Only pages inside the notes section that carry note entries are touched,
    and only standalone bold blocks; bold inside a sentence stays emphasis.
    """
    start = notes_section_start(bodies)
    if start is None:
        return 0
    changed = 0
    for i in range(start, len(bodies)):
        b = bodies[i]
        if not (NOTE_ENTRY.search(b) or NOTE_ENTRY_PLAIN.search(b)
                or NOTE_ENTRY_LI.search(b)):
            continue

        def promote(m):
            nonlocal changed
            changed += 1
            return f"{m.group(1)}{m.group(2)}<h2>{m.group(3)}. {m.group(4)}</h2>"

        bodies[i] = BOLD_HEAD_BLOCK.sub(promote, b)
    return changed


FOOTNOTE_SYMBOLS = ("\\*\\*", "\\*", "†", "‡")


# A paragraph shaped like a footnote entry: opening with a marker — numbered
# or symbol, possibly already wrapped in its link by the symbol pass — and
# then saying something.
# A paragraph shaped like a foot: opening with its marker, then saying
# something. The marker may be set as plain text ("1. According to…") or as a
# superscript ("<sup>2</sup> See:…") — one book uses both, page by page, and
# only the first was understood, which cost it 52 of its 63 footnotes. It may
# also already be wrapped in its link by the symbol pass.
FOOT_MARK = r"(?:<sup>\s*)?(?:<a[^>]*>)?\s*(?:\*{1,2}|†|‡|\d{1,3}[.)]?)\s*" \
            r"(?:</a>)?\s*(?:</sup>)?"


FOOT_SHAPE = re.compile(r"<p[^>]*>\s*" + FOOT_MARK + r"\s*\S")


# The same thing with its number captured: what a foot entry is, wherever the
# page happens to put the block.
#
# A raised number needs no space after it: its closing tag already ends it.
# Gilded Rage sets each foot as "<sup>1</sup><a href=…>", the address hard
# against the number, and 255 of its calls stayed plain for want of a space
# a later pass puts there.
FOOT_ENTRY = re.compile(r"<(p|li)\b[^>]*>\s*(?:<sup>\s*)?(\d{1,3})[.)]?\s*"
                        r"(?:</sup>\s*|\s+)(?=\S)")


def _looks_like_foot_block(below: str) -> bool:
    """
    Is what follows the rule a footnote block, judged by shape alone?

    A footnote block opens with a foot and is feet nearly throughout. A
    section divider is followed by a heading, or by running prose that no
    footnote block contains. Size says nothing — a scholarly page can carry
    more annotation than text — so the test is where the words live: nearly
    all of them inside foot-shaped paragraphs.
    """
    if re.search(r"<h[1-6][\s>]", below):
        return False
    first = re.search(r"<(p|ol|ul|table|figure|blockquote)[^>]*>", below)
    if first is not None and first.group(1) in ("ol", "ul"):
        items = re.findall(r"<li\b[^>]*>(.{0,40})", below, re.S)
        return bool(items) and all(
            re.match(r"\s*" + FOOT_MARK + r"\s*\S", t) for t in items) \
            and not re.search(r"<p\b", below[first.start():])
    if first is None or first.group(1) != "p":
        return False
    if not FOOT_SHAPE.match(below[first.start():]):
        return False
    total = len(_strip_tags(below).split())
    if not total:
        return False
    shaped = sum(len(_strip_tags(m.group(0)).split())
                 for m in re.finditer(r"<p[^>]*>.*?</p>", below, re.S)
                 if FOOT_SHAPE.match(m.group(0)))
    return shaped >= 0.8 * total


def _foot_block_without_a_rule(body: str) -> int:
    """
    Where the feet begin on a page the layout gave no rule, or 0 if unsure.

    The printed rule is what normally separates the body from its feet, and
    where the recogniser sees one there is nothing to guess. It does not
    always see one: seven pages of When Protest Becomes Crime carry a plain
    foot block under no rule at all, and the pass skipped them at its first
    line.

    What stands in for the rule is where the feet sit and how they open. A
    footnote block is the LAST thing on its page and every paragraph in it
    leads with its own number — and a paragraph never leads with a
    superscript otherwise, which is the same reasoning the endnote parser
    uses to tell a citation from a marker in running prose. So the trailing
    run of such paragraphs is the block, and it counts only if what precedes
    it is real prose: a page that is nothing but numbered paragraphs is a
    list, or a notes page, and neither is a page with feet under it.
    """
    # The feet may be a list closing the page, every item opening with its
    # number: Disturbing Times sets seventeen pages' notes that way with no
    # rule over them, and 53 calls in three books stayed plain.
    tail = re.search(r"<ol\b[^>]*>((?:(?!</ol>).)*)</ol>\s*$", body, re.S)
    if tail:
        items = re.findall(r"<li\b[^>]*>(.{0,40})", tail.group(1), re.S)
        if items and all(re.match(r"\s*(?:<sup>\s*)?\d{1,3}(?:\s*</sup>|[.)]?\s)", t)
                         for t in items) \
                and len(_strip_tags(body[:tail.start()]).split()) >= 30:
            return tail.start()
    paras = list(re.finditer(r"<p[^>]*>.*?</p>", body, re.S))
    if len(paras) < 2:
        return 0
    lead = re.compile(r"<p[^>]*>\s*<sup>\s*\d{1,3}\s*</sup>\s*\S")
    k = len(paras)
    while k > 0 and lead.match(paras[k - 1].group(0)):
        k -= 1
    if k == len(paras) or k == 0:
        return 0                      # no trailing run, or the whole page
    # anything after the last body paragraph belongs to the block
    start = paras[k - 1].end()
    body_words = len(_strip_tags(body[:start]).split())
    foot_words = len(_strip_tags(body[start:]).split())
    if body_words < 30 or not foot_words:
        return 0                      # too little page above to be a page
    return start


# A foot printed with a plain number, "20 Kurt W. Weitzmann…".
_BARE_FOOT = re.compile(r"<p[^>]*>\s*(\d{1,3})\s+(?=[^\W\d_]|[‘“'\"(\[<])")


def _bare_feet_without_a_rule(body: str) -> int:
    """Where the feet begin on a page with no rule whose feet are numbered
    plainly, or 0.

    Disturbing Times prints its feet "20 Kurt W. Weitzmann…", with no
    superscript and, on some pages, with no rule the recogniser saw: the
    trailing run of paragraphs that lead with a superscript is how a
    ruleless foot block is found, and a plain number leads none. A plain
    number opening a paragraph is too ordinary to take on its own, so the
    page has to call each of them: the run's numbers climb one at a time
    (a paragraph between them, a note going on, at most one), and each is
    raised exactly once in the page above it, which a numbered list in the
    prose never is."""
    paras = list(re.finditer(r"<p[^>]*>.*?</p>", body, re.S))
    if len(paras) < 2:
        return 0
    k, run, loose = len(paras), [], 0
    while k > 0:
        m = _BARE_FOOT.match(paras[k - 1].group(0))
        if m:
            run.append(int(m.group(1)))
            loose = 0
        elif run and loose == 0:
            loose = 1
        else:
            break
        k -= 1
    k += loose                    # a loose paragraph at the top is above the run
    run.reverse()
    if not run or k == 0 or any(b != a + 1 for a, b in zip(run, run[1:])):
        return 0
    start = paras[k].start()
    above = body[:start]
    if len(_strip_tags(above).split()) < 30:
        return 0
    raised = [int(n) for n in re.findall(r"<sup>\s*(\d{1,3})\s*</sup>", above)]
    if any(raised.count(n) != 1 for n in run):
        return 0
    return start


_ENDS_A_SENTENCE = re.compile(r"[.!?…:;][”’\"')\]]*\s*$")


def _carried_over_tail(below: str, before: str, labelled: bool = False) -> int:
    """Where a page's own notes begin, past the end of a note the page
    before could not finish; 0 if the foot block does not open that way.

    A long footnote runs over the page, and its last lines are the first
    thing under the next page's rule, with no number in front of them. A
    foot block has to open with a foot, so that page was not a foot block,
    and none of its calls or notes were linked: twenty-seven pages of Kill
    Boxes, every note on them, and the same in the other books of its
    series.

    The tail is one or two paragraphs with no number. What makes it a tail
    and not prose under a divider is the page before: its last note stops
    without ending its sentence ("…the Statue of Liberty be-"), or the
    paragraph itself starts in the middle of one. Or the layout says so:
    where it called nearly everything under the rule a footnote (`labelled`),
    the paragraphs with no number in front of the first foot are a note
    going on. Disturbing Times runs a note over the page from under a list
    with no rule, so the page before has no rule to read it from.
    """
    paras = list(re.finditer(r"<p[^>]*>.*?</p>", below, re.S))
    k = next((j for j, m in enumerate(paras) if FOOT_SHAPE.match(m.group(0))),
             None)
    if k is None or not 1 <= k <= 2:
        return 0
    if _strip_tags(below[:paras[0].start()]).strip() \
            or re.search(r"<(?:h[1-6]|table|ul|ol|figure)\b",
                         below[:paras[k].start()]):
        return 0
    first = _strip_tags(paras[0].group(0)).strip()
    if not first:
        return 0
    last = [m.group(0) for m in re.finditer(r"<p[^>]*>.*?</p>",
                                            before.split("<hr/>", 1)[-1], re.S)]
    unfinished = bool(last) and "<hr/>" in before \
        and not _ENDS_A_SENTENCE.search(_strip_tags(last[-1]).strip())
    if not (unfinished or first[0].islower() or labelled):
        return 0
    return paras[k].start()


# How much of what follows the rule the layout must have called footnotes
# for its word to stand in for the block's shape.
FOOT_LABELLED = 0.9


def _opens_with_a_foot(below: str) -> bool:
    first = re.search(r"<(p|ol|ul|table|figure|blockquote|h[1-6])[^>]*>", below)
    return first is not None and first.group(1) == "p" \
        and bool(FOOT_SHAPE.match(below[first.start():]))


# A table's own notes: the table, then straight after it paragraphs that open
# with superscripts numbered from one, answering superscripts in its cells.
# The Modern Corporation and Private Property sets some forty tables that way,
# and none of their notes linked: there is no rule above them, and the page's
# feet, where a page has any, are somewhere else.
TABLE_NOTE = re.compile(r"\s*<p[^>]*>\s*<sup>\s*(\d{1,3})\s*</sup>\s*\S.*?</p>",
                        re.S)
_TABLE = re.compile(r"<table\b[^>]*>.*?</table>", re.S)
_CELL_CALL = re.compile(r"<sup>\s*(\d{1,3})\s*</sup>")
TITLE_WORDS = 40        # a table's title line, not a paragraph of prose


# A table's notes may be marked with symbols instead (Survivability Manual,
# The Horn of Africa): the table, then paragraphs each opening with a
# different one of *, **, †, ‡, each set after a word in the table's cells
# and nowhere else on the page.
TABLE_SYMBOL_NOTE = re.compile(r"\s*<p[^>]*>\s*(\*\*|\*|†|‡)(?![*†‡])\s*\S.*?</p>",
                               re.S)


def _link_symbol_notes(body: str, notes: list, lo: int, hi: int, pos: int,
                       pi: int, t: int, edits: list, claimed: list) -> None:
    """Add to EDITS what links each symbol note in NOTES to its calls: the
    symbol set after a word between LO and HI (the table and its title
    line), and nowhere else on the page outside LO..POS."""
    text_spans = [(x.start(), x.end()) for x in re.finditer(r"[^<]+", body)]

    def in_text(a):
        return any(s <= a < e for s, e in text_spans)
    for k, m in enumerate(notes):
        sym = m.group(1)
        call = re.compile(rf"(?:(?<=[^\s*†‡<>])|(?<=[^\s*†‡<>] ))(?<!\*)"
                          rf"{re.escape(sym)}(?![*†‡])")
        inside = [c for c in call.finditer(body, lo, hi) if in_text(c.start())]
        outside = [c for c in call.finditer(body) if in_text(c.start())
                   and not (lo <= c.start() < pos)]
        if not inside or outside:
            continue
        nid, rid = f"tn-p{pi}-t{t}-s{k + 1}", f"tnref-p{pi}-t{t}-s{k + 1}"
        open_ = body.rfind("<p", 0, m.start(1))
        head = re.match(r"<p[^>]*>\s*" + re.escape(sym), body[open_:])
        edits.append((open_, open_ + head.end(),
                      f'<p id="{nid}"><a href="#{rid}">{sym}</a>'))
        for j, c in enumerate(inside):
            id_attr = f' id="{rid}"' if j == 0 else ""
            edits.append((c.start(), c.end(),
                          f'<sup><a epub:type="noteref"{id_attr} '
                          f'href="#{nid}">{sym}</a></sup>'))
            claimed.append((c.start(), c.end()))


def _table_notes(body: str, pi: int) -> "tuple[str, int]":
    """Link each table's notes on page PI to the calls in that table.

    The scope is the table, and it is settled by the page: the notes number
    from one and climb by one, or are marked each with a different symbol
    (the two kinds may be mixed: The Modern Corporation's appendixes open a
    table's notes with an asterisk on its title and go on with numbers), and
    each is called from the table itself or from the title line set directly
    above it (a note on the whole table hangs there). A note cited from
    several cells is linked from each — in a table that is ordinary, and
    nothing else in so small a scope could claim the number. A number met
    as a bare superscript anywhere else on the page is left alone: the
    page's own feet, or another table's, may be using it."""
    edits: list[tuple[int, int, str]] = []
    claimed: list[tuple[int, int]] = []      # call spans this pass took
    for t, tm in enumerate(_TABLE.finditer(body)):
        notes, syms, pos = [], [], tm.end()
        while True:
            m = TABLE_NOTE.match(body, pos)
            if m and int(m.group(1)) == len(notes) + 1:
                notes.append(m)
                pos = m.end()
                continue
            m = TABLE_SYMBOL_NOTE.match(body, pos)
            if m and m.group(1) not in [x.group(1) for x in syms]:
                syms.append(m)
                pos = m.end()
                continue
            break
        if not notes and not syms:
            continue
        # the title line: the block that ends where the table begins
        lo = tm.start()
        title = re.search(r"<p[^>]*>(?:(?!<p[\s>]).)*</p>\s*$",
                          body[max(0, lo - 600):lo], re.S)
        if title is not None and not TABLE_NOTE.match(title.group(0)) \
                and not TABLE_SYMBOL_NOTE.match(title.group(0)) \
                and len(_strip_tags(title.group(0)).split()) <= TITLE_WORDS:
            lo -= len(title.group(0))
        # a symbol on the title line, but not on a sentence of prose above
        # the table, which may be calling a note of its own
        sym_lo = lo if title is None or lo == tm.start() or not re.search(
            r"[.!?:;][”’\"')\]]*$", _strip_tags(title.group(0)).strip()) else tm.start()
        _link_symbol_notes(body, syms, sym_lo, tm.end(), pos, pi, t, edits, claimed)
        calls: dict[int, list] = {}
        for c in _CELL_CALL.finditer(body, lo, tm.end()):
            calls.setdefault(int(c.group(1)), []).append(c)
        outside = {}
        for c in _CELL_CALL.finditer(body):
            if not (lo <= c.start() < pos):
                outside[int(c.group(1))] = True
        for m in notes:
            n = int(m.group(1))
            if n not in calls or n in outside:
                continue
            nid, rid = f"tn-p{pi}-t{t}-{n}", f"tnref-p{pi}-t{t}-{n}"
            s = m.start(1)
            open_ = body.rfind("<p", 0, s)
            head = re.match(r"<p[^>]*>\s*<sup>\s*\d{1,3}\s*</sup>", body[open_:])
            edits.append((open_, open_ + head.end(),
                          f'<p id="{nid}"><sup><a href="#{rid}">{n}</a></sup>'))
            for j, c in enumerate(calls[n]):
                id_attr = f' id="{rid}"' if j == 0 else ""
                edits.append((c.start(), c.end(),
                              f'<sup><a epub:type="noteref"{id_attr} '
                              f'href="#{nid}">{n}</a></sup>'))
                claimed.append((c.start(), c.end()))
    for a, z, new in sorted(edits, reverse=True):
        body = body[:a] + new + body[z:]
    return body, len(claimed)


# A foot the reader set down somewhere other than the foot of its page: above
# the rule, between two paragraphs, at the top. A paragraph that opens with a
# raised number is a foot wherever it stands — running prose never does —
# and on its own page, with that number raised once in the prose and once at
# the head of such a paragraph, the pair is settled as surely as under a
# rule. The Modern Corporation lost 41 calls to its reading order this way.
_LEAD_FOOT = re.compile(r"<p([^>]*)>(\s*)<sup>\s*(\d{1,3})\s*</sup>(?=\s*\S)")
_BARE_CALL = re.compile(r"<sup>\s*(\d{1,3})\s*</sup>")


def _scattered_feet(body: str, pi: int, before: "int | None" = None
                    ) -> "tuple[str, int]":
    """Pair the feet a page holds out of place with their calls. Under a
    rule the shape tests decide (a rule is not always a footnote rule), so
    on a page that has one, `before` limits this to feet standing above it."""
    feet: dict[int, list] = {}
    for m in _LEAD_FOOT.finditer(body):
        if before is not None and m.start() >= before:
            continue
        if "id=" not in m.group(1):
            feet.setdefault(int(m.group(3)), []).append(m)
    if not feet:
        return body, 0
    starts = {m.end() - len(m.group(0)) + m.group(0).find("<sup>")
              for ms in feet.values() for m in ms}
    calls: dict[int, list] = {}
    for m in _BARE_CALL.finditer(body):
        if m.start() not in starts:
            calls.setdefault(int(m.group(1)), []).append(m)
    edits = []
    for n, fs in feet.items():
        cs = calls.get(n, [])
        if len(fs) != 1 or len(cs) != 1:
            continue
        f, c = fs[0], cs[0]
        fid, rid = f"fn-p{pi}-n{n}", f"fnref-p{pi}-n{n}"
        edits.append((f.start(), f.end(), f'<p{f.group(1)} id="{fid}">{f.group(2)}'
                      f'<sup><a href="#{rid}">{n}</a></sup>'))
        edits.append((c.start(), c.end(), f'<sup><a epub:type="noteref" id="{rid}" '
                      f'href="#{fid}">{n}</a></sup>'))
    for a, z, new in sorted(edits, reverse=True):
        body = body[:a] + new + body[z:]
    return body, len(edits) // 2


def link_footnotes(bodies: list[str], allow_numbered: bool = False,
                   foot_labelled: "list | None" = None) -> dict:
    """
    Same-page footnotes, bound where the page itself settles the match.

    The older convention: an asterisk after a word, answered at the foot of the
    same page below a rule — which the recogniser preserves as a literal <hr/>.
    Scope is the page, so the only question is arity: link a symbol when it
    appears exactly once in the prose above the rule and exactly once below
    it. Any other count on that page and its symbols stay plain.

    Numbered footnotes obey the same page scope and the same arity rule, and
    need guards a symbol does not. Digits are everywhere in a book, so a
    marker counts only as a bare <sup>, never as a number met in prose; a
    rule is not always a footnote rule, so what follows one must be shaped
    like feet before any number binds across it. And the pass runs at all
    only when the CALLER, which can see the whole book, says the book's
    numbered markers are not spoken for: a book with an endnotes section
    resolves its numbers there, and a same-page pass grabbing them first
    would bind marker 1 to whatever paragraph below a rule happens to open
    "1." — The Invisible Government mixes symbol footnotes with numbered
    endnotes on the same pages, and is exactly who this flag protects.
    Every guard costs recall on a page laid out unusually, which is the side
    to err on — an unlinked marker is the page as printed.

    One shape the guards refused is not unusual at all: a note of two
    paragraphs, or one that ends in a list. Its second paragraph has no
    number, so the block was no longer "feet nearly throughout", and no
    note on the page was linked. `foot_labelled` is the layout's own word,
    page by page: the share of the words from the first block it called a
    footnote to the end of the page that it called footnotes. Where that
    is nearly all of them and the block opens with a foot, the paragraphs
    with no number are the notes going on, and the block stands.
    """
    stats = {"linked": 0, "numbered": 0, "skipped": 0, "ruleless": 0,
             "table": 0}
    for pi, body in enumerate(bodies):
        body, tn = _table_notes(body, pi)
        if tn:
            bodies[pi] = body
            stats["linked"] += tn
            stats["table"] += tn
        rule = "<hr/>"
        labelled_page = bool(foot_labelled) and pi < len(foot_labelled) \
            and foot_labelled[pi] >= FOOT_LABELLED
        before = bodies[pi - 1] if pi else ""

        def feet_under(part: str) -> bool:
            cut = _carried_over_tail(part, before, labelled_page)
            return _looks_like_foot_block(part[cut:] if cut else part)
        if rule in body:
            above, below = body.split(rule, 1)
            # The end of a note carried over from the page before can come
            # first, set above a rule of its own, and then the page's prose
            # and its own feet under a second rule. The feet are under the
            # last rule, so that is the one the page divides at when what
            # follows the first is not feet.
            last = body.rfind(rule)
            if last != body.find(rule) and not feet_under(below) \
                    and feet_under(body[last + len(rule):]):
                above, below = body[:last], body[last + len(rule):]
        else:
            # No rule drawn. Only the numbered pass goes on without one, and
            # only where the page's own shape says where the feet begin —
            # symbols have no such trailing-run signature to stand on.
            cut = (_foot_block_without_a_rule(body)
                   or _bare_feet_without_a_rule(body)) if allow_numbered else 0
            if not cut:
                if allow_numbered:
                    body, k = _scattered_feet(body, pi)
                    if k:
                        bodies[pi] = body
                        stats["linked"] += k
                        stats["numbered"] += k
                        stats["scattered"] = stats.get("scattered", 0) + k
                continue
            above, below, rule = body[:cut], body[cut:], ""
            stats["ruleless"] += 1
        for k, sym in enumerate(FOOTNOTE_SYMBOLS):
            foot = list(re.finditer(rf"<p>\s*({sym})(?![*†‡])\s*", below))
            mark = list(re.finditer(rf"(?<=\S)({sym})(?![*†‡])", above))
            if len(foot) != 1 or len(mark) != 1:
                if foot or mark:
                    stats["skipped"] += 1
                continue
            fid = f"fn-p{pi}-{k}"
            rid = f"fnref-p{pi}-{k}"
            m, f = mark[0], foot[0]
            below = (below[:f.start()]
                     + f'<p id="{fid}"><a href="#{rid}">{f.group(1)}</a> '
                     + below[f.end():])
            above = (above[:m.start()]
                     + f'<sup><a epub:type="noteref" id="{rid}" href="#{fid}">'
                     + f'{m.group(1)}</a></sup>'
                     + above[m.end():])
            stats["linked"] += 1

        # numbered footnotes, once the symbols have taken their markers (a
        # linked one is no longer a bare <sup> and cannot be claimed twice)
        tail = ""
        if allow_numbered and not _looks_like_foot_block(below):
            cut = _carried_over_tail(below, before, labelled_page)
            if cut:
                tail, below = below[:cut], below[cut:]
                stats["carried"] = stats.get("carried", 0) + 1
        labelled = bool(foot_labelled) and pi < len(foot_labelled) \
            and foot_labelled[pi] >= FOOT_LABELLED \
            and _opens_with_a_foot(below)
        if allow_numbered and (_looks_like_foot_block(below) or labelled):
            mark_by_n: dict[int, list] = {}
            for m in re.finditer(r"<sup>\s*(\d{1,3})\s*</sup>", above):
                mark_by_n.setdefault(int(m.group(1)), []).append(m)
            foot_by_n: dict[int, list] = {}
            for m in FOOT_ENTRY.finditer(below):
                foot_by_n.setdefault(int(m.group(2)), []).append(m)
            both = sorted(set(mark_by_n) & set(foot_by_n))
            usable = [n for n in both
                      if len(mark_by_n[n]) == 1 and len(foot_by_n[n]) == 1]
            stats["skipped"] += len(both) - len(usable)
            # rewrite from the end of each half so earlier offsets stay valid
            for n in sorted(usable, key=lambda n: foot_by_n[n][0].start(),
                            reverse=True):
                f = foot_by_n[n][0]
                # The entry keeps the dress it was printed in: a superscript
                # stays one, and "1)" or a bare "1" gains no full stop. Every
                # entry used to come out "1.", which put a stop the book never
                # printed after each superscript foot it linked.
                mark = re.match(r"<(p|li)\b([^>]*)>\s*(<sup>)?\s*\d{1,3}([.)]?)",
                                f.group(0))
                num = f'<a href="#fnref-p{pi}-n{n}">{n}{mark.group(4)}</a>'
                # a list item keeps its own attributes: they are its list's
                keep = mark.group(2) if mark.group(1) == "li" else ""
                below = (below[:f.start()]
                         + f'<{mark.group(1)}{keep} id="fn-p{pi}-n{n}">'
                         + (f"<sup>{num}</sup> " if mark.group(3) else f"{num} ")
                         + below[f.end():])
            for n in sorted(usable, key=lambda n: mark_by_n[n][0].start(),
                            reverse=True):
                m = mark_by_n[n][0]
                above = (above[:m.start()]
                         + f'<sup><a epub:type="noteref" id="fnref-p{pi}-n{n}" '
                         + f'href="#fn-p{pi}-n{n}">{n}</a></sup>'
                         + above[m.end():])
            stats["linked"] += len(usable)
            stats["numbered"] += len(usable)

        bodies[pi] = above + rule + tail + below
        if allow_numbered:
            bodies[pi], k = _scattered_feet(
                bodies[pi], pi, before=len(above) if rule else None)
            if k:
                stats["linked"] += k
                stats["numbered"] += k
                stats["scattered"] = stats.get("scattered", 0) + k
    return stats


def _best_body_groups(bodies, notes_start, n_notes):
    """The body's chapters, read the way the notes section says there are.

    Splitting eagerly finds chapters whose notes are one or two citations,
    which a cautious rule folds into their neighbour — one book presented
    four groups against seven and was refused outright, losing all
    thirty-nine of its links. Splitting eagerly also cuts a chapter in two
    where it merely cites its first note twice, and the marker stranded on
    the near side of that cut is then linked to nothing.

    Neither reading is right in general, so both are offered and the notes
    section arbitrates: it knows how many chapters it has. The reading
    whose group count is closest wins, and where they tie — as they do when
    an eager split strands a marker into a group too small to stand — the
    one that leaves fewest markers behind does, because a marker outside
    every group is a marker that cannot link.
    """
    best = None
    # The readings without a headless chapter come first and keep any tie,
    # so a book whose groups already match its notes is read as before.
    for headless in (False, True):
        for climbed in (1, 3):
            raw = find_body_marker_groups(bodies, notes_start, climbed,
                                          headless)
            for smallest in (1, 2):
                # A chapter may cite a single note, and then its body group
                # is one marker long. Discarding every such group by default
                # stranded those markers; keeping them by default invents a
                # chapter out of any stray numeral. It is another thing the
                # notes section can settle, so it is offered as a reading
                # too.
                groups = [g for g in raw if len(g) >= smallest]
                rank = (abs(len(groups) - n_notes),
                        -sum(len(g) for g in groups))
                if best is None or rank < best[0]:
                    best = (rank, groups)
    return best[1]


def _fit_groups(body_groups, note_groups):
    """The one way these two lists of groups line up, or None.

    The counts differ when one side sees a chapter boundary the other
    cannot. A chapter whose notes are a single citation is the case that
    surfaced it: the notes side has an explicit head to open the group,
    while the body side has only the numbers, and it needs an ascending
    run before a return to 1 counts as a new chapter — otherwise an
    author re-citing note 2 would open one. So that chapter's lone marker
    is absorbed by its neighbour and the body comes up one group short.

    Trimming the tail, which is all this used to do, assumes the missing
    group is at the END. Here it was at the FRONT — a preface with one
    note — and trimming shifted all nine pairs, which the agreement test
    then caught and refused: 810 markers left plain on a book whose notes
    were fully parsed and sitting right there.

    Every contiguous fit is offered to that same test instead. The one
    the code already tried is offered FIRST and kept if it passes, so no
    book that pairs today can stop pairing; the others are consulted only
    after it fails, and one is taken only if it is the only one that
    passes. Two candidates agreeing means the numbers cannot separate
    them, and a confident link into the wrong chapter's citations is the
    one outcome worse than no link at all.
    """
    k = min(len(body_groups), len(note_groups))
    body_longer = len(body_groups) > len(note_groups)
    longer = body_groups if body_longer else note_groups

    def paired(off):
        win = longer[off:off + k]
        if body_longer:
            return win, note_groups
        # The notes side keeps a group this side lost, and the markers it
        # lost sit at the front of the first group that remains — as a
        # repeated 1, since the absorbed chapter's numbering started over
        # there too. Pairing them anyway pointed a preface's only
        # citation at the introduction's first source. They are left
        # plain instead: those notes have no marker this code can name.
        bg = list(body_groups)
        if off and bg:
            bg[0] = bg[0][_leading_repeats(bg[0]):]
        return bg, win

    if _paired_groups_agree(*paired(0), min_judged=2):
        return paired(0)
    rest = [paired(off) for off in range(1, len(longer) - k + 1)
            if _paired_groups_agree(*paired(off), min_judged=2)]
    return rest[0] if len(rest) == 1 else None


def _leading_repeats(group) -> int:
    """Where a body group's own numbering starts, past any absorbed one.

    A chapter whose notes are a single citation gives the body splitter
    nothing to split on — it needs an ascending run before a return to 1
    counts as a new chapter, or an author re-citing note 2 would open one
    — so that marker is absorbed by the next chapter and shows up as a
    repeated 1 at the front.

    Only a LEADING run of 1s counts. A 1 further along is the author
    citing note 1 again, and that marker does belong to this chapter.
    """
    k = 0
    while k + 1 < len(group) and group[k][1] == 1 and group[k + 1][1] == 1:
        k += 1
    return k


def _paired_groups_agree(body_groups, note_groups, min_judged: int = 4) -> bool:
    """
    Do these pairings look like the same chapters, or like a shifted list?

    The highest marker in a chapter's prose and the highest note in its group
    are the same citation, so where the pairing is right the two numbers meet.
    Where it is shifted they do not, and no per-marker check can tell: every
    chapter numbers from 1, so each marker finds SOME note and the run reports
    a full set of links.

    Judged loosely on each pair and strictly in aggregate — a single chapter
    may legitimately disagree, where a note went unread or a marker was taken
    as a footnote, but a shifted list disagrees nearly everywhere. And judged
    only with enough pairs to tell those apart: across two groups, one honest
    disagreement and a shift look identical, so the rule keeps quiet unless
    the counts already failed to match, which is the case where the pairing
    was a guess to begin with.
    """
    agree = judged = 0
    for bg, ng in zip(body_groups, note_groups):
        if not bg or not ng:
            continue
        judged += 1
        b_top = max(n for _, n, _ in bg)
        n_top = max(n for _, n, _ in ng)
        if abs(b_top - n_top) <= max(2, 0.25 * max(b_top, n_top)):
            agree += 1
    if judged < min_judged:
        return True                    # too little to tell; other rules apply
    return agree >= 0.75 * judged


# A raised figure standing directly on a title in italics, an abbreviation
# in capitals or a Roman numeral is how an edition or a volume is written:
# SEHRE², SIG³, IG II². In the body of a book whose notes are numbered it is
# also exactly how a note call is written, so it is only taken for the
# edition where it does not continue its chapter's calls: the calls climb,
# and a 2 after call 14 has not climbed. Ste. Croix cites editions this way
# ninety-odd times in a body of 1,275 calls; each would have opened the
# second or third note of whatever section it stood in.
#
# One that happens to continue the calls (PL² met just after call 1) is
# told another way: a title carries its edition wherever it is cited, so
# a figure once shown to be a title's edition is that title's edition
# everywhere in the book. Shown, not merely met twice: "in the EU¹" may
# open two chapters, and both are calls.
_EDITION_MARK = re.compile(
    r"(?:<i>([^<]{1,40})</i>|(?<![A-Za-z])([A-Z]{2,}))?"
    r"(?:\s*(?<![A-Za-z])([IVXL]{1,6}))?$")


def _title_under(body: str, pos: int) -> "str | None":
    """The title, abbreviation or volume the raised figure at `pos` stands
    directly on, or None where it stands on anything else."""
    m = _EDITION_MARK.search(body[max(0, pos - 60):pos])
    if not m or not any(m.groups()):
        return None
    return " ".join(g.strip() for g in m.groups() if g)


def _titles_with_editions(bodies: list[str], body_groups: list,
                          note_groups: list) -> set:
    """(title, figure) pairs the order of the calls shows to be editions:
    met somewhere on that title at or below the highest call its chapter
    had made."""
    shown: set = set()
    for bg, ng in zip(body_groups, note_groups):
        answered = {n for (_, n, _) in ng}
        top = 0
        for (pi, n, pos) in bg:
            if n not in answered:
                continue
            title = _title_under(bodies[pi], pos)
            if title is not None and n <= top:
                shown.add((title, n))
                continue
            top = max(top, n)
    return shown


def _an_editions_number(body: str, pos: int, n: int, top: int,
                        editions: set = frozenset()) -> bool:
    """Whether the raised figure at `pos` is an edition's or a volume's and
    not a call: it stands on a title, and either is no higher than the
    highest call its chapter has made so far or is that title's edition
    elsewhere in the book (`editions`, from _titles_with_editions)."""
    title = _title_under(body, pos)
    return title is not None and (n <= top or (title, n) in editions)


def link_notes(bodies: list[str], dropped: set[int],
               chapter_starts: "list[tuple[int, int]] | None" = None) -> dict:
    """
    Bind prose markers to their endnotes, both ways, where the binding is sure.

    The rule everything here obeys: a marker is linked only when its chapter
    scope is settled and exactly one note answers to its number there. An
    unlinked superscript is the page as it always was; a wrong link sends the
    reader to someone else's citation and looks authoritative doing it. When
    the body's chapter groups and the notes' chapter groups cannot be aligned
    one to one, whole chapters go unlinked and the count is reported, not
    hidden.
    """
    stats = {"linked": 0, "unlinked": 0, "groups": 0, "misaligned": False}
    notes_start, note_groups = parse_notes_section(bodies)
    titled = False
    if notes_start >= 0 and not note_groups:
        # Not one head the patterns know: Behind the Cold War on China heads
        # every group of its notes with its chapter's title alone. Each such
        # heading over an entry 1 is taken for a group here, on the strict
        # terms below — as many groups as the body has, every pair agreeing —
        # since a section of unreadable heads must otherwise be refused.
        notes_start, note_groups = parse_notes_section(bodies, title_heads=True)
        titled = bool(note_groups)
    if notes_start < 0 or not note_groups:
        return stats
    group_numbers = getattr(parse_notes_section, "last_group_numbers", [])
    if chapter_starts and group_numbers and all(n is not None for n in group_numbers):
        # groups name their chapters outright, and the contents told us where
        # each chapter begins: a marker's page places it, no reset-guessing
        markers = []
        for i in range(max(0, notes_start)):
            for m in BODY_MARKER.finditer(bodies[i]):
                markers.append((i, int(m.group(1)), m.start()))
        def chapter_of(page):
            c = None
            for no, start_pg in chapter_starts:
                if start_pg <= page:
                    c = no
            return c
        by_no = {no: [] for no in group_numbers}
        for (pi, n, pos) in markers:
            c = chapter_of(pi)
            if c in by_no:
                by_no[c].append((pi, n, pos))
        body_groups = [by_no[no] for no in group_numbers]
    else:
        body_groups = _best_body_groups(bodies, notes_start, len(note_groups))
    if titled and (len(body_groups) != len(note_groups)
                   or not all(_paired_groups_agree([bg], [ng], min_judged=1)
                              for bg, ng in zip(body_groups, note_groups))):
        stats["misaligned"] = True
        return stats
    if len(body_groups) != len(note_groups):
        if abs(len(body_groups) - len(note_groups)) > 2:
            stats["misaligned"] = True
            return stats
        stats["misaligned"] = True
        fit = _fit_groups(body_groups, note_groups)
        if fit is None:
            return stats
        body_groups, note_groups = fit

    # Pairing by position assumes any group missing from one side is missing
    # from the END. When it is missing from the FRONT — a notes section whose
    # introduction group went unrecognised — every pair is off by one and
    # every marker still finds a note, because chapters all number from 1.
    # Nothing downstream would notice: it reads as a full set of links. So
    # the pairs are checked against each other first. A chapter's last marker
    # and its last note are the same citation, so their numbers agree; paired
    # across a shift they do not, and four markers sitting against
    # twenty-five notes is the shape that gives it away.
    if not _paired_groups_agree(body_groups, note_groups,
                                min_judged=2 if stats["misaligned"] else 4):
        stats["misaligned"] = True
        return stats

    editions = _titles_with_editions(bodies, body_groups, note_groups)
    # rewrite from the last page backward so match offsets stay valid
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for g, (bg, ng) in enumerate(zip(body_groups, note_groups)):
        stats["groups"] += 1
        notes_by_n: dict[int, tuple[int, int]] = {}
        for (pi, n, pos) in ng:
            notes_by_n.setdefault(n, (pi, pos))
        seen_backref: set[int] = set()
        top = 0                 # the highest call this chapter has made
        for (pi, n, pos) in bg:
            target = notes_by_n.get(n)
            if target is None or pi in dropped:
                stats["unlinked"] += 1
                continue
            if _an_editions_number(bodies[pi], pos, n, top, editions):
                stats["unlinked"] += 1
                continue
            top = max(top, n)
            tpi, tpos = target
            m = BODY_MARKER.match(bodies[pi][pos:])
            ref_id = f"ref-g{g}-{n}"
            note_id = f"note-g{g}-{n}"
            # A book may cite the same note twice in a chapter. Every marker
            # links forward; only the first carries the id, because it is the
            # one the note's backlink answers and ids must be unique.
            first = n not in seen_backref
            if first:
                # The entry must anchor the id BEFORE the marker may point at
                # it — a marker linked to an entry that then fails to anchor
                # is a broken href.
                anchored = _entry_anchor(bodies[tpi][tpos:], note_id,
                                         f"page_{pi:04d}.xhtml#{ref_id}", n)
                if anchored is None:
                    stats["unlinked"] += 1
                    continue
                span, back = anchored
                edits.setdefault(tpi, []).append((tpos, tpos + span, back))
                seen_backref.add(n)
            id_attr = f' id="{ref_id}"' if first else ""
            new = (f'<sup><a epub:type="noteref"{id_attr} '
                   f'href="page_{tpi:04d}.xhtml#{note_id}">{n}</a></sup>')
            edits.setdefault(pi, []).append((pos, pos + m.end(), new))
            stats["linked"] += 1
    for pi, page_edits in edits.items():
        b = bodies[pi]
        for a, z, new in sorted(page_edits, reverse=True):
            b = b[:a] + new + b[z:]
        bodies[pi] = b
    return stats
