"""Four properties a correct set of note links has, whatever produced them.

The shelf gate counts links, and counting cannot see the failure that has
actually happened here: 190 links that all resolved and every one of which
pointed at the wrong chapter's note. The count was unchanged, the EPUB was
valid, and the book was ruined.

These four know nothing about how the linkers work. They are what a reader
could check by hand, which is what makes them worth having:

    digits    a marker printed 46 reaches an entry printed 46
    span      one chapter's markers occupy one stretch of the book
    order     chapters appear in the body in the order their notes appear
    reached   every note printed in the back is reached by some marker

`span` is the one that catches mis-grouping: bind a later chapter's markers
to the first chapter's notes and that group's markers suddenly cover the
whole book, overlapping every other group's. Verified by injection rather
than by argument -- see tests/test_linkcheck.py, which does exactly that to
a healthy book and fails if the check stays quiet.

Usage:  python tools/linkcheck.py BOOK.epub [BOOK.epub …]
Exits nonzero if any book violates any property.
"""
from __future__ import annotations

import html
import re
import sys
import zipfile
from collections import defaultdict

PAGE = re.compile(r"page_(\d+)\.xhtml$")
# The backref id is optional: a marker that re-cites a note already cited
# gets no id of its own, because two would collide, and 20 of one book's 905
# are that shape. Requiring it would leave them unchecked.
MARKER = re.compile(r'<a epub:type="noteref"[^>]*?'
                    r'href="page_\d+\.xhtml#(note-[^"]+)"[^>]*>(.*?)</a>', re.S)
ENTRY = re.compile(r'id="(note-[^"]+)"[^>]*>(?=(.{0,80}))', re.S)
GROUP = re.compile(r"note-([gc])(\d+)-(\d+)")


def _leading_number(fragment: str) -> str | None:
    """The number a marker or an entry shows the reader, markup removed."""
    text = html.unescape(re.sub(r"<[^>]+>", " ", fragment)).strip()
    m = re.match(r"(\d{1,3})(?!\d)", text)
    return m.group(1) if m else None


def _group_of(note_id: str):
    m = GROUP.match(note_id)
    return (m.group(1), int(m.group(2))) if m else None


def properties(pages: dict[int, str]) -> dict:
    """Every violation in one book, by property, from its page XHTML."""
    markers, entry_page, entry_number = [], {}, {}
    for i in sorted(pages):
        for m in MARKER.finditer(pages[i]):
            markers.append((i, m.group(1), _leading_number(m.group(2))))
        for m in ENTRY.finditer(pages[i]):
            entry_page.setdefault(m.group(1), i)
            entry_number.setdefault(m.group(1), _leading_number(m.group(2)))

    # digits — a marker and its entry must show the reader the same number.
    # Only where both are legible: an entry whose text opens with a citation
    # number of its own is not evidence about the entry's own number.
    digits = [(t, n, entry_number[t]) for _, t, n in markers
              if n and entry_number.get(t) and n != entry_number[t]]

    marker_pages, entry_pages = defaultdict(list), defaultdict(list)
    for i, t, _ in markers:
        g = _group_of(t)
        if g is None:
            continue
        marker_pages[g].append(i)
        if t in entry_page:
            entry_pages[g].append(entry_page[t])
    groups = sorted(marker_pages)
    mspan = {g: (min(v), max(v)) for g, v in marker_pages.items()}
    espan = {g: (min(v), max(v)) for g, v in entry_pages.items() if v}

    # span — a chapter is one stretch of the book, so its markers cannot
    # reach past where the next chapter's begin
    span = [(a, mspan[a], b, mspan[b]) for a, b in zip(groups, groups[1:])
            if a in mspan and b in mspan and mspan[a][1] > mspan[b][0]]
    # order — and the notes must be in the same order as the chapters
    order = [(a, espan[a], b, espan[b]) for a, b in zip(groups, groups[1:])
             if a in espan and b in espan and espan[a][0] > espan[b][0]]
    # reached — a note nothing points to is a note the reader cannot get to
    reached = sorted(set(entry_page) - {t for _, t, _ in markers})

    return {"links": len(markers), "digits": digits, "span": span,
            "order": order, "unreached": reached}


def from_epub(path: str) -> dict:
    z = zipfile.ZipFile(path)
    pages = {int(m.group(1)): z.read(n).decode("utf-8", "replace")
             for m, n in ((PAGE.search(n), n) for n in z.namelist()) if m}
    return properties(pages)


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[-2])
        return 2
    worst = 0
    for path in argv:
        r = from_epub(path)
        bad = {k: v for k, v in r.items()
               if k != "links" and v}
        head = f"{path.rsplit('/', 1)[-1][:52]:54s}{r['links']:>6} links"
        if not bad:
            print(f"  ok   {head}")
            continue
        worst = 1
        print(f"  FAIL {head}")
        for k, v in bad.items():
            print(f"         {k}: {len(v)}  e.g. {v[0]}")
    return worst


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
