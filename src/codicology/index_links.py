"""The back-of-book index bound to the pages it names: each page number in
an entry becomes a link to the page that prints it.
"""
from __future__ import annotations

import re
from .citations import _outside_markup
from .markup import _strip_tags


# ── the index → the pages it names ───────────────────────────────────────────

INDEX_HEAD = re.compile(r"<(h[1-3])[^>]*>\s*(?:<[^/][^>]*>\s*)*INDEX\s*"
                        r"(?:</[a-zA-Z]+>\s*)*</\1>", re.I)


# An index that says its numbers are not pages is believed. Field manuals
# print the warning outright — "Entries are by paragraph number" — and their
# 10-71 designations would collide with the folio map's own encoding of
# hyphenated page numbers if the warning were ignored.
INDEX_NOT_PAGES = re.compile(r"\bby\s+paragraph|\bto\s+paragraph\s+numbers?",
                             re.I)


# A page reference as an index prints one: a number, possibly a range with
# either dash, possibly abbreviated ("167–8" means 167 to 168). A slash
# neighbour is a name like 9/11, never a page. Nor is a dash neighbour: a
# real range is matched whole from its first number, so a number straight
# after a dash is the tail of something whose head was refused — "(1976–79)"
# refused at the parenthesis, then "79" taken as a page. All 46 such links
# on the shelf were wrong: 20 year-range tails, Covid-19 five times, a bill
# number, a malformed "42–4–5", and a field manual's second halves of
# "6-8-6-15" pointing into its front matter.
INDEX_REF = re.compile(
    r"(?<![\d/(\-–—])(\d{1,4})(\s*[-–]\s*(\d{1,4}))?(?![\d/])")


def _trailing_runs(text: str):
    """The spans of TEXT that are an entry's page references.

    An index entry is a name followed by numbers, and the name may itself
    contain digits — "1848 revolutions, 12" cites one page, not two. What
    separates the name's numbers from the references is position: the
    references are the maximal trailing run of number tokens with nothing
    but separators between them. Semicolons start a fresh sub-entry, so the
    rule applies per segment.
    """
    out = []
    for seg in re.finditer(r"[^;]+", text):
        toks = list(INDEX_REF.finditer(seg.group(0)))
        if not toks:
            continue
        base = seg.start()
        keep = []
        end = len(seg.group(0))
        for m in reversed(toks):
            between = seg.group(0)[m.end():end]
            if re.search(r"[A-Za-z]", re.sub(r"[a-z]{1,2}\b", "", between, count=1)):
                break
            keep.append(m)
            end = m.start()
        tail = seg.group(0)[max(k.end() for k in keep):] if keep else ""
        if keep and re.search(r"[A-Za-z]{3,}", tail):
            pass                          # letters after the run: not trailing
        else:
            out += [(base + m.start(), base + m.end(),
                     int(m.group(1))) for m in keep]
    return out


def link_index(bodies: list[str], folio_to_page: dict, dropped: set) -> dict:
    """
    Make the index's page numbers go where they point.

    The two ends already exist: the index prints folios, and the page-list
    anchored every folio the audit trusts. This joins them. Only numbers in
    an entry's trailing reference run are touched, only when the folio is in
    the map, and a book whose index declares itself a paragraph index is
    skipped with that reason recorded. Cross-references ("See Animal
    Enterprise Protection Act") carry no numbers and pass untouched. The
    word-preservation rule from the citation linker applies whole: an edited
    page that does not read back identical is reverted and counted.
    """
    stats = {"found": False, "linked": 0, "unknown_folio": 0, "reverted": 0,
             "skipped_paragraph_index": False, "pages": 0}
    hit = next(((k, m) for k, b in enumerate(bodies)
                for m in [INDEX_HEAD.search(b)] if m), None)
    if hit is None:
        return stats
    stats["found"] = True
    start, m = hit
    rank = int(m.group(1)[1])
    end = len(bodies)
    for k in range(start + 1, len(bodies)):
        hm = re.search(r"<h([1-3])[^>]*>", bodies[k])
        if hm and int(hm.group(1)) <= rank and not INDEX_HEAD.search(bodies[k]):
            end = k
            break
    intro = _strip_tags(bodies[start])[:400]
    if INDEX_NOT_PAGES.search(intro):
        stats["skipped_paragraph_index"] = True
        return stats

    for k in range(start, end):
        if k in dropped:
            continue
        body = bodies[k]
        before = _strip_tags(body).split()
        edits = []
        # text nodes only, never inside a tag or an anchor
        pos = 0
        for part in re.split(r"(<[^>]+>)", body):
            if part.startswith("<"):
                pos += len(part)
                continue
            for a, z, num in _trailing_runs(part):
                target = folio_to_page.get(num)
                if target is None or target in dropped:
                    stats["unknown_folio"] += 1
                    continue
                if not _outside_markup(body, pos + a, pos + z):
                    continue
                edits.append((pos + a, pos + z,
                              f'<a href="page_{target:04d}.xhtml'
                              f'#pgb-{target:04d}">{body[pos+a:pos+z]}</a>'))
            pos += len(part)
        if not edits:
            continue
        edits.sort(reverse=True)
        taken = len(body) + 1
        applied = 0
        for a, z, new in edits:
            if z > taken:
                continue
            body = body[:a] + new + body[z:]
            taken = a
            applied += 1
        if _strip_tags(body).split() != before:
            stats["reverted"] += 1
            continue
        bodies[k] = body
        stats["linked"] += applied
        stats["pages"] += 1
    return stats
