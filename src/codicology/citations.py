"""In-text citations bound to the bibliography, driven from the
bibliography: the prose is searched only for the names and years the
bibliography holds, never for anything shaped like a citation.
"""
from __future__ import annotations

import re
from .markup import _strip_tags


# ── citations → bibliography ─────────────────────────────────────────────────

BIBLIOGRAPHY_HEAD = re.compile(
    r"<(h[1-6])[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    r"((?:SELECT(?:ED)?\s+)?BIBLIOGRAPHY|REFERENCES|WORKS\s+CITED)"
    r"\s*(?:</[a-zA-Z]+>\s*)*</\1>", re.I)


# an author-date entry opens with the author and puts the year alone in
# parentheses: "Bernards, N. (2019a) 'Understanding…'". A Chicago imprint
# never does — its year shares the parens with a publisher — which is what
# keeps a Chicago bibliography from minting author-date keys.
AD_ENTRY = re.compile(
    r"^\s*(?:<[^/][^>]*>\s*)*"
    r"((?:[A-Z][A-Za-z’'\-]*\s+){0,2}[A-Z][A-Za-z’'\-]{2,})[^<(]{0,120}?"
    r"\((\d{4}[a-z]?)\)")


YEAR_ALONE = re.compile(r"\((\d{4}[a-z]?)\)")


# An author-date bibliography that sets its year BARE, straight after the
# author block, rather than in parentheses: "Abrahams, Roger D., 1964, Deep
# Down in the Jungle" and "Amani, B. 2013. 'Access Copyright…'". Anchored
# twice over — to the head of the entry, and to the FIRST such year — which
# is what stops "(1974 Proceedings of the American Ethnological Society)"
# late in an entry from outranking the 1976 that actually names it. A year
# in parentheses cannot match: its neighbour is a bracket, not a comma.
BARE_YEAR_ENTRY = re.compile(r"^.{0,160}?[.,]\s*(\d{4}[a-z]?)\s*[.,]", re.S)


# how a cited surname may be spelled ahead of its particle
_PARTICLES = {"van", "von", "de", "der", "den", "di", "du", "la", "le",
              "mac", "mc", "o", "ten", "ter"}


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


def _surname_forms(author: str) -> set:
    """The spellings under which this entry's author may be cited."""
    parts = re.sub(r"\s+", " ", author).strip().rstrip(",").split()
    if not parts:
        return set()
    out = {parts[-1]}
    if len(parts) > 1 and parts[-2].lower().rstrip(".") in _PARTICLES:
        out.add(" ".join(parts[-2:]))
    return {s for s in out if len(s) > 2}


def _outside_markup(body: str, start: int, end: int) -> bool:
    """Whether [start:end) is plain text: inside no tag and no anchor."""
    if "<" in body[start:end]:
        return False
    lt, gt = body.rfind("<", 0, start), body.rfind(">", 0, start)
    if lt > gt:
        return False                      # inside a tag
    opens = len(re.findall(r"<a[\s>]", body[:start]))
    return opens == body.count("</a>", 0, start)


def parse_bibliography(bodies: list[str]) -> "dict | None":
    """
    The bibliography as a set of addressable entries, anchored where they sit.

    Two shapes are read, and only two — a style this parser has not met is a
    style it must not guess at. Author-date entries carry their year alone in
    parentheses and key on (surname, year). Chicago entries open surname-first
    with the title in italics and key on (surname, title); an entry with no
    author of its own inherits the surname above it, which is how Chicago
    sets a second work by the same hand.
    """
    hit = next(((k, m) for k, b in enumerate(bodies)
                for m in [BIBLIOGRAPHY_HEAD.search(b)] if m), None)
    if hit is None:
        return None
    start, m = hit
    rank = int(m.group(1)[1])
    end = len(bodies)
    for k in range(start + 1, len(bodies)):
        hm = re.search(r"<h([1-6])[^>]*>", bodies[k])
        if hm and int(hm.group(1)) <= rank \
                and not BIBLIOGRAPHY_HEAD.search(bodies[k]):
            end = k
            break

    entries = []                     # {id, page, surnames, years, title}
    ad_keys: dict = {}               # (form_lower, year) -> entry | "ambiguous"
    ti_keys = []                     # (surname, title_norm, entry)
    last_surname = None
    for k in range(start, end):
        body = bodies[k]
        edits = []
        for bm in re.finditer(r"<(p|li)([^>]*)>(.*?)</\1>", body, re.S):
            own = bm.group(3)
            eid = f"bib-{k:04d}-{bm.start()}"
            im = re.search(r"<i>(.*?)</i>", own, re.S)
            title = _norm_title(re.sub(r"<[^>]+>", "", im.group(1))) if im else ""
            adm = AD_ENTRY.match(own)
            surnames: set = set()
            years: set = set()
            if adm:
                surnames = _surname_forms(adm.group(1))
                years = set(YEAR_ALONE.findall(re.sub(r"<[^>]+>", "", own)))
            elif im:
                lead = re.sub(r"<[^>]+>", "", own[:im.start()]).strip()
                sm = re.match(r"([A-Z][A-Za-z’'\-]{2,}),", lead)
                if sm:
                    surnames = _surname_forms(sm.group(1))
                    last_surname = sm.group(1)
                elif not lead and last_surname:
                    surnames = _surname_forms(last_surname)
            if surnames and not years:
                # The entry names an author but no parenthesised year, which
                # is how a whole style of bibliography is set. Strictly a
                # top-up: the entry keeps the branch, surnames and title key
                # it already earned, and only gains the (surname, year) key
                # it was missing. Anthropology's World holds 260 such entries
                # and could bind none of its author-date citations.
                bym = BARE_YEAR_ENTRY.match(re.sub(r"<[^>]+>", "", own))
                if bym:
                    years = {bym.group(1)}
            if not surnames:
                continue
            entry = {"id": eid, "page": k, "surnames": surnames,
                     "years": years, "title": title}
            entries.append(entry)
            # in a fixed order: a set of strings is listed differently in
            # every process, and the order of the keys is the order their
            # patterns are tried in (see link_citations)
            for s in sorted(surnames):
                for y in sorted(years):
                    # keyed lowercase so Adams and ADAMS collide as they
                    # should, but the ORIGINAL spelling rides along: the
                    # prose is searched for "AFI", not a guessed "Afi"
                    key = (s.lower(), y)
                    ad_keys[key] = ("ambiguous" if key in ad_keys
                                    else (entry, s))
                if len(title) > 3:
                    ti_keys.append((s, title, entry))
            if 'id="' not in bm.group(2):
                edits.append((bm.start(), f"<{bm.group(1)} id=\"{eid}\""
                              + bm.group(2) + ">", bm.end(1) + bm.start()))
        # anchor the entries; ids add no visible words by construction
        for pos, opening, _ in sorted(edits, reverse=True):
            tag_end = body.index(">", pos) + 1
            body = body[:pos] + opening + body[tag_end:]
        bodies[k] = body
    if not entries:
        return None
    return {"start": start, "end": end, "entries": entries,
            "ad_keys": ad_keys, "ti_keys": ti_keys}


def link_citations(bodies: list[str], dropped: set) -> dict:
    """
    Bind in-text citations to the bibliography, driven from the bibliography.

    The prose is never searched for anything shaped like a citation — it is
    searched for the specific names and years the bibliography actually
    holds. "New York: Harper, 1933" can never bind because no author is
    called New; the junk that a general pattern must catch and filter is
    simply never a candidate. A key naming two entries binds nothing.

    Two styles, matching the corpus: author-date in running prose
    ("Bernards (2019a)", "(see Adams 1971)", "(AFI 2010:1)"), and Chicago
    short-form inside NOTE paragraphs only, where the italic title names the
    work ("Altgeld, <i>Live Questions</i>, p. 525"). Note paragraphs are the
    boundary because running prose italicises titles in passing mention,
    and a mention is not a citation.

    Unlike every other linker here this one rewrites inside running
    sentences, so each edited page must read back word-for-word identical —
    a page that does not is reverted whole and counted, never shipped.
    """
    stats = {"authordate": 0, "chicago": 0, "ambiguous": 0,
             "reverted": 0, "entries": 0}
    bib = parse_bibliography(bodies)
    if bib is None:
        return stats
    stats["entries"] = len(bib["entries"])

    ad = {k: e for k, e in bib["ad_keys"].items() if e != "ambiguous"}
    stats["ambiguous"] += sum(1 for e in bib["ad_keys"].values()
                              if e == "ambiguous")
    NOTE_P = re.compile(r"<(p|li)[^>]*(?:id=\"(?:note-|fn-)[^\"]*\")[^>]*>"
                        r"|<(p|li)[^>]*>\s*(?:<sup>|\d{1,3}[.)]\s)")

    # entry patterns depend only on the bibliography, never the page —
    # compiling them inside the page loop cost pages x entries compiles
    # and could churn Python's regex cache on a large bibliography (E5)
    ad_pats = [((form, year), (entry, spelling),
                re.compile(re.escape(spelling)
                           + r"(?:’s|'s)?(?:\s+et\s+al\.?)?[,\s]*\(?\s*"
                           + re.escape(year) + r"(?!\d)"))
               for (form, year), (entry, spelling) in ad.items()]
    for k in range(0, bib["start"]):
        if k in dropped:
            continue
        body = bodies[k]
        before = _strip_tags(body).split()
        edits = []

        # author-date, in any prose on the page
        for (form, year), (entry, spelling), pat in ad_pats:
            for m in pat.finditer(body):
                if not _outside_markup(body, m.start(), m.end()):
                    continue
                edits.append((m.start(), m.end(),
                              f'<a href="page_{entry["page"]:04d}.xhtml'
                              f'#{entry["id"]}">{body[m.start():m.end()]}</a>',
                              "authordate"))

        # Chicago short form, inside note paragraphs only
        for bm in re.finditer(r"<(p|li)([^>]*)>(.*?)</\1>", body, re.S):
            if not NOTE_P.match(bm.group(0)):
                continue
            own, base = bm.group(3), bm.start(3)
            for im in re.finditer(r"<i>(.*?)</i>", own, re.S):
                t = _norm_title(re.sub(r"<[^>]+>", "", im.group(1)))
                if len(t) < 4:
                    continue
                window = re.sub(r"<[^>]+>", " ", own[:im.start()])[-90:]
                cands = {id(e): e for s, bt, e in bib["ti_keys"]
                         if (bt.startswith(t) or t.startswith(bt))
                         and re.search(r"\b" + re.escape(s) + r"\b", window)}
                if len(cands) > 1:
                    stats["ambiguous"] += 1
                    continue
                if len(cands) != 1:
                    continue
                e = next(iter(cands.values()))
                a, z = base + im.start(), base + im.end()
                opens = len(re.findall(r"<a[\s>]", body[:a]))
                if opens != body.count("</a>", 0, a):
                    continue              # already inside an anchor
                edits.append((a, z,
                              f'<a href="page_{e["page"]:04d}.xhtml'
                              f'#{e["id"]}">{body[a:z]}</a>', "chicago"))

        if not edits:
            continue
        # apply without overlaps, last first so offsets hold. Two keys can
        # match from the same place: "Meulemann, H. (2008a) ‘Introduction’,
        # in Meulemann (ed.) (2008)" keys its entry on 2008a and on 2008,
        # and "Meulemann (2008a, p.9)" matches both. The longer is the
        # citation; which came first was the order a set listed the years
        # in, a different one in every process, and Theories of Social
        # Capital linked "(2008a" in one build and "(2008" in the next.
        edits.sort(key=lambda t: (t[0], t[1]), reverse=True)
        taken_from = len(body) + 1
        applied = []
        for a, z, new, kind in edits:
            if z > taken_from:
                continue
            body = body[:a] + new + body[z:]
            taken_from = a
            applied.append(kind)
        # The assertion this linker is not allowed to ship without: an
        # anchor adds no words, so the page must read back identical.
        if _strip_tags(body).split() != before:
            stats["reverted"] += 1
            continue
        bodies[k] = body
        stats["authordate"] += applied.count("authordate")
        stats["chicago"] += applied.count("chicago")
    return stats
