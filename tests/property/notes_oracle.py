"""Checks shared by the note-linker properties: what any linked book must
satisfy whatever the linker decided, written without the linker's own
patterns.

The generators mark every call with a word "c<uid>" set hard against it and
every note entry with a word "e<uid>" as its first word, and record what
each was meant to answer. A link can then be judged by what the generator
meant, not by what the linker found.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

EPUB_NS = "http://www.idpf.org/2007/ops"
SYMBOLS = ("**", "*", "†", "‡")

_NOTEREF = re.compile(r'<a epub:type="noteref"([^>]*)>([^<]*)</a>')
_ID = re.compile(r'\bid="([^"]+)"')
_HREF = re.compile(r'\bhref="([^"]+)"')


def parses(page: str) -> bool:
    """Whether a page body is a well-formed XHTML fragment."""
    try:
        ET.fromstring(f'<div xmlns:epub="{EPUB_NS}">{page}</div>')
    except ET.ParseError:
        return False
    return True


def unlinked(page: str) -> str:
    """The page with every link a linker writes taken back off: anchors
    unwrapped, the ids the linkers write dropped, a symbol call's added
    superscript removed, whitespace removed. Linking adds links and nothing
    else, so this must equal the same of the input."""
    page = re.sub(r'<sup><a epub:type="noteref"[^>]*>(\*\*|\*|†|‡)</a></sup>',
                  r"\1", page)
    page = re.sub(r"<a\b[^>]*>|</a>", "", page)
    page = re.sub(r'\s+id="(?:fn|fnref|tn|tnref|note|ref)-[^"]*"', "", page)
    return re.sub(r"\s+", "", page)


def bare(page: str) -> str:
    """The same of an input page (which may carry paragraph attributes the
    foot linker drops)."""
    return re.sub(r"\s+", "", page)


def text_chars(page: str) -> str:
    """The page's printed characters, markup and spacing gone."""
    return "".join(html.unescape(re.sub(r"<[^>]+>", " ", page)).split())


def page_of(href: str, here: int) -> "tuple[int, str]":
    if href.startswith("#"):
        return here, href[1:]
    m = re.fullmatch(r"page_(\d{4})\.xhtml#(.+)", href)
    assert m, f"unexpected href shape {href!r}"
    return int(m.group(1)), m.group(2)


def check_links(pages: "list[str]") -> None:
    """Every id unique in the book; every href a linker wrote resolves."""
    seen: dict = {}
    for i, p in enumerate(pages):
        for ident in _ID.findall(p):
            assert ident not in seen, \
                f"id {ident!r} on page {i} already on page {seen[ident]}"
            seen[ident] = i
    for i, p in enumerate(pages):
        for href in _HREF.findall(p):
            if not (href.startswith("#") or href.startswith("page_")):
                continue
            pg, ident = page_of(href, i)
            assert 0 <= pg < len(pages), f"href {href!r} on page {i}: no such page"
            assert f'id="{ident}"' in pages[pg], \
                f"href {href!r} on page {i}: no id {ident!r} on page {pg}"


def shown_at(page: str, ident: str) -> str:
    """The figure or symbol a note entry shows, read from where its id is."""
    at = page.index(f'id="{ident}"')
    rest = page[at:]
    rest = rest[rest.index(">") + 1:]
    txt = html.unescape(re.sub(r"<[^>]+>", "", rest[:200])).lstrip()
    m = re.match(r"(\*\*|\*|†|‡|\d{1,3})", txt)
    return m.group(1) if m else ""


def token_after(page: str, ident: str, letter: str = "e") -> "int | None":
    """The generator's token for the entry whose id this is: the first
    word "e<uid>" before that entry's paragraph or item closes."""
    at = page.index(f'id="{ident}"')
    end = re.compile(r"</p>|</li>").search(page, at)
    m = re.compile(rf"\b{letter}(\d+)\b").search(
        page, at, end.start() if end else len(page))
    return int(m.group(1)) if m else None


def noterefs(pages: "list[str]"):
    """(page, call uid or None, shown, target page, target id) per link."""
    for i, p in enumerate(pages):
        for m in _NOTEREF.finditer(p):
            href = _HREF.search(m.group(1)).group(1)
            tp, tid = page_of(href, i)
            before = p[:m.start()]
            before = re.sub(r"<sup>\s*$", "", before)
            cm = re.search(r"\bc(\d+)$", before)
            yield i, (int(cm.group(1)) if cm else None), m.group(2), tp, tid


def check_calls_meet_their_entries(pages, call_intent, entry_intent,
                                   tolerate=lambda want, got, i, tp: False):
    """A linked call showing n reaches an entry showing n, and the entry the
    generator meant it for (where the generator meant one). `tolerate`
    names pairings the page cannot tell from the meant one."""
    for i, uid, shown, tp, tid in noterefs(pages):
        target = shown_at(pages[tp], tid)
        assert target == shown, \
            f"call {shown!r} on page {i} reaches an entry showing {target!r}"
        if uid is None:
            continue
        want = call_intent.get(uid)
        if want is None:
            continue
        got_uid = token_after(pages[tp], tid)
        got = entry_intent.get(got_uid)
        assert got == want or tolerate(want, got, i, tp), (f"call c{uid} ({shown}) on page {i} meant {want}, "
                             f"linked to e{got_uid} = {got}")
