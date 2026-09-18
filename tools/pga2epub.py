#!/usr/bin/env python3
"""Project Gutenberg Australia HTML -> EPUB 3.

PGA pages declare `charset=windows-1252` and lie. Their accented letters
are UTF-8 bytes that were decoded as cp1252 and then entity-encoded, so
an e-acute is stored as `&Atilde;&copy;` and a s-caron as
`&Aring;&iexcl;`. Decoding the entities, encoding the result back to
cp1252 and reading those bytes as UTF-8 undoes it exactly. Saving the
page from a browser performs the same round trip, which makes a saved
copy a second witness rather than a second source.

Two kinds of damage survive that reversal, and the second is the one
that bites:

  * a character is simply **gone** — the source has the first half of
    the pair and nothing after it. The bytes are then invalid UTF-8 and
    `inspect` points straight at them.
  * a character is **wrong but valid**. In one book `&mdash;` (cp1252
    0x97) stood where `&ndash;` (0x96) belonged, so Ohningen read as
    "×hningen" six times and decoded without a murmur. Nothing can find
    that but reading every non-ASCII character in the book, which is why
    `inspect` prints all of them with their context.

Both are repaired from a per-book JSON file, and every repair states how
many times it must fire. A repair that matches a different number of
places than it claims stops the build.

    python tools/pga2epub.py inspect SOURCE [--json]
    python tools/pga2epub.py build BOOK.json [-o OUT.epub]

SOURCE is a URL or a local file. `inspect` reads a book without
building it: the encoding damage, every non-ASCII character in context,
the structural shape, and a starter JSON to fill in.

**On structure.** PGA markup is not consistent between books — one is a
novel in `div.chapter` blocks nested in `div#bookN` parts, the next is a
short story with no structural markup at all. So this reads only the
shapes it has actually been exercised on, names the one it found, and
refuses to guess:

    divs   div.chapter blocks, optionally inside div#book* parts
    flow   one heading level splits the body; everything else is prose

`inspect` says which fits. If neither does, it says that too, and the
right answer is to write the reader against the book in front of you
rather than to widen one of these until it swallows everything.
"""
from __future__ import annotations

import argparse
import datetime as dt
import html as htmlmod
import json
import re
import shutil
import sys
import unicodedata
import urllib.request
import uuid
import zipfile
from pathlib import Path

from lxml import etree, html as lhtml

XHTMLNS = "http://www.w3.org/1999/xhtml"
OPFNS = "http://www.idpf.org/2007/opf"
NCXNS = "http://www.daisy.org/z3986/2005/ncx/"

# An unrepaired byte, as backslashreplace leaves it. Only the high half:
# a real backslash-x in the prose could otherwise look like damage.
STRAY = re.compile(r"\\x[89a-f][0-9a-f]")

# No banner, licence block or footer on that site comes near this; a
# chapter that happened to quote one of the marks would clear it easily.
FURNITURE_CEILING = 2000

FURNITURE_MARKS = (
    "treasure-trove of literature", "Google Site Search", "BROWSE the site",
    "This site is full of FREE ebooks", "gutenberg.net.au/licence",
    "A Project Gutenberg of Australia eBook",
)


# ---------------------------------------------------------------------------
# reading and repairing
# ---------------------------------------------------------------------------

def read_source(src: str) -> str:
    """Raw markup, as text, from a URL or a path."""
    if src.startswith(("http://", "https://")):
        req = urllib.request.Request(
            src, headers={"User-Agent": "codicology/pga2epub"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    else:
        data = Path(src).read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        # A browser-saved copy is already cp1252 bytes; read it as such
        # and the entity pass below has nothing left to do.
        return data.decode("cp1252", errors="backslashreplace")


def unescape_keeping_markup(raw: str) -> str:
    """Resolve entities, but leave the four that are markup alone."""
    def one(m: re.Match) -> str:
        s = htmlmod.unescape(m.group(0))
        return m.group(0) if len(s) == 1 and s in "<>&\"'" else s
    return re.sub(r"&[#a-zA-Z0-9]+;", one, raw)


def mojibake_score(text: str) -> tuple[int, int]:
    """How much of this text reads as UTF-8 seen through cp1252.

    A two-byte UTF-8 sequence looks like a capital in A-grave..eszett
    followed by one of cp1252's punctuation or symbols -- a pair that
    barely occurs in text that means what it says. Counting them
    separates a mojibake page from a page that is simply correct, and
    not every PGA book is mojibake: demojibaking a clean one would
    wreck it.
    """
    data = text.encode("cp1252", errors="replace")
    pairs, i = 0, 0
    while i < len(data) - 1:
        if 0xc2 <= data[i] <= 0xdf and 0x80 <= data[i + 1] <= 0xbf:
            pairs += 1
            i += 2
        else:
            i += 1
    return pairs, sum(1 for ch in text if ord(ch) > 127)


def is_mojibake(text: str) -> bool:
    pairs, non_ascii = mojibake_score(text)
    return pairs >= 3 and pairs * 2 >= non_ascii * 0.5


def demojibake(raw: str) -> str:
    """cp1252 text that was really UTF-8 -> the text it was meant to be.

    Bytes that no longer form valid UTF-8 come back as their own escape,
    `\\xc3`, so a repair can name them in a JSON file as plain text.
    """
    data = raw.encode("cp1252", errors="xmlcharrefreplace")
    return data.decode("utf-8", errors="backslashreplace")


def decode_text(markup: str, force: bool | None = None) -> tuple[str, bool]:
    """A page's text, and whether it had to be demojibaked to get it."""
    plain = unescape_keeping_markup(markup)
    moji = is_mojibake(plain) if force is None else force
    return (demojibake(plain) if moji else plain), moji


def load_text(src: str, force: bool | None = None) -> tuple[str, bool]:
    return decode_text(read_source(src), force)


def apply_repairs(text: str, repairs: list[dict]) -> tuple[str, list[str]]:
    """Each repair replaces exactly the number of places it claims."""
    log = []
    for r in repairs:
        bad, good, want = r["bad"], r["good"], int(r["count"])
        got = text.count(bad)
        if got != want:
            raise SystemExit(
                f"repair {bad!r} -> {good!r}: matches {got} places, "
                f"the book file claims {want}. Re-run inspect: either the "
                f"source changed or the repair is wrong.")
        text = text.replace(bad, good)
        log.append(f"{bad!r} -> {good!r} ({want}x) — {r.get('why', '')}")
    left = sorted(set(STRAY.findall(text)))
    if left:
        raise SystemExit(
            f"unrepaired bytes remain: {', '.join(left)}. Run inspect and "
            f"add a repair for each, or state that it should stay.")
    return text, log


# ---------------------------------------------------------------------------
# inspecting
# ---------------------------------------------------------------------------

def damage(text: str) -> list[tuple[str, str]]:
    out = []
    for m in STRAY.finditer(text):
        out.append((m.group(), " ".join(
            text[max(0, m.start() - 60):m.end() + 40].split())))
    return out


def non_ascii(text: str) -> list[tuple[str, str, str]]:
    """Every non-ASCII character with its context.

    The whole point: a wrong-but-valid substitution is invisible to a
    decoder and obvious to a reader. There are about a hundred of these
    in a 90,000-word book.
    """
    body = re.sub(r"<[^>]+>", " ", text)
    out = []
    for m in re.finditer(r"[^\x00-\x7f]", body):
        c = m.group()
        out.append((c, unicodedata.name(c, "?"), " ".join(
            body[max(0, m.start() - 45):m.start() + 30].split())))
    return out


def parse(text: str):
    """The body, with comments dropped.

    PGA pages carry their analytics and ad tags commented out; those are
    furniture, and a comment node is not an element, which every walk
    below would otherwise have to say out loud.
    """
    doc = lhtml.document_fromstring(text)
    body = doc.body if doc.body is not None else doc
    for node in list(body.iter()):
        if not isinstance(node.tag, str):
            node.drop_tree() if hasattr(node, "drop_tree") else None
    return body


def shape(body) -> dict:
    chapters = body.xpath('.//div[contains(concat(" ", normalize-space(@class),'
                          ' " "), " chapter ")]')
    parts = body.xpath('.//div[starts-with(@id, "book")]')
    heads = {}
    for lvl in range(1, 7):
        heads[f"h{lvl}"] = len(body.xpath(f".//h{lvl}"))
    return {
        "chapter_divs": len(chapters),
        "part_divs": len(parts),
        "headings": {k: v for k, v in heads.items() if v},
        "images": [i.get("src") for i in body.xpath(".//img")],
        "note_targets": len(body.xpath('.//*[starts-with(@id, "note")]')),
        "note_markers": len(body.xpath('.//a[starts-with(@href, "#note")]')),
        "reader": ("divs" if len(chapters) >= 3 else
                   "flow" if any(v >= 3 for k, v in heads.items()
                                 if k in ("h2", "h3")) else "none"),
    }


def header_block(text: str) -> dict:
    """The Title:/Author: block PGA puts in a <pre> near the top."""
    m = re.search(r"<pre>(.*?)</pre>", text, re.S)
    out = {}
    if m:
        for line in m.group(1).splitlines():
            k, _, v = line.partition(":")
            if v.strip() and len(k) < 32:
                out[k.strip().lower()] = v.strip()
    return out


def inspect(src: str, as_json: bool) -> int:
    text, moji = load_text(src)
    body = parse(text)
    sh = shape(body)
    hdr = header_block(text)
    dmg = damage(text)
    census = non_ascii(text)

    skeleton = {
        "source": src,
        "title": hdr.get("title", ""),
        "creator": {"name": hdr.get("author", ""), "file_as": ""},
        "contributors": [],
        "language": "en",
        "date": "",
        "publisher": "Project Gutenberg Australia",
        "rights": "",
        "description": "",
        "subjects": [],
        "reader": sh["reader"],
        "mojibake": moji,
        "repairs": [{"bad": b, "good": "", "count":
                     sum(1 for x, _ in dmg if x == b), "why": ""}
                    for b in sorted({b for b, _ in dmg})],
        "images": {},
        "cover": None,
    }
    if as_json:
        print(json.dumps(skeleton, indent=2, ensure_ascii=False))
        return 0

    pairs, na = mojibake_score(unescape_keeping_markup(read_source(src)))
    print(f"source   {src}")
    print(f"  encoding   {'mojibake, reversed' if moji else 'read as it is'} "
          f"({pairs} cp1252 pairs read as UTF-8, {na} non-ASCII characters)")
    for k, v in hdr.items():
        print(f"  {k:<10} {v}")
    print(f"\nstructure")
    print(f"  chapter divs {sh['chapter_divs']}   part divs "
          f"{sh['part_divs']}   headings {sh['headings']}")
    print(f"  note markers {sh['note_markers']} -> targets "
          f"{sh['note_targets']}   images {len(sh['images'])}")
    print(f"  reader that fits: {sh['reader']}")

    print(f"\nencoding damage — {len(dmg)} characters did not survive")
    for esc, ctx in dmg:
        print(f"  {esc}  {ctx}")

    print(f"\nnon-ASCII census — {len(census)} characters, "
          f"{len({c for c, _, _ in census})} distinct")
    print("  read these: a wrong-but-valid substitution shows up here "
          "and nowhere else")
    for c, name, ctx in census:
        print(f"  {c!r:8} {name:<34} {ctx}")

    print("\nstarter book file: re-run with --json")
    return 0


# ---------------------------------------------------------------------------
# structure
# ---------------------------------------------------------------------------

class Chapter:
    """One chapter, plus the source text it was read from.

    `source_text` is captured off the parse tree before anything is
    cleaned or moved, so the verifier can compare what was emitted
    against what was read rather than against another of the builder's
    own intermediates.
    """

    def __init__(self, label: str, title: str, title_html: str,
                 body: str, source_text: str, notes: str = "",
                 note_ids: tuple = ()):
        self.label = label
        self.title = title
        self.title_html = title_html
        self.body = body
        self.source_text = source_text
        self.notes = notes
        self.note_ids = note_ids

    @property
    def nav(self) -> str:
        if self.label and self.title:
            return f"{self.label}. {self.title}"
        return self.title or self.label


def is_chapter(el) -> bool:
    return (el.tag == "div"
            and "chapter" in (el.get("class") or "").split())


def is_part(el) -> bool:
    return el.tag == "div" and (el.get("id") or "").startswith("book")


def strip_site_furniture(body) -> list[str]:
    """Drop the site's own banner, licence boilerplate and footer.

    A book in chapter divs never sees this text, because the reader only
    ever looks inside the chapters. A book read as a flow does: with no
    structural markup the banner lands in chapter one and the footer in
    the last chapter. Only whole top-level blocks go, only when they
    carry one of the site's own strings, and never a block big enough to
    be the book.
    """
    gone = []
    for node in list(body):
        if not isinstance(node.tag, str) or is_chapter(node) or is_part(node):
            continue
        text = node.text_content()
        if not any(m in text for m in FURNITURE_MARKS):
            continue
        if len(squash(text)) > FURNITURE_CEILING:
            gone.append(f"kept a {len(squash(text))}-character <{node.tag}>: "
                        "too much of the book to be furniture")
            continue
        gone.append(f"<{node.tag}> {' '.join(text.split())[:60]!r}")
        node.drop_tree()
    return gone


def absorb_strays(body) -> int:
    """Give each chapter the blocks a stray </div> pushed out of it.

    One book closes a chapter one div too early, so a paragraph and a
    half of it are the chapter's siblings rather than its children. A
    reader that walks chapter divs alone drops them, and drops them
    silently: the words are ordinary ones the book uses elsewhere, so
    nothing that counts words can see it.
    """
    moved = 0
    containers = [body] + [e for e in body.iterdescendants("div")
                           if is_part(e)]
    for container in containers:
        current = None
        for node in list(container):
            if is_chapter(node):
                current = node
                continue
            if is_part(node) or node.tag in ("h1", "h2", "hr"):
                current = None
                continue
            if current is None or not isinstance(node.tag, str):
                continue
            current.append(node)
            moved += 1
    return moved


VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input",
        "link", "meta", "param", "source", "track", "wbr"}
SELF_CLOSED = re.compile(r"<([a-z][a-z0-9]*)((?:\s[^<>]*?)?)/>")


def expand_empty(markup: str) -> str:
    """`<em/>` is legal XML and a trap: a reader that falls back to an
    HTML parser reads it as an opening tag and italicises the rest of
    the file. Only the void elements may stay self-closed."""
    return SELF_CLOSED.sub(
        lambda m: (m.group(0) if m.group(1) in VOID
                   else f"<{m.group(1)}{m.group(2)}></{m.group(1)}>"),
        markup)


def inner_html(el) -> str:
    """An element's contents, serialised as XML."""
    out = [htmlmod.escape(el.text, quote=False) if el.text else ""]
    for child in el:
        out.append(etree.tostring(child, encoding="unicode", method="xml"))
    return expand_empty("".join(out)).strip()


def clean(el, images: dict) -> None:
    """Turn presentational leftovers into classes the stylesheet knows."""
    for span in list(el.iter("span")):
        style = (span.get("style") or "").replace(" ", "")
        if style == "text-transform:uppercase":
            span.attrib.clear()
            span.set("class", "upper")
        elif style.startswith("text-align:center"):
            span.attrib.clear()
            span.set("class", "center")
        else:
            span.drop_tag()
    for node in list(el.iter()):
        if not isinstance(node.tag, str):
            continue
        if node.tag in ("font", "script", "noscript", "iframe"):
            node.drop_tree()
            continue
        if node.get("align") == "center":
            node.set("class", (node.get("class", "") + " center").strip())
        for attr in ("align", "border", "bgcolor"):
            node.attrib.pop(attr, None)
        if node.tag != "img":
            for attr in ("width", "height"):
                node.attrib.pop(attr, None)
        if node.tag == "div" and "section" in (node.get("class") or "").split():
            node.tag = "section"
            node.set("class", "scene")
        if node.tag == "img":
            base = (node.get("src") or "").rsplit("/", 1)[-1]
            spec = images.get(base)
            if spec is None:
                continue
            node.set("src", "../images/" + spec["name"])
            node.set("alt", spec.get("alt", ""))
            for attr in ("width", "height", "border"):
                node.attrib.pop(attr, None)
            parent = node.getparent()
            if parent is not None and parent.tag == "p":
                parent.set("class",
                           (parent.get("class", "") + " figure").strip())


def head_pair(div, images: dict) -> tuple[str, str, str]:
    """A chapter's label line, its title, and its title as markup.

    One chapter carries a note marker inside its title. Rebuilding the
    heading from its text alone would strand that note, so the markup
    comes back too.
    """
    h3 = next((c for c in div if c.tag == "h3"), None)
    h4 = next((c for c in div if c.tag == "h4"), None)
    label = " ".join(h3.text_content().split()) if h3 is not None else ""
    if h4 is None:
        return label, "", ""
    plain = lhtml.fromstring(etree.tostring(h4, encoding="unicode"))
    for sup in list(plain.iter("sup")):
        (sup.getparent() if sup.getparent().tag == "a" else sup).drop_tree()
    title = " ".join(plain.text_content().split())
    clean(h4, images)
    return label, title, inner_html(h4)


def take_notes(div, images: dict) -> tuple[str, tuple]:
    """Lift a trailing NOTES run into an endnote section with backlinks."""
    head = next((h for h in div.iter("h3")
                 if h.text_content().strip().upper() == "NOTES"), None)
    if head is None:
        return "", ()
    tail = []
    node = head.getnext()
    while node is not None:
        nxt = node.getnext()
        div.remove(node)
        tail.append(node)
        node = nxt
    prev = head.getprevious()
    while prev is not None and prev.tag == "hr":
        gone, prev = prev, prev.getprevious()
        div.remove(gone)
    div.remove(head)

    ids, parts = [], []
    for el in tail:
        if el.tag == "hr":
            continue
        clean(el, images)
        nid = el.get("id") or ""
        if el.tag == "blockquote" and nid.startswith("note"):
            num = nid[4:]
            ids.append(nid)
            b = next(iter(el.iter("b")), None)
            if b is not None:
                printed = b.text_content()
                lead, printed = (("[", printed[1:])
                                 if printed.startswith("[") else ("", printed))
                for child in list(b):
                    b.remove(child)
                b.text = lead
                link = etree.SubElement(b, "a")
                link.set("class", "noteback")
                link.set("href", f"#ref{num}")
                link.set("role", "doc-backlink")
                link.text = printed
            parts.append(
                f'<aside epub:type="endnote" role="doc-endnote" id="{nid}">'
                f"{inner_html(el)}</aside>")
        else:
            parts.append(expand_empty(
                etree.tostring(el, encoding="unicode", method="xml")))
    return ('\n<section epub:type="endnotes" role="doc-endnotes" '
            'class="notes" id="notes">\n<h2>NOTES</h2>\n'
            + "\n".join(parts) + "\n</section>\n"), tuple(ids)


def build_chapter(div, images: dict) -> Chapter:
    source_text = div.text_content()
    label, title, title_html = head_pair(div, images)
    for tag in ("h3", "h4"):
        first = next((c for c in div if c.tag == tag), None)
        if first is not None:
            div.remove(first)
    notes, ids = take_notes(div, images)
    clean(div, images)
    return Chapter(label, title, title_html, inner_html(div), source_text,
                   notes, ids)


def read_divs(body, images: dict) -> list[tuple[str, object]]:
    """div.chapter blocks, optionally inside div#book* parts."""
    out = []
    for el in body:
        if is_part(el):
            label, title, _ = head_pair(el, images)
            out.append(("part", (label, title)))
            for ch in [c for c in el if is_chapter(c)]:
                el.remove(ch)
                out.append(("chapter", build_chapter(ch, images)))
        elif is_chapter(el):
            out.append(("chapter", build_chapter(el, images)))
    return out


def read_flow(body, images: dict, level: str,
              starts: str = "") -> list[tuple[str, object]]:
    """One heading level splits the body; everything else is prose.

    For books with no structural markup at all. The heading level that
    marks chapters is usually also used for a byline or a dedication, so
    `flow_starts` names the first real chapter and everything before it
    is left out -- where the coverage check then insists the book file
    says so.
    """
    groups: list[tuple[str, list]] = [("", [])]
    for el in body:
        if not isinstance(el.tag, str):
            continue
        if el.tag == level:
            groups.append((" ".join(el.text_content().split()), []))
        else:
            groups[-1][1].append(el)
    if starts:
        first = next((i for i, (t, _) in enumerate(groups) if starts in t),
                     None)
        if first is None:
            raise SystemExit(f"flow_starts {starts!r} matches no {level}")
        groups = groups[first:]
    out = []
    for title, nodes in groups:
        if not title and not any(n.text_content().strip() for n in nodes):
            continue
        holder = lhtml.Element("div")
        for n in nodes:
            holder.append(n)
        source_text = title + holder.text_content()
        clean(holder, images)
        out.append(("chapter",
                    Chapter("", title, htmlmod.escape(title),
                            inner_html(holder), source_text)))
    return out


# ---------------------------------------------------------------------------
# emitting
# ---------------------------------------------------------------------------

XHTML = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" \
xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}" xml:lang="{lang}">
<head>
<meta charset="utf-8"/>
<title>{title}</title>
<link rel="stylesheet" type="text/css" href="{css}"/>
</head>
<body epub:type="{etype}"{cls}>
{body}
</body>
</html>
"""

CSS = (Path(__file__).resolve().parent / "pga" / "main.css")


class Doc:
    def __init__(self, name, title, body, *, etype="bodymatter",
                 nav=None, level=1, cls=""):
        self.name, self.title, self.body = name, title, body
        self.etype, self.nav, self.level, self.cls = etype, nav, level, cls
        self.note_ids: tuple = ()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def esc(s: str) -> str:
    return htmlmod.escape(s, quote=False)


def assemble(cfg: dict, sections, build: Path) -> list[Doc]:
    lang = cfg.get("language", "en")
    title = cfg["title"]
    docs: list[Doc] = []

    if cfg.get("cover"):
        docs.append(Doc(
            "cover.xhtml", "Cover",
            '<section epub:type="cover" role="doc-cover">\n'
            f'<img src="../images/cover.png" alt="{esc(cfg["cover"]["alt"])}"/>'
            "\n</section>", etype="cover", cls=' class="cover"'))

    byline = cfg["creator"]["name"]
    extra = "".join(
        f'<p class="translator">{esc(c.get("credit", c["name"]))}</p>\n'
        for c in cfg.get("contributors", []))
    docs.append(Doc(
        "titlepage.xhtml", "Title page",
        '<section epub:type="titlepage" class="titlepage">\n'
        f"<h1>{esc(title)}</h1>\n"
        f'<p class="byline">{esc(byline)}</p>\n{extra}</section>',
        etype="frontmatter", nav="Title page"))

    seq = part = 0
    for kind, payload in sections:
        if kind == "part":
            part += 1
            label, sub = payload
            docs.append(Doc(
                f"part{part}.xhtml", label,
                '<section epub:type="part" role="doc-part" class="part">\n'
                f"<h1>{esc(label)}</h1>\n"
                f'<p class="part-title">{esc(sub)}</p>\n</section>',
                nav=f"{label} — {sub}" if sub else label))
            continue
        seq += 1
        ch = payload
        head = "<header>\n"
        if ch.label:
            head += f'<p class="chapnum">{esc(ch.label)}</p>\n'
        head += f"<h1>{ch.title_html or esc(ch.label)}</h1>\n</header>\n"
        doc = Doc(f"ch{seq:02d}.xhtml", ch.nav,
                  '<section epub:type="chapter" role="doc-chapter">\n'
                  f"{head}{ch.body}{ch.notes}</section>",
                  nav=ch.nav, level=2)
        doc.note_ids = ch.note_ids
        docs.append(doc)

    if cfg.get("the_end"):
        docs[-1].body = docs[-1].body.replace(
            "</section>", f'<p class="theend">{esc(cfg["the_end"])}</p>\n'
                          "</section>")
    return docs


def colophon(cfg: dict, log: list[str]) -> Doc:
    rows = "\n".join(f"<li><code>{esc(line.split(' — ')[0])}</code> — "
                     f"{esc(line.split(' — ', 1)[1])}</li>" if " — " in line
                     else f"<li><code>{esc(line)}</code></li>" for line in log)
    creds = "".join(
        f"<dt>{esc(c.get('label', 'Contributor'))}</dt>"
        f"<dd>{esc(c.get('note') or c['name'])}</dd>\n"
        for c in cfg.get("contributors", []))
    today = dt.date.today().isoformat()
    note = cfg.get("colophon_note_html", "")
    return Doc(
        "colophon.xhtml", "Source and licence",
        '<section epub:type="colophon" class="colophon">\n'
        "<h1>Source and licence</h1>\n<dl>\n"
        f"<dt>Title</dt><dd>{esc(cfg['title'])}</dd>\n"
        f"<dt>Author</dt><dd>{cfg.get('author_html') or esc(cfg['creator']['name'])}</dd>\n"
        f"{creds}"
        f"<dt>Source</dt><dd>{esc(cfg.get('source_note', ''))}<br/>"
        f'<a href="{esc(cfg["source"])}">{esc(cfg["source"])}</a></dd>\n</dl>\n'
        + (f"<p>{cfg['licence_html']}</p>\n" if cfg.get("licence_html") else "")
        + "<h2>About this digital edition</h2>\n"
        f"<p>Built from the HTML edition on {today}: the site’s "
        "navigation furniture removed, the text divided into one file per "
        "chapter, and the printed table of contents replaced by an EPUB "
        "navigation document.</p>\n"
        "<p>The source stores its accented letters as UTF-8 bytes that had "
        "been read as windows-1252 and then entity-encoded. Reversing that "
        "recovers them, except where a character did not survive the round "
        "trip. Those were repaired against the book’s own spellings "
        f"elsewhere:</p>\n<ul>{rows}</ul>\n"
        + (f"<p>{note}</p>\n" if note else "")
        + "</section>",
        etype="backmatter", nav="Source and licence")


def link_notes(docs: list[Doc]) -> int:
    """Bind every `#noteN` marker to its note, and give it an id to
    return to. The source prints the numbers; nothing here renumbers."""
    host = next((d for d in docs if d.note_ids), None)
    marked = 0
    for doc in docs:
        target = "" if doc is host or host is None else host.name

        def mark(m: re.Match) -> str:
            nonlocal marked
            marked += 1
            return (f'<a id="ref{m.group(1)}" class="noteref" '
                    f'epub:type="noteref" role="doc-noteref" '
                    f'href="{target}#note{m.group(1)}">'
                    f"<sup>{m.group(2)}</sup></a>")

        doc.body = re.sub(
            r'<a\s+href=\s*"#note(\d+)"\s*>\s*<sup>(\d+)</sup>\s*</a>',
            mark, doc.body, flags=re.S)
    return marked


def write_nav(docs: list[Doc], build: Path, cfg: dict) -> None:
    items, open_part = [], False
    host = next((d for d in docs if d.note_ids), None)
    for doc in docs:
        if doc.nav is None:
            continue
        label, href = esc(doc.nav), f"text/{doc.name}"
        if doc.level == 1:
            if open_part:
                items.append("</ol></li>")
                open_part = False
            if doc.name.startswith("part"):
                items.append(f'<li><a href="{href}">{label}</a><ol>')
                open_part = True
            else:
                items.append(f'<li><a href="{href}">{label}</a></li>')
        elif doc is host:
            items.append(f'<li><a href="{href}">{label}</a><ol>'
                         f'<li><a href="{href}#notes">Notes</a></li></ol></li>')
        else:
            items.append(f'<li><a href="{href}">{label}</a></li>')
    if open_part:
        items.append("</ol></li>")
    first = next(d for d in docs if d.level == 2 or d.name.startswith("part"))
    nav = ('<nav epub:type="toc" role="doc-toc" id="toc">\n<h1>Contents</h1>\n'
           "<ol>\n" + "\n".join(items) + "\n</ol>\n</nav>\n"
           '<nav epub:type="landmarks" hidden="hidden">\n<ol>\n'
           '<li><a epub:type="titlepage" href="text/titlepage.xhtml">'
           "Title page</a></li>\n"
           f'<li><a epub:type="bodymatter" href="text/{first.name}">'
           "Beginning</a></li>\n"
           '<li><a epub:type="colophon" href="text/colophon.xhtml">'
           "Source and licence</a></li>\n</ol>\n</nav>")
    write(build / "OEBPS" / "nav.xhtml",
          XHTML.format(lang=cfg.get("language", "en"), title="Contents",
                       etype="frontmatter", body=nav, cls="",
                       css="styles/main.css"))


def write_opf(docs: list[Doc], cfg: dict, build: Path, uid: str) -> None:
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    man = ['<item id="nav" href="nav.xhtml" '
           'media-type="application/xhtml+xml" properties="nav"/>',
           '<item id="ncx" href="toc.ncx" '
           'media-type="application/x-dtbncx+xml"/>',
           '<item id="css" href="styles/main.css" media-type="text/css"/>']
    if cfg.get("cover"):
        man.append('<item id="cover-image" href="images/cover.png" '
                   'media-type="image/png" properties="cover-image"/>')
    for spec in cfg.get("images", {}).values():
        name = spec["name"]
        mt = ("image/jpeg" if name.endswith((".jpg", ".jpeg"))
              else "image/gif" if name.endswith(".gif") else "image/png")
        man.append(f'<item id="img-{name.rsplit(".", 1)[0]}" '
                   f'href="images/{name}" media-type="{mt}"/>')
    spine = []
    for i, doc in enumerate(docs):
        ident = f"doc{i:02d}"
        man.append(f'<item id="{ident}" href="text/{doc.name}" '
                   f'media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="{ident}"/>')
        if doc.name == "titlepage.xhtml":
            spine.append('<itemref idref="nav"/>')

    creator = cfg["creator"]
    meta = [f'<dc:identifier id="bookid">{uid}</dc:identifier>',
            f'<dc:title id="t">{esc(cfg["title"])}</dc:title>',
            '<meta refines="#t" property="title-type">main</meta>',
            f'<dc:language>{cfg.get("language", "en")}</dc:language>',
            f'<dc:creator id="creator">{esc(creator["name"])}</dc:creator>',
            '<meta refines="#creator" property="role" '
            'scheme="marc:relators">aut</meta>']
    if creator.get("file_as"):
        meta.append(f'<meta refines="#creator" property="file-as">'
                    f'{esc(creator["file_as"])}</meta>')
    for n, c in enumerate(cfg.get("contributors", [])):
        cid = f"con{n}"
        meta.append(f'<dc:contributor id="{cid}">{esc(c["name"])}'
                    "</dc:contributor>")
        meta.append(f'<meta refines="#{cid}" property="role" '
                    f'scheme="marc:relators">{c.get("role", "ctb")}</meta>')
        if c.get("file_as"):
            meta.append(f'<meta refines="#{cid}" property="file-as">'
                        f'{esc(c["file_as"])}</meta>')
    if cfg.get("date"):
        meta.append(f'<dc:date>{cfg["date"]}</dc:date>')
    for key, tag in (("publisher", "dc:publisher"), ("source", "dc:source"),
                     ("rights", "dc:rights"), ("description",
                                               "dc:description")):
        if cfg.get(key):
            meta.append(f"<{tag}>{esc(cfg[key])}</{tag}>")
    for s in cfg.get("subjects", []):
        meta.append(f"<dc:subject>{esc(s)}</dc:subject>")
    meta.append(f'<meta property="dcterms:modified">{now}</meta>')
    if cfg.get("cover"):
        meta.append('<meta name="cover" content="cover-image"/>')

    nl = "\n    "
    write(build / "OEBPS" / "content.opf",
          '<?xml version="1.0" encoding="utf-8"?>\n'
          '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" '
          f'unique-identifier="bookid" xml:lang="{cfg.get("language", "en")}">\n'
          '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n    '
          + nl.join(meta) + "\n  </metadata>\n  <manifest>\n    "
          + nl.join(man) + '\n  </manifest>\n  <spine toc="ncx">\n    '
          + nl.join(spine) + "\n  </spine>\n</package>\n")


def write_ncx(docs: list[Doc], cfg: dict, build: Path, uid: str) -> None:
    pts = [f'<navPoint id="np{i}" playOrder="{i + 1}">'
           f"<navLabel><text>{esc(d.nav)}</text></navLabel>"
           f'<content src="text/{d.name}"/></navPoint>'
           for i, d in enumerate(d for d in docs if d.nav)]
    write(build / "OEBPS" / "toc.ncx",
          '<?xml version="1.0" encoding="utf-8"?>\n'
          '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" '
          'version="2005-1">\n'
          f'<head><meta name="dtb:uid" content="{uid}"/></head>\n'
          f"<docTitle><text>{esc(cfg['title'])}</text></docTitle>\n<navMap>\n"
          + "\n".join(pts) + "\n</navMap>\n</ncx>\n")


def zip_epub(build: Path, out: Path) -> None:
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, "w") as z:
        z.write(build / "mimetype", "mimetype", zipfile.ZIP_STORED)
        for path in sorted(build.rglob("*")):
            if path.is_file() and path.name != "mimetype":
                z.write(path, str(path.relative_to(build)),
                        zipfile.ZIP_DEFLATED)


# ---------------------------------------------------------------------------
# verifying — runs on every build, not on request
# ---------------------------------------------------------------------------

def squash(t: str) -> str:
    return re.sub(r"\s+", "", t)


def uncovered(body, kept: str, floor: int) -> list[str]:
    """Blocks of the source that reach no chapter and no part page.

    Element by element rather than character by character: a chapter
    orphaned by a stray </div> shows up here as itself, named, instead
    of as a run of characters whose boundaries depend on what happened
    to match elsewhere.
    """
    out = []

    def walk(el):
        for node in el:
            if not isinstance(node.tag, str):
                continue
            if is_chapter(node):
                continue
            if is_part(node):
                walk(node)
                continue
            text = squash(node.text_content())
            if len(text) >= floor and text not in kept:
                out.append(text)

    walk(body)
    return out


def verify(out: Path, sections, cfg: dict, whole) -> list[str]:
    """Re-derive every property from the zip and from what was read.

    Nothing here consults the builder's intermediates: chapters are
    compared against the text captured off the parse tree, `whole` is a
    second, untouched parse of the same source, and the rest comes off
    the file on disk.
    """
    fails = []
    z = zipfile.ZipFile(out)
    names = z.namelist()

    if names[0] != "mimetype" or z.getinfo("mimetype").compress_type != \
            zipfile.ZIP_STORED:
        fails.append("mimetype must be the first entry and stored")

    ids: dict[str, set] = {}
    for n in names:
        if n.endswith((".xhtml", ".opf", ".ncx", ".xml")):
            try:
                root = etree.fromstring(z.read(n))
            except etree.XMLSyntaxError as e:
                fails.append(f"{n} is not well-formed: {e}")
                continue
            if n.endswith(".xhtml"):
                ids[n] = {e.get("id") for e in root.iter() if e.get("id")}

    for n in (x for x in names if x.endswith(".xhtml")):
        base = n.rsplit("/", 1)[0]
        for a in etree.fromstring(z.read(n)).iter(f"{{{XHTMLNS}}}a"):
            href = a.get("href") or ""
            if href.startswith(("http://", "https://", "mailto:")):
                continue
            path, _, frag = href.partition("#")
            target = n if not path else f"{base}/{path}"
            while "/../" in target:
                target = re.sub(r"[^/]+/\.\./", "", target)
            if target not in names:
                fails.append(f"{n} -> {href}: no such file")
            elif frag and frag not in ids.get(target, set()):
                fails.append(f"{n} -> {href}: no such id")

    # Every chapter, character for character, against the text it was
    # read from. Counting words once passed a build that had dropped a
    # paragraph and a half, because the missing words were ordinary ones
    # the book uses elsewhere.
    chapters = [p for k, p in sections if k == "chapter"]
    kept = [squash(lbl + sub) for k, (lbl, sub) in
            ((k, p) for k, p in sections if k == "part")]
    for i, ch in enumerate(chapters, 1):
        name = f"OEBPS/text/ch{i:02d}.xhtml"
        if name not in names:
            fails.append(f"{name} missing")
            continue
        got = squash("".join(
            etree.fromstring(z.read(name)).find(f"{{{XHTMLNS}}}body")
            .itertext()))
        want = squash(ch.source_text)
        kept.append(want)
        if cfg.get("the_end") and i == len(chapters):
            want += squash(cfg["the_end"])
        if want != got:
            k = next((j for j in range(min(len(want), len(got)))
                      if want[j] != got[j]), min(len(want), len(got)))
            fails.append(f"ch{i:02d} diverges at character {k}: "
                         f"{want[k:k + 40]!r} vs {got[k:k + 40]!r}")

    # Nothing in the source is dropped without the book file saying so.
    # This is the check that finds a chapter closed early by a stray
    # </div>: its orphaned paragraphs show up here as an unnamed gap.
    expect = [squash(p) for p in cfg.get("expect_dropped", [])]
    for block in uncovered(whole, "".join(kept), cfg.get("gap_floor", 40)):
        if not any(e and e in block for e in expect):
            fails.append(f"a source block of {len(block)} characters reaches "
                         f"no chapter and no expect_dropped entry names it: "
                         f"{block[:90]!r}")

    # the number a marker prints is the number of the note it reaches
    for n in (x for x in names if x.endswith(".xhtml")):
        body = z.read(n).decode("utf-8")
        for nid, printed in re.findall(
                r'href="[^"]*#note(\d+)"[^>]*>\s*<sup>(\d+)</sup>', body):
            if nid != printed:
                fails.append(f"{n}: a marker printed {printed} reaches "
                             f"note {nid}")
    for mark in FURNITURE_MARKS:
        hits = [n for n in names if n.endswith(".xhtml")
                and mark in z.read(n).decode("utf-8")]
        if hits:
            fails.append(f"site furniture survives in {hits[0]}: {mark!r}")
    return fails


# ---------------------------------------------------------------------------
# cover
# ---------------------------------------------------------------------------

def make_cover(cfg: dict, images_dir: Path, out: Path) -> None:
    """A typographic cover that invents no imagery.

    The mark in the middle, when there is one, is one of the book's own
    illustrations laid on a dark ground.
    """
    from PIL import Image, ImageDraw, ImageFont

    c = cfg["cover"]
    W, H = c.get("size", [1400, 2100])
    ground = tuple(c.get("ground", [14, 40, 44]))
    ink = tuple(c.get("ink", [238, 230, 211]))
    face = c.get("font", "/System/Library/Fonts/Supplemental/Baskerville.ttc")

    img = Image.new("RGB", (W, H), ground)
    d = ImageDraw.Draw(img)

    def font(size, index=0):
        return ImageFont.truetype(face, size, index=index)

    def tracked(y, text, f, track):
        widths = [d.textlength(ch, font=f) for ch in text]
        x = (W - (sum(widths) + track * (len(text) - 1))) / 2
        for ch, w in zip(text, widths):
            d.text((x, y), ch, font=f, fill=ink)
            x += w + track

    def rule(y, width):
        d.line([(W - width) // 2, y, (W + width) // 2, y], fill=ink, width=3)

    rule(150, 260)
    tracked(200, c["author_line"], font(58), 14)
    rule(300, 260)

    y = 430
    for line in c["title_lines"]:
        tracked(y, line, font(c.get("title_size", 172), 1), 10)
        y += 205

    if c.get("plate"):
        plate = Image.open(images_dir / c["plate"]).convert("L")
        w = c.get("plate_width", 1000)
        plate = plate.resize((w, int(w * plate.height / plate.width)),
                             Image.LANCZOS)
        # the plate is a JPEG: clip its grey halo before using it as a mask
        mask = plate.point(lambda v: 0 if v > 205 else 255 if v < 90
                           else int((205 - v) * 255 / 115))
        img.paste(Image.new("RGB", plate.size, ink),
                  ((W - plate.width) // 2, c.get("plate_top", 1240)), mask)

    rule(1700, 160)
    for n, line in enumerate(c.get("foot_lines", [])):
        tracked(1760 + n * 100, line, font(40), 8)
    rule(1940, 160)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, "PNG", optimize=True)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def fetch_image(source: str, href: str, target: Path) -> None:
    """Pull an illustration from beside the page it is printed in."""
    import urllib.parse
    url = urllib.parse.urljoin(source, href)
    if not url.startswith(("http://", "https://")):
        shutil.copy2(url, target)
        return
    req = urllib.request.Request(
        url, headers={"User-Agent": "codicology/pga2epub"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            target.write_bytes(r.read())
    except OSError as e:
        raise SystemExit(f"cannot fetch {url}: {e}") from None


def build(book: Path, out: Path | None, source_file: str | None) -> int:
    cfg = json.loads(book.read_text(encoding="utf-8"))
    src = source_file or cfg["source"]
    text, moji = load_text(src, cfg.get("mojibake"))
    text, log = apply_repairs(text, cfg.get("repairs", []))
    print(f"read {src}")
    print(f"  encoding: {'mojibake, reversed' if moji else 'read as it is'}")
    for line in log:
        print(f"  repaired {line}")

    body = parse(text)
    # a second parse, untouched, is what coverage is measured against
    whole = parse(text)
    for line in strip_site_furniture(body):
        print(f"  dropped {line}")
    moved = absorb_strays(body)
    if moved:
        print(f"  re-attached {moved} stray block(s) to their chapter")

    images = cfg.get("images", {})
    reader = cfg.get("reader", "divs")
    if reader == "divs":
        sections = read_divs(body, images)
    elif reader == "flow":
        sections = read_flow(body, images, cfg.get("split_on", "h2"),
                             cfg.get("flow_starts", ""))
    else:
        raise SystemExit(f"no reader named {reader!r}; inspect the source "
                         "and write one against it")
    chapters = [p for k, p in sections if k == "chapter"]
    if not chapters:
        raise SystemExit(f"reader {reader!r} found no chapters")
    print(f"  {reader}: {len(chapters)} chapters, "
          f"{sum(1 for k, _ in sections if k == 'part')} parts")

    out = out or book.with_suffix(".epub")
    build_dir = out.parent / (out.stem + ".epubdir")
    if build_dir.exists():
        shutil.rmtree(build_dir)

    docs = assemble(cfg, sections, build_dir)
    docs.append(colophon(cfg, log))
    marked = link_notes(docs)
    notes = sum(len(d.note_ids) for d in docs)
    if marked != notes:
        raise SystemExit(f"{marked} note markers but {notes} notes")
    if notes:
        print(f"  {notes} notes, each with a marker and a way back")

    write(build_dir / "mimetype", "application/epub+zip")
    write(build_dir / "META-INF" / "container.xml",
          '<?xml version="1.0" encoding="utf-8"?>\n<container version="1.0" '
          'xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
          '  <rootfiles>\n    <rootfile full-path="OEBPS/content.opf" '
          'media-type="application/oebps-package+xml"/>\n'
          "  </rootfiles>\n</container>\n")
    write(build_dir / "OEBPS" / "styles" / "main.css",
          CSS.read_text(encoding="utf-8"))

    dest = build_dir / "OEBPS" / "images"
    dest.mkdir(parents=True, exist_ok=True)
    local = cfg.get("images_dir")
    for base, spec in images.items():
        target = dest / spec["name"]
        if local:
            shutil.copy2(Path(local).expanduser() / base, target)
        else:
            fetch_image(src, spec.get("src") or base, target)
    if cfg.get("cover"):
        make_cover(cfg, build_dir / "OEBPS" / "images",
                   build_dir / "OEBPS" / "images" / "cover.png")

    lang = cfg.get("language", "en")
    for doc in docs:
        write(build_dir / "OEBPS" / "text" / doc.name,
              XHTML.format(lang=lang, css="../styles/main.css",
                           title=esc(f"{cfg['title']} — {doc.title}"),
                           etype=doc.etype, body=doc.body, cls=doc.cls))
    uid = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, cfg["source"]))
    write_nav(docs, build_dir, cfg)
    write_opf(docs, cfg, build_dir, uid)
    write_ncx(docs, cfg, build_dir, uid)
    zip_epub(build_dir, out)

    fails = verify(out, sections, cfg, whole)
    shutil.rmtree(build_dir)
    for f in fails:
        print(f"  FAIL {f}")
    if fails:
        out.unlink()
        raise SystemExit(f"{len(fails)} check(s) failed; {out.name} not kept")
    print(f"wrote {out} ({out.stat().st_size // 1024} KB), all checks passed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="read a source without building it")
    p.add_argument("source", help="URL or path to the PGA HTML")
    p.add_argument("--json", action="store_true", dest="as_json",
                   help="print a starter book file instead of the report")

    p = sub.add_parser("build", help="build the EPUB a book file describes")
    p.add_argument("book", type=Path, help="the book's JSON file")
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--source-file", default=None,
                   help="read this local copy instead of the recorded URL")

    a = ap.parse_args(argv)
    if a.cmd == "inspect":
        return inspect(a.source, a.as_json)
    return build(a.book, a.out, a.source_file)


if __name__ == "__main__":
    sys.exit(main())
