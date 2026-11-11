"""Pages bound into the files a reader turns through, and taken apart again.

The pipeline works a page at a time, and so does everything that checks it:
page N of the book is page N of the PDF, which is what lets a reading be held
against its ink. Written out that way, one file to a page, a reading system
starts a new screen at every printed page, and a printed page ends where the
paper ran out: of 37,359 page turns in 148 books of one shelf, 18,663 fall
inside a sentence.

So the pages are bound. bind() lays a run of pages end to end as one flow,
the chapter a reader turns through. Where a paragraph runs over the page it
is one paragraph again, with the place the page turned marked inside it.
What stood at the foot of the page it began on (its footnotes, a figure set
at the bottom) and at the head of the next would have fallen in the middle
of the sentence, and is held until the paragraph ends.

Nothing is lost by it. Every page's start is marked, the printed page number
stays where the page begins, and unbind() takes a bound file apart into the
pages it was made of, so that the checks go on reading pages. page_bodies()
does that for a whole book, whichever way it was written.

The marks, all in the flow and none of them shown:

    <span class="pg" id="page-0123"></span>
        page 123 begins here
    <span class="pg pg-joined" id="page-0123"></span>
        the same, inside a paragraph that began on the page before
    <span class="pg pg-joined pg-hyphen" id="page-0123"></span>
        the same, inside a word: the page before ended "con-" and this one
        began "centration", and the hyphen the printer put there is left out
    <div class="pg-held" data-page="0122">…</div>
        what closed page 122 after the paragraph that runs on from it
    <div class="pg-led" data-page="0123">…</div>
        what opened page 123 before the paragraph that runs on into it
"""
from __future__ import annotations

import base64
import html
import json
import os
import re
import zipfile

PAGE_FILE = re.compile(r"(?:.*/)?page_(\d{4})\.xhtml$")
BOUND_FILE = re.compile(r"(?:.*/)?ch_(\d{4})\.xhtml$")

MARK = '<span class="pg" id="page-%04d"></span>'
MARK_JOINED = '<span class="pg pg-joined" id="page-%04d"></span>'
MARK_IN_A_WORD = '<span class="pg pg-joined pg-hyphen" id="page-%04d"></span>'
_MARK = re.compile(r'<span class="pg(?P<joined> pg-joined(?P<hyphen> '
                   r'pg-hyphen)?)?" id="page-(?P<page>\d{4})"\s*'
                   r'(?:/>|>\s*</span>)')
_ASIDE = re.compile(r'<div class="pg-(held|led)" data-page="(\d{4})"\s*>')
# The printed page number, which the build sets at the head of its page.
_FOLIO = re.compile(r'\s*<span\b[^>]*epub:type="pagebreak"[^>]*'
                    r'(?:/>|>\s*</span>)')

_TAG = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9:]*)\b([^<>]*?)(/?)>")
_VOID = {"hr", "br", "img", "meta", "link", "input", "col", "wbr"}
# What a printer sets at the head or the foot of a page because that is
# where there was room, and a sentence runs round.
_FLOATS = {"figure", "table"}
# What begins a line of its own. White space beside one of these says
# nothing; between two things that stand in a line it is a space.
_BLOCK = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "dl",
          "table", "figure", "blockquote", "hr", "pre", "section", "aside"}

# A paragraph that has ended says so. The mark of a note after its stop is
# not part of the sentence ("…transfers.32").
_NOTE_MARKS = re.compile(r"(?:\s*<sup\b[^>]*>(?:(?!</sup>).)*</sup>)+\s*$",
                         re.S)
_ENDS_A_SENTENCE = re.compile(r"[.!?…:;][”’\"')\]]*\s*$")
# …and so does one whose note mark was read as a plain figure after the
# stop ("…or aktivix.6"), which a decimal ("2.9 | million votes") is not.
_NOTE_AFTER_STOP = re.compile(r"(?<!\d)[.!?…][”’\"')\]]*\s*\d{1,3}$")
_CAPTION = re.compile(
    r"\s*(?:(?:Fig(?:ure|\.)|FIG(?:URE|\.)|Table|TABLE|Plate|PLATE|Exhibit|"
    r"EXHIBIT|Chart|CHART|Map|MAP|Box|BOX|Illustration)\s*"
    r"(?:[A-Z]?\d|[IVXLC]+\b|[A-Z]\b)"
    r"|(?:Sources?|SOURCES?|Notes?|NOTES?|Credits?|Photo(?:graph)?s?)\s*:)")
# "Cover design: Hatim Eujayl", "Credit: AP Images/Ng Han Guan": a label
# and after it nothing that begins with a small letter.
_LABEL_AND_NAMES = re.compile(r"[A-Z][\w ]{0,30}:\s+(?:[^a-z\s]\S*\s*)+$")
# A word the printer broke at the foot of the page, and what takes it up.
_BROKEN_WORD = re.compile(r"([A-Za-z]+)-$")
_REST_OF_WORD = re.compile(r"[a-z]+")
# A paragraph that stops without a stop, taken up by a capital or a figure:
# "…except in | Canton on the Pearl River" is one sentence, and so is a line
# of verse followed by the next. Prose is told from the rest by its length.
PROSE_WORDS = 12


def _strip(markup: str) -> str:
    return re.sub(r"<[^>]+>", "", markup)


def blocks(body: str) -> "list[tuple[str, str]]":
    """A body's pieces at its top level, in order: (tag, markup), the tag ""
    for text that stands in no element. White space alone is a piece (a
    space) only between two things that stand in a line, where it is one."""
    out, depth, start, name, pos = [], 0, 0, "", 0
    for m in _TAG.finditer(body):
        closing, tag, void = m.group(1), m.group(2).lower(), \
            bool(m.group(4)) or m.group(2).lower() in _VOID
        if depth == 0:
            if closing:
                continue
            between = body[pos:m.start()]
            if between.strip():
                out.append(("", between))
            elif between and out and out[-1][0] not in _BLOCK \
                    and tag not in _BLOCK:
                out.append(("", " "))
            if void:
                out.append((tag, m.group(0)))
                pos = m.end()
            else:
                depth, name, start = 1, tag, m.start()
        elif not void:
            depth += -1 if closing else 1
            if depth == 0:
                out.append((name, body[start:m.end()]))
                pos = m.end()
    tail = body[start:] if depth else body[pos:]
    if tail.strip():
        out.append((name if depth else "", tail))
    return out


def _feet_like(below: str) -> bool:
    """Whether what follows a rule is the page's notes and not the next
    section of its text: it opens on a note, by its mark or its link."""
    first = re.search(r"<(p|ol|ul|table|figure|blockquote|h[1-6])\b[^>]*>",
                      below)
    if first is None or first.group(1) not in ("p", "ol", "ul"):
        return False
    if 'id="fn-' in below or 'id="note-' in below:
        return True
    return bool(re.match(
        # a note's number, not the start of a year: "1944 was a hard year"
        # after a rule is the next section of the text
        r"<p[^>]*>\s*(?:<sup>\s*)?(?:<a[^>]*>)?\s*(?:\*{1,2}|†|‡|"
        r"\d{1,3}(?!\d)[.)]?)"
        r"\s*(?:</a>)?\s*(?:</sup>)?\s*\S", below[first.start():]))


def _parts(body: str) -> "tuple[str, list, list]":
    """A page as (the page number at its head, its blocks, its feet)."""
    # with the white space after it: before a thing that stands in a line
    # (an equation set as text) it is a space, and is kept as one
    m = re.match(r"(?:%s)*\s*" % _FOLIO.pattern, body)
    head, rest = body[:m.end()], blocks(body[m.end():])
    feet: list = []
    for j in range(len(rest) - 1, -1, -1):
        if rest[j][0] == "hr":
            if _feet_like("".join(h for _, h in rest[j + 1:])):
                rest, feet = rest[:j], rest[j:]
            break
    return head, rest, feet


def _runs_on(a: str, b: str, hopped: bool) -> bool:
    """Whether paragraph b, opening a page, is the rest of paragraph a,
    which closed the page before."""
    if "<br" in a or "<br" in b:
        return False            # lines, not a sentence: an index, verse
    inner = a[a.index(">") + 1:-len("</p>")]
    said = _strip(_NOTE_MARKS.sub("", inner)).rstrip()
    if not said or _ENDS_A_SENTENCE.search(said) \
            or _NOTE_AFTER_STOP.search(said) or _CAPTION.match(_strip(inner)):
        return False
    taken_up = _strip(b[len("<p>"):-len("</p>")]).lstrip()
    if not taken_up or _CAPTION.match(taken_up):
        return False            # "Table 4 Fossil fuel consumption…" opens nothing
    if not re.search(r"[a-z]", taken_up[:80]):
        return False            # a running head the build let through
    words = len(said.split())
    if words < 3 or not re.search(r"[a-z]{2}", said):
        return False            # "B B", a signature mark at the foot
    if words < PROSE_WORDS and _LABEL_AND_NAMES.match(said):
        return False
    if words < PROSE_WORDS and re.search(r"\d[\])]?$", said):
        return False            # "social media 118, 119": an entry, not prose
    if taken_up[0].islower():
        return True
    # a capital, a figure, a quotation mark: only where nothing was hopped
    # over, the paragraph is prose, and it stops on a small word or a comma
    return (not hopped and words >= PROSE_WORDS
            and bool(re.search(r"(?:[a-z]{2,}|,)$", said)))


def book_words(bodies) -> set:
    """Every word a book uses, folded: with the dictionary, what says that
    "con-" and "centration" are one word."""
    from .adjudicate import fold_word
    return {fold_word(w) for b in bodies
            for w in re.findall(r"[A-Za-z][A-Za-z'’]*", _strip(b))}


def _one_word(said: str, b: str, words) -> bool:
    """Whether the page ended inside a word, which the next page finishes:
    the two halves make a word of the dictionary's or of the book's own. A
    compound the printer happened to break at its hyphen (self- | control)
    makes neither, and keeps its hyphen."""
    first, rest = _BROKEN_WORD.search(said), _REST_OF_WORD.match(b)
    if not first or not rest:
        return False
    from .adjudicate import _is_word, fold_word
    whole = fold_word(first.group(1) + rest.group(0))
    return _is_word(whole) or (words is not None and whole in words)


def _join(pending: dict, rest: list) -> "int | None":
    """How many blocks at the head of the next page are hopped over to reach
    the paragraph that takes up `pending`'s, or None where none does."""
    t = 0
    while t < len(rest) and rest[t][0] in _FLOATS:
        t += 1
    if t >= len(rest) or rest[t][0] != "p" or not rest[t][1].startswith("<p>") \
            or not rest[t][1].endswith("</p>"):
        return None
    hopped = bool(t or pending["trailing"])
    return t if _runs_on(pending["a"], rest[t][1], hopped) else None


def _pending(page: int, rest: list, feet: list) -> "tuple[list, dict | None]":
    """Split a page's blocks into what can be written now and the paragraph
    (with what follows it) that the next page may yet take up."""
    k = len(rest)
    while k and rest[k - 1][0] in _FLOATS:
        k -= 1
    if k and rest[k - 1][0] == "p" and rest[k - 1][1].endswith("</p>") \
            and 'class="blank"' not in rest[k - 1][1][:rest[k - 1][1].index(">")]:
        return ([h for _, h in rest[:k - 1]],
                {"page": page, "a": rest[k - 1][1], "after": [],
                 "trailing": [h for _, h in rest[k:]],
                 "feet": [h for _, h in feet]})
    return [h for _, h in rest] + [h for _, h in feet], None


def bind(pages: "list[tuple[int, str]]", words=None) -> str:
    """The pages of one file as one flow. `pages` is (page index, body) in
    reading order; unbind() of the result gives them back. `words` is the
    book's own vocabulary (book_words), which with the dictionary settles
    whether a word broken at the foot of a page is one word."""
    out: list = []
    pending: "dict | None" = None

    def close(p: dict) -> None:
        out.append(p["a"])
        out.extend(p["after"])
        out.extend(p["trailing"])
        out.extend(p["feet"])

    for page, body in pages:
        head, rest, feet = _parts(body)
        t = _join(pending, rest) if pending is not None else None
        after: list = []
        if t is None:
            if pending is not None:
                close(pending)
            # on a line of its own: two pages of loose text must not run
            # together where the one ends and the other begins
            out.append(("\n" if out else "") + MARK % page + head)
        else:
            a, b = pending["a"], rest[t][1]
            said, rest_b = a[:-len("</p>")].rstrip(), b[len("<p>"):].lstrip()
            if _one_word(said, rest_b, words):
                said, glue, mark = said[:-1], "", MARK_IN_A_WORD
            else:
                # a dash set close against its word takes the next close
                # against itself; one with a space before it has one after
                glue = "" if re.search(r"\S[-—–/]$", said) else " "
                mark = MARK_JOINED
            joined = said + glue + mark % page + head.strip() + rest_b
            after = list(pending["after"])
            held = "".join(pending["trailing"] + pending["feet"])
            if held:
                after.append('<div class="pg-held" data-page="%04d">%s</div>'
                             % (pending["page"], held))
            if t:
                after.append('<div class="pg-led" data-page="%04d">%s</div>'
                             % (page, "".join(h for _, h in rest[:t])))
            rest = [("p", joined)] + rest[t + 1:]
        now, pending = _pending(page, rest, feet)
        if after and pending is not None and not now:
            pending["after"] = after    # taken up, and still open
        elif after:
            now[1:1] = after            # it ended on this page: what was
        out.extend(now)                 # held for it follows it
    if pending is not None:
        close(pending)
    return "".join(out)


def _take_asides(flow: str) -> "tuple[str, dict, dict]":
    """The flow without what was held or led, and those by page."""
    held: dict = {}
    led: dict = {}
    out, pos = [], 0
    while True:
        m = _ASIDE.search(flow, pos)
        if not m:
            break
        depth, end = 1, len(flow)
        for t in _TAG.finditer(flow, m.end()):
            if t.group(2).lower() != "div" or t.group(4):
                continue
            depth += -1 if t.group(1) else 1
            if depth == 0:
                end = t.end()
                inner = flow[m.end():t.start()]
                break
        else:
            inner = flow[m.end():]
        (held if m.group(1) == "held" else led).setdefault(
            int(m.group(2)), []).append(inner)
        out.append(flow[pos:m.start()])
        pos = end
    out.append(flow[pos:])
    return "".join(out), held, led


def unbind(flow: str) -> "list[tuple[int, str]]":
    """The pages a bound file was made of: (page index, body), in order."""
    flow, held, led = _take_asides(flow)
    marks = list(_MARK.finditer(flow))
    pages = []
    for k, m in enumerate(marks):
        page = int(m.group("page"))
        end = marks[k + 1].start() if k + 1 < len(marks) else len(flow)
        seg = flow[m.end():end]
        head = re.match(r"(?:%s)*" % _FOLIO.pattern, seg)
        body = seg[:head.end()] + "".join(led.get(page, [])) \
            + ("<p>" if m.group("joined") else "") + seg[head.end():]
        body = body.rstrip()
        if k + 1 < len(marks) and marks[k + 1].group("joined"):
            body += ("-" if marks[k + 1].group("hyphen") else "") + "</p>"
        pages.append((page, body + "".join(held.get(page, []))))
    return pages


def body_of(xhtml: str) -> str:
    """What a content file shows: the inside of its <body>."""
    m = re.match(r".*?<body[^>]*>", xhtml, flags=re.S)
    lo = m.end() if m else 0
    hi = xhtml.find("</body>", lo)
    return xhtml[lo:hi if hi >= 0 else len(xhtml)]


def is_bound(names) -> bool:
    return any(BOUND_FILE.match(n) for n in names)


def page_bodies(epub) -> "dict[int, str]":
    """{page index: the page's body} for a book written either way: a file
    to a page, or pages bound into chapters. `epub` is a path or an open
    ZipFile."""
    z = epub if isinstance(epub, zipfile.ZipFile) else zipfile.ZipFile(epub)
    try:
        out = {}
        for n in z.namelist():
            m = PAGE_FILE.match(n)
            if m:
                # 'replace': one malformed byte in one file must cost that
                # page's words and not the whole reading
                out[int(m.group(1))] = body_of(
                    z.read(n).decode("utf-8", "replace"))
            elif BOUND_FILE.match(n):
                out.update(unbind(body_of(
                    z.read(n).decode("utf-8", "replace"))))
        return out
    finally:
        if z is not epub:
            z.close()


def plan_files(pages: "list[tuple[int, str]]", starts: "set[int]",
               most: int = 60, weight: int = 300_000
               ) -> "list[list[tuple[int, str]]]":
    """The pages divided into the files of the book. A file begins where a
    chapter does (`starts`, by page index), and a chapter that would be a
    file of more than `most` pages or `weight` characters is cut where a
    page ends its paragraph: a reading system lays a whole file out before
    it shows any of it."""
    files: list = []
    pending: "dict | None" = None
    size = 0
    for page, body in pages:
        head, rest, feet = _parts(body)
        runs_on = pending is not None and _join(pending, rest) is not None
        if not files or (not runs_on and (page in starts
                                          or len(files[-1]) >= most
                                          or size >= weight)):
            files.append([])
            size = 0
        files[-1].append((page, body))
        size += len(body)
        if runs_on:
            # as bind() will have it: the paragraph is still open, or not,
            # by how the part of it on this page ends
            t = _join(pending, rest)
            rest = [("p", pending["a"][:-len("</p>")] + " "
                     + rest[t][1][len("<p>"):])] + rest[t + 1:]
        _, pending = _pending(page, rest, feet)
    return files


def relink(markup: str, file_of: "dict[int, str]") -> str:
    """Links written to a page's own file, pointed at the file the page is
    bound into. A link to a page and nothing on it goes to where the page
    begins, which for the page a file opens with is the file."""
    def to(m: "re.Match") -> str:
        page = int(m.group(2))
        name = file_of.get(page)
        if name is None:
            return m.group(0)
        opens = BOUND_FILE.match(name) and int(BOUND_FILE.match(name).group(1)) == page
        return '%s="%s%s"' % (m.group(1), name, m.group(3)
                              or ("" if opens else "#page-%04d" % page))
    return re.sub(r'\b(href|src)="page_(\d{4})\.xhtml(#[^"]*)?"', to, markup)


# ── a whole book ─────────────────────────────────────────────────────────────

class CannotBind(Exception):
    """Why a book is left as it was written."""


MOST = 60               # pages to a file, where no chapter ends sooner
_BLOCKISH = "|".join(sorted(_BLOCK | {"li", "thead", "tbody", "tr", "td",
                                       "th", "figcaption", "dt", "dd"}))
_ITEM = re.compile(r"(\s*)<item\b([^>]*?)/>")
_ITEMREF = re.compile(r"(\s*)<itemref\b([^>]*?)/>")
_ATTR = re.compile(r'([\w:.-]+)="([^"]*)"')
_NOT_A_CHAPTER = re.compile(r"Page \d+$")


def _norm(body: str) -> str:
    """A page's markup less the white space that says nothing: between its
    blocks, and at the edges of them."""
    s = re.sub(r"\s+", " ", body)
    return re.sub(r" ?(</?(?:%s)\b[^>]*>) ?" % _BLOCKISH, r"\1", s).strip()


def same_pages(a, b) -> bool:
    """Whether two readings of a book's pages are the same pages."""
    return [(p, _norm(x)) for p, x in a] == [(p, _norm(x)) for p, x in b]


def _lines(flow: str) -> str:
    """A flow with each of its blocks on a line of its own, for whoever
    opens the file. Only beside a block: a new line between two things
    that stand in a line would be a space between them."""
    out, last = [], None
    for tag, piece in blocks(flow):
        if last is not None and (last in _BLOCK or tag in _BLOCK):
            out.append("\n")
        out.append(piece.strip() if tag in _BLOCK else piece)
        last = tag
    return "".join(out).strip()


def _span(xhtml: str) -> "tuple[int, int]":
    m = re.match(r".*?<body[^>]*>", xhtml, flags=re.S)
    lo = m.end() if m else 0
    hi = xhtml.find("</body>", lo)
    return lo, (hi if hi >= 0 else len(xhtml))


def toc_starts(nav: str) -> "dict[int, str]":
    """{page index: title} for the entries of a book's contents that begin
    a chapter: its top level, and the entries under a heading that has no
    page of its own (a Part). A contents that is only a list of the pages
    names none."""
    import xml.etree.ElementTree as ET
    xh, ep = "{http://www.w3.org/1999/xhtml}", "{http://www.idpf.org/2007/ops}"
    try:
        root = ET.fromstring(nav.encode("utf-8"))
    except ET.ParseError:
        return {}
    starts: dict = {}

    def walk(ol) -> None:
        for li in ol.findall(xh + "li"):
            a = li.find(xh + "a")
            m = PAGE_FILE.match((a.get("href") or "").split("#")[0]) \
                if a is not None else None
            if m:
                title = " ".join("".join(a.itertext()).split())
                if not _NOT_A_CHAPTER.match(title):
                    starts.setdefault(int(m.group(1)), title)
            else:
                for sub in li.findall(xh + "ol"):
                    walk(sub)

    for nav_el in root.iter(xh + "nav"):
        if nav_el.get(ep + "type") == "toc":
            for ol in nav_el.findall(xh + "ol"):
                walk(ol)
    return starts


class _Package:
    """What of a written book the binding has to read and rewrite."""

    def __init__(self, z: zipfile.ZipFile):
        try:
            box = z.read("META-INF/container.xml").decode("utf-8", "replace")
            self.opf_name = re.search(r'full-path="([^"]+)"', box).group(1)
            self.opf = z.read(self.opf_name).decode("utf-8")
        except (KeyError, AttributeError, UnicodeDecodeError) as exc:
            raise CannotBind(f"its package file cannot be read ({exc})")
        self.base = self.opf_name.rpartition("/")[0]
        self.base += "/" if self.base else ""
        self.items = []         # (match, attributes) in the order written
        for m in _ITEM.finditer(self.opf):
            self.items.append((m, dict(_ATTR.findall(m.group(2)))))
        self.refs = [(m, dict(_ATTR.findall(m.group(2))))
                     for m in _ITEMREF.finditer(self.opf)]
        self.by_id = {a.get("id"): a for _, a in self.items}

    def content(self, pattern) -> list:
        """(number, member name, itemref match) for the files of the spine
        whose names match, in reading order."""
        out = []
        for m, a in self.refs:
            href = self.by_id.get(a.get("idref"), {}).get("href", "")
            hit = pattern.match(href)
            if hit:
                out.append((int(hit.group(1)), self.base + href, m))
        return out

    def others(self) -> list:
        """The members that may hold a link to a page and are not pages: the
        contents in both its forms."""
        return [self.base + a["href"] for _, a in self.items
                if a.get("href") and ("nav" in a.get("properties", "").split()
                                      or a["href"].endswith(".ncx"))]


def _rewrite(zin: zipfile.ZipFile, out: str, changed: dict, gone: set,
             added: dict) -> None:
    """The book written again: `changed` members under their own records,
    `gone` left out, and `added` ({name: (after which member, bytes)})
    put where the member they replace stood."""
    try:
        with zipfile.ZipFile(out + ".tmp", "w") as zout:
            for info in zin.infolist():
                # Each entry goes back under its own record: its date, its
                # mode, its compression (mimetype first and stored, as the
                # EPUB contract has it), so that what changed can be read
                # off the files.
                for name, (where, data) in added.items():
                    if where == info.filename:
                        new = zipfile.ZipInfo(name, date_time=info.date_time)
                        new.compress_type = zipfile.ZIP_DEFLATED
                        new.external_attr = info.external_attr
                        zout.writestr(new, data)
                if info.filename in gone:
                    continue
                zout.writestr(info, changed.get(info.filename,
                                                zin.read(info.filename)))
        os.replace(out + ".tmp", out)
    finally:
        if os.path.exists(out + ".tmp"):
            os.remove(out + ".tmp")     # nothing half-written is left behind


def _text(z: zipfile.ZipFile, name: str) -> str:
    """A member the binding is about to rewrite, read strictly: one it
    cannot read exactly is one it must not write."""
    try:
        return z.read(name).decode("utf-8")
    except KeyError:
        raise CannotBind(f"{name} is named in its package and is not in it")
    except UnicodeDecodeError as exc:
        raise CannotBind(f"{name} is not text as it should be ({exc})")


# ── where the reader had got to ──────────────────────────────────────────────
#
# A reading system remembers a place as a file of the spine and steps into
# it. Bound, the files are other files; the place is carried by the page it
# was on, to where that page now begins, which is less than a page from
# where it was. Calibre keeps one in the book itself (and others in its
# library, which carried() serves).

BOOKMARKS = "META-INF/calibre_bookmarks.txt"
_CFI = re.compile(r"epubcfi\(/(\d+)((?:/[^)]*)?)\)")
_LEGACY = "*|!|?|*"


def _spine_names(z: zipfile.ZipFile) -> list:
    pkg = _Package(z)
    return [pkg.base + pkg.by_id.get(a.get("idref"), {}).get("href", "")
            for _, a in pkg.refs]


def _page_marks(doc: str) -> list:
    """(page, (which child of the body, which child of that)) for each
    place a page begins in a bound file, in order; the second is 0 for a
    page that begins between two blocks."""
    import xml.etree.ElementTree as ET
    out = []
    try:
        body = ET.fromstring(doc.encode("utf-8")).find(
            "{http://www.w3.org/1999/xhtml}body")
    except ET.ParseError:
        return out
    for k, el in enumerate(body if body is not None else (), 1):
        m = re.fullmatch(r"page-(\d{4})", el.get("id") or "")
        if m:
            out.append((int(m.group(1)), (k, 0)))
        for j, sub in enumerate(el, 1):
            m = re.fullmatch(r"page-(\d{4})", sub.get("id") or "")
            if m:
                out.append((int(m.group(1)), (k, j)))
    return out


def page_at(z: zipfile.ZipFile, cfi: str, spine=None) -> "int | None":
    """The page a remembered place is on, in a book written either way."""
    m = _CFI.match(cfi or "")
    spine = spine if spine is not None else _spine_names(z)
    if not m or int(m.group(1)) % 2 or not 0 < int(m.group(1)) // 2 <= len(spine):
        return None
    name = spine[int(m.group(1)) // 2 - 1]
    if PAGE_FILE.match(name):
        return int(PAGE_FILE.match(name).group(1))
    if not BOUND_FILE.match(name) or name not in z.namelist():
        return None
    steps = [int(x) for x in re.findall(
        r"/(\d+)", re.sub(r"\[[^\]]*\]", "", m.group(2)).split(":")[0])]
    at = (steps[2] // 2 if len(steps) > 2 else 0,
          steps[3] // 2 if len(steps) > 3 else 0)
    marks = _page_marks(z.read(name).decode("utf-8", "replace"))
    before = [page for page, where in marks if where <= at]
    return before[-1] if before else (marks[0][0] if marks else None)


def page_places(z: zipfile.ZipFile) -> dict:
    """{page: the place where it begins, as a reading system writes one}."""
    out = {}
    for i, name in enumerate(_spine_names(z)):
        step = 2 * (i + 1)
        if PAGE_FILE.match(name):
            out[int(PAGE_FILE.match(name).group(1))] = "epubcfi(/%d/2/4)" % step
        elif BOUND_FILE.match(name) and name in z.namelist():
            for page, (k, j) in _page_marks(z.read(name).decode("utf-8",
                                                                "replace")):
                out[page] = "epubcfi(/%d/2/4/%d%s)" % (
                    step, 2 * k, "/%d" % (2 * j) if j else "")
    return out


def _where(z: zipfile.ZipFile, cfi: str, spine: list):
    """What a remembered place is a place in: ("page", its index), or for a
    file that is no page (the contents) ("file", its name, the steps into
    it), which binding does not change. None for one that cannot be told."""
    m = _CFI.match(cfi or "")
    if not m or int(m.group(1)) % 2 or not 0 < int(m.group(1)) // 2 <= len(spine):
        return None
    name = spine[int(m.group(1)) // 2 - 1]
    if PAGE_FILE.match(name) or BOUND_FILE.match(name):
        page = page_at(z, cfi, spine)
        return None if page is None else ("page", page)
    return ("file", name, m.group(2))


def _there(z: zipfile.ZipFile, wheres: list) -> list:
    """The same places in the book as it is now written."""
    now, spine = page_places(z), _spine_names(z)
    return [None if w is None else now.get(w[1]) if w[0] == "page"
            else "epubcfi(/%d%s)" % (2 * (spine.index(w[1]) + 1), w[2])
            if w[1] in spine else None for w in wheres]


def carried(before: str, after: str, places: list) -> list:
    """Places remembered in the book at `before`, as places in the same
    book written the other way at `after`: the start of the page each was
    on. None for one that cannot be told."""
    with zipfile.ZipFile(before) as z:
        spine = _spine_names(z)
        wheres = [_where(z, c, spine) for c in places]
    with zipfile.ZipFile(after) as z:
        return _there(z, wheres)


def _remembered(raw: str) -> "tuple[list, object]":
    """The places in a book's own bookmarks file, and what puts new ones
    back: (places, put), where put(new places) is the file again. Calibre
    has written the file two ways."""
    if raw.startswith("encoding=json+base64:"):
        rows = json.loads(base64.b64decode(raw.partition(":")[2]))
        if not isinstance(rows, list):
            raise ValueError("not a list of places")
        mine = [r for r in rows if isinstance(r, dict)
                and r.get("pos_type") == "epubcfi"
                and isinstance(r.get("pos"), str)]

        def put(new: list) -> str:
            # a place that cannot be told is taken out: left in, it would
            # be a place in some other part of the book. Every other row
            # is Calibre's and goes back as it came.
            lost = {id(r) for r, cfi in zip(mine, new) if cfi is None}
            for r, cfi in zip(mine, new):
                if cfi is not None:
                    r["pos"] = cfi
            keep = [r for r in rows if id(r) not in lost]
            return "encoding=json+base64:\n" + base64.encodebytes(
                json.dumps(keep).encode("utf-8")).decode("ascii")
        return [r["pos"] for r in mine], put
    lines = raw.splitlines()
    mine = [(n, ln.split(_LEGACY)) for n, ln in enumerate(lines)
            if ln.count(_LEGACY) == 2 and ln.split(_LEGACY)[1].isdigit()]

    def put_legacy(new: list) -> str:
        gone = set()
        for (n, (title, _, _)), cfi in zip(mine, new):
            m = _CFI.match(cfi or "")
            if not m:
                gone.add(n)
                continue
            lines[n] = _LEGACY.join([title, str(int(m.group(1)) // 2 - 1),
                                     m.group(2)])
        return "\n".join(ln for n, ln in enumerate(lines) if n not in gone) \
            + ("\n" if raw.endswith("\n") else "")
    return ["epubcfi(/%d%s)" % (2 * (int(spine) + 1), steps)
            for _, (_, spine, steps) in mine], put_legacy


def _carry_bookmarks(z: zipfile.ZipFile):
    """What the book remembers of its reader, read before it is rewritten:
    (what each place is a place in, put) or None."""
    if BOOKMARKS not in z.namelist():
        return None
    try:
        places, put = _remembered(z.read(BOOKMARKS).decode("utf-8"))
        spine = _spine_names(z)
        return [_where(z, c, spine) for c in places], put
    except Exception:           # not a file as Calibre writes one: it is
        return None             # left as it is, and the book is still bound


def _put_bookmarks(out: str, remembered) -> int:
    """The places written back into the rewritten book; how many."""
    if not remembered:
        return 0
    wheres, put = remembered
    try:
        with zipfile.ZipFile(out) as z:
            new = _there(z, wheres)
            _rewrite(z, out, {BOOKMARKS: put(new).encode("utf-8")}, set(), {})
    except Exception:           # the book is written; its bookmarks stay
        return 0                # as they were
    return sum(c is not None for c in new)


def bind_epub(path: str, out: "str | None" = None, most: int = MOST) -> dict:
    """A book written a file to a page, written again with its pages bound
    into chapters. Nothing but the files changes: every page's text, its
    printed number and what links to it are where they were, and
    page_bodies() of the result is page_bodies() of what it was given.

    Raises CannotBind, and writes nothing, for a book it will not answer
    for: one whose pages would not come apart again exactly, or that has
    one name on two pages of a file."""
    out = out or path
    with zipfile.ZipFile(path) as z:
        pkg = _Package(z)
        pages = pkg.content(PAGE_FILE)
        if not pages:
            raise CannotBind("it is already bound" if pkg.content(BOUND_FILE)
                             else "it has no page files")
        refs = [m for m, _ in pkg.refs]
        at = [refs.index(m) for _, _, m in pages]
        if at != list(range(at[0], at[0] + len(at))):
            raise CannotBind("its pages are not one run of its spine")
        if any("/" in name[len(pkg.base):] for _, name, _ in pages):
            raise CannotBind("its pages are not beside its package file")
        text = {name: _text(z, name) for _, name, _ in pages}
        others = {n: _text(z, n) for n in pkg.others() if n in z.namelist()}
        starts: dict = {}
        for n, t in others.items():
            if n.endswith(".xhtml"):
                starts.update(toc_starts(t))
        name_of = {page: name for page, name, _ in pages}
        bodies = [(page, body_of(text[name])) for page, name, _ in pages]
        words = book_words(b for _, b in bodies)
        files = plan_files(bodies, set(starts), most)

        file_of, flows, stats = {}, [], {"pages": len(pages), "files": len(files),
                                         "joined": 0, "held": 0, "led": 0}
        for group in files:
            flow = bind(group, words)
            if not same_pages(unbind(flow), group):
                raise CannotBind(f"pages {group[0][0]}–{group[-1][0]} would "
                                 f"not come apart again as they went in")
            ids = re.findall(r'\bid="([^"]*)"', flow)
            twice = sorted({i for i in ids if ids.count(i) > 1})[:3] \
                if len(ids) != len(set(ids)) else []
            if twice:
                raise CannotBind(f"pages {group[0][0]}–{group[-1][0]} have "
                                 f"one name on two things ({', '.join(twice)})")
            stats["joined"] += flow.count(" pg-joined")
            stats["words"] = stats.get("words", 0) + flow.count(" pg-hyphen")
            stats["held"] += flow.count('class="pg-held"')
            stats["led"] += flow.count('class="pg-led"')
            first = group[0][0]
            for page, _ in group:
                file_of[page] = "ch_%04d.xhtml" % first
            flows.append((first, flow))

        import xml.etree.ElementTree as ET
        added, changed = {}, {}
        for first, flow in flows:
            src = text[name_of[first]]
            lo, hi = _span(src)
            title = html.escape(starts.get(first, "Page %d" % (first + 1)),
                                quote=False)
            wrap = re.sub(r"<title>[^<]*</title>", lambda _: f"<title>{title}</title>",
                          src[:lo], count=1)
            doc = (wrap + "\n" + _lines(relink(flow, file_of)) + "\n"
                   + src[hi:])
            try:
                ET.fromstring(doc.encode("utf-8"))
            except ET.ParseError as exc:
                raise CannotBind(f"pages from {first} on do not make a "
                                 f"well-formed file ({exc})")
            added[pkg.base + "ch_%04d.xhtml" % first] = (
                name_of[first], doc.encode("utf-8"))
        for n, t in others.items():
            changed[n] = relink(t, file_of).encode("utf-8")

        # the package: one item and one place in the spine for each file,
        # where its first page's were
        opens = {first for first, _ in flows}
        said: dict = {}         # what the package says of a file's pages
        for m, a in pkg.items:
            hit = PAGE_FILE.match(a.get("href", ""))
            if hit and a.get("properties"):
                said.setdefault(file_of.get(int(hit.group(1))), set()).update(
                    a["properties"].split())
        cuts = []
        for m, a in pkg.items:
            hit = PAGE_FILE.match(a.get("href", ""))
            if not hit or pkg.base + a["href"] not in text:
                continue
            page = int(hit.group(1))
            if page not in opens:
                cuts.append((m.start(), m.end(), ""))
                continue
            # the first page's item, under the file's name; whatever any of
            # the file's pages was said to hold (mathematics, a script),
            # the file holds
            item = re.sub(r'\bhref="[^"]*"', 'href="ch_%04d.xhtml"' % page,
                          m.group(0), count=1)
            props = " ".join(sorted(said.get("ch_%04d.xhtml" % page, ())))
            if props:
                item = re.sub(r'\s+properties="[^"]*"', "", item)
                item = item[:-2].rstrip() + ' properties="%s"/>' % props
            cuts.append((m.start(), m.end(), item))
        for page, _, m in pages:
            if page not in opens:
                cuts.append((m.start(), m.end(), ""))
        opf = pkg.opf
        for lo, hi, new in sorted(cuts, reverse=True):
            opf = opf[:lo] + new + opf[hi:]
        changed[pkg.opf_name] = opf.encode("utf-8")
        remembered = _carry_bookmarks(z)
        _rewrite(z, out, changed, set(text), added)
    stats["places"] = _put_bookmarks(out, remembered)
    return stats


def unbind_epub(path: str, out: "str | None" = None) -> dict:
    """A bound book written again a file to a page, as the build first
    wrote it."""
    out = out or path
    with zipfile.ZipFile(path) as z:
        pkg = _Package(z)
        bound = pkg.content(BOUND_FILE)
        if not bound:
            raise CannotBind("it is not bound")
        text = {name: _text(z, name) for _, name, _ in bound}
        groups = {name: unbind(body_of(text[name])) for name in text}
        # A bound file: the page it opens with. A name in a bound file:
        # the page it is on. By file, because a name need only be the one
        # of its kind in its file, and a link names the file it means.
        home: dict = {}
        for _, name, _ in bound:
            short = name[len(pkg.base):]
            for k, (page, body) in enumerate(groups[name]):
                if k == 0:
                    home[short] = page
                home[short, "page-%04d" % page] = page
                for i in re.findall(r'\bid="([^"]*)"', body):
                    home.setdefault((short, i), page)

        def back(markup: str) -> str:
            def to(m: "re.Match") -> str:
                frag = (m.group(3) or "")[1:]
                # a name that is on no page of the file led nowhere before,
                # and goes on leading to the file
                page = home.get((m.group(2), frag), home.get(m.group(2))) \
                    if frag else home.get(m.group(2))
                if page is None:
                    return m.group(0)
                keep = "" if not frag or frag == "page-%04d" % page \
                    else "#" + frag
                return '%s="page_%04d.xhtml%s"' % (m.group(1), page, keep)
            return re.sub(r'\b(href|src)="(ch_\d{4}\.xhtml)(#[^"]*)?"', to,
                          markup)

        added, changed, n = {}, {}, 0
        new_items, new_refs = {}, {}
        ref_of = {name: m for _, name, m in bound}
        item_of = {pkg.base + a.get("href", ""): (m, a) for m, a in pkg.items}
        for _, name, _ in bound:
            src = text[name]
            lo, hi = _span(src)
            m_item, a = item_of[name]
            items, refs = [], []
            for k, (page, body) in enumerate(groups[name]):
                wrap = re.sub(r"<title>[^<]*</title>",
                              lambda _: "<title>Page %d</title>" % (page + 1),
                              src[:lo], count=1)
                doc = wrap + "\n" + _lines(back(body)) + "\n" + src[hi:]
                added[pkg.base + "page_%04d.xhtml" % page] = (
                    name, doc.encode("utf-8"))
                ident = a["id"] if k == 0 else "page_%04d" % page
                item = re.sub(r'\bhref="[^"]*"', 'href="page_%04d.xhtml"' % page,
                              m_item.group(0), count=1)
                items.append(re.sub(r'\bid="[^"]*"', 'id="%s"' % ident, item,
                                    count=1))
                refs.append('%s<itemref idref="%s"/>'
                            % (ref_of[name].group(1), ident))
                n += 1
            new_items[name], new_refs[name] = "".join(items), "".join(refs)
        cuts = [(item_of[name][0].start(), item_of[name][0].end(),
                 new_items[name]) for name in text]
        cuts += [(ref_of[name].start(), ref_of[name].end(), new_refs[name])
                 for name in text]
        opf = pkg.opf
        for lo, hi, new in sorted(cuts, reverse=True):
            opf = opf[:lo] + new + opf[hi:]
        changed[pkg.opf_name] = opf.encode("utf-8")
        for name in pkg.others():
            if name in z.namelist():
                changed[name] = back(_text(z, name)).encode("utf-8")
        remembered = _carry_bookmarks(z)
        _rewrite(z, out, changed, set(text), added)
    _put_bookmarks(out, remembered)
    return {"pages": n, "files": len(text)}


def rewrite_pages(path: str, out: str, change) -> int:
    """The book written again with some of its pages changed, whichever
    way it is laid out. `change(page index, body)` returns the page's new
    body, or None to leave it; the number of pages changed is returned."""
    n = 0
    with zipfile.ZipFile(path) as z:
        changed = {}
        words = None
        for name in z.namelist():
            page, bound = PAGE_FILE.match(name), BOUND_FILE.match(name)
            if not (page or bound):
                continue
            src = z.read(name).decode("utf-8")
            lo, hi = _span(src)
            if page:
                new = change(int(page.group(1)), src[lo:hi])
                if new is None or new == src[lo:hi]:
                    continue
                n += 1
            else:
                was = unbind(src[lo:hi])
                now = [(p, b if got is None else got) for p, b in was
                       for got in [change(p, b)]]
                if now == was:
                    continue
                n += sum(a != b for a, b in zip(was, now))
                if words is None:
                    words = book_words(page_bodies(z).values())
                new = "\n" + _lines(bind(now, words)) + "\n"
            changed[name] = (src[:lo] + new + src[hi:]).encode("utf-8")
        _rewrite(z, out, changed, set(), {})
    return n
