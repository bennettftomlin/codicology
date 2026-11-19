"""What a page's reading is made of. Today one PageItem per block in reading
order, as the reader and the cache hold it; and the typed Page and Block the
assembly half is moving onto.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from PIL import Image
from typing import NamedTuple


class PageItem(NamedTuple):
    """One element of a page in reading order: either markup or an extracted figure."""

    html: str = ""
    figure: Image.Image | None = None
    is_caption: bool = False
    # A running head or foot. Kept rather than dropped, because the folio audit
    # needs exactly this text and the recognition here — full resolution, full
    # page context — is the best reading the pipeline will ever have of it. It
    # never reaches the book's body; every consumer of page items filters it.
    is_furniture: bool = False
    # Where the block sits on its page, as fractions of the page's own size
    # (x0, y0, x1, y1). Fractions, not pixels, so the record survives any
    # raster scale a consumer works at. None on entries cached before
    # geometry was kept — those can still build an EPUB, just not a
    # positioned text layer.
    box: tuple | None = None
    # What the model reported about its own certainty for this block. Kept
    # because it costs nothing to keep and cannot be recovered later; whether
    # it is worth anything as a guard is an empirical question, and one that
    # cannot even be asked while the number is being discarded.
    conf: float | None = None
    # The layout model's own name for the block — Footnote, Formula,
    # SectionHeader, TableOfContents, Handwriting, Text… — kept verbatim
    # because every one of those is an assertion some heuristic here
    # re-derives from shape. None on entries cached before labels were
    # kept, and every consumer must treat None as "label unknown",
    # never as "label Text".
    label: "str | None" = None
    # The page itself, standing where its words would have: what a page is
    # given when what was read from it is not fit to print. Never cached;
    # it is decided each time the page is assembled.
    whole_page: bool = False


# ── a page as the assembly half receives it ──────────────────────────────────
#
# The reading half has always known, of every block, the layout's name for
# it, where it stands and how sure the recogniser was, and then joined each
# page into one string of markup for the assembly half, which re-derived by
# pattern what that string no longer said: which paragraphs are footnotes,
# where a running head stood, which table is which. A Page is that page
# before it is joined: its blocks in reading order, each as the markup it
# contributes and what the reading knew of it, and the furniture set aside
# from it. Its rendering is the string the assembly half has always had.
#
# A block's kind is the label in the words the assembly half will ask in;
# the label itself is kept verbatim beside it. "rule" is the line the build
# itself draws above a page's footnotes where the layout named them and
# the recogniser drew none.
KIND_OF_LABEL = {
    "Footnote": "footnote", "SectionHeader": "heading", "PageHeader": "furniture",
    "PageFooter": "furniture", "Table": "table", "Form": "form",
    "TableOfContents": "toc", "Caption": "caption", "ListItem": "list-item",
    "Equation": "equation", "Formula": "equation", "Code": "code",
    "Handwriting": "handwriting", "Picture": "figure", "Figure": "figure",
    "Diagram": "figure", "Text": "text",
}


def kind_of(item: PageItem) -> str:
    """The kind of block a reading's item makes: a figure where it holds one,
    furniture where it was set aside as such, else what its label says, and
    "text" where it has no label the build knows."""
    if item.figure is not None or item.whole_page:
        return "figure"
    if item.is_furniture:
        return "furniture"
    return KIND_OF_LABEL.get(item.label or "", "text")


@dataclass
class Block:
    """One piece of a page's body: the markup it contributes, and what the
    reading knew of it. A figure's block holds the figure with its caption,
    as one <figure>."""
    kind: str
    html: str
    label: "str | None" = None
    box: "tuple | None" = None
    conf: "float | None" = None
    caption: "str | None" = None        # a figure's caption, as read

    @classmethod
    def of(cls, item: PageItem, html: "str | None" = None, **more) -> "Block":
        return cls(kind=kind_of(item), html=item.html if html is None else html,
                   label=item.label, box=item.box, conf=item.conf, **more)


@dataclass
class Page:
    """A page as the assembly half receives it: its blocks in reading order
    and the furniture set aside from it."""
    index: int
    path: str
    blocks: "list[Block]" = field(default_factory=list)
    furniture: "list[Block]" = field(default_factory=list)
    furniture_known: bool = True        # False: cached before heads were kept

    def render(self) -> str:
        """The page as one string of markup, as the assembly half reads it."""
        return "".join(b.html for b in self.blocks)

    def cut(self, start: int, end: int) -> None:
        """Take the span [start, end) of the rendered page out of its blocks:
        a block wholly inside it goes, with its label; a block it reaches
        into keeps the rest of its markup. The page then renders as the
        string with that span cut out, and no block's label passes to
        another's text."""
        kept, at = [], 0
        for b in self.blocks:
            s, e = at, at + len(b.html)
            at = e
            if start <= s and e <= end and s < e:
                continue
            if s < end and start < e:
                b.html = b.html[:max(start - s, 0)] + b.html[max(end - s, 0):]
            kept.append(b)
        self.blocks[:] = kept


# ── keeping a page true to the string a pass rewrote ─────────────────────────
#
# The assembly half is some thirty passes over each page's string. Until each
# is a pass over blocks, a page's blocks are kept true to its string by
# carrying them through every edit: where a pass has changed a page, its old
# rendering is compared with the new string, every boundary between two
# blocks is carried to its place in the new one, and the new string is cut
# there. A block keeps its kind, label and box; its markup becomes what the
# pass made of it. A change that straddles a boundary (a pass that joins two
# paragraphs) goes to the earlier block, and is counted, so it can be seen
# how often the carrying is a judgement and not a fact.

# Compared token by token (a tag or a word, each with the spaces after it):
# a page is some thousands of characters, and a pass that sets every quote on
# it changes most of them, which compared character by character cost a long
# book more time than everything else in its assembly together. A boundary
# between blocks stands between tags, so it is always between two tokens.
_TOKEN = None


def _tokens(text: str) -> "tuple[list[str], list[int]]":
    """The tokens of `text` and where each starts, with its length last."""
    import re
    global _TOKEN
    if _TOKEN is None:
        _TOKEN = re.compile(r"<[^<>]*>\s*|[^<\s]+\s*|\s+|<")
    toks, starts = [], []
    for m in _TOKEN.finditer(text):
        toks.append(m.group())
        starts.append(m.start())
    starts.append(len(text))
    return toks, starts


def _key(token: str) -> str:
    """What a token is compared by: a tag by its name, a word as it is."""
    if token.startswith("<") and len(token) > 1:
        import re
        m = re.match(r"<(/?)([A-Za-z][\w:.-]*)", token)
        if m:
            return "<" + m.group(1) + m.group(2).lower() + (" /" if token.rstrip().endswith("/>") else "")
    return token


def _carry(old: str, new: str, cuts: "list[int]") -> "tuple[list[int], int]":
    """Where each cut in `old` lands in `new`, and how many fell inside a
    change rather than between unchanged text."""
    import bisect
    import difflib
    if old == new:
        return list(cuts), 0
    # The unchanged ends are taken back to where a tag opens, so the middle
    # is compared as whole tokens: compared from inside a shared "<p>", a
    # page whose first block was cut gave that block's label to the
    # paragraph after it, which kept nothing. Which end is measured first
    # matters: a block cut from the top wants the shared end measured whole
    # (or "<blockquote><p>…" loses its label), a block cut from the foot the
    # shared start. Both are tried, and the one leaving less to compare kept.
    def common_head(limit):
        h = 0
        while h < limit and old[h] == new[h]:
            h += 1
        return max(old.rfind("<", 0, h + 1), 0)

    def common_tail(limit):
        t = 0
        while t < limit and old[-1 - t] == new[-1 - t]:
            t += 1
        at = old.find("<", len(old) - t)
        return len(old) - at if at >= 0 else 0

    shortest = min(len(old), len(new))
    h1 = common_head(shortest)
    t1 = min(common_tail(shortest - h1), len(old) - h1)
    t2 = common_tail(shortest)
    h2 = common_head(shortest - t2)
    whole = set(_balanced(old))

    def rank(h, t):
        # more left unchanged first; then a middle that is whole elements
        return (h + t, h in whole and (len(old) - t) in whole)
    head, tail = (h1, t1) if rank(h1, t1) >= rank(h2, t2) else (h2, t2)
    if head > len(old) - tail or head > len(new) - tail:
        head, tail = h1, 0
    mid_old, mid_new = old[head:len(old) - tail], new[head:len(new) - tail]
    t_old, s_old = _tokens(mid_old)
    t_new, s_new = _tokens(mid_new)
    # Tags are matched by name, their attributes aside: a block cut while
    # the next one's opening tag gained a class is a deletion and a retag,
    # not one change across the boundary.
    k_old, k_new = [_key(t) for t in t_old], [_key(t) for t in t_new]
    ops = difflib.SequenceMatcher(None, k_old, k_new, autojunk=False).get_opcodes()
    out, straddled = [], 0
    for c in cuts:
        if c <= head:
            out.append(c)
        elif c >= len(old) - tail:
            out.append(c + len(new) - len(old))
        else:
            m = c - head
            t = bisect.bisect_right(s_old, m) - 1      # the token m stands in
            inner = m - s_old[t]
            for tag, i1, i2, j1, j2 in ops:
                if (i1 <= t < i2) if inner else (i1 <= t <= i2):
                    if tag == "equal" and t < i2:
                        out.append(head + s_new[j1 + (t - i1)] + inner)
                    elif t == i1 and inner == 0:
                        out.append(head + s_new[j1])
                    elif t == i2 and inner == 0:
                        out.append(head + s_new[j2])
                    else:                       # inside a change: to the earlier block
                        out.append(head + s_new[j2])
                        straddled += 1
                    break
            else:
                out.append(head + len(mid_new))
    for k in range(1, len(out)):                # never backwards
        out[k] = max(out[k], out[k - 1])
    return out, straddled


_TAG = None
_VOID = {"br", "hr", "img", "meta", "link", "input", "col", "area", "base",
         "wbr", "source", "track", "embed", "param"}


def _balanced(markup: str) -> "list[int]":
    """The positions in `markup` that stand between whole elements: where
    every element opened before has been closed."""
    import re
    global _TAG
    if _TAG is None:
        _TAG = re.compile(r"<(/?)([A-Za-z][\w:.-]*)\b[^>]*?(/?)>")
    depth, out, at = 0, [], 0
    for m in _TAG.finditer(markup):
        if depth == 0:
            out.extend(range(at, m.start() + 1))
        closing, name, selfclosed = m.group(1), m.group(2).lower(), m.group(3)
        if closing:
            depth = max(0, depth - 1)
        elif not selfclosed and name not in _VOID:
            depth += 1
        at = m.end()
    if depth == 0:
        out.extend(range(at, len(markup) + 1))
    return out


def sync_page(page: "Page", body: str) -> int:
    """Carry a page's blocks to the string a pass made of it (see above).
    Returns how many boundaries fell inside a change."""
    old = page.render()
    if old == body:
        return 0
    cuts, at = [], 0
    for b in page.blocks[:-1]:
        at += len(b.html)
        cuts.append(at)
    if not page.blocks:
        page.blocks.append(Block(kind="text", html=body))
        return 0
    new_cuts, straddled = _carry(old, body, cuts)
    # a cut inside an element moves on to where it closes, so every block
    # stays whole markup: two paragraphs a pass joined go to the first block
    import bisect
    whole = _balanced(body)
    for k, c in enumerate(new_cuts):
        at = bisect.bisect_left(whole, c)
        new_cuts[k] = whole[at] if at < len(whole) else len(body)
    for k in range(1, len(new_cuts)):
        new_cuts[k] = max(new_cuts[k], new_cuts[k - 1])
    edges = [0] + new_cuts + [len(body)]
    for b, s, e in zip(page.blocks, edges, edges[1:]):
        b.html = body[s:e]
    return straddled


def sync_pages(pages: "list[Page]", bodies: "list[str]") -> int:
    """Carry every page whose string a pass has changed; the number of
    boundaries that fell inside a change."""
    return sum(sync_page(p, b) for p, b in zip(pages, bodies))
