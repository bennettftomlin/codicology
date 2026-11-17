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

def _carry(old: str, new: str, cuts: "list[int]") -> "tuple[list[int], int]":
    """Where each cut in `old` lands in `new`, and how many fell inside a
    change rather than between unchanged text."""
    import difflib
    if old == new:
        return list(cuts), 0
    head = 0
    while head < min(len(old), len(new)) and old[head] == new[head]:
        head += 1
    tail = 0
    while tail < min(len(old), len(new)) - head and old[-1 - tail] == new[-1 - tail]:
        tail += 1
    mid_old, mid_new = old[head:len(old) - tail], new[head:len(new) - tail]
    ops = difflib.SequenceMatcher(None, mid_old, mid_new, autojunk=False).get_opcodes()
    out, straddled = [], 0
    for c in cuts:
        if c <= head:
            out.append(c)
        elif c >= len(old) - tail:
            out.append(c + len(new) - len(old))
        else:
            m = c - head
            for tag, i1, i2, j1, j2 in ops:
                if i1 <= m <= i2:
                    if tag == "equal":
                        out.append(head + j1 + (m - i1))
                    elif m == i1:
                        out.append(head + j1)
                    else:                       # inside a change: to the earlier block
                        out.append(head + j2)
                        straddled += m != i2
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
