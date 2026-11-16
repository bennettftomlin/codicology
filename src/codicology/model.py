"""What a page's reading is made of. Today one PageItem per block in reading
order, as the reader and the cache hold it; the typed Page and Block of the
plan (PLAN-2026-11-12.md) are to live here too.
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
