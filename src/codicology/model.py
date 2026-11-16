"""What a page's reading is made of. Today one PageItem per block in reading
order, as the reader and the cache hold it; the typed Page and Block of the
plan (PLAN-2026-11-12.md) are to live here too.
"""
from __future__ import annotations

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
