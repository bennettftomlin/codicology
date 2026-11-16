"""The book's notes. For now, what the layout itself says of a page's foot,
read from the page's blocks; linking calls to their notes is still in
pipeline.py and comes here as phase 3 of the plan types it.
"""
from __future__ import annotations

from .markup import _strip_tags
from .model import Page


def footnote_share(pages: "list[Page]") -> "list[float]":
    """For each page, how much of its foot the layout itself called
    footnotes: of the words from its first Footnote block on, the share in
    Footnote blocks. 0 on a page with no Footnote block. The foot linker
    takes a page the layout has named this way at its word (see
    link_footnotes)."""
    out = []
    for p in pages:
        first = next((k for k, b in enumerate(p.blocks) if b.label == "Footnote"), None)
        if first is None:
            out.append(0.0)
            continue
        words = lambda b: len(_strip_tags(b.html).split()) if b.html else 0
        after = sum(words(b) for b in p.blocks[first:])
        out.append(sum(words(b) for b in p.blocks[first:] if b.label == "Footnote") / after
                   if after else 0.0)
    return out
