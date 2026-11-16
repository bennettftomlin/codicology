"""The book's contents: what the layout labelled as headings and as a
printed contents page, read from each page's blocks. The rest of the
contents (the printed contents parsed, the running heads' chapters merged
in, the heads promoted) is still in pipeline.py and comes here as phase 3
of the plan types it.
"""
from __future__ import annotations

import re

from .markup import _strip_tags
from .model import Page


def labelled_heads(pages: "list[Page]") -> "list[tuple[int, int, str]]":
    """Every heading the layout labelled (SectionHeader): the page it stands
    on, its rank from its <hN>, and its text, cut at ninety characters. A
    bare number is a note marker the layout took for a heading, and is left
    out."""
    out = []
    for p in pages:
        for b in p.blocks:
            if b.label != "SectionHeader" or not b.html:
                continue
            t = re.sub(r"\s+", " ", _strip_tags(b.html)).strip()
            rank = re.match(r"\s*<h([1-6])", b.html)
            if t and not re.fullmatch(r"\d{1,3}", t):
                out.append((p.index, int(rank.group(1)) if rank else 3, t[:90]))
    return out


def labelled_contents_pages(pages: "list[Page]") -> "set[int]":
    """The pages on which the layout labelled a printed contents."""
    return {p.index for p in pages for b in p.blocks if b.label == "TableOfContents"}
