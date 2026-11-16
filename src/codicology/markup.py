"""Small helpers for the markup a page is held in, shared by the modules
that work on it. Nothing here imports the rest of codicology.
"""
from __future__ import annotations

import html
import re


def _strip_tags(fragment: str) -> str:
    """The words of a fragment, with markup removed.

    Removing a tag leaves a word boundary behind, not nothing: a page whose
    heading and subheading are adjacent blocks flattened to "Chapter
    5Elasticity", which is why a chapter opener could not be recognised by
    its own printed title. Inline tags inside a word (<i>, <b>) would suffer
    the opposite fault, so only tags that open or close a block introduce
    the space.
    """
    spaced = re.sub(r"</(?:p|div|h[1-6]|li|tr|td|th|table|section|figure|"
                    r"figcaption|blockquote|ol|ul|br)\s*>|<(?:br|hr)\s*/?>",
                    " ", fragment, flags=re.I)
    return html.unescape(re.sub(r"<[^>]+>", "", spaced)).strip()
