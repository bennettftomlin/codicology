"""The witness: what a classical reader with no language model, tesseract,
finds on a page. It cannot invent, so its silence where the recogniser read
paragraphs is evidence of invention, and its words are what a page about to
be dropped as blank is held against. Every caller reaches it here
(witness._classical_text), so a test that stands in for tesseract does so
in one place.
"""
from __future__ import annotations

import functools
import re


def witness_available() -> bool:
    """Whether the classical reader this pipeline testifies with is installed."""
    import shutil
    return shutil.which("tesseract") is not None


@functools.lru_cache(maxsize=None)
def _classical_text(page_path: str) -> "str | None":
    """What the classical reader makes of the page, or None if unavailable."""
    import shutil
    import subprocess
    if not shutil.which("tesseract"):
        return None
    try:
        return subprocess.run(["tesseract", page_path, "stdout", "--psm", "3"],
                              capture_output=True, text=True,
                              timeout=300).stdout
    except Exception:
        return None


@functools.lru_cache(maxsize=None)
def _classical_word_count(page_path: str, min_len: int = 2) -> "int | None":
    """
    How many words a classical OCR finds on the page, or None if unavailable.

    Tesseract has no generator, so it cannot invent: on a blank or bleed-
    through page it returns nothing at all — measured 0 words on every blank
    verso, and exactly "TIHKAL The Continuation Epilogue" on the part-title
    whose ghost text seeded a 91-word invention. That silence makes it a
    witness against fabrication on pages with no text layer to consult. It
    also goes silent on some REAL pages (a faded stamp, a dedication), so
    its silence may only ever raise suspicion, never convict on its own.
    """
    import shutil
    import subprocess
    if not shutil.which("tesseract"):
        return None
    try:
        r = subprocess.run(["tesseract", page_path, "stdout", "--psm", "3"],
                           capture_output=True, text=True, timeout=120)
        return len(re.findall(r"[A-Za-z]{%d,}" % min_len, r.stdout))
    except Exception:
        return None
