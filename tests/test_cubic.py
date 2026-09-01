"""The cubic-sheet rung: the pages the line finder cannot read.

The synthetic recipe is chosen so rung 1 genuinely declines — seven-odd
lines, below leptonica's bar — and the cubic sheet must then both model
the page and remove the bow that was put in.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from codicology import cubic as C
from codicology.dewarp import dewarp_page
from test_dewarp import _text_page, _bowed, _bow_of

needs_pd = pytest.mark.skipif(not C.available(), reason="no page_dewarp")


def _thin_bowed():
    return _bowed(_text_page(w=1500, h=800, pitch=100, seed=5), amp=8.0)


@needs_pd
def test_the_ladder_hands_off_where_the_line_finder_declines(vtb, tmp_path):
    page = _thin_bowed()
    _, rung1 = dewarp_page(page)
    assert not rung1, "recipe must decline rung 1 or the test tests nothing"
    out, rung2 = C.cubic_dewarp(page, str(tmp_path))
    assert rung2
    assert out.ndim == 3, "the facsimile keeps its colour"
    assert _bow_of(out) < _bow_of(page) * 0.5


def _pitch(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (g < 128).astype(np.float32)
    prof = ink.sum(1)
    thr = prof.max() * 0.4
    centres, y = [], 0
    while y < len(prof):
        if prof[y] > thr:
            j = y
            while j < len(prof) and prof[j] > thr:
                j += 1
            centres.append((y + j) / 2)
            y = j
        else:
            y += 1
    return float(np.median(np.diff(centres))) if len(centres) > 2 else None


@needs_pd
def test_the_type_is_not_rescaled(vtb, tmp_path):
    """The canvas is sized to the modelled page plane, so on a synthetic
    page with wide empty borders it legitimately shrinks — but the TYPE
    must ride through at its own size, or the facsimile is a zoom."""
    page = _thin_bowed()
    out, ok = C.cubic_dewarp(page, str(tmp_path))
    assert ok
    a, b = _pitch(page), _pitch(out)
    assert a and b and abs(b - a) < a * 0.30, (a, b)


def test_missing_library_degrades_to_a_no_op(vtb, tmp_path, monkeypatch):
    monkeypatch.setitem(C._state, "tried", True)
    monkeypatch.setitem(C._state, "mod", None)
    page = _text_page(w=600, h=500, pitch=60)
    out, ok = C.cubic_dewarp(page, str(tmp_path))
    assert not ok and out is page
    assert not C.available()


@needs_pd
def test_the_witness_reads_a_page(vtb, tmp_path):
    words, conf, span = C.witness(_text_page(w=900, h=700, pitch=60),
                                  str(tmp_path))
    assert words >= 0 and 0.0 <= conf <= 100.0 and 0.0 <= span <= 1.0


@needs_pd
def test_scan_opt_in_rewrites_only_what_the_ladder_changed(vtb, tmp_path):
    """--dewarp-scans: a bowed scan page is corrected on disk; a blank page
    is left byte-identical — nothing is re-encoded for no reason."""
    import os
    bowed = _thin_bowed()
    flat = np.full((700, 900, 3), 245, np.uint8)
    p1 = str(tmp_path / "page_0000.png")
    p2 = str(tmp_path / "page_0001.png")
    cv2.imwrite(p1, bowed)
    cv2.imwrite(p2, flat)
    sig2 = open(p2, "rb").read()
    # Synthetic stroke-rows are not words: the real witness reads none and
    # the blind-page gate would (rightly) refuse the sheet. This test is
    # about rewrite-only-what-changed, so the witness testifies by proxy.
    monkey = lambda img, wd, **k: (30, 80.0, 0.9)
    orig = vtb._cubic.witness
    vtb._cubic.witness = monkey
    try:
        vtb.dewarp_scan_pages([p1, p2], str(tmp_path))
    finally:
        vtb._cubic.witness = orig
    after = cv2.imread(p1)
    assert _bow_of(after) < _bow_of(bowed) * 0.6
    assert open(p2, "rb").read() == sig2, "untouched page must not be rewritten"
