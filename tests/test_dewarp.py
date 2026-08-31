"""The dewarp ladder, tested on pages built to bow.

The synthetic page is words drawn as blocks — leptonica's line finder wants
lines with word-like gaps, not solid rules — and the bow is a known
sinusoidal displacement, so the test can ask the one question that matters:
did the correction remove what was put in, without touching what wasn't.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from codicology import dewarp as D


def _text_page(w=1500, h=2000, pitch=34, seed=7):
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 245, np.uint8)
    y = 120
    while y < h - 120:
        x = 130
        while x < w - 130:
            wd = int(rng.integers(28, 90))
            img[y:y + 13, x:min(x + wd, w - 130)] = 20
            x += wd + int(rng.integers(10, 22))
        y += pitch
    return img


def _bowed(img, amp=9.0):
    h, w = img.shape[:2]
    xs = np.tile(np.arange(w, dtype=np.float32), (h, 1))
    bow = (amp * np.sin(np.pi * xs / w)).astype(np.float32)
    ys = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w)) - bow
    return cv2.remap(img, xs, ys, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REPLICATE)


def _bow_of(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ink = (g < 128).astype(np.float32)
    h, w = ink.shape
    prof = ink.sum(1)
    thr = prof.max() * 0.4
    res = []
    y = 0
    while y < h:
        if prof[y] > thr:
            j = y
            while j < h and prof[j] > thr:
                j += 1
            band = ink[max(0, y - 3):j + 3]
            cs = []
            for sxi in range(9):
                sl = band[:, int(w * sxi / 9):int(w * (sxi + 1) / 9)]
                if sl.sum() < 4:
                    continue
                yy = np.arange(sl.shape[0])
                cs.append(float((sl.sum(1) * yy).sum() / sl.sum()))
            if len(cs) >= 6:
                res.append(max(cs) - min(cs))
            y = j
        else:
            y += 1
    return float(np.median(res)) if res else 0.0


needs_lept = pytest.mark.skipif(not D.available(), reason="no leptonica")


@needs_lept
def test_a_bowed_page_is_straightened(vtb):
    page = _bowed(_text_page(), amp=9.0)
    before = _bow_of(page)
    out, modelled = D.dewarp_page(page)
    assert modelled
    assert out.shape == page.shape
    after = _bow_of(out)
    assert after < before * 0.5, (before, after)


@needs_lept
def test_a_wide_capture_is_modelled_at_half_size(vtb):
    """A 48MP-class page sits past the line finder's envelope at full
    resolution; the width-chosen redfactor=2 path must carry it, and the
    output must stay at the full size — the shipped page is never reduced."""
    page = _bowed(_text_page(w=3400, h=4600, pitch=64), amp=16.0)
    out, modelled = D.dewarp_page(page)
    assert modelled
    assert out.shape == page.shape
    assert _bow_of(out) < _bow_of(page) * 0.6


@needs_lept
def test_a_page_without_lines_is_declined_untouched(vtb):
    plate = np.full((1600, 1200, 3), 240, np.uint8)
    plate[300:1300, 200:1000] = 90                      # one big illustration
    out, modelled = D.dewarp_page(plate)
    assert not modelled
    assert out is plate


def test_missing_leptonica_degrades_to_a_no_op(vtb, monkeypatch):
    monkeypatch.setattr(D, "_load", lambda: None)
    page = _text_page(w=600, h=800)
    out, modelled = D.dewarp_page(page)
    assert not modelled and out is page


def test_a_quad_that_crops_the_text_is_pushed_back_out(vtb):
    """One chapter-opening spread put letters against both canvas edges:
    the detected quad ran inside the curled fore-edge and the gutter
    shadow, and no later stage can restore ink the warp never sampled.
    The text pushes the quad back out."""
    frame = np.full((1000, 1400, 3), 30, np.uint8)         # dark desk
    frame[100:900, 300:1100] = 245                         # the page
    for y in range(180, 820, 24):                          # text to the edges,
        frame[y:y + 5, 310:1090] = 20                      # at stroke width
    tight = np.array([[340, 120], [1060, 120],
                      [1060, 880], [340, 880]], np.float32)  # crops both sides
    naive = vtb.warp_page(frame, tight)
    guarded = vtb.warp_page_guarded(frame, tight)
    assert "left" in vtb._clipped_sides(naive) or "right" in vtb._clipped_sides(naive)
    assert vtb._clipped_sides(guarded) == [], vtb._clipped_sides(guarded)


def test_a_well_framed_page_is_warped_exactly_as_before(vtb):
    """The guard must cost nothing where the quad was right."""
    frame = np.full((1000, 1400, 3), 30, np.uint8)
    frame[100:900, 300:1100] = 245
    for y in range(200, 800, 24):
        frame[y:y + 5, 360:1040] = 20
    quad = np.array([[300, 100], [1100, 100],
                     [1100, 900], [300, 900]], np.float32)
    a = vtb.warp_page(frame, quad)
    b = vtb.warp_page_guarded(frame, quad)
    assert a.shape == b.shape and np.array_equal(a, b)


def test_a_transform_that_presses_text_to_the_edge_is_indicted():
    """The cubic sheet once ate 270px of a chapter opening's width and the
    confidence witness accepted it — half-letters barely move the mean. A
    transform may not create a clipped side the input did not have; one the
    input already had cannot indict it."""
    import numpy as np
    from codicology.pipeline import _adds_clip

    def page(x0):
        a = np.full((900, 600), 235, dtype=np.uint8)
        for row in range(150, 750, 40):        # 15 strokes, 5px thick
            a[row:row + 5, x0:x0 + 200] = 30
        return a

    margined, pressed = page(40), page(0)
    assert _adds_clip(margined, pressed)
    assert not _adds_clip(margined, page(60))
    assert not _adds_clip(pressed, pressed)
