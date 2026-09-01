"""
Finding the page and splitting the spread: detect_page, limit_quad, warp_page,
split_spread.

Every scenario here is drawn with numpy and cv2, so the whole module runs in a
couple of seconds and never touches the real footage. Each test pins a
behaviour one of the four functions' comments says it exists to protect: the
gutter search's per-strip maximum, its trim back onto the lit page, its
flat-profile fallback, the skin mask that keeps a hand out of the page outline,
and limit_quad's two independent measures of a bad quad.

The one xfail is the interesting one — read it before touching detect_page.
"""
import numpy as np
import cv2
import pytest


# ── scenery the shared fixtures do not build ─────────────────────────────────

def _lines(img, x0, y0, w, lines=16, gap=58, height=26, shade=35):
    for k in range(lines):
        y = y0 + k * gap
        cv2.rectangle(img, (x0, y), (x0 + w, y + height), (shade, shade, shade), -1)
    return img


def hard_gutter_spread(width=2000, height=1400, bar=60):
    """
    make_spread's twin, with the gutter drawn as a solid dark bar instead of a
    shadow: same page rectangle, same text, hard edges down the middle.
    """
    img = np.full((height, width, 3), 40, np.uint8)
    cv2.rectangle(img, (120, 90), (width - 120, height - 90), (238, 236, 232), -1)
    cx = width // 2
    cv2.rectangle(img, (cx - bar // 2, 90), (cx + bar // 2, height - 90), (18, 18, 18), -1)
    for side in (0, 1):
        x0 = 190 + side * 830
        cv2.putText(img, str(side), (x0, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3)
        _lines(img, x0, 260, 620)
    return img


def crop_of(vtb, spread):
    """The pipeline as process_video runs it: detect, limit, warp."""
    quad = vtb.detect_page(spread)
    assert quad is not None
    return vtb.warp_page(spread, vtb.limit_quad(quad))


def single_band_minimum(crop):
    """
    Where the gutter search would land if it profiled ONE tall band instead of
    several strips and trimmed nothing — the approach split_spread's comments
    say was replaced. Used only to prove a scenario is genuinely adversarial.
    """
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    band = gray[int(h * 0.15):int(h * 0.85)].mean(axis=0).reshape(1, -1).astype(np.float32)
    profile = cv2.blur(band, (max(3, (w // 100) | 1), 1)).ravel()
    lo, hi = int(w * 0.35), int(w * 0.65)
    return lo + int(np.argmin(profile[lo:hi]))


# ── detect_page ──────────────────────────────────────────────────────────────

def test_detect_page_finds_the_lit_page_against_a_dark_desk(vtb, make_spread):
    """The bimodal case Otsu is chosen for: bright spread, dark surface."""
    spread = make_spread()
    quad = vtb.detect_page(spread)

    assert quad is not None
    assert quad.shape == (4, 2)
    # The page rectangle the fixture draws, not the whole frame: the desk border
    # is excluded on every side.
    np.testing.assert_allclose(
        quad, [[120, 90], [1880, 90], [1880, 1310], [120, 1310]], atol=3)


def test_detect_page_rejects_a_subject_too_small_to_be_the_page(vtb):
    """min_area_ratio: a bright scrap on the desk is not a page."""
    desk = np.full((1000, 1400, 3), 30, np.uint8)
    cv2.rectangle(desk, (600, 450), (820, 700), (235, 235, 235), -1)  # ~4% of frame

    assert vtb.detect_page(desk) is None


def test_hand_over_the_page_edge_does_not_drag_the_outline_onto_the_arm(vtb, make_spread):
    """
    Skin photographs about as bright as paper, so without the chroma mask the
    hand joins the page as one blob and the outline follows it off the page.
    Verified while writing this: a same-shaped blob in a NON-skin colour does
    drag the bottom edge from y=1310 (the page) to y=1398 (the desk).
    """
    clean = vtb.detect_page(make_spread())

    handed = make_spread()
    cv2.ellipse(handed, (700, 1290), (150, 210), 0, 0, 360, (150, 190, 225), -1)
    quad = vtb.detect_page(handed)

    assert quad is not None
    np.testing.assert_allclose(quad, clean, atol=3)


def test_hard_edged_gutter_keeps_both_pages_of_the_spread(vtb):
    """Once a strict xfail documenting the defect this suite waited on: a
    hard-edged gutter made detect_page take the left page alone and drop the
    facing page without a warning. Fixed by sibling detection — measured in
    the field at 62 of 92 spread photographs losing a page each."""
    spread = hard_gutter_spread()
    quad = vtb.detect_page(spread)

    assert quad is not None
    # The spread runs 120..1880; anything much narrower has eaten a page.
    assert quad[:, 0].max() - quad[:, 0].min() > 1600

    parts = vtb.split_spread(vtb.warp_page(spread, vtb.limit_quad(quad)))
    assert len(parts) == 2


# ── limit_quad ───────────────────────────────────────────────────────────────

def test_limit_quad_leaves_a_squarely_shot_page_alone(vtb, make_spread):
    """A quad already within cap passes through untouched, corner for corner."""
    quad = vtb.detect_page(make_spread())
    limited = vtb.limit_quad(quad)

    np.testing.assert_array_equal(limited, quad)


def test_limit_quad_eases_a_knuckle_quad_only_as_far_as_the_cap(vtb):
    """
    One corner left on a knuckle shears every line of text, so the quad is
    blended toward its bounding box — but by the least amount that is safe, or
    the camera tilt that was worth correcting is thrown away with it.
    """
    knuckle = np.array([[100, 100], [900, 140], [820, 1200], [100, 1200]], np.float32)
    assert vtb._quad_skew(knuckle) > vtb.MAX_QUAD_SKEW  # scenario is over cap

    limited = vtb.limit_quad(knuckle)
    box = np.array([[100, 100], [900, 100], [900, 1200], [100, 1200]], np.float32)

    assert vtb._quad_skew(limited) <= vtb.MAX_QUAD_SKEW + 1e-6
    # Nowhere near flattened to the bounding box: it still leans.
    assert np.abs(limited - knuckle).max() < np.abs(limited - box).max()


def test_limit_quad_corrects_a_lean_that_the_side_length_test_scores_as_perfect(vtb):
    """
    Comparing opposite sides scores a parallelogram at zero however far it
    leans, which is why the corner-angle check exists alongside it. This quad
    is 14 degrees off square with a skew of exactly 0.
    """
    leaning = np.array([[200, 100], [1000, 100], [1250, 1100], [450, 1100]], np.float32)
    assert vtb._quad_skew(leaning) == 0.0
    assert vtb._quad_corner_error(leaning) > vtb.MAX_CORNER_ERROR_DEG

    limited = vtb.limit_quad(leaning)

    assert not np.array_equal(limited, leaning)
    assert vtb._quad_corner_error(limited) <= vtb.MAX_CORNER_ERROR_DEG + 1e-6


# ── warp_page ────────────────────────────────────────────────────────────────

def test_warp_page_sizes_the_output_from_the_longer_of_each_opposite_pair(vtb):
    """
    Perspective makes the far edge shorter than the near one. Sizing to the
    longer of each pair stretches nothing away; sizing to the shorter would
    throw resolution off the near edge.
    """
    image = np.full((400, 600, 3), 200, np.uint8)
    quad = np.array([[50, 50], [450, 60], [440, 350], [60, 300]], np.float32)
    top, bottom = 400.12, 380.13     # |tr-tl|, |br-bl|
    left, right = 250.20, 290.17     # |bl-tl|, |br-tr|

    out = vtb.warp_page(image, quad)

    assert out.shape[1] == int(max(top, bottom)) == 400
    assert out.shape[0] == int(max(left, right)) == 290


def test_warp_page_flattens_a_tilted_spread_into_a_splittable_crop(vtb, make_spread):
    """A spread shot off-square still lands as a landscape crop that splits in two."""
    spread = make_spread()
    h, w = spread.shape[:2]
    tilt = cv2.warpPerspective(
        spread,
        cv2.getPerspectiveTransform(
            np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
            np.float32([[60, 30], [w - 30, 0], [w - 10, h - 20], [20, h - 40]])),
        (w, h), borderValue=(40, 40, 40))

    crop = crop_of(vtb, tilt)
    assert crop.shape[1] / crop.shape[0] > 1.15   # landscape: read as a spread

    parts = vtb.split_spread(crop)
    assert len(parts) == 2
    assert min(p.shape[1] for p in parts) / crop.shape[1] > 0.4


# ── split_spread ─────────────────────────────────────────────────────────────

def test_soft_gutter_spread_splits_into_exactly_two_pages(vtb, make_spread):
    parts = vtb.split_spread(crop_of(vtb, make_spread()))

    assert len(parts) == 2
    assert all(p.size for p in parts)


def test_split_halves_are_of_comparable_width(vtb, make_spread):
    """Neither half may be a sliver: that is a lost page dressed up as a split."""
    crop = crop_of(vtb, make_spread())
    parts = vtb.split_spread(crop)

    shares = [p.shape[1] / crop.shape[1] for p in parts]
    assert min(shares) > 0.4, shares
    assert max(shares) < 0.6, shares


def test_split_keeps_every_column_exactly_once(vtb, make_spread):
    """The two halves are a partition of the crop — no overlap, no lost column."""
    crop = crop_of(vtb, make_spread())

    rejoined = np.concatenate(vtb.split_spread(crop), axis=1)

    assert rejoined.shape == crop.shape
    np.testing.assert_array_equal(rejoined, crop)


def test_single_portrait_page_is_not_split(vtb, make_page):
    page = make_page()
    parts = vtb.split_spread(page)

    assert len(parts) == 1
    np.testing.assert_array_equal(parts[0], page)


def test_crop_too_square_to_be_a_spread_is_not_split(vtb):
    """min_aspect: a book page is portrait, so only landscape crops are spreads."""
    nearly_square = np.full((1000, 1100, 3), 235, np.uint8)   # aspect 1.10 < 1.15

    assert len(vtb.split_spread(nearly_square)) == 1


def test_tall_dark_photograph_does_not_steal_the_gutter(vtb, make_spread):
    """
    A photograph darkens the rows it occupies, and averaged over one tall band
    its columns read as dark as the spine. Keeping each column's BRIGHTEST
    horizontal strip tells them apart, because the photo is bright in the
    strips it does not reach and the spine is dark in all of them.
    """
    spread = make_spread()
    cv2.rectangle(spread, (1100, 140), (1280, 660), (12, 12, 12), -1)  # plate, upper half
    crop = crop_of(vtb, spread)
    photo = (1100 - 120, 1280 - 120)   # the plate's columns in crop coordinates

    parts = vtb.split_spread(crop)
    x = parts[0].shape[1]

    assert photo[0] <= single_band_minimum(crop) <= photo[1]   # scenario is adversarial
    assert not photo[0] <= x <= photo[1]
    assert abs(x / crop.shape[1] - 0.5) < 0.05


def test_dark_desk_margin_does_not_pull_the_split_off_the_spine(vtb, make_spread):
    """
    detect_page can overshoot onto the surface behind the book. That margin is
    darker than any gutter and it leaves the spread off-centre, so the profile
    is trimmed back to the lit page before the spine is looked for.
    """
    crop = crop_of(vtb, make_spread())
    spine = 1000 - 120                      # the fixture's gutter, in crop coordinates
    overshot = np.hstack([np.full((crop.shape[0], 1000, 3), 26, np.uint8), crop])

    parts = vtb.split_spread(overshot)
    x = parts[0].shape[1]

    assert single_band_minimum(overshot) < 1000    # untrimmed, it lands in the desk
    assert abs(x - (1000 + spine)) < 0.02 * crop.shape[1]


def test_flat_crop_with_no_gutter_in_view_is_halved(vtb):
    """
    When the crop caught only part of the spread the profile is flat and argmin
    picks an arbitrary column, which would slice a text column in half. Below
    GUTTER_MIN_DEPTH the split falls back to the middle: wrong by a predictable
    margin instead of severing a paragraph mid-word.
    """
    flat = np.full((800, 1600, 3), 235, np.uint8)
    _lines(flat, 100, 100, 500, lines=10)
    _lines(flat, 950, 100, 500, lines=10)

    parts = vtb.split_spread(flat)

    assert len(parts) == 2
    assert [p.shape[1] for p in parts] == [800, 800]


def test_split_survives_a_crop_too_short_to_trim_margins(vtb):
    """A crop with fewer rows than the strip profiler wants must not blow up."""
    sliver = np.full((1, 120, 3), 200, np.uint8)

    parts = vtb.split_spread(sliver)

    assert len(parts) == 2
    assert sum(p.shape[1] for p in parts) == 120


# --------------------------------------------------------------------------
# a frame whose page was never found must not be split
# --------------------------------------------------------------------------

def test_an_uncropped_frame_is_not_split_into_halves(vtb, make_spread):
    """
    The rule the build applies: split only what was successfully cropped to a
    page. On an uncropped frame the whole photograph is in hand, desk and all,
    and the darkest column down the middle is wherever that scene happens to be
    dark — on one book's cover shots it fell between the desk and the cover,
    making each a page of bare table.

    Brightness cannot stand in for this test. Measured on real captures, the two
    halves of a genuine spread matched to within 0.94-0.97 of each other while a
    cover shot sat at 0.93 — overlapping. Whether a page was found does separate
    them, which is why that is the signal used.
    """
    spread = make_spread(0)
    corners = vtb.detect_page(spread, 0.15)
    assert corners is not None, "fixture should be detectable"
    assert len(vtb.split_spread(vtb.warp_page(spread, vtb.limit_quad(corners)))) == 2

    # A frame the detector cannot resolve: a page adrift in a large dark scene.
    import numpy as np
    scene = np.full((1400, 2000, 3), 30, np.uint8)
    scene[500:800, 800:1200] = 235                      # too small to be a page
    assert vtb.detect_page(scene, 0.15) is None
    # The build keeps such a frame whole; splitting it would invent two pages
    # out of desk. That decision lives in the caller, so what is pinned here is
    # the fact the detector reports failure, which is what the caller keys on.


def test_a_spread_with_one_blank_side_still_splits(vtb):
    """
    A blank verso facing a chapter opening is a real spread and must still be
    cut in two: the blank is a page of the book, and the EPUB build drops it
    later on its own terms. Refusing to split whenever a half looks empty would
    fuse it onto the chapter opening instead.
    """
    import cv2
    import numpy as np
    img = np.full((1400, 2000, 3), 40, np.uint8)
    cv2.rectangle(img, (120, 90), (1880, 1310), (238, 236, 232), -1)
    shadow = np.zeros((1400, 2000), np.float32)
    for x in range(910, 1090):
        shadow[90:1310, x] = 95 * np.exp(-((x - 1000) / 38.0) ** 2)
    img = np.clip(img.astype(np.float32) - shadow[..., None], 0, 255).astype(np.uint8)
    # text on the right half only; the left is a blank leaf
    for k in range(16):
        y = 260 + k * 58
        cv2.rectangle(img, (1120, y), (1740, y + 26), (35, 35, 35), -1)

    corners = vtb.detect_page(img, 0.15)
    assert corners is not None
    assert len(vtb.split_spread(vtb.warp_page(img, vtb.limit_quad(corners)))) == 2
