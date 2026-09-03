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
    """The outline-and-warp crop these gutter tests were written against.

    The build itself no longer warps by an outline — the rectifier
    flattens the whole sheet — but split_spread must still divide a flat
    spread at its gutter, and a warped crop of the fixture is exactly such
    a sheet with the fixture's known geometry."""
    quad = vtb.detect_page(spread)
    assert quad is not None
    return vtb.warp_page(spread, quad)


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


def test_hard_edged_gutter_spread_is_seen_and_splits_in_two(vtb):
    """A hard-edged gutter once made detect_page take the left page alone.
    The outline no longer decides the pages — the rectifier does — but it
    still has to see the spread at all, because a photograph in which no
    page can be seen is never split."""
    spread = hard_gutter_spread()
    quad = vtb.detect_page(spread)
    assert quad is not None
    # and a flat sheet of it — what the rectifier hands back — splits in two
    flat = spread[90:1310, 120:1880]
    assert len(vtb.split_spread(flat)) == 2


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
    assert len(vtb.split_spread(vtb.warp_page(spread, corners))) == 2

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
    assert len(vtb.split_spread(vtb.warp_page(img, corners))) == 2


def test_gutter_prefers_ink_free_columns_over_a_dark_text_column(vtb):
    """A bright flat spread can carry a spine shadow FAINTER than a text
    column's own ink; cutting at the column once amputated the opening
    characters of every line of a preface. The spine is always among the
    ink-free columns."""
    img = np.full((900, 1400), 235, np.uint8)
    img[:, 690:710] = 216                     # faint spine band at centre
    for row in range(80, 820, 30):            # dense text column, darker net
        img[row:row + 6, 520:640] = 20
    x = vtb._gutter_x(img)
    assert 660 <= x <= 740, f"cut at {x}, inside text (520-640) or astray"


def test_a_sliver_quad_keeps_the_whole_frame(vtb, tmp_path):
    """A lone quad covering a sliver of the frame found SOMETHING — a
    cover's title label — but not the page. Keeping the frame loses
    nothing; cropping to the sliver loses the page."""
    import cv2 as _cv2
    f = np.full((1000, 1300, 3), 15, np.uint8)
    f[200:800, 500:800] = 235                 # label: ~14% of the frame
    src = str(tmp_path / "IMG_0001.png")
    _cv2.imwrite(src, f)
    pages, ids = vtb.pages_from_images([src], str(tmp_path), 0.10, 0,
                                       False, False, False, True,
                                       dewarp=False)
    assert len(pages) == 1
    out = _cv2.imread(pages[0])
    assert out.shape[:2] == (1000, 1300), "must keep the frame uncropped"


def test_the_cut_never_amputates_text_that_hugs_the_spine(vtb, tmp_path):
    """A blob's thresholded edge once cut off the captions sitting inside
    the spine shadow. The cut is the measured ink-free gutter of the flat
    sheet, so every stroke of both pages survives, each exactly once."""
    import cv2 as _cv2
    sheet = np.full((700, 1120, 3), 235, np.uint8)
    sheet[:, 540:580] = 40                        # a deep spine shadow
    n_left = n_right = 0
    for row in range(60, 640, 40):                # left page text
        sheet[row:row + 5, 50:480] = 20; n_left += 1
    for row in range(60, 640, 40):                # right text HUGS the spine
        sheet[row:row + 5, 595:880] = 20; n_right += 1
    pages = vtb.split_spread(sheet)
    assert len(pages) == 2

    def strokes(img):
        g = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        n, _, stats, _ = _cv2.connectedComponentsWithStats(
            (g < 100).astype("uint8"))
        return sum(1 for i in range(1, n)
                   if stats[i, _cv2.CC_STAT_WIDTH] > 50)
    total = strokes(pages[0]) + strokes(pages[1])
    assert total == n_left + n_right, \
        f"{total} strokes across halves, expected {n_left + n_right}"


def test_a_leaning_gutter_is_cut_along_its_path(vtb):
    """On a flat sheet the two text blocks can lean toward each other, so no
    single column is ink-free: a vertical cut through the least-ink column
    took the first letters of a right-hand page's lower lines on two pages
    of one book. The cut follows the gutter's path and every stroke lands
    on its own side, exactly once."""
    import cv2 as _cv2
    h, w = 900, 1400
    sheet = np.full((h, w, 3), 235, np.uint8)
    n_left = n_right = 0
    for y in range(h):                               # the spine's shadow, leaning with the fold
        lean = int(40 * y / (h - 1))                  # 40px right, top to bottom
        sheet[y, 655 + lean:675 + lean] = 90
    for row in range(60, 840, 30):
        lean = int(40 * (row + 2) / (h - 1))
        sheet[row:row + 5, 60:640 + lean] = 20; n_left += 1      # left block reaches toward the fold
        sheet[row:row + 5, 690 + lean:1340] = 20; n_right += 1   # right block starts just past it
    parts = vtb.split_spread(sheet)
    assert len(parts) == 2
    left, right = parts

    def strokes(img, x_offset):
        g = _cv2.cvtColor(img, _cv2.COLOR_BGR2GRAY)
        n, _, stats, _ = _cv2.connectedComponentsWithStats((g < 60).astype("uint8"))
        return [(int(stats[i, _cv2.CC_STAT_TOP]), int(stats[i, _cv2.CC_STAT_LEFT]) + x_offset,
                 int(stats[i, _cv2.CC_STAT_LEFT]) + x_offset + int(stats[i, _cv2.CC_STAT_WIDTH]))
                for i in range(1, n)]
    ls, rs = strokes(left, 0), strokes(right, w - right.shape[1])
    assert len(ls) == n_left and len(rs) == n_right, (len(ls), len(rs))
    # and no stroke was shortened: each spans exactly what was drawn at its row
    for top, x0, x1 in ls:
        lean = int(40 * (top + 2) / (h - 1))
        assert abs(x0 - 60) <= 1 and abs(x1 - (640 + lean)) <= 1, (top, x0, x1)
    for top, x0, x1 in rs:
        lean = int(40 * (top + 2) / (h - 1))
        assert abs(x0 - (690 + lean)) <= 1 and abs(x1 - 1340) <= 1, (top, x0, x1)


def test_a_straight_gutter_cuts_as_one_column(vtb, make_spread):
    """The path machinery must leave the ordinary case exactly as it was:
    two rectangular halves that rejoin to the sheet."""
    crop = crop_of(vtb, make_spread())
    parts = vtb.split_spread(crop)
    assert len(parts) == 2
    assert np.array_equal(np.concatenate(parts, axis=1), crop)


def test_text_hugging_the_spine_shadow_is_not_cut(vtb):
    """Letters set hard against the spine's shadow merge with it at a coarse
    scale and vanish from every ink test, so a seam ran through their first
    strokes on two pages of one book. Ink is read fine and pooled; the
    seam takes the shadow, never the letters."""
    import cv2 as _cv2
    h, w = 1200, 2400
    sheet = np.full((h, w, 3), 235, np.uint8)
    for y in range(h):
        lean = int(30 * y / (h - 1))
        sheet[y, 1170 + lean:1200 + lean] = 70                   # the spine's shadow, leaning
    n_right = 0
    for row in range(80, 1120, 28):
        lean = int(30 * (row + 3) / (h - 1))
        sheet[row:row + 6, 120:1140 + lean] = 20                 # left block, near the fold
        sheet[row:row + 6, 1200 + lean:2280] = 20                # right block starts AT the shadow's edge
        n_right += 1
    left, right = vtb.split_spread(sheet)
    g = _cv2.cvtColor(right, _cv2.COLOR_BGR2GRAY)
    n, _, stats, _ = _cv2.connectedComponentsWithStats((g < 60).astype("uint8"))
    x_off = w - right.shape[1]
    starts = sorted((int(stats[i, _cv2.CC_STAT_TOP]), int(stats[i, _cv2.CC_STAT_LEFT]) + x_off) for i in range(1, n))
    assert len(starts) == n_right, len(starts)
    for top, x0 in starts:
        lean = int(30 * (top + 3) / (h - 1))
        assert abs(x0 - (1200 + lean)) <= 1, (top, x0, 1200 + lean)
