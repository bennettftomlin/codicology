"""Pages out of video and photographs: choosing the still frames, finding the
page in each, flattening it, cutting a spread at its gutter, straightening
and cleaning it. Nothing here reads text.
"""
from __future__ import annotations

import cv2
import numpy as np
import os
import shutil
from PIL import Image, ImageEnhance
from . import rectify as _rectify


# ── Frame extraction ──────────────────────────────────────────────────────────

def _sharpness(frame: np.ndarray, width: int = 480) -> float:
    """Blur score on a downscaled copy — only relative values matter, and this is much faster."""
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, max(1, int(h * width / w))))
    return blur_score(small)


def _frame_score(frame: np.ndarray, width: int = 480) -> tuple[float, float]:
    """
    Rank frames within one still stretch. Returns (score, true sharpness).

    Sharpness alone picks the wrong frame whenever a hand is holding the page
    down: a hand pressed flat is perfectly in focus and photographs sharper than
    the same page a moment later, half-released and slightly bowed. So the page
    that reaches the book is the one with fingers across it. Discounting the
    score by how much of the frame is skin lets a marginally softer but unobscured
    frame win, which is the one a reader wants.
    """
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (width, max(1, int(h * width / w))))
    sharp = blur_score(small)

    # Look for the hand only where the page is. Measured across the whole frame
    # this reads 22-29% on every single frame, because the hands holding the book
    # steady never leave the edges — a constant that ranks nothing. Over the
    # middle it reads 0.0% on a clear page and 5-7% when fingers lie across the
    # text, which is the distinction worth acting on.
    sh, sw = small.shape[:2]
    middle = _skin_mask(small)[int(sh * 0.18):int(sh * 0.82),
                               int(sw * 0.20):int(sw * 0.80)]
    hand = float((middle > 0).mean())
    return sharp * max(0.0, 1.0 - HAND_PENALTY * hand), sharp


# ── Blur detection ────────────────────────────────────────────────────────────

def blur_score(image: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# ── Page detection & perspective correction ───────────────────────────────────

def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Order four points: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype=np.float32)


DETECT_WIDTH = 1000


# A column this much darker than the page counts as the spine shadow. Measured
# gutters run 60+ grey levels below the page; a crop that missed the spine is
# flat to within ~15, so anything shallower is noise rather than a gutter.
GUTTER_MIN_DEPTH = 30.0


# Columns above this fraction of the page's own brightness are page rather than
# the surface behind it.
PAGE_LEVEL_RATIO = 0.6


# Horizontal strips the gutter search profiles independently, so a photograph
# tall enough to darken one strip's average but not another cannot be mistaken
# for a spine that is dark the same way at every height.
GUTTER_STRIPS = 4


# Hand pixels in YCrCb. Chroma separates skin from both the page (neutral) and
# the desk (dark) far more cleanly than brightness, which is the whole problem.
SKIN_LO = (0, 135, 85)


SKIN_HI = (255, 180, 135)


# How far off an edge a contour point may sit and still be fitted to it, as a
# fraction of the page's short side.
EDGE_BAND = 0.06


# How heavily a hand lying across the page discounts that frame when choosing
# which moment of a still stretch to keep. Measured over the middle of the frame,
# a clear page reads 0.0% and fingers across the text read 5-7%, so at 8.0 a
# fingered frame scores roughly half a clear one and only wins when the
# alternative is genuinely blurred. Above 12.5% the frame scores zero — still
# kept if nothing better exists in that stretch, since a page is never dropped
# for this.
HAND_PENALTY = 8.0


# How many of a still stretch's best-scoring frames stay eligible for the final
# pick. Only needs to be large enough that a detectable frame is usually among
# the top few even when the single best-scoring one is not; detect_page runs
# once per candidate in this pool, so it trades a little of the cropping pass's
# time for not losing a page to a scoring quirk.
CANDIDATE_POOL = 6


# How much a frame may move and still be worth photographing. Measured on this
# footage against pages checked by eye: frames that read cleanly all sat at or
# below 2.52, every visibly ghosted one at or above 2.73. The gate sits in that
# gap. It is far below the threshold that splits runs (5.40), which only has to
# decide when a page stopped moving — not whether it had gone still enough for
# the camera, which is a much stricter question.
QUIET_GATE = 2.5


# How many of a run's least-moving frames to keep as a fallback for holds where
# nothing at all clears QUIET_GATE — a shaky stretch still has to yield a page.
# Measured on this footage the only two clean frames of one such hold came in at
# motion ranks 1 and 2 of 18, so a handful is plenty.
QUIET_POOL = 5


# Frames this far into a still stretch are excluded from the settled pool
# entirely (see select_page_frames). At ~30fps this is a fraction of a second —
# enough to clear a hand still mid-withdrawal, small enough that even the
# shortest permitted run (min_still_frames, default 15) keeps most of its
# length eligible.
SETTLE_MARGIN = 5


# Text deskew. The search runs to 15 degrees because a page can sit further off
# level than a quad ever admits to; below a quarter of a degree, rotating costs
# a resample and buys nothing an OCR engine would notice.
DESKEW_WIDTH = 600


DESKEW_MAX_DEG = 15.0


DESKEW_STEP_DEG = 0.25


DESKEW_MIN_DEG = 0.25


DESKEW_MIN_INK = 0.008


# Roughly a text stroke's thickness at DESKEW_WIDTH; anything broader than this
# is page furniture or the desk, not a letter.
DESKEW_STROKE = 9


# How much the best angle must beat a typical one before the page is believed to
# have lines of text at all. Prose ruled into parallel lines scores 2.0 upwards
# even on a sparse copyright page. Artwork does not: a cover collaging newspaper
# clippings at odd angles has plenty of print on it but no single direction, and
# scores 1.14 — so it must fall below this, or the cover gets straightened to
# whichever clipping happened to win.
DESKEW_PEAK_RATIO = 1.80


def _quad_from_contour(contour: np.ndarray) -> np.ndarray:
    """
    Reduce a contour to four corners.

    A photographed book is never a clean quadrilateral — pages curl, corners are
    rounded, and the outline picks up the reader's fingers — so demanding that
    approxPolyDP return exactly 4 vertices rejects perfectly usable pages. Try
    progressively coarser simplification, then fall back to the tightest
    enclosing rotated rectangle.
    """
    peri = cv2.arcLength(contour, True)
    for frac in (0.02, 0.04, 0.06, 0.08, 0.12):
        approx = cv2.approxPolyDP(contour, frac * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).astype(np.float32)
    return cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Pixels that look like the reader's hand rather than paper."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, SKIN_LO, SKIN_HI)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)


def _corners_from_edges(contour: np.ndarray) -> np.ndarray | None:
    """
    Corners as the intersections of the page's four edges.

    Taking the extremes of the outline puts a corner wherever the outline
    reaches furthest, which is a knuckle whenever a hand overlaps the page. A
    line fitted along a whole edge is barely moved by that: the fingers are a
    short bulge among many points lying straight, and Huber weighting discounts
    them. Where the true corner is hidden under a hand it is recovered rather
    than guessed, as the meeting point of the two edges that are still visible.
    """
    hull = cv2.convexHull(contour).reshape(-1, 2).astype(np.float32)
    box = _order_corners(cv2.boxPoints(cv2.minAreaRect(hull)))
    short = min(np.linalg.norm(box[1] - box[0]), np.linalg.norm(box[3] - box[0]))
    if short < 1.0:
        return None

    edges = []
    for i in range(4):
        a, b = box[i], box[(i + 1) % 4]
        span = np.linalg.norm(b - a)
        along = (b - a) / (span + 1e-9)
        normal = np.array([-along[1], along[0]], dtype=np.float32)
        offset = np.abs((hull - a) @ normal)
        travel = (hull - a) @ along
        near = hull[(offset < short * EDGE_BAND)
                    & (travel > -short * 0.05)
                    & (travel < span + short * 0.05)]
        if len(near) < 8:
            edges.append((a, along))  # too little of this edge in view
            continue
        vx, vy, x0, y0 = cv2.fitLine(near, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        edges.append((np.array([x0, y0], np.float32), np.array([vx, vy], np.float32)))

    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = edges[i - 1], edges[i]
        basis = np.array([d1, -d2], dtype=np.float32).T
        if abs(float(np.linalg.det(basis))) < 1e-6:
            return None  # parallel edges cannot meet
        t = np.linalg.solve(basis, p2 - p1)
        corners.append(p1 + t[0] * d1)
    return _order_corners(np.array(corners, dtype=np.float32))


def detect_page(image: np.ndarray, min_area_ratio: float = 0.15) -> np.ndarray | None:
    """
    Find the page (or two-page spread) in the frame.
    Returns ordered (4, 2) float32 corners in source coordinates, or None.

    Used to score frames and to tell a photograph with a page in it from
    one of bare desk; the rectifier, not this outline, flattens the page.
    """
    quads = detect_page_quads(image, min_area_ratio)
    if not quads:
        return None
    return quads[0]


def detect_page_quads(image: np.ndarray,
                      min_area_ratio: float = 0.15) -> "list[np.ndarray]":
    """
    The page quads in the frame, in reading order. [] when none is found.

    One quad is the usual case: a single page, or an open spread whose
    sheet forms one bright blob for the gutter split to divide later. Two
    quads mean the gutter shadow already divided the spread at detection —
    and the facing pages of an open book lie in two different planes, so
    warping each by its own quad corrects a keystone the union warp would
    bake into both halves. Measured on one book: 62 of 92 spread
    photographs lost their facing page entirely when only the larger blob
    was taken, including a chapter opening discarded while its blank verso
    shipped.

    Detection runs on a downscaled copy: contour finding on a 24-megapixel
    phone photo is slow, and the edges we want survive downscaling easily.
    """
    h, w = image.shape[:2]
    scale = DETECT_WIDTH / float(w) if w > DETECT_WIDTH else 1.0
    small = cv2.resize(image, (int(w * scale), int(h * scale))) if scale < 1.0 else image

    sh, sw = small.shape[:2]
    min_area = sw * sh * min_area_ratio
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Skin photographs about as bright as paper, so a hand resting on the page
    # joins it as one blob and drags the outline off along the arm. Nothing
    # downstream can tell the two apart once merged, so drop it here.
    hands = _skin_mask(small)

    # Otsu first: a lit page on a dark surface is exactly the bimodal case it
    # handles, and it yields one solid blob. Canny fragments on real paper
    # texture, and adaptive thresholding finds every letter as its own contour.
    strategies = [
        cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        cv2.dilate(cv2.Canny(blurred, 50, 150), np.ones((3, 3), np.uint8), iterations=1),
        cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv2.THRESH_BINARY_INV, 11, 2),
    ]

    for mask in strategies:
        mask = mask.copy()
        mask[hands > 0] = 0
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ranked = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        for contour in ranked:
            if cv2.contourArea(contour) < min_area:
                break
            quad = _corners_from_edges(contour)
            if quad is None or cv2.contourArea(quad) < min_area:
                quad = _quad_from_contour(contour)
                if cv2.contourArea(quad) < min_area:
                    continue
            return [_order_corners(quad) / scale]
    return []


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    codes = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE,
    }
    return cv2.rotate(image, codes[degrees]) if degrees in codes else image


def split_spread(image: np.ndarray, min_aspect: float = 1.15) -> list[np.ndarray]:
    """
    Split a two-page spread into left and right pages at the gutter.

    Only landscape crops are treated as spreads; a single book page is portrait.
    The gutter is the darkest ink-free band near the middle — the shadow where
    the pages meet the spine — measured on a column profile smoothed enough that
    an inter-word gap in a text column cannot win.

    The cut follows the gutter's PATH, not one column: on a flat sheet the two
    text blocks can lean toward each other, and a single vertical cut through
    the least-ink column took the first letters of a right-hand page's lower
    lines on two pages of one book. The gutter is hunted in horizontal strips,
    a line is fitted through the strips' finds, and each page keeps every
    pixel on its side of that line with paper painted over the other side.
    A gutter that stands straight cuts exactly as it always did.
    """
    h, w = image.shape[:2]
    if h == 0 or w / float(h) < min_aspect:
        return [image]

    path = _gutter_path(image)
    if path.max() - path.min() < GUTTER_PATH_MIN_LEAN * w:
        x = int(round(float(path.mean())))
        return [image[:, :x], image[:, x:]]
    paper = np.median(image.reshape(-1, image.shape[-1]), axis=0) if image.ndim == 3 \
        else np.median(image)
    xs = np.arange(w)[None, :]
    left_of = xs < path[:, None]
    left = image.copy()
    left[~left_of] = paper
    right = image.copy()
    right[left_of] = paper
    x_hi = int(np.ceil(path.max())) + 1
    x_lo = int(np.floor(path.min()))
    return [left[:, :x_hi], right[:, x_lo:]]


# A gutter path whose ends differ by less than this fraction of the width
# is a straight gutter, cut as one column as before.
GUTTER_PATH_MIN_LEAN = 0.003


# The seam may wander this far from the whole-sheet gutter column: a gap
# it cannot find within a tenth of the width on either side is not there.
_SEAM_UNAVAILABLE_SAID = False     # the one-line note, once a run


GUTTER_SEAM_REACH = 0.10


# Seam costs. A letter must outweigh any detour the grid allows: the grid
# is at most 2000 cells wide and a few thousand tall, so a whole traverse
# through shadow costs under 2000 * 0.001 — nothing beside one letter.
GUTTER_SEAM_COST_LETTER = 1e4


GUTTER_SEAM_COST_PLATE = 1e2


GUTTER_SEAM_COST_SHADOW = 1e-3


# Cells within this fraction of the width of any content cost up to this
# much more, so the seam keeps its distance where it can.
GUTTER_SEAM_MARGIN = 0.01


GUTTER_SEAM_COST_NEAR = 1e-2


# An ink component at least this tall (fraction of the sheet) is a line,
# not a letter: a crease, a rule, the edge of the spine's shadow.
GUTTER_SEAM_LINE_HEIGHT = 0.30


# Content a component may span of the sheet's height and still be content
# (text, a figure, a plate); a dark band taller than that is the spine's
# shadow, which the seam is free to run through.
GUTTER_SEAM_MAX_CONTENT_HEIGHT = 0.70


def _gutter_path(image: np.ndarray) -> np.ndarray:
    """The gutter's x for every row, as the minimum-cost seam through the
    gap between the pages.

    The classical page-splitting answer (ScanTailor finds candidate lines
    and scores how well they separate the pages' content; seam carving
    finds the least-cost top-to-bottom path) applied to a flat sheet:
    every pixel of content — text ink, and any dark region that does not
    span the sheet's height, a figure or a plate — costs; the spine's
    shadow, which does span the height, is free; and a small pull toward
    the whole-sheet gutter column keeps the seam in the shadow where the
    gap is wide. The seam may move one probe pixel per row, so it follows
    a gap that leans or curves and never crosses a letter while a clear
    path exists. A seam that wanders less than a straight gutter would
    hands back the whole-sheet column, and the cut is exactly as it was.
    """
    h, w = image.shape[:2]
    whole = float(_gutter_x(image))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # The seam's grid is two thousand pixels across: at a thousand, the
    # gaps between lines of type closed up once ink was pooled onto the
    # grid, and a seam that must slip sideways between two lines to get
    # round a block that leans into the fold had no row to do it in.
    probe_w = min(w, 2000)
    scale = probe_w / float(w)
    sh = max(8, int(h * scale))
    small = cv2.resize(gray, (probe_w, sh))
    # Ink is read at a finer scale and max-pooled onto the seam's grid: at
    # the grid's own scale a stroke is under a pixel wide and letters become
    # blobs, and a blob hugging the spine's shadow merges with it and
    # vanishes from every test — which is exactly where the cut mattered.
    # At the finer scale each stroke is its own thin dark feature, seen
    # beside the shadow or not.
    # Read at the sheet's own resolution (capped): at three thousand pixels
    # across, the nine-pixel kernel was wider than the gap between lines of
    # type, adjacent lines merged into one block of ink, and the seam had
    # no clear row to slip sideways through. At full size the gap is wide
    # and each line stands alone.
    fine_w = min(w, 6000)
    fine = gray if fine_w == w else cv2.resize(gray, (fine_w, max(8, int(h * fine_w / w))))
    fhat = cv2.morphologyEx(fine, cv2.MORPH_BLACKHAT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ink = cv2.resize((fhat > 40).astype(np.float32), (probe_w, sh),
                     interpolation=cv2.INTER_AREA) > 0
    # Letters are short. A thin dark feature that runs down a third of the
    # sheet is a crease at the spine's edge or a rule, and costing it as
    # letters pushed the seam out of the gutter and through the edge of a
    # plate that reached the fold. Tall ink components are not letters;
    # what they are, the dark and shadow rules below decide.
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    tall = [i for i in range(1, n) if stats[i, cv2.CC_STAT_HEIGHT] >= GUTTER_SEAM_LINE_HEIGHT * sh]
    if tall:
        ink &= ~np.isin(labels, tall)
    core = small[int(sh * 0.2):max(int(sh * 0.2) + 1, int(sh * 0.8)),
                 int(probe_w * 0.2):int(probe_w * 0.8)]
    paper = float(np.percentile(core, 75))
    dark = small < paper * PAGE_LEVEL_RATIO
    # The spine's shadow is dark down most of a column; a plate, a figure
    # or a letter is dark for only part of one. Graded costs, so the seam
    # takes the shadow over a plate and a plate over a letter when it has
    # no free path: a leaning shadow is dark for less of each column and
    # reads as plate-like, and still beats the text beside it. The grades
    # are far apart on purpose: with letters at ten and shadow at a half,
    # a twenty-cell detour through the shadow cost the same as cutting one
    # letter, and the seam cut letters — eighteen on one page. No detour
    # the grid allows may ever be dearer than a single letter.
    shadow_cols = dark.mean(axis=0) >= GUTTER_SEAM_MAX_CONTENT_HEIGHT
    cost_map = np.zeros((sh, probe_w), dtype=np.float32)
    cost_map[dark] = GUTTER_SEAM_COST_PLATE
    cost_map[:, shadow_cols] = np.where(dark[:, shadow_cols], GUTTER_SEAM_COST_SHADOW, 0.0)
    cost_map[ink] = GUTTER_SEAM_COST_LETTER
    # Keep a margin: a seam that costs nothing anywhere in the gap hugged
    # the edge of a plate that reached the fold, tracing its outline cell
    # by cell, and the page came out with a ragged left border. Cells near
    # content cost a little, rising toward it, so the seam runs down the
    # middle of whatever gap there is and touches nothing it need not.
    away = cv2.distanceTransform((~(ink | dark)).astype(np.uint8), cv2.DIST_L2, 3)
    margin = max(2.0, GUTTER_SEAM_MARGIN * probe_w)
    cost_map += GUTTER_SEAM_COST_NEAR * np.clip(1.0 - away / margin, 0.0, 1.0).astype(np.float32)
    wx = whole * scale
    lo = max(0, int(wx - GUTTER_SEAM_REACH * probe_w))
    hi = min(probe_w, int(wx + GUTTER_SEAM_REACH * probe_w) + 1)
    if hi - lo < 8:
        return np.full(h, whole, dtype=np.float32)
    xs = np.arange(lo, hi)
    cost = cost_map[:, lo:hi] \
        + (np.abs(xs - wx) / probe_w).astype(np.float32)[None, :] * 1e-4
    # scikit-image's minimum cost path does the seam; a free row above and
    # below lets it start and end in whichever column it likes. It comes
    # with the `geometry` extra, and this was the one place that assumed
    # the extra: without it a photographed spread ended the capture in an
    # ImportError. It is cut straight down the gutter instead, and said.
    try:
        from skimage.graph import route_through_array
    except ImportError:
        if not _SEAM_UNAVAILABLE_SAID:
            globals()["_SEAM_UNAVAILABLE_SAID"] = True
            print("  [~] scikit-image is not installed, so a spread is cut "
                  "straight down its gutter and not along the fold's own "
                  "line (pip install 'codicology[geometry]')")
        return np.full(h, whole, dtype=np.float32)
    free = np.full((1, hi - lo), 1e-6, dtype=np.float32)
    grid = np.vstack([free, cost + 1e-6, free])
    start_col = int(np.clip(round(wx) - lo, 0, hi - lo - 1))
    route, _total = route_through_array(grid, (0, start_col), (sh + 1, start_col),
                                        fully_connected=True, geometric=False)
    seam = np.full(sh, float(start_col), dtype=np.float32)
    for r, cidx in route:
        if 1 <= r <= sh:
            seam[r - 1] = cidx
    seam = (seam + lo + 0.5) / scale
    path = np.interp(np.arange(h), (np.arange(sh) + 0.5) / scale, seam,
                     left=seam[0], right=seam[-1])
    if path.max() - path.min() < GUTTER_PATH_MIN_LEAN * w:
        return np.full(h, whole, dtype=np.float32)
    return np.clip(path, 0, w - 1).astype(np.float32)


def _gutter_x(image: np.ndarray) -> int:
    """The column where the spine divides a spread crop.

    The darkest smoothed vertical band near the middle, with the fallbacks
    split_spread has always used: trim to the lit page first, and take the
    middle when no pronounced shadow exists.
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    kernel = max(3, (w // 100) | 1)

    # A genuine spine shadow runs the full height of the page; a photograph only
    # darkens the rows it occupies. Averaged over one tall band the two can look
    # alike — a photo's column averages as dark as the spine even though it is
    # bright above and below the photo's own borders. Profiling several
    # horizontal strips independently and keeping each column's BRIGHTEST strip
    # tells them apart: a real spine stays dark in every strip, so even its
    # brightest showing is dark; a photo is bright in whichever strips it does
    # not reach, and that bright strip is what gives it away.
    y0, y1 = int(h * 0.15), int(h * 0.85)
    if y1 <= y0:  # a crop too short for the margin trim to leave any rows
        y0, y1 = 0, h
    edges = np.linspace(y0, y1, GUTTER_STRIPS + 1).astype(int)
    strips = [cv2.blur(gray[a:b].mean(axis=0).reshape(1, -1).astype(np.float32), (kernel, 1)).ravel()
              for a, b in zip(edges[:-1], edges[1:]) if b > a]
    if not strips:  # still nothing — fewer rows than GUTTER_STRIPS to divide
        strips = [gray[y0:y1].mean(axis=0).astype(np.float32)]
    smooth = np.maximum.reduce(strips) if len(strips) > 1 else strips[0]

    # detect_page can overshoot the book onto the dark surface behind it. That
    # margin is darker than any gutter, and it leaves the spread off-centre in
    # the crop, so the spine no longer sits near the middle. Trim back to the
    # lit page before going looking for it.
    page_level = float(np.percentile(smooth, 75))
    bright = np.flatnonzero(smooth > page_level * PAGE_LEVEL_RATIO)
    left, right = (int(bright[0]), int(bright[-1]) + 1) if bright.size else (0, w)
    if right - left < w * 0.5:  # nothing page-like found; trust the whole crop
        left, right = 0, w
    inner = right - left

    lo, hi = left + int(inner * 0.35), left + int(inner * 0.65)
    # The darkest band is not always the spine: on a bright, flat spread the
    # spine shadow can be fainter than a TEXT COLUMN's own ink, and cutting
    # at the column amputated the opening characters of every line of one
    # book's preface. Measured there: the chosen column carried a black-hat
    # ink fraction of 0.19 while 156 of the 300 window columns were ink-free
    # (<0.002) — the spine is always among the ink-free ones, because a
    # shadow darkens paper smoothly and strokes do not live in the fold.
    # So the hunt runs over ink-free columns first; only a window with no
    # ink-free candidate falls back to the plain darkest.
    probe_w = 1000
    small_g = cv2.resize(gray, (probe_w, max(8, int(h * probe_w / w))))
    hat = cv2.morphologyEx(small_g, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ink_cols = np.interp(np.arange(w), np.linspace(0, w - 1, probe_w),
                         (hat > 40).mean(axis=0))
    window = np.arange(lo, hi)
    inkfree = window[ink_cols[lo:hi] < 0.002]
    if inkfree.size:
        x = int(inkfree[np.argmin(smooth[inkfree])])
    else:
        # No ink-free column at all — a tight layout runs text hard
        # against the spine on both sides. The least-ink column is still
        # the fold; the DARKEST column is a text column, and cutting
        # there amputates it.
        x = int(window[np.argmin(ink_cols[lo:hi])])

    # A spine in view is a pronounced shadow. When the crop caught only part of
    # the spread the profile across the middle is flat instead, and argmin picks
    # an arbitrary column — often pinned to the window edge — which would slice a
    # text column in half. Halving the crop is wrong there too, but it is wrong
    # by a predictable margin instead of severing a paragraph mid-word.
    if float(np.median(smooth[left:right])) - float(smooth[x]) < GUTTER_MIN_DEPTH:
        x = left + inner // 2
    return x


def _warp_transform(corners: np.ndarray):
    """The perspective matrix and output size warp_page would use."""
    tl, tr, br, bl = corners
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(corners, dst), out_w, out_h


def warp_page(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct the detected page to a flat rectangle."""
    M, out_w, out_h = _warp_transform(corners)
    return cv2.warpPerspective(image, M, (out_w, out_h))


# ── Page deduplication ────────────────────────────────────────────────────────

def _thumb(image: np.ndarray, size: int = 64) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.resize(gray, (size, size)).astype(np.float32)


def scan_motion(video_path: str, stride: int = 1) -> list[float]:
    """Per-frame motion for the whole video. Thumbnails only, so this is cheap."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    motions: list[float] = []
    prev: np.ndarray | None = None
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if stride > 1 and idx % stride:
            idx += 1
            continue
        idx += 1
        thumb = _thumb(frame)
        motions.append(0.0 if prev is None else float(np.abs(thumb - prev).mean()))
        prev = thumb

    cap.release()
    return motions


def _otsu_split(values: list[float], empty: float) -> float:
    """
    Find the valley between two clusters in a 1D distribution.

    Otsu's criterion, evaluated exactly on the sorted samples rather than through
    a 256-bin histogram. When the two clusters are far apart the histogram form
    has a plateau of equally optimal thresholds and OpenCV returns the lowest of
    them, which can sit *inside* the lower cluster; the threshold then lands one
    sample short and admits the very value it was meant to exclude.

    Returns the midpoint of the dividing gap, so no sample sits on the boundary.
    """
    arr = np.sort(np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64))
    n = arr.size
    if n < 2 or arr[-1] - arr[0] < 1e-9:
        return empty

    total = float(arr.sum())
    csum = np.cumsum(arr)[:-1]
    k = np.arange(1, n, dtype=np.float64)
    w0 = k / n
    mu0 = csum / k
    mu1 = (total - csum) / (n - k)
    variance = w0 * (1.0 - w0) * (mu0 - mu1) ** 2

    best = float(variance.max())
    ties = np.flatnonzero(variance >= best - 1e-12)
    widest = ties[int(np.argmax(arr[ties + 1] - arr[ties]))]
    return float(arr[widest] + arr[widest + 1]) / 2.0


def auto_motion_threshold(motions: list[float]) -> float:
    """
    Split still frames from page turns using Otsu's method.

    A fixed threshold cannot work across footage: the baseline depends on the
    camera, the lighting and how steadily the book is held. The motion histogram
    is reliably bimodal though — a dense low cluster while a page is held, a
    sparse high cluster during turns — so we find the valley between them.
    """
    return _otsu_split(motions, float("inf"))


MIN_CLUSTER_RATIO = 3.0


def _cluster_cut(values: list[float], min_ratio: float = MIN_CLUSTER_RATIO) -> float:
    """
    Cut between a low and a high cluster of positive values, or -inf if the values
    do not actually divide into two.

    Split in log space: both quantities this measures — Laplacian variance, and how
    many frames a page was held — span orders of magnitude, so on a linear axis the
    high cluster is a thin tail and Otsu lands far too low.

    A single cluster still has a widest internal gap, and splitting there would
    throw away good pages. Requiring the two sides to differ by min_ratio lets the
    cut opt out instead, for footage that has nothing to reject.
    """
    positive = [float(v) for v in values if np.isfinite(v) and v > 0]
    if len(positive) < 2:
        return float("-inf")
    cut = _otsu_split([float(np.log10(v)) for v in positive], float("-inf"))
    if not np.isfinite(cut):
        return float("-inf")

    arr = np.asarray(positive)
    cut = float(10.0 ** cut)
    lower, upper = arr[arr < cut], arr[arr >= cut]
    if lower.size == 0 or upper.size == 0 or upper.min() < min_ratio * lower.max():
        return float("-inf")
    return cut


def auto_blur_threshold(scores: list[float]) -> float:
    """Sharpness below which a frame is a motion-smeared fragment of a turn."""
    return _cluster_cut(scores)


def auto_hold_threshold(run_lengths: list[int]) -> float:
    """Still-frame count below which a stretch is a pause mid-turn, not a page."""
    return _cluster_cut([float(r) for r in run_lengths])


def select_page_frames(
    video_path: str,
    output_dir: str,
    motion_threshold: float,
    min_still_frames: int,
    stride: int = 1,
    min_area_ratio: float = 0.15,
) -> tuple[list[str], list[float], list[float], list[int]]:
    """
    Stream the video and save the single best frame for each physical page.

    Comparing page *content* cannot work here: two consecutive pages of prose are
    near-identical once downsampled, so perceptual hashes score them as duplicates
    and real pages silently vanish. We segment on motion instead — turning a page
    moves the image, holding it still does not — and keep the sharpest frame from
    each still stretch.

    Motion is measured on every frame rather than on a subsample, because a fast
    page turn can complete entirely between two subsampled frames, hiding the
    boundary and merging two pages into one.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    saved: list[str] = []
    sharpness: list[float] = []
    held_frames: list[int] = []
    motions: list[float] = []
    prev_thumb: np.ndarray | None = None
    # The few best-scoring frames seen so far this run, worst-first, capped at
    # CANDIDATE_POOL. A hand pressed flat on the page can be sharp enough to
    # outscore every alternative even after the penalty, and detect_page is not
    # run here to check — that only happens once, later, on whichever single
    # frame this function hands back. Keeping a short list rather than one
    # frame means a later detectability check has other options to fall back
    # on, instead of the run's answer being fixed before that check ever runs.
    candidates: list[tuple[float, float, np.ndarray]] = []  # (score, sharp, frame)
    # A second pool that excludes the run's first SETTLE_MARGIN frames. Motion
    # dropping below threshold is not the same moment as the page coming to
    # rest — a hand still withdrawing at that instant is often at a corner or
    # edge, outside the centre band the hand penalty measures, so it can score
    # as if no hand were present at all. Measured on this footage: the run's
    # very first frame was the frame select_page_frames actually picked in 28%
    # of the cases where the hand-penalised score changed the answer, against
    # 2% for plain sharpness — a bias in the metric, not a real quality signal.
    # Keeping a separate pool that only starts scoring after the settle margin
    # removes that frame from contention outright, rather than trying to
    # penalise it correctly after the fact.
    settled: list[tuple[float, float, np.ndarray]] = []
    # The best-scoring frames among those that actually held still — stillness
    # decides who is eligible, the score decides between them. Ranking on
    # stillness alone picks hands: a hand resting flat on the page is the
    # stillest thing in the run, so the frame it obscures wins every time.
    # Gating on stillness and then ranking on the hand-penalised score keeps
    # mid-turn frames out without handing the page back to the hand.
    steady: list[tuple[float, float, np.ndarray]] = []
    # The run's least-moving frames, worst-first, capped at QUIET_POOL, used
    # only when no frame in the run cleared QUIET_GATE. Entries are
    # (unsteadiness, score, sharp, frame).
    quiet: list[tuple[float, float, float, np.ndarray]] = []
    # A frame's unsteadiness is the larger of the motion measured into it and
    # the motion measured out of it, because a page that starts moving during
    # an exposure blurs that exposure just as much as one that was already
    # moving. The motion out of a frame is not known until the next frame is
    # read, so each frame waits here for one iteration before it can be ranked.
    pending: tuple[float, float, float, np.ndarray] | None = None
    run_len = 0

    def admit_pending(next_motion: float) -> None:
        """Rank the held-back frame now that the motion after it is known."""
        nonlocal pending, steady, quiet
        if pending is None:
            return
        p_motion, p_score, p_sharp, p_frame = pending
        pending = None
        unsteady = max(p_motion, next_motion)
        if unsteady < QUIET_GATE:
            if len(steady) < CANDIDATE_POOL or p_score > steady[0][0]:
                steady.append((p_score, p_sharp, p_frame))
                steady.sort(key=lambda c: c[0])   # worst first, like the others
                if len(steady) > CANDIDATE_POOL:
                    steady.pop(0)
        if len(quiet) < QUIET_POOL or unsteady < quiet[0][0]:
            quiet.append((unsteady, p_score, p_sharp, p_frame))
            quiet.sort(key=lambda c: -c[0])      # worst first, like the others
            if len(quiet) > QUIET_POOL:
                quiet.pop(0)

    def flush_run() -> None:
        nonlocal candidates, settled, steady, quiet, pending, run_len
        if run_len >= min_still_frames and (steady or quiet or settled or candidates):
            # Frames that held still, ranked by the hand-penalised score. The
            # remaining pools are fallbacks: the stillest frames for a hold that
            # never settled below the gate, then the score-ranked pools for a
            # run too short for anything to clear the settle margin — never
            # fewer pages than before, only a better-informed choice among what
            # a long enough run actually offers.
            if steady:
                best_frame, best_sharp = _pick_detectable(steady, min_area_ratio)
            elif quiet:
                best_frame, best_sharp = _pick_quietest(quiet, min_area_ratio)
            else:
                pool = settled if settled else candidates
                best_frame, best_sharp = _pick_detectable(pool, min_area_ratio)
            path = os.path.join(output_dir, f"raw_{len(saved):04d}.png")
            cv2.imwrite(path, best_frame)
            saved.append(path)
            sharpness.append(best_sharp)   # report focus, not the hand-adjusted rank
            held_frames.append(run_len)
        candidates, settled, steady, quiet, pending, run_len = [], [], [], [], None, 0

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if stride > 1 and idx % stride:
            idx += 1
            continue
        idx += 1

        thumb = _thumb(frame)
        motion = 0.0 if prev_thumb is None else float(np.abs(thumb - prev_thumb).mean())
        prev_thumb = thumb
        motions.append(motion)

        # Resolve the previous frame before this one is classified, so that the
        # last frame of a run is rated against the motion that ended the run —
        # which is exactly the lift that ghosts it, and exactly what used to let
        # it win.
        admit_pending(motion)

        if motion < motion_threshold:
            run_len += 1
            score, sharp = _frame_score(frame)
            wants_candidates = len(candidates) < CANDIDATE_POOL or score > candidates[0][0]
            wants_settled = (run_len > SETTLE_MARGIN
                            and (len(settled) < CANDIDATE_POOL or score > settled[0][0]))
            # Unsteadiness is at least this frame's own motion, so a frame that
            # already moves more than the gate — or more than the worst one held
            # — can never make either pool and need not be copied. Without that
            # test a 200-frame hold would copy a 4K frame per iteration for
            # nothing.
            wants_quiet = (run_len > SETTLE_MARGIN
                           and (motion < QUIET_GATE
                                or len(quiet) < QUIET_POOL
                                or motion < quiet[0][0]))
            if wants_candidates or wants_settled or wants_quiet:
                frame_copy = frame.copy()
                if wants_candidates:
                    candidates.append((score, sharp, frame_copy))
                    candidates.sort(key=lambda c: c[0])
                    if len(candidates) > CANDIDATE_POOL:
                        candidates.pop(0)
                if wants_settled:
                    settled.append((score, sharp, frame_copy))
                    settled.sort(key=lambda c: c[0])
                    if len(settled) > CANDIDATE_POOL:
                        settled.pop(0)
                if wants_quiet:
                    pending = (motion, score, sharp, frame_copy)
        else:
            flush_run()

    # The video ended rather than the page moving, so there is no motion after
    # the held frame to hold against it; its own reading is all there is.
    if pending is not None:
        admit_pending(pending[0])
    flush_run()
    cap.release()
    return saved, motions, sharpness, held_frames


def _pick_quietest(
    quiet: list[tuple[float, float, float, np.ndarray]],
    min_area_ratio: float,
) -> tuple[np.ndarray, float]:
    """
    Among a run's least-moving frames, take the stillest one that is usable.

    Sharpness cannot tell ghosting from detail. Motion doubles every edge on the
    page, and a Laplacian variance counts those doubled edges as more detail, so
    a half-ghosted frame routinely outscores a clean one — on this footage the
    frame the sharpness score shipped for one page scored the highest regional
    reading in its entire run while being visibly smeared, and the frames a
    human picked out as the only clean ones in a shaky hold ranked 1st and 2nd
    by motion but 10th and worse by sharpness.

    So the ranking is stillness, measured as the larger of the motion into and
    out of the frame. The old score is still consulted, but only to step over a
    frame with a hand across the text (HAND_PENALTY drives those to zero) or one
    no page can be found in — never to reorder on focus, which is the judgement
    it has been shown to get backwards.
    """
    ordered = sorted(quiet, key=lambda c: c[0])
    for _unsteady, score, sharp, frame in ordered:
        if score > 0 and detect_page(frame, min_area_ratio) is not None:
            return frame, sharp
    # Every still frame has a hand on it: keep the stillest one a page can still
    # be cropped out of rather than reaching back to a moving frame.
    for _unsteady, _score, sharp, frame in ordered:
        if detect_page(frame, min_area_ratio) is not None:
            return frame, sharp
    return ordered[0][3], ordered[0][2]


def _pick_detectable(
    candidates: list[tuple[float, float, np.ndarray]],
    min_area_ratio: float,
) -> tuple[np.ndarray, float]:
    """
    Among a run's best-scoring frames, prefer one where a page is actually found.

    The hand penalty only ever compares scores; nothing about it stops the
    top-scoring frame from being one where detect_page fails entirely, if every
    higher-scoring alternative happens to have a hand over the spine or an edge
    out of frame. Checking a few real candidates for detectability, rather than
    committing to whichever one the score alone preferred, is what stops a
    scoring change here from silently costing a page it could have kept.

    Probed at the same min_area_ratio the main pass will use, since a frame
    that "detects" here under a looser threshold could still fail there.
    """
    for score, sharp, frame in sorted(candidates, key=lambda c: -c[0]):
        if detect_page(frame, min_area_ratio) is not None:
            return frame, sharp
    best = max(candidates, key=lambda c: c[0])
    return best[2], best[1]


# ── Image enhancement ─────────────────────────────────────────────────────────

def text_angle(image: np.ndarray) -> float:
    """
    Angle of the page's text lines, in degrees, or 0.0 when there is no text.

    Rotating the page sweeps its lines of text through horizontal. At the moment
    they are level, every line's ink falls in one row band and the gaps between
    them are empty, so the row totals swing between extremes; at any other angle
    the lines smear across each other and the totals flatten. The angle whose row
    profile varies most is therefore the angle the text sits at.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    if min(h, w) < 40:
        return 0.0
    scale = DESKEW_WIDTH / float(w)
    small = cv2.resize(gray, (DESKEW_WIDTH, max(8, int(h * scale))))
    sh, sw = small.shape[:2]
    core = small[int(sh * 0.1) : int(sh * 0.9), int(sw * 0.1) : int(sw * 0.9)]

    # Take the strokes rather than everything dark. Thresholding on darkness
    # alone reads the desk beside the page as ink, and the desk is broader and
    # blacker than any letter, so its straight edge wins the vote and the page is
    # levelled against the table instead of its own text. A black-hat keeps only
    # what is dark *and* narrower than the kernel, which is letters and not desk.
    stroke = cv2.getStructuringElement(cv2.MORPH_RECT, (DESKEW_STROKE, DESKEW_STROKE))
    strokes = cv2.morphologyEx(core, cv2.MORPH_BLACKHAT, stroke)
    if strokes.max() < 12:
        return 0.0
    ink = (strokes > max(12.0, float(np.percentile(strokes, 99)) * 0.35)).astype(np.float32)
    if ink.mean() < DESKEW_MIN_INK:
        return 0.0  # a blank or near-blank page has no baseline to read

    centre = (core.shape[1] / 2.0, core.shape[0] / 2.0)
    angles = np.arange(-DESKEW_MAX_DEG, DESKEW_MAX_DEG + 1e-9, DESKEW_STEP_DEG)
    variances = []
    for angle in angles:
        M = cv2.getRotationMatrix2D(centre, float(angle), 1.0)
        rotated = cv2.warpAffine(ink, M, (core.shape[1], core.shape[0]))
        variances.append(float(rotated.sum(axis=1).var()))

    # Lines of text give a sharp peak: level them and the row totals swing, tilt
    # them a degree and the swing collapses. A page with no text still produces a
    # winning angle — cloth on a sleeve, wood grain on the desk — but its
    # response is nearly flat, and rotating on that turns a page for no reason.
    peak = float(np.max(variances))
    typical = float(np.median(variances))
    if typical <= 0 or peak / typical < DESKEW_PEAK_RATIO:
        return 0.0
    return float(angles[int(np.argmax(variances))])


def deskew_page(image: np.ndarray) -> np.ndarray:
    """
    Rotate the page so its text sits level.

    Detection can only ever be as square as the outline it found, and some pages
    come through tilted however carefully the corners are fitted — a rotated
    rectangle has equal sides and right-angled corners, so no test on the quad
    can see the lean. The text itself can: it is the one thing on the page whose
    correct orientation is known. Rotation is rigid, so unlike a re-warp this can
    only turn the page, never stretch it.
    """
    angle = text_angle(image)
    if abs(angle) < DESKEW_MIN_DEG:
        return image

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    out_w, out_h = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += out_w / 2.0 - w / 2.0
    M[1, 2] += out_h / 2.0 - h / 2.0
    # Turning the page opens up four wedges that were never photographed. Filling
    # them by repeating the edge pixel drags whatever sat on the border — desk,
    # a fingertip — into long smears that look like scanner damage. Paper-white
    # reads as margin, which is what the corner of a page should look like.
    return cv2.warpAffine(image, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(255, 255, 255))


def enhance_page(image: np.ndarray) -> np.ndarray:
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil = ImageEnhance.Contrast(pil).enhance(1.3)
    pil = ImageEnhance.Sharpness(pil).enhance(1.5)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


# Fewer words than this on the photograph and the witness has nothing to
# compare: a plate, a blank, a cover. The rectified sheet is then taken on
# the model's word alone, which on such pages is the only word there is.
WITNESS_MIN_WORDS = 20


def capture_pages(image: np.ndarray, *, rectify: bool = True, enhance: bool = True,
                  deskew: bool = True, split_spreads: bool = True,
                  workdir: "str | None" = None, min_area_ratio: float = 0.15,
                  ) -> "tuple[list[np.ndarray], dict]":
    """One photograph (already rotated) → its pages, flat and upright.

    The rectifier flattens the whole sheet; a landscape sheet is a spread
    and is split at the ink-free gutter; each page is trimmed back to its
    paper, enhanced, and levelled against its own text. Nothing here
    guesses at page outlines, corner quads or curve models — that was the
    ladder this replaced, and its failures were the worst pages in two
    books.

    The witness is the one gate: tesseract reads the photograph and the
    flat sheet, and a sheet that has lost most of the photograph's words
    (a folded map, a boundary that excluded a page) is refused — the
    photograph is kept for that shot and the build says so. The returned
    dict reports what happened: rectified, split, and why the photograph
    was kept if it was.
    """
    info = {"rectified": False, "split": False, "kept_photo": None}
    sheet = image
    if rectify:
        cand, ok = _rectify.rectify(image)
        if not ok:
            info["kept_photo"] = "no rectifier"
        else:
            sheet = cand
            info["rectified"] = True
            if workdir and shutil.which("tesseract"):
                # No testimony (a timed-out or failed read) is not a
                # verdict: the sheet stands unless both readings exist.
                w_before = _rectify.witness(image, workdir)
                w_after = _rectify.witness(sheet, workdir) if w_before else None
                if w_before and w_after:
                    before, after = w_before[0], w_after[0]
                    if before >= WITNESS_MIN_WORDS and after < _rectify.WITNESS_KEEP_RATIO * before:
                        info.update(rectified=False, kept_photo=(before, after))
                        sheet = image
    # A flat sheet is a spread when it is landscape — but only a photograph
    # that shows a page, and more than a label's worth of one, is split at
    # all. The gutter hunt on a frame of bare desk finds whatever is darkest
    # down the middle, and a cover shot's title label (14% of its frame,
    # where honest pages cover 42% or more) makes a landscape "sheet" of the
    # whole frame whose halves are two pages of table.
    seen = False
    if split_spreads:
        outline = detect_page(image, min_area_ratio)
        seen = outline is not None and cv2.contourArea(
            outline.astype(np.float32)) >= 0.30 * image.shape[0] * image.shape[1]
        if seen and not info["rectified"]:
            # A photograph kept as shot is the whole frame, landscape
            # whatever it holds. Whether it holds a spread is the
            # OUTLINE's shape: a closed book's cover is a portrait
            # outline in a landscape frame, and halving the frame made
            # two pages of half a cover.
            _x, _y, ow, oh = cv2.boundingRect(outline.astype(np.float32))
            seen = oh > 0 and ow / float(oh) >= 1.15
    parts = split_spread(sheet) if seen else [sheet]
    info["split"] = len(parts) > 1
    out = []
    info["squared"] = 0
    for part in parts:
        if info["rectified"]:
            # Trim twice. Before deskew, the surround's bulk comes off
            # while the sheet's edges are still the image's edges. After
            # deskew and squaring the spine's shadow is vertical — leaning
            # with the page it darkened no single column's median and
            # survived on 23 of 130 pages — and the second pass takes it.
            # Neither pass alone does both: the wedges deskew paints into
            # the rotation corners are white, and a trim that only ran
            # after deskew stopped at them on every page that leaned.
            part = _rectify.paper_crop(part)
        if enhance:
            part = enhance_page(part)
        if deskew:
            # Deskew stays for rectified pages too. It was dropped once on
            # the strength of tesseract's count on three inset pages, and
            # the surviving-edge-line count went from 41 to 163 pages on
            # one book: the pages squaring cannot measure — front matter,
            # plates, insets — are exactly the ones nothing else levels,
            # and a leaning surround band is invisible to the trim.
            part = deskew_page(part)
        if info["rectified"]:
            # The residual the rectifier leaves is a trapezoid or a
            # lean; squaring takes both, and levels the surround with
            # the block so the second trim can see it.
            before = part
            part, _conv = _rectify.square(part)
            info["squared"] += part is not before
            part = _rectify.paper_crop(part)
        out.append(part)
    return out, info
