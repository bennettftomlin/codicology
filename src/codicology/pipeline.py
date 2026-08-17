"""
codicology convert — the pipeline.

Turn a book into a clean PDF and an OCR'd EPUB, from a video, a stack of
photographs, or a PDF someone else made: a library scan, a born-digital
manual, or a facsimile this pipeline wrote earlier.

Dependencies (installed with the package):
    opencv-python numpy Pillow img2pdf pypdfium2 pymupdf lxml ebooklib
        pypdfium2   reading pages out of a PDF (--pages-from), and taking
                    figures from the source rather than from a re-render
        pymupdf     writing the searchable text layer (--pdf-text-layer)
        lxml        parsing a printed contents page into a real table of
                    contents

REQUIRED, and not installable with pip:
    tesseract       brew install tesseract  /  apt install tesseract-ocr

    Not an OCR backend here but the witness. It has no generator, so it
    cannot invent: its silence on a page the model read paragraphs from is
    evidence, and two checks rest entirely on it —

        * whether a page with no text layer to consult was invented, where
          it is the ONLY witness (photographs, video, image-only scans);
        * whether a page about to be deleted as blank was merely one this
          pipeline failed to read, which no other check can tell apart —
          and that one applies to every source.

    Without it both simply pass, so the run refuses to start. --without-
    witness overrides that and reports at the end how many pages went
    unexamined. Its word boxes also place the searchable layer far more
    precisely than layout blocks can.

OCR backend:
    surya      pip install surya-ocr

    This pipeline was built around Surya and only Surya. Everything it does
    beyond reading characters — figures, captions, running heads, headings,
    reading order, block geometry — depends on the layout Surya returns.

    The other names --ocr accepts (easyocr, paddleocr, tesseract, gcv) are
    residue. Not one of them has ever been run against a real book here, and
    none is tested. Read the code and it is plain why: each returns FLAT
    TEXT and nothing else, so a book built on one would arrive with no
    figures, no furniture stripped, no headings, no contents, and no note
    links — silently, since nothing checks for it. They are left in place
    because the seam they sit behind is honest, not because they work.
    Treat them as unimplemented.

Usage:
    # a PDF — the usual case. Always pass --ocr-cache: it makes a rebuild
    # cost minutes instead of hours, and every fix means a rebuild.
    codicology convert --pages-from book.pdf --epub book.epub \
        --ocr-cache book.ocr.gz --check-folios --link-notes --title "Book"

    # photographs of the pages, which hold far more detail than video frames
    codicology convert --from-images ./shots book.pdf --epub book.epub \
        --ocr-cache book.ocr.gz

    # a video, and a facsimile PDF that searches like a born-digital file
    codicology convert recording.mp4 book.pdf --pdf-text-layer \
        --epub book.epub --ocr-cache book.ocr.gz

Checking the result — run this, every time:
    codicology verify book.epub source.pdf

    It answers the one question no run log answers: is anything missing? A
    page cached empty, a page deleted as a duplicate, a page whose text was
    swallowed by a drawn border — each shipped silently, each showed up here
    as "the source has words on this page and we have none". It also reports
    whether the source is a scan or born-digital, which decides what the word
    counts mean.

    codicology compare book.epub source.pdf

    Word-by-word disagreement with the source's own text. On a born-digital
    book the layer is the publisher's typesetting and every disagreement is
    our error. On a scan it is another OCR — usually worse than ours — so
    disagreements need a human eye, not a ratio. Beware: matching word COUNTS
    prove only completeness. Two readings of one page differ constantly and
    still count the same.

What happens without being asked:
    - A page rendered from a PDF is stored losslessly, so nothing is thrown
      away before the recogniser or the figure crops see it.
    - Figures are taken from the PDF itself where it holds them, at their
      own resolution, rather than cut out of a re-rendered page. Artwork the
      layout mistook for text — placeholders, un-boxed images, drawn
      diagrams — is recovered from the source and put back.
    - Each figure ships in whichever of PNG or JPEG is smaller, which lands
      lossless on line art and compact on photographs.
    - A page whose text loops, or which claims more words than its ink can
      account for, is read again and then LEFT EMPTY rather than filled with
      invention. On a born-digital page it is restored from the publisher's
      own text instead. A deliberately blank page is a refusal, not a bug —
      the run log says so with [!].
    - Where the source is born-digital, its text is authority: isolated words
      are corrected to match it, and an unreadable page is restored from it.
      Where the source is a scan, its text layer is somebody else's OCR and
      is used only as a witness, never as a correction.

Filming tips (video and --from-images):
    - Hold each page flat and still for ~1-2 seconds, at a steady pace: pages
      are told apart from half-finished turns partly by how long they were held
    - Use even lighting; avoid glare and cast shadows
    - A dark background behind the book improves page detection
    - Keep the camera directly overhead or at a consistent angle
"""

from __future__ import annotations

import argparse
import base64
import glob
import gzip
import hashlib
import json
import html
import io
import os
import re
import shutil
import statistics
from collections import Counter
import sys
import tempfile
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from . import progress


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

# Opposite sides of the page quad may disagree by this much and still be read as
# camera tilt rather than a mis-placed corner. A roughly overhead shot measures
# 0-7% once the corners are fitted from the page edges; an outline that wandered
# onto the reader's fingers measures 16-25%. Do not raise this to accommodate the
# synthetic fixtures, which are posed at a steeper angle than real footage: an
# under-corrected page costs nothing an OCR engine cannot absorb, while trusting
# a bad corner slants every line of text.
MAX_QUAD_SKEW = 0.08
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

# Duplicate detection, on OCR'd text. Five-word runs are long enough that two
# different pages of prose share almost none, and Jaccard stays high across the
# word or two an OCR pass gets wrong between two shots of the same page.
SHINGLE_WORDS = 5
# Checked against the printed page numbers across a whole book: every drop was a
# true repeat, and the repeats scored 0.80 to 0.996 — far enough above this that
# the margin is real rather than lucky. Beware of tuning this against page
# numbers alone: OCR misreads a running head often enough that a kept page can
# look like it is missing when only its header was garbled.
DUPE_SIMILARITY = 0.55
DUPE_WINDOW = 10
MIN_SHINGLES = 12
# How far back to look for a repeat. A spread re-shot three or four times puts
# its copies further apart than the two pages a single re-take would, so a window
# of 6 let a genuine duplicate through at a distance of 8.
# Scores between this and DUPE_SIMILARITY are close enough to be worth a look.
# Measured over a book, almost nothing lands here - which is the point: a
# borderline verdict is rare, so reviewing them costs a glance.
DUPE_BORDERLINE = 0.35
REVIEW_THUMB_WIDTH = 150
# Pages read before the OCR cache is written out again. Small enough that an
# interrupted run loses minutes rather than an hour, large enough that
# re-compressing the file is not what the run spends its time on.
CACHE_CHECKPOINT_PAGES = 25

# A corner may sit this far from a right angle before the quad stops looking like
# a photograph of a rectangle. Genuine tilt bends a corner by a few degrees.
MAX_CORNER_ERROR_DEG = 12.0

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

    Detection runs on a downscaled copy: contour finding on a 24-megapixel phone
    photo is slow, and the edges we want survive downscaling easily.
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
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            if cv2.contourArea(contour) < min_area:
                break
            quad = _corners_from_edges(contour)
            if quad is None or cv2.contourArea(quad) < min_area:
                quad = _quad_from_contour(contour)
                if cv2.contourArea(quad) < min_area:
                    continue
            return _order_corners(quad) / scale
    return None


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
    The gutter is the darkest vertical band near the middle — the shadow where
    the pages meet the spine — measured on a column profile smoothed enough that
    an inter-word gap in a text column cannot win.
    """
    h, w = image.shape[:2]
    if h == 0 or w / float(h) < min_aspect:
        return [image]

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
    x = lo + int(np.argmin(smooth[lo:hi]))

    # A spine in view is a pronounced shadow. When the crop caught only part of
    # the spread the profile across the middle is flat instead, and argmin picks
    # an arbitrary column — often pinned to the window edge — which would slice a
    # text column in half. Halving the crop is wrong there too, but it is wrong
    # by a predictable margin instead of severing a paragraph mid-word.
    if float(np.median(smooth[left:right])) - float(smooth[x]) < GUTTER_MIN_DEPTH:
        x = left + inner // 2
    return [image[:, :x], image[:, x:]]


def _quad_corner_error(corners: np.ndarray) -> float:
    """Largest departure of a corner from a right angle, in degrees."""
    worst = 0.0
    for i in range(4):
        a = corners[i - 1] - corners[i]
        b = corners[(i + 1) % 4] - corners[i]
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-6 or nb < 1e-6:
            continue
        cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
        worst = max(worst, abs(float(np.degrees(np.arccos(cos))) - 90.0))
    return worst


def _quad_unsafe(corners: np.ndarray) -> bool:
    """
    Whether warping through this quad would distort the page rather than flatten it.

    Two different failures, and neither metric sees the other. Comparing the
    lengths of opposite sides catches a trapezoid but scores a parallelogram at
    zero, because its opposite sides are equal however far it leans. Comparing
    corners against a right angle catches the lean but not the taper.
    """
    return (_quad_skew(corners) > MAX_QUAD_SKEW
            or _quad_corner_error(corners) > MAX_CORNER_ERROR_DEG)


def _quad_skew(corners: np.ndarray) -> float:
    """How far the quad is from a plausible photo of a rectangle, 0 = perfect."""
    tl, tr, br, bl = corners
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    # A collapsed quad has no meaningful skew; it fails the area check upstream.
    horizontal = abs(top - bottom) / max(top, bottom) if max(top, bottom) else 0.0
    vertical = abs(left - right) / max(left, right) if max(left, right) else 0.0
    return max(horizontal, vertical)


def limit_quad(corners: np.ndarray, cap: float = MAX_QUAD_SKEW) -> np.ndarray:
    """
    Ease an over-skewed quad back toward its bounding box until it is within cap.

    A homography has eight degrees of freedom, so a single corner left on a
    knuckle shears the whole page — every line of text slants — rather than
    merely framing it badly. Correcting real camera tilt is worth keeping, so
    rather than abandon the warp, blend the quad toward the rectangle it should
    resemble by the least amount that brings it back into range. Quads already
    within cap pass through untouched.
    """
    if not _quad_unsafe(corners):
        return corners

    x0, y0 = corners.min(axis=0)
    x1, y1 = corners.max(axis=0)
    box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)

    lo, hi = 0.0, 1.0  # bisect for the gentlest blend that is safe
    for _ in range(24):
        mid = (lo + hi) / 2
        if _quad_unsafe((1 - mid) * corners + mid * box):
            lo = mid
        else:
            hi = mid
    return ((1 - hi) * corners + hi * box).astype(np.float32)


def warp_page(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Perspective-correct the detected page to a flat rectangle."""
    tl, tr, br, bl = corners
    out_w = int(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl)))
    out_h = int(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr)))
    dst = np.array(
        [[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(corners, dst)
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


# ══ OCR backends ══════════════════════════════════════════════════════════════

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


class OCRBackend:
    """Takes a batch of PIL page images, returns one result per page."""

    name = "base"
    batch_size = 1

    def __init__(self, langs: list[str]) -> None:
        self.langs = langs

    def run(self, images: list[Image.Image]) -> list[str]:
        """Plain text, one string per page."""
        raise NotImplementedError

    def run_html(self, images: list[Image.Image]) -> list[str]:
        """XHTML body fragment per page. Backends that recover layout override this."""
        return [
            "".join(f"<p>{html.escape(p)}</p>" for p in reflow_paragraphs(text))
            for text in self.run(images)
        ]

    def run_items(self, images: list[Image.Image]) -> list[list[PageItem]]:
        """
        Page contents in reading order.

        Backends that locate figures override this so illustrations survive into
        the EPUB; the rest degrade to a single markup item per page.
        """
        return [[PageItem(html=body)] for body in self.run_html(images)]


def _lines_to_text(lines: list[tuple[float, float, str]]) -> str:
    """Sort detected line boxes into reading order and join them."""
    # Quantise y so that words on the same visual line don't get split by
    # sub-pixel bbox jitter, then order left-to-right within each line.
    lines.sort(key=lambda r: (round(r[0] / 10.0), r[1]))
    return "\n".join(text for _, _, text in lines if text)


FIGURE_LABELS = frozenset({"Picture", "Figure"})

# A ruled box smaller than this fraction of the page is an initial or an
# ornament; larger than the ceiling it is the page's own border.
BOX_MIN_AREA, BOX_MAX_AREA = 0.015, 0.35
# How empty a ruled box's interior must be to read as a drawing. A chemical
# structure is sparse strokes on white; a boxed paragraph or a filled table
# carries far more ink than this.
BOX_MAX_INK = 0.20


def find_ruled_boxes(image: Image.Image,
                     claimed: "list[tuple[float, float, float, float]] | None" = None
                     ) -> list[tuple[int, int, int, int]]:
    """
    Rectangles drawn on the page that the layout model never made blocks of.

    TiHKAL sets every chemical structure inside a ruled box with body text
    wrapped around it, and the layout pass swallows the whole region into the
    text block: no Picture, no garbage, nothing — fifty-five structure drawings
    silently absent from the book. The box border itself is the reliable
    signal: a closed dark rectangle of plausible size whose interior is mostly
    white. That is a contour question, not a layout question.

    `claimed` lists regions (x0, y0, x1, y1) already emitted as figures, so a
    framed photograph is not added twice.
    """
    g = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = g.shape[:2]
    ink = (g < 200).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(w * h)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh / page_area
        if not BOX_MIN_AREA <= area <= BOX_MAX_AREA:
            continue
        if not 0.25 <= bw / max(1, bh) <= 6.0:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        # the border must actually be drawn on all four sides
        t = 3
        sides = [ink[y:y + t, x:x + bw], ink[y + bh - t:y + bh, x:x + bw],
                 ink[y:y + bh, x:x + t], ink[y:y + bh, x + bw - t:x + bw]]
        # 0.3, not higher: the rule is one drawn line and anti-aliasing puts
        # some of it just outside the contour's bounding strip — a real box on
        # a real page measured 0.496 on its weakest side and was rejected at
        # 0.5. Prose never forms a four-cornered contour of this size at all,
        # so the corner and hollowness tests carry the discrimination.
        if min(float(sd.mean()) for sd in sides if sd.size) < 0.3:
            continue
        interior = ink[y + t * 2:y + bh - t * 2, x + t * 2:x + bw - t * 2]
        if interior.size == 0 or float(interior.mean()) > BOX_MAX_INK:
            continue
        if any(not (x1 <= x or x0 >= x + bw or y1 <= y or y0 >= y + bh)
               for x0, y0, x1, y1 in (claimed or [])):
            continue
        boxes.append((x, y, x + bw, y + bh))
    # nested rectangles (a rule drawn double) collapse to the outermost
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes:
        if not any(b[0] >= k[0] - 6 and b[1] >= k[1] - 6
                   and b[2] <= k[2] + 6 and b[3] <= k[3] + 6 for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: b[1])
CAPTION_LABELS = frozenset({"Caption"})
FURNITURE_LABELS = frozenset({"PageHeader", "PageFooter"})

# A running head that the layout pass missed, recognised by its shape instead:
# a short line at the very top or bottom of a page, carrying a page number at
# one end. Kept deliberately narrow — it only ever examines the first and last
# fragment of a page, so a numeral opening a sentence mid-page is never at risk.
RUNNING_HEAD = re.compile(r"""
    ^\s*(?: \d{1,3}\s*[•·|:]          # "28 • Citadel of Sin"
         |  \d{1,3}\s*$               # a bare folio on its own line
    )""", re.X)
RUNNING_FOOT = re.compile(r"[•·|:]\s*\d{1,3}\s*$")   # "Chapter 13 • 199"
RUNNING_HEAD_MAX_CHARS = 60


def _crop_block(image: Image.Image, block, page) -> Image.Image | None:
    """
    Cut a detected block's polygon out of the page scan.

    Taken exactly, with no padding: blocks sit only a few pixels apart, so even a
    small margin pulls the running header or the caption into the figure.
    """
    poly = getattr(block, "polygon", None)
    if not poly:
        return None
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]

    # Block coordinates are in the frame the model saw, which need not match the
    # scan we hold, so rescale via the reported page box.
    box = getattr(page, "image_bbox", None) or [0, 0, image.width, image.height]
    sx = image.width / max(1e-6, float(box[2]) - float(box[0]))
    sy = image.height / max(1e-6, float(box[3]) - float(box[1]))

    x0 = int(max(0, min(xs) * sx))
    y0 = int(max(0, min(ys) * sy))
    x1 = int(min(image.width, max(xs) * sx))
    y1 = int(min(image.height, max(ys) * sy))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    # crop() is lazy; copy so the result survives closing the source image.
    return image.crop((x0, y0, x1, y1)).copy()


# A picture the layout declined to call one. Surya sometimes returns a block
# whose entire content is an <img> with no source: it has seen artwork and
# labelled the region text. The caption beside it survives, so the book shows
# "Figure 6.19" with nothing underneath — sixteen figures across four books
# were lost this way, silently, in every build since the beginning. The block
# carries its own box, so the picture can be cut from the page after the fact,
# which means every cache already on disk can be repaired without re-reading.
PLACEHOLDER_IMG = re.compile(r"<img(?![^>]*\bsrc=)[^>]*>")


def _picture_signature(data: bytes) -> "tuple | None":
    """A coarse fingerprint of what a picture LOOKS like, not what it weighs.

    Size alone would be reckless: a field manual draws every diagram to one
    column width, so matching on dimensions would collapse hundreds of
    distinct figures into one. Two images match here only when the same
    light-and-dark pattern survives being reduced to a 16x32 thumbnail —
    which a decorative band repeated down a margin does, and two different
    diagrams of the same width do not.
    """
    try:
        with Image.open(io.BytesIO(data)) as im:
            small = im.convert("L").resize((16, 32))
            px = list(small.getdata())
    except Exception:
        return None
    if not px:
        return None
    avg = sum(px) / len(px)
    return (len(px), tuple(p > avg for p in px))


# How many separate pages must carry the same picture before it stops being a
# figure and becomes decoration. Two is a book showing one thing twice, which
# happens and is content — a chart repeated for reference, a plate reproduced
# in a later chapter. Three is a pattern.
FURNITURE_PAGES = 3


def find_image_furniture(page_figures: list[list[str]],
                         figure_data: dict) -> set:
    """The pictures that are page furniture rather than figures.

    A running head is not content and the pipeline has always known it; the
    same is true of the ornament printed down the margin of every chapter
    opening, which is a running head that happens to be drawn rather than
    typeset. It was being recovered as artwork — one book carried the same
    decorative band eight times over, 2.5MB of a 7MB file, sitting with the
    folio numbers it was printed beside.

    Recurrence is what separates them, and it is the book's own evidence: a
    figure illustrates its page and appears there. Anything that appears
    identically on three or more pages is furniture, and no unique picture
    can be caught by this however it is shaped or wherever it sits.
    """
    by_sig: dict = {}
    for page_no, names in enumerate(page_figures):
        for name in names:
            data = figure_data.get(name)
            if data is None:
                continue
            sig = _picture_signature(data)
            if sig is None:
                continue
            entry = by_sig.setdefault(sig, ([], set()))
            entry[0].append(name)
            entry[1].add(page_no)
    doomed = set()
    for names, pages in by_sig.values():
        if len(names) >= FURNITURE_PAGES and len(pages) >= FURNITURE_PAGES:
            doomed.update(names)
    return doomed


def recover_placeholder_figures(items: list, page_path: str) -> list:
    """Turn source-less image placeholders back into the pictures they mark."""
    if not any(it.figure is None and it.html and it.box
               and PLACEHOLDER_IMG.search(it.html) for it in items):
        return items
    try:
        page = Image.open(page_path)
        page.load()
    except Exception:
        return items
    out = []
    for it in items:
        if not (it.figure is None and it.html and it.box
                and PLACEHOLDER_IMG.search(it.html)):
            out.append(it)
            continue
        rest = _strip_tags(PLACEHOLDER_IMG.sub("", it.html)).strip()
        if not rest:
            x0, y0 = int(it.box[0] * page.width), int(it.box[1] * page.height)
            x1, y1 = int(it.box[2] * page.width), int(it.box[3] * page.height)
            if x1 - x0 >= 16 and y1 - y0 >= 16:
                out.append(PageItem(figure=page.crop((x0, y0, x1, y1)).copy(),
                                    box=it.box, conf=it.conf))
                continue
        # a placeholder sitting among real words: drop the empty tag, which
        # renders as a broken image in every reader, and keep the text
        out.append(it._replace(html=PLACEHOLDER_IMG.sub("", it.html)))
    page.close()
    return out


def recover_orphan_artwork(items: list, page_path: str) -> list:
    """
    Artwork the layout never boxed at all.

    A figure is normally found because the recognition labelled that region a
    picture. When it labels the region text instead — which lossless pages
    made it do on label-dense diagrams — the artwork is not merely mis-cropped,
    it is absent: a first-aid manual lost the pressure-point diagram off
    thirteen pages while gaining the prose around it. The PDF still holds the
    picture, and its placement says exactly where it belongs, so anything the
    emitted figures do not already cover can be taken directly.
    """
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile"):
            return items            # the page IS the picture; nothing orphaned
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(src["pdf"])
        page = doc[src["page"]]
        pw, ph = page.get_size()
    except Exception:
        return items

    covered = [it.box for it in items if it.figure is not None and it.box]
    orphans = []
    try:
        for obj in page.get_objects():
            if not isinstance(obj, pdfium.PdfImage):
                continue
            b = obj.get_bounds()
            placed = (b[0] / pw, 1 - b[3] / ph, b[2] / pw, 1 - b[1] / ph)
            area = (placed[2] - placed[0]) * (placed[3] - placed[1])
            if not 0.02 <= area <= 0.9:
                continue            # a logo, or the whole page
            if any(_box_iou(placed, c) >= 0.3 for c in covered):
                continue            # already emitted
            orphans.append((placed, obj))
    except Exception:
        return items
    if not orphans:
        return items

    out = list(items)
    for placed, obj in orphans:
        art = None
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                stem = os.path.join(td, "orphan")
                obj.extract(stem)
                found = next(iter(sorted(glob.glob(stem + ".*"))), None)
                if found:
                    with Image.open(found) as native:
                        native.load()
                        art = native.convert("RGB")
        except Exception:
            art = None
        ref = _render_region(page, placed, art.width if art else 800)
        if art is None or ref is None or not _looks_like_same_picture(art, ref):
            art = ref               # the page as drawn, if the stored pixels
        if art is None:             # cannot be confirmed
            continue
        # slot it in by where it sits, so it reads in the right place
        at = len(out)
        for k, it in enumerate(out):
            if it.box and it.box[1] > placed[1]:
                at = k
                break
        out.insert(at, PageItem(figure=art, box=placed))
    return out


def recover_vector_artwork(items: list, page_path: str) -> list:
    """
    Drawings made of ink rather than pixels.

    A manual's diagrams are often DRAWN — hundreds of path objects, no
    embedded image anywhere — so when the layout reads the region as text,
    there is nothing for orphan recovery to find and the artwork vanishes
    whole: one urban-operations figure of 1,048 paths left only its caption
    behind. The paths themselves say where the drawing is. Dense clusters of
    them that no emitted figure covers are rendered straight from the page,
    which for vector art is the best possible copy.
    """
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile"):
            return items
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(src["pdf"])
        page = doc[src["page"]]
        pw, ph = page.get_size()
    except Exception:
        return items

    boxes = []
    try:
        for obj in page.get_objects():
            if obj.type != 2:                  # paths only
                continue
            b = obj.get_bounds()
            w, h = b[2] - b[0], b[3] - b[1]
            if w > 0.8 * pw and h < 3:
                continue                        # the running head's rule
            boxes.append(b)
    except Exception:
        return items
    if len(boxes) < 15:
        return items

    # union-find nearby boxes into clusters
    parent = list(range(len(boxes)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    PAD = 8
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b2 = boxes[i], boxes[j]
            if (a[0] - PAD <= b2[2] and b2[0] - PAD <= a[2]
                    and a[1] - PAD <= b2[3] and b2[1] - PAD <= a[3]):
                parent[find(i)] = find(j)
    groups: dict = {}
    for i in range(len(boxes)):
        groups.setdefault(find(i), []).append(boxes[i])

    covered = [it.box for it in items if it.figure is not None and it.box]
    tabled = [it.box for it in items
              if it.html and "<table" in it.html and it.box]
    out = list(items)
    for members in groups.values():
        if len(members) < 15:
            continue
        x0 = min(b[0] for b in members); y0 = min(b[1] for b in members)
        x1 = max(b[2] for b in members); y1 = max(b[3] for b in members)
        placed = (x0 / pw, 1 - y1 / ph, x1 / pw, 1 - y0 / ph)
        area = (placed[2] - placed[0]) * (placed[3] - placed[1])
        if not 0.02 <= area <= 0.9 or (placed[3] - placed[1]) < 0.03:
            continue
        if any(_box_iou(placed, c) >= 0.3 for c in covered):
            continue                            # already a figure
        if any(_box_iou(placed, t) >= 0.3 for t in tabled):
            continue                            # a drawn table, kept as one
        # A cluster that contains real prose is not a diagram — it is a
        # frame drawn around the page's text, and three pages shipped with
        # two words each because the rule below swallowed whole paragraphs
        # as "labels". Labels are short; a paragraph is not.
        inside_items = [it for it in out
                        if it.figure is None and it.box
                        and it.box[0] >= placed[0] - 0.01
                        and it.box[1] >= placed[1] - 0.01
                        and it.box[2] <= placed[2] + 0.01
                        and it.box[3] <= placed[3] + 0.01]
        prose_inside = sum(len(_strip_tags(it.html).split())
                           for it in inside_items if it.html)
        if prose_inside > 40:
            continue
        art = _render_region(page, placed, 1400)
        if art is None:
            continue
        # short labels inside the drawing belong to the drawing
        swallow = {id(it) for it in inside_items
                   if len(_strip_tags(it.html or "").split()) < 15}
        out = [it for it in out if id(it) not in swallow]
        at = len(out)
        for k, it in enumerate(out):
            if it.box and it.box[1] > placed[1]:
                at = k
                break
        out.insert(at, PageItem(figure=art, box=placed))
    return out


def _box_iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / max(1e-9, union)


def _looks_like_same_picture(a: "Image.Image", b: "Image.Image") -> bool:
    """
    Whether two renderings show the same artwork.

    This is the guard that makes taking a figure from the PDF safe. An
    extracted image can lose its transparency mask and come back on black; a
    placement can be rotated or flipped, and the stored pixels are neither; a
    figure assembled from several bitmaps matches none of them. All of those
    fail here, and the crop we already have is kept instead.
    """
    import numpy as np
    try:
        size = (96, 96)
        x = np.asarray(a.convert("L").resize(size), dtype=np.float32).ravel()
        y = np.asarray(b.convert("L").resize(size), dtype=np.float32).ravel()
    except Exception:
        return False
    flat_x, flat_y = x.std() < 1e-3, y.std() < 1e-3
    if flat_x != flat_y:
        # one is a blank field and the other is a picture — the shape a lost
        # transparency mask takes when it comes back as solid black
        return False
    if flat_x and flat_y:
        return abs(x.mean() - y.mean()) < 8
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr >= 0.85


def _render_region(page, box, want_w: int) -> "Image.Image | None":
    """The page as drawn, cropped to a fractional box, about want_w wide."""
    try:
        pw, _ = page.get_size()
        scale = max(0.5, min(8.0, want_w / max(1e-6, (box[2] - box[0]) * pw)))
        full = page.render(scale=scale).to_pil().convert("RGB")
        x0, y0 = int(box[0] * full.width), int(box[1] * full.height)
        x1, y1 = int(box[2] * full.width), int(box[3] * full.height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return full.crop((x0, y0, x1, y1)).copy()
    except Exception:
        return None


def better_figure(page_path: str, box, crop: "Image.Image") -> "Image.Image":
    """
    The best available rendering of a figure: the PDF's own image where one
    sits exactly there, a fresh lossless render of that region otherwise, and
    the raster crop when neither can be confirmed.

    Measured on two books, 77% and 65% of figures correspond one-to-one with
    an embedded image, and taking those at their native size skips an upscale
    of 3.12x and 1.82x respectively — the interpolation that made line art
    look harsh — along with a generation of JPEG.
    """
    if box is None or crop is None:
        return crop
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(src["pdf"])
        page = doc[src["page"]]
        pw, ph = page.get_size()
    except Exception:
        return crop

    best = None
    try:
        for obj in page.get_objects():
            if not isinstance(obj, pdfium.PdfImage):
                continue
            b = obj.get_bounds()
            placed = (b[0] / pw, 1 - b[3] / ph, b[2] / pw, 1 - b[1] / ph)
            score = _box_iou(box, placed)
            if score >= 0.6 and (best is None or score > best[0]):
                best = (score, obj, placed)
    except Exception:
        best = None

    if best is not None:
        try:
            import tempfile
            with tempfile.TemporaryDirectory() as td:
                stem = os.path.join(td, "fig")
                best[1].extract(stem)
                found = next(iter(sorted(glob.glob(stem + ".*"))), None)
                if found:
                    with Image.open(found) as native:
                        native.load()
                        native = native.convert("RGB")
                    # Confirm against the page as DRAWN in this image's own
                    # footprint, not against the layout block's crop. The two
                    # framings differ — a block is often tighter than the
                    # picture it holds, and comparing a whole drawing against
                    # a clipped one rejects a perfectly good pass-through.
                    ref = _render_region(page, best[2], native.width)
                    if ref is not None and _looks_like_same_picture(native, ref):
                        crop.close()
                        return native
        except Exception:
            pass

    # No embedded image to take, or it could not be confirmed: render this
    # region straight from the page at the crop's own resolution. Vector
    # artwork gains from being drawn rather than resampled, and either way
    # the figure never passes through the page raster's JPEG.
    if src.get("facsimile"):
        # The page IS a photograph — this pipeline's own output, or a scan
        # stored whole. Rendering a region of it only resamples pixels the
        # crop already holds exactly, so the crop is the better copy.
        return crop
    fresh = _render_region(page, box, max(1, crop.width))
    if fresh is not None and _looks_like_same_picture(fresh, crop):
        crop.close()
        return fresh
    return crop


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


def _to_xhtml(fragment: str) -> str:
    """Reparse model-generated HTML into well-formed XHTML for the EPUB."""
    try:
        from lxml import etree
        from lxml import html as lhtml

        div = lhtml.fragment_fromstring(fragment, create_parent="div")
    except Exception:
        return f"<p>{html.escape(_strip_tags(fragment))}</p>"

    # Loose text needs a block wrapper; a bare text node in <body> isn't valid XHTML.
    parts = [f"<p>{html.escape(div.text.strip())}</p>"] if (div.text or "").strip() else []
    parts += [etree.tostring(child, method="xml", encoding="unicode") for child in div]
    return "".join(parts)


# Surya's llama.cpp backend sizes the server's KV cache as parallel-slots x
# 12k tokens, and matches its client workers to the same number — but its
# default of 8 slots is double the 4-page batches this pipeline ever sends.
# The surplus is pure memory: ~48k tokens of idle KV cache, which is exactly
# the pressure that made grammar initialisation start failing when two OCR
# stacks ran at once. Sized to the work, set before surya reads its settings.
os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "4")


class SuryaBackend(OCRBackend):
    """
    Surya's API has shifted repeatedly across releases, so we probe for the
    shape that's installed:
      - 0.6-0.2x: surya.recognition.RecognitionPredictor, called as rec(images).
        Detection/layout run internally and results come back as layout blocks
        carrying HTML, ordered by `reading_order`.
      - older:    surya.ocr.run_ocr(...) with separately loaded models, returning
        flat `text_lines` that we sort by bounding box.

    Note: 0.2x delegates inference to llama.cpp and needs a `llama-server`
    binary on PATH (`brew install llama.cpp`), or LLAMA_CPP_BINARY set.
    """

    name = "surya"
    batch_size = 4

    def __init__(self, langs: list[str]) -> None:
        super().__init__(langs)
        self._shape: str | None = None
        self._load()

    def _load(self) -> None:
        try:
            from surya.recognition import RecognitionPredictor
        except ImportError:
            pass
        else:
            self._rec = RecognitionPredictor()
            self._shape = "predictor"
            return

        try:
            from surya.model.detection.segformer import (
                load_model as load_det_model,
                load_processor as load_det_processor,
            )
            from surya.model.recognition.model import load_model as load_rec_model
            from surya.model.recognition.processor import load_processor as load_rec_processor
            from surya.ocr import run_ocr
        except ImportError as exc:
            raise RuntimeError(
                "Could not find a supported Surya API. Install with `pip install surya-ocr`.\n"
                "If Surya is installed but this still fails, its API has likely changed "
                "again — check the installed version with\n"
                "  python -c \"import importlib.metadata as m; print(m.version('surya-ocr'))\"\n"
                "then adjust SuryaBackend._load()/_predict().\n"
                f"Underlying import error: {exc}"
            ) from exc

        self._run_ocr = run_ocr
        self._det_model, self._det_processor = load_det_model(), load_det_processor()
        self._rec_model, self._rec_processor = load_rec_model(), load_rec_processor()
        self._shape = "run_ocr"

    def _predict(self, images: list[Image.Image]):
        if self._shape == "run_ocr":
            return self._run_ocr(
                images,
                [self.langs] * len(images),
                self._det_model,
                self._det_processor,
                self._rec_model,
                self._rec_processor,
            )
        return self._rec(images)

    def _page_fragments(self, images: list[Image.Image]) -> list[list[str]]:
        """Per page, the usable HTML fragments in reading order."""
        pages: list[list[str]] = []
        for pred in self._predict(images):
            blocks = getattr(pred, "blocks", None)
            if blocks is None:
                # Legacy run_ocr shape: flat text lines positioned by bbox.
                lines = []
                for line in getattr(pred, "text_lines", []) or []:
                    bbox = getattr(line, "bbox", None) or [0.0, 0.0, 0.0, 0.0]
                    lines.append((float(bbox[1]), float(bbox[0]), getattr(line, "text", "")))
                text = _lines_to_text(lines)
                pages.append([f"<p>{html.escape(p)}</p>" for p in reflow_paragraphs(text)])
                continue

            ordered = sorted(blocks, key=lambda b: getattr(b, "reading_order", 0))
            pages.append([
                b.html for b in ordered
                if getattr(b, "html", "") and not getattr(b, "skipped", False)
                and not getattr(b, "error", False)
            ])
        return pages

    def run(self, images: list[Image.Image]) -> list[str]:
        return [
            "\n\n".join(filter(None, (_strip_tags(f) for f in frags)))
            for frags in self._page_fragments(images)
        ]

    def run_html(self, images: list[Image.Image]) -> list[str]:
        return ["".join(_to_xhtml(f) for f in frags) for frags in self._page_fragments(images)]

    def run_items(self, images: list[Image.Image]) -> list[list[PageItem]]:
        pages: list[list[PageItem]] = []
        for image, pred in zip(images, self._predict(images)):
            blocks = getattr(pred, "blocks", None)
            if blocks is None:
                pages.append([PageItem(html=frag)
                              for frag in self._page_fragments([image])[0]])
                continue

            items: list[PageItem] = []
            tops: list[float] = []          # each item's top edge, for box placement
            claimed: list[tuple[float, float, float, float]] = []
            box = getattr(pred, "image_bbox", None) or [0, 0, image.width, image.height]
            sx = image.width / max(1e-6, float(box[2]) - float(box[0]))
            sy = image.height / max(1e-6, float(box[3]) - float(box[1]))

            def block_top(b) -> float:
                poly = getattr(b, "polygon", None)
                return min(float(q[1]) for q in poly) * sy if poly else 0.0

            def block_box(b) -> tuple | None:
                poly = getattr(b, "polygon", None)
                if not poly:
                    return None
                xs = [float(q[0]) * sx for q in poly]
                ys = [float(q[1]) * sy for q in poly]
                return (min(xs) / image.width, min(ys) / image.height,
                        max(xs) / image.width, max(ys) / image.height)

            for b in sorted(blocks, key=lambda b: getattr(b, "reading_order", 0)):
                label = getattr(b, "label", "")
                if label in FURNITURE_LABELS:
                    # A running head belongs to the paper, not to the book — it
                    # never reaches the body. But its text is the folio audit's
                    # raw material, read here at full resolution with the whole
                    # page for context, so it rides along flagged instead of
                    # being discarded and re-derived later at lower quality.
                    frag = getattr(b, "html", "")
                    if frag and not getattr(b, "skipped", False) \
                            and not getattr(b, "error", False):
                        items.append(PageItem(html=_to_xhtml(frag),
                                              is_furniture=True,
                                              box=block_box(b),
                                              conf=getattr(b, "confidence", None),
                                              label=label or None))
                        tops.append(block_top(b))
                    continue
                if label in FIGURE_LABELS:
                    # Figures come back with skipped=True and no HTML — there is
                    # no text to read — so the pixels are the only content.
                    crop = _crop_block(image, b, pred)
                    if crop is not None:
                        items.append(PageItem(figure=crop, box=block_box(b),
                                              conf=getattr(b, "confidence", None),
                                              label=label or None))
                        tops.append(block_top(b))
                        poly = getattr(b, "polygon", None)
                        if poly:
                            xs = [float(q[0]) * sx for q in poly]
                            ys = [float(q[1]) * sy for q in poly]
                            claimed.append((min(xs), min(ys), max(xs), max(ys)))
                    continue
                frag = getattr(b, "html", "")
                if not frag or getattr(b, "skipped", False) or getattr(b, "error", False):
                    continue
                items.append(PageItem(html=_to_xhtml(frag),
                                      is_caption=label in CAPTION_LABELS,
                                      box=block_box(b),
                                      conf=getattr(b, "confidence", None),
                                      label=label or None))
                tops.append(block_top(b))

            # Drawings the layout never segmented — a boxed structure with text
            # wrapped around it leaves no block of its own, so the border is
            # found directly and the crop slotted in by where it sits on the
            # page, after whatever text stands above it.
            for (x0, y0, x1, y1) in find_ruled_boxes(image, claimed):
                pad = 4
                crop = image.crop((max(0, x0 - pad), max(0, y0 - pad),
                                   min(image.width, x1 + pad),
                                   min(image.height, y1 + pad)))
                at = next((k for k, t in enumerate(tops) if t > y0), len(items))
                items.insert(at, PageItem(figure=crop, box=(
                    x0 / image.width, y0 / image.height,
                    x1 / image.width, y1 / image.height)))
                tops.insert(at, float(y0))
            pages.append(items)
        return pages


class EasyOCRBackend(OCRBackend):
    name = "easyocr"
    batch_size = 1

    def __init__(self, langs: list[str]) -> None:
        super().__init__(langs)
        import easyocr

        self._reader = easyocr.Reader(langs)

    def run(self, images: list[Image.Image]) -> list[str]:
        pages = []
        for img in images:
            lines = []
            for box, text, _conf in self._reader.readtext(np.array(img)):
                ys = [p[1] for p in box]
                xs = [p[0] for p in box]
                lines.append((float(min(ys)), float(min(xs)), text))
            pages.append(_lines_to_text(lines))
        return pages


class PaddleOCRBackend(OCRBackend):
    name = "paddleocr"
    batch_size = 1

    def __init__(self, langs: list[str]) -> None:
        super().__init__(langs)
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(use_angle_cls=True, lang=langs[0], show_log=False)

    def run(self, images: list[Image.Image]) -> list[str]:
        pages = []
        for img in images:
            result = self._ocr.ocr(np.array(img), cls=True) or []
            lines = []
            for block in result:
                for box, (text, _conf) in block or []:
                    ys = [p[1] for p in box]
                    xs = [p[0] for p in box]
                    lines.append((float(min(ys)), float(min(xs)), text))
            pages.append(_lines_to_text(lines))
        return pages


class TesseractBackend(OCRBackend):
    name = "tesseract"
    batch_size = 1

    def __init__(self, langs: list[str]) -> None:
        super().__init__(langs)
        import pytesseract

        self._pytesseract = pytesseract
        self._lang = "+".join(langs)

    def run(self, images: list[Image.Image]) -> list[str]:
        return [self._pytesseract.image_to_string(img, lang=self._lang) for img in images]


class GoogleVisionBackend(OCRBackend):
    name = "gcv"
    batch_size = 1

    def __init__(self, langs: list[str]) -> None:
        super().__init__(langs)
        from google.cloud import vision

        self._vision = vision
        self._client = vision.ImageAnnotatorClient()

    def run(self, images: list[Image.Image]) -> list[str]:
        import io

        pages = []
        for img in images:
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=92)
            response = self._client.document_text_detection(
                image=self._vision.Image(content=buf.getvalue())
            )
            if response.error.message:
                raise RuntimeError(f"Google Vision error: {response.error.message}")
            pages.append(response.full_text_annotation.text)
        return pages


OCR_BACKENDS: dict[str, type[OCRBackend]] = {
    "surya": SuryaBackend,
    "easyocr": EasyOCRBackend,
    "paddleocr": PaddleOCRBackend,
    "tesseract": TesseractBackend,
    "gcv": GoogleVisionBackend,
}


def load_backend(name: str, langs: list[str]) -> OCRBackend:
    if name != "surya":
        # Said once, plainly, at the moment it stops being hypothetical.
        print(f"[!] --ocr {name}: this pipeline was built around Surya and has "
              f"never been run with anything else. {name} returns flat text "
              f"with no layout, so the book will have no figures, no headings, "
              f"no contents and no note links, and nothing downstream will "
              f"notice. This path is untested.")
    try:
        return OCR_BACKENDS[name](langs)
    except ImportError as exc:
        sys.exit(f"OCR backend '{name}' is not installed: {exc}\nSee the header of this file for install commands.")
    except RuntimeError as exc:
        sys.exit(str(exc))


# ── Text reflow for EPUB ──────────────────────────────────────────────────────

def reflow_paragraphs(text: str) -> list[str]:
    """Join OCR'd lines back into paragraphs so the EPUB reflows properly."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    # A line noticeably shorter than the page's typical line usually ends a
    # paragraph; full-width lines are mid-paragraph wraps.
    typical = statistics.median(len(ln) for ln in lines)
    paragraphs: list[str] = []
    buf = ""

    for line in lines:
        if buf.endswith("-"):
            buf = buf[:-1] + line
        elif buf:
            buf += " " + line
        else:
            buf = line
        if len(line) < 0.75 * typical:
            paragraphs.append(buf)
            buf = ""

    if buf:
        paragraphs.append(buf)
    return paragraphs


# ── Duplicate pages ───────────────────────────────────────────────────────────

def _shingles(text: str, k: int = SHINGLE_WORDS) -> set[str]:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def strip_running_heads(items: list[PageItem]) -> list[PageItem]:
    """
    Drop the running head and folio from the ends of a page.

    The layout pass labels most of them and they never reach here. It misses
    some, and those are worth catching by shape, because a running head is the
    one line on the page guaranteed to be wrong in an EPUB: the text reflows,
    the page it numbered no longer exists, and the title interrupts a sentence
    that ran across the page break.
    """
    if not items:
        return items
    keep = list(items)
    for end in (0, -1):
        while keep:
            item = keep[end]
            if item.figure is not None or item.is_caption:
                break
            text = _strip_tags(item.html).strip()
            if not text or len(text) > RUNNING_HEAD_MAX_CHARS:
                break
            # Either shape can appear at either end: this book numbers versos at
            # the front of the line ("28 • Citadel of Sin") and rectos at the
            # back ("Chapter 13 • 199"), and both sit at the top of the page.
            if not (RUNNING_HEAD.match(text) or RUNNING_FOOT.search(text)):
                break
            keep.pop(end)
    return keep


class PageVerdict(NamedTuple):
    """What the duplicate pass concluded about one page, and on what evidence."""
    index: int
    status: str          # kept | duplicate | unjudged | borderline
    match: int | None    # the page it was measured against
    score: float


def review_pages(texts: list[str], window: int = DUPE_WINDOW) -> list[PageVerdict]:
    """
    Judge every page, recording the reasoning rather than only the conclusion.

    Two of these verdicts deserve a second look and the numbers say which. Across
    a whole book the scores pile up at the ends — near zero for a page unlike
    anything before it, above 0.8 for a page shot twice — and the middle is all
    but empty, so a *borderline* score is rare and genuinely close. Far more
    common is *unjudged*: a page holding too few words to compare at all, which
    is what a hand across the paper leaves behind. Those are invisible to this
    pass and obvious to a reader, so they are worth surfacing, not silently
    keeping.
    """
    fingerprints = [_shingles(t) for t in texts]
    verdicts: dict[int, PageVerdict] = {}
    duplicates: set[int] = set()

    for i in range(len(texts)):
        if len(fingerprints[i]) < MIN_SHINGLES:
            verdicts[i] = PageVerdict(i, "unjudged", None, 0.0)
            continue

        best_score, best_j = 0.0, None
        for j in range(max(0, i - window), i):
            if j in duplicates or len(fingerprints[j]) < MIN_SHINGLES:
                continue
            smaller = min(len(fingerprints[i]), len(fingerprints[j]))
            if not smaller:
                continue
            score = len(fingerprints[i] & fingerprints[j]) / smaller
            if score > best_score:
                best_score, best_j = score, j

        if best_j is not None and best_score >= DUPE_SIMILARITY:
            loser = i if len(texts[i]) <= len(texts[best_j]) else best_j
            keeper = best_j if loser == i else i
            duplicates.add(loser)
            verdicts[loser] = PageVerdict(loser, "duplicate", keeper, best_score)
            if loser != i:
                verdicts[i] = PageVerdict(i, "kept", best_j, best_score)
        elif best_j is not None and best_score >= DUPE_BORDERLINE:
            verdicts[i] = PageVerdict(i, "borderline", best_j, best_score)
        else:
            verdicts[i] = PageVerdict(i, "kept", best_j, best_score)

    return [verdicts[i] for i in range(len(texts))]


def find_duplicate_pages(texts: list[str], window: int = DUPE_WINDOW) -> set[int]:
    """
    Indices of pages that repeat one already kept, judged on their OCR'd text.

    A spread that is settled, nudged, and settled again is photographed twice,
    and motion segmentation cannot tell that from two separate pages: nothing
    moved differently. Pixels cannot settle it either — that is why detection
    does not try, since two pages of prose look alike once downsampled and a
    pixel rule that is strict enough to catch a re-shoot also deletes real
    pages. The words are the first signal that actually distinguishes them: two
    different pages of prose share almost no five-word runs, while the same page
    twice shares nearly all of them, even through OCR noise.

    Only a short window back is compared. Re-shoots land within a few pages of
    each other, whereas a genuine repeat far later in a book — a part title, a
    blank verso — is not a duplicate and must survive.
    """
    fingerprints = [_shingles(t) for t in texts]
    duplicates: set[int] = set()
    for i in range(len(texts)):
        if len(fingerprints[i]) < MIN_SHINGLES:
            continue  # too little text to judge; a blank verso is not a duplicate
        for j in range(max(0, i - window), i):
            if j in duplicates or len(fingerprints[j]) < MIN_SHINGLES:
                continue
            shared = len(fingerprints[i] & fingerprints[j])
            smaller = min(len(fingerprints[i]), len(fingerprints[j]))
            if not smaller:
                continue
            # Score by how much of the *smaller* page is contained in the larger,
            # not by how much the two have in common overall. A page shot twice
            # is often shot badly once — a hand across a column, a corner curled
            # away — so the poorer read holds only part of the text. Against the
            # union that scores as half a match and the repeat survives; against
            # its own length it is what it is, a fragment of a page already kept.
            if shared / smaller >= DUPE_SIMILARITY:
                # Keep whichever read better; a re-shoot is often the cleaner one.
                duplicates.add(i if len(texts[i]) <= len(texts[j]) else j)
                break
    return duplicates


class OCRCache:
    """
    Remembers what the OCR engine made of each page, so a second opinion about
    duplicates does not cost a second reading of the book.

    Recognising the page is by far the slowest part of this pipeline and by far
    the most settled: the words on a page do not change because the rule for
    discarding repeats did. Everything downstream — which pages are duplicates,
    which the reader struck out, how the book is assembled — is cheap to redo
    and is exactly what gets iterated on.

    Keyed on the bytes of the page image together with the engine and languages,
    so a re-crop or a different backend misses rather than serving a stale read.
    """

    def __init__(self, path: str, backend_name: str, langs: list[str]) -> None:
        self.path = path
        self.tag = f"{backend_name}|{','.join(langs)}"
        self.entries: dict[str, list[dict]] = {}
        self.hits = 0
        self.dirty = False
        self.unsaved = 0
        if os.path.exists(path):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    stored = json.load(fh)
                if stored.get("tag") == self.tag:
                    self.entries = stored.get("pages", {})
            except (OSError, ValueError, KeyError):
                pass  # an unreadable cache is a cold cache, never an error

    def _key(self, page_path: str) -> str:
        with open(page_path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()

    def get(self, page_path: str) -> list[PageItem] | None:
        entry = self.entries.get(self._key(page_path))
        if entry is None:
            return None
        raw = entry["items"] if isinstance(entry, dict) else entry
        self.hits += 1
        items = []
        for it in raw:
            fig = None
            if it.get("fig"):
                fig = Image.open(io.BytesIO(base64.b64decode(it["fig"])))
                fig.load()
            items.append(PageItem(html=it.get("html", ""), figure=fig,
                                  is_caption=it.get("cap", False),
                                  is_furniture=it.get("furn", False),
                                  box=tuple(it["box"]) if it.get("box") else None,
                                  conf=it.get("conf"),
                                  label=it.get("lab")))
        return items

    def knows_furniture(self, page_path: str) -> bool:
        """Whether this entry was written by a version that keeps running heads."""
        return isinstance(self.entries.get(self._key(page_path)), dict)

    def put(self, page_path: str, items: list[PageItem]) -> None:
        raw = []
        for it in items:
            enc = None
            if it.figure is not None:
                buf = io.BytesIO()
                # Higher quality than the copy that reaches the book: this one is
                # encoded again on the way out, and stacking two lossy passes on
                # a halftone photograph is visible where one is not.
                it.figure.convert("RGB").save(buf, "JPEG", quality=97)
                enc = base64.b64encode(buf.getvalue()).decode("ascii")
            raw.append({"html": it.html, "fig": enc, "cap": it.is_caption,
                        "furn": it.is_furniture,
                        "box": list(it.box) if it.box else None,
                        "conf": it.conf, "lab": it.label})
        # The version marker is what tells a page with no running head apart
        # from a page read before heads were kept at all — without it, an old
        # cache would silently audit as a book with no folios anywhere.
        self.entries[self._key(page_path)] = {"v": 3, "items": raw}
        self.dirty = True
        self.unsaved += 1
        # Checkpoint as we go. Reading a book takes the better part of an hour,
        # and a cache written only at the end is a cache that a stopped run
        # leaves with nothing in it — losing exactly the work it exists to save.
        if self.unsaved >= CACHE_CHECKPOINT_PAGES:
            self.save()

    def save(self) -> None:
        if not self.dirty:
            return
        self.unsaved = 0
        tmp = self.path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump({"tag": self.tag, "pages": self.entries}, fh)
        os.replace(tmp, self.path)  # never leave a half-written cache behind


def _thumb_data_uri(path: str, width: int = REVIEW_THUMB_WIDTH) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((width, width * 2))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class Folio(NamedTuple):
    """The printed page number read off one page, and how much to trust it."""
    index: int
    number: int | None
    text: str            # the running head as recognised, for a human to check
    confident: bool


def parse_folio(text: str) -> int | None:
    """
    The page number out of a running head, or None if the line is not one.

    Books put it at either end — "60 - Citadel of Sin" on a verso, "Chapter 7 - 61"
    on a recto — so both are read, but only from a line short enough to be
    furniture and only when the rest of the line carries letters. A bare number
    is rejected: a stray numeral from the body text would otherwise sail through
    as a folio and quietly corrupt the ordering it was meant to check.
    """
    line = " ".join(text.split())
    if not line or len(line) > RUNNING_HEAD_MAX_CHARS:
        return None
    # Number at either end, set off by a separator or by nothing but space:
    # "60 - Citadel of Sin" and "THE NEW CAREER IN THE NEW ERA 51" are the same
    # convention in different dress. Three digits at most, so a year in a head
    # ("CHICAGO IN 1893") can never be read as a folio — four digits simply do
    # not match, and \d{1,3} anchored to a boundary cannot take the tail of one.
    # Manuals number pages by chapter — "4-9" is chapter four, page nine —
    # and that convention carries the same information a plain folio does.
    # Encoded as chapter*1000+page it stays an integer, so every mechanism
    # built on plain folios (the resolver, the audit, the contents mapper)
    # works unchanged; only the audit's idea of a gap needs to know that the
    # jump from 1-27 to 2-1 is a chapter boundary and not 974 missing pages.
    # A reader's own pagination — "Page 303 of 496" — is a folio too: proof of
    # position from the screen rather than the print. It outranks the generic
    # patterns because the same line often ends with a percentage whose digits
    # would otherwise be read as the number.
    pm = re.search(r"\bPage (\d{1,3}) of \d{1,4}\b", line)
    if pm:
        return int(pm.group(1))
    cm = (re.match(r"^(\d{1,2})-(\d{1,3})(?:\s|$)", line)
          or re.search(r"\s(\d{1,2})-(\d{1,3})$", line))
    # "FM 4-25.11" clears a two-letter bar where prose needs three; a lone
    # "1-2" is the whole line and vouches for itself. "60 - 61" matches
    # neither — the compound is written tight, no spaces round the dash.
    # A zero-padded page component is a DESIGNATION, not pagination: no book
    # numbers a page "3-06", but a manual calls ITSELF that in every running
    # head, and one seeded sixty-three false anchors before this test.
    if cm and (re.search(r"[A-Za-z]{2}", line) or line == cm.group(0).strip()):
        if not cm.group(2).startswith("0"):
            ch, pg = int(cm.group(1)), int(cm.group(2))
            if 1 <= ch <= 99 and 1 <= pg <= 999:
                return ch * 1000 + pg
    # Appendix pages paginate by letter — "A-2", "B-14" — the same
    # chapter-page convention with a letter for the chapter. Encoded past
    # every numeric chapter so appendix order sorts after the body.
    am = re.fullmatch(r"([A-Z])-([1-9]\d{0,2})", line.strip())
    if am:
        return (100 + ord(am.group(1)) - ord("A")) * 1000 + int(am.group(2))
    if not re.search(r"[A-Za-z]{3}", line):
        return None
    m = (re.match(r"^(\d{1,3})(?:\s*[•·|:\-]|\s)\s*\S", line)
         or re.search(r"(?:[•·|:\-]\s*|\s)(\d{1,3})$", line))
    if not m:
        return None
    n = int(m.group(1))
    return n if 1 <= n <= 999 else None


def folios_from_furniture(page_furniture: list[list[str]]) -> list[Folio]:
    """Folio records straight from already-recognised running heads and feet."""
    out: list[Folio] = []
    for i, texts in enumerate(page_furniture):
        number, shown = None, ""
        # Rank the candidates instead of taking the first that parses. A page
        # offers its head and its foot together, and the head is sometimes a
        # chapter title that opens with a number — "48 Hours" — which the
        # verso pattern ("60 Citadel of Sin") reads as folio 48 while the real
        # folio sits beside it as "((9". Taking the first match seeded three
        # false anchors into one book and left chapter one unplaceable.
        # A line that is nothing but a numeral outranks a number quarried
        # out of a title.
        best_rank = 99
        for t in texts:
            t = " ".join(t.split())
            if t and not shown:
                shown = t
            n = parse_folio(t)
            rank = 1 if n is not None else 99
            # Folios arrive dressed: "(56)" in one book's parentheses, "((54"
            # where the scan double-struck the paren. The layout pass already
            # vouched this line is furniture, so stripping decoration before
            # the number test cannot promote prose — only undress a folio.
            bare = re.sub(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$", "", t)
            if re.fullmatch(r"[1-9]\d?-[1-9]\d{0,2}", bare):
                ch, pg = bare.split("-")
                n, rank = int(ch) * 1000 + int(pg), 0
            elif re.fullmatch(r"[A-Z]-[1-9]\d{0,2}", bare):
                ch, pg = bare.split("-")
                n, rank = (100 + ord(ch) - ord("A")) * 1000 + int(pg), 0
            elif re.fullmatch(r"[1-9]\d{0,2}", bare):
                n, rank = int(bare), 0
            if n is None and re.fullmatch(r"[1-9]\d{0,2}", t):
                # A bare numeral in running text is shrapnel, but these lines
                # were judged furniture by the layout pass, and a chapter
                # opening's foot is often nothing but the number. The context
                # that makes bare numbers dangerous is exactly what is absent.
                n, rank = int(t), 0
            if n is not None and rank < best_rank:
                number, shown, best_rank = n, t, rank
                if rank == 0:
                    break            # a naked folio needs no second opinion
        out.append(Folio(i, number, shown, number is not None))
    # A folio ascends; a designation is constant. "FM 5-103" parses as
    # chapter 5, page 103 — no zero-padding to give it away — and seeded
    # eighty-one identical anchors. No real page number recurs like that: a
    # scan that shot a leaf twice repeats it twice, not eighty times. Any
    # value this frequent is the book naming itself, whatever its shape.
    from collections import Counter
    freq = Counter(f.number for f in out if f.number is not None)
    constant = {n for n, c in freq.items() if c > 5}
    if constant:
        out = [f if f.number not in constant else
               Folio(f.index, f.number, f.text, False) for f in out]
    return out


def read_folios(page_paths: list[str], backend: "OCRBackend",
                locate_width: int = 1100, batch: int = 8) -> list[Folio]:
    """
    Read each page's printed number from its running head or foot.

    One full-page pass per page, downscaled and batched. Everything narrower was
    tried and failed: reading just the head's strip through the OCR stack comes
    back empty or hallucinated more often than not — the layout stage has no
    page around the ribbon to orient by — and running full pages at full
    resolution once doubled a build. With the whole (small) page in view the
    same models read the same heads correctly, and what the head says arrives
    in the layout blocks the pass produces anyway.

    Each page is read once. The defence against a misread is not a second
    reading — a dropped repeated digit misleads the same way at any size — but
    the sequence itself: audit_folios discards a number its neighbours disown.
    """
    pages: list[Image.Image | None] = []
    for path in page_paths:
        try:
            with Image.open(path) as im:
                page = im.convert("RGB").copy()
        except OSError:
            pages.append(None)
            continue
        if page.width > locate_width:
            f = locate_width / page.width
            page = page.resize((locate_width, max(8, int(page.height * f))),
                               Image.LANCZOS)
        pages.append(page)

    out: list[Folio] = []
    live = [(i, p) for i, p in enumerate(pages) if p is not None]
    results: dict[int, Folio] = {}
    for c0 in range(0, len(live), batch):
        chunk = live[c0:c0 + batch]
        try:
            preds = list(backend._predict([p for _, p in chunk]))
        except Exception:
            preds = [None] * len(chunk)
        for (i, _), pred in zip(chunk, preds):
            texts = [_strip_tags(getattr(b, "html", "") or "").strip()
                     for b in (getattr(pred, "blocks", None) or [])
                     if getattr(b, "label", "") in FURNITURE_LABELS]
            f = folios_from_furniture([texts])[0]
            results[i] = Folio(i, f.number, f.text, f.confident)
        done = min(c0 + batch, len(live))
        print(f"    read {done}/{len(live)} pages…")
        progress.emit(event="progress", phase="folios",
                      done=done, total=len(live))
    for i in range(len(page_paths)):
        out.append(results.get(i, Folio(i, None, "", False)))
    return out


def _ocr_one(image: Image.Image, backend: "OCRBackend") -> str:
    try:
        return _strip_tags(backend.run([image])[0]).strip()
    except Exception:
        return ""


class FolioAudit(NamedTuple):
    """What the printed numbers say about the order of the book."""
    read: int
    total: int
    inversions: list[tuple[int, int]]          # (page index, page index) out of order
    gaps: list[tuple[int, list[int]]]          # (page index after which, numbers absent)
    duplicates: list[tuple[int, list[int]]]    # (folio number, page indices)
    misreads: list[tuple[int, int]]            # (page index, number read but disbelieved)


def audit_folios(folios: list[Folio]) -> FolioAudit:
    """
    Compare the printed numbers against the order the pages are actually in.

    Only confident readings take part. Pages with no running head at all are not
    evidence of anything — a chapter's first page prints none, and neither does
    front matter — so they are stepped over rather than counted as a break. A
    gap is only reported where there are fewer pages between two folios than
    there are missing numbers, since a blank verso the pipeline dropped is a
    legitimate reason for a number to be absent.
    """
    good = [f for f in folios if f.confident and f.number is not None]

    # Numbering domains do not mix. A manual's folios are chapter-page
    # compounds ("1-13"); its chapter openings also carry a large bare digit —
    # the chapter's own number — and its appendices number their pages afresh.
    # Read together they fake an inversion at every chapter break: 1-13, then
    # "2", then 2-1. When compound folios carry the book, a bare number in the
    # furniture is a heading, not a page.
    compound = sum(1 for f in good if f.number >= 1000)
    if compound > len(good) // 2:
        good = [f for f in good if f.number >= 1000]

    # A reading that fits NEITHER neighbour, while its neighbours fit each
    # other, is a misread and not an ordering fault. Both defences upstream can
    # miss the same error: reading "444" as "44" drops a repeated digit the same
    # way at every magnification, so the two readings agree and the confidence
    # gate waves it through. Left in, one such number detonates into an
    # inversion, a duplicate AND a hundreds-long gap — which is exactly what a
    # real book produced. The sequence itself is the third, decisive check.
    misreads: list[tuple[int, int]] = []
    kept: list[Folio] = []
    for k, f in enumerate(good):
        prev = good[k - 1] if k > 0 else None
        nxt = good[k + 1] if k + 1 < len(good) else None
        if prev is not None and nxt is not None:
            span = nxt.index - prev.index
            neighbours_fit = 0 < nxt.number - prev.number <= span * 2 + 2
            fits_between = prev.number < f.number < nxt.number
            # A transposition is not a misread: two pages bound out of order
            # still carry real numbers, and exchanging them repairs the
            # sequence. A misread repairs nothing. Only the irreparable is
            # discarded; the repairable is exactly what the inversion report
            # below exists to say.
            swap_left = prev.number > f.number and f.number > (
                good[k - 2].number if k >= 2 else f.number - 1)
            swap_right = nxt.number < f.number and f.number < (
                good[k + 2].number if k + 2 < len(good) else f.number + 1)
            # Equality with a neighbour is its own finding: the same page
            # captured twice reads the same number twice, and the duplicate
            # report below is where that belongs.
            repeats_neighbour = f.number in (prev.number, nxt.number)
            if (neighbours_fit and not fits_between
                    and not (swap_left or swap_right) and not repeats_neighbour):
                misreads.append((f.index, f.number))
                continue
        kept.append(f)
    good = kept
    inversions: list[tuple[int, int]] = []
    gaps: list[tuple[int, list[int]]] = []
    for a, b in zip(good, good[1:]):
        if b.number <= a.number:
            inversions.append((a.index, b.index))
        elif b.number > a.number + 1:
            if b.number // 1000 != a.number // 1000:
                continue          # a new chapter restarts the count; no gap
            missing = list(range(a.number + 1, b.number))
            between = b.index - a.index - 1
            if len(missing) > between:
                gaps.append((a.index, missing))
    seen: dict[int, list[int]] = {}
    for f in good:
        seen.setdefault(f.number, []).append(f.index)
    duplicates = [(n, idxs) for n, idxs in sorted(seen.items()) if len(idxs) > 1]
    return FolioAudit(len(good), len(folios), inversions, gaps, duplicates, misreads)


def fill_folio_gaps(folios: list[Folio]) -> tuple[dict[int, int], list[tuple]]:
    """
    The numbers the printer never inked, restored where the book's own
    arithmetic supplies them.

    Most pages that read no folio never carried one: chapter openings set
    their number as display type, part-titles and blank versos print nothing
    at all. The page still occupies a number in the publisher's pagination —
    it simply is not on the leaf. Measured on Schwartz's Chinese Communism,
    every one of the twenty-one fillable pages was of that kind, and not one
    was a page where the reading merely failed.

    So a run is filled only when the arithmetic closes: bounded by confident
    anchors on BOTH sides, and the difference between their numbers exactly
    equal to the number of pages between them. Where a plate is bound in or a
    leaf is missing from the capture, the difference exceeds the page count,
    nothing is known about which page holds which number, and the run is
    refused and reported. Nothing is ever extrapolated past the outermost
    anchor: front matter is commonly numbered in Roman, and running Arabic
    backwards off the first anchor would invent a numbering the book does not
    use.

    Confidence is judged again here, against the one thing a book guarantees:
    folios strictly increase in page order. The lone-misread demotion upstream
    cannot catch misreads that SUPPORT each other — two adjacent pages both
    read as folio 16 between 215 and 218 look mutually plausible, and shipped
    a page-list that ran 215, 16, 16, 218. So the anchors are cut to their
    longest strictly-increasing run first; a reading outside it is distrusted
    however confidently it was read. The dividend: once the false 16s are
    gone, the pages they sat on interpolate to the 216 and 217 the printer
    actually bound there.

    Returns (number by page index,
             runs refused as (lo, hi, lo_no, hi_no),
             page indices whose readings were distrusted for breaking order).
    """
    known = {f.index: f.number for f in folios
             if f.confident and f.number is not None}
    idx = sorted(known)
    # longest strictly-increasing subsequence of the numbers in page order
    best_len = [1] * len(idx)
    prev = [-1] * len(idx)
    for i in range(len(idx)):
        for j in range(i):
            if known[idx[j]] < known[idx[i]] and best_len[j] + 1 > best_len[i]:
                best_len[i], prev[i] = best_len[j] + 1, j
    keep: set[int] = set()
    if idx:
        i = max(range(len(idx)), key=lambda k: best_len[k])
        while i != -1:
            keep.add(idx[i])
            i = prev[i]
    distrusted = [i for i in idx if i not in keep]
    known = {i: known[i] for i in keep}
    filled, refused = dict(known), []
    for lo, hi in zip(sorted(known), sorted(known)[1:]):
        if hi - lo < 2:
            continue                       # adjacent anchors leave no gap
        if hi - lo != known[hi] - known[lo]:
            refused.append((lo, hi, known[lo], known[hi]))
            continue
        for i in range(lo + 1, hi):
            filled[i] = known[lo] + (i - lo)
    return filled, refused, distrusted


# A figure carrying less ink than this is blank paper, not a picture. The two
# populations do not overlap remotely: measured across a scanned book, the
# blank-page crops ran 0.00000-0.00071 and everything real — a cloth cover, a
# library stamp, a halftone plate — ran 0.149-0.398, a gap of more than two
# hundred fold. Anywhere in between would do; this sits in the middle of it.
FIGURE_MIN_DETAIL = 0.01


def figure_has_content(img: Image.Image) -> bool:
    """
    Whether a detected figure holds a picture or just the paper it was printed on.

    The layout pass marks a whole blank leaf as a figure — reasonably, since
    there is nothing else on it to call the region — and those crops then reach
    the book as full-page images of nothing. In one 518-page scan, 44 of the 50
    figures were blank leaves.

    Measured as ink against the image's OWN local background rather than against
    an absolute level, because the paper here is tan and a fixed darkness
    threshold reads the whole sheet as ink.
    """
    try:
        g = img.convert("L")
    except (OSError, ValueError):
        return True                     # unreadable: keep it, and let a human judge
    if g.width < 8 or g.height < 8:
        return False
    a = np.asarray(g, dtype=np.float32)
    bg = np.asarray(g.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32)
    return float(((bg - a) > 12).mean()) >= FIGURE_MIN_DETAIL


class TocEntry(NamedTuple):
    """One line of a book's own table of contents, as printed."""
    title: str
    folio: int | None        # None: a roman-numbered or unnumbered line
    folio_text: str          # the number as printed, roman numerals included
    depth: int               # 0 = top grouping, 1 = subgroup, 2 = leaf


def parse_printed_toc(bodies: list[str], limit: int = 25
                      ) -> tuple[list[TocEntry], set[int]]:
    """
    The book's own contents pages, read back as structure.

    The layout pass renders a printed contents page as a real table — one row
    per line, folio in its own right-aligned cell, group headings as <h3>/<h4>
    — so this is a parse, not a reconstruction. Books put their own names on
    their chapters; deriving a contents list any other way guesses at what the
    book already states.

    A page qualifies as a contents page if it says so in its opening text, or
    if it directly continues one and still looks like one (folio-terminated
    rows). TiHKAL's contents run two pages and only the first is titled.
    """
    try:
        from lxml import html as lhtml
    except ImportError:
        return [], set()
    entries: list[TocEntry] = []
    toc_pages: set[int] = set()
    prev_was_toc = False
    for page_i, body in enumerate(bodies[:limit]):
        text_open = _strip_tags(body)[:1000].lower()
        rows_here = body.count("</tr>")
        # A contents page need not announce itself. This one opens straight
        # into "Preface . . . . . vii" and was skipped for want of the word
        # "contents", costing the book its whole structure. What makes a
        # printed contents page recognisable is the dot leader — a run of
        # dots carrying the eye to a folio — which no ordinary table has.
        leaders = len(re.findall(r"[.·…]\s*[.·…]\s*[.·…]", body))
        looks_like_contents = rows_here >= 5 and leaders >= max(3, rows_here // 3)
        # A bare list of titles runs past its first page as bare paragraphs,
        # with no rows to recognise it by — the Saylor contents names its
        # first 23 chapters on one page and its last 11 on the next, and
        # stopping at the page break loses a third of the book's structure.
        short_lines = len([m for m in re.finditer(r"<p>([^<]{3,120})</p>", body)
                           if re.search(r"[A-Za-z]{3}", m.group(1))])
        is_toc = ("contents" in text_open or looks_like_contents
                  or (prev_was_toc and (rows_here >= 5 or short_lines >= 5)))
        prev_was_toc = is_toc
        if not is_toc:
            continue
        toc_pages.add(page_i)
        try:
            root = lhtml.fragment_fromstring(body, create_parent="div")
        except Exception:
            continue
        # A contents page that lists titles and no page numbers at all — the
        # Saylor economics volumes name 34 chapters as bare paragraphs. There
        # is nothing to resolve against the folio map, so these carry no folio
        # and are placed later by hunting the title through the book. Only
        # taken when the page yields no rows, so a real contents table is
        # never competing with the prose around it.
        if not body.count("</tr>"):
            for el in root.iter("p"):
                t = " ".join((el.text_content() or "").split())
                t = re.sub(r"[\s.·…]+$", "", t)
                if 3 <= len(t) <= 120 and re.search(r"[A-Za-z]{3}", t) \
                        and not re.fullmatch(r"[\d\s.,:-]+", t):
                    entries.append(TocEntry(t, None, "", 2))
            continue
        for el in root.iter():
            if el.tag in ("h1", "h2", "h3", "h4"):
                t = " ".join((el.text_content() or "").split())
                if t and "contents" not in t.lower():
                    entries.append(TocEntry(t, None, "", 0 if el.tag in ("h1", "h2", "h3") else 1))
            elif el.tag == "tr":
                cells = [" ".join(td.text_content().split()) for td in el.iter("td", "th")]
                cells = [c for c in cells if c]
                if not cells:
                    continue
                folio_text = cells[-1]
                title = " ".join(cells[:-1]) if len(cells) > 1 else ""
                # printed dot leaders survive OCR as trailing runs of dots
                title = re.sub(r"[\s.·…]+$", "", title)
                m = re.fullmatch(r"(\d{1,3})", folio_text)
                cm = re.fullmatch(r"(\d{1,2})-(\d{1,3})", folio_text)
                roman = re.fullmatch(r"[ivxlc]+", folio_text.lower())
                if cm and title:
                    entries.append(TocEntry(title, int(cm.group(1)) * 1000
                                            + int(cm.group(2)), folio_text, 2))
                elif m and title:
                    entries.append(TocEntry(title, int(m.group(1)), folio_text, 2))
                elif roman and title:
                    entries.append(TocEntry(title, None, folio_text, 2))
                else:
                    # A row with no folio at all is usually noise — but a
                    # part heading is printed exactly this way: eagle's
                    # contents groups its chapters under "BOOK SEVEN: The
                    # Revolt" lines that carry no page number, and dropping
                    # them flattened the book's own declared structure.
                    full = re.sub(r"[\s.·…]+$", "", " ".join(cells))
                    if re.match(r"(BOOK|PART|VOLUME)\b", full, re.I) \
                            and 2 <= len(full.split()) <= 8:
                        entries.append(TocEntry(full, None, "", 0))
    return _infer_row_depths(entries), toc_pages


def _infer_row_depths(entries: "list[TocEntry]") -> "list[TocEntry]":
    """The printed page declares its own hierarchy in the entries' prefixes:
    Russian Purge lists roman-numeral chapters with runs of (1)…(6)
    subsections beneath them, and a nav that flattens all forty-nine lines
    is our failure, not the book's. Rows parse flat; this pass reads the
    declaration back. Only parenthesized prefixes are believed — (1), (a) —
    and only when at least three rows carry one, so a flat contents stays
    flat. A row is promoted to section rank only when a subsection actually
    follows it: a chapter with none stays a plain leaf, because an empty
    grouping in the nav is noise."""
    sub = re.compile(r"^\([0-9a-z]{1,3}\)\s")
    rows = [i for i, e in enumerate(entries) if e.depth == 2]
    n_sub = sum(1 for i in rows if sub.match(entries[i].title))
    if n_sub < 3 or n_sub == len(rows):
        return entries
    out = list(entries)
    for k, i in enumerate(rows):
        if sub.match(entries[i].title):
            continue
        nxt = rows[k + 1] if k + 1 < len(rows) else None
        if nxt is not None and sub.match(entries[nxt].title):
            out[i] = entries[i]._replace(depth=1)
    return out


def folio_resolver(folios: "list[Folio]") -> "callable":
    """
    Printed folio → position in the book, exact where read, pinned where not.

    Chapter openings print no running head, so the very pages a contents line
    points at are the ones with no direct reading. Their neighbours hold them
    in place: with hundreds of confident anchors, the page carrying folio F
    sits a known distance from the nearest anchor on each side, and both sides
    must agree before the answer is trusted.
    """
    anchors = sorted((f.number, f.index) for f in folios
                     if f.confident and f.number is not None)

    def resolve(folio: int) -> int | None:
        if not anchors:
            return None
        import bisect
        i = bisect.bisect_left(anchors, (folio, -1))
        if i < len(anchors) and anchors[i][0] == folio:
            return anchors[i][1]
        left = anchors[i - 1] if i > 0 else None
        right = anchors[i] if i < len(anchors) else None
        cand_l = left[1] + (folio - left[0]) if left else None
        cand_r = right[1] - (right[0] - folio) if right else None
        if cand_l is not None and cand_r is not None:
            return cand_l if cand_l == cand_r else None   # neighbours disagree: refuse
        return cand_l if cand_l is not None else cand_r

    return resolve


def _title_on_page(title: str, text: str) -> bool:
    """Whether enough of a contents line's words appear in a page's text."""
    words = [w for w in re.findall(r"[a-z0-9']+", title.lower()) if len(w) > 2]
    if not words:
        return False
    hay = text.lower()
    hits = sum(1 for w in words if w in hay)
    # Rounded UP: for a two-word title, 60% rounded down is one word, and one
    # word is how "Comments on the Title" matched the cover art. Every word
    # short of the fraction has to be there.
    need = -(-len(words) * 3 // 5)
    return hits >= max(1, need)


def _title_names_this_page(title: str, text: str) -> bool:
    """
    A stricter title test, for lines with no folio to corroborate them.

    The ordinary test asks whether enough of a title's words appear, matching
    substrings so plurals still count. Hunting a chapter across a whole book
    that is far too easy to satisfy: "Chapter 7: The Analysis of Consumer
    Choice" matched a section called "The Analysis of Maximizing Behavior"
    because the prose beneath it mentioned consumers. Here every LONG word of
    the title must be present — the distinctive ones, which prose in the
    neighbourhood does not supply by accident.
    """
    # A chapter announces itself by printing its title, whole and unbroken.
    # Bag-of-words cannot tell that apart from a page that merely discusses
    # the same subject: "Chapter 10: Monopoly" matched a page inside chapter
    # 9 that mentioned monopoly while saying "in this chapter", and "The
    # Analysis of Consumer Choice" matched a section called "The Analysis of
    # Maximizing Behavior" whose prose spoke of consumers. A contiguous
    # phrase is not supplied by accident.
    flat = lambda t: " " + re.sub(r"[^a-z0-9]+", " ", t.lower()).strip() + " "
    hay = flat(text)
    needle = flat(title)
    if len(needle) > 6 and needle in hay:
        return True
    # A contents line carries the chapter's number; the opening page often
    # prints only its name. "I THE INVISIBLE GOVERNMENT" — where the OCR read
    # a printed 1 as a roman I — never matched the page that says just "The
    # Invisible Government", and chapter one lost its link. Try again without
    # the leading numeral, roman or arabic, which the page is entitled to
    # omit and the contents is entitled to carry.
    bare = re.sub(r"^\s*(?:[0-9]{1,3}|[ivxlcIVXLC]{1,5})[\s.:)-]+", "", title)
    if bare != title:
        needle = flat(bare)
        # Without its numeral a chapter title is just the book's own words,
        # and a half-title page says exactly that and nothing else. A chapter
        # OPENS: it carries its title and then it starts talking.
        if len(needle) > 6 and needle in hay and len(text.split()) >= 40:
            return True
    return False



def _place_toc_entries(bodies: list[str], folios,
                       dropped: "set[int]" = frozenset()):
    """Resolve each printed-contents line to a page, refusing placements that
    break the contents' own order.

    Resolution goes wrong in one specific, nasty way: a chapter opener prints
    its chapter NUMBER where a folio normally sits, the furniture pass reads
    it as a folio, and some other chapter's contents line then resolves onto
    that page exactly. Title checking cannot be relied on to catch it —
    "2 48 HOURS" verified happily on the page whose prose says "48 hours".
    What does catch it is the contents page itself: its lines run in reading
    order, so the pages they resolve to must ascend. Placements off the
    longest non-decreasing run are refused rather than emitted wrong.
    """
    printed, toc_pages = parse_printed_toc(bodies)
    resolve = folio_resolver(folios)
    placed: list[tuple] = []          # (entry, target | None, verified)
    for e in printed:
        # groupings without a printed folio have nothing to resolve; a
        # promoted chapter row still carries its folio and resolves like
        # any leaf
        if e.folio is None and e.depth < 2:
            placed.append((e, None, False))
            continue
        target = resolve(e.folio) if e.folio is not None else None
        verified = False
        if target is not None:
            # a contents line may point a page or two off after drops and
            # unread folios; the title's own words settle it
            cands = [t for t in (target, target + 1, target - 1)
                     if 0 <= t < len(bodies) and t not in dropped]
            hit = next((t for t in cands
                        if _title_on_page(e.title, _strip_tags(bodies[t]))),
                       None)
            if hit is not None:
                target, verified = hit, True
        placed.append((e, target, verified))
    leaf = [(k, t) for k, (e, t, _) in enumerate(placed) if t is not None]

    def lis(seq):
        n = len(seq)
        best, prev = [1] * n, [-1] * n
        for i in range(n):
            for j in range(i):
                if seq[j][1] <= seq[i][1] and best[j] + 1 > best[i]:
                    best[i], prev[i] = best[j] + 1, j
        i = max(range(n), key=lambda k: best[k])
        out = set()
        while i != -1:
            out.add(seq[i][0])
            i = prev[i]
        return out

    if len(leaf) >= 3:
        # A printed contents is not always ONE ascending sequence: a manual
        # follows its chapters with a list of figures whose folios start
        # over, and a single longest-run pass kept the figure list while
        # refusing every chapter. So long runs are extracted repeatedly —
        # each is a real sequence the book printed — and only the short
        # residue is refused, which is still exactly what a poisoned anchor
        # leaves behind: a spike belongs to no long run.
        # The first run is the book's principal sequence and is kept at any
        # length, as the single-pass guard always did. A FURTHER run must be
        # long enough to be a second printed list rather than salvage of
        # scattered spikes.
        keep: set = lis(leaf)
        remaining = [x for x in leaf if x[0] not in keep]
        while len(remaining) >= 5:
            run = lis(remaining)
            if len(run) < 5:
                break
            keep |= run
            remaining = [x for x in remaining if x[0] not in run]
        for k, t in leaf:
            if k not in keep:
                e, _, _ = placed[k]
                placed[k] = (e, None, False)
    return placed, toc_pages


# An entry head opens its own block — heading or paragraph, bolded or not —
# while a passing mention ("described in PIHKAL, entry #173") sits mid-sentence.
# Only the position is matched here; the name is taken from the stripped text
# that follows, because the books set Greek letters as inline markup and a name
# like "α,N-DMT" begins with a tag no character class survives. TiHKAL spread
# its fifty-five heads across four different renderings of the same convention.
ENTRY_HEAD = re.compile(r"<(?:h[1-4][^>]*|p)>\s*(?:<b>\s*)?#(\d{1,3})\.")
# …and a fifth rendering: a bare <b> as the page's first element, no block
# wrapper at all. Position is what separates it from a mid-sentence mention —
# and the position is the START OF THE FRAGMENT, not a <body> tag: fragments
# are what this sees, and the wrapper is added long after, at page assembly.
# The first version anchored on <body>, and its test passed by constructing
# exactly the wrapped string the code would never receive.
ENTRY_HEAD_BARE = re.compile(r"^\s*(?:<b>\s*)?#(\d{1,3})\.")


GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
         "mu": "μ", "pi": "π", "psi": "ψ", "epsilon": "ε", "omega": "ω"}


def normalize_math(body: str) -> str:
    """
    LaTeX out, letters in.

    The recogniser wraps Greek in <math>\\alpha</math>, and no EPUB reader
    renders LaTeX: the reading experience is a raw backslash-word in the middle
    of a chemical name, over a thousand times across one book's chemistry half.
    The content is only ever letters and simple wrappers, so it translates
    exactly; anything unrecognised keeps its text with the markup stripped,
    which is never worse than what the reader would have shown.
    """
    def clean(m):
        t = m.group(1)
        t = re.sub(r"\\(?:text|mathrm|mathit)\{([^}]*)\}", r"\1", t)
        for name, ch in GREEK.items():
            t = t.replace("\\" + name, ch)
        t = t.replace("\\,", ",").replace("\\", "")
        return t
    # Display math is exempt: a block the layout model LABELLED Formula is
    # an equation, not a Greek letter in a chemical name, and flattening it
    # to letters destroys structure a MathML-capable reader would show. The
    # assembly loop stamps those blocks display="block" on the way in.
    return re.sub(r"<math(?![^>]*display=\"block\")[^>]*>(.*?)</math>",
                  clean, body, flags=re.S)


# The five renderings of an entry head, each capturing the head's own text.
_HEAD_FORMS = [re.compile(x, re.S) for x in (
    r"<h[1-4][^>]*>\s*(?:<b>\s*)?(#\d{1,3}\..*?)(?:</b>\s*)?</h[1-4]>",
    r"<p>\s*<b>\s*(#\d{1,3}\..*?)</b>\s*</p>",
    r"^\s*<b>\s*(#\d{1,3}\..*?)</b>",
    r"<p>\s*(#\d{1,3}\..*?)</p>",
)]


def normalize_entry_heads(body: str) -> str:
    """
    One look for every entry head, whatever the recogniser made of it.

    The same printed convention arrived as five different markups — <h2>, <h4>
    with bold, bold paragraphs, plain paragraphs, and a bare <b> opening the
    page — so entries open at random visual weights through the book. Each is
    rewritten to the same heading, full name list kept, Greek already restored
    by normalize_math. Only the page's first match is touched: a mid-page
    mention is prose and stays prose.
    """
    for form in _HEAD_FORMS:
        # Anywhere on the page, not an opening window: entries begin mid-page
        # whenever the previous one ends there, and a fixed window left thirty
        # of these unstyled the first time — the same mistake the entry FINDER
        # made and fixed a round earlier. The forms themselves carry the
        # safety: a heading tag is a heading wherever it sits, a whole-bold or
        # #-opening paragraph likewise, and the bare-<b> form stays anchored to
        # the page start inside its own pattern.
        m = form.search(body)
        if m:
            text = " ".join(_strip_tags(m.group(1)).split())
            head = f'<h3 class="entry">{html.escape(text)}</h3>'
            return body[:m.start()] + head + body[m.end():]
    return body


def _load_cover(spec: str, page_paths: list[str],
                page_ids: list[str] | None) -> bytes | None:
    """
    The cover image, from a page of this book or a file of the reader's choosing.

    A page selector reuses the same names as --drop-pages; a path takes the
    file as it stands, which is how an unenhanced original gets in.
    """
    path: str | None = None
    if os.path.exists(spec):
        path = spec
    else:
        try:
            idx = resolve_selector(spec, page_ids)
            if 0 <= idx < len(page_paths):
                path = page_paths[idx]
        except ValueError as exc:
            print(f"  [!] --cover {spec}: {exc}")
            return None
    if path is None:
        return None
    img = read_image(path)
    if img is None:
        return None
    buf = io.BytesIO()
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def find_numbered_entries(bodies: list[str]) -> list[tuple[int, str, int]]:
    """
    Reference-work entries ("#41. 5,6-MeO-MIPT") by the page they open on.

    TiHKAL and PiHKAL's second halves are organised by these, not by chapters,
    and they are what a reader actually navigates to. The head is set bold at
    the top of the entry's first page, which the recogniser preserves.
    """
    out: list[tuple[int, str, int]] = []
    seen: set[int] = set()
    for i, body in enumerate(bodies):
        m = ENTRY_HEAD.search(body) or ENTRY_HEAD_BARE.search(body[:160])
        if m:
            n = int(m.group(1))
            if n in seen:                    # continuation pages repeat nothing
                continue
            seen.add(n)
            name = _strip_tags(body[m.end():m.end() + 300])
            name = name.replace("alpha", "α").replace("beta", "β") \
                if "\\" in name else name
            name = re.sub(r"text\{([^}]*)\}", r"\1", name)
            name = name.replace("\\", "").split(";")[0].strip()[:60]
            out.append((i, f"#{n}. {name}" if name else f"#{n}.", n))
    return out


NOTE_ENTRY = re.compile(r"<p>\s*<sup>\s*(\d{1,3})\s*</sup>")
BODY_MARKER = re.compile(r"<sup>\s*(\d{1,3})\s*</sup>")
NOTES_HEAD = re.compile(r"<h\d[^>]*>\s*NOTES?\s*</h\d>", re.I)
# The names a notes section gives its groups. Chapter forms, spelled out or
# numbered — and the sections a book's front and back matter use, because a
# book that annotates its introduction gives those notes a group of their
# own. Missing that group does not merely lose it: groups pair by position,
# so an unrecognised first group shifts every one after it by one, and the
# markers then link confidently to the wrong chapter's citations.
GROUP_NAMES = (r"CHAPTER\s+[A-Z\-]+|CHAPTER\s+\d{1,3}"
               r"|INTRODUCTION|PREFACE|PROLOGUE|FOREWORD|FORWARD"
               r"|CONCLUSION|EPILOGUE|AFTERWORD|POSTSCRIPT")
# Inline markup between the heading and its words is invisible to a reader
# and fatal to a pattern that does not expect it: Work, Sex & Power sets
# every one of its eighteen group heads as <h2><i>Chapter One</i></h2>, and
# not one was recognised, so all 359 of its markers went unlinked. The
# designation must still be the WHOLE heading — a heading that merely opens
# with the word is prose, not a head.
GROUP_HEAD = re.compile(
    r"<(h[1-6]|p)\b[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    r"(" + GROUP_NAMES + r")"
    r"\s*(?:</[a-zA-Z]+>\s*)*</\1>", re.I)


def parse_notes_section(bodies: list[str]) -> tuple[int, list[list[tuple[int, int, int]]]]:
    """
    The book's endnotes, grouped as the book groups them.

    Returns (first_notes_page, groups) where each group is the notes of one
    chapter, in order, as (page_index, number, match_start). A note is a
    paragraph that OPENS with a superscript number — the position is what
    tells a citation apart from a marker in running prose, which never leads
    its paragraph.
    """
    start = next((i for i, b in enumerate(bodies) if NOTES_HEAD.search(b)), None)
    if start is None:
        return -1, []
    groups: list[list[tuple[int, int, int]]] = []
    group_nos: list[int | None] = []
    current: list[tuple[int, int, int]] | None = None
    empty_streak = 0
    for i in range(start, len(bodies)):
        b = bodies[i]
        sup_notes = [(m.start(), "note", int(m.group(1))) for m in NOTE_ENTRY.finditer(b)]
        # a section that sets its notes as "1. Source…" rather than with a
        # superscript; only consulted where the superscript form is absent,
        # since "1. " also opens ordinary numbered lists in prose
        plain_notes = [] if sup_notes else \
            [(m.start(), "note", int(m.group(1)))
             for pat in (NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
             for m in pat.finditer(b)]
        heads = []
        for m in GROUP_HEAD.finditer(b):
            heads.append((m.start(), "group", None))
        for pat in (GROUP_HEAD_NUMBERED, GROUP_HEAD_NUMBERED_B):
            for m in pat.finditer(b):
                heads.append((m.start(), "group", int(m.group(1))))
        for pat in (GROUP_HEAD_ROMAN, GROUP_HEAD_ROMAN_P):
            for m in pat.finditer(b):
                no = _roman_to_int(m.group(1))
                if no is not None:
                    heads.append((m.start(), "group", no))
        events = sorted(heads + sup_notes + plain_notes, key=lambda e: e[0])
        if i > start and not events and current:
            # one event-less page may be a blank leaf inside the notes; two
            # in a row mean the notes have ended and this is the index
            empty_streak += 1
            if empty_streak >= 2:
                break
            continue
        empty_streak = 0
        for pos, kind, n in events:
            if kind == "group":
                current = []
                groups.append(current)
                group_nos.append(n)
            elif current is not None:
                current.append((i, n, pos))
    keep = [(no, g) for no, g in zip(group_nos, groups) if g]
    if not keep:
        solo = _ungrouped_notes(bodies, start)
        if solo:
            parse_notes_section.last_group_numbers = [None]
            return start, [solo]
    parse_notes_section.last_group_numbers = [no for no, _ in keep]
    return start, [g for _, g in keep]


def _ungrouped_notes(bodies: list[str], start: int) -> list:
    """
    The whole notes section as one group, for a book that does not group.

    Not every book divides its endnotes by chapter. A Critical History of
    Poverty Finance cites author-date in the prose and keeps numbered notes
    for archival sources only, so its twenty-one notes run 1 to 21 straight
    through under a single "Notes" heading with no chapter heads at all. Every
    entry was recognised and then discarded, because a group must be opened
    before an entry has anywhere to go.

    The absence of grouping is what makes this safe rather than risky: with
    one sequence the numbers are unique across the whole book, so the scope
    question the group machinery exists to answer does not arise — note 7 is
    note 7 wherever it is cited. That is also the condition. The numbers must
    climb without restarting; a section that starts over without a heading to
    say so is genuinely ambiguous and keeps its refusal.

    The section ends at the next part of the book. A bibliography or an index
    opens with a heading of the notes head's own rank, and numbered entries
    beyond that boundary are somebody else's.
    """
    rank = re.search(r"<(h[1-6])[^>]*>\s*NOTES?\s*</\1>", bodies[start], re.I)
    stop = rf"<{rank.group(1)}[^>]*>" if rank else r"<h1[^>]*>"
    entries: list[tuple[int, int, int]] = []
    for i in range(start, len(bodies)):
        if i > start and re.search(stop, bodies[i], re.I):
            break
        sup = [(m.start(), int(m.group(1))) for m in NOTE_ENTRY.finditer(bodies[i])]
        plain = [] if sup else [(m.start(), int(m.group(1)))
                                for pat in (NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
                                for m in pat.finditer(bodies[i])]
        entries += [(i, n, pos) for pos, n in sorted(sup + plain)]
    if len(entries) < 3:
        return []
    nums = [n for _, n, _ in entries]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        return []
    return entries


def _entry_anchor(entry: str, note_id: str, back_href: str, n: int):
    """
    (span, replacement) anchoring a note entry's opening with its id and
    backlink, in whichever of the three renderings the entry uses — or None
    when the entry's opening matches none of them, in which case the caller
    must leave the marker unlinked: a forward href to an id that was never
    written is a broken link.
    """
    nm = NOTE_ENTRY.match(entry)
    if nm is not None:
        return nm.end(), (f'<p id="{note_id}"><sup><a href="{back_href}">'
                          f'{n}</a></sup>')
    pm = re.match(r"<(p|li)[^>]*>\s*(\d{1,3})\.", entry)
    if pm is not None:
        return pm.end(), (f'<{pm.group(1)} id="{note_id}">'
                          f'<a href="{back_href}">{n}.</a>')
    return None


def link_chapter_notes(bodies: list[str], dropped: set) -> dict:
    """
    Chapter-endnotes layout: a "Notes" section closes each chapter, entries
    numbered from 1, markers scattered through the chapter body before it.

    The scoping needs no guessing of any kind — the sections partition the
    book by position. Markers between one section and the next belong to the
    next, and that is the whole rule; no reset detection, no chapter numbers,
    no contents page. Entries may run past the section's first page, accepted
    only while their numbers keep ascending by one, so a numbered list in the
    following chapter's prose ("1. First, …") cannot be swallowed as more
    notes.
    """
    stats = {"linked": 0, "unlinked": 0, "sections": 0}
    heads = [(i, m.start()) for i, b in enumerate(bodies)
             for m in [NOTES_HEAD.search(b)] if m]
    if len(heads) < 2:
        return stats
    edits: dict[int, list[tuple[int, int, str]]] = {}
    prev_head_page = -1
    for k, (hp, hpos) in enumerate(heads):
        # entries: from just past the head, across pages while numbers ascend
        entries: dict[int, tuple[int, int]] = {}
        expect = 1
        pi, from_pos = hp, hpos
        while pi < len(bodies):
            found_here = False
            for pat in (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI):
                for m in pat.finditer(bodies[pi]):
                    if pi == hp and m.start() < from_pos:
                        continue
                    n = int(m.group(1))
                    if n == expect:
                        entries.setdefault(n, (pi, m.start()))
                        expect += 1
                        found_here = True
            if not found_here and pi > hp:
                break
            pi += 1
            if k + 1 < len(heads) and pi > heads[k + 1][0]:
                break                       # never read into the next section
        # markers: strictly the body span since the previous section's head
        markers = []
        for bpi in range(prev_head_page + 1, hp + 1):
            for m in BODY_MARKER.finditer(bodies[bpi]):
                if bodies[bpi][:m.start()].rstrip().endswith("<p>"):
                    continue                # an entry, not a marker
                markers.append((bpi, int(m.group(1)), m.start()))
        prev_head_page = hp
        if not entries or not markers:
            continue
        stats["sections"] += 1
        seen: set[int] = set()
        for (bpi, n, pos) in markers:
            target = entries.get(n)
            if target is None or bpi in dropped:
                stats["unlinked"] += 1
                continue
            tpi, tpos = target
            ref_id, note_id = f"ref-c{k}-{n}", f"note-c{k}-{n}"
            first = n not in seen
            if first:
                anchored = _entry_anchor(bodies[tpi][tpos:], note_id,
                                         f"page_{bpi:04d}.xhtml#{ref_id}", n)
                if anchored is None:
                    stats["unlinked"] += 1
                    continue
                span, back = anchored
                edits.setdefault(tpi, []).append((tpos, tpos + span, back))
                seen.add(n)
            mm = BODY_MARKER.match(bodies[bpi][pos:])
            id_attr = f' id="{ref_id}"' if first else ""
            new_html = (f'<sup><a epub:type="noteref"{id_attr} '
                        f'href="page_{tpi:04d}.xhtml#{note_id}">{n}</a></sup>')
            edits.setdefault(bpi, []).append((pos, pos + mm.end(), new_html))
            stats["linked"] += 1
    for pi, page_edits in edits.items():
        b = bodies[pi]
        for a, z, new_html in sorted(page_edits, reverse=True):
            b = b[:a] + new_html + b[z:]
        bodies[pi] = b
    return stats


def find_body_marker_groups(bodies: list[str], stop: int) -> list[list[tuple[int, int, int]]]:
    """
    Superscript markers in the prose, split into chapters by their own resets.

    Note numbering restarts at 1 every chapter, so the numbers themselves mark
    the chapter boundaries: a marker at or below its predecessor after an
    ascending run means a new chapter has begun. No table of contents is
    consulted — the markers carry their own structure.
    """
    # flatten first: the reset test needs one-marker lookahead
    stream: list[tuple[int, int, int]] = []
    for i in range(max(0, stop)):
        for m in BODY_MARKER.finditer(bodies[i]):
            if bodies[i][:m.start()].rstrip().endswith("<p>"):
                continue               # a note entry, not a marker
            stream.append((i, int(m.group(1)), m.start()))
    groups: list[list[tuple[int, int, int]]] = []
    current: list[tuple[int, int, int]] = []
    prev = 0
    for k, (i, n, pos) in enumerate(stream):
        # A chapter start resets the count to exactly 1 and stays low; a
        # re-citation dips and then returns high. "…16, 17, 2, 18…" is the
        # author citing note 2 again, and calling it a chapter would bind that
        # marker to the NEXT chapter's note 2 — a wrong link found by this
        # module's own regression test before any reader could find it.
        nxt = stream[k + 1][1] if k + 1 < len(stream) else None
        if n == 1 and prev >= 3 and (nxt is None or nxt <= 3):
            if current:
                groups.append(current)
            current = []
        current.append((i, n, pos))
        prev = n
    if current:
        groups.append(current)
    return groups


NOTE_ENTRY_PLAIN = re.compile(r"<p>\s*(\d{1,3})\.\s+\S")
# The layout pass does not render a notes section consistently: The Invisible
# Government's ran ten pages and came back as <h2> heads with <p> entries on
# most, but one page as bare <b> heads with <ol><li> entries. Same printed
# convention, two renderings — both must parse or every group after the odd
# page is silently lost.
NOTE_ENTRY_LI = re.compile(r"<li[^>]*>\s*(\d{1,3})\.\s+\S")
GROUP_HEAD_NUMBERED = re.compile(r"<h\d[^>]*>\s*(\d{1,2})\.\s")
GROUP_HEAD_NUMBERED_B = re.compile(r"<b>\s*(\d{1,2})\.\s+\S")
# A book may number its chapters in Roman, and then its endnote groups are
# headed "I.", "II.", "III." where another book heads them "1.". Schwartz's
# Chinese Communism does, and without this every one of its 697 markers went
# unlinked: no head was recognized, so no group ever opened.
GROUP_HEAD_ROMAN = re.compile(r"<h\d[^>]*>\s*([IVXL]{1,6})\.\s")
# The same book set twelve of its thirteen group heads as headings and one as
# an ordinary paragraph, so the paragraph form has to parse too — but it is
# the risky one, because a note continuing across a page break also opens a
# paragraph. Three things keep it honest: the block must stand alone and end
# right there, it must be short the way a title is and not a citation, and it
# must not be an author's initials, which is how "I. V. Stalin" opens a note
# in exactly the literature this pattern reads.
GROUP_HEAD_ROMAN_P = re.compile(
    r"<p[^>]*>\s*([IVXL]{1,6})\.\s+(?![A-Z]\.)(?:<[^>]+>|[^<]){1,70}</p>")


def _roman_to_int(s: str) -> "int | None":
    """The value of a Roman numeral, or None if it is not one.

    Strict: it must re-render to exactly what was read, so "IIII", "VX" and
    other things that merely use the letters are refused rather than guessed
    at. A misread group number would scope a whole chapter's notes wrongly.
    """
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = prev = 0
    for ch in reversed(s.upper()):
        if ch not in values:
            return None
        v = values[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    if not 1 <= total <= 99:
        return None
    tens = ["", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"]
    ones = ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
    return total if tens[total // 10] + ones[total % 10] == s.upper() else None
# A bold block standing on its own where a numbered group head belongs — at
# the fragment start or hard against another block's close, never bold inside
# running text, which is emphasis and must not be promoted.
BOLD_HEAD_BLOCK = re.compile(
    r"(^|</(?:p|h\d|ol|ul|table|figure)>)(\s*)<b>\s*(\d{1,2})\.\s+([^<]+?)\s*</b>")


# A heading that opens a chapter: the word, then the number the book counts
# in. Roman is admitted because half the books here number that way, and the
# strict parser refuses anything that is merely made of those letters.
CHAPTER_HEAD = re.compile(
    r"<(h[1-6])[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    r"(CHAPTER|Chapter)\s+(\d{1,3}|[IVXL]{1,7})"
    r"\s*[.:—-]?\s*(.*?)</\1>", re.S)


def _chapter_head_number(token: str) -> "int | None":
    return int(token) if token.isdigit() else _roman_to_int(token)


def normalize_chapter_heads(bodies: list[str]) -> int:
    """
    Give every chapter opening the same rank, and its title back.

    The layout pass reads each page alone, so it has no way to notice that it
    called chapter five an h1 and chapter six an h2 — Principles of
    Macroeconomics came out seven of one and fourteen of the other for the
    same job. It also splits the opening in two, the number in one heading
    and the title in the next, which leaves neither able to say on its own
    what chapter this is.

    Both are repaired from the same recognition. Rank goes to the SHALLOWEST
    level the book itself used for a chapter: a chapter is the outermost
    division a book names, so where the layout ever managed to see one as
    top-level, that is what it is — taking the commonest instead would have
    pushed macroeconomics' chapters down to h2, level with the very titles
    they own. The title is then folded back into the heading, giving the
    shape the books that got it right already have ("CHAPTER 1. INVASION").

    Only an exactly-numbered heading is touched, so a book whose openings are
    already whole is left alone, and a heading carrying prose is never
    merged into.
    """
    found = []                                   # (page, match, level)
    for i, body in enumerate(bodies):
        for m in CHAPTER_HEAD.finditer(body):
            if _chapter_head_number(m.group(3)) is None:
                continue
            found.append((i, m, int(m.group(1)[1])))
    if len(found) < 2:
        return 0
    rank = min(level for _, _, level in found)
    changed = 0
    for i in sorted({p for p, _, _ in found}, reverse=True):
        body = bodies[i]
        edits = []
        for m in CHAPTER_HEAD.finditer(body):
            if _chapter_head_number(m.group(3)) is None:
                continue
            word, num = m.group(2), m.group(3)
            title = re.sub(r"\s+", " ", _strip_tags(m.group(4))).strip()
            end = m.end()
            if not title:
                # the title is the next heading along, if that is what it is:
                # adjacent, no deeper than a title should be, and short enough
                # to be one rather than a paragraph that got promoted
                nxt = re.match(r"\s*<(h[1-6])[^>]*>(.*?)</\1>", body[m.end():], re.S)
                if nxt and int(nxt.group(1)[1]) >= rank:
                    cand = re.sub(r"\s+", " ", _strip_tags(nxt.group(2))).strip()
                    if 0 < len(cand) <= 90 and not re.match(
                            r"^(CHAPTER|Chapter)\s", cand):
                        title, end = cand, m.end() + nxt.end()
            label = f"{word} {num}" + (f". {title}" if title else "")
            new = f"<h{rank}>{html.escape(label)}</h{rank}>"
            if body[m.start():end] != new:
                edits.append((m.start(), end, new))
        for a, z, new in sorted(edits, reverse=True):
            body = body[:a] + new + body[z:]
            changed += 1
        bodies[i] = body
    return changed


# An item that has already printed its own marker: the numeral or bullet the
# page shows, before any of the item's words.
SELF_MARKED_NUM = re.compile(r"<li[^>]*>\s*(?:<[^/][^>]*>\s*)*\d{1,3}[.)]\s")
SELF_MARKED_BUL = re.compile(r"<li[^>]*>\s*(?:<[^/][^>]*>\s*)*[•·▪‣∙]\s")
LIST_OPEN = re.compile(r"<(ol|ul)\b([^>]*)>", re.I)


def normalize_list_markers(bodies: list[str]) -> int:
    """
    Stop a list drawing a marker the item has already printed.

    The recogniser hands back a real <ol> whose items still carry the number
    the page showed — "<li>1. Suppose the multiplier…" — so a reader draws
    its own "1." beside ours and the item reads "1. 1. Suppose". Principles
    of Economics shipped 813 of those, and 450 doubled bullets besides. The
    recogniser already writes list-style-type: none on some lists, which is
    exactly why other books came out clean; it simply does not do it
    consistently.

    The marker in the text is what the page printed, so it stays: a list may
    begin at five, or number by citation rather than position, and stripping
    it would delete something the book actually says. What is suppressed is
    the marker the READER would add — presentation only, no text touched.

    Only a list whose items mostly mark themselves is changed, so a list
    where one item merely opens with a date or a figure number is left as it
    is.
    """
    changed = 0
    for bi, body in enumerate(bodies):
        edits = []
        for om in LIST_OPEN.finditer(body):
            kind, attrs = om.group(1).lower(), om.group(2)
            if "list-style" in attrs.lower():
                continue                      # the recogniser already said so
            # find this list's close, respecting nesting
            depth, pos, close = 1, om.end(), None
            while depth and pos < len(body):
                nxt = re.search(rf"<(/?){kind}\b[^>]*>", body[pos:], re.I)
                if not nxt:
                    break
                depth += -1 if nxt.group(1) else 1
                pos += nxt.end()
                if depth == 0:
                    close = pos
            if close is None:
                continue
            # judge this list on ITS OWN items: a nested list that marks
            # itself must not talk its parent into being restyled too
            own = body[om.end():close]
            while True:
                stripped = re.sub(
                    r"<(ol|ul)\b[^>]*>(?:(?!<(?:ol|ul)\b).)*?</\1>", "",
                    own, flags=re.S | re.I)
                if stripped == own:
                    break
                own = stripped
            items = len(re.findall(r"<li\b", own, re.I))
            if not items:
                continue
            pat = SELF_MARKED_NUM if kind == "ol" else SELF_MARKED_BUL
            marked = len(pat.findall(own))
            if marked * 2 < items:
                # Not the list's convention — a marker on one item of four is
                # the recogniser catching a glyph it missed on the rest, and
                # it renders as "• •" on that item alone. Suppressing the
                # list's own markers would strip the other three of theirs,
                # so the stray goes instead. Only ever a bullet: the <li>
                # already says "list item", so the glyph carries nothing a
                # number might, and it is not a word by any count taken here.
                if marked and kind == "ul":
                    fixed = re.sub(
                        r"(<li[^>]*>\s*(?:<[^/][^>]*>\s*)*)[•·▪‣∙]\s+",
                        r"\1", body[om.end():close])
                    if fixed != body[om.end():close]:
                        edits.append((om.end(), close, fixed))
                continue
            style = ' style="list-style-type: none;"'
            edits.append((om.start(), om.end(),
                          f"<{kind}{attrs}{style}>"))
        for a, z, new in sorted(edits, reverse=True):
            body = body[:a] + new + body[z:]
            changed += 1
        bodies[bi] = body
    return changed


# A superscript marker the layout orphaned into its own block stands exactly
# one text line tall. Measured on eagle's 274 specimens: markers 0.019-0.020
# of page height, the shortest multi-line text block 0.038 — this sits in
# the gap. A display chapter numeral is several times taller.
MARKER_BLOCK_MAX_H = 0.03


def reattach_orphan_markers(items: list, furn_numbers: set,
                            seq_state: "dict | None" = None) -> tuple:
    """
    Sort a page's standalone digit blocks into markers and phantoms.

    The layout model does two different things that look identical in the
    markup. Sometimes it segments a real superscript marker into its own
    block — recognition renders <p>2</p> because block-internally there is
    nothing to be superscript to. And sometimes it invents a tiny block
    over ordinary prose that recognition then "reads" as a small digit —
    eagle carried 220 of those, verified against the ink: the boxed regions
    hold body text and no standalone digit at all. Fabrication at the
    block level, below every page-level guard's sightline.

    Geometry cannot separate the two — both stand one line tall. What
    separates them is the marker SEQUENCE: a real orphan continues the
    inline markers (last seen + 1) or opens a chapter (a 1 whose next
    inline marker is 2). A phantom fits nowhere, because it was never part
    of any sequence. Fits are reattached to the preceding prose; misfits
    are suppressed and counted, never shipped as the stray digits they
    have been.

    Returns (reattached, suppressed).
    """
    if seq_state is None:
        seq_state = {"last": None}
    reattached = suppressed = 0
    # inline markers on this page, in order, for lookahead and tracking
    inline_positions = []
    for idx, it in enumerate(items):
        if it.html and it.figure is None:
            for m in re.finditer(r"<sup>\s*(\d{1,3})\s*</sup>", it.html):
                inline_positions.append((idx, int(m.group(1))))

    for i, it in enumerate(list(items)):
        if it.figure is not None or not it.html:
            continue
        t = _strip_tags(it.html).strip()
        if not re.fullmatch(r"\d{1,3}", t):
            continue
        if it.box is None or (it.box[3] - it.box[1]) > MARKER_BLOCK_MAX_H:
            continue
        if t in furn_numbers:
            continue
        val = int(t)
        # advance last-seen over inline markers that precede this block
        for idx, n in inline_positions:
            if idx < i:
                seq_state["last"] = n
        nxt_inline = next((n for idx, n in inline_positions if idx > i), None)
        fits = (seq_state["last"] is not None
                and val == seq_state["last"] + 1)             or (val == 1 and nxt_inline == 2)
        prev = next((p for p in reversed(items[:i])
                     if p.html and p.figure is None), None)
        if fits and prev is not None \
                and len(_strip_tags(prev.html).split()) >= 8:
            m = re.search(r"</(p|li|h[1-6]|blockquote)>\s*$", prev.html)
            sup = f"<sup>{t}</sup>"
            nh = (prev.html[:m.start()] + sup + prev.html[m.start():]
                  if m else prev.html + f"<p>{sup}</p>")
            items[items.index(prev)] = prev._replace(html=nh)
            items[i] = it._replace(html="")
            seq_state["last"] = val
            reattached += 1
        else:
            items[i] = it._replace(html="")
            suppressed += 1
    if reattached or suppressed:
        items[:] = [x for x in items if x.html or x.figure is not None]
    # advance over any trailing inline markers for the next page
    for idx, n in inline_positions:
        seq_state["last"] = max(seq_state["last"] or 0, 0) or None             if False else n
    return reattached, suppressed


def promote_missing_chapter_heads(bodies: list[str]) -> int:
    """
    Recover a chapter opening the layout pass ran into the prose.

    A chapter that begins on a fresh page is easy: its number and title stand
    alone and come back as headings. One that begins halfway down a page,
    under the previous chapter's exercises, does not — Principles of
    Macroeconomics welded the whole of chapter nine's opening onto the end of
    chapter eight's last problem, "…over the two decades. Chapter 9 The
    Nature and Creation of Money Start Up: How Many Macks…". The reader lost
    the chapter break itself, not merely a line in the contents.

    "Chapter 9" also appears in ordinary prose — a textbook cross-references
    its own chapters constantly — so recognition alone would be reckless.
    What makes this safe is that only a HOLE is filled: the number must be
    one the sequence is missing, it must lie between the pages of the
    chapters either side of it, it must open a sentence, and it must be the
    only such mention in that range. A cross-reference fails all of these at
    once; the genuine opening passes them by construction.

    The title is deliberately left where it is. Where it ends in run-together
    prose cannot be known — "…Creation of Money Start Up: How Many Macks…"
    offers no honest boundary — and a heading that swallows half the next
    one would be invention. The break is what was lost; the break is what is
    restored.
    """
    found = []
    for i, body in enumerate(bodies):
        for m in CHAPTER_HEAD.finditer(body):
            no = _chapter_head_number(m.group(3))
            if no is not None:
                found.append((no, i, int(m.group(1)[1])))
    if len(found) < 2:
        return 0                              # nothing to be a gap between
    found.sort()
    rank = min(level for _, _, level in found)
    seen = {no: page for no, page, _ in found}
    pages_in_order = [seen[no] for no in sorted(seen)]
    if any(b < a for a, b in zip(pages_in_order, pages_in_order[1:])):
        return 0                              # sequence disagrees with the book
    filled = 0
    for missing in range(min(seen) + 1, max(seen)):
        if missing in seen:
            continue
        lo, hi = seen.get(missing - 1), seen.get(missing + 1)
        if lo is None or hi is None or hi <= lo:
            continue
        word = "CHAPTER" if any(
            CHAPTER_HEAD.search(bodies[p]) and
            CHAPTER_HEAD.search(bodies[p]).group(2) == "CHAPTER"
            for p in (lo, hi)) else "Chapter"
        pat = re.compile(
            r"(?<=[.!?])(\s+)(" + word + r"\s+" + str(missing) + r")\s+(?=[A-Z])")
        hits = [(p, m) for p in range(lo, hi + 1)
                for m in pat.finditer(bodies[p])]
        if len(hits) != 1:
            continue
        p, m = hits[0]
        body = bodies[p]
        # Split the paragraph: what came before stays in it, the opening
        # becomes a heading of the rank its siblings use, the remainder
        # resumes in a fresh paragraph.
        head = f"</p><h{rank}>{word} {missing}</h{rank}><p>"
        bodies[p] = body[:m.start()] + head + body[m.end():]
        filled += 1
    return filled


def chapter_head_contents(bodies: list[str]) -> "list[tuple[int, str]]":
    """
    (page, label) for each chapter opening, as a contents list of last resort.

    Some books simply have no printed contents page — Principles of
    Macroeconomics runs copyright, preface, chapter one — and until now that
    meant the reader got one navigation entry per page, eight hundred and
    twenty-six of them reading "Page 1", "Page 2". The book still knows where
    its chapters begin; nothing was asking it. This is only ever consulted
    when the printed page yielded nothing, so a real contents always wins.
    """
    out, seen = [], set()
    for i, body in enumerate(bodies):
        for m in CHAPTER_HEAD.finditer(body):
            no = _chapter_head_number(m.group(3))
            if no is None or no in seen:
                continue
            seen.add(no)
            label = re.sub(r"\s+", " ", _strip_tags(m.group(0))).strip()
            if label:
                out.append((i, label))
    # numbers must climb with the pages, or this is not a chapter sequence
    if len(out) < 3:
        return []
    nums = [_chapter_head_number(CHAPTER_HEAD.search(bodies[i]).group(3))
            for i, _ in out]
    if any(b <= a for a, b in zip(nums, nums[1:])):
        return []
    return out


def normalize_note_heads(bodies: list[str]) -> int:
    """
    Promote bold blocks that the notes parser recognizes as group heads into
    the heading form their siblings use.

    The layout pass rendered one notes page's heads as bare <b> where every
    other page used <h2>. The parser learned to read both — but once a bold
    block is identified as a head, the markup itself should say so: the
    reader gets one consistent rendering, not a page that silently differs.
    Only pages inside the notes section that carry note entries are touched,
    and only standalone bold blocks; bold inside a sentence stays emphasis.
    """
    start = next((i for i, b in enumerate(bodies) if NOTES_HEAD.search(b)), None)
    if start is None:
        return 0
    changed = 0
    for i in range(start, len(bodies)):
        b = bodies[i]
        if not (NOTE_ENTRY.search(b) or NOTE_ENTRY_PLAIN.search(b)
                or NOTE_ENTRY_LI.search(b)):
            continue

        def promote(m):
            nonlocal changed
            changed += 1
            return f"{m.group(1)}{m.group(2)}<h2>{m.group(3)}. {m.group(4)}</h2>"

        bodies[i] = BOLD_HEAD_BLOCK.sub(promote, b)
    return changed
FOOTNOTE_SYMBOLS = ("\\*\\*", "\\*", "†", "‡")

# A paragraph shaped like a footnote entry: opening with a marker — numbered
# or symbol, possibly already wrapped in its link by the symbol pass — and
# then saying something.
# A paragraph shaped like a foot: opening with its marker, then saying
# something. The marker may be set as plain text ("1. According to…") or as a
# superscript ("<sup>2</sup> See:…") — one book uses both, page by page, and
# only the first was understood, which cost it 52 of its 63 footnotes. It may
# also already be wrapped in its link by the symbol pass.
FOOT_MARK = r"(?:<sup>\s*)?(?:<a[^>]*>)?\s*(?:\*{1,2}|†|‡|\d{1,3}[.)]?)\s*" \
            r"(?:</a>)?\s*(?:</sup>)?"
FOOT_SHAPE = re.compile(r"<p[^>]*>\s*" + FOOT_MARK + r"\s*\S")
# The same thing with its number captured: what a foot entry is, wherever the
# page happens to put the block.
FOOT_ENTRY = re.compile(r"<p[^>]*>\s*(?:<sup>\s*)?(\d{1,3})[.)]?\s*"
                        r"(?:</sup>)?\s+(?=\S)")


def _looks_like_foot_block(below: str) -> bool:
    """
    Is what follows the rule a footnote block, judged by shape alone?

    A footnote block opens with a foot and is feet nearly throughout. A
    section divider is followed by a heading, or by running prose that no
    footnote block contains. Size says nothing — a scholarly page can carry
    more annotation than text — so the test is where the words live: nearly
    all of them inside foot-shaped paragraphs.
    """
    if re.search(r"<h[1-6][\s>]", below):
        return False
    first = re.search(r"<(p|ol|ul|table|figure|blockquote)[^>]*>", below)
    if first is None or first.group(1) != "p":
        return False
    if not FOOT_SHAPE.match(below[first.start():]):
        return False
    total = len(_strip_tags(below).split())
    if not total:
        return False
    shaped = sum(len(_strip_tags(m.group(0)).split())
                 for m in re.finditer(r"<p[^>]*>.*?</p>", below, re.S)
                 if FOOT_SHAPE.match(m.group(0)))
    return shaped >= 0.8 * total


def _foot_block_without_a_rule(body: str) -> int:
    """
    Where the feet begin on a page the layout gave no rule, or 0 if unsure.

    The printed rule is what normally separates the body from its feet, and
    where the recogniser sees one there is nothing to guess. It does not
    always see one: seven pages of When Protest Becomes Crime carry a plain
    foot block under no rule at all, and the pass skipped them at its first
    line.

    What stands in for the rule is where the feet sit and how they open. A
    footnote block is the LAST thing on its page and every paragraph in it
    leads with its own number — and a paragraph never leads with a
    superscript otherwise, which is the same reasoning the endnote parser
    uses to tell a citation from a marker in running prose. So the trailing
    run of such paragraphs is the block, and it counts only if what precedes
    it is real prose: a page that is nothing but numbered paragraphs is a
    list, or a notes page, and neither is a page with feet under it.
    """
    paras = list(re.finditer(r"<p[^>]*>.*?</p>", body, re.S))
    if len(paras) < 2:
        return 0
    lead = re.compile(r"<p[^>]*>\s*<sup>\s*\d{1,3}\s*</sup>\s*\S")
    k = len(paras)
    while k > 0 and lead.match(paras[k - 1].group(0)):
        k -= 1
    if k == len(paras) or k == 0:
        return 0                      # no trailing run, or the whole page
    # anything after the last body paragraph belongs to the block
    start = paras[k - 1].end()
    body_words = len(_strip_tags(body[:start]).split())
    foot_words = len(_strip_tags(body[start:]).split())
    if body_words < 30 or not foot_words:
        return 0                      # too little page above to be a page
    return start


def link_footnotes(bodies: list[str], allow_numbered: bool = False) -> dict:
    """
    Same-page footnotes, bound where the page itself settles the match.

    The older convention: an asterisk after a word, answered at the foot of the
    same page below a rule — which the recogniser preserves as a literal <hr/>.
    Scope is the page, so the only question is arity: link a symbol when it
    appears exactly once in the prose above the rule and exactly once below
    it. Any other count on that page and its symbols stay plain.

    Numbered footnotes obey the same page scope and the same arity rule, and
    need guards a symbol does not. Digits are everywhere in a book, so a
    marker counts only as a bare <sup>, never as a number met in prose; a
    rule is not always a footnote rule, so what follows one must be shaped
    like feet before any number binds across it. And the pass runs at all
    only when the CALLER, which can see the whole book, says the book's
    numbered markers are not spoken for: a book with an endnotes section
    resolves its numbers there, and a same-page pass grabbing them first
    would bind marker 1 to whatever paragraph below a rule happens to open
    "1." — The Invisible Government mixes symbol footnotes with numbered
    endnotes on the same pages, and is exactly who this flag protects.
    Every guard costs recall on a page laid out unusually, which is the side
    to err on — an unlinked marker is the page as printed.
    """
    stats = {"linked": 0, "numbered": 0, "skipped": 0, "ruleless": 0}
    for pi, body in enumerate(bodies):
        rule = "<hr/>"
        if rule in body:
            above, below = body.split(rule, 1)
        else:
            # No rule drawn. Only the numbered pass goes on without one, and
            # only where the page's own shape says where the feet begin —
            # symbols have no such trailing-run signature to stand on.
            cut = _foot_block_without_a_rule(body) if allow_numbered else 0
            if not cut:
                continue
            above, below, rule = body[:cut], body[cut:], ""
            stats["ruleless"] += 1
        for k, sym in enumerate(FOOTNOTE_SYMBOLS):
            foot = list(re.finditer(rf"<p>\s*({sym})(?![*†‡])\s*", below))
            mark = list(re.finditer(rf"(?<=\S)({sym})(?![*†‡])", above))
            if len(foot) != 1 or len(mark) != 1:
                if foot or mark:
                    stats["skipped"] += 1
                continue
            fid = f"fn-p{pi}-{k}"
            rid = f"fnref-p{pi}-{k}"
            m, f = mark[0], foot[0]
            below = (below[:f.start()]
                     + f'<p id="{fid}"><a href="#{rid}">{f.group(1)}</a> '
                     + below[f.end():])
            above = (above[:m.start()]
                     + f'<sup><a epub:type="noteref" id="{rid}" href="#{fid}">'
                     + f'{m.group(1)}</a></sup>'
                     + above[m.end():])
            stats["linked"] += 1

        # numbered footnotes, once the symbols have taken their markers (a
        # linked one is no longer a bare <sup> and cannot be claimed twice)
        if allow_numbered and _looks_like_foot_block(below):
            mark_by_n: dict[int, list] = {}
            for m in re.finditer(r"<sup>\s*(\d{1,3})\s*</sup>", above):
                mark_by_n.setdefault(int(m.group(1)), []).append(m)
            foot_by_n: dict[int, list] = {}
            for m in FOOT_ENTRY.finditer(below):
                foot_by_n.setdefault(int(m.group(1)), []).append(m)
            both = sorted(set(mark_by_n) & set(foot_by_n))
            usable = [n for n in both
                      if len(mark_by_n[n]) == 1 and len(foot_by_n[n]) == 1]
            stats["skipped"] += len(both) - len(usable)
            # rewrite from the end of each half so earlier offsets stay valid
            for n in sorted(usable, key=lambda n: foot_by_n[n][0].start(),
                            reverse=True):
                f = foot_by_n[n][0]
                below = (below[:f.start()]
                         + f'<p id="fn-p{pi}-n{n}">'
                         + f'<a href="#fnref-p{pi}-n{n}">{n}.</a> '
                         + below[f.end():])
            for n in sorted(usable, key=lambda n: mark_by_n[n][0].start(),
                            reverse=True):
                m = mark_by_n[n][0]
                above = (above[:m.start()]
                         + f'<sup><a epub:type="noteref" id="fnref-p{pi}-n{n}" '
                         + f'href="#fn-p{pi}-n{n}">{n}</a></sup>'
                         + above[m.end():])
            stats["linked"] += len(usable)
            stats["numbered"] += len(usable)

        bodies[pi] = above + rule + below
    return stats


def _paired_groups_agree(body_groups, note_groups, min_judged: int = 4) -> bool:
    """
    Do these pairings look like the same chapters, or like a shifted list?

    The highest marker in a chapter's prose and the highest note in its group
    are the same citation, so where the pairing is right the two numbers meet.
    Where it is shifted they do not, and no per-marker check can tell: every
    chapter numbers from 1, so each marker finds SOME note and the run reports
    a full set of links.

    Judged loosely on each pair and strictly in aggregate — a single chapter
    may legitimately disagree, where a note went unread or a marker was taken
    as a footnote, but a shifted list disagrees nearly everywhere. And judged
    only with enough pairs to tell those apart: across two groups, one honest
    disagreement and a shift look identical, so the rule keeps quiet unless
    the counts already failed to match, which is the case where the pairing
    was a guess to begin with.
    """
    agree = judged = 0
    for bg, ng in zip(body_groups, note_groups):
        if not bg or not ng:
            continue
        judged += 1
        b_top = max(n for _, n, _ in bg)
        n_top = max(n for _, n, _ in ng)
        if abs(b_top - n_top) <= max(2, 0.25 * max(b_top, n_top)):
            agree += 1
    if judged < min_judged:
        return True                    # too little to tell; other rules apply
    return agree >= 0.75 * judged


# ── typography restoration ───────────────────────────────────────────────────

# Words that legitimately OPEN with an apostrophe, where the mark elides
# letters rather than opening a quotation: "'tis", "'em", "the '90s". An
# opening single quote before these would be wrong far more often than right.
_ELISIONS = re.compile(r"(?:tis|twas|twere|em|cause|til|till|round|bout|"
                       r"neath|gainst|\d0s?)\b", re.I)


def _requote(text: str) -> str:
    """Directional quotes for one run of plain text, letters untouched.

    The printed page had directional quotes; the recogniser flattened them
    to straight ones — two and a half thousand in one book. Putting them
    back is restoration, not modernisation, which is why this pass exists
    and the spelling-modernising kind never will. The rules are the
    typesetter's own: a quote after space or an opening bracket opens, any
    other closes; a straight apostrophe between letters, after a letter, or
    eliding ("'tis", "'90s") is ’.
    """
    out = []
    for i, ch in enumerate(text):
        if ch == '"':
            prev = text[i - 1] if i else " "
            out.append("“" if prev in " \t\n(—–[{‘“"
                       else "”")
        elif ch == "'":
            prev = text[i - 1] if i else " "
            nxt = text[i + 1] if i + 1 < len(text) else " "
            if prev.isalnum():
                out.append("’")           # contraction or possessive
            elif _ELISIONS.match(text[i + 1:]):
                out.append("’")           # elision keeps the apostrophe
            elif prev in " \t\n(—–[{“" and (nxt.isalnum()
                                                          or nxt in "“\""):
                out.append("‘")
            else:
                out.append("’")
        else:
            out.append(ch)
    return "".join(out)


def normalize_typography(bodies: list[str],
                         page_paths: "list[str] | None" = None) -> dict:
    """
    Restore what the recogniser flattened: quote direction, and the double
    spaces no printed page contains. Deliberately NOT normalised: spaced
    ellipses (". . ." is how older houses set them — that is the book),
    hyphens standing where a dash might (a guess between minus, range and
    dash would be wrong somewhere), and anything involving letters, which
    belong to the OCR and its witnesses. Text inside tags is never touched.

    Neither is a born-digital page. Its text is the publisher's own
    typesetting — restored or reconciled from the layer — and its
    punctuation is therefore the real thing, not a flattening to repair. A
    heuristic that overwrote it would be guessing over an answer sheet.
    Only pages the OCR alone speaks for are touched.
    """
    stats = {"quotes": 0, "spaces": 0, "native_skipped": 0}
    for i, body in enumerate(bodies):
        if page_paths is not None and i < len(page_paths) \
                and os.path.exists(page_paths[i] + ".native"):
            stats["native_skipped"] += 1
            continue
        parts = re.split(r"(<[^>]+>)", body)
        for j, part in enumerate(parts):
            if part.startswith("<"):
                continue
            n_q = part.count('"') + part.count("'")
            if n_q:
                part = _requote(part)
                stats["quotes"] += n_q
            collapsed = re.sub(r"(?<=\S)  +(?=\S)", " ", part)
            if collapsed != part:
                stats["spaces"] += 1
                part = collapsed
            parts[j] = part
        bodies[i] = "".join(parts)
    return stats


def unusual_characters(bodies: list[str]) -> "list[tuple[str, int]]":
    """Characters that usually mean the recogniser stumbled, with counts —
    the broken-ligature ¬, box-drawing strays, replacement characters.
    Reported for a human eye; nothing is ever changed on this evidence."""
    sus = Counter()
    for body in bodies:
        text = re.sub(r"<[^>]+>", "", body)
        for ch in text:
            o = ord(ch)
            if ch in "¬¦§¤°^~`|\\" or 0x2500 <= o <= 0x257F \
                    or ch == "�" or 0x0080 <= o <= 0x009F:
                sus[ch] += 1
    return sus.most_common(8)


# ── citations → bibliography ─────────────────────────────────────────────────

BIBLIOGRAPHY_HEAD = re.compile(
    r"<(h[1-6])[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    r"((?:SELECT(?:ED)?\s+)?BIBLIOGRAPHY|REFERENCES|WORKS\s+CITED)"
    r"\s*(?:</[a-zA-Z]+>\s*)*</\1>", re.I)
# an author-date entry opens with the author and puts the year alone in
# parentheses: "Bernards, N. (2019a) 'Understanding…'". A Chicago imprint
# never does — its year shares the parens with a publisher — which is what
# keeps a Chicago bibliography from minting author-date keys.
AD_ENTRY = re.compile(
    r"^\s*(?:<[^/][^>]*>\s*)*"
    r"((?:[A-Z][A-Za-z’'\-]*\s+){0,2}[A-Z][A-Za-z’'\-]{2,})[^<(]{0,120}?"
    r"\((\d{4}[a-z]?)\)")
YEAR_ALONE = re.compile(r"\((\d{4}[a-z]?)\)")
# how a cited surname may be spelled ahead of its particle
_PARTICLES = {"van", "von", "de", "der", "den", "di", "du", "la", "le",
              "mac", "mc", "o", "ten", "ter"}


def _norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", re.sub(r"\s+", " ", t.lower())).strip()


def _surname_forms(author: str) -> set:
    """The spellings under which this entry's author may be cited."""
    parts = re.sub(r"\s+", " ", author).strip().rstrip(",").split()
    if not parts:
        return set()
    out = {parts[-1]}
    if len(parts) > 1 and parts[-2].lower().rstrip(".") in _PARTICLES:
        out.add(" ".join(parts[-2:]))
    return {s for s in out if len(s) > 2}


def _outside_markup(body: str, start: int, end: int) -> bool:
    """Whether [start:end) is plain text: inside no tag and no anchor."""
    if "<" in body[start:end]:
        return False
    lt, gt = body.rfind("<", 0, start), body.rfind(">", 0, start)
    if lt > gt:
        return False                      # inside a tag
    opens = len(re.findall(r"<a[\s>]", body[:start]))
    return opens == body.count("</a>", 0, start)


def parse_bibliography(bodies: list[str]) -> "dict | None":
    """
    The bibliography as a set of addressable entries, anchored where they sit.

    Two shapes are read, and only two — a style this parser has not met is a
    style it must not guess at. Author-date entries carry their year alone in
    parentheses and key on (surname, year). Chicago entries open surname-first
    with the title in italics and key on (surname, title); an entry with no
    author of its own inherits the surname above it, which is how Chicago
    sets a second work by the same hand.
    """
    hit = next(((k, m) for k, b in enumerate(bodies)
                for m in [BIBLIOGRAPHY_HEAD.search(b)] if m), None)
    if hit is None:
        return None
    start, m = hit
    rank = int(m.group(1)[1])
    end = len(bodies)
    for k in range(start + 1, len(bodies)):
        hm = re.search(r"<h([1-6])[^>]*>", bodies[k])
        if hm and int(hm.group(1)) <= rank \
                and not BIBLIOGRAPHY_HEAD.search(bodies[k]):
            end = k
            break

    entries = []                     # {id, page, surnames, years, title}
    ad_keys: dict = {}               # (form_lower, year) -> entry | "ambiguous"
    ti_keys = []                     # (surname, title_norm, entry)
    last_surname = None
    for k in range(start, end):
        body = bodies[k]
        edits = []
        for bm in re.finditer(r"<(p|li)([^>]*)>(.*?)</\1>", body, re.S):
            own = bm.group(3)
            eid = f"bib-{k:04d}-{bm.start()}"
            im = re.search(r"<i>(.*?)</i>", own, re.S)
            title = _norm_title(re.sub(r"<[^>]+>", "", im.group(1))) if im else ""
            adm = AD_ENTRY.match(own)
            surnames: set = set()
            years: set = set()
            if adm:
                surnames = _surname_forms(adm.group(1))
                years = set(YEAR_ALONE.findall(re.sub(r"<[^>]+>", "", own)))
            elif im:
                lead = re.sub(r"<[^>]+>", "", own[:im.start()]).strip()
                sm = re.match(r"([A-Z][A-Za-z’'\-]{2,}),", lead)
                if sm:
                    surnames = _surname_forms(sm.group(1))
                    last_surname = sm.group(1)
                elif not lead and last_surname:
                    surnames = _surname_forms(last_surname)
            if not surnames:
                continue
            entry = {"id": eid, "page": k, "surnames": surnames,
                     "years": years, "title": title}
            entries.append(entry)
            for s in surnames:
                for y in years:
                    # keyed lowercase so Adams and ADAMS collide as they
                    # should, but the ORIGINAL spelling rides along: the
                    # prose is searched for "AFI", not a guessed "Afi"
                    key = (s.lower(), y)
                    ad_keys[key] = ("ambiguous" if key in ad_keys
                                    else (entry, s))
                if len(title) > 3:
                    ti_keys.append((s, title, entry))
            if 'id="' not in bm.group(2):
                edits.append((bm.start(), f"<{bm.group(1)} id=\"{eid}\""
                              + bm.group(2) + ">", bm.end(1) + bm.start()))
        # anchor the entries; ids add no visible words by construction
        for pos, opening, _ in sorted(edits, reverse=True):
            tag_end = body.index(">", pos) + 1
            body = body[:pos] + opening + body[tag_end:]
        bodies[k] = body
    if not entries:
        return None
    return {"start": start, "end": end, "entries": entries,
            "ad_keys": ad_keys, "ti_keys": ti_keys}


# ── the index → the pages it names ───────────────────────────────────────────

INDEX_HEAD = re.compile(r"<(h[1-3])[^>]*>\s*(?:<[^/][^>]*>\s*)*INDEX\s*"
                        r"(?:</[a-zA-Z]+>\s*)*</\1>", re.I)
# An index that says its numbers are not pages is believed. Field manuals
# print the warning outright — "Entries are by paragraph number" — and their
# 10-71 designations would collide with the folio map's own encoding of
# hyphenated page numbers if the warning were ignored.
INDEX_NOT_PAGES = re.compile(r"\bby\s+paragraph|\bto\s+paragraph\s+numbers?",
                             re.I)
# A page reference as an index prints one: a number, possibly a range with
# either dash, possibly abbreviated ("167–8" means 167 to 168). A slash
# neighbour is a name like 9/11, never a page.
INDEX_REF = re.compile(r"(?<![\d/(])(\d{1,4})(\s*[-–]\s*(\d{1,4}))?(?![\d/])")


def _trailing_runs(text: str):
    """The spans of TEXT that are an entry's page references.

    An index entry is a name followed by numbers, and the name may itself
    contain digits — "1848 revolutions, 12" cites one page, not two. What
    separates the name's numbers from the references is position: the
    references are the maximal trailing run of number tokens with nothing
    but separators between them. Semicolons start a fresh sub-entry, so the
    rule applies per segment.
    """
    out = []
    for seg in re.finditer(r"[^;]+", text):
        toks = list(INDEX_REF.finditer(seg.group(0)))
        if not toks:
            continue
        base = seg.start()
        keep = []
        end = len(seg.group(0))
        for m in reversed(toks):
            between = seg.group(0)[m.end():end]
            if re.search(r"[A-Za-z]", re.sub(r"[a-z]{1,2}\b", "", between, count=1)):
                break
            keep.append(m)
            end = m.start()
        tail = seg.group(0)[max(k.end() for k in keep):] if keep else ""
        if keep and re.search(r"[A-Za-z]{3,}", tail):
            pass                          # letters after the run: not trailing
        else:
            out += [(base + m.start(), base + m.end(),
                     int(m.group(1))) for m in keep]
    return out


def link_index(bodies: list[str], folio_to_page: dict, dropped: set) -> dict:
    """
    Make the index's page numbers go where they point.

    The two ends already exist: the index prints folios, and the page-list
    anchored every folio the audit trusts. This joins them. Only numbers in
    an entry's trailing reference run are touched, only when the folio is in
    the map, and a book whose index declares itself a paragraph index is
    skipped with that reason recorded. Cross-references ("See Animal
    Enterprise Protection Act") carry no numbers and pass untouched. The
    word-preservation rule from the citation linker applies whole: an edited
    page that does not read back identical is reverted and counted.
    """
    stats = {"linked": 0, "unknown_folio": 0, "reverted": 0,
             "skipped_paragraph_index": False, "pages": 0}
    hit = next(((k, m) for k, b in enumerate(bodies)
                for m in [INDEX_HEAD.search(b)] if m), None)
    if hit is None:
        return stats
    start, m = hit
    rank = int(m.group(1)[1])
    end = len(bodies)
    for k in range(start + 1, len(bodies)):
        hm = re.search(r"<h([1-3])[^>]*>", bodies[k])
        if hm and int(hm.group(1)) <= rank and not INDEX_HEAD.search(bodies[k]):
            end = k
            break
    intro = _strip_tags(bodies[start])[:400]
    if INDEX_NOT_PAGES.search(intro):
        stats["skipped_paragraph_index"] = True
        return stats

    for k in range(start, end):
        if k in dropped:
            continue
        body = bodies[k]
        before = _strip_tags(body).split()
        edits = []
        # text nodes only, never inside a tag or an anchor
        pos = 0
        for part in re.split(r"(<[^>]+>)", body):
            if part.startswith("<"):
                pos += len(part)
                continue
            for a, z, num in _trailing_runs(part):
                target = folio_to_page.get(num)
                if target is None or target in dropped:
                    stats["unknown_folio"] += 1
                    continue
                if not _outside_markup(body, pos + a, pos + z):
                    continue
                edits.append((pos + a, pos + z,
                              f'<a href="page_{target:04d}.xhtml'
                              f'#pgb-{target:04d}">{body[pos+a:pos+z]}</a>'))
            pos += len(part)
        if not edits:
            continue
        edits.sort(reverse=True)
        taken = len(body) + 1
        applied = 0
        for a, z, new in edits:
            if z > taken:
                continue
            body = body[:a] + new + body[z:]
            taken = a
            applied += 1
        if _strip_tags(body).split() != before:
            stats["reverted"] += 1
            continue
        bodies[k] = body
        stats["linked"] += applied
        stats["pages"] += 1
    return stats


def apply_reviewer_decisions(bodies: list, decisions_path: str) -> dict:
    """A review sheet's exported decisions, reapplied at rebuild time —
    before the linkers run, so a corrected word can still earn its anchor.

    Same contract as `codicology apply`: text nodes only, the recorded
    occurrence, and a site that no longer exists is stale — reported and
    left as read, never guessed at. The reviewer read the shipped book;
    a rebuild may differ from it in quote shape, so the straight/curly
    variants of the site are tried before declaring it stale."""
    from .adjudicate import _TYPO
    from .review import _apply_to_xhtml
    try:
        with open(decisions_path) as fh:
            rows = json.load(fh).get("decisions", [])
    except (OSError, ValueError) as exc:
        print(f"    [!] decisions file unreadable, ignored: {exc}")
        return {"applied": 0, "stale": 0}
    curl = str.maketrans({"'": "’"})
    applied = stale = 0
    # highest occurrence first, so an earlier replacement of the same word
    # on the same page cannot renumber a later one
    for d in sorted(rows, key=lambda d: -d.get("occurrence", 0)):
        i = d.get("page")
        if not isinstance(i, int) or not (0 <= i < len(bodies)):
            stale += 1
            continue
        got = None
        tried = set()
        for old in (d["old"], d["old"].translate(curl),
                    d["old"].translate(_TYPO)):
            if old in tried:
                continue
            tried.add(old)
            got = _apply_to_xhtml(bodies[i], old, d["new"],
                                  d.get("occurrence", 0))
            if got is not None:
                break
        if got is None:
            stale += 1
            print(f"    stale decision: p{i} {d['old']!r} -> {d['new']!r} "
                  f"— site changed since review, kept as read")
        else:
            bodies[i] = got
            applied += 1
    return {"applied": applied, "stale": stale}


def link_citations(bodies: list[str], dropped: set) -> dict:
    """
    Bind in-text citations to the bibliography, driven from the bibliography.

    The prose is never searched for anything shaped like a citation — it is
    searched for the specific names and years the bibliography actually
    holds. "New York: Harper, 1933" can never bind because no author is
    called New; the junk that a general pattern must catch and filter is
    simply never a candidate. A key naming two entries binds nothing.

    Two styles, matching the corpus: author-date in running prose
    ("Bernards (2019a)", "(see Adams 1971)", "(AFI 2010:1)"), and Chicago
    short-form inside NOTE paragraphs only, where the italic title names the
    work ("Altgeld, <i>Live Questions</i>, p. 525"). Note paragraphs are the
    boundary because running prose italicises titles in passing mention,
    and a mention is not a citation.

    Unlike every other linker here this one rewrites inside running
    sentences, so each edited page must read back word-for-word identical —
    a page that does not is reverted whole and counted, never shipped.
    """
    stats = {"authordate": 0, "chicago": 0, "ambiguous": 0,
             "reverted": 0, "entries": 0}
    bib = parse_bibliography(bodies)
    if bib is None:
        return stats
    stats["entries"] = len(bib["entries"])

    ad = {k: e for k, e in bib["ad_keys"].items() if e != "ambiguous"}
    stats["ambiguous"] += sum(1 for e in bib["ad_keys"].values()
                              if e == "ambiguous")
    NOTE_P = re.compile(r"<(p|li)[^>]*(?:id=\"(?:note-|fn-)[^\"]*\")[^>]*>"
                        r"|<(p|li)[^>]*>\s*(?:<sup>|\d{1,3}[.)]\s)")

    for k in range(0, bib["start"]):
        if k in dropped:
            continue
        body = bodies[k]
        before = _strip_tags(body).split()
        edits = []

        # author-date, in any prose on the page
        for (form, year), (entry, spelling) in ad.items():
            pat = re.compile(re.escape(spelling)
                             + r"(?:’s|'s)?(?:\s+et\s+al\.?)?[,\s]*\(?\s*"
                             + re.escape(year) + r"(?!\d)")
            for m in pat.finditer(body):
                if not _outside_markup(body, m.start(), m.end()):
                    continue
                edits.append((m.start(), m.end(),
                              f'<a href="page_{entry["page"]:04d}.xhtml'
                              f'#{entry["id"]}">{body[m.start():m.end()]}</a>',
                              "authordate"))

        # Chicago short form, inside note paragraphs only
        for bm in re.finditer(r"<(p|li)([^>]*)>(.*?)</\1>", body, re.S):
            if not NOTE_P.match(bm.group(0)):
                continue
            own, base = bm.group(3), bm.start(3)
            for im in re.finditer(r"<i>(.*?)</i>", own, re.S):
                t = _norm_title(re.sub(r"<[^>]+>", "", im.group(1)))
                if len(t) < 4:
                    continue
                window = re.sub(r"<[^>]+>", " ", own[:im.start()])[-90:]
                cands = {id(e): e for s, bt, e in bib["ti_keys"]
                         if (bt.startswith(t) or t.startswith(bt))
                         and re.search(r"\b" + re.escape(s) + r"\b", window)}
                if len(cands) > 1:
                    stats["ambiguous"] += 1
                    continue
                if len(cands) != 1:
                    continue
                e = next(iter(cands.values()))
                a, z = base + im.start(), base + im.end()
                opens = len(re.findall(r"<a[\s>]", body[:a]))
                if opens != body.count("</a>", 0, a):
                    continue              # already inside an anchor
                edits.append((a, z,
                              f'<a href="page_{e["page"]:04d}.xhtml'
                              f'#{e["id"]}">{body[a:z]}</a>', "chicago"))

        if not edits:
            continue
        # apply without overlaps, last first so offsets hold
        edits.sort(key=lambda t: t[0], reverse=True)
        taken_from = len(body) + 1
        applied = []
        for a, z, new, kind in edits:
            if z > taken_from:
                continue
            body = body[:a] + new + body[z:]
            taken_from = a
            applied.append(kind)
        # The assertion this linker is not allowed to ship without: an
        # anchor adds no words, so the page must read back identical.
        if _strip_tags(body).split() != before:
            stats["reverted"] += 1
            continue
        bodies[k] = body
        stats["authordate"] += applied.count("authordate")
        stats["chicago"] += applied.count("chicago")
    return stats


def link_notes(bodies: list[str], dropped: set[int],
               chapter_starts: "list[tuple[int, int]] | None" = None) -> dict:
    """
    Bind prose markers to their endnotes, both ways, where the binding is sure.

    The rule everything here obeys: a marker is linked only when its chapter
    scope is settled and exactly one note answers to its number there. An
    unlinked superscript is the page as it always was; a wrong link sends the
    reader to someone else's citation and looks authoritative doing it. When
    the body's chapter groups and the notes' chapter groups cannot be aligned
    one to one, whole chapters go unlinked and the count is reported, not
    hidden.
    """
    stats = {"linked": 0, "unlinked": 0, "groups": 0, "misaligned": False}
    notes_start, note_groups = parse_notes_section(bodies)
    if notes_start < 0 or not note_groups:
        return stats
    group_numbers = getattr(parse_notes_section, "last_group_numbers", [])
    if chapter_starts and group_numbers and all(n is not None for n in group_numbers):
        # groups name their chapters outright, and the contents told us where
        # each chapter begins: a marker's page places it, no reset-guessing
        markers = []
        for i in range(max(0, notes_start)):
            for m in BODY_MARKER.finditer(bodies[i]):
                markers.append((i, int(m.group(1)), m.start()))
        def chapter_of(page):
            c = None
            for no, start_pg in chapter_starts:
                if start_pg <= page:
                    c = no
            return c
        by_no = {no: [] for no in group_numbers}
        for (pi, n, pos) in markers:
            c = chapter_of(pi)
            if c in by_no:
                by_no[c].append((pi, n, pos))
        body_groups = [by_no[no] for no in group_numbers]
    else:
        body_groups = [g for g in find_body_marker_groups(bodies, notes_start)
                       if len(g) >= 2]
    if len(body_groups) != len(note_groups):
        # order-based pairing has no anchor if the counts disagree; pair the
        # longest matching prefix only when the tail is what differs
        k = min(len(body_groups), len(note_groups))
        if abs(len(body_groups) - len(note_groups)) > 2:
            stats["misaligned"] = True
            return stats
        body_groups, note_groups = body_groups[:k], note_groups[:k]
        stats["misaligned"] = True

    # Pairing by position assumes any group missing from one side is missing
    # from the END. When it is missing from the FRONT — a notes section whose
    # introduction group went unrecognised — every pair is off by one and
    # every marker still finds a note, because chapters all number from 1.
    # Nothing downstream would notice: it reads as a full set of links. So
    # the pairs are checked against each other first. A chapter's last marker
    # and its last note are the same citation, so their numbers agree; paired
    # across a shift they do not, and four markers sitting against
    # twenty-five notes is the shape that gives it away.
    if not _paired_groups_agree(body_groups, note_groups,
                                min_judged=2 if stats["misaligned"] else 4):
        stats["misaligned"] = True
        return stats

    # rewrite from the last page backward so match offsets stay valid
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for g, (bg, ng) in enumerate(zip(body_groups, note_groups)):
        stats["groups"] += 1
        notes_by_n: dict[int, tuple[int, int]] = {}
        for (pi, n, pos) in ng:
            notes_by_n.setdefault(n, (pi, pos))
        seen_backref: set[int] = set()
        for (pi, n, pos) in bg:
            target = notes_by_n.get(n)
            if target is None or pi in dropped:
                stats["unlinked"] += 1
                continue
            tpi, tpos = target
            m = BODY_MARKER.match(bodies[pi][pos:])
            ref_id = f"ref-g{g}-{n}"
            note_id = f"note-g{g}-{n}"
            # A book may cite the same note twice in a chapter. Every marker
            # links forward; only the first carries the id, because it is the
            # one the note's backlink answers and ids must be unique.
            first = n not in seen_backref
            if first:
                # The entry must anchor the id BEFORE the marker may point at
                # it — a marker linked to an entry that then fails to anchor
                # is a broken href.
                anchored = _entry_anchor(bodies[tpi][tpos:], note_id,
                                         f"page_{pi:04d}.xhtml#{ref_id}", n)
                if anchored is None:
                    stats["unlinked"] += 1
                    continue
                span, back = anchored
                edits.setdefault(tpi, []).append((tpos, tpos + span, back))
                seen_backref.add(n)
            id_attr = f' id="{ref_id}"' if first else ""
            new = (f'<sup><a epub:type="noteref"{id_attr} '
                   f'href="page_{tpi:04d}.xhtml#{note_id}">{n}</a></sup>')
            edits.setdefault(pi, []).append((pos, pos + m.end(), new))
            stats["linked"] += 1
    for pi, page_edits in edits.items():
        b = bodies[pi]
        for a, z, new in sorted(page_edits, reverse=True):
            b = b[:a] + new + b[z:]
        bodies[pi] = b
    return stats


class PageFlag(NamedTuple):
    """A page worth a human's attention, and what specifically is wrong."""
    index: int
    kind: str            # hand | sparse | uncropped | blank
    detail: str
    action: str


# A hand over the page reads this much skin across the text body. Measured on
# this footage: a clear page sits under 0.03 and a page with fingers across the
# column reaches 0.10-0.23. The threshold sits above the clear pages and below
# anything that actually obscures words.
FLAG_HAND = 0.06
# A page holding this fraction of its neighbours' word count has almost
# certainly lost text to an occlusion or a bad crop, rather than simply being a
# short page — chapter ends run short but not this short.
FLAG_SPARSE = 0.35
# How far a page's shape may sit from the book's own median before it reads as
# a frame that was never cropped to the page at all.
FLAG_ASPECT = 0.35


# How much shorter than its neighbours a page must read before it is even
# considered a fragment rather than a page.
TURN_SHORTER = 0.5
# and how much of the little it does say must already appear on a page nearby.
# High deliberately: this is the test that makes discarding safe, because a
# fragment whose every phrase is already in the book costs nothing to lose.
TURN_CONTAINED = 0.7
# How far away to look for the page it repeats. A leaf caught in flight shows
# the page immediately beside it and no other, so this stays tight. Widening it
# starts eating real pages: at four, the title page was condemned because a
# preface several leaves later happens to name both authors, and matching on
# characters means short common words turn up in any long text.
TURN_WINDOW = 2


def find_page_turns(page_paths: list[str], texts: list[str] | None = None
                    ) -> list[tuple[int, str]]:
    """
    Captures that caught a leaf in flight rather than a page at rest.

    Identified by what they say, not by how they look. Three image measures were
    tried first and all three failed on a known fragment from this book — a page
    eighty percent hidden behind the next leaf, whose few visible characters were
    cut off mid-word. It reads as perfectly ordinary: the covering leaf's edge and
    the desk behind it put its dark-pixel count *above* the book's median, its
    proportions are those of any page, and that same leaf edge spreads its
    apparent text across the full width. Meanwhile the genuinely low-ink pages
    were the title page and the chapter openings — the ones it would be worst to
    lose. Geometry cannot tell a covered page from an uncovered one.

    What is decisive is that a fragment repeats a page that is already in the
    book. So a capture is called a turn only when it says much less than the
    pages around it *and* nearly everything it does say is already on a page
    close by. That pairing is what makes discarding it safe: not a judgement that
    it looks bad, but a demonstration that nothing is lost with it.
    """
    if texts is None or not page_paths:
        return []

    # Matched on characters rather than on words, because a covered page's text
    # is cut off mid-word: "Chicag", "orgies a", "disrepai". Those are prefixes
    # of what the full page says, so as words they match nothing at all — the
    # first attempt at this scored a fragment's overlap with the very page it
    # was a fragment of at zero. As character runs they match exactly.
    def squash(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", t.lower())

    def stubs(t: str) -> list[str]:
        return [s for s in (re.sub(r"[^a-z0-9]+", "", w.lower()) for w in t.split())
                if len(s) >= 4]

    words = [len(t.split()) for t in texts]
    squashed = [squash(t) for t in texts]
    turns: list[tuple[int, str]] = []

    for i in range(len(page_paths)):
        near = [words[j] for j in range(max(0, i - 2), min(len(words), i + 3)) if j != i]
        typical = statistics.median(near) if near else 0
        # Too short to be a page here — but a page has to be substantial for
        # "shorter than its neighbours" to mean anything at all.
        if typical < 60 or words[i] >= typical * TURN_SHORTER:
            continue
        mine = stubs(texts[i])
        if len(mine) < 5:
            continue
        best, best_j = 0.0, None
        for j in range(max(0, i - TURN_WINDOW), min(len(page_paths), i + TURN_WINDOW + 1)):
            if j == i or len(squashed[j]) < 200:
                continue
            share = sum(1 for s in mine if s in squashed[j]) / len(mine)
            if share > best:
                best, best_j = share, j
        if best >= TURN_CONTAINED and best_j is not None:
            turns.append((i, f"{words[i]} words against about {typical:.0f} nearby, and "
                             f"{best * 100:.0f}% of them already on page {best_j}"))
    return turns


def flag_pages(page_paths: list[str], texts: list[str],
               page_ids: list[str] | None = None) -> list[PageFlag]:
    """
    Find pages a person should look at, from measurements that can be trusted.

    Deliberately narrow. The obvious extra signal — reading the printed folio
    from the running head and checking it follows the last one — is not here,
    because at this capture resolution that type is around eleven pixels tall
    and every reader tried on it, human and machine, misreads it often enough
    that the flag would cry wolf. What is measurable is measurable: skin across
    the text, a page far shorter than its neighbours, a frame that was never
    cropped, a page with no ink on it.
    """
    flags: list[PageFlag] = []
    words = [len(t.split()) for t in texts]
    aspects: list[float] = []
    for p in page_paths:
        try:
            with Image.open(p) as im:
                aspects.append(im.width / max(1, im.height))
        except OSError:
            aspects.append(0.0)
    median_aspect = statistics.median([a for a in aspects if a > 0] or [0.75])

    def name(i: int) -> str:
        return page_ids[i] if page_ids and i < len(page_ids) else str(i)

    # The skin mask is chroma bounds, and the warm paper of an older book sits
    # inside them wholesale — a 1938 printing measured 99.9% "skin" on every
    # page, which flagged the entire book as handled. What means something is
    # not the absolute reading but a page standing out from this book's own
    # paper: a hand on white stock is far above a near-zero baseline, and on tan
    # stock nothing can stand out at all — reporting nothing there is honest,
    # because the measurement genuinely cannot see a hand against that paper.
    skins: list[float] = []
    for path in page_paths:
        img = cv2.imread(path)
        if img is None:
            skins.append(0.0)
            continue
        h, w = img.shape[:2]
        body = img[int(h * 0.10):int(h * 0.92), int(w * 0.06):int(w * 0.94)]
        skins.append(float((_skin_mask(body) > 0).mean()) if body.size else 0.0)
    hand_bar = max(FLAG_HAND, statistics.median(skins) + 0.05) if skins else FLAG_HAND

    for i, path in enumerate(page_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        skin = skins[i]
        if skin > hand_bar:
            flags.append(PageFlag(
                i, "hand", f"{skin * 100:.0f}% of the page body reads as skin",
                f"re-shoot this page: --patch {name(i)}=<photo>"))

        near = [words[j] for j in range(max(0, i - 2), min(len(words), i + 3)) if j != i]
        typical = statistics.median(near) if near else 0
        if typical >= 80 and words[i] < typical * FLAG_SPARSE:
            flags.append(PageFlag(
                i, "sparse", f"{words[i]} words against about {typical:.0f} nearby",
                f"check for a hand or a bad crop: --patch {name(i)}=<photo>"))

        if words[i] == 0:
            flags.append(PageFlag(
                i, "blank", "no text recognised",
                f"drop it if it is a blank leaf: --drop-pages {name(i)}"))

        if aspects[i] and abs(aspects[i] - median_aspect) / median_aspect > FLAG_ASPECT:
            flags.append(PageFlag(
                i, "uncropped", f"shape {aspects[i]:.2f} against a book median of "
                                f"{median_aspect:.2f} — the page outline was probably missed",
                f"re-shoot or drop: --patch {name(i)}=<photo>"))

    for i, why in find_page_turns(page_paths, texts):
        flags.append(PageFlag(
            i, "turn", f"looks like a page caught mid-turn — {why}",
            f"drop it: --drop-pages {name(i)}   (or --drop-turns for all of these)"))
    return flags


def write_review_sheet(
    output_html: str,
    page_paths: list[str],
    verdicts: list[PageVerdict],
    title: str,
    flags: list[PageFlag] | None = None,
    page_ids: list[str] | None = None,
) -> None:
    """
    Write a self-contained page showing every call the duplicate pass was unsure
    of, so it can be overruled.

    Only the doubtful pages go in. A sheet of all several hundred is a sheet
    nobody reads, and the confident verdicts are confident for a reason: the
    scores behind them sit at the extremes.

    Nothing here edits the book. The sheet's output is a list of page numbers to
    feed back through --drop-pages, which re-runs from the PDF in about a second
    rather than putting OCR through the whole book again.
    """
    dupes = [v for v in verdicts if v.status == "duplicate"]
    unjudged = [v for v in verdicts if v.status == "unjudged"]
    borderline = [v for v in verdicts if v.status == "borderline"]
    flags = flags or []
    if not (dupes or unjudged or borderline or flags):
        print("  nothing doubtful to review")
        return

    # A page that beats one earlier duplicate can still lose to a later one:
    # four shots of the same page chain as 60 loses to 62, then 62 itself
    # loses to 64. Shown pairwise, a card reading "62 kept" is only true
    # against 60 — three cards later, 62 has its own "dropped" card, and
    # nothing on the first card said so. Following each chain to wherever it
    # actually ends is what the reader needs on that first card, not the
    # immediate match, which is often just a stepping stone.
    by_index = {v.index: v for v in verdicts}

    def final_survivor(idx: int) -> int:
        seen = set()
        while idx not in seen:
            seen.add(idx)
            v = by_index.get(idx)
            if v is None or v.status != "duplicate" or v.match is None:
                return idx
            idx = v.match
        return idx  # a cycle would mean a scoring bug elsewhere; break rather than loop

    def card(idx: int, caption: str, checked: bool) -> str:
        uri = _thumb_data_uri(page_paths[idx])
        mark = " checked" if checked else ""
        return (f'<label class="c"><input type="checkbox" data-i="{idx}"{mark}>'
                f'<img src="{uri}" alt="page {idx}"><span>{caption}</span></label>')

    parts = [f"<h1>{html.escape(title)} — review</h1>",
             "<p class='lede'>Tick a page to leave it out of the book. Pages already "
             "dropped as duplicates start ticked, so the list below begins as what the "
             "duplicate pass decided; untick one to keep it, tick anything else to drop "
             "it. Run the whole command as given — it turns the duplicate pass off, so "
             "that what you tick here is what happens, rather than being decided again "
             "and overruling you.</p>"]

    if flags:
        # These are not duplicate calls and no checkbox fixes them — a page with
        # a thumb across the text needs a new photograph, which is a thing only
        # the person holding the book can do. Grouped by page so one trip to the
        # shelf settles everything about it.
        by_page: dict[int, list[PageFlag]] = {}
        for f in flags:
            by_page.setdefault(f.index, []).append(f)
        parts.append(
            f"<h2>Worth your eyes ({len(by_page)})</h2>"
            "<p>Pages the measurements say are damaged rather than repeated. Ticking "
            "does nothing useful here: most want a fresh photograph, which you can feed "
            "back with <code>--patch</code> without re-running the video. The suggested "
            "command is under each page. Note that a page can be perfectly sharp and "
            "still be wrong in a way no measurement catches — a misprinted folio, a leaf "
            "photographed out of turn — so this is a floor, not a guarantee.</p><div class='g'>")
        for idx in sorted(by_page):
            fs = by_page[idx]
            label = (page_ids[idx] if page_ids and idx < len(page_ids) else f"#{idx}")
            why = "; ".join(f.detail for f in fs)
            act = fs[0].action
            uri = _thumb_data_uri(page_paths[idx])
            parts.append(
                f'<div class="c"><img src="{uri}" alt="page {idx}">'
                f'<span><b>{html.escape(label)}</b> — {html.escape(why)}<br>'
                f'<code>{html.escape(act)}</code></span></div>')
        parts.append("</div>")

    if dupes:
        parts.append(f"<h2>Dropped as duplicates ({len(dupes)})</h2>"
                     "<p>Each pair is the page that was dropped beside the one kept "
                     "in its place — the page actually still in the book, even where "
                     "that took outlasting more than one other repeat.</p><div class='g'>")
        for v in dupes:
            survivor = final_survivor(v.match)
            label = (f"#{survivor} kept" if survivor == v.match
                     else f"#{survivor} kept · also outlasted #{v.match}")
            parts.append("<div class='pair'>"
                         + card(v.index, f"#{v.index} dropped · {v.score:.2f}", True)
                         + card(survivor, label, False)
                         + "</div>")
        parts.append("</div>")

    if unjudged:
        parts.append(f"<h2>Too little text to judge ({len(unjudged)})</h2>"
                     "<p>Blank pages, and pages a hand covered enough that OCR read "
                     "almost nothing. All were kept. Repeats sit next to each other.</p>"
                     "<div class='g'>")
        parts += [card(v.index, f"#{v.index}", False) for v in unjudged]
        parts.append("</div>")

    if borderline:
        parts.append(f"<h2>Borderline ({len(borderline)})</h2><div class='g'>")
        for v in borderline:
            parts.append("<div class='pair'>"
                         + card(v.index, f"#{v.index} · {v.score:.2f}", False)
                         + card(v.match, f"#{v.match}", False) + "</div>")
        parts.append("</div>")

    body = "\n".join(parts)
    doc = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(title)} — review</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:1.5rem 2rem 8rem;color:#111;background:#fafafa}}
 h1{{margin:0 0 .25rem}} h2{{margin:2rem 0 .25rem;font-size:1.05rem}}
 .lede,h2+p{{color:#555;margin:.15rem 0 .9rem;font-size:.9rem}}
 .g{{display:flex;flex-wrap:wrap;gap:.75rem}}
 .pair{{display:flex;gap:.35rem;padding:.4rem;background:#fff;border:1px solid #e3e3e3;border-radius:8px}}
 .c{{display:block;width:{REVIEW_THUMB_WIDTH}px;cursor:pointer;position:relative}}
 .c img{{width:100%;display:block;border-radius:5px;border:1px solid #ddd;background:#fff}}
 .c span{{display:block;font-size:.75rem;color:#666;padding-top:.2rem}}
 .c input{{position:absolute;top:.3rem;left:.3rem;width:1.1rem;height:1.1rem;z-index:1}}
 .c input:checked ~ img{{outline:3px solid #c0392b;opacity:.45}}
 footer{{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:1px solid #ddd;
   padding:.7rem 2rem;display:flex;gap:.75rem;align-items:center}}
 code{{background:#f2f2f2;padding:.35rem .5rem;border-radius:5px;flex:1;overflow:auto;white-space:nowrap}}
 button{{font:inherit;padding:.4rem .8rem;border-radius:6px;border:1px solid #bbb;background:#f6f6f6;cursor:pointer}}
 @media (prefers-color-scheme:dark){{
   body{{background:#151515;color:#eee}} .pair{{background:#1e1e1e;border-color:#333}}
   .c img{{border-color:#333;background:#222}} .lede,h2+p,.c span{{color:#aaa}}
   footer{{background:#1e1e1e;border-color:#333}} code{{background:#262626}}
   button{{background:#2a2a2a;border-color:#444;color:#eee}}}}
</style>
{body}
<footer><code id="out">--no-dedupe --drop-pages </code>
<button onclick="navigator.clipboard.writeText(document.getElementById('out').textContent.trim())">Copy</button></footer>
<script>
 const out=document.getElementById('out');
 function sync(){{
   const v=[...document.querySelectorAll('input[type=checkbox]')]
     .filter(c=>c.checked).map(c=>c.dataset.i).sort((a,b)=>a-b);
   out.textContent='--no-dedupe --drop-pages '+(v.join(',')||'(none)');
 }}
 document.addEventListener('change',sync); sync();
</script>"""
    with open(output_html, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"  review sheet: {output_html}  "
          f"({len(dupes)} dropped, {len(unjudged)} unjudged, {len(borderline)} borderline)")


# ── EPUB builder ──────────────────────────────────────────────────────────────

def _restrict_page_list_to_pagebreaks() -> None:
    """Make the EPUB's page-list mean what its name says.

    ebooklib decides what belongs in the page-list by asking whether an
    element carries both an epub:type and an id — never which epub:type. Our
    note markers carry exactly that pair, and necessarily: epub:type="noteref"
    is what earns a popup in a reading system, and the id is what the note's
    backlink answers. So every marker was filed as a printed page. One book
    shipped a page-list of 567 footnote markers labelled 1, 2, 3, restarting
    at each chapter. It is hidden, so nobody browsing a contents list would
    see it — but a reading system offering "go to page" would have obeyed it,
    which is a wrong link wearing the authority of the printed edition.

    The harvester is replaced rather than worked around. Dodging it — moving
    the id off the anchor so the pair never co-occurs — would work today and
    break silently the moment note ENTRIES take an epub:type="footnote" for
    popup support, which is a thing worth doing later. Correcting the rule
    survives that; the page-list purity test is what catches it if this stops
    being applied at all.
    """
    from ebooklib import utils as _eu

    if not hasattr(_eu, "get_pages"):       # loudly, not silently
        raise RuntimeError("ebooklib.utils.get_pages is gone: the page-list "
                           "harvester must be re-checked before shipping")

    def only_pagebreaks(item):
        body = _eu.parse_html_string(item.get_body_content())
        out = []
        for elem in body.iter():
            kind = (elem.get("epub:type")
                    or elem.get("{http://www.idpf.org/2007/ops}type") or "")
            if "pagebreak" not in kind.split() or elem.get("id") is None:
                continue
            label = ((elem.text or "").strip() or elem.get("aria-label")
                     or elem.get("title") or elem.get("id"))
            out.append((item.get_name(), elem.get("id"), label))
        return out

    _eu.get_pages = only_pagebreaks

def build_epub(
    page_paths: list[str],
    output_epub: str,
    backend: OCRBackend,
    title: str,
    embed_images: bool,
    dedupe: bool = True,
    review_sheet: str | None = None,
    ocr_cache: str | None = None,
    strip_furniture: bool = True,
    page_ids: list[str] | None = None,
    check_folios: bool = False,
    drop_blank: bool = True,
    cover_spec: str | None = "auto",
    link_notes_flag: bool = False,
    link_citations_flag: bool = False,
    link_index_flag: bool = False,
    typography_flag: bool = False,
    decisions_path: str | None = None,
) -> None:
    try:
        from ebooklib import epub
    except ImportError:
        sys.exit("EPUB output needs ebooklib: pip install ebooklib")
    _restrict_page_list_to_pagebreaks()

    print(f"  OCR backend: {backend.name} (batch size {backend.batch_size})")
    progress.emit(event="phase", phase="ocr",
                  message=f"OCR backend: {backend.name}")
    cache = OCRCache(ocr_cache, backend.name, backend.langs) if ocr_cache else None
    if cache and cache.entries:
        print(f"  OCR cache: {len(cache.entries)} pages remembered in {ocr_cache}")

    book = epub.EpubBook()
    book.set_title(title)
    book.set_language(backend.langs[0] if backend.langs else "en")

    # Load each batch from disk rather than holding every page in memory: a
    # 300-page book at full resolution is several GB of decoded pixels. Figures
    # go straight into the book for the same reason.
    bodies: list[str] = []
    page_figures: list[list[str]] = []  # figure files each page put in the book
    figure_data: dict[str, bytes] = {}  # their bytes, for spotting repeats
    page_furniture: list[list[str]] = []   # running head/foot text, per page
    furniture_known: list[bool] = []       # False = cached before heads were kept
    n_figures = 0
    n_blank_figures = 0
    n_label_rules = 0
    n_reattached = 0
    n_phantoms = 0
    marker_seq_state: dict = {"last": None}
    label_heads: list = []      # (page, rank, text) from SectionHeader
    toc_label_pages: set = set()
    total = len(page_paths)

    nonlocal_page_i = [0]      # the page emit_figure is currently working on

    def emit_figure(item: PageItem, caption: PageItem | None, caption_first: bool) -> str:
        """Store the crop in the book and wrap it, with its caption, in one <figure>."""
        nonlocal n_figures, n_blank_figures
        if not figure_has_content(item.figure):
            # A blank leaf the layout pass boxed for want of anything else to
            # call it. Its caption, if any, is still worth keeping: a plate's
            # caption sometimes sits on the facing blank.
            n_blank_figures += 1
            item.figure.close()
            return f"<p>{caption.html}</p>" if caption is not None else ""
        art = better_figure(page_paths[nonlocal_page_i[0]], item.box, item.figure)
        # Encode both ways and keep the smaller. Line art and charts come out
        # smaller as PNG *and* lossless — a drawing measured 6KB against 17KB
        # — because JPEG spends its bits ringing the hard black edges that
        # made this artwork look harsh in the first place. A photograph goes
        # the other way by more than four to one, and takes JPEG on merit.
        jb = io.BytesIO()
        art.save(jb, "JPEG", quality=88)
        pb = io.BytesIO()
        art.save(pb, "PNG", optimize=True)
        if len(pb.getvalue()) <= len(jb.getvalue()):
            data, ext, mime = pb.getvalue(), "png", "image/png"
        else:
            data, ext, mime = jb.getvalue(), "jpg", "image/jpeg"
        if art is not item.figure:
            art.close()
        name = f"images/fig_{n_figures:04d}.{ext}"
        current_figures.append(name)
        figure_data[name] = data
        book.add_item(epub.EpubImage(
            uid=f"fig{n_figures}",
            file_name=name,
            media_type=mime,
            content=data,
        ))
        try:
            item.figure.close()
        except Exception:
            pass
        n_figures += 1

        img_tag = f'<img src="{name}" alt="Figure" style="max-width:100%"/>'
        if caption is None:
            return f"<figure>{img_tag}</figure>"
        # figcaption is valid as either the first or last child, so keep it on the
        # side the printed page had it.
        cap = f"<figcaption>{caption.html}</figcaption>"
        return f"<figure>{cap}{img_tag}</figure>" if caption_first \
            else f"<figure>{img_tag}{cap}</figure>"

    for start in range(0, total, backend.batch_size):
        chunk_paths = page_paths[start : start + backend.batch_size]

        # Read only the pages this cache has not seen; the rest come straight back.
        remembered = [cache.get(p) if cache else None for p in chunk_paths]
        unseen = [p for p, hit in zip(chunk_paths, remembered) if hit is None]
        fresh: dict[str, list[PageItem]] = {}
        if unseen:
            opened = [Image.open(p).convert("RGB") for p in unseen]
            fresh = _read_pages_resiliently(unseen, opened, backend, cache)
            for im in opened:
                im.close()

        for path, hit in zip(chunk_paths, remembered):
            nonlocal_page_i[0] = len(bodies)   # the page these figures sit on
            items = hit if hit is not None else fresh[path]
            # Furniture is set aside, never merely deleted: its text is the
            # folio audit's evidence, already recognised at full resolution.
            page_furniture.append([_strip_tags(it.html) for it in items
                                   if it.is_furniture and it.html])
            furniture_known.append((hit is None)
                                   or (cache is not None and cache.knows_furniture(path)))
            page_no_for_labels = len(bodies)
            for it in items:
                if it.label in ("SectionHeader", "Title") and it.html \
                        and not it.is_furniture:
                    _t = re.sub(r"\s+", " ", _strip_tags(it.html)).strip()
                    _rk = re.match(r"\s*<h([1-6])", it.html)
                    if _t and not re.fullmatch(r"\d{1,3}", _t):
                        label_heads.append((page_no_for_labels,
                                            int(_rk.group(1)) if _rk else 3,
                                            _t[:90]))
                if it.label == "TableOfContents" and not it.is_furniture:
                    toc_label_pages.add(page_no_for_labels)
            furn_numbers = {m for it in items if it.is_furniture and it.html
                            for m in re.findall(r"\d{1,3}",
                                                _strip_tags(it.html))}
            items = [it for it in items if not it.is_furniture]
            _ra, _sp = reattach_orphan_markers(items, furn_numbers,
                                               marker_seq_state)
            n_reattached += _ra
            n_phantoms += _sp
            if strip_furniture:
                items = strip_running_heads(items)
            items = recover_placeholder_figures(items, path)
            items = recover_orphan_artwork(items, path)
            items = recover_vector_artwork(items, path)
            parts: list[str] = []
            current_figures: list[str] = []
            i = 0
            while i < len(items):
                item = items[i]
                nxt = items[i + 1] if i + 1 < len(items) else None

                # Books caption above as well as below, so bind a caption on
                # either side of the image rather than assuming one convention.
                if item.is_caption and nxt is not None and nxt.figure is not None:
                    parts.append(emit_figure(nxt, item, caption_first=True))
                    i += 2
                    continue

                if item.figure is not None:
                    if nxt is not None and nxt.is_caption:
                        parts.append(emit_figure(item, nxt, caption_first=False))
                        i += 2
                    else:
                        parts.append(emit_figure(item, None, caption_first=False))
                        i += 1
                    continue

                # The layout model's own word for a block outranks the shape
                # heuristics downstream — where a label exists. A page whose
                # feet the model has NAMED but whose printed rule the
                # recogniser did not draw gets its boundary here: the <hr/>
                # the foot linker keys on, synthesised from the label. Seven
                # pages of When Protest Becomes Crime needed exactly this and
                # got a trailing-run heuristic instead; the label is the
                # primary witness, the heuristic stays as the fallback for
                # every cache written before labels were kept.
                if item.label == "Footnote" \
                        and "<hr/>" not in "".join(parts) \
                        and not any(it.label == "Footnote" for it in items[:i]):
                    parts.append("<hr/>")
                    n_label_rules += 1
                html_out = item.html
                if item.label == "Formula" and "<math" in html_out:
                    # the label's assertion, stamped so normalize_math keeps
                    # its hands off real equations
                    html_out = re.sub(r"<math(?![^>]*display)",
                                      '<math display="block"', html_out)
                parts.append(html_out)
                i += 1
            bodies.append("".join(parts))
            page_figures.append(current_figures)
        done = min(start + len(chunk_paths), total)
        note = f" ({cache.hits} from cache)" if cache and cache.hits else ""
        print(f"    OCR'd {done}/{total} pages…{note}")
        progress.emit(event="progress", phase="ocr", done=done, total=total,
                      note=f"{cache.hits} from cache"
                           if cache and cache.hits else "")

    if cache:
        cache.save()
    if n_figures:
        print(f"    extracted {n_figures} figures")
    if n_blank_figures:
        print(f"    skipped {n_blank_figures} blank-page image(s)")
    if n_label_rules:
        print(f"    drew {n_label_rules} footnote rule(s) the layout labels "
              f"assert but the page never showed")
    if n_reattached:
        print(f"    reattached {n_reattached} superscript marker(s) the "
              f"layout had orphaned into standalone blocks")
    if n_phantoms:
        print(f"    suppressed {n_phantoms} phantom digit block(s) the "
              f"layout invented over prose — verified against the ink")

    # Decoration that recurs down the book is furniture, not artwork, and is
    # dropped here — before anything downstream counts what a page holds, so
    # a page carrying nothing but an ornament is judged on what it really has.
    furniture = find_image_furniture(page_figures, figure_data)
    if furniture:
        for i, names in enumerate(page_figures):
            hit = [n for n in names if n in furniture]
            if not hit:
                continue
            for name in hit:
                bodies[i] = re.sub(
                    r"<figure>(?:(?!</figure>).)*?" + re.escape(name)
                    + r"(?:(?!</figure>).)*?</figure>", "", bodies[i], flags=re.S)
                bodies[i] = re.sub(r"<img[^>]*" + re.escape(name) + r"[^>]*/?>",
                                   "", bodies[i])
            page_figures[i] = [n for n in names if n not in furniture]
        book.items = [it for it in book.items
                      if getattr(it, "file_name", None) not in furniture]
        freed = sum(len(figure_data[n]) for n in furniture if n in figure_data)
        for name in furniture:
            figure_data.pop(name, None)
        print(f"    dropped {len(furniture)} repeated decoration(s) as page "
              f"furniture, freeing {freed / 1048576:.1f} MB")

    bodies = [normalize_entry_heads(normalize_math(b)) for b in bodies]
    n_promoted = normalize_note_heads(bodies)
    if n_promoted:
        print(f"    promoted {n_promoted} bold note head(s) to headings")
    n_reconciled = reconcile_native_text(bodies, page_paths)
    if n_reconciled:
        print(f"    reconciled {n_reconciled} word(s) against the book's "
              f"own born-digital text")
    # After reconciliation, the lesson the chapter separator taught: the
    # publisher's layer keeps straight quotes, and reconciling after this
    # pass would quietly put them back.
    if typography_flag:
        tst = normalize_typography(bodies, page_paths)
        if tst["quotes"]:
            print(f"    typography: {tst['quotes']} quote mark(s) set "
                  f"directional, {tst['spaces']} double space(s) collapsed"
                  + (f"; {tst['native_skipped']} born-digital page(s) left "
                     f"to the publisher's own typesetting"
                     if tst["native_skipped"] else ""))
        odd = unusual_characters(bodies)
        if odd:
            print("    [~] characters that usually mean the OCR stumbled: "
                  + ", ".join(f"{c!r} x{n}" for c, n in odd))
    # After reconciliation, never before it. Reconciling corrects our reading
    # against the publisher's text, word by word; the merged opening
    # "Chapter 1. Economics" has no counterpart there, so running it first
    # let the correction quietly take the separator back out again.
    n_chapters = normalize_chapter_heads(bodies)
    if n_chapters:
        print(f"    squared up {n_chapters} chapter heading(s) to one rank")
    n_recovered = promote_missing_chapter_heads(bodies)
    if n_recovered:
        print(f"    recovered {n_recovered} chapter opening(s) the layout ran "
              f"into the prose")
    n_lists = normalize_list_markers(bodies)
    if n_lists:
        print(f"    stopped {n_lists} list(s) double-marking items that "
              f"already carry their own number or bullet")
    texts = [_strip_tags(b) for b in bodies]
    folios = folios_from_furniture(page_furniture) if page_furniture else None

    if check_folios:
        print("  Reading printed page numbers…")
        progress.emit(event="phase", phase="folios",
                      message="Reading printed page numbers")
        folios = folios or []
        # Pages cached before running heads were kept carry no furniture record,
        # and for those absence means nothing. Only they pay for a second look.
        legacy = [i for i, known in enumerate(furniture_known) if not known]
        if legacy:
            print(f"    {len(legacy)} page(s) cached before heads were kept — "
                  f"re-reading those")
            for i, f in zip(legacy, read_folios([page_paths[i] for i in legacy],
                                                backend)):
                folios[i] = Folio(i, f.number, f.text, f.confident)
                # Fold the reading back into the cache, upgrading the entry to
                # the head-keeping format: the second pass costs hours on a
                # long book, and without this every rebuild pays it again. A
                # head the pass could not read is recorded as absent — a small
                # untruth against what a fresh full-resolution pass might find,
                # but the honest alternative is paying for that pass, and the
                # pages it misses here were mostly true absences when measured.
                if cache is not None:
                    old_items = cache.get(page_paths[i])
                    if old_items is not None:
                        if f.text:
                            old_items = old_items + [PageItem(
                                html=f"<p>{html.escape(f.text)}</p>",
                                is_furniture=True)]
                        cache.put(page_paths[i], old_items)
        audit = audit_folios(folios)
        print(f"    read {audit.read} of {audit.total} confidently"
              + ("" if audit.read else " — too few to say anything about the order"))
        if audit.read:
            if audit.inversions:
                for a, b in audit.inversions:
                    fa = next(f.number for f in folios if f.index == a)
                    fb = next(f.number for f in folios if f.index == b)
                    print(f"    [!] out of order: page {a} is folio {fa}, "
                          f"page {b} is folio {fb} — consider --swap {a}:{b}")
            if audit.duplicates:
                for n, idxs in audit.duplicates:
                    print(f"    [!] folio {n} appears on pages {idxs}")
            if audit.gaps:
                for after, missing in audit.gaps:
                    shown = missing if len(missing) <= 8 else \
                        missing[:4] + ["…"] + missing[-2:]
                    print(f"    [!] {len(missing)} folio(s) missing after page {after}: {shown}")
            for idx, n in audit.misreads:
                print(f"    [~] page {idx} read as folio {n}, which fits neither "
                      f"neighbour — treated as a misread, not an ordering fault")
            if not (audit.inversions or audit.duplicates or audit.gaps):
                print("    order, numbering and coverage all check out")
        if audit.misreads:
            # The audit knows these readings fit nowhere, so the resolver must
            # not anchor on them: a chapter opener printing its chapter number
            # where the folio sits is exactly how "folio 8" landed chapter 2's
            # contents line on chapter 8's opening page. Demote, keep density.
            bad = {idx for idx, _ in audit.misreads}
            folios = [f if f.index not in bad else
                      Folio(f.index, f.number, f.text, False) for f in folios]

    verdicts = review_pages(texts)
    if review_sheet:
        flags = flag_pages(page_paths, texts, page_ids)
        if flags:
            kinds = ", ".join(f"{k}×{sum(1 for f in flags if f.kind == k)}"
                              for k in dict.fromkeys(f.kind for f in flags))
            print(f"    {len(flags)} page(s) flagged for a human look ({kinds})")
        write_review_sheet(review_sheet, page_paths, verdicts, title, flags, page_ids)

    if dedupe:
        # Duplicate detection defends against a page CAPTURED twice — a video
        # dwelling on one spread, a photographer shooting a page again. A
        # born-digital page cannot be that: the publisher's pagination is
        # authoritative, and two of its pages are two pages however alike they
        # read. Textbook problem sets are alike on purpose — consecutive
        # problems share their whole template and differ in one variable, and
        # one such page, whose own repeated sub-question left it few distinct
        # fragments to divide by, scored 0.569 against its neighbour and was
        # dropped with 290 words on it.
        native = {i for i, p in enumerate(page_paths)
                  if os.path.exists(p + ".native")}
        # How alike two pages must be before one is deleted depends on where
        # they came from. A camera shooting the same page twice produces two
        # different readings of it, so a loose bar is right. A PDF that holds
        # a page twice holds it identically — measured at 1.00 on a scan that
        # really did duplicate two leaves. At the camera's bar, that same
        # scan lost compound #20 of a chemical reference because its entry is
        # laid out like compound #24's: 0.59, and gone.
        from_pdf = any(os.path.exists(p + ".source.json") for p in page_paths)
        floor = 0.90 if from_pdf else DUPE_SIMILARITY
        dropped = {v.index for v in verdicts
                   if v.status == "duplicate" and v.index not in native
                   and v.score >= floor}
        if dropped:
            # The figures those pages contributed are now unreferenced.
            orphans = {name for i in dropped for name in page_figures[i]}
            book.items = [it for it in book.items
                          if getattr(it, "file_name", None) not in orphans]
            print(f"    dropped {len(dropped)} duplicate pages "
                  f"({total - len(dropped)} kept)")
    else:
        dropped = set()

    if drop_blank:
        # A page with no text and no picture on it has nothing to show. In a
        # facsimile PDF a blank leaf is worth keeping, because pagination is the
        # point; in a reflowable book it is an empty screen the reader swipes
        # past for no reason. Emptiness here is not a judgement about quality —
        # every fragment the page produced has already been assembled into
        # `bodies`, so a page that is empty at this point genuinely carries
        # nothing, and dropping it cannot lose anything.
        blank = {i for i, body in enumerate(bodies)
                 if i not in dropped and not _strip_tags(body).strip()
                 and "<img" not in body}
        # An empty body means one of two very different things: an empty
        # leaf, or a page this pipeline failed to read. Both fabrication
        # witnesses are deaf here — they judge text we produced, and we
        # produced none — so a failed page would be deleted with the word
        # "blank" beside it and nothing to tell it apart. Ask the classical
        # reader, which cannot invent: real words on a page we read as empty
        # mean it is not empty. Show-through from a facing map yields a few
        # junk fragments, so it takes a sentence's worth to overrule us.
        if blank:
            second_opinion = set()
            for i in sorted(blank):
                n = _classical_word_count(page_paths[i], min_len=3)
                if n is None:
                    # Nothing can tell an empty leaf from a page we failed to
                    # read, so this page is about to be deleted on our own
                    # word alone. Counted, and reported at the end.
                    UNWITNESSED["blank"] += 1
                elif n >= 10:
                    second_opinion.add(i)
            if second_opinion:
                blank -= second_opinion
                for i in sorted(second_opinion):
                    print(f"    [!] page {i} read as empty, but a classical "
                          f"reader finds text on it — kept, not dropped")
        if blank:
            orphans = {name for i in blank for name in page_figures[i]}
            book.items = [it for it in book.items
                          if getattr(it, "file_name", None) not in orphans]
            dropped |= blank
            print(f"    dropped {len(blank)} blank page(s) "
                  f"({total - len(dropped)} kept)")

    if decisions_path:
        dst = apply_reviewer_decisions(bodies, decisions_path)
        if dst["applied"] or dst["stale"]:
            print(f"    decisions: {dst['applied']} reviewer correction(s) "
                  f"reapplied"
                  + (f", {dst['stale']} stale" if dst["stale"] else ""))

    if link_notes_flag:
        # chapter starts, where the printed contents can supply them: a leaf
        # titled "5 THE CASE OF…" placed at page P starts chapter 5 there.
        # Books number their chapters in Arabic or in Roman and both readings
        # land on the same scale, so "I" and "1" alike open chapter one. That
        # also covers the OCR reading a printed 1 as an I, which is what the
        # narrower rule here used to be for.
        chapter_starts = []
        if folios is not None:
            placed_ch, _tocp = _place_toc_entries(bodies, folios, dropped)
            for e, tgt, _ok in placed_ch:
                mnum = re.match(r"^([IVXL]{1,6}|\d{1,2})\s+\S", e.title)
                # depth 1 included: a chapter row promoted to section rank
                # because subsections follow it is still a chapter start
                if e.depth >= 1 and mnum and tgt is not None:
                    tok = mnum.group(1)
                    no = int(tok) if tok.isdigit() else _roman_to_int(tok)
                    if no is not None:
                        chapter_starts.append((no, tgt))
        # Which linker answers a numbered marker is a property of the BOOK.
        # A Notes section, back-of-book or per-chapter, claims every numbered
        # superscript in the prose; only a book without one may read numbers
        # at the foot of the page as footnotes. Symbols carry no such
        # ambiguity — no endnote scheme uses an asterisk — so the symbol pass
        # runs regardless, and The Invisible Government, which mixes symbol
        # footnotes with numbered endnotes, gets both linkers each taking
        # only what is theirs.
        n_note_heads = sum(1 for b in bodies if NOTES_HEAD.search(b))
        fstats = link_footnotes(bodies, allow_numbered=(n_note_heads == 0))
        if fstats["linked"] or fstats["skipped"]:
            print(f"    footnotes: {fstats['linked']} same-page notes linked"
                  + (f" ({fstats['numbered']} numbered)"
                     if fstats["numbered"] else "")
                  + f", {fstats['skipped']} left unsure")
        if n_note_heads >= 2:
            # several "Notes" heads mean chapter-endnotes: each section
            # closes the chapter it follows, and position alone scopes them
            cstats = link_chapter_notes(bodies, dropped)
            if cstats["sections"]:
                print(f"    notes: {cstats['linked']} markers linked across "
                      f"{cstats['sections']} chapter sections, "
                      f"{cstats['unlinked']} left plain")
            else:
                # Counting "Notes" headings is a guess at the layout, not a
                # reading of it: a back-of-book section whose title happens to
                # appear twice looks like chapter endnotes and parses to
                # nothing. Falling back costs one more pass; not falling back
                # cost one book all 893 of its links, silently.
                nstats = link_notes(bodies, dropped=dropped,
                                    chapter_starts=chapter_starts or None)
                if nstats["groups"]:
                    print(f"    notes: {nstats['linked']} markers linked across "
                          f"{nstats['groups']} chapters, {nstats['unlinked']} "
                          f"left plain (chapter-section layout did not apply)")
                elif nstats["misaligned"]:
                    print("    notes: body and endnote chapters would not "
                          "align — nothing linked rather than linked wrongly")
                else:
                    print(f"    notes: {n_note_heads} Notes heads found but "
                          f"neither layout parsed — nothing linked")
        else:
            nstats = link_notes(bodies, dropped=dropped,
                                chapter_starts=chapter_starts or None)
            if nstats["groups"]:
                print(f"    notes: {nstats['linked']} markers linked across "
                      f"{nstats['groups']} chapters, {nstats['unlinked']} left plain"
                      + (", group alignment imperfect — tail chapters unlinked"
                         if nstats["misaligned"] else ""))
            elif nstats["misaligned"]:
                print("    notes: body and endnote chapters would not align — "
                      "nothing linked rather than linked wrongly")
            elif n_note_heads:
                print("    notes: a Notes head was found but its shape was "
                      "not understood — nothing linked")

    if link_citations_flag:
        # After the note linkers on purpose: the Chicago pass finds its note
        # paragraphs by the ids they leave behind.
        cst = link_citations(bodies, dropped)
        if cst["authordate"] or cst["chicago"]:
            print(f"    citations: {cst['authordate']} author-date and "
                  f"{cst['chicago']} note citation(s) bound to the "
                  f"bibliography's {cst['entries']} entries"
                  + (f", {cst['ambiguous']} ambiguous left alone"
                     if cst["ambiguous"] else ""))
        elif cst["entries"]:
            print(f"    citations: {cst['entries']} bibliography entries "
                  f"parsed but no citation bound")
        else:
            print("    citations: no bibliography found to bind against")
        if cst["reverted"]:
            print(f"    [!] citations: {cst['reverted']} page(s) reverted — "
                  f"a link would have changed the page's words")

    # The printed page numbers, written where a reading system can act on
    # them: a page-list is how "go to page 147" and a citation to the paper
    # edition survive reflowable text. The id is the page's own index, which
    # is unique by construction; the number goes in the label, where a folio
    # repeated by a misprint cannot collide with anything.
    if check_folios and folios:
        numbers, refused, distrusted = fill_folio_gaps(folios)
        if distrusted:
            print(f"    [~] {len(distrusted)} folio reading(s) distrusted for "
                  f"breaking page order, pages {distrusted[:8]}"
                  + ("…" if len(distrusted) > 8 else ""))
        for lo, hi, lo_no, hi_no in refused:
            print(f"    [~] pages {lo + 1}-{hi - 1} left out of the page list: "
                  f"{hi - lo} pages between folios {lo_no} and {hi_no} but "
                  f"{hi_no - lo_no} numbers — which page holds which is unknown")
        read_at = {f.index for f in folios
                   if f.confident and f.number is not None}
        anchored = restored = 0
        for i in sorted(numbers):
            if i in dropped or i >= len(bodies):
                continue
            bodies[i] = (f'<span epub:type="pagebreak" role="doc-pagebreak" '
                         f'id="pgb-{i:04d}" aria-label="{numbers[i]}"></span>'
                         + bodies[i])
            anchored += 1
            restored += i not in read_at
        if anchored:
            print(f"    page list: {anchored} printed page numbers anchored"
                  + (f", {restored} of them restored where the page carried "
                     f"no printed number" if restored else ""))
        if link_index_flag:
            # After the anchors exist, from the same folio map they used. A
            # folio the audit distrusted never entered `numbers`, so the
            # index cannot link through a misread.
            folio_to_page = {n: i for i, n in numbers.items()
                             if i not in dropped}
            ist = link_index(bodies, folio_to_page, dropped)
            if ist["skipped_paragraph_index"]:
                print("    index: declares paragraph references, not pages — "
                      "left untouched as printed")
            elif ist["linked"]:
                print(f"    index: {ist['linked']} page reference(s) linked "
                      f"across {ist['pages']} index page(s)"
                      + (f", {ist['unknown_folio']} named folios outside the "
                         f"map left plain" if ist["unknown_folio"] else ""))
            if ist["reverted"]:
                print(f"    [!] index: {ist['reverted']} page(s) reverted — "
                      f"a link would have changed the page's words")
    elif link_index_flag:
        print("    index: needs --check-folios for the folio map — nothing "
              "linked")

    chapters = []
    for i, page_body in enumerate(bodies):
        if i in dropped:
            continue
        body = ""
        if embed_images:
            # The page keeps whatever encoding it was stored in — a rendered
            # page is lossless PNG now, and naming it .jpg with a JPEG media
            # type would describe the file wrongly to every reader.
            ext = os.path.splitext(page_paths[i])[1].lower() or ".jpg"
            mime = {".png": "image/png", ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg", ".webp": "image/webp",
                    ".tif": "image/tiff", ".tiff": "image/tiff"}.get(
                        ext, "image/jpeg")
            name = f"images/page_{i:04d}{ext}"
            with open(page_paths[i], "rb") as fh:
                book.add_item(epub.EpubImage(
                    uid=f"img{i}",
                    file_name=name,
                    media_type=mime,
                    content=fh.read(),
                ))
            body += f'<img src="{name}" alt="Page {i + 1}" style="max-width:100%"/>'

        body += page_body

        if not body.strip():
            # A page can legitimately end up with nothing on it — a blank leaf
            # whose only "content" was the blank-page image just dropped. An
            # empty body is not merely ugly: ebooklib parses every page when it
            # writes the spine, and an empty document raises there, taking the
            # whole build down at the last step.
            body = f'<p class="blank">[blank page {i + 1}]</p>'

        ch = epub.EpubHtml(title=f"Page {i + 1}", file_name=f"page_{i:04d}.xhtml", lang="en")
        ch.content = f"<html><body>{body}</body></html>"
        book.add_item(ch)
        chapters.append(ch)

    # The book's own contents, pointed at the right pages via the folios. A
    # flat list of "Page N" names most cases; where the printed contents page
    # parsed and the folio map holds, the reader gets the chapters the book
    # itself declares, with the second half's numbered entries nested beneath.
    kept = [i for i in range(len(bodies)) if i not in dropped]
    pos_of = {orig: k for k, orig in enumerate(kept)}
    toc_built = None
    if folios is not None:
        placed, toc_pages = _place_toc_entries(bodies, folios, dropped)
        links: list = []
        verified = missed = 0
        # Titles are hunted in reading order, starting past the contents —
        # a book's half-title repeats the chapter's words before the book
        # has begun, and would otherwise win chapter one every time.
        hunt_from = max(toc_pages) if toc_pages else -1
        stack: list[tuple[int, list]] = [(-1, links)]
        for e, target, ok in placed:
            if ok:
                verified += 1
            elif target is not None and target not in pos_of:
                target = None
            if target is None:
                # No folio to resolve: the title text is the only address the
                # line has. Front matter sits in the opening pages, but a
                # chapter list without page numbers points anywhere in the
                # book, so the search runs forward from the last thing placed
                # — which keeps the contents in their own order for free — and
                # demands the title near the TOP of a page, where a chapter
                # announces itself, so a passing mention in prose cannot win.
                # The contents pages are excluded: every title appears there
                # by definition, which is how "Foreword" once pointed at the
                # table that listed it.
                search = [t for t in kept if t > hunt_from and t not in toc_pages]
                target = next((t for t in search
                               if _title_names_this_page(
                                   e.title, _strip_tags(bodies[t])[:300])),
                              None)
                if target is None and hunt_from < 0:
                    # Front matter — a Preface, a Foreword — sits in the
                    # opening pages and carries roman folios the audit never
                    # sees. Only before anything else has been placed, and
                    # only on the same strict terms: this fallback once put
                    # "Chapter 3: Demand and Supply" on chapter 2's opening
                    # page, which shared two of its three words.
                    target = next((t for t in kept[:30] if t not in toc_pages
                                   and _title_names_this_page(
                                       e.title, _strip_tags(bodies[t])[:400])),
                                  None)
                if target is not None:
                    hunt_from = target
            if e.depth < 2:
                # a grouping: BOOK SEVEN as printed, or a chapter promoted
                # because subsections follow it. A chapter that nests
                # children is still a destination, so it keeps its link
                # when it resolved; a grouping that never resolves is a
                # heading and no more.
                sect: list = []
                while stack and stack[-1][0] >= e.depth:
                    stack.pop()
                head = (epub.Section(e.title, href=f"page_{target:04d}.xhtml")
                        if target is not None and target in pos_of
                        else epub.Section(e.title))
                stack[-1][1].append((head, sect))
                stack.append((e.depth, sect))
                continue
            if target is None or target not in pos_of:
                missed += 1
                continue
            stack[-1][1].append(epub.Link(f"page_{target:04d}.xhtml", e.title,
                                          f"toc{len(links)}-{pos_of[target]}"))
        entries = [(i, t) for i, t, _ in find_numbered_entries(bodies) if i in pos_of]
        if entries:
            sect = [epub.Link(f"page_{i:04d}.xhtml", t, f"entry{i}") for i, t in entries]
            links.append((epub.Section("Entries"), sect))
        if links:
            toc_built = links
            print(f"    contents: {len(placed)} printed lines, "
                  f"{verified} title-verified, {missed} unplaced"
                  + (f", {len(entries)} numbered entries" if entries else ""))
    if not toc_built and len(label_heads) >= 4:
        # The layout model NAMED the book's headings; after the running-head
        # strip and the digit filter (markers masquerade as SectionHeader),
        # what survives is the book's own skeleton — measured 107 distinct
        # real heads on one book. Top rank becomes chapters, deeper ranks
        # nest beneath the chapter they follow.
        top = min(r for _, r, _ in label_heads)
        outline = []
        for pg, r, t in label_heads:
            if pg not in pos_of:
                continue
            link = epub.Link(f"page_{pg:04d}.xhtml", t,
                             f"lh{pg}-{len(outline)}")
            if r == top or not outline:
                outline.append([link, []])
            else:
                outline[-1][1].append(link)
        if len(outline) >= 3:
            toc_built = [(l, kids) if kids else l for l, kids in outline]
            n_kids = sum(len(k) for _, k in outline)
            print(f"    contents: no printed contents page; built "
                  f"{len(outline)} chapter(s)"
                  + (f" and {n_kids} nested section(s)" if n_kids else "")
                  + " from the layout's own heading labels")
    if not toc_built and toc_label_pages:
        print(f"    [~] layout labels mark page(s) "
              f"{sorted(toc_label_pages)[:4]} as a printed contents the "
              f"parser did not use")
    if not toc_built:
        # No printed contents parsed. Before falling back to a list of pages —
        # which is not a table of contents, and which the page list now
        # provides properly anyway — ask the book where its chapters start.
        heads = [(i, t) for i, t in chapter_head_contents(bodies)
                 if i in pos_of]
        if heads:
            toc_built = [epub.Link(f"page_{i:04d}.xhtml", t, f"chap{i}")
                         for i, t in heads]
            print(f"    contents: no printed contents page; built from "
                  f"{len(heads)} chapter headings in the text")
    book.toc = tuple(toc_built) if toc_built else tuple(chapters)

    if cover_spec == "auto":
        # A cover is an image with nearly nothing to say in words. The first
        # page of every book done so far — two cloth covers, two cover arts —
        # passes both halves of that; a book that opens straight into prose
        # fails the word test and gets no cover rather than a wall of text.
        first = kept[0] if kept else None
        cover_spec = None
        if first is not None and len(texts[first].split()) < 25:
            cover_spec = page_paths[first]
    if cover_spec:
        cov = _load_cover(cover_spec, page_paths, page_ids)
        if cov is not None:
            book.set_cover("cover.jpg", cov, create_page=False)
            print("    cover set")
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + chapters
    epub.write_epub(output_epub, book)
    print(f"  EPUB saved: {output_epub}")
    if UNWITNESSED["fabrication"] or UNWITNESSED["blank"]:
        print(f"  [!] built without a witness: "
              f"{UNWITNESSED['fabrication']} page(s) passed the fabrication "
              f"check unexamined, {UNWITNESSED['blank']} page(s) were "
              f"deleted as blank on this pipeline's word alone. Install "
              f"tesseract and rebuild to have either checked.")


# ── Main pipeline ─────────────────────────────────────────────────────────────

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic", ".heif", ".webp", ".bmp")

_HEIF_READY = False


def read_image(path: str) -> np.ndarray | None:
    """
    One photograph off disk, as OpenCV expects it.

    OpenCV cannot open HEIC at all, and that is the format every recent iPhone
    writes by default — so a folder straight off a phone reads as entirely empty
    without this. Pillow can, once pillow-heif registers itself, so the fallback
    goes through Pillow and hands back the same BGR array the rest of the
    pipeline works in.
    """
    img = cv2.imread(path)
    if img is not None:
        return img
    global _HEIF_READY
    if not _HEIF_READY:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            print("  [!] cannot read HEIC/HEIF without pillow-heif: "
                  "pip install 'pillow-heif<1.0'")
            return None
        _HEIF_READY = True
    try:
        with Image.open(path) as im:
            # Phones record which way up they were held rather than rotating the
            # pixels; honouring that here means --rotate only ever has to deal
            # with how the book sat, not with how the camera was held.
            return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(im).convert("RGB")),
                                cv2.COLOR_RGB2BGR)
    except (OSError, ValueError) as exc:
        print(f"  [!] unreadable image {os.path.basename(path)}: {exc}")
        return None


def collect_image_paths(spec: str) -> list[str]:
    """
    The photographs to read, in the order they were taken.

    Sorted by name, because cameras number files in the order the shutter fired
    and that is the order the pages were turned in. Any other order would have
    to be recovered from EXIF timestamps, which phones and cameras disagree
    about often enough not to trust ahead of the filename.
    """
    if os.path.isdir(spec):
        names = [os.path.join(spec, n) for n in os.listdir(spec)]
    else:
        names = glob.glob(spec)
    paths = sorted(p for p in names
                   if os.path.isfile(p) and p.lower().endswith(IMAGE_SUFFIXES))
    if not paths:
        sys.exit(f"No images found in {spec} "
                 f"(looking for {', '.join(IMAGE_SUFFIXES)})")
    return paths


def group_burst(paths: list[str], threshold: float | None = None) -> list[list[int]]:
    """
    Gather consecutive photographs of the same page into one group.

    Shooting a burst while turning pages is the least tedious way to photograph
    a book, and it produces the same thing a video does — many frames per page,
    most of them mid-turn. So it is treated the same way: consecutive shots that
    look alike are one page, and the run of them is scored for the best. One
    deliberate photograph per page falls out of this untouched, since every
    consecutive pair then differs and every group holds a single image.
    """
    thumbs = []
    for p in paths:
        img = read_image(p)
        thumbs.append(None if img is None else _thumb(img))
    diffs = [float(np.abs(b - a).mean()) if (a is not None and b is not None) else 255.0
             for a, b in zip(thumbs, thumbs[1:])]
    if not diffs:
        return [[0]] if paths else []
    if threshold is None:
        # The gap between "the same page again" and "a different page" is what
        # the threshold has to land in — but only if there IS a gap. Otsu always
        # returns a cut, even for values that form a single cluster, and one
        # deliberate photograph per page produces exactly that: consecutive
        # differences all large and all alike. Taking the cut regardless read
        # half of them as repeats and silently merged distinct pages. So the
        # same opt-out the blur and hold thresholds use applies here: if the two
        # sides do not really differ, group nothing and treat every photograph
        # as its own page.
        threshold = _cluster_cut(diffs)
        if not np.isfinite(threshold):
            threshold = 0.0
    groups, cur = [], [0]
    for i, d in enumerate(diffs, start=1):
        if d < threshold:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def _prepare_page_image(img: np.ndarray, min_area_ratio: float, rotate: int,
                        no_warp: bool, enhance: bool, deskew: bool) -> np.ndarray:
    """Put one image through the same treatment a captured page receives."""
    if rotate:
        img = rotate_image(img, rotate)
    if not no_warp:
        corners = detect_page(img, min_area_ratio)
        if corners is not None:
            img = warp_page(img, limit_quad(corners))
    if enhance:
        img = enhance_page(img)
    if deskew:
        img = deskew_page(img)
    return img


def load_patches(spec: str) -> dict[str, str]:
    """
    Replacement sources for named pages: {"r060p1": "photos/folio60.jpg", ...}

    Some pages cannot be rescued by choosing a better frame, because the video
    never holds the information — a page whose running head is eleven pixels
    tall does not become legible by being enlarged. Re-photographing that one
    page is the only real fix, and it is a minute's work for whoever has the
    book. Accepts a JSON file of the same shape, or inline "id=path" pairs.
    """
    if os.path.isfile(spec):
        try:
            with open(spec) as fh:
                loaded = json.load(fh)
        except (OSError, ValueError) as exc:
            sys.exit(f"--patch could not read {spec}: {exc}")
        if not isinstance(loaded, dict):
            sys.exit(f"--patch expects an object of page-id → file, in {spec}")
        patches = {str(k): str(v) for k, v in loaded.items()}
    else:
        patches = {}
        for tok in re.split(r"[,\s]+", spec.strip()):
            if not tok:
                continue
            if "=" not in tok:
                sys.exit(f"--patch wants id=file pairs or a JSON file, got: {tok}")
            k, val = tok.split("=", 1)
            patches[k.strip()] = val.strip()

    for k, val in patches.items():
        if not k.strip():
            sys.exit("--patch: a patch is named for an empty page")
        if not os.path.exists(val):
            sys.exit(f"--patch: no such file for {k}: {val}")
    return patches


def _patch_frames(path: str) -> list[np.ndarray]:
    """Every usable frame of a patch source — a still, or the frames of a clip."""
    ext = os.path.splitext(path)[1].lower()
    if ext in {".mov", ".mp4", ".m4v", ".avi", ".mkv"}:
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        return frames
    img = read_image(path)
    return [img] if img is not None else []


def apply_patches(page_paths: list[str], page_ids: list[str] | None,
                  patches: dict[str, str], pages_dir: str,
                  min_area_ratio: float, rotate: int, no_warp: bool,
                  enhance: bool, deskew: bool) -> int:
    """
    Swap in re-shot sources for the pages named, in place.

    A patch is held to the same standard as a capture: it goes through the same
    detect/warp/enhance/deskew path, so a patched page is not distinguishable
    from one the video gave up. A clip is treated as a miniature scan of a
    single page — the stillest frame wins, by the same measure used everywhere
    else, since a hand-held phone shot has the same wobble the book video does.
    """
    applied = 0
    for sel, src in patches.items():
        try:
            idx = resolve_selector(sel, page_ids)
        except ValueError as exc:
            sys.exit(f"--patch {sel}: {exc}")
        if not 0 <= idx < len(page_paths):
            sys.exit(f"--patch {sel}: outside this book (0-{len(page_paths) - 1})")
        frames = _patch_frames(src)
        if not frames:
            print(f"  [!] --patch {sel}: no readable image in {src}")
            continue
        if len(frames) > 1:
            scored = [(_frame_score(f)[0], i) for i, f in enumerate(frames)]
            best = max(scored)[1]
            print(f"    {sel}: {len(frames)} frames from {os.path.basename(src)}, "
                  f"keeping #{best}")
            frame = frames[best]
        else:
            frame = frames[0]
        img = _prepare_page_image(frame, min_area_ratio, rotate, no_warp, enhance, deskew)
        out = os.path.join(pages_dir, f"patch_{idx:04d}.jpg")
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(
            out, "JPEG", quality=92, dpi=(300, 300)
        )
        page_paths[idx] = out
        applied += 1
        label = f"#{idx}" + (f" ({page_ids[idx]})" if page_ids else "")
        print(f"  Patched page {label} ← {src}")
    return applied


def pages_from_images(
    image_paths: list[str], pages_dir: str, min_area_ratio: float, rotate: int,
    no_warp: bool, enhance: bool, deskew: bool, split_spreads: bool,
    burst_threshold: float | None = None,
) -> tuple[list[str], list[str]]:
    """
    Build the book from photographs rather than from a video.

    Nothing downstream changes: the same cropping, gutter split, enhancement and
    deskew run, and the review sheet and patching work the same way. What is
    skipped is the whole business of deciding when the book was held still,
    because a photograph is by definition a moment someone chose — which is why
    a set of stills sidesteps the failure this pipeline works hardest to avoid.
    """
    groups = group_burst(image_paths, burst_threshold)
    multi = sum(1 for g in groups if len(g) > 1)
    print(f"  {len(image_paths)} photographs → {len(groups)} pages"
          + (f" ({multi} taken as bursts, keeping the best of each)" if multi else ""))

    page_paths: list[str] = []
    page_ids: list[str] = []
    n_nopage = n_split = 0
    for group in groups:
        best_path, best_score = image_paths[group[0]], None
        if len(group) > 1:
            for i in group:
                img = read_image(image_paths[i])
                if img is None:
                    continue
                score = _frame_score(rotate_image(img, rotate) if rotate else img)[0]
                if best_score is None or score > best_score:
                    best_path, best_score = image_paths[i], score
        img = read_image(best_path)
        if img is None:
            print(f"  [!] unreadable image, skipped: {best_path}")
            continue
        if rotate:
            img = rotate_image(img, rotate)
        cropped = True
        if not no_warp:
            corners = detect_page(img, min_area_ratio)
            if corners is None:
                n_nopage += 1
                cropped = False
            else:
                img = warp_page(img, limit_quad(corners))

        stem = os.path.splitext(os.path.basename(best_path))[0]
        # Splitting a frame whose page was never found is guesswork: what is in
        # hand is the whole photograph, desk and all, and the "gutter" the
        # search settles on is wherever that scene happens to be darkest down
        # the middle. On the cover shots of one book it fell between the desk
        # and the cover, turning each into a page of bare table. Brightness
        # cannot tell those apart from a real spread either — measured here, two
        # halves of a genuine spread match to within 0.94-0.97 of each other and
        # one cover shot sat at 0.93 — but whether a page was found at all
        # separates them completely.
        parts = split_spread(img) if (split_spreads and cropped) else [img]
        n_split += len(parts) > 1
        for part_idx, part in enumerate(parts):
            if enhance:
                part = enhance_page(part)
            if deskew:
                part = deskew_page(part)
            out = os.path.join(pages_dir, f"page_{len(page_paths):04d}.jpg")
            Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB)).save(
                out, "JPEG", quality=92, dpi=(300, 300)
            )
            page_paths.append(out)
            page_ids.append(page_id(stem, part_idx))

    print(f"  {len(page_paths)} pages kept ({n_nopage} kept uncropped, "
          f"no page outline found)")
    if n_split:
        print(f"  {n_split} two-page spreads split at the gutter")
    return page_paths, page_ids


def load_pages_from_pdf(pdf_path: str, pages_dir: str) -> list[str]:
    """
    Take the page images straight back out of a PDF this script wrote earlier.

    Re-reading the video costs the better part of an hour before OCR can even
    start, which makes iterating on OCR settings painful for no reason: the pages
    were already chosen and cropped. Pulling the embedded JPEG out rather than
    re-rendering the page hands OCR the original image instead of a re-encode.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        sys.exit("--pages-from needs pypdfium2: pip install pypdfium2")

    doc = pdfium.PdfDocument(pdf_path)
    page_paths: list[str] = []
    for i, page in enumerate(doc):
        stem = os.path.join(pages_dir, f"page_{i:04d}")
        images = [o for o in page.get_objects() if isinstance(o, pdfium.PdfImage)]
        pw_pt, ph_pt = page.get_size()
        out: str | None = None
        extracted = False
        if len(images) == 1:
            # The shortcut is only honest when the one image IS the page — the
            # shape this script's own output has. A born-digital page holding a
            # single small illustration also has exactly one image object, and
            # taking the shortcut there discards every word of the page's
            # vector text and keeps the illustration as "the page": a field
            # manual lost forty-seven pages of procedure prose to it, silently,
            # while its picture-free pages came through perfectly.
            try:
                bm = images[0].get_bitmap()
                pw, ph = page.get_size()
                covers = (bm.width * bm.height) >= 0.5 * (pw / 72 * 150) * (ph / 72 * 150)
                # …and it must cover the page, not merely a lot of it. A
                # facsimile IS the page and measures 1.00; a photograph with
                # a caption beside it measures 0.80-0.89, and extracting it
                # as "the page" threw the caption away on every plate in a
                # book. Measured across four scans and one born-digital
                # title, nothing legitimate falls between.
                bounds = images[0].get_bounds()
                placed = (((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]))
                          / max(1e-6, pw * ph))
                covers = covers and placed >= 0.97
                # A page facsimile has the page's own shape. A born-digital
                # page can hold one LARGE figure — a chart wider than it is
                # tall — and taking it as "the page" discards every word of
                # vector text beside it; what surya then reads is a textless
                # chart, and what it returned, four separate times, was an
                # invented table. Coverage alone passed that chart; shape
                # does not.
                page_aspect = pw / max(1e-6, ph)
                img_aspect = bm.width / max(1e-6, bm.height)
                covers = covers and (
                    0.8 <= img_aspect / max(1e-6, page_aspect) <= 1.25)
            except Exception:
                covers = False
            if covers:
                try:
                    images[0].extract(stem)  # keeps the original encoding
                    out = next(iter(sorted(glob.glob(stem + ".*"))), None)
                    extracted = out is not None
                except Exception:
                    out = None
        if out is None:
            # A page this script did not write, or one built from several images,
            # is safer to render whole than to pick apart.
            # Lossless, always. This raster is what the recogniser reads and
            # what a figure falls back to when it cannot be taken from the
            # PDF directly, so any loss here is permanent and lands hardest
            # on photographic plates — the very pages a size-based choice
            # would have encoded as JPEG. On text and line art PNG is also
            # the smaller file (631KB against 1182KB on a manual page); on a
            # plate it is larger, and that costs scratch space in a
            # directory this run deletes when it ends.
            out = stem + ".png"
            page.render(scale=300 / 72).to_pil().convert("RGB").save(
                out, "PNG", dpi=(300, 300)
            )
        page_paths.append(out)
        # The page's own text layer rides along as a sidecar. It is never used
        # as content — this pipeline reads the ink — but it is a witness: a
        # recognition that shares almost no vocabulary with what the page says
        # about itself has invented something, and without the sidecar that
        # comparison cannot be made at read time.
        try:
            layer = page.get_textpage().get_text_bounded().strip()
        except Exception:
            layer = ""
        if layer:
            with open(out + ".layer.txt", "w", encoding="utf-8") as fh:
                fh.write(layer)
            # Born-digital, or somebody else's OCR? The distinction decides
            # whether this layer may correct our words, restore a page we
            # could not read, or only bear witness — and getting it wrong
            # imports a worse reading as if it were the truth. "The page was
            # rendered rather than extracted" is not the test: a library scan
            # stored as two MRC layers is rendered too, and forty-three pages
            # of one were treated as a publisher's typesetting, pulling in
            # sixty of Google's words and filling two blank endpapers with
            # OCR of the embossed seal. What settles it is what sits behind
            # the text: a scan has a picture of the whole page under it, and
            # a born-digital page does not. A photographic plate covers at
            # most about nine tenths of its page, leaving room for the
            # caption; a facsimile covers all of it.
            behind = 0.0
            for o in images:
                try:
                    b = o.get_bounds()
                    behind = max(behind, ((b[2] - b[0]) * (b[3] - b[1]))
                                 / max(1e-6, pw_pt * ph_pt))
                except Exception:
                    pass
            if not extracted and behind < 0.9:
                open(out + ".native", "w").close()
        # Where this page came from. A figure cut out of the page raster is a
        # crop of a crop — the raster already resampled the artwork, at 3.12x
        # on one manual's line drawings, and JPEG rang every hard edge it
        # made. With the source in hand a figure can be taken from the PDF
        # itself instead.
        try:
            with open(out + ".source.json", "w") as fh:
                json.dump({"pdf": os.path.abspath(pdf_path), "page": i,
                           "facsimile": bool(extracted)}, fh)
        except OSError:
            pass

    if not page_paths:
        sys.exit(f"No pages found in {pdf_path}")
    print(f"  {len(page_paths)} pages loaded from {pdf_path}")
    return page_paths


# A page's position in the book is not a durable name for it. Re-extracting with
# a different frame-selection rule keeps the same still-runs but can change how
# many parts a run splits into, and every index after that point shifts — which
# silently re-points a hand-curated drop list at the wrong pages. The run a page
# came from, and which half of that spread it was, survive all of it: selection
# decides *which frame* represents a run, never which runs exist.
POSITIONAL = re.compile(r"^\d+$")


def page_id(source: int | str, part: int) -> str:
    """
    A durable name for a page: where it came from, and which half of it.

    From video that is the still-run's ordinal; from a folder of photographs it
    is the file's own name, which is better still — a filename survives even a
    change of capture, where a run ordinal only survives a change of selection.
    """
    stem = f"r{source:03d}" if isinstance(source, int) else source
    return f"{stem}p{part}"


def resolve_selector(tok: str, page_ids: list[str] | None) -> int:
    """A page named either by position in the book, or by where it came from."""
    tok = tok.strip()
    if POSITIONAL.match(tok):
        return int(tok)
    if not page_ids:
        raise ValueError(
            f"'{tok}' names a page by source, but no page map is available. "
            "Extract from video or images (either writes one), or pass --page-map."
        )
    try:
        return page_ids.index(tok)
    except ValueError:
        raise ValueError(f"no page {tok} in this book") from None


def load_page_map(path: str) -> list[str] | None:
    """Page ids written beside a raw PDF by an earlier extraction, if present."""
    for cand in (path + ".pages.json", os.path.splitext(path)[0] + ".pages.json"):
        if os.path.exists(cand):
            try:
                with open(cand) as fh:
                    ids = json.load(fh).get("page_ids")
                if isinstance(ids, list) and ids:
                    print(f"  Page map: {cand} ({len(ids)} ids)")
                    return [str(i) for i in ids]
            except (OSError, ValueError):
                pass
    return None



# A generative recogniser has a failure mode a classical one cannot have: given
# a page it cannot read, it invents. Measured here on 47 reads, the fabrications
# announce themselves two ways — they loop (every six-word window repeating), or
# they appear on a page that has almost no ink to have produced them. Neither
# alone is enough. One fabrication repeated three reads verbatim, so re-reading
# cannot catch it; every genuinely sparse page (a part title, a chapter's last
# two lines) reads back identically, so ink cannot condemn one on its own.
# Looping is the one signal that survived testing. Two others were tried and
# withdrawn, each after condemning real pages: "text on a near-blank page"
# threw away two books' title pages, and "characters per unit of ink" flagged
# 276 of Shadow Libraries' 321 pages — ink varies more across typefaces and
# scan styles than fabrication varies from truth, so no fixed threshold on it
# transfers between books. Every damaging fabrication observed loops; the
# short ones that do not are indistinguishable from a real dedication page
# and are accepted as a known blind spot rather than guessed at.
DEGENERATE_REP6 = 0.4        # legitimate pages measured at most 0.14


def _repeated_six_gram_share(text: str) -> float:
    """How much of the text is six-word runs that occur more than once."""
    w = re.findall(r"[a-z0-9]{2,}", text.lower())
    if len(w) < 40:
        return 0.0
    six = [" ".join(w[i:i + 6]) for i in range(len(w) - 5)]
    counts: dict[str, int] = {}
    for g in six:
        counts[g] = counts.get(g, 0) + 1
    return sum(v for v in counts.values() if v > 1) / max(1, len(six))


def _page_ink(image) -> float:
    """Fraction of the page clearly darker than its paper, border excluded.

    The border matters: a photographed page carries the desk and the book's
    own shadow, and counting those put a blank leaf at 0.11 when the paper
    itself was at 0.006 — nearly the reading of a printed page.
    """
    import numpy as np
    a = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = a.shape
    a = a[int(h * .06):int(h * .94), int(w * .06):int(w * .94)]
    if a.size == 0:
        return 1.0
    return float((a < np.percentile(a, 90) - 40).mean())


def _looks_fabricated(items, image) -> str | None:
    """Why this reading is suspect, or None if it looks like a real one."""
    text = " ".join(_strip_tags(it.html) for it in items
                    if it.html and not it.is_furniture).strip()
    if not text:
        return None                      # an empty read is the other guard's
    if _repeated_six_gram_share(text) >= DEGENERATE_REP6:
        return "text loops on itself"
    return None


# Below this share of our words appearing in the page's own text layer, the
# reading is talking about something the page is not. Fabricated pages measured
# 0.00-0.04 against their layers; genuine readings of the same pages, 0.5 and
# up even against badly garbled scanner layers, because prose shares its own
# vocabulary however badly either side mangles the edges. The word floor keeps
# the rule off pages too short to have a vocabulary worth comparing.
LAYER_MIN_OVERLAP = 0.2
LAYER_MIN_WORDS = 40


def _contradicts_layer(items, page_path: str) -> str | None:
    """Why this reading disagrees with the page's own text layer, or None."""
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            layer = fh.read()
    except OSError:
        return None                        # no layer, no witness — not guilt
    words = re.findall(r"[a-z]{3,}", " ".join(
        _strip_tags(it.html) for it in items if it.html).lower())
    # The floor counts words READ, not distinct ones: a fabricated table says
    # "Project" eighty times in a thousand words — plenty of text, almost no
    # vocabulary — and a floor on distinct words would wave exactly the
    # fabrications through.
    if len(words) < LAYER_MIN_WORDS:
        return None
    ours = set(words)
    theirs = set(re.findall(r"[a-z]{3,}", layer.lower()))
    # A witness must know enough to testify. One TiHKAL page carries a real
    # 261-word aside about the size of a trillion over a layer that says only
    # "how many zeros" — the scanner's own OCR failed there — and judging
    # vocabulary against three words condemns the truth. A sparse layer means
    # no verdict here; the classical witness takes such pages instead.
    if len(theirs) < LAYER_MIN_WORDS:
        return None
    if len(ours & theirs) / len(ours) < LAYER_MIN_OVERLAP:
        return "reads nothing like the page's own text layer"
    return None


def _native_layer_body(page_path: str) -> "list[PageItem] | None":
    """
    The page rebuilt from its own born-digital text, or None where there is
    no such authority (a scan, a photograph, an empty layer).

    The last resort when every reading of a native page fails the guards:
    the publisher's text IS the page, so restoring from it is not a guess —
    unlike a scan, where a failed page must stay honestly unread. Leading
    watermark and folio lines are dropped by shape (a URL, a bare number):
    they are the page's furniture, not its words.
    """
    if not os.path.exists(page_path + ".native"):
        return None
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            layer = fh.read()
    except OSError:
        return None
    lines = [ln.strip() for ln in layer.splitlines() if ln.strip()]
    while lines and (re.fullmatch(r"[0-9ivxlc]+", lines[0].lower())
                     or "http" in lines[0].lower()):
        lines.pop(0)
    text = " ".join(lines)
    if len(text) < 40:
        return None
    return [PageItem(html=f"<p>{html.escape(text)}</p>")]


def _layer_can_testify(page_path: str) -> bool:
    """Whether the page has a text layer rich enough to judge a reading."""
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            return len(re.findall(r"[a-z]{3,}", fh.read().lower())) \
                >= LAYER_MIN_WORDS
    except OSError:
        return False


def witness_available() -> bool:
    """Whether the classical reader this pipeline testifies with is installed."""
    import shutil
    return shutil.which("tesseract") is not None


# Set when the operator has said, in as many words, to run without a witness.
# Nothing consults it to decide anything; it exists so the count of checks
# that went unwitnessed can be reported at the end of the run instead of the
# checks quietly not happening.
WITNESS_WAIVED = False
UNWITNESSED = {"fabrication": 0, "blank": 0}


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


def _reading_agrees(a_items, b_items) -> float:
    """Word-set agreement between two readings of one page."""
    def ws(items):
        return set(re.findall(r"[a-z]{3,}", " ".join(
            _strip_tags(it.html) for it in items if it.html).lower()))
    wa, wb = ws(a_items), ws(b_items)
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(1, len(wa | wb))


def _read_pages_resiliently(paths: list[str], images: list,
                            backend: "OCRBackend", cache: "OCRCache | None"
                            ) -> "dict[str, list[PageItem]]":
    """
    One OCR round with a guard: an empty read of an inked page is a failed
    call, not a blank page.

    The inference server behind recognition fails transiently — measured once
    at 1,536 errors in a run, which blanked 191 of 197 pages — and an empty
    result written to the cache poisons every later build. Each empty-but-inked
    page gets one solo retry, and nothing empty is cached unless the page
    itself is genuinely blank.
    """
    def has_text(items):
        return any(_strip_tags(it.html).strip() for it in items
                   if it.html and not it.is_furniture)

    out: dict[str, list[PageItem]] = {}
    for path, img, items in zip(paths, images, backend.run_items(images)):
        if not has_text(items) and figure_has_content(img):
            retry = backend.run_items([img])[0]
            if has_text(retry):
                items = retry
            else:
                # Twice empty on an inked page. On a scan that stays empty —
                # honest, and rereadable later since nothing is cached. On a
                # born-digital page the book's own text is at hand, and a
                # blank leaf distinguishes itself: its layer holds only the
                # watermark and folio, which the restorer strips, leaving
                # nothing to restore. A page that had real words gets them.
                restored = _native_layer_body(path)
                if restored:
                    items = restored
                    print(f"    [!] {os.path.basename(path)}: read empty "
                          f"twice — restored from the page's own "
                          f"born-digital text")

        # …and the opposite failure: text where the page has none to give.
        # Invented prose attributed to the author is worse than a page left
        # unread, so this one refuses rather than guesses. A second reading
        # decides: a real page reads back the same, a made-up one usually
        # does not — and where it does, it is still looping, which shows.
        why = _looks_fabricated(items, img) or _contradicts_layer(items, path)
        if why is None and not _layer_can_testify(path):
            # No layer to bear witness — a photographed source. A classical
            # OCR's silence on a page the model read paragraphs from is
            # grounds for suspicion (it cannot invent, so on fabrication-
            # prone pages it reads nothing), but never for conviction: it
            # also goes silent on faint real pages. Suspicion means one
            # re-read; the verdict belongs to stability, because every real
            # page measured reads back the same and inventions mostly don't.
            n_ours = len(re.findall(r"[a-z]{3,}", " ".join(
                _strip_tags(it.html) for it in items
                if it.html and not it.is_furniture).lower()))
            if n_ours >= LAYER_MIN_WORDS:
                n_tess = _classical_word_count(path)
                if n_tess is None:
                    # The only witness this page had, and it is not here. The
                    # page passes unexamined — which is the honest outcome,
                    # but it must be counted and said, not simply skipped.
                    UNWITNESSED["fabrication"] += 1
                elif n_tess < 5:
                    second = backend.run_items([img])[0]
                    if _reading_agrees(items, second) < 0.5:
                        why = ("read paragraphs where a classical OCR reads "
                               "nothing, and told a different story twice")
                        restored = _native_layer_body(path)
                        items = restored or []
                        print(f"    [!] {os.path.basename(path)}: {why} — "
                              + ("restored from the page's own born-digital "
                                 "text" if restored else
                                 "left unread rather than filled with "
                                 "invention"))
                        if not restored:
                            _read_pages_resiliently.refused.append(path)
        elif why is not None:
            second = backend.run_items([img])[0]
            why2 = _looks_fabricated(second, img) \
                or _contradicts_layer(second, path)
            if why2 is None:
                # A clean reading is the answer, however little it resembles
                # the invented one it replaces — disagreement with a
                # fabrication is expected, not evidence against the rescue.
                items, why = second, None
            else:
                why = (why if why == why2 else
                       f"{why}, and again on a second look ({why2})")
                items = _native_layer_body(path) or []
            if why is not None:
                restored = bool(items)
                print(f"    [!] {os.path.basename(path)}: {why} — "
                      + ("restored from the page's own born-digital text"
                         if restored else
                         "left unread rather than filled with invention"))
                if not restored:
                    _read_pages_resiliently.refused.append(path)

        out[path] = items
        if cache is not None:
            if has_text(items) or not figure_has_content(img):
                cache.put(path, items)
    return out


_read_pages_resiliently.refused = []


def reconcile_native_text(bodies: list[str], page_paths: list[str]) -> int:
    """
    Where a page's text is born-digital, defer to it word by word.

    On a scan, the embedded layer is another OCR's guess and ours is usually
    better. On a born-digital page the layer is not a guess — it is the
    publisher's own typesetting, so every disagreement is by definition our
    error, including the flattering ones: a recogniser that reads
    "competitiveness" where the book printed "competiveness" has silently
    edited the author. Surya still supplies everything structural — layout,
    reading order, figures, furniture — but inside tightly aligned prose
    runs, the native spelling wins.

    Only the unambiguous case substitutes: a run of at most three words that
    differs, bounded by words both sides agree on, with the same number of
    words on each side. Everything else — tables surya restructured, split
    hyphens the layer kept, regions that align loosely — stays ours.
    """
    import difflib

    changed = 0
    for pi, body in enumerate(bodies):
        if pi >= len(page_paths) or not body:
            continue
        base = page_paths[pi]
        if not os.path.exists(base + ".native"):
            continue
        try:
            with open(base + ".layer.txt", encoding="utf-8") as fh:
                layer_words = fh.read().split()
        except OSError:
            continue
        if len(layer_words) < 10:
            continue
        # split the html into tags and text nodes; tokens live in text nodes
        parts = re.split(r"(<[^>]+>)", body)
        toks: list[tuple[int, int, int, str]] = []   # (part, start, end, word)
        for k, part in enumerate(parts):
            if part.startswith("<"):
                continue
            for m in re.finditer(r"\S+", part):
                toks.append((k, m.start(), m.end(), html.unescape(m.group(0))))
        ours = [t[3] for t in toks]
        sm = difflib.SequenceMatcher(a=ours, b=layer_words, autojunk=False)
        subs: list[tuple[int, str]] = []
        for tag, a0, a1, b0, b1 in sm.get_opcodes():
            if tag != "replace" or (a1 - a0) != (b1 - b0) or (a1 - a0) > 3:
                continue
            for off in range(a1 - a0):
                cand = layer_words[b0 + off]
                # Some layers carry control characters at line-break points
                # ("Cana\x02da"); deferring to those would corrupt a word we
                # read correctly. Authority does not extend to garbage.
                if any(ord(c) < 32 for c in cand):
                    continue
                if ours[a0 + off] != cand:
                    subs.append((a0 + off, cand))
        if not subs:
            continue
        # apply from the back so recorded offsets stay valid
        for ti, new_word in sorted(subs, reverse=True):
            k, a, z, _ = toks[ti]
            parts[k] = parts[k][:a] + html.escape(new_word) + parts[k][z:]
            changed += 1
        bodies[pi] = "".join(parts)
    return changed


def _collect_page_items(page_paths: list[str], backend: "OCRBackend",
                        ocr_cache: str | None) -> list[list[PageItem]]:
    """Every page's items, through the same cache the EPUB build fills."""
    cache = OCRCache(ocr_cache, backend.name, getattr(backend, "langs", ["en"])) \
        if ocr_cache else None
    out: list[list[PageItem]] = []
    for start in range(0, len(page_paths), backend.batch_size):
        chunk = page_paths[start:start + backend.batch_size]
        remembered = [cache.get(p) if cache else None for p in chunk]
        unseen = [p for p, hit in zip(chunk, remembered) if hit is None]
        fresh: dict[str, list[PageItem]] = {}
        if unseen:
            opened = [Image.open(p).convert("RGB") for p in unseen]
            fresh = _read_pages_resiliently(unseen, opened, backend, cache)
            for im in opened:
                im.close()
        out.extend(hit if hit is not None else fresh[p]
                   for p, hit in zip(chunk, remembered))
    if cache is not None:
        cache.save()
    return out


# Invisible text may be set as small as it likes — nobody reads it, and search
# does not care about size. The floor exists only so a font size stays a
# positive number the writer will accept.
_LAYER_MIN_FONT = 0.4


def _force_text_into(page, rect, text: str) -> None:
    """
    Write text inside rect whatever it takes, as a last resort.

    Sized from the box's area rather than tried and retried, so the fit is
    arithmetic instead of hopeful: every character goes in. The layout is
    nothing like the printed page's, which costs a highlight that covers the
    block loosely — the alternative on this path is losing the words.
    """
    import math

    fs = math.sqrt(max(1.0, abs(rect.get_area())) / (0.9 * max(1, len(text))))
    fs = max(0.02, min(fs, max(0.02, rect.height)))
    per_line = max(1, int(rect.width / (fs * 0.62)))
    y = rect.y0 + fs
    for i in range(0, len(text), per_line):
        page.insert_text((rect.x0, min(y, rect.y1 + fs)), text[i:i + per_line],
                         fontsize=fs, fontname="helv", render_mode=3)
        y += fs * 1.2


def _tesseract_word_boxes(page_path: str) -> "list[dict] | None":
    """
    Word-level geometry from a classical OCR, or None if unavailable.

    Surya reports geometry per layout block only, so a paragraph is one
    rectangle and selection drifts inside it. Tesseract reads words WITH
    their boxes; its words are not trusted as text — surya's reading stays
    the truth — but where both engines read the same word inside the same
    block, tesseract's box places it. Measured on six pages: words sitting
    on their own ink went from 70.6% (block placement) to 92.9%.
    """
    import shutil
    import subprocess
    if not shutil.which("tesseract"):
        return None
    try:
        r = subprocess.run(["tesseract", page_path, "stdout", "tsv"],
                           capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    out = []
    for line in r.stdout.splitlines()[1:]:
        c = line.split("\t")
        if len(c) == 12 and c[0] == "5" and c[11].strip():
            try:
                x, y, w, h, conf = map(float, (c[6], c[7], c[8], c[9], c[10]))
            except ValueError:
                continue
            if conf >= 30:
                t = re.sub(r"^[^\w]+|[^\w]+$", "", c[11].lower())
                if t:
                    out.append({"t": t, "x": x, "y": y, "w": w, "h": h,
                                "used": False})
    return out


def build_searchable_pdf(page_paths: list[str],
                         items_per_page: list[list[PageItem]],
                         out_path: str) -> tuple[int, int, int]:
    """
    The cleaned page images with the OCR's text laid invisibly over each
    block, so the PDF searches, selects and copies like a born-digital file
    while showing exactly the pixels the camera kept.

    Placement is per layout block — the finest geometry surya records; it
    reports no line or character boxes, so a highlight covers the block that
    holds the hit rather than the exact words.

    No block's text is ever dropped. Refusing to place something is right
    where a wrong answer would assert a falsehood — a note link, a folio —
    but this layer has no reader: text nobody sees cannot mislead anyone.
    The costs are lopsided the other way. A block laid out imperfectly puts
    a highlight rectangle slightly wrong; a block left out makes the words
    unfindable, and silently, since nothing in the file says they are gone.
    So a block that will not fit at a natural size is shrunk, and one that
    will not fit at any size is written in a forced layout of its own.
    Returns (blocks placed, blocks forced).
    """
    import math
    import unicodedata
    import pymupdf

    doc = pymupdf.open()
    placed = forced = word_placed = 0
    for path, items in zip(page_paths, items_per_page):
        with Image.open(path) as im:
            w_px, h_px = im.size
            dpi = (im.info.get("dpi") or (300, 300))[0] or 300
        scale_pt = 72.0 / dpi
        w_pt, h_pt = w_px * scale_pt, h_px * scale_pt
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(page.rect, filename=path)
        tess = _tesseract_word_boxes(path)
        for it in items:
            if not it.html or it.box is None:
                continue
            text = unicodedata.normalize(
                "NFKC", html.unescape(_strip_tags(it.html))).strip()
            if not text:
                continue
            if tess:
                # Words both engines read take tesseract's box. Words only
                # surya read are interpolated from their matched neighbours
                # — between the end of the previous placed word and the start
                # of the next is almost always where the word actually sits —
                # and every word is inserted in READING order, because the
                # content stream's order is what search and copy follow: a
                # layer whose words are shuffled breaks phrase search across
                # every matched/unmatched seam.
                bx0, by0 = it.box[0] * w_px, it.box[1] * h_px
                bx1, by1 = it.box[2] * w_px, it.box[3] * h_px
                words = text.split()
                wrects: list = [None] * len(words)
                for wi, word in enumerate(words):
                    nw = re.sub(r"^[^\w]+|[^\w]+$", "", word.lower())
                    if len(nw) < 2:
                        continue
                    for cand in tess:
                        if cand["used"] or cand["t"] != nw:
                            continue
                        cx = cand["x"] + cand["w"] / 2
                        cy = cand["y"] + cand["h"] / 2
                        if bx0 - 20 <= cx <= bx1 + 20 \
                                and by0 - 20 <= cy <= by1 + 20:
                            cand["used"] = True
                            wrects[wi] = pymupdf.Rect(
                                cand["x"] * scale_pt, cand["y"] * scale_pt,
                                (cand["x"] + cand["w"]) * scale_pt,
                                (cand["y"] + cand["h"]) * scale_pt)
                            break
                anchors = [wi for wi, r in enumerate(wrects) if r is not None]
                if anchors:
                    heights = [wrects[wi].height for wi in anchors]
                    line_h = sum(heights) / len(heights)
                    cw = sum(wrects[wi].width / max(1, len(words[wi]))
                             for wi in anchors) / len(anchors)
                    blk_l, blk_r = it.box[0] * w_pt, it.box[2] * w_pt
                    est = lambda w: cw * (len(w) + 0.5)
                    for wi, word in enumerate(words):
                        if wrects[wi] is not None:
                            continue
                        prev = next((j for j in reversed(anchors) if j < wi), None)
                        nxt = next((j for j in anchors if j > wi), None)
                        if prev is not None and nxt is not None:
                            a, b = wrects[prev], wrects[nxt]
                            same_line = abs((a.y0 + a.y1) - (b.y0 + b.y1)) / 2 \
                                < line_h * 0.6
                            if same_line:
                                # share the gap between the two anchors
                                run = [k for k in range(prev + 1, nxt)
                                       if wrects[k] is None]
                                slot = (b.x0 - a.x1) / max(1, len(run))
                                off = run.index(wi)
                                x = a.x1 + slot * off
                                wrects[wi] = pymupdf.Rect(
                                    x, a.y0, x + max(slot, est(word)), a.y1)
                            else:
                                # a line break sits in the run: fill toward
                                # the block edge, then before the next anchor
                                x = a.x1 + cw * 0.4
                                if x + est(word) > blk_r:
                                    x = max(blk_l, b.x0 - est(word))
                                    a = b
                                wrects[wi] = pymupdf.Rect(
                                    x, a.y0, x + est(word), a.y1)
                        elif prev is not None:
                            a = wrects[prev]
                            x = a.x1 + cw * 0.4
                            if x + est(word) > blk_r:
                                x, a = blk_l, pymupdf.Rect(
                                    blk_l, a.y0 + line_h * 1.1,
                                    blk_r, a.y1 + line_h * 1.1)
                            wrects[wi] = pymupdf.Rect(x, a.y0,
                                                      x + est(word), a.y1)
                        else:
                            b = wrects[nxt]
                            x = max(blk_l, b.x0 - est(word) - cw * 0.4)
                            wrects[wi] = pymupdf.Rect(x, b.y0,
                                                      x + est(word), b.y1)
                        # synthesized boxes stay honest about their origin:
                        # inside the block, near the anchors, never beyond
                        wrects[wi] = wrects[wi] & pymupdf.Rect(
                            blk_l, it.box[1] * h_pt, blk_r, it.box[3] * h_pt) \
                            or wrects[wi]
                    for wi, word in enumerate(words):
                        r = wrects[wi]
                        unit = pymupdf.get_text_length(word, fontname="helv",
                                                       fontsize=1)
                        fs = max(1.0, min(r.height * 1.05,
                                          r.width / max(0.1, unit)))
                        try:
                            page.insert_text((r.x0, r.y1 - 0.15 * fs), word,
                                             fontsize=fs, fontname="helv",
                                             render_mode=3)
                        except Exception:
                            safe = word.encode("latin-1", "replace") \
                                .decode("latin-1")
                            page.insert_text((r.x0, r.y1 - 0.15 * fs), safe,
                                             fontsize=fs, fontname="helv",
                                             render_mode=3)
                    word_placed += len(anchors)
                    placed += 1
                    continue
            x0, y0, x1, y1 = it.box
            rect = pymupdf.Rect(x0 * w_pt, y0 * h_pt, x1 * w_pt, y1 * h_pt)
            rect.normalize()
            if rect.width < 1 or rect.height < 1:
                # Degenerate geometry — a sliver of a polygon. Give it a
                # minimal box where the block claimed to be, so the words
                # still enter the file.
                rect = pymupdf.Rect(
                    min(rect.x0, w_pt - 24), min(rect.y0, h_pt - 8),
                    min(w_pt, max(rect.x1, rect.x0 + 24)),
                    min(h_pt, max(rect.y1, rect.y0 + 8)))
            # Start at the size that would just fill the box rather than
            # walking down from an arbitrary maximum: a dense block lands in
            # a step or two instead of eight, and a sparse one keeps a size
            # close to the ink it sits under.
            fs = min(rect.height * 0.9,
                     math.sqrt(max(1.0, abs(rect.get_area()))
                               / (0.62 * max(1, len(text)))))
            fs = max(_LAYER_MIN_FONT, min(fs, 72.0))
            ok = cleaned = False
            while True:
                try:
                    leftover = page.insert_textbox(
                        rect, text, fontsize=fs, fontname="helv",
                        render_mode=3)
                except Exception:
                    if cleaned:
                        break
                    # a glyph the built-in font lacks: replace once, retry
                    text = text.encode("latin-1", "replace").decode("latin-1")
                    cleaned = True
                    continue
                if leftover >= 0:
                    ok = True
                    break
                if fs <= _LAYER_MIN_FONT:
                    break
                fs = max(_LAYER_MIN_FONT, fs * 0.75)
            if ok:
                placed += 1
            else:
                _force_text_into(page, rect, text)
                forced += 1
    doc.set_metadata({"producer": "video_to_book"})
    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    return placed, forced, word_placed


def page_texts(page_paths: list[str], backend: "OCRBackend",
               ocr_cache: str | None = None) -> list[str]:
    """
    The words on each page, through the same cache the EPUB build uses.

    Sharing the cache is the whole point: reading the book is by far the slowest
    thing here, and a caller that needs the text before deciding what to keep
    would otherwise pay for it twice.
    """
    cache = OCRCache(ocr_cache, backend.name, getattr(backend, "langs", ["en"])) \
        if ocr_cache else None
    out: list[str] = []
    batch = max(1, getattr(backend, "batch_size", 4))
    for start in range(0, len(page_paths), batch):
        chunk = page_paths[start:start + batch]
        pending, cached = [], {}
        for p in chunk:
            hit = cache.get(p) if cache else None
            if hit is None:
                pending.append(p)
            else:
                cached[p] = " ".join(_strip_tags(it.html) for it in hit
                                     if it.html and not it.is_furniture)
        if pending:
            opened = [Image.open(p).convert("RGB") for p in pending]
            got = _read_pages_resiliently(pending, opened, backend, cache)
            for im in opened:
                im.close()
            for p in pending:
                cached[p] = " ".join(_strip_tags(it.html) for it in got[p]
                                     if it.html and not it.is_furniture)
        out.extend(cached.get(p, "") for p in chunk)
        print(f"    read {min(start + batch, len(page_paths))}/{len(page_paths)} pages…")
    if cache:
        cache.save()
    return out


def _finish(
    page_paths: list[str],
    output_pdf: str | None,
    output_epub: str | None,
    backend: "OCRBackend | None",
    title: str,
    embed_images: bool,
    img2pdf,
    dedupe: bool = True,
    review_sheet: str | None = None,
    drop_pages: list[str] | None = None,
    ocr_cache: str | None = None,
    swap_pairs: list[tuple[str, str]] | None = None,
    page_ids: list[str] | None = None,
    drop_turns: bool = False,
    check_folios: bool = False,
    drop_blank: bool = True,
    cover_spec: str | None = "auto",
    link_notes_flag: bool = False,
    link_citations_flag: bool = False,
    link_index_flag: bool = False,
    typography_flag: bool = False,
    pdf_text_layer: bool = False,
    decisions_path: str | None = None,
) -> None:
    """Write whichever outputs were asked for, from pages already on disk."""
    # Both numbered against the pages as they went in, which is what the review
    # sheet and any manual page-by-page check show, so the numbers stay
    # meaningful regardless of what else has been dropped or reordered. Tracking
    # the original index alongside each path, rather than trusting position,
    # is what lets drop and swap compose in either order without one silently
    # renumbering what the other meant.
    numbered = list(enumerate(page_paths))

    if drop_turns:
        # Turn fragments are told apart by what they SAY — that they repeat a
        # neighbouring page — so this needs the pages read first. Reading them
        # here rather than leaving it to the EPUB is what lets the PDF be
        # filtered too; the cache means the EPUB's own pass costs nothing extra.
        if backend is None:
            sys.exit("--drop-turns needs the pages read first: add --epub "
                     "(and --ocr-cache to keep it cheap on a re-run)")
        print("\n[+] Reading pages to find page-turn fragments…")
        turn_texts = page_texts(page_paths, backend, ocr_cache)
        turns = find_page_turns(page_paths, turn_texts)
        if turns:
            names = ", ".join(
                (page_ids[i] if page_ids and i < len(page_ids) else str(i))
                for i, _ in turns)
            print(f"\n[+] Discarding {len(turns)} page-turn fragment(s): {names}")
            for i, why in turns:
                print(f"      #{i}: {why}")
            drop_ix = {i for i, _ in turns}
            numbered = [(i, p) for i, p in numbered if i not in drop_ix]
        else:
            print("\n[+] No page-turn fragments found")

    def resolve(tok: str) -> int:
        idx = resolve_selector(tok, page_ids)
        if not 0 <= idx < len(page_paths):
            raise ValueError(f"page {tok} is outside this book (0-{len(page_paths) - 1})")
        return idx

    def name(idx: int) -> str:
        """How a page is best referred to back to the reader."""
        return f"#{idx}" + (f" ({page_ids[idx]})" if page_ids and idx < len(page_ids) else "")

    if drop_pages:
        try:
            drop = {resolve(t) for t in drop_pages}
        except ValueError as exc:
            sys.exit(f"--drop-pages: {exc}")
        before = len(numbered)
        numbered = [(i, p) for i, p in numbered if i not in drop]
        print(f"\n[+] Dropping {before - len(numbered)} pages named on the "
              f"command line ({len(numbered)} left)")

    if swap_pairs:
        position = {i: pos for pos, (i, _) in enumerate(numbered)}
        applied = 0
        for ta, tb in swap_pairs:
            try:
                a, b = resolve(ta), resolve(tb)
            except ValueError as exc:
                sys.exit(f"--swap: {exc}")
            if a in position and b in position:
                pa, pb = position[a], position[b]
                numbered[pa], numbered[pb] = numbered[pb], numbered[pa]
                position[a], position[b] = pb, pa
                applied += 1
            else:
                missing = a if a not in position else b
                print(f"  [!] --swap {ta}:{tb} skipped — page {name(missing)} was "
                      f"already dropped")
        print(f"  Swapped {applied} page pair(s) into reading order")

    page_paths = [p for _, p in numbered]
    # Ids follow their pages through the reordering, so the review sheet names a
    # page by the run it came from even after drops and swaps have moved it.
    kept_ids = ([page_ids[i] for i, _ in numbered]
                if page_ids and all(i < len(page_ids) for i, _ in numbered) else None)

    if output_pdf and not pdf_text_layer:
        print(f"\n[+] Saving PDF → {output_pdf}")
        with open(output_pdf, "wb") as fh:
            fh.write(img2pdf.convert(page_paths))
        print(f"  PDF saved: {output_pdf}  ({len(page_paths)} pages)")

    if output_epub and backend is not None:
        print(f"\n[+] Building EPUB → {output_epub}")
        build_epub(page_paths, output_epub, backend, title, embed_images, dedupe,
                   review_sheet, ocr_cache, page_ids=kept_ids,
                   check_folios=check_folios, drop_blank=drop_blank,
                   cover_spec=cover_spec, link_notes_flag=link_notes_flag,
                   link_citations_flag=link_citations_flag,
                   link_index_flag=link_index_flag,
                   typography_flag=typography_flag,
                   decisions_path=decisions_path)

    if output_pdf and pdf_text_layer:
        # After the EPUB, so a shared cache is already warm — and the reads
        # carry geometry only if made by a version that keeps it, so a stale
        # cache yields an honest count of unplaced blocks, not wrong ones.
        if backend is None:
            sys.exit("--pdf-text-layer needs an OCR backend")
        print(f"\n[+] Saving searchable PDF → {output_pdf}")
        items_pp = _collect_page_items(page_paths, backend, ocr_cache)
        placed, forced, wordp = build_searchable_pdf(page_paths, items_pp,
                                                     output_pdf)
        boxed = sum(1 for pg in items_pp for it in pg if it.box is not None)
        unpositioned = sum(1 for pg in items_pp for it in pg
                           if it.html and _strip_tags(it.html).strip()
                           and it.box is None)
        print(f"  PDF saved: {output_pdf}  ({len(page_paths)} pages, "
              f"{placed} text blocks placed"
              + (f", {wordp} words at word precision" if wordp else "")
              + (f", {forced} force-fitted" if forced else "")
              + (f", {unpositioned} with no geometry left out of the layer"
                 if unpositioned else "")
              + (" — cache predates geometry; delete it and re-OCR"
                 if not boxed else "") + ")")


def process_video(
    video_path: str | None,
    output_pdf: str | None,
    output_epub: str | None = None,
    ocr: str = "surya",
    langs: list[str] | None = None,
    title: str = "Scanned Book",
    embed_images: bool = False,
    stride: int = 1,
    blur_threshold: float | None = None,
    min_area_ratio: float = 0.15,
    motion_threshold: float | None = None,
    min_still_frames: int = 3,
    enhance: bool = True,
    deskew: bool = True,
    no_warp: bool = False,
    split_spreads: bool = True,
    rotate: int = 0,
    pages_from: str | None = None,
    dedupe: bool = True,
    review_sheet: str | None = None,
    drop_pages: list[str] | None = None,
    ocr_cache: str | None = None,
    swap_pairs: list[tuple[str, str]] | None = None,
    page_map: list[str] | None = None,
    patches: dict[str, str] | None = None,
    from_images: str | None = None,
    drop_turns: bool = False,
    check_folios: bool = False,
    drop_blank: bool = True,
    cover_spec: str | None = "auto",
    link_notes_flag: bool = False,
    link_citations_flag: bool = False,
    link_index_flag: bool = False,
    typography_flag: bool = False,
    pdf_text_layer: bool = False,
    decisions_path: str | None = None,
) -> None:
    try:
        import img2pdf
    except ImportError:
        sys.exit("Missing dependency: pip install img2pdf")

    # Fail before spending minutes on frame extraction if the backend is unusable.
    backend = load_backend(ocr, langs or ["en"]) \
        if (output_epub or pdf_text_layer) else None

    tmpdir = tempfile.mkdtemp(prefix="book_scan_")
    frames_dir = os.path.join(tmpdir, "frames")
    pages_dir = os.path.join(tmpdir, "pages")
    os.makedirs(frames_dir)
    os.makedirs(pages_dir)

    try:
        if pages_from:
            print(f"\n[1/2] Loading pages from {pages_from}…")
            page_paths = load_pages_from_pdf(pages_from, pages_dir)
            ids = page_map or load_page_map(pages_from)
            if ids and len(ids) != len(page_paths):
                print(f"  [!] page map lists {len(ids)} pages but the PDF holds "
                      f"{len(page_paths)} — ignoring it rather than mis-naming pages")
                ids = None
            if patches:
                print(f"\n[+] Applying {len(patches)} patch source(s)…")
                apply_patches(page_paths, ids, patches, pages_dir, min_area_ratio,
                              rotate, no_warp, enhance, deskew)
            _finish(page_paths, output_pdf, output_epub, backend, title,
                    embed_images, img2pdf, dedupe, review_sheet, drop_pages,
                    ocr_cache, swap_pairs, ids, drop_turns, check_folios, drop_blank,
                    cover_spec, link_notes_flag, link_citations_flag, link_index_flag,
                    typography_flag, pdf_text_layer,
                    decisions_path=decisions_path)
            return

        if from_images:
            image_paths = collect_image_paths(from_images)
            print(f"\n[1/2] Reading {len(image_paths)} photographs from {from_images}…")
            page_paths, page_ids = pages_from_images(
                image_paths, pages_dir, min_area_ratio, rotate, no_warp,
                enhance, deskew, split_spreads, motion_threshold,
            )
            if not page_paths:
                sys.exit("No pages could be built from those images.")
            if patches:
                print(f"\n[+] Applying {len(patches)} patch source(s)…")
                apply_patches(page_paths, page_ids, patches, pages_dir,
                              min_area_ratio, rotate, no_warp, enhance, deskew)
            if output_pdf and page_ids:
                map_path = os.path.splitext(output_pdf)[0] + ".pages.json"
                try:
                    with open(map_path, "w") as fh:
                        json.dump({"page_ids": page_ids, "pages": len(page_ids)}, fh)
                    print(f"  Page map → {map_path}")
                except OSError as exc:
                    print(f"  [!] could not write page map: {exc}")
            _finish(page_paths, output_pdf, output_epub, backend, title,
                    embed_images, img2pdf, dedupe, review_sheet, drop_pages,
                    ocr_cache, swap_pairs, page_ids, drop_turns, check_folios, drop_blank,
                    cover_spec, link_notes_flag, link_citations_flag, link_index_flag,
                    typography_flag, pdf_text_layer,
                    decisions_path=decisions_path)
            return

        print("\n[1/3] Scanning video for pages held still…")
        motions = scan_motion(video_path, stride)
        if not motions:
            sys.exit(f"No frames read from {video_path}")

        ordered = sorted(motions)
        if motion_threshold is None:
            motion_threshold = auto_motion_threshold(motions)
            origin = "auto"
        else:
            origin = "manual"
        still_pct = 100.0 * sum(m < motion_threshold for m in motions) / len(motions)
        print(f"  Scanned {len(motions)} frames. Inter-frame motion: "
              f"min {ordered[0]:.2f}, median {ordered[len(ordered) // 2]:.2f}, "
              f"max {ordered[-1]:.2f}")
        print(f"  Motion threshold {motion_threshold:.2f} ({origin}) "
              f"→ {still_pct:.0f}% of frames held still")

        selected, _, sharpness, held_frames = select_page_frames(
            video_path, frames_dir, motion_threshold, min_still_frames, stride,
            min_area_ratio,
        )
        print(f"  {len(selected)} candidate pages found")

        if len(selected) < 2 and len(motions) > 30:
            print("  WARNING: suspiciously few pages. If your book has more, the motion\n"
                  "           threshold is likely too high — try --motion-threshold 2")

        print("\n[2/3] Cropping and enhancing…")
        # Each page goes straight to disk; keeping them all decoded in memory
        # costs several GB on a full-length book.
        # A page turn is not one clean movement: the motion dips below threshold
        # partway through, so some candidates are fragments of a turn rather than
        # pages. Neither measure catches them alone — a page shot slightly out of
        # focus is blurrier than a crisp mid-turn fragment, and a page flipped
        # quickly is held no longer than a pause during a turn. Demanding both
        # means either measure can misfire without costing a real page.
        auto_blur = auto_blur_threshold(sharpness)
        auto_hold = auto_hold_threshold(held_frames)
        if sharpness:
            ss = sorted(sharpness)
            hs = sorted(held_frames)
            print(f"  Sharpness: min {ss[0]:.0f}, median {ss[len(ss) // 2]:.0f}, "
                  f"max {ss[-1]:.0f}")
            print(f"  Held for: min {hs[0]}, median {hs[len(hs) // 2]}, "
                  f"max {hs[-1]} frames")
            if blur_threshold is not None:
                print(f"  Discarding sharpness under {blur_threshold:.0f} (manual)")
            elif np.isfinite(auto_blur) and np.isfinite(auto_hold):
                print(f"  Discarding candidates both blurrier than {auto_blur:.0f} "
                      f"and held under {auto_hold:.0f} frames (auto)")
            else:
                print("  Keeping every candidate: no separable cluster of turn fragments")

        page_paths: list[str] = []
        page_ids: list[str] = []
        n_rejected = n_nopage = n_split = 0
        for run_ordinal, (path, score, held) in enumerate(
                zip(selected, sharpness, held_frames)):
            if blur_threshold is not None:
                reject = score < blur_threshold
            else:
                reject = score < auto_blur and held < auto_hold
            if reject:
                n_rejected += 1
                os.remove(path)
                continue
            img = cv2.imread(path)
            if img is None:
                continue
            if rotate:
                img = rotate_image(img, rotate)
            cropped = True
            if not no_warp:
                corners = detect_page(img, min_area_ratio)
                if corners is None:
                    # Keep the frame uncropped: an untrimmed page still OCRs,
                    # whereas dropping it loses that page from the book entirely.
                    n_nopage += 1
                    cropped = False
                else:
                    img = warp_page(img, limit_quad(corners))

        # Splitting a frame whose page was never found is guesswork: what is in
        # hand is the whole photograph, desk and all, and the "gutter" the
        # search settles on is wherever that scene happens to be darkest down
        # the middle. On the cover shots of one book it fell between the desk
        # and the cover, turning each into a page of bare table. Brightness
        # cannot tell those apart from a real spread either — measured here, two
        # halves of a genuine spread match to within 0.94-0.97 of each other and
        # one cover shot sat at 0.93 — but whether a page was found at all
        # separates them completely.
            parts = split_spread(img) if (split_spreads and cropped) else [img]
            n_split += len(parts) > 1
            for part_idx, part in enumerate(parts):
                # Enhance first: deskew reads the page's own text to find level,
                # and it reads it more surely once the contrast has been lifted.
                if enhance:
                    part = enhance_page(part)
                if deskew:
                    part = deskew_page(part)
                out = os.path.join(pages_dir, f"page_{len(page_paths):04d}.jpg")
                Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB)).save(
                    out, "JPEG", quality=92, dpi=(300, 300)
                )
                page_paths.append(out)
                page_ids.append(page_id(run_ordinal, part_idx))
            os.remove(path)  # raw frame is no longer needed

        print(f"  {len(page_paths)} pages kept (discarded {n_rejected} mid-turn "
              f"fragments; {n_nopage} kept uncropped, no page outline found)")
        if n_split:
            print(f"  {n_split} two-page spreads split at the gutter")

        if not page_paths:
            sys.exit(
                "No pages detected. Suggestions:\n"
                "  --motion-threshold higher  (camera shake is masking the still stretches)\n"
                "  --min-still-frames lower   (pages were not held still for long)\n"
                "  --no-warp                  (skip perspective detection)\n"
                "  --min-area lower           (allow smaller page detections)"
            )

        if patches:
            print(f"\n[+] Applying {len(patches)} patch source(s)…")
            apply_patches(page_paths, page_ids, patches, pages_dir, min_area_ratio,
                          rotate, no_warp, enhance, deskew)

        # Written beside the PDF so a later run — with different selection, or
        # only --pages-from — can still be told which page to drop or move by
        # the run it came from rather than by a position that may have shifted.
        if output_pdf and page_ids:
            map_path = os.path.splitext(output_pdf)[0] + ".pages.json"
            try:
                with open(map_path, "w") as fh:
                    json.dump({"page_ids": page_ids, "pages": len(page_ids)}, fh)
                print(f"  Page map → {map_path}")
            except OSError as exc:
                print(f"  [!] could not write page map: {exc}")

        _finish(page_paths, output_pdf, output_epub, backend, title,
                embed_images, img2pdf, dedupe, review_sheet, drop_pages,
                ocr_cache, swap_pairs, page_ids, drop_turns, check_folios, drop_blank,
                    cover_spec, link_notes_flag, link_citations_flag, link_index_flag,
                    typography_flag, pdf_text_layer,
                    decisions_path=decisions_path)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main(argv: "list[str] | None" = None) -> None:
    parser = argparse.ArgumentParser(
        prog="codicology convert",
        description="Convert a book — video, photographs, or PDF — into a "
                    "clean PDF and an OCR'd EPUB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("video", nargs="?",
                        help="Input video file (MP4, MOV, etc.). Omit with --pages-from")
    parser.add_argument("pdf", nargs="?",
                        help="Output PDF path. Optional with --pages-from, which can "
                             "produce only an EPUB")
    parser.add_argument("--pages-from", metavar="PDF",
                        help="Skip frame extraction and take the pages from a PDF this "
                             "script wrote earlier. Re-running OCR on a book already "
                             "extracted takes seconds instead of re-reading the video")
    parser.add_argument("--epub", metavar="FILE", help="Also produce an OCR'd EPUB")
    parser.add_argument("--pdf-text-layer", action="store_true",
                        help="Give the --pdf output a positioned, invisible text "
                             "layer from the OCR, so it searches and copies like a "
                             "born-digital file. Needs a cache written with geometry "
                             "(re-OCR with a fresh cache for books read before)")
    parser.add_argument("--ocr", choices=sorted(OCR_BACKENDS), default="surya",
                        help="OCR backend used for --epub (default: surya). "
                             "Surya is the only one this pipeline was built "
                             "for or has ever been run with; the others "
                             "return flat text with no layout, so a book "
                             "built on one arrives with no figures, headings, "
                             "contents or note links. Treat them as "
                             "unimplemented")
    parser.add_argument("--lang", default="en",
                        help="Comma-separated OCR language codes (default: en)")
    parser.add_argument("--title", default="Scanned Book", help="EPUB title metadata")
    parser.add_argument("--progress-json", action="store_true",
                        help="Emit machine-readable progress as one JSON object per "
                             "line on stderr, for a program driving this pipeline "
                             "through a pipe (the Calibre plugin does). The human "
                             "log on stdout is unchanged")
    parser.add_argument("--ocr-cache", metavar="FILE",
                        help="Remember what OCR read on each page here, and reuse it. "
                             "Re-running to change deduplication or --drop-pages then "
                             "costs seconds instead of reading the book again")
    parser.add_argument("--review-sheet", metavar="HTML",
                        help="Write a page of everything the duplicate pass was unsure "
                             "of, for checking by eye. Its output is a --drop-pages list")
    parser.add_argument("--drop-pages", metavar="LIST",
                        help="Leave these pages out, named as the review sheet shows them: "
                             "either by position (3,7,12) or by the run they came from "
                             "(r060p1), or a file with one per line. Prefer the run ids — "
                             "positions shift if the book is extracted again, run ids do not")
    parser.add_argument("--swap", metavar="PAIRS",
                        help="Trade the reading-order position of two pages, named the "
                             "same way as --drop-pages: a:b or a:b,c:d for more than one "
                             "pair. For when a page was filmed out of turn and its folio "
                             "reads before the one it should follow")
    parser.add_argument("--link-notes", action="store_true",
                        help="Bind superscript markers in the prose to the book's endnotes, "
                             "both directions, where the binding is certain. A marker is "
                             "linked only when its chapter scope is settled and exactly one "
                             "note answers its number there — an unlinked superscript is the "
                             "page as printed, a wrong link is misinformation in a suit")
    parser.add_argument("--link-citations", action="store_true",
                        help="Bind in-text citations to the bibliography, driven from the "
                             "bibliography itself: only names and years it actually holds "
                             "are ever sought, and a key naming two entries binds nothing. "
                             "Author-date in prose (Bernards 2019a) and Chicago short-form "
                             "in notes (the italic title). Every edited page must read "
                             "back word-for-word identical or it is reverted whole")
    parser.add_argument("--link-index", action="store_true",
                        help="Make the index's page numbers into links. The index "
                             "prints folios and the page list anchors them; this "
                             "joins the two. An index that declares its numbers "
                             "are paragraphs, not pages, is believed and skipped")
    parser.add_argument("--typography", action="store_true",
                        help="Restore what the recogniser flattened: straight quotes "
                             "become directional and double spaces collapse. Letters are "
                             "never touched, and neither are spaced ellipses — '. . .' is "
                             "how the book set them. Also reports characters that usually "
                             "mean the OCR stumbled")
    parser.add_argument("--apply-decisions", metavar="JSON", default=None,
                        help="review-sheet decisions to reapply at build time, "
                             "before the linkers run — the reviewer's "
                             "corrections survive rebuilds; sites that no "
                             "longer exist are reported stale, never guessed")
    parser.add_argument("--cover", metavar="PAGE_OR_FILE", default="auto",
                        help="The EPUB's cover: a page named like --drop-pages names them, "
                             "a path to an image file, or 'none'. Default 'auto' takes the "
                             "book's first page when it plausibly is a cover — mostly image, "
                             "hardly any text — and otherwise sets nothing, since a page of "
                             "prose posing as a cover is worse than a reader's default")
    parser.add_argument("--keep-blank-pages", action="store_true",
                        help="Keep pages that hold neither text nor a picture. They are "
                             "left out by default: a blank leaf matters in a facsimile PDF, "
                             "where pagination is the point, but in a reflowable book it is "
                             "an empty screen to swipe past. Nothing is lost either way — "
                             "an empty page is empty after every fragment has been assembled")
    parser.add_argument("--drop-turns", action="store_true",
                        help="Discard captures that caught a leaf in flight rather than a "
                             "page at rest. Deliberately cautious — a page has to carry far "
                             "less ink than this book's pages do AND be shaped unlike one, "
                             "because losing a real page costs more than keeping a bad "
                             "frame. Without this they are only flagged in the review sheet")
    parser.add_argument("--check-folios", action="store_true",
                        help="Read the printed page numbers and check they ascend, reporting "
                             "any out of order, repeated or missing. Each head is read twice "
                             "at different magnifications and only agreement is trusted, so "
                             "a capture too coarse to read reports nothing rather than "
                             "guessing. Needs --epub for the OCR backend")
    parser.add_argument("--from-images", metavar="DIR",
                        help="Build the book from photographs instead of a video: a folder "
                             "or glob, read in filename order. Consecutive shots of the "
                             "same page are grouped and the best kept, so interval or "
                             "burst shooting works as-is. Stills carry far more detail "
                             "than video frames, which is what makes small type readable")
    parser.add_argument("--patch", metavar="SPEC",
                        help="Replace named pages with re-shot sources: a JSON file of "
                             "{\"r060p1\": \"folio60.jpg\"} or inline r060p1=folio60.jpg "
                             "pairs. Takes a photo or a short clip (stillest frame wins) "
                             "and runs it through the same crop/enhance path as a captured "
                             "page. For pages the video simply never resolved — no amount "
                             "of re-processing recovers detail the camera did not record")
    parser.add_argument("--page-map", metavar="FILE",
                        help="Page ids from an earlier extraction, so pages can be named "
                             "by the run they came from (r060p1) instead of by position. "
                             "Extraction writes one beside the PDF automatically and "
                             "--pages-from picks it up; this is for pointing at another")
    parser.add_argument("--no-dedupe", action="store_true",
                        help="Keep every page in the EPUB. By default a page whose OCR'd "
                             "text repeats one just before it is dropped, which removes "
                             "spreads the camera caught twice")
    parser.add_argument("--embed-images", action="store_true",
                        help="Include the page scans alongside the text in the EPUB")
    parser.add_argument("--stride", type=int, default=1,
                        help="Examine every Nth frame. 1 (default) is recommended: skipping "
                             "frames can hide a fast page turn and merge two pages into one. "
                             "Raise only to speed up a very long video")
    parser.add_argument("--blur-threshold", type=float, default=None,
                        help="Discard any page whose Laplacian variance falls below this, "
                             "regardless of how long it was held. By default blur is only "
                             "held against frames that were also barely held still")
    parser.add_argument("--min-area", type=float, default=0.15,
                        help="Minimum page area as a fraction of the frame (default: 0.15)")
    parser.add_argument("--motion-threshold", type=float, default=None,
                        help="Inter-frame motion below which the page counts as held still. "
                             "Chosen automatically from the footage by default; pass a number "
                             "to override (the scan prints the observed range)")
    parser.add_argument("--min-still-frames", type=int, default=3,
                        help="Frames a page must be held still to count. Raise if you get "
                             "spurious pages from mid-turn pauses (default: 3)")
    parser.add_argument("--without-witness", action="store_true",
                        help="Run even though tesseract is not installed. Two "
                             "checks depend on it and both then pass without "
                             "examining anything: whether a page with no text "
                             "layer was invented, and whether a page about to "
                             "be deleted as blank was merely one this pipeline "
                             "failed to read. The run says at the end how many "
                             "pages went unwitnessed")
    parser.add_argument("--no-warp", action="store_true",
                        help="Skip perspective correction (useful when detection fails)")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip contrast/sharpness enhancement")
    parser.add_argument("--no-deskew", action="store_true",
                        help="Skip straightening each page against its own text lines")
    parser.add_argument("--no-split-spreads", action="store_true",
                        help="Keep two-page spreads as a single page. By default a landscape "
                             "crop is split at the gutter into two pages")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Rotate frames before processing, for footage that comes in "
                             "sideways (default: 0)")

    args = parser.parse_args(argv)
    if args.progress_json:
        progress.enable()
    try:
        _convert(args, parser)
    except SystemExit as exc:
        # sys.exit(message) is this pipeline's idiom for a refusal; hand the
        # message to whoever is driving before the status code ends us.
        if exc.code not in (None, 0):
            progress.emit(event="error", message=str(exc.code))
        raise
    except Exception as exc:
        progress.emit(event="error", message=f"{type(exc).__name__}: {exc}")
        raise


def _convert(args, parser) -> None:
    # The witness is checked here, before any work, because finding out after
    # forty minutes of reading that nothing was witnessed is finding out too
    # late. It is the tesseract BINARY, not the pytesseract package, which is
    # a different thing that only the tesseract OCR backend uses — so no pip
    # install satisfies this and the dependency list cannot enforce it.
    if not witness_available():
        if not args.without_witness:
            sys.exit(
                "codicology needs tesseract installed: it is the witness two "
                "checks depend on — whether a page with no text layer was "
                "invented, and whether a page about to be deleted as blank "
                "was merely one this pipeline failed to read. Without it both "
                "pass without examining anything.\n"
                "    macOS:  brew install tesseract\n"
                "    Debian: apt install tesseract-ocr\n"
                "Or pass --without-witness to build anyway, accepting that "
                "those pages go unchecked. The run will say how many did.")
        globals()["WITNESS_WAIVED"] = True
        print("[!] Running WITHOUT a witness. Pages with no text layer will "
              "pass the fabrication check unexamined, and pages deleted as "
              "blank will be deleted on this pipeline's word alone.")

    if args.pages_from or args.from_images:
        # Neither mode reads a video, so a lone positional is the output PDF.
        output_pdf = args.pdf or args.video
        if args.pages_from and not os.path.isfile(args.pages_from):
            sys.exit(f"PDF not found: {args.pages_from}")
        if args.from_images and not os.path.exists(args.from_images) \
                and not glob.glob(args.from_images):
            sys.exit(f"No such folder or pattern: {args.from_images}")
        if not output_pdf and not args.epub:
            sys.exit("give an output: a PDF path, --epub, or both")
    else:
        if not args.video or not args.pdf:
            parser.error("give a video and an output PDF, or use --pages-from")
        if not os.path.isfile(args.video):
            sys.exit(f"Video not found: {args.video}")
        output_pdf = args.pdf

    drop_pages: set[int] = set()
    # Selectors stay as written until the page list exists, because a page may
    # be named by run ("r060p1") rather than by position, and only the book being
    # assembled knows where that page ended up.
    def check(tok: str) -> str:
        # A page id may be a run ("r060p1") or a filename ("IMG_4471p0"), so
        # anything that is not blank is accepted here and checked for real
        # against the page map once the book exists.
        if not tok.strip():
            sys.exit("empty page selector")
        return tok.strip()

    if args.from_images and args.video and not args.pdf:
        # "video_to_book.py --from-images shots out.pdf" puts the output in the
        # slot the video would occupy; take it as the output rather than refusing.
        args.pdf, args.video = args.video, None

    if args.drop_pages:
        raw = args.drop_pages
        if os.path.isfile(raw):
            with open(raw) as fh:
                raw = fh.read()
        drop_pages = [check(tok) for tok in re.split(r"[,\s]+", raw.strip()) if tok]

    swap_pairs: list[tuple[str, str]] = []
    if args.swap:
        for tok in re.split(r"[,\s]+", args.swap.strip()):
            if not tok:
                continue
            if ":" not in tok:
                sys.exit(f"--swap wants a:b pairs, got: {tok}")
            a, b = tok.split(":", 1)
            swap_pairs.append((check(a), check(b)))

    page_map = load_page_map(args.page_map) if args.page_map else None
    if args.page_map and page_map is None:
        try:
            with open(args.page_map) as fh:
                loaded = json.load(fh)
            page_map = [str(i) for i in (loaded.get("page_ids") if isinstance(loaded, dict) else loaded)]
        except (OSError, ValueError, AttributeError, TypeError):
            sys.exit(f"--page-map could not be read as a page map: {args.page_map}")

    process_video(
        video_path=args.video,
        output_pdf=output_pdf,
        output_epub=args.epub,
        ocr=args.ocr,
        langs=[s.strip() for s in args.lang.split(",") if s.strip()],
        title=args.title,
        embed_images=args.embed_images,
        stride=args.stride,
        blur_threshold=args.blur_threshold,
        min_area_ratio=args.min_area,
        motion_threshold=args.motion_threshold,
        min_still_frames=args.min_still_frames,
        enhance=not args.no_enhance,
        deskew=not args.no_deskew,
        no_warp=args.no_warp,
        split_spreads=not args.no_split_spreads,
        rotate=args.rotate,
        pages_from=args.pages_from,
        dedupe=not args.no_dedupe,
        review_sheet=args.review_sheet,
        drop_pages=drop_pages,
        ocr_cache=args.ocr_cache,
        swap_pairs=swap_pairs,
        page_map=page_map,
        patches=load_patches(args.patch) if args.patch else None,
        from_images=args.from_images,
        drop_turns=args.drop_turns,
        check_folios=args.check_folios,
        drop_blank=not args.keep_blank_pages,
        cover_spec=None if args.cover == "none" else args.cover,
        link_notes_flag=args.link_notes,
        link_citations_flag=args.link_citations,
        link_index_flag=args.link_index,
        typography_flag=args.typography,
        pdf_text_layer=args.pdf_text_layer,
        decisions_path=args.apply_decisions,
    )
    progress.emit(event="result", epub=args.epub, pdf=output_pdf)


if __name__ == "__main__":
    main()
