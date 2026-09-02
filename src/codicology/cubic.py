"""The second geometry rung: a cubic-sheet model for pages the line finder
cannot read.

Leptonica's dewarper wants fifteen-odd full-measure text lines, and on one
photographed book 63 of 67 declined pages never model at ANY threshold —
pages built around insets, facsimiles and plates. page_dewarp finds text by
contour clustering and fits Zucker's cubic sheet instead, and it modelled
every one of a twelve-page sample of those declines. Its own OUTPUT is
disqualified for a facsimile — cropped to the text block, greyscale — but
the transform never was: the sheet is solved in scale-free normalised
coordinates, so this module takes the solved parameters and rebuilds the
remap at full resolution, in colour, with margins extended past the
modelled block (the cubic extrapolates calmly over blank paper).

The dependency is OPTIONAL and the degrade is graceful: without
page_dewarp installed this module reports unavailable, the affected pages
keep the plain deskew — exactly the old behaviour — and the build says so
once, because a fallback nobody hears about is a result nobody questions.

A page this rung corrects must still pass the witness: tesseract reads the
page before and after, and a correction that costs more than a point of
confidence is discarded. Measured before integration: mean +0.85 across
twelve declined pages, wins to +5.6, and the one page the witness rejects
is an inset-newsprint page where the "loss" is tesseract reading more tiny
newsprint — the acceptance test refusing it is the system working.
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess

import cv2
import numpy as np

_state = {"tried": False, "mod": None}
MARGIN = 0.07                 # page-plane units past the modelled block


def _load():
    if _state["tried"]:
        return _state["mod"]
    _state["tried"] = True
    try:
        import page_dewarp.image as pdi
        from page_dewarp.projection import project_xy
        from page_dewarp.normalisation import norm2pix
        from page_dewarp.options.core import Config

        class _NullRemap:
            """Capture nothing: the solver is wanted, its file output is not."""

            def __init__(self, *a, **k):
                self.threshfile = None

        pdi.RemappedImage = _NullRemap
        _state["mod"] = (pdi, project_xy, norm2pix, Config)
    except Exception:
        _state["mod"] = None
    return _state["mod"]


def available() -> bool:
    return _load() is not None


def _input_extent(project_xy, norm2pix, params, dims, shape,
                  reach: float = 0.6, step: float = 0.02):
    """The sheet-coordinate range (x0, x1, y0, y1) that covers the input.

    Starting from the fitted box, each side steps outward while at least
    half of a line of sample points still projects inside the image; the
    walk stops where the page ends (or the model, extrapolating into desk,
    folds away). Bounded by `reach` sheet units so a folded extrapolation
    cannot run off to infinity. A margin of one step is kept so the last
    inside line is inside the canvas too.
    """
    h, w = shape[:2]

    def inside_frac(xs, ys):
        xy = np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1)]).astype(np.float32)
        pts = norm2pix(shape, project_xy(xy, params), False).reshape(-1, 2)
        ok = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
        return float(ok.mean())

    ys_line = np.linspace(0, dims[1], 24)
    xs_line = np.linspace(0, dims[0], 24)
    x1 = dims[0]
    while x1 < dims[0] + reach and inside_frac(np.full_like(ys_line, x1 + step), ys_line) >= 0.5:
        x1 += step
    x0 = 0.0
    while x0 > -reach and inside_frac(np.full_like(ys_line, x0 - step), ys_line) >= 0.5:
        x0 -= step
    y1 = dims[1]
    while y1 < dims[1] + reach and inside_frac(xs_line, np.full_like(xs_line, y1 + step)) >= 0.5:
        y1 += step
    y0 = 0.0
    while y0 > -reach and inside_frac(xs_line, np.full_like(xs_line, y0 - step)) >= 0.5:
        y0 -= step
    return x0 - step, x1 + step, y0 - step, y1 + step


def cubic_dewarp(image: np.ndarray, workdir: str) -> tuple[np.ndarray, bool]:
    """The page under the cubic-sheet correction, or untouched.

    The solver reads from a file and narrates to stdout; both are kept out
    of the build's way. A page it cannot model comes back unchanged.
    """
    mod = _load()
    if mod is None or image.ndim != 3:
        return image, False
    pdi, project_xy, norm2pix, Config = mod
    src = os.path.join(workdir, "_cubic_in.png")
    if not cv2.imwrite(src, image):
        return image, False
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            wi = pdi.WarpedImage(src, config=Config())
        if not getattr(wi, "written", False):
            return image, False
        params = np.asarray(wi.params)
        dims = np.asarray(wi.page_dims, dtype=float)
    except Exception:
        return image, False
    finally:
        try:
            os.remove(src)
        except OSError:
            pass
    # 0.5 · dims · max-dim: the solver's normalised unit is 2/max(h, w) of
    # the source, so the max dimension — not the height — is what keeps the
    # type at its own size. Inherited as shape[0] it was exact on portrait
    # pages and rendered a landscape page's type at h/w scale; a synthetic
    # landscape page caught it at exactly 0.53x.
    h_img = max(image.shape[:2])
    # The canvas must cover the INPUT's extent under the model, not the
    # solver's fitted text box: the box is fitted to the text it found, and
    # everything it under-covers — a second column, a wide caption — used
    # to land outside the output. One page shipped with its right column
    # cut (351px, folio 86) and a synthetic came back shifted 8.6% -> 21%
    # on the left and 8.6% -> 0.25% on the right. Each side walks outward
    # from the box until the model's projections leave the image.
    x0, x1, y0, y1 = _input_extent(project_xy, norm2pix, params, dims,
                                   image.shape)
    height = 0.5 * (y1 - y0) * h_img
    height = max(64, int(np.round(height / 16) * 16))
    width = int(np.round(height * (x1 - x0) / max(1e-6, (y1 - y0)) / 16) * 16)
    hs, ws = max(4, height // 16), max(4, width // 16)
    xr = np.linspace(x0, x1, ws)
    yr = np.linspace(y0, y1, hs)
    xc, yc = np.meshgrid(xr, yr)
    xy = np.hstack([xc.reshape(-1, 1), yc.reshape(-1, 1)]).astype(np.float32)
    try:
        pts = project_xy(xy, params)
        pts = norm2pix(image.shape, pts, False)
    except Exception:
        return image, False
    mx = cv2.resize(pts[:, 0, 0].reshape(xc.shape), (width, height),
                    interpolation=cv2.INTER_CUBIC).astype(np.float32)
    my = cv2.resize(pts[:, 0, 1].reshape(yc.shape), (width, height),
                    interpolation=cv2.INTER_CUBIC).astype(np.float32)
    out = cv2.remap(image, mx, my, cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_REPLICATE)
    return out, True


def witness(image: np.ndarray, workdir: str,
            scale: float = 0.6) -> tuple[int, float, float]:
    """(words, mean confidence, y-span) from tesseract — the acceptance judge.

    The third value is COVERAGE: the fraction of ten horizontal bands
    that contain at least one word. The sheet models page-wide curvature,
    so testimony must be spread across the page: a plate's caption
    clusters plus a folio can stretch a raw top-to-bottom span past any
    bar while still describing almost nothing — one such page was smeared
    twice, surviving both a word-count floor and a span test. Bands
    cannot be stretched; they are either inhabited or empty.
    """
    p = os.path.join(workdir, "_cubic_wit.png")
    h, w = image.shape[:2]
    cv2.imwrite(p, cv2.resize(image, (int(w * scale), int(h * scale))))
    try:
        r = subprocess.run(["tesseract", p, "stdout", "--psm", "3", "tsv"],
                           capture_output=True, timeout=300)
    except Exception:
        return 0, 0.0
    finally:
        try:
            os.remove(p)
        except OSError:
            pass
    confs, centers = [], []
    for line in r.stdout.decode("utf-8", "replace").splitlines()[1:]:
        f = line.split("\t")
        if len(f) >= 12 and f[0] == "5" and f[11].strip():
            try:
                c = float(f[10])
                top, height = float(f[7]), float(f[9])
            except ValueError:
                continue
            if c >= 0:
                confs.append(c)
                centers.append(top + height / 2.0)
    if confs:
        bands = {min(9, int(10.0 * y / max(1.0, h * scale))) for y in centers}
        coverage = len(bands) / 10.0
    else:
        coverage = 0.0
    return len(confs), (float(np.mean(confs)) if confs else 0.0), coverage
