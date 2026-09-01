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
    height = 0.5 * (dims[1] + 2 * MARGIN) * h_img
    height = max(64, int(np.round(height / 16) * 16))
    width = int(np.round(height * (dims[0] + 2 * MARGIN)
                         / (dims[1] + 2 * MARGIN) / 16) * 16)
    hs, ws = max(4, height // 16), max(4, width // 16)
    xr = np.linspace(-MARGIN, dims[0] + MARGIN, ws)
    yr = np.linspace(-MARGIN, dims[1] + MARGIN, hs)
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

    y-span is the fraction of the page height the witnessed words cover.
    The sheet models page-wide curvature, so testimony clustered in one
    band — a plate's captions — is testimony about almost none of what
    the transform touches; two plate pages were once smeared into swirls
    while their captions read the same before and after.
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
    confs, tops, bottoms = [], [], []
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
                tops.append(top)
                bottoms.append(top + height)
    span = ((max(bottoms) - min(tops)) / max(1.0, h * scale)) if confs else 0.0
    return len(confs), (float(np.mean(confs)) if confs else 0.0), float(span)
