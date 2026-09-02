"""Page geometry from a photograph: one learned rectifier, then the split.

Every photographed page must reach the OCR as a flat, upright rectangle.
For a long time that took a ladder here — outline detection, a perspective
warp per page plane, guards on the warp, a deskew, two curl models under a
witness — and the ladder's failures were the pages that looked worst: a
page shipped rotated inside its canvas with black wedges, a column cut
where a canvas was sized to the wrong box. Measured on two books (see the
commit that introduced this module), a document rectifier trained for
exactly this problem — UVDoc (Verhoeven, Magne & Sorkine-Hornung, SIGGRAPH
Asia 2023), served through transformers — read the same words at equal or
higher confidence on every page, levelled the pages the ladder had left
rotated, and returned the column the ladder had cut. It replaces the
ladder outright.

The model predicts, for a regular grid over the flattened sheet, where in
the photograph each grid point lies. Sampling the photograph along that
grid IS the rectification: perspective, the curl into the gutter and the
crease at the spine come out of one map. What remains classical is small
and lives beside it: the output is sized so the sheet keeps the
proportions and scale it was photographed at; dark surround the model's
boundary let through is trimmed back to paper; and a witness reads the
result against the photograph so a model failure is reported, never
shipped quietly. Splitting a spread at its gutter, enhancing and the final
deskew are the pipeline's own and unchanged.
"""
from __future__ import annotations

import os
import subprocess

import cv2
import numpy as np

MODEL_ID = "PaddlePaddle/UVDoc_safetensors"

# Columns and rows at the sheet's edge darker than this fraction of the
# page's own brightness are surround, not paper: the desk, a cover strip,
# the spine shadow the split left on a page's inner edge. Shared with the
# gutter hunt's notion of "page rather than the surface behind it".
PAPER_LEVEL_RATIO = 0.6
# Along columns the mark is higher: a spine shadow's darkest column sat
# at 0.59-0.60 of paper on three pages and slipped under a 0.6 cutoff. No
# text column's median comes near 0.7 of paper; only a figure spanning
# the full height could, and it would have to sit inside the outer 12%.
PAPER_LEVEL_RATIO_COLS = 0.7
# A column or row is judged by this percentile of its pixels: dark here
# means dark along nine tenths of its length. Deskew's white wedges and a
# squared page's paper fill touch an edge column for far less than that.
PAPER_PROFILE_PCT = 90
# The trim never takes more than this fraction of a side. The model's
# boundary is close; a trim that wants more is reading a dark plate or a
# dark page as surround, and a plate is content.
PAPER_MAX_TRIM = 0.12
# A bright gap this narrow (fraction of the side) between the cut and a
# dark run is the facing page's sliver, not the page's margin; measured
# gutter slivers run 1-2%, the narrowest real margin several times that.
PAPER_SLIVER = 0.03
# A wider gap is still surround when it is dimmer than this fraction of
# the page's paper: the facing page's penumbra measured 0.65-0.8 of paper,
# a real margin 0.95 and up.
PAPER_DIM_RATIO = 0.85
# A rectified sheet that reads fewer than this fraction of the photograph's
# words has lost text — a folded map, a boundary that excluded a page. The
# build keeps the photograph for that page and says so.
WITNESS_KEEP_RATIO = 0.6

_state: dict = {"tried": False, "model": None, "proc": None, "device": None,
                "error": None}


def _load() -> bool:
    if _state["tried"]:
        return _state["model"] is not None
    _state["tried"] = True
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel
        model = AutoModel.from_pretrained(MODEL_ID)
        proc = AutoImageProcessor.from_pretrained(MODEL_ID)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        model.to(device).eval()
        _state.update(model=model, proc=proc, device=device)
        return True
    except Exception as exc:  # missing extra, no network for the first fetch, ...
        _state["error"] = f"{type(exc).__name__}: {exc}"
        return False


def available() -> bool:
    """Whether the rectifier can run here (the extra installed, weights on hand)."""
    return _load()


def unavailable_reason() -> str:
    _load()
    return _state["error"] or "not tried"


def grid_for(image: np.ndarray) -> "np.ndarray | None":
    """The rectifier's map for one photograph: (rows, cols, 2) positions in
    the photograph, in pixels, for a regular grid over the flattened sheet.
    None when the model is not available."""
    if not _load():
        return None
    import torch
    from PIL import Image
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    inputs = _state["proc"](images=rgb, return_tensors="pt")
    inputs.pop("original_images", None)
    with torch.no_grad():
        out = _state["model"](**{k: v.to(_state["device"]) for k, v in inputs.items()})
    g = out.last_hidden_state[0].float().cpu().numpy()  # 2 x rows x cols, in -1..1
    h, w = image.shape[:2]
    return np.stack([(g[0] + 1.0) / 2.0 * (w - 1), (g[1] + 1.0) / 2.0 * (h - 1)],
                    axis=-1).astype(np.float32)


def sheet_size(grid: np.ndarray) -> tuple[int, int]:
    """(width, height) the flattened sheet is rendered at.

    The model says nothing about the sheet's proportions — it maps a grid,
    and any output size can be asked of it. Rendering at the photograph's
    own size (the reference implementation's choice) squeezes a spread
    into the frame's aspect and stretches a single page to landscape. The
    sheet's size as photographed is the grid's own extent: the mean length
    of its rows and of its columns, walked point to point in the
    photograph. For a shot taken roughly overhead that is the sheet at the
    scale the camera recorded it, so the type keeps its size and the OCR
    cache its resolution.
    """
    rows = np.linalg.norm(np.diff(grid, axis=1), axis=-1).sum(axis=1).mean()
    cols = np.linalg.norm(np.diff(grid, axis=0), axis=-1).sum(axis=0).mean()
    # A span of W-1 pixel steps is W pixels wide.
    return max(8, int(round(float(rows))) + 1), max(8, int(round(float(cols))) + 1)


def sample(image: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """The photograph resampled along the grid: the flattened sheet.

    The coarse grid is upsampled with the corners pinned to the corners
    (align_corners), as the model's own renderer does — an off-by-half-cell
    upsampling would rescale the sheet by (cols-1)/cols, three percent on
    a 31-column grid, and replicate its edge cells.
    """
    import torch
    import torch.nn.functional as F
    out_w, out_h = sheet_size(grid)
    t = torch.from_numpy(np.ascontiguousarray(grid.transpose(2, 0, 1))).unsqueeze(0)
    dense = F.interpolate(t, size=(out_h, out_w), mode="bilinear",
                          align_corners=True)[0].numpy()
    mx, my = np.ascontiguousarray(dense[0]), np.ascontiguousarray(dense[1])
    return cv2.remap(image, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rectify(image: np.ndarray) -> tuple[np.ndarray, bool]:
    """The photograph as a flat sheet, or the photograph itself when the
    rectifier is not available. The flag says which."""
    if image.ndim != 3:
        return image, False
    grid = grid_for(image)
    if grid is None:
        return image, False
    return sample(image, grid), True


def paper_crop(page: np.ndarray) -> np.ndarray:
    """Trim dark surround from the sheet's edges, back to the paper.

    The model's boundary is close to the sheet's but not on it: a strip
    of desk, the red of a cover, the spine's shadow left on a page's inner
    edge by the split. Each side walks inward while the column (row) is
    dark along nearly its whole length — its 90th percentile below the
    page's brightness ratio, so a finger over one stretch of a column, or
    a plate covering half of it, cannot read as surround — and never past
    PAPER_MAX_TRIM of the side.
    """
    h, w = page.shape[:2]
    if h < 16 or w < 16:
        return page
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
    pw = 400
    small = cv2.resize(gray, (pw, max(8, int(h * pw / w))))
    sh, sw = small.shape
    core = small[int(sh * 0.2):max(int(sh * 0.2) + 1, int(sh * 0.8)),
                 int(sw * 0.2):max(int(sw * 0.2) + 1, int(sw * 0.8))]
    paper = float(np.percentile(core, 75))
    dim = paper * PAPER_DIM_RATIO
    # The profile is each column's (row's) 90th percentile, not its
    # median: surround — desk, cover, the spine's shadow — is dark along
    # the WHOLE column, while a large dark plate is dark along only the
    # part it covers. Judged by the median, a plate covering 55% of a
    # column read as surround, the walk chained through the text beside
    # it to reach it, and the text column at the page's inner edge was
    # cut. A column must be dark in nine tenths of its length to count.
    cols = np.percentile(small, PAPER_PROFILE_PCT, axis=0)
    rows = np.percentile(small, PAPER_PROFILE_PCT, axis=1)

    def walk(p, chain):
        level = paper * (PAPER_LEVEL_RATIO_COLS if chain else PAPER_LEVEL_RATIO)
        """How far in from one edge the surround reaches: dark runs, and —
        along columns only — dark runs behind a gap that is not paper.
        The split leaves the facing page's edge between the cut and the
        spine's shadow: a sliver at most a few percent wide, or a wider
        penumbra that is dim (measured 145-173 grey against paper at
        220). A walk that stopped there left the shadow's line on 27 of
        130 pages. A gap at paper brightness is the page's own margin,
        so a rule behind a real margin is never reached; rows never chain
        at all."""
        lim = len(p)
        sliver = max(2, int(lim * PAPER_SLIVER / PAPER_MAX_TRIM))
        i = 0
        while i < lim:
            k = i
            while k < lim and p[k] >= level:
                k += 1
            if k >= lim or (k > i and not chain):
                break
            # The gap's MEAN, not its brightest column: a penumbra fades
            # from near-paper at the cut to the shadow, and its first
            # column alone sat above the mark on two pages.
            gap_is_surround = (k - i <= sliver) or float(np.mean(p[i:k])) < dim
            if k > i and not gap_is_surround:
                break
            j = k
            while j < lim and p[j] < level:
                j += 1
            i = j
        return i

    def trim(prof, n, chain):
        lim = int(n * PAPER_MAX_TRIM)
        a = walk(prof[:lim], chain)
        b = n - walk(prof[n - lim:][::-1], chain)
        return a, b

    x0, x1 = trim(cols, sw, chain=True)
    y0, y1 = trim(rows, sh, chain=False)
    X0, X1 = int(x0 * w / sw), int(np.ceil(x1 * w / sw))
    Y0, Y1 = int(y0 * h / sh), int(np.ceil(y1 * h / sh))
    if X1 - X0 < w * 0.5 or Y1 - Y0 < h * 0.5:
        return page
    return page[Y0:Y1, X0:X1]


# The text block's two edges must converge by at least this before the
# page is squared — below it the fit's own noise is as large as the
# correction — and a convergence above the cap is a misread of the block
# (a figure taken for lines), not perspective, and is left alone.
SQUARE_MIN_DEG = 0.4
SQUARE_MAX_DEG = 4.0
SQUARE_MIN_LINES = 10


def text_edges(page: np.ndarray, probe_w: int = 1200):
    """The text block's left and right edges as lines x = a*y + b, in the
    page's own pixels, from the full-measure lines of justified text.

    Returns (left, right, n_lines) with each edge as (a, b), or None when
    fewer than SQUARE_MIN_LINES full lines can be read. Lines are taken
    at the probe scale, an edge is the extent of each line's ink, and only
    lines near the dominant extents count — a figure, a centred heading
    or an indented last line would otherwise swing the fit.
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY) if page.ndim == 3 else page
    h, w = gray.shape
    s = probe_w / float(w)
    g = cv2.resize(gray, (probe_w, max(8, int(h * s))))
    ph, pw = g.shape
    hat = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                           cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    ink = (hat > max(12, np.percentile(hat, 99) * 0.35)).astype(np.uint8)
    m = int(0.02 * ph), int(0.02 * pw)
    ink[:m[0]] = 0; ink[-m[0]:] = 0; ink[:, :m[1]] = 0; ink[:, -m[1]:] = 0
    merged = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((1, 15), np.uint8))
    rows = merged.mean(axis=1) > 0.03
    L, R, Y = [], [], []
    y = 0
    while y < ph:
        if rows[y]:
            j = y
            while j < ph and rows[j]:
                j += 1
            if 4 <= j - y <= 0.05 * ph:
                cols = np.where(merged[y:j].max(axis=0) > 0)[0]
                if len(cols) and cols[-1] - cols[0] > 0.45 * pw:
                    L.append(cols[0]); R.append(cols[-1]); Y.append((y + j) / 2.0)
            y = j
        else:
            y += 1
    if len(L) < SQUARE_MIN_LINES:
        return None
    L, R, Y = np.array(L, float), np.array(R, float), np.array(Y, float)
    keepL = np.abs(L - np.median(L)) < 0.03 * pw
    keepR = np.abs(R - np.median(R)) < 0.03 * pw

    def fit(x, yy):
        if len(x) < SQUARE_MIN_LINES:
            return None
        for _ in range(2):
            a, b = np.polyfit(yy, x, 1)
            res = x - (a * yy + b)
            keep = np.abs(res) < 3 * (np.median(np.abs(res)) + 1e-6) + 2
            if keep.sum() < SQUARE_MIN_LINES:
                return None
            x, yy = x[keep], yy[keep]
        a, b = np.polyfit(yy, x, 1)
        return float(a), float(b / s)  # slope is scale-free; intercept back to page pixels

    left, right = fit(L[keepL], Y[keepL]), fit(R[keepR], Y[keepR])
    if left is None or right is None:
        return None
    return left, right, int(min(keepL.sum(), keepR.sum()))


def square(page: np.ndarray) -> tuple[np.ndarray, float]:
    """The page with its text block's edges made parallel.

    A learned rectifier leaves a small, consistent trapezoid — measured at
    0.9° median across two books where a correct homography leaves 0.45°
    — because its smooth grid cannot hold the crease at the spine
    exactly. The block's own edges say how much: the trapezoid they span
    is mapped to the rectangle of the same height and mean width, and the
    whole page goes with it. Returns the page and the convergence found
    (degrees); an unmeasurable or already-square page comes back as it
    was, with 0.0.
    """
    found = text_edges(page)
    if found is None:
        return page, 0.0
    (aL, bL), (aR, bR), _n = found
    conv = float(np.degrees(np.arctan(aR) - np.arctan(aL)))
    if abs(conv) < SQUARE_MIN_DEG or abs(conv) > SQUARE_MAX_DEG:
        return page, conv
    h, w = page.shape[:2]
    yt, yb = 0.15 * h, 0.85 * h
    xl = lambda y: aL * y + bL
    xr = lambda y: aR * y + bR
    src = np.float32([[xl(yt), yt], [xr(yt), yt], [xr(yb), yb], [xl(yb), yb]])
    x0 = (xl(yt) + xl(yb)) / 2.0
    wm = ((xr(yt) - xl(yt)) + (xr(yb) - xl(yb))) / 2.0
    dst = np.float32([[x0, yt], [x0 + wm, yt], [x0 + wm, yb], [x0, yb]])
    H = cv2.getPerspectiveTransform(src, dst)
    # The canvas holds the WHOLE warped page: widening the block's narrow
    # end pushes that end's margin outward, and a canvas of the page's
    # own size would lose whatever sat there — a line that reaches the
    # edge, a folio. The page's corners are mapped, the canvas sized to
    # their extent, and the warp shifted into it.
    corners = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]).reshape(-1, 1, 2)
    moved = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    x_min, y_min = np.floor(moved.min(axis=0)); x_max, y_max = np.ceil(moved.max(axis=0))
    shift = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    out_w, out_h = int(x_max - x_min) + 1, int(y_max - y_min) + 1
    paper = tuple(float(v) for v in np.median(page.reshape(-1, page.shape[-1]) if page.ndim == 3 else page.reshape(-1, 1), axis=0))
    out = cv2.warpPerspective(page, shift @ H, (out_w, out_h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=paper)
    return out, conv


def witness(image: np.ndarray, workdir: str,
            scale: float = 0.6) -> tuple[int, float, float]:
    """(words, mean confidence, coverage) from tesseract.

    The third value is the fraction of ten horizontal bands that contain
    at least one word: testimony spread across the page, which a caption
    cluster plus a folio cannot fake. Used to compare a rectified sheet
    with its photograph — a sheet that reads far fewer words than the
    photograph has lost text — and offered to anything else that needs an
    independent reader.
    """
    p = os.path.join(workdir, "_witness.png")
    h, w = image.shape[:2]
    cv2.imwrite(p, cv2.resize(image, (max(8, int(w * scale)), max(8, int(h * scale)))))
    try:
        r = subprocess.run(["tesseract", p, "stdout", "--psm", "3", "tsv"],
                           capture_output=True, timeout=300)
    except Exception:
        return 0, 0.0, 0.0
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
