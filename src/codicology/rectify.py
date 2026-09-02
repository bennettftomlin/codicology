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
# The trim never takes more than this fraction of a side. The model's
# boundary is close; a trim that wants more is reading a dark plate or a
# dark page as surround, and a plate is content.
PAPER_MAX_TRIM = 0.12
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
    edge by the split. Each side walks inward while the column (row)
    median stays below PAPER_LEVEL_RATIO of the page's brightness — a
    median, so a finger over one stretch of an otherwise bright column, or
    a photograph in the middle of a page, cannot read as surround — and
    never past PAPER_MAX_TRIM of the side.
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
    level = float(np.percentile(core, 75)) * PAPER_LEVEL_RATIO
    cols = np.median(small, axis=0)
    rows = np.median(small, axis=1)

    def trim(prof, n):
        lim = int(n * PAPER_MAX_TRIM)
        a = 0
        while a < lim and prof[a] < level:
            a += 1
        b = n
        while n - b < lim and prof[b - 1] < level:
            b -= 1
        return a, b

    x0, x1 = trim(cols, sw)
    y0, y1 = trim(rows, sh)
    X0, X1 = int(x0 * w / sw), int(np.ceil(x1 * w / sw))
    Y0, Y1 = int(y0 * h / sh), int(np.ceil(y1 * h / sh))
    if X1 - X0 < w * 0.5 or Y1 - Y0 < h * 0.5:
        return page
    return page[Y0:Y1, X0:X1]


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
