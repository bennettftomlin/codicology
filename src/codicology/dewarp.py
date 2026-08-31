"""Straighten a page's text lines using leptonica's dewarper.

An open book is a cylinder, and the four-corner warp that flattens the
photograph cannot flatten the binding's curve; the rigid deskew that follows
can only rotate. What survives both is bow — measured on one photographed
book at 3-4px typical and 5-6px at the ninth decile against a 30px line
pitch — plus the occasional shear the outline-fitted warp leaves behind.

Leptonica models both from the page's own text lines and applies the
correction as one remap. It is already on every machine this pipeline
supports, because tesseract links it. Three facts, each learned the hard
way, are load-bearing here:

- The textline finder is built for scans in the 300 dpi class: on the same
  page it models happily up to ~440 dpi and refuses at ~600 — and a 48MP
  phone photo of a small hardback IS a ~600 dpi capture. The native answer
  is redfactor=2: the model is built from a half-size binarised copy and
  the disparity scaled up when applied, so the page itself is never
  resampled below full resolution.
- In leptonica's 1bpp world, 1 is ink. Hand it white-on-black and it sees
  a blank page and reports "textline centers not found".
- It abstains: fewer than fifteen usable lines and no model is built. A
  declined page is returned untouched, which is exactly the ladder the
  caller wants — dewarp where the type can prove a curve, the rigid
  deskew alone everywhere else.

Measured before integration (130 pages, every text page of one book):
63 modelled, tesseract confidence mean +2.5 with not one page falling more
than a point, +25 words read per modelled page, and Surya↔tesseract
agreement rising on every sampled page while block, furniture and figure
counts held still.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import io
import os

import cv2
import numpy as np
from PIL import Image

_CANDIDATES = [
    "/opt/homebrew/opt/leptonica/lib/libleptonica.dylib",
    "/opt/homebrew/lib/libleptonica.dylib",
    "/usr/local/lib/libleptonica.dylib",
    "/usr/lib/x86_64-linux-gnu/liblept.so.5",
]

IFF_PNG = 3
_lib = None
_tried = False

# The model is built from the page's own text lines, and the finder's
# morphology suits pages up to roughly 2600px wide (measured; ~440 dpi on
# the book that set these numbers). Wider than this and the model must be
# built at half size — leptonica's redfactor=2 — with the disparity scaled
# back up by leptonica itself when applied.
FULL_RES_LIMIT = 2800


def _load():
    global _lib, _tried
    if _tried:
        return _lib
    _tried = True
    path = next((p for p in _CANDIDATES if os.path.exists(p)), None)
    path = path or ctypes.util.find_library("leptonica")
    if not path:
        return None
    try:
        lib = ctypes.CDLL(path)
        lib.pixReadMem.restype = ctypes.c_void_p
        lib.pixReadMem.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
        lib.pixWriteMem.restype = ctypes.c_int
        lib.pixWriteMem.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.c_ubyte)),
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p, ctypes.c_int]
        lib.pixDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.lept_free.argtypes = [ctypes.c_void_p]
        lib.dewarpaCreate.restype = ctypes.c_void_p
        lib.dewarpaCreate.argtypes = [ctypes.c_int] * 5
        lib.dewarpaUseBothArrays.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.dewarpCreate.restype = ctypes.c_void_p
        lib.dewarpCreate.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.dewarpaInsertDewarp.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.dewarpBuildPageModel.restype = ctypes.c_int
        lib.dewarpBuildPageModel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.dewarpaApplyDisparity.restype = ctypes.c_int
        lib.dewarpaApplyDisparity.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p]
        lib.dewarpaDestroy.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        if hasattr(lib, "setMsgSeverity"):
            lib.setMsgSeverity(5)          # errors only; Info/Warning are chat
        _lib = lib
    except (OSError, AttributeError):
        _lib = None
    return _lib


def available() -> bool:
    return _load() is not None


def _pix_from_png(data: bytes):
    return _load().pixReadMem(data, len(data))


def _png_from_pix(pix) -> bytes | None:
    lib = _load()
    buf = ctypes.POINTER(ctypes.c_ubyte)()
    size = ctypes.c_size_t()
    if lib.pixWriteMem(ctypes.byref(buf), ctypes.byref(size), pix, IFF_PNG):
        return None
    try:
        return bytes(bytearray(ctypes.cast(
            buf, ctypes.POINTER(ctypes.c_ubyte * size.value)).contents))
    finally:
        lib.lept_free(buf)


def _binarised_png(image: np.ndarray, redfactor: int,
                   binarise: str = "otsu") -> bytes:
    """A 1bpp PNG of the page's ink, at the model-building scale.

    The ink stays dark: a PNG's black bit becomes leptonica's 1, which is
    ink. This is the polarity that works; the other reads as a blank page.
    Otsu first — a global threshold suits an evenly lit page — and adaptive
    as the second look, for the page whose photograph or plate drags a
    global threshold away from the type.
    """
    img = image
    if redfactor == 2:
        img = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2),
                         interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    if binarise == "adaptive":
        binary = cv2.adaptiveThreshold(gray, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 51, 18)
    else:
        _, binary = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out = io.BytesIO()
    Image.fromarray(binary).convert("1").save(out, format="PNG")
    return out.getvalue()


def _attempt(lib, page_png: bytes, image: np.ndarray, redfactor: int,
             minlines: int = 15, binarise: str = "otsu"):
    pixs = _pix_from_png(page_png)
    pixb = _pix_from_png(_binarised_png(image, redfactor, binarise))
    if not pixs or not pixb:
        return None
    dewa = lib.dewarpaCreate(1, 0, redfactor, minlines, 0)
    pixd = ctypes.c_void_p()
    try:
        lib.dewarpaUseBothArrays(dewa, 1)
        dew = lib.dewarpCreate(pixb, 0)
        if not dew:
            return None
        lib.dewarpaInsertDewarp(dewa, dew)          # dewa owns dew now
        if lib.dewarpBuildPageModel(dew, None) != 0:
            return None
        if lib.dewarpaApplyDisparity(dewa, 0, pixs, 255, 0, 0,
                                     ctypes.byref(pixd), None) != 0 or not pixd:
            return None
        return _png_from_pix(pixd)
    finally:
        da = ctypes.c_void_p(dewa)
        lib.dewarpaDestroy(ctypes.byref(da))
        for p in (ctypes.c_void_p(pixs), ctypes.c_void_p(pixb), pixd):
            if p:
                lib.pixDestroy(ctypes.byref(p))


def dewarp_page(image: np.ndarray,
                minlines: int = 10,
                binarise: str = "otsu") -> tuple[np.ndarray, bool]:
    """The page with its text lines straightened, or the page untouched.

    The build scale is chosen from the page itself — full resolution where
    the finder's envelope allows it, half where the capture outruns it —
    and the other factor is tried when the first declines, so a mixed book
    needs no per-book setting. Both declining falls through to the caller's
    rigid deskew, which has already run.

    minlines is leptonica's acceptance bar. Its own default is 15; the
    default here is 10, measured on one book: the 10-14-line cohort was
    three pages, every one improving under the witness (+2.3 mean, +0.6
    worst), and below 10 the bar admits nothing at all — 63 of 67 declined
    pages never model at ANY threshold, so the finder's eyesight, not its
    standards, is what limits coverage.
    """
    lib = _load()
    if lib is None or image.ndim != 3:
        return image, False
    ok, enc = cv2.imencode(".png", image)
    if not ok:
        return image, False
    page_png = enc.tobytes()
    order = (2, 1) if image.shape[1] > FULL_RES_LIMIT else (1, 2)
    for redfactor in order:
        out_png = _attempt(lib, page_png, image, redfactor, minlines,
                           binarise)
        if out_png is None:
            continue
        out = cv2.imdecode(np.frombuffer(out_png, np.uint8), cv2.IMREAD_COLOR)
        if out is None or out.shape != image.shape:
            continue
        if np.array_equal(out, image):
            continue
        return out, True
    return image, False
