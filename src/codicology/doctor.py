"""codicology doctor — can this environment actually run a build?

Quick mode answers in under a second, from facts that can be checked
without loading anything heavy: binaries, package versions, tesseract's
languages. --full also OCRs one synthetic page through the real backend,
because the expensive failure cannot be caught any other way: surya 0.2x
imports cleanly without the llama-server it delegates to, and fails on the
first real page of the first real book. An import check is insufficient by
construction.

--json prints a single JSON object on stdout for a program driving this
pipeline (the Calibre plugin does). Exit status is 0 only when a build
could run honestly — which includes the witness: tesseract is required,
for the same reasons convert refuses to start without it.
"""
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys

from . import __version__

PACKAGES = ("surya-ocr", "torch", "numpy", "Pillow",
            "pymupdf", "pypdfium2", "ebooklib", "img2pdf", "lxml")

# OpenCV is asked for by what imports, not by what pip lists. Every one of
# its wheels unpacks into the same cv2 directory, so two of them installed
# together leave whichever pip wrote last, and `pip check` is satisfied with
# both. This project used to ask for opencv-python while Surya pins
# opencv-python-headless==4.11.0.86: both were installed, every book was
# read on OpenCV 5.0.0, and nothing said so.
OPENCV_WHEELS = ("opencv-python-headless", "opencv-python",
                 "opencv-contrib-python-headless", "opencv-contrib-python")

# What the smoke test's page says, and the overlap that counts as reading it.
_TEST_LINES = ("CODICOLOGY DOCTOR", "the quick brown fox reads page 42")
_EXPECTED = {"codicology", "doctor", "quick", "brown", "fox", "reads", "page"}
_MIN_MATCHED = 3


def _run(argv, timeout=10):
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout)
    except Exception:
        return None


def _pkg_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _surya_opencv_pin():
    """The OpenCV release surya-ocr's own metadata pins, if it pins one."""
    try:
        requires = importlib.metadata.requires("surya-ocr") or []
    except importlib.metadata.PackageNotFoundError:
        return None
    for req in requires:
        m = re.match(r"opencv[-_.]python[-_.]headless\s*==\s*([\w.]+)", req,
                     re.I)
        if m:
            return m.group(1)
    return None


def _opencv():
    """The OpenCV that `import cv2` would load, read from the version file
    its wheel ships rather than by loading it."""
    info = {"wheels": {w: v for w in OPENCV_WHEELS
                       if (v := _pkg_version(w))},
            "surya_pins": _surya_opencv_pin()}
    try:
        spec = importlib.util.find_spec("cv2")
    except (ImportError, ValueError):
        spec = None
    info["present"] = bool(spec and spec.origin)
    if info["present"]:
        try:
            with open(os.path.join(os.path.dirname(spec.origin), "version.py"),
                      encoding="utf-8") as fh:
                m = re.search(r'opencv_version\s*=\s*"([^"]+)"', fh.read())
            if m:
                info["version"] = m.group(1)
        except OSError:
            pass
    return info


def _opencv_warning(cv):
    """What to say when the OpenCV that imports is not the one Surya pins,
    or when two wheels are sharing the directory."""
    wheels, pin, version = cv["wheels"], cv.get("surya_pins"), cv.get("version")
    wrong = bool(pin and version and version != pin)
    if len(wheels) < 2 and not wrong:
        return None
    listed = ", ".join(f"{w} {v}" for w, v in wheels.items()) or "none listed"
    said = (f"{len(wheels)} OpenCV wheels are installed ({listed}) into the "
            "one cv2 directory, and the last one written is what imports"
            if len(wheels) > 1 else f"OpenCV is installed as {listed}")
    if wrong:
        said += f": cv2 is {version} where surya-ocr pins {pin}"
    elif version:
        said += f": cv2 is {version}"
    others = [w for w in wheels if w != "opencv-python-headless"]
    fix = "pip install --force-reinstall --no-deps opencv-python-headless" \
          + (f"=={pin}" if pin else "")
    if others:
        fix = f"pip uninstall {' '.join(others)}, then " + fix
    return f"{said} — {fix}"


def _tesseract():
    path = shutil.which("tesseract")
    if not path:
        return {"present": False}
    info = {"present": True, "path": path}
    r = _run([path, "--version"])
    if r:
        m = re.search(r"tesseract\s+v?([\w.]+)", (r.stdout or "") + (r.stderr or ""))
        if m:
            info["version"] = m.group(1)
    r = _run([path, "--list-langs"])
    if r:
        # First line is the "List of available languages (N):" header.
        text = (r.stdout or "") + (r.stderr or "")
        info["langs"] = [l.strip() for l in text.splitlines()[1:] if l.strip()]
    return info


def _vision():
    """Apple Vision is the adjudication ladder's optional fourth rung —
    macOS only, graceful everywhere else. Absent, books build identically
    and review sheets simply carry more open abstains; this line exists
    so a fat sheet on another platform explains itself."""
    try:
        import Vision  # noqa: F401
        return {"present": True}
    except ImportError:
        return {"present": False}


def _llama_server():
    override = os.environ.get("LLAMA_CPP_BINARY")
    if override and os.path.exists(override):
        return {"present": True, "path": override, "via": "LLAMA_CPP_BINARY"}
    path = shutil.which("llama-server")
    return {"present": bool(path), "path": path}


def _nvidia():
    path = shutil.which("nvidia-smi")
    if not path:
        return {"present": False}
    r = _run([path, "--query-gpu=name,memory.total", "--format=csv,noheader"],
             timeout=5)
    gpus = [l.strip() for l in (r.stdout if r else "").splitlines() if l.strip()]
    return {"present": True, "gpus": gpus}


def surya_needs_server(version) -> bool:
    """v0.20.0 rewrote surya around an external inference server —
    llama.cpp on CPU and Apple Silicon, vllm on NVIDIA."""
    if not version:
        return False
    try:
        major, minor = (int(x) for x in version.split(".")[:2])
    except ValueError:
        return True          # unparseable is newer than anything we know
    return (major, minor) >= (0, 20)


def report(full=False, ocr="surya", langs=("en",)):
    rep = {
        "codicology": __version__,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "machine": platform.machine(),
        "packages": {p: _pkg_version(p) for p in PACKAGES},
        "opencv": _opencv(),
        "tesseract": _tesseract(),
        "llama_server": _llama_server(),
        "vision": _vision(),
        "nvidia": _nvidia(),
    }

    problems, warnings = [], []
    if not rep["tesseract"]["present"]:
        problems.append(
            "tesseract is required — it is the witness two checks depend on, "
            "and convert refuses to start without it "
            "(macOS: brew install tesseract; Debian: apt install tesseract-ocr)")

    surya = rep["packages"]["surya-ocr"]
    if ocr == "surya":
        if not surya:
            problems.append("surya-ocr is not installed in this environment")
        elif surya_needs_server(surya) and not rep["llama_server"]["present"]:
            if rep["nvidia"]["present"]:
                warnings.append(
                    "no llama-server found; assuming vllm is configured for "
                    "the NVIDIA GPU — if it is not, the first page will fail")
            else:
                problems.append(
                    f"surya {surya} serves inference through llama.cpp on "
                    "this hardware and no llama-server was found "
                    "(brew install llama.cpp, or set LLAMA_CPP_BINARY)")
    if rep["nvidia"]["present"] and rep["llama_server"]["present"]:
        warnings.append(
            "an NVIDIA GPU is present, but unless vllm is configured surya "
            "reads on CPU through llama.cpp — hours instead of minutes on a "
            "long book")

    for pkg in ("pymupdf", "pypdfium2", "ebooklib", "Pillow"):
        if not rep["packages"][pkg]:
            problems.append(f"{pkg} is not installed")
    if not rep["opencv"]["present"]:
        problems.append("OpenCV is not installed "
                        "(pip install opencv-python-headless)")
    elif (said := _opencv_warning(rep["opencv"])):
        warnings.append(said)

    if full:
        rep["smoke"] = smoke(ocr, list(langs))
        if not rep["smoke"]["ok"]:
            problems.append("the OCR smoke test failed: "
                            + rep["smoke"].get("error",
                                               "the backend did not read the test page"))

    rep["warnings"] = warnings
    rep["problems"] = problems
    rep["ok"] = not problems
    return rep


def _test_page():
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (1400, 500), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=72)
    except TypeError:                        # Pillow < 10 has no size arg
        font = ImageFont.load_default()
    draw.text((80, 90), _TEST_LINES[0], fill="black", font=font)
    draw.text((80, 260), _TEST_LINES[1], fill="black", font=font)
    return img


def _torch_devices():
    try:
        import torch
    except Exception:
        return None
    mps = getattr(torch.backends, "mps", None)
    return {"cuda": bool(torch.cuda.is_available()),
            "mps": bool(mps and mps.is_available())}


def smoke(ocr, langs):
    """Read one synthetic page through the real backend and report what
    actually answered. This is the check imports cannot perform."""
    import time
    from .pipeline import load_backend
    t0 = time.time()
    try:
        backend = load_backend(ocr, langs)
    except SystemExit as exc:                # load_backend refuses via exit
        return {"ok": False, "backend": ocr, "error": str(exc.code)}
    except Exception as exc:
        return {"ok": False, "backend": ocr,
                "error": f"{type(exc).__name__}: {exc}"}
    try:
        text = backend.run([_test_page()])[0]
    except Exception as exc:
        return {"ok": False, "backend": backend.name,
                "seconds": round(time.time() - t0, 1),
                "error": f"{type(exc).__name__}: {exc}"}
    words = set(re.findall(r"[a-z]+", text.lower()))
    matched = sorted(_EXPECTED & words)
    out = {"ok": len(matched) >= _MIN_MATCHED, "backend": backend.name,
           "seconds": round(time.time() - t0, 1),
           "read": " ".join(text.split())[:120], "matched": matched}
    shape = getattr(backend, "_shape", None)
    if shape:
        out["shape"] = shape
    if shape == "predictor":
        out["layout"] = layout_smoke()
        if not out["layout"]["ok"]:
            out["ok"] = False
            out["error"] = ("surya's layout step returned no boxes — it is "
                            "the fallback for pages the full-page read "
                            "fails, and without it they come back empty"
                            + (f" ({out['layout']['error']})"
                               if out["layout"].get("error") else ""))
    devices = _torch_devices()
    if devices:
        out["devices"] = devices
    return out


def layout_smoke():
    """Ask surya's layout step, guided, for the boxes on the test page.

    The page read above never exercises it: layout only runs when a
    full-page read fails, so a layout step the server refuses goes unseen
    until a hard page needs it — and that page then ships blank. It went
    unseen for two months that way (llama.cpp could not compile the
    schema's `\\d`; see pipeline.llamacpp_safe_schemas)."""
    try:
        from surya.layout import LayoutPredictor
        from .pipeline import llamacpp_safe_schemas
        llamacpp_safe_schemas()
        res = LayoutPredictor()([_test_page()])[0]
    except Exception as exc:
        return {"ok": False, "boxes": 0,
                "error": f"{type(exc).__name__}: {exc}"}
    return {"ok": bool(res.bboxes), "boxes": len(res.bboxes)}


def _human(rep):
    print(f"codicology {rep['codicology']} on Python {rep['python']} "
          f"({rep['machine']})")
    print(f"  {rep['executable']}")
    pk = rep["packages"]
    have = ", ".join(f"{k} {v}" for k, v in pk.items() if v)
    missing = [k for k, v in pk.items() if not v]
    print(f"  packages: {have or 'none found'}")
    if missing:
        print(f"  missing:  {', '.join(missing)}")
    cv = rep["opencv"]
    print("  opencv: " + (f"{cv.get('version', '?')} (from "
                          + (", ".join(cv["wheels"]) or "an unlisted wheel")
                          + ")" if cv["present"] else "NOT FOUND"))
    t = rep["tesseract"]
    if t["present"]:
        langs = t.get("langs") or []
        shown = ", ".join(langs[:8]) + (f", … {len(langs)} total"
                                        if len(langs) > 8 else "")
        print(f"  tesseract {t.get('version', '?')} at {t['path']} — "
              f"languages: {shown or '?'}")
    else:
        print("  tesseract: NOT FOUND")
    l = rep["llama_server"]
    print(f"  llama-server: {l['path'] if l['present'] else 'not found'}")
    v = rep.get("vision", {})
    try:
        from .dewarp import available as _dewarp_ok
        print("  leptonica: " + ("found (text-line dewarp active)"
                                 if _dewarp_ok() else
                                 "not found — pages keep the rigid deskew "
                                 "alone"))
    except Exception:
        pass
    print("  apple vision: " + ("present (adjudication's optional rung)"
                                if v.get("present") else
                                "absent — review sheets will carry more "
                                "open abstains; builds are unaffected"))
    n = rep["nvidia"]
    print(f"  nvidia: {', '.join(n['gpus']) if n.get('gpus') else ('present' if n['present'] else 'none')}")
    s = rep.get("smoke")
    if s is None:
        print("  smoke test: skipped (pass --full to read one synthetic page)")
    elif s["ok"]:
        extra = f" (shape: {s['shape']})" if s.get("shape") else ""
        if s.get("devices"):
            extra += f" — cuda {s['devices']['cuda']}, mps {s['devices']['mps']}"
        print(f"  smoke test: {s['backend']} read the test page in "
              f"{s['seconds']}s{extra}")
        print(f"    read: {s['read']!r}")
        if "layout" in s:
            print(f"    layout (the fallback): {s['layout']['boxes']} box(es)")
    else:
        print(f"  smoke test FAILED"
              + (f" after {s['seconds']}s" if "seconds" in s else "")
              + f": {s.get('error', 'did not read the test page')}")
    for w in rep["warnings"]:
        print(f"  [~] {w}")
    for p in rep["problems"]:
        print(f"  [!] {p}")
    print("VERDICT:", "ready" if rep["ok"] else "NOT READY")


def main(full=False, ocr="surya", lang="en", as_json=False) -> int:
    langs = [s.strip() for s in lang.split(",") if s.strip()] or ["en"]
    rep = report(full=full, ocr=ocr, langs=langs)
    if as_json:
        print(json.dumps(rep, indent=2))
    else:
        _human(rep)
    return 0 if rep["ok"] else 1
