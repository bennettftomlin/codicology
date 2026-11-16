"""Pages from photographs and patches: a folder of photographs read in the
order they were taken (HEIC too), bursts of one page grouped and the best
kept, patch photographs put in place of the pages they mend, scanned pages
dewarped, and the durable names pages are known by across extractions.
"""
from __future__ import annotations

import cv2
import glob
import json
import numpy as np
import os
import re
import sys
from PIL import Image, ImageOps
from . import rectify as _rectify
from .capture import _cluster_cut, _frame_score, _thumb, capture_pages, rotate_image


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
                        no_warp: bool, enhance: bool, deskew: bool,
                        dewarp: bool = False,
                        workdir: str | None = None) -> np.ndarray:
    """Put one image through the same treatment a captured page receives.

    With dewarp set (and a workdir for the witness), a patch photograph
    climbs the same geometry ladder as the pages around it — a patched
    page used to stop at the rigid deskew and sit less corrected than its
    neighbours.
    """
    if rotate:
        img = rotate_image(img, rotate)
    if no_warp and not enhance and not deskew:
        return img
    parts, _info = capture_pages(img, rectify=not no_warp, enhance=enhance,
                                 deskew=deskew, split_spreads=False,
                                 workdir=workdir if dewarp else None,
                                 min_area_ratio=min_area_ratio)
    return parts[0]


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
        img = _prepare_page_image(frame, min_area_ratio, rotate, no_warp,
                                  enhance, deskew, dewarp=deskew,
                                  workdir=pages_dir)
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
    burst_threshold: float | None = None, dewarp: bool = True,
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
    n_rectified = n_split = n_no_model = 0
    refused: list[tuple[str, int, int]] = []
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
        stem = os.path.splitext(os.path.basename(best_path))[0]
        parts, info = capture_pages(img, rectify=not no_warp and dewarp,
                                    enhance=enhance, deskew=deskew,
                                    split_spreads=split_spreads, workdir=pages_dir,
                                    min_area_ratio=min_area_ratio)
        n_rectified += info["rectified"]
        n_split += info["split"]
        if info["kept_photo"] == "no rectifier":
            n_no_model += 1
        elif info["kept_photo"]:
            refused.append((stem, *info["kept_photo"]))
        for part_idx, part in enumerate(parts):
            out = os.path.join(pages_dir, f"page_{len(page_paths):04d}.jpg")
            Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB)).save(
                out, "JPEG", quality=92, dpi=(300, 300)
            )
            page_paths.append(out)
            page_ids.append(page_id(stem, part_idx))

    print(f"  {len(page_paths)} pages kept; {n_rectified} photograph(s) "
          f"flattened by the rectifier")
    if n_split:
        print(f"  {n_split} two-page spreads split at the gutter")
    _report_kept_photos(n_no_model, refused)
    return page_paths, page_ids


def _report_kept_photos(n_no_model: int, refused: list) -> None:
    """Say which shots were kept as photographed, and why."""
    if n_no_model:
        print(f"  [!] {n_no_model} photograph(s) kept as shot — the page "
              f"rectifier is not available ({_rectify.unavailable_reason()}); "
              f"pip install 'codicology[geometry]' flattens them")
    for stem, before, after in refused:
        print(f"  [!] {stem}: the rectified sheet read {after} words where the "
              f"photograph read {before} — kept the photograph; look at this page")


def dewarp_scan_pages(page_paths: list[str], pages_dir: str) -> None:
    """Opt-in geometry for a third-party scan loaded with --pages-from.

    The round-trip case must stay untouched — pages this script wrote have
    been through the ladder once, and warping them again would drift — and
    a shelf of already-square library scans has caches keyed to its exact
    pixels. So nothing here runs by default. For the scan that genuinely
    leans or curls, this applies the capture ladder minus the crop: no
    outline detection (a scan fills its frame), then deskew, the line-model
    dewarp, and the cubic sheet under the witness. Every changed page
    invalidates its OCR cache entry, which is the point of opting in.
    """
    n_rectified = n_no_model = 0
    refused: list[tuple[str, int, int]] = []
    for path in page_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        parts, info = capture_pages(img, rectify=True, enhance=False, deskew=True,
                                    split_spreads=False, workdir=pages_dir)
        n_rectified += info["rectified"]
        if info["kept_photo"] == "no rectifier":
            n_no_model += 1
        elif info["kept_photo"]:
            refused.append((os.path.basename(path), *info["kept_photo"]))
        out = parts[0]
        if out is not img:
            if path.lower().endswith((".jpg", ".jpeg")):
                cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(path, out)
    print(f"  scan geometry: {n_rectified} page(s) flattened by the rectifier "
          f"and levelled")
    _report_kept_photos(n_no_model, refused)


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
