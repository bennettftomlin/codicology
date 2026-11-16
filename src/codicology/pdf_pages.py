"""A PDF's pages as the images the recogniser reads: a scan's own picture
taken out as it is stored where it is the page, anything else rendered,
each with its sidecars (its text layer, whether it is born-digital, where
it came from); rendered across worker processes for a long book; and a
page that a book scanned two to a sheet prints sideways, noted for turning.
"""
from __future__ import annotations

import glob
import hashlib
import json
import numpy as np
import os
import re
import sys
from PIL import Image
from .cache import _as_formerly_written, _pixel_key, page_key
from .capture import split_spread
from .pdf_frame import native_orientation


def _union_fraction(boxes: list, pw: float, ph: float) -> float:
    """How much of the page the given rectangles cover between them.

    Overlap counts once, so two halves of a tiled scan come to the whole
    page rather than to twice it. Coordinate compression rather than a
    sweep line: a page carries a handful of images, so the grid is tiny and
    the code can be read for correctness at a glance.
    """
    area = max(1e-6, pw * ph)
    if not boxes:
        return 0.0
    xs = sorted({v for b in boxes for v in (b[0], b[2])})
    ys = sorted({v for b in boxes for v in (b[1], b[3])})
    covered = 0.0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cx, cy = (xs[i] + xs[i + 1]) / 2, (ys[j] + ys[j + 1]) / 2
            if any(b[0] <= cx <= b[2] and b[1] <= cy <= b[3] for b in boxes):
                covered += (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
    # An image may be placed hanging over the page edge; the page cannot be
    # more than covered.
    return min(1.0, covered / area)


PAGE_PICTURE_MIN = 0.5      # of its page, for a picture to be the page


def _is_picture_of_itself(page) -> bool:
    """
    Whether everything this page prints sits inside a picture on it.

    A scanned page is a photograph of a page: its words are pixels, and any
    text the PDF carries is a recogniser's layer laid over them. A typeset
    page that holds a plate prints something beside it in type — a caption,
    a folio, a running head — and that is the difference this looks for:
    a picture over at least half the page, and no text or line anywhere
    off the pictures.
    """
    import pypdfium2 as pdfium
    try:
        pw, ph = page.get_size()
        pictures, ink = [], []
        for o in page.get_objects():
            if not (isinstance(o, pdfium.PdfImage) or o.type in (1, 2)):
                continue                    # 1 is text, 2 a path
            b = o.get_bounds()
            b = (min(b[0], b[2]), min(b[1], b[3]),
                 max(b[0], b[2]), max(b[1], b[3]))
            (pictures if isinstance(o, pdfium.PdfImage) else ink).append(b)
    except Exception:
        return False
    if not pictures or max((b[2] - b[0]) * (b[3] - b[1]) for b in pictures) \
            < PAGE_PICTURE_MIN * pw * ph:
        return False
    pad = 2
    for b in ink:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        if not any(m[0] - pad <= cx <= m[2] + pad
                   and m[1] - pad <= cy <= m[3] + pad for m in pictures):
            return False
    return True


def _scanned_pages(doc) -> set:
    """
    The pages of a scanned book that are pictures of themselves.

    One such page proves nothing — a typeset book's cover is a picture and
    nothing else too. A book of them is a scan, and the shelf divides
    cleanly on it: of 151 books, 115 have such a page on under 2% of their
    pages or not at all, one picture book on 9%, and the 35 scans on 82% or
    more. So the pages count only where they are most of the book.

    It matters where the scan does not fill its page. Google Books crops
    each page image to the type area and sets it inside the margins, at
    0.72 of the page; another scan sits at 0.7 to 0.8 beside a library's
    stamp. Neither is "a picture of the whole page", so both were handled
    as typeset pages holding an illustration: the history of Lloyd's
    shipped with the scan of every one of its 438 pages inserted above that
    page's text as artwork the layout had missed, 130MB of it, and
    Confusion de Confusiones had its scanner's text layer trusted as the
    publisher's own.
    """
    found = {i for i, page in enumerate(doc) if _is_picture_of_itself(page)}
    return found if 2 * len(found) > len(doc) else set()


def _scans_are_spreads(doc, scanned: set) -> bool:
    """Whether a scanned book holds two pages to a sheet: most of its scans
    are wider than tall (see TURN_SPREAD_ASPECT)."""
    wide = 0
    for i in scanned:
        try:
            w, h = doc[i].get_size()
        except Exception:
            continue
        wide += w / max(1e-6, h) >= TURN_SPREAD_ASPECT
    return bool(scanned) and 2 * wide > len(scanned)


# Pages are rendered in worker processes once a book is long enough to repay
# starting them. A warm rebuild of Kill or Get Killed (440 pages, every page
# already read) spent 126 of its 202 seconds here, one page at a time, on a
# machine with ten cores: 107 of them in the PNG encoder. Each worker opens
# the PDF for itself, because pdfium is not safe to share, and writes the
# same files the one-at-a-time loop writes, byte for byte. On five books
# (1,862 pages) loading went from 545 seconds to 81 with every file
# identical; naming pages for their pixels (see page_key) then took the
# slow encoder out of what was left.
#
# The workers are plain subprocesses handed their pages on stdin, not a
# multiprocessing pool. A pool's workers begin by importing the caller's
# main module again, and a script that calls this without a __main__ guard
# would be run a second time in each of them.
PDF_POOL_MIN_PAGES = 24


PDF_POOL_WORKERS = 8


PDF_WORKERS_ENV = "CODICOLOGY_PDF_WORKERS"


_PDF_WORKER = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); "
    "from {module} import _load_pdf_share; "
    "json.dump(_load_pdf_share(json.load(sys.stdin)), sys.stdout)")


# A scanned page printed sideways is read sideways. The ST 31-91B medical
# handbook is scanned two pages to a sheet, and on two sheets a page is
# drawings captioned in type that runs down the page: how to wrap dry
# goods, fold gauze, make cotton balls. The recogniser read those captions
# in a different order every time it was asked, dropping a heading one
# time and a measurement the next, and no reading could be kept. Upright,
# the page reads as a page.
#
# Which way is up is tesseract's orientation check (`--psm 0`); tesseract
# is already what the adjudicator reads with, and without it nothing is
# turned. It says how far to turn the picture clockwise and how sure it
# is. Only a quarter turn either way is taken, and only where it is sure:
# TURN_MIN_CONFIDENCE.
#
# What is turned is the page's body, not the page. Its running head and
# its page number are printed upright above and below the turned type, and
# the first reading of the page knows them for what they are; turned with
# the body they were read as neither. The medical handbook's appendices are
# known by the letters of their page numbers (F-4, G-1), and with three of
# those lying on their side its chapters could no longer be told, its
# contents came out as every heading in the book, and two page numbers sat
# in the text as words. So the page is read first as it lies, the blocks
# that reading calls furniture are cut out and set back upright above and
# below, and only what is left is turned (turn_sideways_bodies).
#
# A sheet of two pages is turned a page at a time, the left page set above
# the right: turned whole, the right-hand page would come out on top and
# be read first. A single page set on its side is left as it lies. The
# check was put to all 10,616 scanned pages on the shelf, and outside the
# medical handbook it found four: a register and a table in the history of
# Lloyd's, a chart in KGB, a plate in Kill or Get Killed. Each was already
# read well as it lay. Nothing else on the shelf was called sideways with
# any confidence at all.
TURN_MIN_CONFIDENCE = 5.0


TURN_SPREAD_ASPECT = 1.15


TURN_CHECK_LONGEST = 2400       # pixels: the check needs no more than this


TURN_FURNITURE_PAD = 0.006      # of the sheet: room kept round a cut-out head


_TURNS = {90: Image.Transpose.ROTATE_270, 270: Image.Transpose.ROTATE_90}


def _which_way_up(im: "Image.Image") -> "tuple[int, float] | None":
    """How far tesseract's orientation check says to turn a picture
    clockwise to read it (0, 90, 180 or 270) and how sure it is. None where
    it cannot say: not installed, no type to judge by, a failed call."""
    import shutil
    import subprocess
    import tempfile
    if shutil.which("tesseract") is None:
        return None
    g = im.convert("L")
    scale = TURN_CHECK_LONGEST / max(g.size)
    if scale < 1:
        g = g.resize((max(1, int(g.width * scale)), max(1, int(g.height * scale))),
                     Image.LANCZOS)
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, "page.png")
        g.save(f)
        try:
            r = subprocess.run(["tesseract", f, "-", "--psm", "0"],
                               capture_output=True, text=True, timeout=120,
                               env=dict(os.environ, OMP_THREAD_LIMIT="1"))
        except (OSError, subprocess.SubprocessError):
            return None
    said = r.stdout + r.stderr
    turn = re.search(r"Rotate: (\d+)", said)
    sure = re.search(r"Orientation confidence: ([\d.]+)", said)
    if not (turn and sure):
        return None
    return int(turn.group(1)), float(sure.group(1))


def _halves(im: "Image.Image") -> "list[tuple[Image.Image, int]]":
    """The two pages of a sheet, each with where it starts across the sheet;
    one part, the whole sheet, where no gutter divides it."""
    parts = split_spread(np.asarray(im.convert("RGB")),
                         min_aspect=TURN_SPREAD_ASPECT)
    if len(parts) != 2:
        return [(im.convert("RGB"), 0)]
    return [(Image.fromarray(parts[0]), 0),
            (Image.fromarray(parts[1]), im.width - parts[1].shape[1])]


def _sideways(im: "Image.Image", spread: bool, said: dict) -> "list[int] | None":
    """How far to turn each page of a sheet, left and right (0 where it reads
    as it lies); None where nothing is to be turned, it cannot be judged, or
    it is not a sheet of two pages. `said` holds what the check has already
    said of this picture, by half ("left", "right"), and gains what it says
    now."""
    def turn_of(part, name):
        if name not in said:
            got = _which_way_up(part)
            if got is None:
                return 0            # cannot say, and is asked again next time
            said[name] = list(got)
        turn, sure = said[name]
        return turn if turn in _TURNS and sure >= TURN_MIN_CONFIDENCE else 0

    if not spread or im.width / max(1, im.height) < TURN_SPREAD_ASPECT:
        return None                     # a single page is left as it lies
    halves = _halves(im)
    if len(halves) != 2:
        return None
    turns = [turn_of(h, n) for (h, _), n in zip(halves, ("left", "right"))]
    return turns if any(turns) else None


def _body_turned(im: "Image.Image", turns: "list[int]",
                 furniture: "list[tuple]") -> "Image.Image":
    """A sheet of two pages with the body of each sideways page turned to
    read, its furniture (boxes, as fractions of the sheet) set back upright
    above and below it as it was printed, and the left page above the
    right."""
    pad = TURN_FURNITURE_PAD
    out = []
    for (half, x0), turn in zip(_halves(im), turns):
        if not turn:
            out.append(half)
            continue
        w, h = half.size
        paper = tuple(int(v) for v in np.median(
            np.asarray(half).reshape(-1, 3), axis=0))
        body = half.copy()
        heads, feet = [], []
        for bx in furniture:
            l = bx[0] * im.width - x0
            r = bx[2] * im.width - x0
            if not (0 <= (l + r) / 2 < w):
                continue                # the other page's
            box = (max(0, int(l - pad * im.width)), max(0, int((bx[1] - pad) * im.height)),
                   min(w, int(r + pad * im.width)), min(h, int((bx[3] + pad) * im.height)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            (heads if (box[1] + box[3]) / 2 < h / 2 else feet).append(
                (box[0], half.crop(box)))
            body.paste(paper, box)
        body = body.transpose(_TURNS[turn])

        def band(cuts):
            if not cuts:
                return None
            strip = Image.new("RGB", (body.width, max(c.height for _, c in cuts)), paper)
            for x, c in cuts:
                strip.paste(c, (min(max(0, int(x * body.width / w)),
                                    max(0, body.width - c.width)), 0))
            return strip
        parts = [p for p in (band(heads), body, band(feet)) if p is not None]
        page = Image.new("RGB", (body.width, sum(p.height for p in parts)), paper)
        y = 0
        for p in parts:
            page.paste(p, (0, y))
            y += p.height
        out.append(page)
    sheet = Image.new("RGB", (max(p.width for p in out),
                              sum(p.height for p in out)), "white")
    y = 0
    for p in out:                       # the left page above the right
        sheet.paste(p, (0, y))
        y += p.height
    return sheet


# What the check said of each scanned page, kept beside the book's reading
# cache under the page's name: the check costs a second or more a page,
# every page of a scan is asked, and the answer for a picture never
# changes. What is kept is what tesseract said, not what was done with it,
# so a change of TURN_MIN_CONFIDENCE needs no page checked again.
TURNS_SUFFIX = ".turns.json"


def load_turns(path: "str | None") -> dict:
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return dict(json.load(fh).get("pages") or {})
    except (OSError, ValueError, AttributeError):
        return {}


def save_turns(path: "str | None", pages: dict) -> None:
    if not path:
        return
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({"check": "tesseract --psm 0", "pages": pages}, fh,
                      sort_keys=True)
        os.replace(tmp, path)
    except OSError:
        pass


def _note_sideways(out: str, spread: bool,
                   remembered: "dict | None") -> "dict | None":
    """Whether a scanned page's sheet has a page printed sideways: how far to
    turn each, or None. What the check said is left beside the image
    (.turn.json) for the caller to remember. Nothing is turned here: that
    waits for the page's first reading (turn_sideways_bodies)."""
    try:
        name = page_key(out)
        said = dict((remembered or {}).get(name) or {})
        with Image.open(out) as im:
            im.load()
            turns = _sideways(im, spread, said)
    except Exception:
        return None
    if said:
        with open(out + ".turn.json", "w", encoding="utf-8") as fh:
            json.dump({"page": name, "said": said}, fh)
    return {"degrees": turns, "spread": True} if turns else None


def _load_pdf_page(doc, i: int, pdf_path: str, pages_dir: str,
                   is_scan: bool, former_keys: bool = False,
                   spreads: bool = False,
                   turns: "dict | None" = None) -> str:
    """Write page `i` of an open PDF as the image the recogniser will read,
    with its sidecars, and return the image's path. `spreads`: the book's
    scans hold two pages to a sheet (see TURN_SPREAD_ASPECT); `turns`, what
    the orientation check said of pages before (see TURNS_SUFFIX)."""
    import pypdfium2 as pdfium
    page = doc[i]
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
            # The image's size as the PDF records it. get_bitmap() gave the
            # same two numbers by decoding the whole image to count them:
            # 8.0 ms a page against 0.005.
            bm_w, bm_h = images[0].get_px_size()
            pw, ph = page.get_size()
            covers = (bm_w * bm_h) >= 0.5 * (pw / 72 * 150) * (ph / 72 * 150)
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
            img_aspect = bm_w / max(1e-6, bm_h)
            covers = covers and (
                0.8 <= img_aspect / max(1e-6, page_aspect) <= 1.25)
        except Exception:
            covers = False
        # A page placed by a flipping matrix must not be taken as its
        # own bytes: the shortcut keeps the original encoding, which is
        # its whole point, and re-encoding a turned copy would give
        # that up. Render such a page instead — correct beats cheap.
        if covers and native_orientation(images[0]) == []:
            try:
                images[0].extract(stem)  # keeps the original encoding
                out = next(iter(sorted(glob.glob(stem + ".*"))), None)
                extracted = out is not None
            except Exception:
                out = None
            # A JPEG or JPEG 2000 comes out as the PDF's own bytes, and
            # those are the page's name. Anything else pdfium decodes and
            # Pillow encodes again, so it is named for its pixels like a
            # page rendered here (see page_key).
            if out and not out.lower().endswith((".jpg", ".jpeg", ".jp2")):
                try:
                    with Image.open(out) as im:
                        key = _pixel_key(im)
                    with open(out + ".key", "w", encoding="ascii") as fh:
                        fh.write(key)
                except Exception:
                    pass
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
        im = page.render(scale=300 / 72,
                         draw_annots=False).to_pil().convert("RGB")
        # Named before it is written, from the pixels, so the encoder is
        # free to be quick: level 1 where the name used to demand the
        # default, a sixth of the time for a file a third larger, in a
        # directory this run deletes.
        with open(out + ".key", "w", encoding="ascii") as fh:
            fh.write(_pixel_key(im))
        if former_keys:
            with open(out + ".key1", "w", encoding="ascii") as fh:
                fh.write(hashlib.sha256(_as_formerly_written(im)).hexdigest())
        im.save(out, "PNG", dpi=(300, 300), compress_level=1)
    # A scan printed sideways is set upright before anything reads it (see
    # TURN_MIN_CONFIDENCE). The turned picture is still the page and
    # nothing else, as a facsimile is, but it no longer stands where the
    # PDF draws it: nothing may be asked of the PDF by position on it.
    # A page printed sideways is noted here and turned once it has been
    # read as it lies (see turn_sideways_bodies).
    sideways = _note_sideways(out, spreads, turns) if (is_scan and spreads) else None
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
        # The union of the images, not the largest of them. "Is there a
        # picture of the whole page behind this text" is a question about
        # coverage, and a scan does not have to answer it with one image:
        # stored as two tiles, or as MRC layers, its largest piece can sit
        # well under the threshold while the page is completely covered.
        # One book's cover does exactly that — largest 0.71, union 1.0 —
        # and was being handed to the reconciler as a publisher's
        # typesetting. The error is not symmetric, which is why this
        # errs upward: calling a born-digital page a scan costs a
        # correction we could have made, while calling a scan
        # born-digital lets somebody else's OCR rewrite the author.
        boxes = []
        for o in images:
            try:
                b = o.get_bounds()
                boxes.append((min(b[0], b[2]), min(b[1], b[3]),
                              max(b[0], b[2]), max(b[1], b[3])))
            except Exception:
                pass
        behind = _union_fraction(boxes, pw_pt, ph_pt)
        # …nor does the picture have to reach the page's edges. A scan
        # set inside its margins covers seven tenths of the page and is
        # no more typeset for it; what gives it away is the book.
        if not extracted and behind < 0.9 and not is_scan:
            open(out + ".native", "w").close()
    # Where this page came from. A figure cut out of the page raster is a
    # crop of a crop — the raster already resampled the artwork, at 3.12x
    # on one manual's line drawings, and JPEG rang every hard edge it
    # made. With the source in hand a figure can be taken from the PDF
    # itself instead — which needs to know when the largest picture
    # on the page is not a figure at all but the scan of the page.
    try:
        with open(out + ".source.json", "w") as fh:
            json.dump({"pdf": os.path.abspath(pdf_path), "page": i,
                       "facsimile": bool(extracted),
                       "scan": is_scan,
                       **({"sideways": sideways} if sideways else {})}, fh)
    except OSError:
        pass

    return out


def _load_pdf_share(job: dict) -> dict:
    """A worker's share of a PDF: the pages it was dealt, by index. Returns
    each page's file name."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(job["pdf"])
    scanned = set(job["scanned"])
    try:
        return {str(i): os.path.basename(
                    _load_pdf_page(doc, i, job["pdf"], job["dir"], i in scanned,
                                   job.get("former_keys", False),
                                   job.get("spreads", False),
                                   job.get("turns")))
                for i in job["pages"]}
    finally:
        try:
            doc.close()
        except Exception:
            pass


def _load_pdf_pages_in_workers(pdf_path: str, pages_dir: str, n_pages: int,
                               scanned: set, workers: int,
                               former_keys: bool = False,
                               spreads: bool = False,
                               turns: "dict | None" = None) -> "list[str] | None":
    """Render a PDF's pages across `workers` processes. None if any of them
    failed, and the caller renders the book itself."""
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    argv = [sys.executable, "-c", _PDF_WORKER.format(module=__name__),
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    # Dealt round, not cut into runs: a book's plates sit together, and a
    # run of them would leave one worker with all the slow pages.
    jobs = [json.dumps({"pdf": pdf_path, "dir": pages_dir,
                        "pages": list(range(k, n_pages, workers)),
                        "scanned": sorted(scanned),
                        "former_keys": former_keys, "spreads": spreads,
                        "turns": turns or {}})
            for k in range(workers)]

    def run(job: str) -> "dict | None":
        try:
            done = subprocess.run(argv, input=job, capture_output=True,
                                  text=True)
            return json.loads(done.stdout) if done.returncode == 0 else None
        except (OSError, ValueError):
            return None

    with ThreadPoolExecutor(max_workers=workers) as waiting:
        shares = list(waiting.map(run, jobs))
    names: dict = {}
    for share in shares:
        if share is None:
            return None
        names.update(share)
    if len(names) != n_pages:
        return None
    return [os.path.join(pages_dir, names[str(i)]) for i in range(n_pages)]


def _pdf_pool_workers(n_pages: int) -> int:
    """How many processes to render with: one means stay in this one."""
    if n_pages < PDF_POOL_MIN_PAGES:
        return 1
    # Whoever runs several builds at once says how many each may start:
    # four builds of the shelf harness, eight workers apiece, were fifty
    # processes on ten cores and a load average over two hundred.
    most = PDF_POOL_WORKERS
    said = os.environ.get(PDF_WORKERS_ENV, "")
    if said.isdigit() and int(said) >= 1:
        most = min(most, int(said))
    return max(1, min(most, os.cpu_count() or 1))


def load_pages_from_pdf(pdf_path: str, pages_dir: str,
                        former_keys: bool = False,
                        turns_path: "str | None" = None) -> list[str]:
    """
    Take the page images straight back out of a PDF this script wrote earlier.

    Re-reading the video costs the better part of an hour before OCR can even
    start, which makes iterating on OCR settings painful for no reason: the pages
    were already chosen and cropped. Pulling the embedded JPEG out rather than
    re-rendering the page hands OCR the original image instead of a re-encode.

    `former_keys` also leaves, beside each rendered page, the name its
    reading had in a cache written before pages were named for their pixels
    (see page_key). It costs the encoding that naming used to cost, so it is
    asked for only when such a cache is about to be read.

    A scanned page printed sideways is turned upright (see
    TURN_MIN_CONFIDENCE); `turns_path` is where what the check said of each
    page is remembered (see TURNS_SUFFIX).
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        sys.exit("--pages-from needs pypdfium2: pip install pypdfium2")

    doc = pdfium.PdfDocument(pdf_path)
    scanned = _scanned_pages(doc)
    n_pages = len(doc)
    spreads = _scans_are_spreads(doc, scanned)
    turns = load_turns(turns_path) if spreads else {}
    page_paths: list[str] = []
    workers = _pdf_pool_workers(n_pages)
    if workers > 1:
        page_paths = _load_pdf_pages_in_workers(pdf_path, pages_dir, n_pages,
                                                scanned, workers, former_keys,
                                                spreads, turns)
        if page_paths is None:
            print("  [~] a page-rendering worker failed; rendering the pages "
                  "one at a time instead")
            workers = 1
    if workers == 1:
        page_paths = [_load_pdf_page(doc, i, pdf_path, pages_dir, i in scanned,
                                     former_keys, spreads, turns)
                      for i in range(n_pages)]

    if not page_paths:
        sys.exit(f"No pages found in {pdf_path}")
    if spreads:
        learned, upright = dict(turns), []
        for k, p in enumerate(page_paths):
            try:
                with open(p + ".turn.json", encoding="utf-8") as fh:
                    said = json.load(fh)
                learned[said["page"]] = said["said"]
            except (OSError, ValueError, KeyError):
                pass
            try:
                with open(p + ".source.json", encoding="utf-8") as fh:
                    how = json.load(fh).get("sideways")
                if how:
                    upright.append(f"{k + 1} ({'/'.join(map(str, how['degrees']))})")
            except (OSError, ValueError):
                pass
        if learned != turns:
            save_turns(turns_path, learned)
        if upright:
            print(f"  printed sideways, to be turned once read as they lie: "
                  f"page(s) {', '.join(upright)}")
    print(f"  {len(page_paths)} pages loaded from {pdf_path}")
    return page_paths
