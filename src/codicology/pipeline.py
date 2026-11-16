"""
codicology convert — the pipeline.

Turn a book into a clean PDF and an OCR'd EPUB, from a video, a stack of
photographs, or a PDF someone else made: a library scan, a born-digital
manual, or a facsimile this pipeline wrote earlier.

Dependencies (installed with the package):
    opencv-python-headless numpy Pillow img2pdf pypdfium2 pymupdf lxml ebooklib
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

    There were four other names behind --ocr once (easyocr, paddleocr,
    tesseract, gcv). None was ever run against a real book or tested, and
    each returned flat text and nothing else, so a book built on one would
    have arrived with no figures, no headings, no contents and no note
    links, with nothing to say so. They are gone. OCRBackend is still the
    seam a second engine would sit behind, and it would have to return
    layout to be one.

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
import functools
import glob
import gzip
import zlib
import hashlib
import json
import difflib
import html
import io
import os
import re
import shutil
import statistics
from collections import Counter
import sys
import tempfile
import time
import unicodedata
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from . import rectify as _rectify

from . import progress
from .markup import _strip_tags
from . import switches
from . import witness


from .capture import (  # noqa: F401  (moved; kept importable from here)
    _sharpness, _frame_score, blur_score, _order_corners, DETECT_WIDTH,
    GUTTER_MIN_DEPTH, PAGE_LEVEL_RATIO, GUTTER_STRIPS, SKIN_LO, SKIN_HI,
    EDGE_BAND, HAND_PENALTY, CANDIDATE_POOL, QUIET_GATE, QUIET_POOL,
    SETTLE_MARGIN, DESKEW_WIDTH, DESKEW_MAX_DEG, DESKEW_STEP_DEG,
    DESKEW_MIN_DEG, DESKEW_MIN_INK, DESKEW_STROKE, DESKEW_PEAK_RATIO,
    _quad_from_contour, _skin_mask, _corners_from_edges, detect_page,
    detect_page_quads, rotate_image, split_spread, GUTTER_PATH_MIN_LEAN,
    _SEAM_UNAVAILABLE_SAID, GUTTER_SEAM_REACH, GUTTER_SEAM_COST_LETTER,
    GUTTER_SEAM_COST_PLATE, GUTTER_SEAM_COST_SHADOW, GUTTER_SEAM_MARGIN,
    GUTTER_SEAM_COST_NEAR, GUTTER_SEAM_LINE_HEIGHT,
    GUTTER_SEAM_MAX_CONTENT_HEIGHT, _gutter_path, _gutter_x,
    _warp_transform, warp_page, _thumb, scan_motion, _otsu_split,
    auto_motion_threshold, MIN_CLUSTER_RATIO, _cluster_cut,
    auto_blur_threshold, auto_hold_threshold, select_page_frames,
    _pick_quietest, _pick_detectable, text_angle, deskew_page,
    enhance_page, WITNESS_MIN_WORDS, capture_pages,
)



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


from .cache import (  # noqa: F401  (moved; kept importable from here)
    CACHE_CHECKPOINT_PAGES, CACHE_VERSION, PAGE_KEYING, _PAGE_KEY,
    _pixel_key, _key_beside, _bytes_key, page_key, _as_formerly_written,
    former_page_key, OCRCache,
)




from .model import PageItem  # noqa: F401  (moved; kept importable from here)


from .backends import (  # noqa: F401  (moved; kept importable from here)
    OCRBackend, FIGURE_LABELS, BOX_MIN_AREA, BOX_MAX_AREA, BOX_MAX_INK,
    find_ruled_boxes, CAPTION_LABELS, FURNITURE_LABELS, _crop_block,
    _BLOCK_TAGS, _FOLD_INTO, _NO_SEAM_AFTER, _needs_seam_space,
    _fold_loose_inline, _STANDS_ALONE, _wrap_loose_inline, _to_xhtml,
    llamacpp_safe_schemas, set_pages_in_flight, SuryaBackend,
    OCR_BACKENDS, load_backend, reflow_paragraphs,
)



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
    bits = tuple(p > avg for p in px)
    # A flat image has no pattern to fingerprint: every pixel sits at the
    # mean and the signature collapses to all-False — identical for a
    # solid divider, a blank plate, and a grey placeholder alike, which
    # once grouped three unrelated images as "the same recurring picture"
    # and deleted them all. Featureless means unidentifiable: decline to
    # sign rather than sign everything the same (A2, 2026-08-18 ledger).
    if not any(bits):
        return None
    return (len(px), bits)


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


# A ruled box is looked for on the page so that a drawing the layout never
# made a block of is not lost (find_ruled_boxes). It is told which regions
# are already figures, and nothing else, so wherever the layout DID make a
# block of a ruled region the box was cut out as a picture all the same.
# Across the shelf's caches (38,423 pages, 6,847 pictures) that is 417
# pictures. 360 sit over a block that has its text: 332 tables, most of
# them in three economics textbooks, which ship as a table and then again as
# a picture of the table; and 28 boxed notes, warnings and contents lists.
# The other 57 sit over a block that is only a picture waiting to be cut
# (a chart or a chemical structure the recogniser returned as an empty
# <img>), and recover_placeholder_figures then cut the same region a second
# time: Where There Is No Dentist printed its diagnostic charts twice.
#
# Only a picture found by its ruled border is ever dropped, and only where
# another block covers the same box. Sixty of the 360 were looked at, the
# tables with the least text for their size and all 28 of the others: every
# one is the block's own content.
#
# The box that is dropped is still a region the page has accounted for, and
# the passes that look for artwork nobody boxed are told so. They ask
# whether a figure already covers what they have found, and with the
# picture gone the answer changed on five pages of the sixteen books first
# rebuilt under this rule. On three the same picture came straight back,
# cut from the PDF instead. On two, in Principles of Microeconomics, a
# table's rules, the banner over them and a tinted line between made one
# drawing, which went in as a picture and took the banner's heading, and
# once the line's sentence, with it as its labels. Told of the box, those
# passes do exactly what they did while the picture stood there.
DOUBLED_BOX_OVERLAP = 0.8


from .opinions import (  # noqa: F401  (moved; kept importable from here)
    EQUATION_SAME_BLOCK, EQUATION_NEARER, _flat_tex, _tex_words,
    _flat_printed, _undotted, _borne_out, _likeness, _FIGURES_ONLY,
    _differ_in_figures_alone, _differences, _stands, majority_reading,
    _LINES_OF, _tex_rows, _stands_alone, _tidy_tex, mend_equations,
    _FIGURE, _figures, _typeset_text_in, _share_held, SecondOpinion,
    ThirdOpinion, TABLE_QUESTIONED, TABLE_TAKEN, TABLE_BETTER_BY,
    TABLE_REACH, TABLE_UNPRINTED, mend_tables,
)




from .tex import (  # noqa: F401  (moved; kept importable from here)
    _TEX_DOTS, _tex_name, _brace_group, _unfrac, GREEK, MATH_SIGNS,
    _DISPLAY_MATH,
)


# One block with the box's own outline is the plain case. A box may also
# hold two blocks (a WARNING and its sentence), or be one cell of a table
# that is a single block, and then no block has its outline: FM 4-25 printed
# each boxed warning as a picture, a one-row table and a paragraph, and the
# cells of its tables again as pictures after the table. Of the 675
# ruled-border pictures in the shelf's caches that no single block matches,
# 414 sit over text this way.
#
# What settles those is the PDF, where the page is typeset: whether anything
# is placed or drawn in the box besides type and rules, and whether the
# words the box prints are words the page has been read as saying. A box of
# nothing but type the page already has is dropped. A box with a picture or
# a drawing in it stays, and so does a box whose words were not read: its
# picture is the only copy of them. Of 408 such boxes on typeset pages the
# PDF calls 383 type and nothing else; fifty-six of those were looked at
# and every one is a warning, a sidebar or a table cell. Of the 25 it holds
# back, five are drawings and the rest are text set on a tint, which stays
# doubled rather than be guessed at. A scan has no such record and its boxes
# are left as they were: TiHKAL's chemical structures are boxes inside text.
BOX_WORDS_READ = 0.9     # of the words the box prints, on the page as read


def _box_holds_only_type(page, box) -> bool:
    """Whether the PDF puts nothing inside this box but type and rules."""
    frame = _page_frame(page)
    inner = (box[0] + 0.008, box[1] + 0.006, box[2] - 0.008, box[3] - 0.006)
    span = (inner[2] - inner[0]) * (inner[3] - inner[1])
    if span <= 0:
        return False

    def inside(nb):
        return max(0.0, min(nb[2], inner[2]) - max(nb[0], inner[0])) \
            * max(0.0, min(nb[3], inner[3]) - max(nb[1], inner[1]))

    whole = (box[2] - box[0]) * (box[3] - box[1])
    for obj in page.get_objects(max_depth=4):
        if obj.type not in (2, 3, 4):       # a path, an image, a shading
            continue
        b = obj.get_bounds()
        nb = _pdf_box_to_normalized(b, frame)
        if nb[2] < inner[0] or nb[0] > inner[2] \
                or nb[3] < inner[1] or nb[1] > inner[3]:
            continue
        if obj.type == 3:
            if inside(nb) > 0.02 * span:
                return False
            continue
        if obj.type == 4:
            return False
        w, h = b[2] - b[0], b[3] - b[1]
        if min(w, h) < 2.5 or max(w, h) < 9:
            continue                        # a rule; a bullet or a tick
        area = (nb[2] - nb[0]) * (nb[3] - nb[1])
        if area > 0 and max(0.0, min(nb[2], box[2]) - max(nb[0], box[0])) \
                * max(0.0, min(nb[3], box[3]) - max(nb[1], box[1])) \
                >= 0.85 * whole:
            continue                        # the box's own border
        if inside(nb) >= 0.6 * area or inside(nb) > 0.05 * span:
            return False
    return True


_BOX_WORD = re.compile(r"[a-z0-9]+")


def _typeset_boxes_already_read(items: list, ruled: list, page_path: str
                                ) -> set:
    """ids of the ruled-border pictures that hold only type the page has."""
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile") or src.get("scan"):
            return set()
        import pypdfium2 as pdfium
        page = pdfium.PdfDocument(src["pdf"])[src["page"]]
        frame = _page_frame(page)
        textpage = page.get_textpage()
    except Exception:
        return set()
    said = Counter(_BOX_WORD.findall(" ".join(
        _strip_tags(it.html) for it in items
        if it.html and it.figure is None).lower()))
    gone = set()
    for fig in ruled:
        try:
            printed = Counter(_BOX_WORD.findall(textpage.get_text_bounded(
                **_normalized_to_pdf(fig.box, frame)).lower()))
            total = sum(printed.values())
            if total < 2:
                continue                    # an empty box says nothing
            read = sum(min(n, said.get(w, 0)) for w, n in printed.items())
            if read >= BOX_WORDS_READ * total \
                    and _box_holds_only_type(page, fig.box):
                gone.add(id(fig))
        except Exception:
            continue
    return gone


# A table of figures is where a generative reader does its worst work and
# shows it least: a column slides one cell left, a row is filed under the
# wrong country, a third of the cells are dropped, and what is left still
# looks like a table. Table 12 of Burning Up kept four of its six columns.
# On a typeset page the PDF says what the table prints, so the count can at
# least be checked: of 910 rebuilt tables on such pages with six figures or
# more, 49 hold under nine tenths of the figures printed in their place and
# 10 under six tenths. Below seven tenths, with ten figures or more printed,
# the table is named when the book is built. It is not replaced: the eight
# such tables on the shelf have not each been looked at, and two of them are
# indexes the layout took for tables.
TABLE_MIN_FIGURES = 10
TABLE_FIGURES_HELD = 0.7


def tables_short_of_figures(items: list, page_path: "str | None"
                            ) -> "list[tuple[int, int]]":
    """(figures printed, figures held) for each rebuilt table on a typeset
    page that holds too few of the figures the PDF prints in its box."""
    tables = [it for it in items if it.html and it.box and it.figure is None
              and "<table" in it.html and not it.is_furniture]
    text_in = _typeset_text_in(page_path) if tables else None
    if text_in is None:
        return []
    short = []
    for it in tables:
        printed = _figures(text_in(it.box))
        total = sum(printed.values())
        if total < TABLE_MIN_FIGURES:
            continue
        held = _figures(_strip_tags(it.html))
        got = sum(min(n, held.get(w, 0)) for w, n in printed.items())
        if got < TABLE_FIGURES_HELD * total:
            short.append((total, got))
    return short


def drop_doubled_boxes(items: list, page_path: "str | None" = None
                       ) -> "tuple[list, int, list]":
    """Return items without the pictures that only repeat what the page
    already has, how many were dropped, and the boxes they stood in."""
    ruled = [it for it in items
             if it.figure is not None and it.label is None and it.box]
    if not ruled:
        return items, 0, []
    gone = set()
    for fig in ruled:
        for it in items:
            if it is fig or it.figure is not None or not it.box \
                    or not it.html \
                    or _box_iou(fig.box, it.box) < DOUBLED_BOX_OVERLAP:
                continue
            if _strip_tags(PLACEHOLDER_IMG.sub("", it.html)).strip():
                gone.add(id(fig))   # the block says what the box holds
            elif PLACEHOLDER_IMG.search(it.html):
                gone.add(id(it))    # the same picture, about to be cut again
    if page_path:
        gone |= _typeset_boxes_already_read(
            items, [f for f in ruled if id(f) not in gone], page_path)
    return ([it for it in items if id(it) not in gone], len(gone),
            [it.box for it in items if id(it) in gone])


def _lies_inside(box, other, share: float = 0.9) -> bool:
    """Whether nearly all of `box` falls within `other`."""
    ix = max(0.0, min(box[2], other[2]) - max(box[0], other[0]))
    iy = max(0.0, min(box[3], other[3]) - max(box[1], other[1]))
    area = (box[2] - box[0]) * (box[3] - box[1])
    return area > 0 and ix * iy >= share * area


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


def recover_orphan_artwork(items: list, page_path: str,
                           held: "list | tuple" = ()) -> list:
    """
    Artwork the layout never boxed at all.

    A figure is normally found because the recognition labelled that region a
    picture. When it labels the region text instead — which lossless pages
    made it do on label-dense diagrams — the artwork is not merely mis-cropped,
    it is absent: a first-aid manual lost the pressure-point diagram off
    thirteen pages while gaining the prose around it. The PDF still holds the
    picture, and its placement says exactly where it belongs, so anything the
    emitted figures do not already cover can be taken directly. `held` is
    the boxes of pictures left out because a block holds what they showed
    (drop_doubled_boxes): covered, though no figure stands there.
    """
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile"):
            return items            # the page IS the picture; nothing orphaned
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(src["pdf"])
        page = doc[src["page"]]
        frame = _page_frame(page)
    except Exception:
        return items

    covered = [it.box for it in items if it.figure is not None and it.box]
    covered += list(held)
    orphans = []
    try:
        placements = []
        for obj in page.get_objects():
            if not isinstance(obj, pdfium.PdfImage):
                continue
            b = obj.get_bounds()
            placed = _pdf_box_to_normalized(b, frame)
            area = (placed[2] - placed[0]) * (placed[3] - placed[1])
            placements.append((area, placed, obj))
        # On a scanned page the largest picture is the page, at whatever
        # share of the sheet it was set, and it is what the layout was
        # reading — not artwork the layout missed. What can be missed there
        # is a plate stored as its own layer, and a plate lies on its page:
        # a strip out in the PDF's margin is the scanner's bed or a
        # library's stamp, which no reader of the book ever saw.
        own = max(placements, key=lambda t: t[0]) \
            if src.get("scan") and placements else None
        for area, placed, obj in placements:
            if not 0.02 <= area <= 0.9:
                continue            # a logo, or the whole page
            if own is not None:
                if obj is own[2]:
                    continue        # the page itself
                cx, cy = (placed[0] + placed[2]) / 2, (placed[1] + placed[3]) / 2
                if not (own[1][0] <= cx <= own[1][2]
                        and own[1][1] <= cy <= own[1][3]):
                    continue        # off the page the scan shows
            if any(_box_iou(placed, c) >= 0.3 or _lies_inside(placed, c)
                   for c in covered):
                # Already emitted, or part of a picture that was: a drawing
                # set inside a chart is in the chart, and scored against the
                # chart's whole box it overlaps too little to count.
                continue
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


# A table the page prints is ruled, and its rules are paths like any
# drawing's. recover_vector_artwork knew a group of paths for a table where
# the two overlapped by 0.3, which holds while the rules are all the group
# is. A textbook that sets a heading on a banner and a line of prose on a
# tint above each table draws paths that touch the rules, and the group is
# then three to five times the table. Principles of Microeconomics printed
# banner, prose and table as one picture on pp. 205 and 899, and on p. 205
# the heading and the table's only row went with it as the picture's
# labels. Pages 372 and 716 were spared the same only while a picture of
# the table stood there to overlap the group, by 0.3045 and 0.3103.
#
# So a path lying in a table the page prints is that table's, and is not
# looked at; what else is drawn is judged without it. The rule a table is
# drawn in runs a little outside the recogniser's box for the table: by a
# median 0.007 of the page across the 343 ruled tables in the shelf's
# caches, and within this reach on 333. Over the 1,738 cached pages that
# hold a table the pass now differs on pp. 205 and 899 and nowhere else,
# whether the reach is 0.01, 0.02 or 0.04. No page on the shelf sets a
# drawing hard against a table; one that did had it grouped with the
# rules, where it overlapped the table enough to be left out with them,
# and it is found now.
TABLE_RULE_REACH = 0.02


def recover_vector_artwork(items: list, page_path: str,
                           held: "list | tuple" = ()) -> list:
    """
    Drawings made of ink rather than pixels.

    A manual's diagrams are often DRAWN — hundreds of path objects, no
    embedded image anywhere — so when the layout reads the region as text,
    there is nothing for orphan recovery to find and the artwork vanishes
    whole: one urban-operations figure of 1,048 paths left only its caption
    behind. The paths themselves say where the drawing is. Dense clusters of
    them that no emitted figure covers are rendered straight from the page,
    which for vector art is the best possible copy. `held` is as in
    recover_orphan_artwork. The rules of a table the page prints are not
    counted among the paths (TABLE_RULE_REACH).
    """
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile"):
            return items
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(src["pdf"])
        page = doc[src["page"]]
        frame = _page_frame(page)
        pw, ph = frame[2], frame[3]
    except Exception:
        return items

    tabled = [it.box for it in items
              if it.html and "<table" in it.html and it.box]
    boxes = []
    try:
        for obj in page.get_objects():
            if obj.type != 2:                  # paths only
                continue
            b = obj.get_bounds()
            w, h = b[2] - b[0], b[3] - b[1]
            if w > 0.8 * pw and h < 3:
                continue                        # the running head's rule
            at = _pdf_box_to_normalized(b, frame)
            if any(at[0] >= t[0] - TABLE_RULE_REACH
                   and at[1] >= t[1] - TABLE_RULE_REACH
                   and at[2] <= t[2] + TABLE_RULE_REACH
                   and at[3] <= t[3] + TABLE_RULE_REACH for t in tabled):
                continue                        # a rule of a table we print
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
    covered += list(held)
    out = list(items)
    for members in groups.values():
        if len(members) < 15:
            continue
        x0 = min(b[0] for b in members); y0 = min(b[1] for b in members)
        x1 = max(b[2] for b in members); y1 = max(b[3] for b in members)
        placed = _pdf_box_to_normalized((x0, y0, x1, y1), frame)
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


from .pdf_frame import (  # noqa: F401  (moved; kept importable from here)
    _page_frame, _pdf_box_to_normalized, _normalized_to_pdf,
)




def _render_region(page, box, want_w: int) -> "Image.Image | None":
    """The page as drawn, cropped to a fractional box, about want_w wide."""
    try:
        pw, _ = page.get_size()
        scale = max(0.5, min(8.0, want_w / max(1e-6, (box[2] - box[0]) * pw)))
        full = page.render(scale=scale,
                           draw_annots=False).to_pil().convert("RGB")
        x0, y0 = int(box[0] * full.width), int(box[1] * full.height)
        x1, y1 = int(box[2] * full.width), int(box[3] * full.height)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        return full.crop((x0, y0, x1, y1)).copy()
    except Exception:
        return None


def native_orientation(obj) -> "list | None":
    """How an embedded image must be turned to face the way the page draws
    it — or None when the shortcut of taking it should be declined.

    A PDF image is painted into the unit square and placed by a matrix. The
    stored raster's first row lands wherever that matrix sends y=1, so a
    negative vertical scale hangs the picture upside down relative to its
    own bytes, and a negative horizontal one mirrors it. FM 20-3's spectrum
    chart is placed with d=-159: extracted as stored it shipped inverted,
    and it passed the confirmation check because a broad banded diagram
    correlates with its own reflection at thumbnail size.

    A matrix with rotation or skew (b or c non-zero) is not a flip and is
    not corrected here; the caller renders the region instead, which is
    always right and only slower.
    """
    try:
        m = obj.get_matrix()
    except Exception:
        return None
    if abs(m.b) > 1e-6 or abs(m.c) > 1e-6:
        return None
    ops = []
    if m.d < 0:
        ops.append(Image.FLIP_TOP_BOTTOM)
    if m.a < 0:
        ops.append(Image.FLIP_LEFT_RIGHT)
    return ops


def _encodable(img: "Image.Image") -> "Image.Image":
    """The same picture in a mode BOTH encoders accept, or it unchanged.

    A figure is encoded as JPEG and as PNG so the smaller can ship, and each
    format refuses a mode the other takes. PNG cannot hold CMYK, which is
    how a plate arrives from a press-ready PDF — After Queer Theory lost a
    whole 225-page build to one such plate — and JPEG cannot hold alpha, the
    same crash waiting on the other line. Alpha is composited onto white
    rather than dropped, because dropping it uncovers whatever colour was
    hiding under a transparent mask instead of the page it was drawn for.
    """
    if img.mode in ("L", "RGB"):
        return img
    if img.mode == "P" or "A" in img.mode:
        rgba = img.convert("RGBA")
        flat = Image.new("RGB", rgba.size, "white")
        flat.paste(rgba, mask=rgba.split()[-1])
        return flat
    return img.convert("RGB")


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
        if src.get("turned"):
            return crop     # the page was turned: the PDF draws it otherwise
        import pypdfium2 as pdfium
        doc = pdfium.PdfDocument(src["pdf"])
        page = doc[src["page"]]
        frame = _page_frame(page)
    except Exception:
        return crop

    best = None
    try:
        for obj in page.get_objects():
            if not isinstance(obj, pdfium.PdfImage):
                continue
            b = obj.get_bounds()
            placed = _pdf_box_to_normalized(b, frame)
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
                turns = native_orientation(best[1])
                found = None
                if turns is not None:
                    best[1].extract(stem)
                    found = next(iter(sorted(glob.glob(stem + ".*"))), None)
                if found:
                    with Image.open(found) as native:
                        native.load()
                        native = native.convert("RGB")
                    for op in turns:
                        native = native.transpose(op)
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


# ── Duplicate pages ───────────────────────────────────────────────────────────

def _shingles(text: str, k: int = SHINGLE_WORDS) -> set[str]:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def demote_mislabelled_heads(items: "list[PageItem]",
                             page_name: str = "") -> "list[PageItem]":
    """Return items with a mislabelled section heading restored to the body.

    The layout pass sometimes labels a section's display heading as page
    furniture — measured once on a notes section whose FOOTNOTES sat at the
    top of a column, exactly where a running head lives; the label read
    clean text at 99.6% confidence and the body silently lost the heading,
    which unlinked half a chapter's notes and cross-linked the rest by
    position. The demotion demands page-local evidence: the text must BE a
    notes heading and the same page must carry the section's own first
    numbered entry. Any other furniture that carries a full line of words
    is only reported — visibility, not guesswork — because a true running
    head repeated across a section must stay out of the body.
    """
    has_entry = any(not j.is_furniture and j.html
                    and re.search(r"(?:^|>)\s*1\.\s", j.html)
                    for j in items)
    # A page that already shows a heading has lost nothing, and the demotion
    # exists only to restore what was lost. Without this, every CONTINUATION
    # page of a multi-chapter notes section qualifies — a new chapter's notes
    # open with "1." there, so the entry test passes — and the running head
    # printed at the top of every page of the section is copied into the body
    # beside the group head that was never missing. Thirty such demotions on
    # the shelf, in seven books, every one of them on a page carrying its own
    # head ("IV. Collaboration", "CHAPTER 2"); three of one book's reached
    # the table of contents.
    has_head = any(not j.is_furniture and j.html
                   and (j.label == "SectionHeader"
                        or re.match(r"\s*<h[1-6]", j.html))
                   for j in items)
    out = []
    for it in items:
        if it.is_furniture and it.html:
            t = re.sub(r"\s+", " ", _strip_tags(it.html)).strip()
            if has_entry and not has_head \
                    and re.fullmatch(r"(?:FOOT|END)?NOTES?", t, re.I):
                print(f"    [~] {page_name}: {t!r} was labelled page "
                      f"furniture but names the section beside it — kept "
                      f"in the body")
                out.append(it._replace(is_furniture=False))
                continue
            if len(t) > 60 or len(re.findall(r"[A-Za-z]{2,}", t)) > 8:
                print(f"    [~] {page_name}: furniture label on a full "
                      f"line of text — left out of the body: {t[:70]!r}")
        out.append(it)
    return out


# A block closed, and then content that is not a block. Cheap enough to run
# over every cached fragment in a book; true for 137 of 123,377.
_CLOSED_EARLY = re.compile(
    r"</(?:p|h[1-6]|li|blockquote|figcaption)>\s*"
    r"(?!<(?:p|h[1-6]|li|ul|ol|table|div|figure|blockquote|pre|hr)\b)(?=\S)")


# A fragment that begins with running text rather than a block has text
# outside every block, whatever else it holds (see _wrap_loose_inline).
_OPENS_ON_A_BLOCK = re.compile(
    r"\s*<(?:p|h[1-6]|li|ul|ol|table|div|figure|blockquote|pre|hr|math|img)\b")


def refold_cached_fragments(items: "list[PageItem]") -> "list[PageItem]":
    """Fold loose inline content in fragments that predate the repair.

    The cache stores each fragment ALREADY converted, so a warm rebuild
    never passes back through _to_xhtml — and a repair made there alone
    would reach only freshly OCR'd books, which is no book anyone rebuilds.
    This was not a guess: the first attempt shipped the fold in _to_xhtml,
    eleven books were rebuilt from warm caches, and every one came back
    byte-identical.

    Re-running the conversion is what repairs it, so the fragment takes
    exactly the path a fresh reading would, and the cheap test above keeps
    that to the handful of fragments that need it.
    """
    out = []
    for it in items:
        if it.html and (_CLOSED_EARLY.search(it.html)
                        or not _OPENS_ON_A_BLOCK.match(it.html)):
            it = it._replace(html=_to_xhtml(it.html))
        out.append(it)
    return out


def route_form_blocks(items: "list[PageItem]") -> "tuple[list, int]":
    """Items with the layout's Form blocks kept out of the body.

    A Form is a document facsimile — a death certificate, a printed
    blank — not narrative: its fields read as prose beside the figure of
    the document ("1. PLACE OF DEATH STATE OF TEXAS Brooks COUNTY OF…").
    The layout pass labels it, so the routing is a label, not a guess.
    Its text is not lost: the PDF text layer collects items on its own
    path and keeps every block, so the facsimile stays searchable where
    the document actually is. A Form item carrying a figure keeps the
    figure. Measured on the shelf: two books carry the class at all.
    """
    out, dropped = [], 0
    for it in items:
        if it.label == "Form" and it.figure is None and it.html:
            dropped += 1
            continue
        out.append(it)
    return out, dropped


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


def head_shape(text: str) -> str:
    """A running head with its folio masked, so every page's is one shape."""
    masked = re.sub(r"\d+", "#", " ".join(text.split()))
    return re.sub(r"[^A-Za-z#]+", " ", masked).strip().lower()


# A block with nothing block-level inside it, which is what a running head is.
# Written so it can never swallow a figure, a list or a table by accident.
_LEAF_BLOCK = (r"<(p|h[1-6])\b[^>]*>"
               r"(?:(?!</?(?:p|h[1-6]|figure|div|table|ul|ol|blockquote)\b).)*?"
               r"</\1>")
_FIRST_LEAF = re.compile(r"\A\s*" + _LEAF_BLOCK, re.S)
_LAST_LEAF = re.compile(_LEAF_BLOCK + r"\s*\Z", re.S)
# The same head, not given a block of its own: the layout pass put the
# running head and the first line of prose in ONE paragraph with a <br/>
# between them. One book carries 21 of these, and they survived the first
# version of this pass untouched — it read a block's WHOLE text, which for
# a fused head is the head plus the paragraph it interrupts, far past any
# length a running head has. The check written to confirm that fix shared
# the same assumption and reported the book clean.
_INLINE = r"(?:(?!<br\s*/?>|</?(?:p|h[1-6]|figure|div|table|ul|ol|blockquote)\b).)*?"
_FIRST_FUSED = re.compile(r"\A\s*<(?:p|h[1-6])\b[^>]*>(" + _INLINE
                          + r")(<br\s*/?>\s*)", re.S)
_LAST_FUSED = re.compile(r"(\s*<br\s*/?>)(" + _INLINE
                         + r")</(?:p|h[1-6])>\s*\Z", re.S)
_OPENS_HEADING = re.compile(r"\A\s*<h[1-6]\b")
_ENDS_HEADING = re.compile(r"</h[1-6]>\s*\Z")
# Four places a head can sit: alone in the page's first or last block, or
# fused by a <br/> into the front of the first block or the back of the
# last. Whole blocks first, so a page that has both is not read through the
# wrong lens.
_HEAD_SITES = ((_FIRST_LEAF, True, False), (_LAST_LEAF, False, False),
               (_FIRST_FUSED, True, True), (_LAST_FUSED, False, True))
# A running head that carries no folio is known by its rhythm instead: the
# same line opening page after page, never more than this many pages between
# one sighting and the next — every page, or every recto.
HEAD_CADENCE = 2


def _head_site(body: str, pattern, at_top: bool, fused: bool):
    """The span a running head at this site would be cut from, and its text,
    or None where the page has no such block."""
    m = pattern.search(body)
    if m is None:
        return None
    if fused:
        # the head is one group and the break the other; the span to cut
        # covers both, so the paragraph closes up behind it
        return ((m.start(1), m.end(2)),
                " ".join(_strip_tags(m.group(1 if at_top else 2)).split()))
    return (m.start(), m.end()), " ".join(_strip_tags(m.group(0)).split())


def _head_key(text: str) -> str:
    """One line however each page happens to set it: case, the setting of
    its quotes and dashes, and its spacing aside."""
    return " ".join(text.translate(_SETTING).split()).casefold()


def _head_runs(bodies: list[str], page_furniture: list[list[str]]) -> dict:
    """{(line, page): (sightings in its run, the run's first page)} for every
    short line that opens a page, or that the layout called furniture there,
    split into runs wherever more than HEAD_CADENCE pages pass without it."""
    seen: dict[str, set] = {}
    for i, texts in enumerate(page_furniture):
        for t in texts:
            seen.setdefault(_head_key(t), set()).add(i)
    for i, body in enumerate(bodies):
        for site in _HEAD_SITES:
            if not site[1]:
                continue
            hit = _head_site(body, *site)
            if hit and hit[1] and len(hit[1]) <= RUNNING_HEAD_MAX_CHARS:
                seen.setdefault(_head_key(hit[1]), set()).add(i)
    runs = {}
    for key, pages in seen.items():
        pages = sorted(pages)
        start = 0
        for k in range(1, len(pages) + 1):
            if k < len(pages) and pages[k] - pages[k - 1] <= HEAD_CADENCE:
                continue
            for p in pages[start:k]:
                runs[(key, p)] = (k - start, pages[start])
            start = k
    return runs


def _worded_head(text: str) -> bool:
    """Whether a line of page furniture carries words, as a running head
    does. A designation does not: "SWM 6-77" names a paragraph of the
    manual as a page number names a page, "AGO 8641A" is the printer's
    and "DOI: 10.21983/…" the publisher's. One short word in capitals
    beside a figure is that, and a page that carries it may still have a
    running head the layout took for a heading: the Small Wars Manual sets
    one under its designation on 198 pages, 115 of them in the middle of a
    sentence that runs over the page."""
    text = html.unescape(text)
    words = [w for w in re.findall(r"[A-Za-z]+", text)
             if not re.fullmatch(r"[ivxlcdm]+", w, re.I)]
    if not any(len(w) >= 3 for w in words):
        return False
    longer = [w for w in words if len(w) >= 2]
    return not (len(longer) == 1 and len(longer[0]) <= 3
                and longer[0].isupper() and re.search(r"\d", text))


def strip_missed_running_heads(bodies: list[str],
                               page_furniture: list[list[str]]
                               ) -> list[tuple[int, str]]:
    """
    Drop the running heads the layout pass called content, and say which.

    The layout model is not consistent about its own furniture. On one book
    it labelled the line "10 THE REBEL PASSION" a PageHeader on 209 pages
    and a SectionHeader on 49 — the same place on the page, the same 0.99
    confidence — so 47 running heads shipped inside the prose, and 35 of
    them as HEADINGS, dropped into sentences that ran across the page
    break. No test of shape or position can separate the two groups,
    because there is no difference between them to find. Only the book's
    own testimony can, and the book gives it freely: a running head is
    printed on every page of a section, so a shape recurring under a
    furniture label is the book naming that line.

    Three sightings, the threshold FURNITURE_PAGES already sets for
    repeated decoration and for the same reason: two is a coincidence,
    three is a habit. Recurrence alone is not enough, though, and the two
    conditions that guard it were measured on the shelf rather than
    supposed. parse_folio must read a page number out of the line, which
    refuses the largest class of lookalikes — a bare chapter number
    standing over its title, 20-odd of each digit across the books. And
    the block beside it must not itself be a heading, because a chapter's
    OPENING page prints no running head: a head-shaped line there is the
    chapter number above the chapter's name.

    Across every cached book the rule takes 57 lines in 8 books, while the
    two conditions spare 129 repeated section titles and 3 chapter
    openings.

    A book that numbers its pages at the foot gives its running heads no
    folio to read, and there the folio condition refuses every leak: China's
    Engine of Environmental Collapse set its recto head REFERENCES as a
    heading on all 36 notes pages it heads, and never once as furniture, so
    the census never saw it either. What such a head has instead is rhythm —
    the identical line, case and typesetting aside, opening page after page
    with never more than HEAD_CADENCE pages between sightings — where a
    repeated section title returns once a chapter and a chapter number
    changes every time. The run's first page is the section's own heading,
    since an opening page prints no running head, so it stays, and the head
    must be seen FURNITURE_PAGES times besides it. The rhythm is read at the
    head of the page only, because the foot is where a page repeats its own
    apparatus, and not on a page whose furniture already carries words: that
    page has its running head. Across the shelf this takes 217 more lines in
    13 books — recto heads through notes, conclusions and indexes, leaked
    book titles, a manual's designation — while those conditions spare a
    chart's source line under four charts running, a field manual's NOTE
    callouts, and a title printed on a cover, a half-title and a title page.
    """
    census: dict[str, int] = {}
    for texts in page_furniture:
        for t in texts:
            s = head_shape(t)
            if s:
                census[s] = census.get(s, 0) + 1
    removed: list[tuple[int, str]] = []
    if not census:
        return removed
    runs = _head_runs(bodies, page_furniture)
    # A page whose furniture already carries words, not just a number, has
    # its running head; a second line the rhythm points at is its own.
    has_head = [any(_worded_head(t) for t in texts)
                for texts in page_furniture]
    has_head += [False] * (len(bodies) - len(has_head))
    for i, body in enumerate(bodies):
        for pattern, at_top, fused in _HEAD_SITES:
            site = _head_site(body, pattern, at_top, fused)
            if site is None:
                continue
            cut, text = site
            if not text or len(text) > RUNNING_HEAD_MAX_CHARS:
                continue
            if parse_folio(text) is not None:
                if census.get(head_shape(text), 0) < FURNITURE_PAGES:
                    continue
            else:
                n, first = runs.get((_head_key(text), i), (0, i))
                if not at_top or first == i or n - 1 < FURNITURE_PAGES \
                        or has_head[i]:
                    continue
            rest = body[cut[1]:] if at_top else body[:cut[0]]
            if not fused and \
                    (_OPENS_HEADING if at_top else _ENDS_HEADING).search(rest):
                continue
            left = body[:cut[0]] + body[cut[1]:]
            # A head printed on an otherwise empty leaf interrupts nothing,
            # and taking it leaves a body that reads as blank — which the
            # blank pass then deletes, costing a page and its page-list
            # anchor to tidy one line. Losing a page is the larger harm.
            if "<img" not in left and not _strip_tags(left).strip():
                continue
            body = left
            bodies[i] = body
            removed.append((i, text))
    return removed


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


# A figure is encoded both ways and the smaller kept. Line art and charts
# come out smaller as PNG *and* lossless — a drawing measured 6KB against
# 17KB — because JPEG spends its bits ringing the hard black edges that made
# this artwork look harsh in the first place. A photograph goes the other
# way by more than four to one, and takes JPEG on merit.
#
# The PNG that competes is the optimised one, and it is the slow one: on
# the 6,847 figures in the shelf's caches it takes thirteen times as long
# as a PNG written at the fastest setting, and it loses to the JPEG for
# 5,823 of them. The fast PNG says whether the contest is worth holding. No
# figure whose optimised PNG won had a fast PNG more than 1.69 times the
# size of its JPEG, so past 2.5 times the optimised one is not written:
# that is 47% of the figures and three quarters of the time it took, and
# the same choice, byte for byte, for every one of the 6,847.
FIGURE_PNG_OUT_OF_REACH = 2.5
FIGURE_ENCODERS = max(1, min(8, os.cpu_count() or 1))   # threads, at once


def encode_figure(art: Image.Image) -> "tuple[bytes, str, str]":
    """(bytes, extension, media type) of a figure as the book will carry it."""
    jb = io.BytesIO()
    art.save(jb, "JPEG", quality=88)
    quick = io.BytesIO()
    art.save(quick, "PNG", compress_level=1)
    if quick.tell() <= FIGURE_PNG_OUT_OF_REACH * jb.tell():
        pb = io.BytesIO()
        art.save(pb, "PNG", optimize=True)
        if pb.tell() <= jb.tell():
            return pb.getvalue(), "png", "image/png"
    return jb.getvalue(), "jpg", "image/jpeg"


def _same_file(a: str, b: str) -> bool:
    """Whether two paths name one file, by any spelling or link."""
    try:
        return os.path.samefile(a, b)
    except OSError:             # one of them does not exist yet
        return os.path.abspath(a) == os.path.abspath(b)


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


# Front matter paginates in Roman, and those pages must sort before Arabic 1
# while staying integers, because the audit, the gap filler and the resolver
# all compare and subtract folios. Roman i…xcix is carried as i - 1000, so
# the whole numbering space reads in order: Roman below zero, then plain
# Arabic 1…999, then the manual compound forms above.
ROMAN_FOLIO_BASE = 1000
_ROMAN_TENS = ["", "x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc"]
_ROMAN_ONES = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]


def folio_label(n: int) -> str:
    """A folio as the page prints it, from the integer it is carried as.

    The encodings exist so that every mechanism built on plain folios keeps
    working, and they were never meant to be read by anyone. They were:
    three manuals shipped a page-list labelled 1001, 1002, 2001 for pages
    printed 1-1, 1-2, 2-1, so "go to page 2-1" could not find them and the
    numbers a reader saw belonged to no page in the book.

    Roman comes back lowercase, which is what front matter prints; a book
    that sets its front matter in capitals is rendered against its own
    house style, and nothing but the label is affected.
    """
    if n < 0:                                  # front matter, Roman
        k = n + ROMAN_FOLIO_BASE
        return _ROMAN_TENS[k // 10] + _ROMAN_ONES[k % 10] if 1 <= k <= 99 else str(n)
    if n >= 100 * 1000:                        # appendix, "B-14"
        return f"{chr(ord('A') + n // 1000 - 100)}-{n % 1000}"
    if n >= 1000:                              # chapter-page, "4-9"
        return f"{n // 1000}-{n % 1000}"
    return str(n)


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


class Chapter(NamedTuple):
    """A chapter as the running heads describe it: a title and the span of
    pages that carried it."""
    title: str
    start: int           # first page index whose head named this chapter
    end: int             # last such page, inclusive


def head_title(line: str) -> str:
    """A running head with its folio, and whatever separated them, taken off.

    parse_folio reads the number out of the same line; this reads the rest.
    Books set the pair a dozen ways — "60 · Citadel of Sin", "THE NEW CAREER
    51", "Chapter 7 — 61" — so both ends are tried and the separator goes
    with the number.
    """
    t = html.unescape(" ".join((line or "").split()))
    t = re.sub(r"^\d{1,3}\s*[•·|:.\-–—]*\s*", "", t)
    t = re.sub(r"\s*[•·|:.\-–—]*\s*\d{1,3}$", "", t)
    # Front matter paginates in roman, and a lone "vi" left standing becomes
    # a two-page chapter called vi. Only a bare numeral is taken — a chapter
    # actually titled "I" keeps company with words and survives.
    if re.fullmatch(r"(?i)[ivxlcdm]{1,7}", t):
        return ""
    return " ".join(t.strip(" •·|:.-–—").split())


def _same_head(a: str, b: str) -> bool:
    """Whether two lines are the same running head read twice.

    A head is re-recognised on every page it appears, and the readings
    differ: one page's TIHKAL is another's THIKAL, one escapes an
    apostrophe where the next prints it. Exact matching splits a single
    chapter into several — measured, one stray misread turned an 18-page
    chapter into a 12 and a 6 — so heads are compared for closeness.
    """
    if not a or not b:
        return False
    a, b = a.lower(), b.lower()
    if a == b:
        return True
    if min(len(a), len(b)) < 4:
        return False
    # A single substituted letter is the commonest misreading of a head, and
    # on a short title it does not survive a ratio test: "Invasiom" against
    # "Invasion" scores 0.875 and would open a second chapter halfway
    # through the first. Same length, one letter apart, is that misreading.
    if len(a) == len(b) and sum(x != y for x, y in zip(a, b)) == 1:
        return True
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.90


# A stream of heads that changes on nearly every page is not chapters — it is
# marginal notes, or an edited volume naming a different contributor each
# leaf. Measured across the shelf: real chapter partitions run 8-12 pages to
# the run, degenerate ones run 1.0-1.2, and nothing legitimate falls between.
CHAPTER_RUN_MIN_MEAN = 4.0

# What a running head can say that is not a chapter's name. A manual prints
# its issue date and the bare word "Chapter" where a book prints a title,
# and those passed every structural guard — the runs were long and evenly
# spread — while naming nothing. They matched no heading on any page, and
# by existing at all they kept the printed contents from being consulted.
_NOT_A_CHAPTER_NAME = re.compile(
    r"^(chapter|part|section|book|volume|appendix|figure|table|page|"
    r"continued|cont|contents|table of contents|index of contents)$"
    r"|^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*\d{1,4}$"
    r"|^[\d\s./–—-]+$"
    # a scan's margins can carry the archive's own furniture: one book's
    # every page bottom reads http://www.archive.org/details/… and the run
    # is as sustained as any chapter's
    r"|.*(https?://|www\.)", re.I)


def chapters_from_furniture(page_furniture: list[list[str]]
                            ) -> "tuple[list[Chapter], dict]":
    """The book's chapters, as its own running heads report them.

    A running head is the one piece of structure a book repeats on every
    page, and it names the chapter the reader is inside. Two streams share
    the position — the verso usually carries the book's title and the recto
    the chapter's — so the constant one is identified and set aside, and
    what varies is the chapter.

    This is evidence the printed contents cannot give: a span asserted
    twenty times over rather than a single line whose page must be inferred
    from a folio. Where a contents page parses it still outranks this, being
    the book's own declaration of hierarchy; where none does, or where it
    parses badly, this is the better skeleton.

    Returns (chapters, stats). An empty list means the heads did not
    describe chapters, which is a real answer for a book whose head is only
    its own title, and for a field manual whose head is a document number.
    """
    n = len(page_furniture)
    stats = {"pages": n, "with_head": 0, "runs": 0, "mean_run": 0.0,
             "refused": ""}
    if n < 8:
        stats["refused"] = "too few pages"
        return [], stats
    titles = [[head_title(t) for t in (page or [])] for page in page_furniture]
    flat = [t for page in titles for t in page if t]
    if not flat:
        stats["refused"] = "no running heads carry text"
        return [], stats
    # Which stream is the book's own title? Not the most frequent one —
    # frequency alone mistakes a long chapter for the title and eats it.
    # The distinguishing property is SPREAD: a title recurs from the first
    # leaf to the last, while a chapter head owns a band and no more. Where
    # no head spans the book, there is no title stream to set aside, and
    # every head is a chapter — some books print the chapter on both leaves.
    from collections import Counter
    seed = Counter(flat).most_common(1)[0][0]
    seen_at = [i for i, page in enumerate(titles)
               if any(_same_head(t, seed) for t in page if t)]
    spans_book = (seen_at and
                  (seen_at[-1] - seen_at[0] + 1) >= 0.6 * n)
    varying: list[str | None] = []
    for page in titles:
        rest = [t for t in page
                if t and not (spans_book and _same_head(t, seed))]
        varying.append(rest[0] if rest else None)
    stats["with_head"] = sum(1 for v in varying if v)
    if stats["with_head"] < 4:
        stats["refused"] = "no second head stream (the head is the title)"
        return [], stats
    # Walk the book. A page with no chapter head does not end a chapter — a
    # verso carrying only the book's title sits INSIDE one, and so does the
    # chapter's own opening page, which by convention prints no head at all.
    runs: list[list] = []
    for i, v in enumerate(varying):
        if v is None:
            continue
        if runs and _same_head(runs[-1][0], v):
            runs[-1][2] = i
        else:
            runs.append([v, i, i])
    # A single page bearing a head unlike its neighbours, between two that
    # agree, is a misread — not a one-page chapter between two halves of the
    # same one.
    merged: list[list] = []
    for k, r in enumerate(runs):
        if (r[1] == r[2] and merged and k + 1 < len(runs)
                and _same_head(merged[-1][0], runs[k + 1][0])):
            continue
        if merged and _same_head(merged[-1][0], r[0]):
            merged[-1][2] = r[2]
        else:
            merged.append(list(r))
    # A run named by something that is not a name is not a chapter.
    merged = [r for r in merged if not _NOT_A_CHAPTER_NAME.match(r[0].strip())]
    stats["runs"] = len(merged)
    if len(merged) < 3:
        stats["refused"] = "fewer than three chapters"
        return [], stats
    stats["mean_run"] = round(stats["with_head"] / len(merged), 2)
    if stats["mean_run"] < CHAPTER_RUN_MIN_MEAN:
        stats["refused"] = (f"heads change every {stats['mean_run']} pages — "
                            f"marginalia, not chapters")
        return [], stats
    # If one head owns most of the book it is not a chapter, it is the
    # constant stream wearing the varying one's clothes — a manual whose
    # every page names the manual, where the head set aside as "the title"
    # was something rarer. Measured: a legitimate widest chapter takes a
    # quarter of its book, this failure took 96% of one.
    widest = max(b - a + 1 for _, a, b in merged)
    if widest > 0.6 * n:
        stats["refused"] = (f"one head spans {widest} of {n} pages — the "
                            f"book's own title, not a chapter")
        return [], stats
    # A partition that is mostly one-page runs is a list of section names
    # repeating, not a book divided into chapters.
    if sum(1 for _, a, b in merged if b - a + 1 >= 3) < 3:
        stats["refused"] = "fewer than three chapters span three pages"
        return [], stats
    # A chapter's opening page prints no running head — that is the
    # convention, the drop cap stands where the head would be — so the first
    # page NAMING a chapter is usually its second. Step back onto the
    # opening where no other chapter has claimed it.
    claimed = {p for _, a, b in merged for p in range(a, b + 1)}
    out = []
    for t, a, b in merged:
        if a - 1 >= 0 and (a - 1) not in claimed:
            a -= 1
        out.append(Chapter(t, a, b))
    return out, stats


# How a book designates a division before naming it: "Chapter 1", "PART
# ONE", "III.", "7". A contents line usually carries the designation and
# the page it points at usually does not, so both are stripped before the
# titles are compared. Four digits are left alone — a chapter called "1984"
# is a title, not a number.
_DESIGNATION = r"(?:chapter|part|section|book|appendix)"

# A number a book spells out. Only ever read where a designation introduces
# it: bare, "one" opens too many real titles to strip — "One Hundred Years
# of Solitude" is not chapter one of anything.
#
# Longest spelling first, and a boundary at the end. Alternation takes the
# first branch that fits, not the longest: with the small numbers leading,
# "FOURTEEN" matched "four" and left "TEEN" standing as the chapter's title.
# That renumbered fourteen as four in the body AND in the endnotes, so the
# two could no longer be aligned and every note link in the book was
# withheld. Either guard alone would do; both are cheap.
_NUMBER_WORD = (
    r"(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)"
    r"(?:[-\s](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen"
    r"|ten|eleven|twelve"
    r"|one|two|three|four|five|six|seven|eight|nine)\b")

_CHAPTER_MARK = re.compile(
    r"^\s*(?:"
    rf"{_DESIGNATION}\s+{_NUMBER_WORD}"
    rf"|{_DESIGNATION}?\s*(?:\d{{1,3}}|[ivxlc]{{1,7}})"
    r")\s*[:.\-—]?\s+", re.I)


def _bare_title(t: str) -> str:
    """A division's title with its designation taken off."""
    return " ".join(_CHAPTER_MARK.sub("", t).split())


def _names_chapter(heading: str, head: str) -> bool:
    """Whether a heading on the page is the chapter this running head names.

    A running head ABBREVIATES, and it abbreviates in one particular way: it
    drops the SUBTITLE. The head reads "INTRODUCTION" where the page is
    titled "Introduction: Going Inside", "BEING THERE" where the page says
    "Being There: Social Life in the Centre" — scoring 0.59 and 0.48, so
    similarity alone matches neither. What licenses the match is the colon.
    A plain space does not: "Border" must not claim the section called
    "Border security", and on real books every abbreviation observed was cut
    at punctuation, never mid-phrase.
    """
    a = " ".join(heading.lower().split())
    b = " ".join(head.lower().split())
    if _same_head(a, b):
        return True
    # A contents line and the page it points at rarely agree about the
    # chapter's number: the contents says "Chapter 1 Understanding the
    # Strategic Context" and the page is headed by the title alone, or the
    # reverse. Strip the designation from both and compare what is left.
    ba, bb = _bare_title(a), _bare_title(b)
    if ba and bb and _same_head(ba, bb):
        return True
    a, b = ba or a, bb or b
    if len(b) < 6 or len(a) < 6:
        return False
    lo, hi = (b, a) if len(b) <= len(a) else (a, b)
    if hi.startswith(lo) and hi[len(lo):len(lo) + 1] in (":", ".", ",",
                                                         "—", "–", "-"):
        return True
    # The mirror image: a head that abbreviates to the SUBTITLE. One book's
    # margins run THE COLLAPSE OF THE MIDDLE AGES where the contents line is
    # "CHAPTER ONE. “The King's in His Castle …”: The Collapse of the
    # Middle Ages" — the head is the post-colon half, and matching only the
    # front half called five of its chapters unaccounted for. The same
    # guards apply: cut at punctuation, never mid-phrase.
    rest = hi[:len(hi) - len(lo)].rstrip()
    return hi.endswith(lo) and bool(rest) and rest[-1] in (":", ".", ",",
                                                           "—", "–", "-")


def promote_chapter_headings(bodies: list[str],
                             chapters: "list[Chapter]") -> int:
    """Put each chapter's own title at the top of the document outline.

    Measured across the shelf, h1 is mostly occupied by the title page —
    often in pieces, and often misread ("Inthropology'sWorld", "SEX &
    POLYNER") — while the chapters that actually structure the book sit at
    h2 or h3, and one manual reached the reader with no h1 at all. A
    reading system builds its outline from these tags, so the book's
    skeleton was the one thing they did not describe.

    Only a heading whose text is the chapter's own name, on the page where
    that chapter opens or the one after, is promoted. Nothing is demoted:
    a title page's h1 is not wrong, it is just not a chapter, and moving it
    would trade one arguable outline for another.
    """
    n = 0
    floor = 0
    for c in chapters:
        # The head names a chapter from the page AFTER its opening, and on a
        # verso-title book the opening can be two leaves back. Search a small
        # window, never past where the previous chapter ended.
        window = [p for p in range(c.start - 2, c.start + 2)
                  if floor <= p < len(bodies)]
        floor = c.end + 1
        for pi in window:
            if not 0 <= pi < len(bodies):
                continue
            hit = False
            for m in re.finditer(r"<h([1-6])([^>]*)>(.*?)</h\1>",
                                 bodies[pi], re.S):
                rank, attrs, inner = m.group(1), m.group(2), m.group(3)
                text = " ".join(_strip_tags(inner).split())
                if rank == "1" or not _names_chapter(text, c.title):
                    continue
                bodies[pi] = (bodies[pi][:m.start()]
                              + f"<h1{attrs}>{inner}</h1>"
                              + bodies[pi][m.end():])
                n += 1
                hit = True
                break
            if hit:
                break
    return n


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
            if n is None and _roman_to_int(bare) is not None:
                # Front matter's own pagination, on the same terms: the whole
                # furniture line and nothing else. Roman numerals are letters,
                # so the guards are what keep prose out — _roman_to_int
                # re-renders what it read and refuses anything that does not
                # come back identical, which turns away every English word
                # spelled from the same letters, and it stops at xcix, past
                # any front matter a book actually has. A stray reading that
                # survives all that still has to join the strictly increasing
                # run in fill_folio_gaps, where a Roman numeral sitting among
                # Arabic pages cannot.
                n, rank = _roman_to_int(bare) - ROMAN_FOLIO_BASE, 0
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


# A signature mark: the letter a printer set at the foot of the first page
# of each gathering, to tell the binder the order of the sheets. "B", "D",
# "F" down The Knights of Industry, "A A", "B B" once the alphabet is used
# up in Lloyd's. The layout calls it text, with good confidence, and it is
# the last paragraph of its page: bound into chapters, it stands between
# two halves of a sentence. One alone cannot be told from a paragraph that
# says "B". A run of them can: they climb through the alphabet as the book
# goes on, and nothing else at the foot of a page does. On 152 books the
# rule takes 12 marks in those two and nothing else; a table's lone "H"
# and an index's letter "K" are single and stay.
_SIGNATURE = re.compile(
    r"<p>\s*((?:\d\s?)?[A-Za-z](?:\s?[A-Za-z])?(?:\s?\d)?)\s*</p>\s*$")
SIGNATURES_WANTED = 3


def _signature_rank(mark: str) -> "tuple | None":
    """Where a mark comes in a printer's series: B before C, and a doubled
    letter after every single one. None for two different letters."""
    letters = [ch.lower() for ch in mark if ch.isalpha()]
    if len(set(letters)) != 1:
        return None
    return len(letters), letters[0]


def strip_signature_marks(bodies: list[str]) -> list[tuple[int, str]]:
    """Take out the printer's signature marks; return (page, mark) for each."""
    found = []
    for i, body in enumerate(bodies):
        m = _SIGNATURE.search(body)
        if m and len(m.group(1).replace(" ", "")) <= 3:
            rank = _signature_rank(m.group(1))
            if rank is not None and body[:m.start()].strip():
                found.append((i, m, rank))
    # the longest run that climbs: what is not in it is something else
    best: list = []
    runs: list = []
    for k, (_, _, rank) in enumerate(found):
        before = [r for j, r in enumerate(runs) if found[j][2] <= rank]
        runs.append(max(before, key=len, default=[]) + [k])
        if len(runs[-1]) > len(best):
            best = runs[-1]
    ranks = {found[k][2] for k in best}
    if len(best) < SIGNATURES_WANTED or len(ranks) < SIGNATURES_WANTED:
        return []
    removed = []
    for k in best:
        i, m, _ = found[k]
        bodies[i] = bodies[i][:m.start()] + bodies[i][m.end():]
        removed.append((i, m.group(1)))
    return removed


# A page's own number, run into its last line. Where the number is printed
# close under the text the layout draws one box round both, and the page
# ends "…his valuable bag-89": bound, "bag-89 gage". Only the page's own
# number, only as the last thing on the page and a whole figure there (not
# the 200 of "29,200"), only after a letter, a hyphen, or a comma and a
# space (after a full stop it would be a sentence's last figure), and not
# after a word that names what the number counts. That a page's text should
# end on a figure that happens to be the page's own number is rare enough
# to leave to chance: on 152 books the rule takes seven, six of them in The
# Man Who Stole Portugal, and each is the number under the text.
_TAKES_A_NUMBER = re.compile(
    r"(?:pages?|pp?|chapters?|ch|chap|tables?|figures?|figs?|sections?|sec|"
    r"vols?|volumes?|nos?|numbers?|parts?|lines?|notes?|items?|"
    r"paragraphs?|paras?)\.?$", re.I)


def strip_fused_folios(bodies: list[str], numbers: dict) -> list:
    """Take a page's own printed number off the end of its last line;
    return (page, number) for each."""
    removed = []
    for i, number in numbers.items():
        if not isinstance(number, int) or number <= 0 or i >= len(bodies):
            continue
        m = re.search(r"(?:(?<=[A-Za-z\-–])|(?<=[A-Za-z,(]\s)|(?<=[A-Za-z]\s\s))"
                      r"(\s*)%d(\s*(?:</[^>]+>\s*)*)$" % number, bodies[i])
        if not m:
            continue
        kept = bodies[i][:m.start(1)].rstrip()
        said = _strip_tags(kept).split()
        if not said or _TAKES_A_NUMBER.match(said[-1].strip("([\"“‘'")):
            continue
        bodies[i] = kept + m.group(2)
        removed.append((i, number))
    return removed


def strip_folio_superscripts(bodies: list[str],
                             folios: "list[Folio] | None") -> int:
    """A page's own printed number, mistaken for a note marker.

    Anthropology's World ended a page with `…Anthropology?”2<sup>38</sup>` on
    the leaf whose folio is 38: the layout lifted the page number into a
    superscript and left the real marker beside it as plain text. Two such
    phantoms cost that book all 167 of its note links — the chapter-reset
    test looks one marker ahead, so a stray 38 sitting just after a chapter's
    opening 1 hides the boundary, merges three chapters into one run, and the
    body's group count no longer matches the notes'; the misalignment guard
    then correctly refuses to link anything at all.

    Deliberately narrow, and the narrowness is measured: across 3,253 books
    on the shelf this fires on exactly those two pages. A superscript
    qualifies only as the last one on its page, only when nothing but markup
    follows it, and only when it reads as that page's own folio. The demoted
    marker beside it is left alone — recovering it would be a guess, and one
    unlinked note is a smaller loss than a wrong link.
    """
    if not folios:
        return 0
    n = 0
    for f in folios:
        if f is None or f.number is None or not 0 <= f.index < len(bodies):
            continue
        b = bodies[f.index]
        marks = list(BODY_MARKER.finditer(b))
        if not marks:
            continue
        m = marks[-1]
        if int(m.group(1)) != f.number:
            continue
        if _strip_tags(b[m.end():]).strip():
            continue                      # prose follows: a real marker
        # The layout merged marker and folio into one run and superscripted
        # only its tail, so the marker is sitting in the prose as digits
        # GLUED to the preceding word — "locations.1" is note 1 on the leaf
        # whose folio is 59. Restoring it is what puts the chapter boundary
        # back, and it cannot ship a wrong link even if it is wrong: the
        # group-alignment check downstream refuses to pair what it cannot
        # reconcile, exactly as it did before this ran.
        head, rest = b[:m.start()], b[m.end():]
        dm = re.search(r"(?<=[^\s\d>])(\d{1,3})\Z", head)
        if dm:
            head = head[:dm.start(1)] + f"<sup>{dm.group(1)}</sup>"
        bodies[f.index] = head + rest
        n += 1
    return n


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
        # the previous neighbour is the last reading that SURVIVED — a
        # just-discarded misread must not vouch for the one after it, or
        # two consecutive misreads slip a false anchor into kept (B5)
        prev = kept[-1] if kept else None
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
        if known[hi] // 1000 != known[lo] // 1000:
            # Front matter giving way to Arabic 1, or a manual's chapter
            # turning over: the count restarts, so the arithmetic between
            # these two anchors means nothing and there is no gap to report.
            # The same test the audit uses to tell a restart from a gap.
            continue
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


# A figure crop that is mostly the reader's hand is not a figure. The layout
# pass boxes a thumb resting on a blank verso as a Picture — reasonably, it
# is the only thing on the page — and the detail test passes it, because
# skin has texture. Measured on the crop that shipped: a 179x504 thumbnail
# of a thumbnail. The same skin classifier page detection uses to keep a
# hand out of the page outline keeps it out of the book here.
FIGURE_MAX_SKIN = 0.5


# A hand carries no printed line-work. Skin alone cannot say what a crop
# is: aged paper sits in the same band, and a photograph OF PEOPLE is
# mostly skin by definition, so the colour test on its own condemned 488
# of the shelf's 10,881 figures — among them 67 of one book's 71 plates,
# a faint survey map, and a portrait. What separates them is ink: black-hat
# keeps dark features narrower than its kernel, which is what a map's
# lines, a halftone's dots and a photograph's detail all leave behind, and
# what smooth flesh does not. Measured over every mostly-skin figure on the
# shelf, the one real hand scores 0.00001 and the faintest real figure
# 0.0022 — two orders of magnitude of daylight, and this sits in it.
FIGURE_HAND_INK = 0.001


def figure_is_a_hand(img: Image.Image) -> bool:
    """Whether the crop is the reader's hand rather than the book's picture.

    Mostly skin AND carrying no fine dark structure. Both conditions, because
    either alone is wrong: skin colour catches every portrait and every tan
    page, and an absence of ink catches every blank leaf — which is
    figure_has_content's business, and is kept rather than discarded.
    """
    try:
        rgb = np.asarray(img.convert("RGB"))
    except (OSError, ValueError):
        return False
    if rgb.shape[0] < 8 or rgb.shape[1] < 8:
        return False
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    if float((_skin_mask(bgr) > 0).mean()) <= FIGURE_MAX_SKIN:
        return False
    grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    ink = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)
    return float((ink > 30).mean()) < FIGURE_HAND_INK


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


# What a division of a book calls itself, in the contents and on its own
# title page: "BOOK SEVEN", "Part III", "SECTION 2". The row branch has
# always required this before grouping a folio-less line; the heading
# branch now does too, so a book's own title reprinted atop a second
# contents page cannot adopt the chapters beneath it.
# The book's apparatus, as a contents page names it. These lines belong to
# no chapter: they are the front and back matter, and a chapter that
# appears to subordinate them is an artefact of their being the only plain
# rows on a page of bold ones.
_APPARATUS_WORD = re.compile(
    r"\b(notes?|references?|index|bibliography|works cited|"
    r"acknowledge?ments?|preface|foreword|afterword|"
    r"glossary|chronolog(?:y|ies)|appendix|appendices|abbreviations|contributors|"
    r"further reading|about the author|credits|permissions)\b", re.I)


def APPARATUS_ROW(title: str):
    """Whether a contents line names the book's apparatus rather than a
    part of its argument.

    Usually the word opens the line. But a book may qualify it — "PERSONAL
    ACKNOWLEDGMENTS" — and matching only at the start stopped the
    back-matter walk dead there, stranding the NOTES and BIBLIOGRAPHY
    printed above it inside the last chapter. So a short line counts
    wherever the word falls. Short is the whole guard: at four words the
    test starts claiming real sections ("The PCE Price Index"), while
    every apparatus line a book actually prints is one or two.
    """
    t = " ".join((title or "").split())
    return _APPARATUS_WORD.match(t) or (
        _APPARATUS_WORD.search(t) if len(t.split()) <= 3 else None)

# What a contents row can name that is not a division of the book. A
# figure caption set in the same weight as a chapter title reads as a
# chapter to any typographic test — one book promoted "Exhibit 3.2",
# "Exhibit 3.5" and "Exhibit 3.6" to chapter rank and left Chapter One and
# Chapter Two as leaves beneath them.
NOT_A_DIVISION = re.compile(
    r"\s*(exhibit|figure|fig\.|table|plate|map|chart|diagram|box|"
    r"illustration|appendix table)\b", re.I)

PART_HEAD = re.compile(
    r"\s*(BOOK|PART|VOLUME|SECTION|APPENDI(X|CES))\b", re.I)


class TocEntry(NamedTuple):
    """One line of a book's own table of contents, as printed."""
    title: str
    folio: int | None        # None: an unnumbered line. Roman is carried
                             # as i - ROMAN_FOLIO_BASE, as the audit carries it
    folio_text: str          # the number as printed, roman numerals included
    depth: int               # 0 = top grouping, 1 = subgroup, 2 = leaf
    # How the page itself declared this line's rank. A typesetter marks a
    # chapter by setting it bold and indents what belongs under it; both
    # survive recognition and both were being discarded at parse time.
    bold: bool = False
    indent: int = 0
    src_page: int = -1


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


# A contents line that is nothing but the designation — the book puts
# "CHAPTER ONE" on one line and names the chapter on the next.
_DESIGNATION_ONLY = re.compile(
    rf"^\s*{_DESIGNATION}\s+(?:\d{{1,3}}|[ivxlcdm]+|{_NUMBER_WORD})"
    r"\s*[.:—–-]?\s*$", re.I)

# A line carrying no title at all: a folio, roman or arabic, or the date
# range that finishes the title above it.
_FOLIO_ONLY = re.compile(r"^\s*(?:[ivxlcdm]+|[\d\s.,:–—-]+)\s*$", re.I)


def _line_text(el) -> str:
    """An element's text with its line breaks kept as spaces.

    A contents entry set over two lines carries a <br/>, and text_content()
    closes it up: one book's chapter one arrived as "All's Right with the
    World":The Collapse of the Middle Ages, which then matched no heading
    anywhere in the book.
    """
    for br in el.iter("br"):
        br.tail = " " + (br.tail or "")
    return " ".join((el.text_content() or "").split())


def _fold_contents_lines(lines: "list[str]") -> "list[str]":
    """Put a designation back together with the title it introduces.

    A contents page with no table gives each of these its own line, and read
    separately neither says what chapter it is: sixteen chapters arrived as
    thirty-two entries, half of them bare numbers. The same page also breaks
    a long title across lines, and where it does the first half ends in a
    comma — which is the only evidence offered that the line is unfinished.
    """
    out: "list[str]" = []
    for line in lines:
        starts_one = bool(_DESIGNATION_ONLY.match(line))
        folio = bool(_FOLIO_ONLY.match(line))
        unfinished = bool(out) and out[-1].rstrip().endswith((",", ":"))
        if out and _DESIGNATION_ONLY.match(out[-1]) and not starts_one \
                and not folio:
            sep = "" if out[-1].rstrip().endswith((":", ".", "-", "—", "–")) else "."
            out[-1] = f"{out[-1].rstrip()}{sep} {line}"
        elif out and unfinished and not starts_one:
            # An unfinished line. A colon or a comma ending a contents line
            # is the book saying the title continues, and on a page with no
            # rows to align it is the only such evidence offered. This is
            # also the one thing that keeps a year range: "…and Socialism,"
            # is finished by "1871–1914" and by nothing else.
            out[-1] = f"{out[-1].rstrip()} {line}"
        elif folio:
            continue                 # a page number, standing on its own
        else:
            out.append(line)
    return out


# A page calls itself a contents page, as against merely mentioning the word.
# "The top should be opened and the contents allowed to melt slowly" is a
# manual talking about a thermos, and reading it as a contents page harvested
# a page of prose about canteens into the book's structure. What separates
# them is the article: a label stands on its own, a common noun is preceded
# by one. "TABLE OF CONTENTS" keeps its "of" and is unaffected.
_NAMES_ITSELF_CONTENTS = re.compile(
    r"(?<!\bthe )(?<!\bits )(?<!\btheir )(?<!\bthis )(?<!\bwhose )contents\b")


# The table's own column headings. They sit in a row like any other and,
# where the leading cell is blank, look exactly like a subordinate entry.
_COLUMN_HEADING = re.compile(
    r"^(?:page|pages|contents|chapter|section|paragraphs?|title|no\.?)$", re.I)


def _blank_lead_runs(body: str) -> bool:
    """Whether this page's blank-leading-cell rows ever come two in a row.

    Both a subsection and a chapter's byline are set the same way — the
    number cell left empty, the title cell holding text — so the row alone
    cannot say which it is. What tells them apart is company. Sections come
    in runs; a byline is one line naming who wrote the chapter above it, and
    a book that gives every chapter an author gives every one of them
    exactly one. One edited collection listed nine contributors that way and
    each arrived as a subsection of the chapter they had written.
    """
    run = 0
    for row in re.findall(r"<tr[^>]*>(.*?)</tr>", body, re.S):
        cells = re.findall(r"<td[^>]*/>|<td[^>]*>(.*?)</td>", row, re.S)
        vals = [" ".join(re.sub(r"<[^>]+>", "", c or "").split()) for c in cells]
        if len(vals) >= 2 and not vals[0] and vals[1]:
            run += 1
            if run >= 2:
                return True
        else:
            run = 0
    return False


def _leads_indented_rows(el, indent: int, need: int = 2) -> bool:
    """Whether the contents rows after this one are set further in than it.

    A part banner carries no folio, and so does a stray line of prose that
    happened to begin with the word Section. What separates them is what
    follows: a grouping has its members set under it. Two are required, so
    that one indented straggler cannot promote a line on its own.
    """
    deeper = 0
    for sib in el.itersiblings("tr"):
        raw = [td.text_content() or "" for td in sib.iter("td", "th")]
        cells = [" ".join(r.split()) for r in raw]
        if not any(cells):
            continue
        if next((len(r) - len(r.lstrip())
                 for r, c in zip(raw, cells) if c), 0) <= indent:
            return False
        deeper += 1
        if deeper >= need:
            return True
    return False


def _peer_banners(el, word: str) -> bool:
    """Whether the page prints other folio-less lines of the same designation.

    A book that sets its chapters flush and indents the part banners instead
    — one here does exactly that — offers no positional evidence at all.
    What it does offer is a series: Part I, Part II, Part III, each printed
    the same way and each without a page number. A line of prose that merely
    opens with the word Section keeps no such company.
    """
    for sib in (list(el.itersiblings("tr", preceding=True))
                + list(el.itersiblings("tr"))):
        cells = [c for c in (" ".join((td.text_content() or "").split())
                             for td in sib.iter("td", "th")) if c]
        if len(cells) != 1:
            continue
        m = PART_HEAD.match(cells[0])
        if m and m.group(1).lower() == word:
            return True
    return False


def parse_printed_toc(bodies: list[str], limit: int = 25,
                      book_title: "str | None" = None
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
        # Rows or list items: the layout renders a block it has labelled a
        # contents as <li> items rather than a table.
        lines_here = rows_here + body.count("</li>")
        looks_like_contents = lines_here >= 5 and leaders >= max(3, lines_here // 3)
        # A bare list of titles runs past its first page as bare paragraphs,
        # with no rows to recognise it by — the Saylor contents names its
        # first 23 chapters on one page and its last 11 on the next, and
        # stopping at the page break loses a third of the book's structure.
        short_lines = len([m for m in re.finditer(r"<p>([^<]{3,120})</p>", body)
                           if re.search(r"[A-Za-z]{3}", m.group(1))])
        is_toc = (bool(_NAMES_ITSELF_CONTENTS.search(text_open))
                  or looks_like_contents
                  or (prev_was_toc and (rows_here >= 5 or short_lines >= 5)))
        prev_was_toc = is_toc
        if not is_toc:
            continue
        toc_pages.add(page_i)
        try:
            root = lhtml.fragment_fromstring(body, create_parent="div")
        except Exception:
            continue
        subordinate_runs = _blank_lead_runs(body)
        # A contents page that lists titles and no page numbers at all — the
        # Saylor economics volumes name 34 chapters as bare paragraphs. There
        # is nothing to resolve against the folio map, so these carry no folio
        # and are placed later by hunting the title through the book. Only
        # taken when the page yields no rows, so a real contents table is
        # never competing with the prose around it.
        if not body.count("</tr>"):
            # Headings as well as paragraphs. A page with no table still
            # sets its designations as headings — one book puts CHAPTER ONE
            # through SEVEN in <h2> and the rest in <p>, and reading only
            # the paragraphs lost the first half of the book. They are not
            # part divisions here: nothing on a table-less page is ranked,
            # so everything arrives at one level, as before.
            # List items as well as paragraphs: the layout renders a block
            # it has labelled a contents as <li> items rather than a table
            # — one book's seventeen chapters arrived that way and the nav
            # was built from nothing. And a line that ends in a dot leader
            # and a folio is a complete entry, folio and all; it is read
            # here as the table branch reads a row, so the nav is not left
            # to hunt titles it was handed the pages for.
            pending: list = []

            def flush():
                for t in _fold_contents_lines(pending):
                    entries.append(TocEntry(t, None, "", 2))
                pending.clear()

            for el in root.iter("p", "h1", "h2", "h3", "h4", "li"):
                raw_t = " ".join(_line_text(el).split())
                led = re.match(r"^(.*?\S)\s*[.·…]{2,}[\s.·…]*(\d{1,3}|[ivxlcdm]{1,7})$",
                               raw_t, re.I)
                t = re.sub(r"[\s.·…]+$", "", raw_t)
                # A folio is carried this far, and no further: it is the only
                # thing that can finish a title broken before its year range,
                # and the fold drops it when it finishes nothing.
                if not (t and len(t) <= 120):
                    continue
                if led and re.search(r"[A-Za-z]{3}", led.group(1)) \
                        and "contents" not in led.group(1).lower():
                    flush()
                    title = re.sub(r"[\s.·…]+$", "", led.group(1))
                    folio_text = led.group(2)
                    folio = _contents_folio(folio_text)
                    entries.append(TocEntry(title, folio, folio_text, 2, False, 0, page_i))
                    continue
                if not _FOLIO_ONLY.match(t) and not (
                        3 <= len(t) and re.search(r"[A-Za-z]{3}", t)):
                    continue
                if "contents" in t.lower():
                    continue
                if book_title and _same_head(t, book_title):
                    continue
                pending.append(t)
            flush()
            continue
        for el in root.iter():
            if el.tag in ("h1", "h2", "h3", "h4"):
                t = " ".join((el.text_content() or "").split())
                # A heading on a contents page is a part division — or it
                # is the book announcing itself again. A contents running
                # onto a second page reprints its own title at the head of
                # it, and taken as a division that title adopted every
                # chapter below: chapters 5 and 6, the notes and the index
                # all hung under "AFTER QUEER THEORY" in one book's nav.
                # The test is that identity, not a vocabulary of division
                # words — one book's parts are headed "HOW THE NORTHWEST
                # WAS LOST TO FRANCE", which names no division and is one.
                if t and "contents" not in t.lower() \
                        and not (book_title and _same_head(t, book_title)):
                    entries.append(TocEntry(t, None, "", 0 if el.tag in ("h1", "h2", "h3") else 1))
            elif el.tag == "tr":
                tds = list(el.iter("td", "th"))
                # A cell holding a title set over two lines carries a <br/>,
                # and text_content() closes it up: "Migrant Communities" and
                # "in the United Kingdom" arrived as "Communitiesin", which
                # then matched no heading in the book. The tail carries the
                # space so the leading whitespace an indent is read from is
                # left alone.
                for br in el.iter("br"):
                    br.tail = " " + (br.tail or "")
                raw = [td.text_content() or "" for td in tds]
                cells = [" ".join(r.split()) for r in raw]
                # the title is every cell but the folio; its own typography
                # is the book's statement about rank
                bold = any(len(td.findall(".//b")) or len(td.findall(".//strong"))
                           for td in tds[:-1] or tds)
                indent = next((len(r) - len(r.lstrip())
                               for r, c in zip(raw, cells) if c), 0)
                cells = [c for c in cells if c]
                if not cells:
                    continue
                folio_text = cells[-1]
                title = " ".join(cells[:-1]) if len(cells) > 1 else ""
                # printed dot leaders survive OCR as trailing runs of dots
                title = re.sub(r"[\s.·…]+$", "", title)
                m = re.fullmatch(r"(\d{1,3})", folio_text)
                cm = re.fullmatch(r"(\d{1,2})-(\d{1,3})", folio_text)
                # the appendix form the furniture reader already parses:
                # "A-12" encodes exactly as folios_from_furniture encodes
                # it, so folio_resolver can match the two (B4)
                lm = re.fullmatch(r"([A-Z])-([1-9]\d{0,2})", folio_text)
                roman = re.fullmatch(r"[ivxlc]+", folio_text.lower())
                if lm and title:
                    entries.append(TocEntry(
                        title,
                        (100 + ord(lm.group(1)) - ord("A")) * 1000
                        + int(lm.group(2)), folio_text, 2, bold, indent, page_i))
                elif cm and title:
                    entries.append(TocEntry(title, int(cm.group(1)) * 1000
                                            + int(cm.group(2)), folio_text, 2,
                                            bold, indent, page_i))
                elif m and title:
                    entries.append(TocEntry(title, int(m.group(1)), folio_text,
                                            2, bold, indent, page_i))
                elif roman and title:
                    entries.append(TocEntry(title, _contents_folio(folio_text),
                                            folio_text, 2, bold, indent, page_i))
                else:
                    # A row with no folio at all is usually noise — but a
                    # part heading is printed exactly this way: eagle's
                    # contents groups its chapters under "BOOK SEVEN: The
                    # Revolt" lines that carry no page number, and dropping
                    # them flattened the book's own declared structure.
                    # Length was standing in for structure. It fits a banner
                    # named "BOOK SEVEN: The Revolt" and not one named for
                    # everything it contains — "Part I: YUGOSLAVIA, GREECE,
                    # POLAND AND LATVIA – Between the blocs" runs eleven
                    # words and was thrown away, taking the book's four
                    # parts with it. A line that the rows beneath it are
                    # indented under is a grouping however long its name,
                    # and that is the same evidence the depth rule reads.
                    # A chapter that puts its number in a cell of its own
                    # and prints no page against it. The number is not a
                    # folio and the title is not a page, so the row parsed
                    # as neither and was dropped whole — three chapters of
                    # one book, whose other two put the number and the title
                    # in a single cell and came through. Rejoining them
                    # gives every chapter the shape the survivors had.
                    if (len(cells) == 2
                            and re.fullmatch(r"\d{1,3}|[ivxlcdm]{1,7}",
                                             cells[0], re.I)
                            and re.search(r"[A-Za-z]{3}", cells[1])):
                        entries.append(TocEntry(f"{cells[0]} {cells[1]}", None,
                                                "", 2, bold, indent, page_i))
                        continue
                    full = re.sub(r"[\s.·…]+$", "", " ".join(cells))
                    words = len(full.split())
                    mark = PART_HEAD.match(full)
                    if mark and words >= 2 and (
                            words <= 8
                            or _leads_indented_rows(el, indent)
                            or _peer_banners(el, mark.group(1).lower())):
                        entries.append(TocEntry(full, None, "", 0))
                    elif (len(cells) == 1 and len(raw) >= 2
                            and subordinate_runs
                            and not raw[0].strip()
                            and 3 <= len(full) <= 120
                            and re.search(r"[A-Za-z]{3}", full)
                            and not _FOLIO_ONLY.match(full)
                            and not _COLUMN_HEADING.match(full)):
                        # A subsection printed with no page against it. What
                        # says so is the EMPTY cell in front of it: this book
                        # sets a chapter as number-cell plus title-cell, and
                        # everything under that chapter with the number cell
                        # left blank. The row parsed as neither a title nor a
                        # folio and was dropped, taking every subsection of
                        # three chapters with it.
                        #
                        # The empty leading cell is the whole guard. Keeping
                        # any lone folio-less cell instead pulled in column
                        # headings ("Page", "CONTENTS") and, from one book,
                        # its copyright page — LIBRARY OF CONGRESS, Two
                        # Copies Received, JAN 27 1908.
                        entries.append(TocEntry(full, None, "", 2, bold,
                                                indent, page_i))
    # The prefix rule first: where a book numbers its subsections (1)…(6)
    # that is an explicit declaration and outranks an inference from type.
    # Where it does not, the page's own setting is the declaration.
    # Indentation is relative to the page it was set on. One manual sets
    # its early chapters at four spaces with their sections at eight, and
    # then, pages later, sets chapters flush left — so a baseline taken
    # across the whole contents made the flush chapters heads and demoted
    # the indented ones to sit beside their own sections. Each page's
    # setting is measured against itself.
    # Only rows carry a measured indent. A heading on the page — a part
    # division, or the book's own name — is not set in the table and
    # arrives at zero by default, which would make it the page's baseline
    # and every real row look indented beneath it.
    by_page: dict = {}
    for e in entries:
        if e.depth == 2:
            by_page.setdefault(e.src_page, []).append(e.indent)
    # Per page, and by the same test: the floor is the shallowest indent
    # that several rows share. A single flush-left row — the manual's own
    # name above its chapters — is an outlier, not the page's baseline.
    from collections import Counter as _C
    floor = {}
    for pg, v in by_page.items():
        tally = _C(v)
        real = [ind for ind, n in tally.items() if n >= 2]
        floor[pg] = min(real) if real else min(v)
    entries = [e._replace(indent=e.indent - floor.get(e.src_page, 0))
               for e in entries]

    ranked = _lift_trailing_apparatus(_infer_row_depths(entries))
    if ranked == entries:            # the prefix rule found nothing to read
        ranked = _lift_trailing_apparatus(_depths_from_typography(ranked))
    return _close_group_on_outdent(ranked), toc_pages


def _close_group_on_outdent(entries: "list[TocEntry]") -> "list[TocEntry]":
    """End a part when the contents stops setting lines inside it.

    A row below a part banner belongs to that part, which is right until the
    book finishes with its parts and goes back to speaking for itself. One
    book's Conclusion is set flush with its Introduction while every chapter
    inside a part is indented under it — so once the parts were read, the
    Conclusion, being merely the next line after the last of them, was filed
    inside Part IV.

    A line cannot belong to a group whose members are set further in than it
    is. That is the whole test: if anything between the banner and this row
    is indented deeper than this row, the group has closed.
    """
    out = list(entries)
    open_group = False
    inside = None                # how far in this group's members are set
    for i, e in enumerate(entries):
        if e.depth == 0:
            open_group, inside = True, None
            continue
        if not open_group:
            continue
        if inside is None:
            inside = e.indent
        elif e.indent < inside:
            out[i] = e._replace(depth=0)
            inside = None
    return out


def _lift_trailing_apparatus(entries: "list[TocEntry]") -> "list[TocEntry]":
    """The book's back matter belongs to the book, not to its last chapter.

    A contents ends with its apparatus — notes, bibliography, index,
    glossary, the author's note — set at the same indent as a chapter's
    subsections. Nested by that indent alone, all of it disappeared inside
    whichever chapter happened to come last: fourteen such entries across
    seven books, including one where the index sat under a chapter it has
    nothing to do with.

    Position is what separates the two cases, not the words. A References
    line BETWEEN two chapters is that chapter's, and stays where it is;
    only the run at the very END is the book's own, and it is lifted to
    the top level where the book prints it.
    """
    # Only where the contents has a hierarchy to be buried in. A flat
    # list has no last chapter to fall into, and lifting its tail would
    # invent the structure this exists to correct.
    if not any(e.depth < 2 for e in entries):
        return entries
    # Lift to whatever rank this book's top level IS. A book divided into
    # parts puts them at 0 and its chapters at 1, so lifting the back
    # matter to 1 made it a sibling of the CHAPTERS — which the nav then
    # nests inside the last part, exactly the burial this is meant to
    # undo. Four books came out unchanged for that reason.
    top = min(e.depth for e in entries)
    out = list(entries)
    for i in range(len(out) - 1, -1, -1):
        if out[i].depth != 2 or not APPARATUS_ROW(out[i].title):
            break
        out[i] = out[i]._replace(depth=top)
    return out


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
        if nxt is None or not sub.match(entries[nxt].title):
            continue
        # the subsection must FOLLOW this chapter, not open the next BOOK:
        # an intervening lower-depth heading between the two rows means
        # the (1) belongs to someone else, and promoting across it makes
        # exactly the empty grouping this pass exists to avoid (B3)
        if any(e.depth < 2 for e in entries[i + 1:nxt]):
            continue
        if NOT_A_DIVISION.match(entries[i].title):
            continue
        out[i] = entries[i]._replace(depth=1)
    return out


def _depths_from_typography(entries: "list[TocEntry]") -> "list[TocEntry]":
    """The other way a contents page declares its hierarchy: by setting it.

    Prefixes are one convention. The commoner one is typographic — the
    chapter is bold, and what belongs under it is indented — and both cues
    survive recognition into the markup, where the parser was discarding
    them. One book listed forty entries this way and shipped every one of
    them flat, chapters level with their own subsections.

    Bold leads, because it is the sturdier of the two: a contents that runs
    across two pages can lose its indentation on the second (measured: cells
    at indent 0 throughout) while the bold still marks every chapter.
    Indentation is read only where nothing is bold.

    The guards are the prefix rule's, for the same reasons: three of each
    kind at least, so a uniformly bold contents stays flat; and a row is
    promoted only where a subsection actually follows it, so no chapter
    gains an empty grouping.
    """
    rows = [i for i, e in enumerate(entries) if e.depth == 2]
    if len(rows) < 6:
        return entries
    heads = [i for i in rows if entries[i].bold]
    if len(heads) < 3 or len(heads) == len(rows):
        # Nothing bold, or everything: fall back to the indentation, which
        # says the same thing when the compositor used it instead.
        # The baseline is the shallowest indent that is a LEVEL rather than
        # an outlier. One manual prints its own name as a single flush-left
        # row above chapters set at four spaces; taking the plain minimum
        # made that one row the baseline, left every chapter looking
        # indented beneath it, and wrapped the book's nav in its own title.
        # Taking the commonest indent instead fails the other way, since
        # subsections usually outnumber chapters. A level needs members.
        from collections import Counter
        tally = Counter(entries[i].indent for i in rows)
        levels = [ind for ind, n in tally.items() if n >= 3]
        if not levels:
            return entries
        base = min(levels)
        heads = [i for i in rows if entries[i].indent == base]
        if len(heads) < 3 or len(heads) == len(rows):
            return entries
        subs = {i for i in rows if i not in heads}
    else:
        subs = {i for i in rows if not entries[i].bold}
    # The apparatus is not a subsection. Where a book lists chapters and no
    # subsections at all, its contents is bold chapters and a plain tail of
    # Notes, References and Index — and the last chapter, being the only one
    # a plain row follows, adopted all three. Three books flattened that way
    # into "7 Conclusion" parenting the back matter. A row that names the
    # apparatus is evidence of nothing.
    subs = {i for i in subs if not APPARATUS_ROW(entries[i].title)}
    if len(subs) < 3:
        return entries
    # Every row at the head level is a head. Requiring a subsection to
    # follow each one seemed safe — it stops an empty grouping — but rank
    # is a property of the level, not of what happens to come next: one
    # book's Chapter Five is set exactly like its Chapter Four and was
    # left inside it, because the only row following Five was its
    # References, which the apparatus test had already set aside. A
    # chapter with nothing under it is a top-level entry, which is what
    # the book prints. The guard that matters is the one above: unless
    # this contents really has subsections, none of this runs at all.
    out = list(entries)
    for i in rows:
        if i not in heads or NOT_A_DIVISION.match(entries[i].title):
            continue
        out[i] = entries[i]._replace(depth=1)
    return out


def _contents_folio(text: str) -> "int | None":
    """A contents line's printed folio, in the encoding the folio audit uses.

    Roman included. Front matter paginates in Roman and the audit already
    reads Roman off a running head, carrying it as i - ROMAN_FOLIO_BASE so
    the whole numbering space stays ordered. The contents parser recognised
    the same numerals and threw them away — "FIGURES ..... iv" arrived with
    no address at all — and a line with no address falls through to the
    title hunt, which only moves FORWARD. One manual's list of figures was
    placed 233 pages past the page its own contents named, and because the
    hunt's floor moves with every placement, every line after it inherited
    that floor: four front-matter entries and a notes section lost.

    _roman_to_int is strict and stops at 99, which is what front matter is.
    """
    t = text.strip()
    if t.isdigit():
        # Three digits at most, the same bar parse_folio holds a running
        # head to: a book prints no page 1997, and a contents line that
        # ends in a year would otherwise be given one as an address.
        return int(t) if 1 <= len(t) <= 3 and int(t) else None
    if re.fullmatch(r"[ivxlc]+", t, re.I):
        r = _roman_to_int(t)
        if r is not None:
            return r - ROMAN_FOLIO_BASE
    return None


def _folio_series(n: int) -> int:
    """Which numbering a folio belongs to: Roman front matter, plain Arabic,
    or one of the compound forms. Interpolating across two of them is
    meaningless arithmetic — the distance from folio iv to folio 7 is not a
    number of pages — so the resolver only ever reasons within one."""
    return -1 if n < 0 else n // 1000


def folio_resolver(folios: "list[Folio]") -> "callable":
    """
    Printed folio → position in the book, exact where read, pinned where not.

    Chapter openings print no running head, so the very pages a contents line
    points at are the ones with no direct reading. Their neighbours hold them
    in place: with hundreds of confident anchors, the page carrying folio F
    sits a known distance from the nearest anchor on each side, and both sides
    must agree before the answer is trusted.
    """
    by_series: dict = {}
    for f in folios:
        if f.confident and f.number is not None:
            by_series.setdefault(_folio_series(f.number), []).append(
                (f.number, f.index))
    for v in by_series.values():
        v.sort()

    def resolve(folio: int) -> int | None:
        anchors = by_series.get(_folio_series(folio), [])
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
                       dropped: "set[int]" = frozenset(),
                       book_title: "str | None" = None):
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
    printed, toc_pages = parse_printed_toc(bodies, book_title=book_title)
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

# The fractions Unicode sets as one character. Any other is set the way
# a compositor without the sort would: raised numerator, fraction slash,
# lowered denominator.
VULGAR = {"1/2": "½", "1/3": "⅓", "2/3": "⅔", "1/4": "¼", "3/4": "¾",
          "1/5": "⅕", "2/5": "⅖", "3/5": "⅗", "4/5": "⅘", "1/6": "⅙",
          "5/6": "⅚", "1/7": "⅐", "1/8": "⅛", "3/8": "⅜", "5/8": "⅝",
          "7/8": "⅞", "1/9": "⅑", "1/10": "⅒"}
_RAISED = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
_LOWERED = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_FRACTION = re.compile(r"\\[dt]?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_MATH_FORMS = {"text", "mathrm", "mathit", "mathbf", "frac", "dfrac", "tfrac",
               "circ"}


def _set_fraction(m) -> str:
    over, under = m.group(1).strip(), m.group(2).strip()
    if re.fullmatch(r"[0-9]+", over) and re.fullmatch(r"[0-9]+", under):
        return (VULGAR.get(f"{over}/{under}")
                or over.translate(_RAISED) + "⁄" + under.translate(_LOWERED))

    def term(t):        # "100 cc./kg." over 10 is (100 cc./kg.)/10
        if not re.search(r"[\s+\-−×·/]", t):
            return t
        depth = 0           # already one bracketed group: leave it
        for k, ch in enumerate(t):
            depth += (ch == "(") - (ch == ")")
            if depth == 0 and k < len(t) - 1:
                return f"({t})"
        return t if t.startswith("(") else f"({t})"
    return f"{term(over)}/{term(under)}"


def _math_as_text(t: str) -> str:
    """What the recogniser wrote as TeX, as the letters and signs it
    stands for: Greek by name, a fraction as one, a degree sign, and every
    other command and brace taken away."""
    def name(m):
        word = m.group(1)
        if word in _MATH_FORMS:
            return m.group(0)       # has arguments: set below
        return GREEK.get(word) or MATH_SIGNS.get(word) or word

    # Names first, while each still ends where the recogniser ended it:
    # \mu\text{g} is μ and then g, and unwrapped first it is \mug.
    t = re.sub(r"\\([A-Za-z]+)", name, t)
    t = re.sub(r"\\(?:text|mathrm|mathit|mathbf)\{([^}]*)\}", r"\1", t)
    while True:     # innermost first, so a fraction over a fraction sets
        t, n = _FRACTION.subn(_set_fraction, t)
        if not n:
            break
    t = re.sub(r"\^\s*(?:\{\s*\\circ\s*\}|\\circ)", "°", t)
    t = t.replace("\\,", ",").replace("\\", "")
    return t


_SUPERSCRIPT = str.maketrans("0123456789+-−=()ni", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁻⁼⁽⁾ⁿⁱ")
_SCRIPT = re.compile(r"([_^])\s*(?:\{([^{}]*)\}|([^\s{}_^]))")


def _scripts_as_html(t: str) -> str:
    """TeX's subscripts and powers, in what is otherwise text, as HTML."""
    def one(m):
        inner = m.group(2) if m.group(2) is not None else m.group(3)
        inner = inner.strip()
        if not inner:
            return ""
        if m.group(1) == "_":
            return f"<sub>{inner}</sub>"
        if all(ch in "0123456789+-−=()ni" for ch in inner):
            return inner.translate(_SUPERSCRIPT)
        return f'<sup class="power">{inner}</sup>'
    return _SCRIPT.sub(one, t)


def normalize_math(body: str, scripts: bool = False) -> str:
    """
    LaTeX out, letters in.

    The recogniser wraps Greek in <math>\\alpha</math>, and no EPUB reader
    renders LaTeX: the reading experience is a raw backslash-word in the middle
    of a chemical name, over a thousand times across one book's chemistry half.
    The content is only ever letters and simple wrappers, so it translates
    exactly; anything unrecognised keeps its text with the markup stripped,
    which is never worse than what the reader would have shown.

    It wraps a printed fraction the same way — 7<math>\\frac{1}{2}</math>
    inches — in 25 of 151 books, and stripping that left "7frac{1}{2} inches"
    on the page. A fraction, a degree sign and the common operators are set
    as the characters they are.

    With `scripts`, a subscript or a power too, which are otherwise left as
    written ("P_1", "C_4H_{10}N^+"). A subscript is set as <sub>. A power is
    set as superscript characters where each of its characters has one (n²,
    C₄H₁₀N⁺), and otherwise as <sup class="power">: a bare <sup> holding a
    figure is what every note linker here takes for a call, and a power is
    not one. The caller asks for it only on a page with no born-digital
    text (see build_epub).
    """
    def clean(m):
        t = _math_as_text(m.group(1))
        return _scripts_as_html(t) if scripts else t
    # Display math is exempt: an equation set on its own line is not a Greek
    # letter in a chemical name, and its structure is all it has. The
    # recogniser marks it display="block" itself, on 410 of the 419 blocks
    # the layout labels Equation in the shelf's caches. The label is not
    # consulted: the eight it would add are expressions run into a sentence
    # ("In perfect competition, MRP = MP \times P"), which read better set
    # as text. (The assembly loop used to stamp blocks labelled "Formula",
    # a label Surya does not have, so it never stamped one.)
    return re.sub(r"<math(?![^>]*display=\"block\")[^>]*>(.*?)</math>",
                  clean, body, flags=re.S)


# A raised 2 or 3 standing directly on a unit of length is a power. The
# recogniser writes "60,000 km<sup>2</sup>" exactly as it writes a note
# call, and where the chapter had a note 2 the linker bound the unit to
# it: ten square kilometres on the shelf pointed at somebody's citation,
# and Small is Necessary's list of abbreviations ("ft² square feet")
# would have. These five are not words, so nothing else can be meant; "in"
# is one, and a raised 1 on "ft" in a table was a note mark, so neither is
# taken. Set as the character, it is what the page prints and nothing a
# linker looks at. (It is set once the page's own text layer has been
# deferred to: see where build_epub calls this.)
_UNIT_POWER = re.compile(
    r"(?<![A-Za-z])(mm|cm|km|m|ft)<sup>\s*([23])\s*</sup>")


def set_unit_powers(body: str) -> str:
    return _UNIT_POWER.sub(
        lambda m: m.group(1) + ("²" if m.group(2) == "2" else "³"), body)


# An equation set on its own line, written as MathML. The recogniser writes
# it as TeX inside <math display="block">, which is no MathML at all: it
# has no namespace, so every reading system showed the TeX, backslashes and
# braces, 429 times in 14 books (two economics textbooks hold 288 of them).
# MathML is part of EPUB 3 itself, and a reading system that draws pages
# must draw it; pandoc, where it is installed, turns the TeX into MathML.
# What it cannot parse (a recogniser's misreading, "P } d } r = P") and
# every equation when pandoc is not there is set as its plain text inside
# the same element, which draws as text everywhere. Either way the element
# carries that text as its alttext, for a reader that speaks it.
#
# pandoc also keeps the TeX inside, as an annotation. It is left out here:
# no reader shows it, and every check this pipeline makes of a built book
# counts the words of its pages, where "frac" and "Delta" would be words.
#
# This is the last thing done to a page's text, after every decision has
# been applied, so a decisions file made against a book built before it
# still finds its words: the token stream it addresses is the TeX one.
MATHML_NS = "http://www.w3.org/1998/Math/MathML"
PANDOC_ENV = "CODICOLOGY_PANDOC"
_SEPARATOR = "CODICOLOGYEQUATIONSEPARATOR"


def _pandoc() -> "str | None":
    import shutil
    said = os.environ.get(PANDOC_ENV)
    if said:
        return said if os.access(said, os.X_OK) else None
    return shutil.which("pandoc")


def _mathml_from_pandoc(texs: "list[str]") -> "list[str | None]":
    """Each TeX equation as the inside of a MathML <math> element, or None
    where pandoc is not there or could not parse it. One pandoc for all."""
    exe = _pandoc()
    if not exe or not texs:
        return [None] * len(texs)
    import subprocess
    # A bare $ is money ("AE = $300 + 0.8Y") to the recogniser and the end
    # of the equation to pandoc, which then gives up on it: 35 of the 57
    # equations it could not parse on the shelf. As \$ it is a dollar sign.
    doc = f"\n\n{_SEPARATOR}\n\n".join(
        "$$" + re.sub(r"(?<!\\)\$", r"\\$", re.sub(r"\s+", " ", t).strip())
        + "$$" for t in texs)
    try:
        done = subprocess.run([exe, "-f", "markdown", "-t", "html5",
                               "--mathml"], input=doc, capture_output=True,
                              text=True, timeout=300)
    except (OSError, subprocess.SubprocessError):
        return [None] * len(texs)
    if done.returncode != 0:
        return [None] * len(texs)
    parts = re.split(rf"<p>\s*{_SEPARATOR}\s*</p>", done.stdout)
    if len(parts) != len(texs):
        return [None] * len(texs)       # never pair one with another's
    out = []
    for part in parts:
        m = re.search(r"<math\b[^>]*>\s*<semantics>(.*?)"
                      r"<annotation\b.*?</annotation>\s*</semantics>\s*</math>",
                      part, re.S)
        out.append(m.group(1).strip() if m else None)
    return out


def write_display_math(bodies: list) -> dict:
    """Every display equation of the book as MathML (see MATHML_NS).
    Returns how many were set by pandoc and how many as text."""
    found = [(i, m) for i, b in enumerate(bodies)
             for m in _DISPLAY_MATH.finditer(b)]
    if not found:
        return {"mathml": 0, "text": 0}
    texs = [html.unescape(m.group(1)) for _, m in found]
    made = _mathml_from_pandoc(texs)
    stats = {"mathml": 0, "text": 0}
    new = {}
    for (i, m), tex, inner in zip(found, texs, made):
        plain = re.sub(r"\s+", " ", _math_as_text(tex)).strip()
        if inner is None:
            inner = f"<mtext>{html.escape(plain, quote=False)}</mtext>"
            stats["text"] += 1
        else:
            stats["mathml"] += 1
        new.setdefault(i, []).append((m.start(), m.end(), (
            f'<math xmlns="{MATHML_NS}" display="block" '
            f'alttext="{html.escape(plain)}">{inner}</math>')))
    for i, edits in new.items():
        b = bodies[i]
        for lo, hi, text in sorted(edits, reverse=True):
            b = b[:lo] + text + b[hi:]
        bodies[i] = b
    return stats


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
# FOOTNOTES and ENDNOTES included: one book heads every chapter-end notes
# section "FOOTNOTES", and the bare word blocked all of them — the linker,
# the group walk, and the rank probe all gate on this one pattern. The word
# must still be the whole heading.
#
# And the word a publisher puts before it. Every heading on the shelf that
# says "notes" was listed on 12 November 2026, 151 books: besides the bare
# word there are "REFERENCE NOTES" (Allen's Reconstruction, whose 192
# markers all stayed plain for it) and "End Notes" written as two words.
# Everything else that says the word is a heading ABOUT something ("A Note
# on the Text", "Notes on Contributors", "Notes payable to order") and is
# still refused, because the designation must be the whole heading.
#
# "Source Notes" is refused too, though it reads like one of these. Both
# books that have it are field manuals whose source notes are a list keyed
# by page, not numbered notes, and admitted, the heading made a notes
# section of FM 3-06's back matter: a stray raised 2 in the body was bound
# to the second sense of a word in the glossary, thirty-eight pages on.
# The first whole-shelf run found it; the link probe, which skips a book
# with so few markers, had not.
_NOTES_WORDS = r"(?:(?:FOOT|END|REFERENCE)[\s-]*)?NOTES?"
NOTES_HEAD = re.compile(rf"<h\d[^>]*>\s*{_NOTES_WORDS}\s*</h\d>", re.I)
# A publisher may call the same thing by the other name: one book heads its
# endnotes "References" and then groups them PREFACE, INTRODUCTION, CHAPTER
# 1…, numbered from 1 in each — a notes section in every respect but the
# word. Nothing was found, so nothing was grouped, and all 810 of its
# markers stayed plain.
#
# It is a SEPARATE pattern because the word carries different weight in the
# two questions asked of it. "Where does the back-of-book section start?"
# may read References; "how many notes sections does this book have?" may
# not, because a repeated NOTES head means chapter endnotes while a
# repeated REFERENCES head is nearly always the running head over one
# back-of-book section. Answering the second with this pattern sent that
# same book down the chapter-endnotes path, where position alone scopes
# the markers: 190 links, and chapters five, six and eight all pointed at
# the preface's only citation.
#
# The word decides nothing beyond the start, which is what makes it safe
# there: an author-date bibliography under the same heading opens no group,
# because a group needs a chapter head, and its entries carry no numbers,
# so the ungrouped fallback refuses it for want of an ascending run. A
# section has to be shaped like notes to be read as notes.
NOTES_SECTION_HEAD = re.compile(
    rf"<h\d[^>]*>\s*(?:{_NOTES_WORDS}|REFERENCES?)\s*</h\d>", re.I)
# The names a notes section gives its groups. Chapter forms, spelled out or
# numbered — and the sections a book's front and back matter use, because a
# book that annotates its introduction gives those notes a group of their
# own. Missing that group does not merely lose it: groups pair by position,
# so an unrecognised first group shifts every one after it by one, and the
# markers then link confidently to the wrong chapter's citations.
#
# A notes section may also say whose notes they are in so many words:
# Cressey's Other People's Money heads its groups "Notes to Chapter 1",
# and none of the five was a head, so its 102 markers had five groups too
# few to pair with.
GROUP_NAMES = (r"(?:NOTES\s+(?:TO|FOR)\s+(?:THE\s+)?)?"
               r"(?:CHAPTER\s+[A-Z\-]+|CHAPTER\s+\d{1,3}"
               r"|INTRODUCTION|PREFACE|PROLOGUE|FOREWORD|FORWARD"
               r"|CONCLUSION|EPILOGUE|AFTERWORD|POSTSCRIPT)")
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
# A group's head may carry the chapter's title after its number: The War
# Correspondent heads its eight "CHAPTER 3. JOURNALISM, OBJECTIVITY AND
# WAR". GROUP_HEAD wants the designation to be the whole heading, for the
# reason given above, so not one of the eight opened a group and all 188
# markers stayed plain. Here the heading must be a HEADING (a paragraph
# that opens "Chapter 3: see…" is somebody's note), the number must be
# one the strict parsers accept, and a mark must stand between it and the
# title, which is what tells "CHAPTER 3. THE PRESS" from a heading that
# merely begins with the words.
GROUP_HEAD_TITLED = re.compile(
    r"<(h[1-6])\b[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    rf"CHAPTER\s+(\d{{1,3}}|[IVXL]{{1,7}}|{_NUMBER_WORD})(?!\w)"
    r"\s*[.:—–-]\s*\S", re.I)

# And a group may be named by where in the book its notes belong and by
# nothing else. Ste. Croix's Class Struggle in the Ancient Greek World
# numbers its notes afresh in every section of every chapter and heads each
# group with the section's designation alone: "[I.ii]", "[IV.vi]". Thirty-
# eight heads, not one of them a head. The whole heading must be the
# designation: numerals joined by full stops, in brackets or not.
_SECTION_NUMERAL = r"(?:[IVXL]{1,7}|[ivxl]{1,7}|\d{1,3})"
GROUP_HEAD_SECTION = re.compile(
    r"<(h[1-6])\b[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    rf"[\[(]?\s*{_SECTION_NUMERAL}(?:\s*\.\s*{_SECTION_NUMERAL})+\s*[\])]?"
    r"\s*(?:</[a-zA-Z]+>\s*)*</\1>")


def _notes_follow(bodies: list[str], page: int, pos: int) -> bool:
    """Whether what stands under a NOTES head is notes: a numbered entry,
    in any dress the parsers read, before the page after next begins.

    The word alone is not the thing. A field manual sets a caution under
    the heading NOTE two dozen times, and Small is Necessary opens with a
    one-page NOTE on its terms; each was counted as a notes section. The
    count decides which linker a book gets, and the first such heading is
    where the back-of-book section is taken to begin, so that one page cost
    Small is Necessary 361 links: its real Notes, two hundred pages on,
    were read as the tail of a section that had begun in the front matter.
    """
    under = bodies[page][pos:]
    if page + 1 < len(bodies):
        under += bodies[page + 1]
    return any(pat.search(under) for pat in (
        NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI, NOTE_ENTRY_BARE))


def notes_head_on(bodies: list[str], page: int, pattern=None):
    """The first NOTES head on this page that notes follow, or None."""
    for m in (pattern or NOTES_HEAD).finditer(bodies[page]):
        if _notes_follow(bodies, page, m.end()):
            return m
    return None


def count_notes_heads(bodies: list[str]) -> int:
    """How many pages open a notes section."""
    return sum(1 for i in range(len(bodies)) if notes_head_on(bodies, i))


def notes_section_start(bodies: list[str]) -> "int | None":
    """The page the back-of-book notes begin on: the first with a notes
    head (or the other name for one) that notes follow. A book none of
    whose heads is followed by anything this can read keeps its first, as
    before: the parser may still find what this glance did not."""
    first = None
    for i, b in enumerate(bodies):
        if not NOTES_SECTION_HEAD.search(b):
            continue
        if notes_head_on(bodies, i, NOTES_SECTION_HEAD):
            return i
        if first is None:
            first = i
    return first


def _bare_numbered(bodies: list[str], start: int) -> bool:
    """Whether the notes section opening at START numbers its entries bare,
    "1 Source…" — at least ten such entries, and not one in any other dress
    before the section ends. A single "1." or superscript entry says the
    book uses that form, and then a bare numeral is somebody's citation
    running on from the page before. Ten, because a notes section set this
    way runs to dozens of entries, while three bare numerals were the panel
    labels of a figure in FM 31-70's appendix, under a REFERENCES head."""
    rank = re.search(rf"<(h[1-6])[^>]*>\s*{_NOTES_WORDS}\s*</\1>",
                     bodies[start], re.I)
    stop = rf"<{rank.group(1)}[^>]*>" if rank else r"<h1[^>]*>"
    bare = 0
    for i in range(start, len(bodies)):
        if i > start and re.search(stop, bodies[i], re.I):
            break
        if any(p.search(bodies[i]) for p in (NOTE_ENTRY, NOTE_ENTRY_PLAIN,
                                              NOTE_ENTRY_LI, NOTE_ENTRY_BR)):
            return False
        bare += len(NOTE_ENTRY_BARE.findall(bodies[i]))
    return bare >= 10


def parse_notes_section(bodies: list[str]) -> tuple[int, list[list[tuple[int, int, int]]]]:
    """
    The book's endnotes, grouped as the book groups them.

    Returns (first_notes_page, groups) where each group is the notes of one
    chapter, in order, as (page_index, number, match_start). A note is a
    paragraph that OPENS with a superscript number — the position is what
    tells a citation apart from a marker in running prose, which never leads
    its paragraph.
    """
    start = notes_section_start(bodies)
    if start is None:
        return -1, []
    bare = _bare_numbered(bodies, start)
    entry_pats = (NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI) + (
        (NOTE_ENTRY_BARE,) if bare else ())
    groups: list[list[tuple[int, int, int]]] = []
    group_nos: list[int | None] = []
    current: list[tuple[int, int, int]] | None = None
    empty_streak = 0
    # The section is read first and grouped after, because one question
    # asked while grouping needs what comes next (_lists_inside_notes).
    stream: list[tuple[int, int, str, "int | None"]] = []
    seen_group = open_notes = False       # a group is open and has a note
    for i in range(start, len(bodies)):
        b = bodies[i]
        sup_notes = [(m.start(), "note", int(m.group(1))) for m in NOTE_ENTRY.finditer(b)]
        if sup_notes:
            sup_notes += _br_run_notes(b, sup=True)
        # a section that sets its notes as "1. Source…" rather than with a
        # superscript; only consulted where the superscript form is absent,
        # since "1. " also opens ordinary numbered lists in prose
        plain_notes = [] if sup_notes else \
            [(m.start(), "note", int(m.group(1)))
             for pat in entry_pats
             for m in pat.finditer(b)] + _br_run_notes(b, bare=bare)
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
        for m in GROUP_HEAD_TITLED.finditer(b):
            no = _chapter_head_number(m.group(2))
            if no is not None:
                heads.append((m.start(), "group", no))
        for m in GROUP_HEAD_SECTION.finditer(b):
            heads.append((m.start(), "group", None))
        events = sorted(heads + sup_notes + plain_notes, key=lambda e: e[0])
        if i > start and not events and open_notes:
            # one event-less page may be a blank leaf inside the notes; two
            # in a row mean the notes have ended and this is the index
            empty_streak += 1
            if empty_streak >= 2:
                break
            continue
        empty_streak = 0
        for pos, kind, n in events:
            if kind == "group":
                seen_group, open_notes = True, False
            elif seen_group:
                open_notes = True
            stream.append((i, pos, kind, n))
    skip = _lists_inside_notes(stream)
    for at, (i, pos, kind, n) in enumerate(stream):
        if at in skip:
            continue
        if kind == "group":
            current = []
            groups.append(current)
            group_nos.append(n)
        elif current is not None:
            if n == 1 and current and current[-1][1] > 1:
                # The numbering came back to 1. A book restarts its
                # notes at 1 only where a new chapter's notes begin, so
                # the section's own sequence declares the boundary — a
                # structural fact no typographic pattern can supply, and
                # the only evidence available when a head is dressed in
                # a way the patterns miss. Gendzier's Development
                # Against Democracy heads all eight of its groups and
                # four are invisible here: three wrap the number in
                # <i>, and chapter four's is a centred paragraph with a
                # <br/> inside it. Eight groups parsed as four, too far
                # from the body's eight to pair, and all 705 markers
                # went unlinked.
                #
                # A RETURN TO 1 specifically, not any number that fails
                # to climb: a repeated or misread number is a damaged
                # entry, and splitting on one would invent a group and
                # shift every pairing after it. The numbering must also
                # have climbed above 1 first, so a group whose own
                # first entries read 1, 1 stays whole.
                current = []
                groups.append(current)
                group_nos.append(None)
            current.append((i, n, pos))
    keep = [(no, g) for no, g in zip(group_nos, groups) if g]
    if not keep:
        solo = _ungrouped_notes(bodies, start)
        if solo:
            parse_notes_section.last_group_numbers = [None]
            return start, [solo]
    parse_notes_section.last_group_numbers = [no for no, _ in keep]
    return start, [g for _, g in keep]


def _lists_inside_notes(stream: list) -> set:
    """Which entries of a notes section are a numbered list inside a note,
    as positions in `stream` ((page, pos, kind, number), in reading order).

    A return to 1 is how a notes section says a new chapter's notes begin
    where its head went unread (see parse_notes_section). But a note may
    itself hold a numbered list, and that returns to 1 as well: KGB: Masters
    of the Soviet Union has one "1." inside note 15 of a chapter, and the
    phantom group it opened put every later group one out against the
    body's, so that 643 markers, the whole book's, were refused.

    What tells the two apart is what comes after. A chapter's notes go on
    from 1; a list inside note 15 is followed by note 16. So a run that
    climbs from 1 by ones, begun where the numbers had passed 1 and with
    no head before it, is a list inside a note exactly when the entry
    after it is the next number of the notes it interrupted.
    """
    skip: set = set()
    prev = None                 # the last note of the group being read
    at = 0
    while at < len(stream):
        _, _, kind, n = stream[at]
        if kind == "group":
            prev = None
            at += 1
            continue
        if n == 1 and prev is not None and prev > 1:
            end = at + 1
            while (end < len(stream) and stream[end][2] == "note"
                   and stream[end][3] == stream[end - 1][3] + 1):
                end += 1
            after = stream[end] if end < len(stream) else None
            if after and after[2] == "note" and after[3] == prev + 1:
                skip.update(range(at, end))
                at = end
                continue
        prev = n
        at += 1
    return skip


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
    rank = re.search(rf"<(h[1-6])[^>]*>\s*{_NOTES_WORDS}\s*</\1>",
                     bodies[start], re.I)
    stop = rf"<{rank.group(1)}[^>]*>" if rank else r"<h1[^>]*>"
    bare = _bare_numbered(bodies, start)
    entry_pats = (NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI) + (
        (NOTE_ENTRY_BARE,) if bare else ())
    entries: list[tuple[int, int, int]] = []
    for i in range(start, len(bodies)):
        if i > start and re.search(stop, bodies[i], re.I):
            break
        sup = [(m.start(), int(m.group(1))) for m in NOTE_ENTRY.finditer(bodies[i])]
        if sup:
            sup += [(pos, n) for pos, _kind, n in
                    _br_run_notes(bodies[i], sup=True)]
        plain = [] if sup else [(m.start(), int(m.group(1)))
                                for pat in entry_pats
                                for m in pat.finditer(bodies[i])] \
            + [(pos, n) for pos, _kind, n in _br_run_notes(bodies[i],
                                                            bare=bare)]
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
    # A bare-numbered entry (NOTE_ENTRY_BARE) keeps its own setting: no full
    # stop is written that the book never printed.
    bare = re.match(r"<(p|li)[^>]*>\s*(\d{1,3})(?=\s)", entry)
    if bare is not None and int(bare.group(2)) == n:
        return bare.end(), (f'<{bare.group(1)} id="{note_id}">'
                            f'<a href="{back_href}">{n}</a>')
    # An entry inside a <br/>-divided run has no block of its own to carry
    # the id, so the number itself becomes the anchor. The break and the
    # space after it are reproduced exactly: they are the run's typesetting,
    # and swallowing them would run two notes together on one line.
    bm = re.match(r"(<br\s*/?>)(\s*)(\d{1,3})\.", entry)
    if bm is not None:
        return bm.end(), (f'{bm.group(1)}{bm.group(2)}'
                          f'<a id="{note_id}" href="{back_href}">{n}.</a>')
    bb = re.match(r"(<br\s*/?>)(\s*)(\d{1,3})(?=\s)", entry)
    if bb is not None and int(bb.group(3)) == n:
        return bb.end(), (f'{bb.group(1)}{bb.group(2)}'
                          f'<a id="{note_id}" href="{back_href}">{n}</a>')
    sm = re.match(r"(<br\s*/?>)(\s*)<sup>\s*(\d{1,3})\s*</sup>", entry)
    if sm is not None:
        # the superscript is the book's own typesetting and stays; only the
        # number inside it becomes the anchor, exactly as a first entry's does
        return sm.end(), (f'{sm.group(1)}{sm.group(2)}<sup>'
                          f'<a id="{note_id}" href="{back_href}">{n}</a></sup>')
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
    heads = [(i, m.start()) for i in range(len(bodies))
             for m in [notes_head_on(bodies, i)] if m]
    # A section whose display heading the OCR dropped still announces
    # itself: a bare chapter designation followed at once by entries
    # numbered from one. One book lost a single FOOTNOTES head that way,
    # and the cost was double — that chapter's high markers went unlinked,
    # and its low markers bound BY POSITION to the next chapter's entries,
    # confidently and wrongly. Three ascending entries are required, the
    # same floor the single-section walk uses, so a chapter OPENING page
    # ("Chapter III" over prose) can never qualify.
    head_pages = {i for i, _ in heads}
    for i, b in enumerate(bodies):
        if i in head_pages:
            continue
        for gm in GROUP_HEAD.finditer(b):
            tail = b[gm.end():gm.end() + 400]
            ns = [n for _, n in sorted(
                (m.start(), int(m.group(1)))
                for pat in (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
                for m in pat.finditer(tail))]
            if ns[:3] == [1, 2, 3]:
                heads.append((i, gm.start()))
                break
    heads.sort()
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
            # all three entry forms gathered FIRST, then walked in page
            # order with one ascending counter — three sequential passes
            # shared the counter and a page whose entries alternate forms
            # (sup 1, plain 2, sup 3) permanently skipped the later sup
            matches = sorted(
                ((m.start(), int(m.group(1)))
                 for pat in (NOTE_ENTRY, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI)
                 for m in pat.finditer(bodies[pi])
                 if not (pi == hp and m.start() < from_pos)),
                key=lambda t: t[0])
            for pos, n in matches:
                if n == expect:
                    entries.setdefault(n, (pi, pos))
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


def find_body_marker_groups(bodies: list[str], stop: int,
                            climbed: int = 1, headless: bool = False
                            ) -> list[list[tuple[int, int, int]]]:
    """
    Superscript markers in the prose, split into chapters by their own resets.

    Note numbering restarts at 1 every chapter, so the numbers themselves mark
    the chapter boundaries: a marker at or below its predecessor after an
    ascending run means a new chapter has begun. No table of contents is
    consulted — the markers carry their own structure.

    `climbed` is how far the run before a 1 must have got for that 1 to open
    a chapter, and no single value of it is right for every book. Two 1s in
    a row are a chapter with one note followed by the next chapter, or one
    chapter citing its first note twice, and nothing in the body tells them
    apart — the marker after both readings is 2. So the caller tries both
    and lets the notes section, which knows how many chapters it has,
    decide; see `_best_body_groups`.

    `headless` also opens a chapter where the numbers fall to 2 or 3 and
    climb from there, with no 1 before them: a chapter whose first call was
    not read as one. Burning Up's tenth chapter lost its 1, so the run went
    "…44, 45, 2, 3, 4" and was read as chapter nine citing its early notes
    again: forty-three calls opened the ninth chapter's notes, and every
    chapter after it was paired one group out, 104 calls in all. A
    re-citation falls and comes straight back ("…16, 17, 2, 18"); this
    falls and stays down for two more calls. It is one more reading for the
    notes section to choose between, not a rule.
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
        # What tells a chapter boundary from an author citing note 1 again
        # is the marker AFTER it: a new chapter goes on to 2, a re-citation
        # returns to wherever the chapter had got to. Requiring the run
        # before it to have reached 3 as well was a second guard on the
        # same thing, and it cost more than it caught — a chapter with one
        # or two notes never reaches 3, so its markers were absorbed by the
        # next chapter and the body came up short of the notes. Measured
        # over every built book, dropping it links 365 markers that went
        # plain — one book from none to 296 — puts not one link in the
        # wrong chapter, and costs a single marker in a single book.
        if n == 1 and prev >= climbed and (nxt is None or nxt <= 3):
            if current:
                groups.append(current)
            current = []
        elif headless and 1 < n <= 3 and prev >= n + 5 \
                and k + 2 < len(stream) \
                and n < nxt < stream[k + 2][1] <= n + 4:
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
# A book may hang its note numbers in a column with no full stop at all:
# "1  James Joyce, Ulysses…". How the West Came to Rule does, as a list on
# its first notes page and as paragraphs after that, and every pattern here
# wanted "1." or a superscript — its notes section parsed empty and all
# 1,667 markers went plain. A bare numeral opening a paragraph is too
# ordinary to trust everywhere (a citation continued from the page before can
# open with a page number), so this form is read only in a notes section
# that shows no other: see _bare_numbered.
NOTE_ENTRY_BARE = re.compile(
    r"<(?:p|li)[^>]*>\s*(\d{1,3})\s+(?=[^\W\d_]|[‘“'\"(\[<])")
# A run of notes may be set as ONE paragraph, its entries divided by <br/>
# rather than by blocks. Every pattern above anchors the number to a block's
# opening, so only the run's first line is seen and the group's highest note
# stops there — every marker past it has nothing to point at. Two such runs,
# both at a chapter's tail, cost one book 34 links.
NOTE_ENTRY_BR = re.compile(r"<br\s*/?>\s*(\d{1,3})\.\s+\S")
# The same run set the other way: the number in a superscript and no period
# after it, which is how a book that superscripts its entry numbers renders a
# column the layout pass failed to split. Eagle Forgotten hides 67 entries in
# four such paragraphs, and 69 of its markers went plain for want of them.
NOTE_ENTRY_BR_SUP = re.compile(r"<br\s*/?>\s*<sup>\s*(\d{1,3})\s*</sup>")
# And the bare dress: How the West Came to Rule lost notes 60 to 87 into one
# paragraph, "<p>59 Shaw,<br/>60 ĪnalcIk,<br/>61 Coles,…".
NOTE_ENTRY_BR_BARE = re.compile(
    r"<br\s*/?>\s*(\d{1,3})\s+(?=[^\W\d_]|[‘“'\"(\[<])")


def _br_run_notes(body: str, sup: bool = False, bare: bool = False) -> list:
    """The note entries a paragraph hides behind its <br/>s.

    Returned as (position, "note", number), the position being the <br/>
    that opens each — which is where the anchor step rewrites.

    `sup` picks the rendering. A book that numbers its entries "1. Source…"
    continues a run as "<br/>2. Source…"; a book that superscripts them
    writes "<sup>2</sup>Source…" with no period. `bare` is the third dress,
    "<br/>2 Source…", read only in a section _bare_numbered says is set that
    way. All are the same column of notes with the same break between them,
    and the opening of the paragraph says which one this book uses — so they
    are never mixed, and a run is only ever read in the dress its own first
    entry wears.

    The guard against reading an ordinary line break this way is the
    numbering. The paragraph must already BE a note entry, so a block of
    running prose is never examined, and the numbers must climb from the
    one the paragraph opens with. A note that merely carries a <br/> before
    a numeral — a date, an address, a line of verse, a table of figures —
    does not climb, and then none of its lines are taken. Refusing the
    whole paragraph rather than its first few lines is deliberate: a run
    read half-way would end its group early, which is the very failure
    this exists to repair.
    """
    head, tail = ((NOTE_ENTRY, NOTE_ENTRY_BR_SUP) if sup
                  else (NOTE_ENTRY_BARE, NOTE_ENTRY_BR_BARE) if bare
                  else (NOTE_ENTRY_PLAIN, NOTE_ENTRY_BR))
    out = []
    for block in re.finditer(r"<p\b[^>]*>.*?</p>", body, re.S):
        opening = head.match(block.group(0))
        if opening is None:
            continue
        run = [(block.start() + m.start(), int(m.group(1)))
               for m in tail.finditer(block.group(0))]
        if not run:
            continue
        nums = [int(opening.group(1))] + [n for _, n in run]
        if any(b <= a for a, b in zip(nums, nums[1:])):
            continue
        out += [(pos, "note", n) for pos, n in run]
    return out


# The period is optional: Beyond Money heads its note groups "1 capital and
# crises" where another book writes "1. Capital and crises", and requiring
# the dot cost it every one of its 410 links — no head parsed, so no group
# ever opened and every entry was discarded before it could be paired.
GROUP_HEAD_NUMBERED = re.compile(r"<h\d[^>]*>\s*(\d{1,2})\.?\s+\S")
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
# strict parser refuses anything that is merely made of those letters. Books
# that spell the number out count too — two here do, and leaving them out
# left sixty openings split between a heading holding the number and a
# heading holding the title, neither able to say what chapter it was.
# The number must end where it ends. Without the boundary, "Chapter
# Introduction" was chapter I with the title "ntroduction", and "Chapter
# Vital Signs" chapter V, "ital Signs".
CHAPTER_HEAD = re.compile(
    r"<(h[1-6])[^>]*>\s*(?:<[^/][^>]*>\s*)*"
    rf"(CHAPTER|Chapter)\s+(\d{{1,3}}|[IVXL]{{1,7}}|(?i:{_NUMBER_WORD}))"
    r"(?!\w)\s*[.:—-]?\s*(.*?)</\1>", re.S)

_WORD_VALUE = {w: v for v, w in enumerate(
    "one two three four five six seven eight nine ten eleven twelve thirteen "
    "fourteen fifteen sixteen seventeen eighteen nineteen".split(), start=1)}
_WORD_VALUE.update({w: v for v, w in zip(
    range(20, 100, 10),
    "twenty thirty forty fifty sixty seventy eighty ninety".split())})


def _chapter_head_number(token: str) -> "int | None":
    if token.isdigit():
        return int(token)
    parts = [p for p in re.split(r"[-\s]+", token.lower()) if p]
    if parts and all(p in _WORD_VALUE for p in parts):
        return sum(_WORD_VALUE[p] for p in parts)
    return _roman_to_int(token)


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
    # the next page's fit test continues from this page's HIGHEST marker,
    # not its last in reading order — a trailing backward re-citation
    # ("16, 17, 2, 18" is real in this material) would otherwise reset the
    # sequence and suppress the next page's genuine marker as a phantom
    if inline_positions:
        seq_state["last"] = max(n for _, n in inline_positions)
    return reattached, suppressed


def furniture_absent_from_contents(placed, furn_chapters, pos_of):
    """The chapters the running heads describe that the contents never gave
    the reader: no placed entry lands on the chapter's opening pages, and no
    entry's title names it.

    Title cover counts even for an entry that failed to place — the contents
    still declares that chapter, and adding a second copy from the margins
    would duplicate it rather than rescue it.
    """
    targets = {t for _, t, _ in placed if t is not None}
    # This expects the RESOLVED placements — the list nav_from_placed hands
    # back after every title hunt has settled. Judged before the hunt, a
    # wholly folio-less contents has no targets at all, and one book merged
    # all nineteen of its chapters as "missing" while its nav named every
    # one. Only entries the reader actually gets can cover: resolved
    # leaves, and groupings, which emit as headings even unresolved. An
    # unresolved leaf is skipped from the nav entirely, and counting it as
    # cover blocked the rescue in exactly the case rescue helps.
    emitted = [e for e, t, _ in placed if t is not None or e.depth < 2]

    def covered(c):
        # A head names its chapter from the page after the opening — two
        # leaves back on a verso-title book — so the window is the one the
        # heading promoter already uses, not a tighter one.
        if any(abs(t - c.start) <= 2 for t in targets):
            return True
        return any(_names_chapter(e.title, c.title)
                   or _names_chapter(c.title, e.title)
                   for e in emitted)

    return [c for c in furn_chapters
            if c.start in pos_of and not covered(c)]


def merge_missing_furniture(placed, missing, furn_total):
    """Give the nav the chapters the margins describe, where the printed
    contents plainly failed to.

    The printed contents is the book's own declaration and outranks an
    inference from its running heads — which is enforced per chapter, not
    per book: a chapter counts as missing only when no placed entry lands
    within the verso window of its opening and no entry the reader will
    actually see names it, under the same fuzzy matching the heads
    themselves are grouped by. A chapter that survives all of that is
    evidenced by a sustained run of its own margins and absent from the
    nav, and there is no number of its peers that makes leaving it out
    better for the reader. An aggregate gate stood here once — at least
    three missing and half the book's — and its only effect was to keep
    proven chapters out of thin navs.

    Merged, never replaced: everything the contents did place is kept, and
    the absent chapters are added in page order, each anchored where its
    own running heads begin. The one refusal is a title with no run of
    three letters — one manual's appendix heads are the bare letters A and
    B, real destinations but meaningless as labels.
    """
    depth = min((e.depth for e, _, _ in placed), default=2)
    out = list(placed)
    merged = 0
    # Resolution before insertion. Where an UNRESOLVED entry — leaf or
    # grouping — names the chapter the margins located, the entry is given
    # the margins' page rather than a copy of the margins' words: the
    # contents keeps its own title and its own position, the furniture
    # supplies the destination, and nothing duplicates. Inserted alongside
    # instead, one book's chapter three arrived twice — an abbreviated
    # head-link above the real grouping, with the neighbouring chapter's
    # section caught between them. The head may be cut mid-word (MEDIA
    # CONTENT, 2 is the front of a title ending 2006), so the naming test
    # here is looser than coverage's, and safely so: one direction only,
    # the head abbreviating the entry, and its worst failure is a link on
    # a heading that already stands where the book put it.
    lent: set = set()
    unresolved = [(k, e) for k, (e, t, _) in enumerate(placed) if t is None]
    for c in sorted(missing, key=lambda c: c.start):
        if not re.search(r"[A-Za-z]{3}", c.title):
            continue
        head_core = re.sub(r"[^a-z]", "", _bare_title(c.title.lower()))
        home = None
        for k, e in unresolved:
            if k in lent:
                continue
            entry_core = re.sub(r"[^a-z]", "", _bare_title(e.title.lower()))
            if (_names_chapter(e.title, c.title)
                    or _names_chapter(c.title, e.title)
                    or (len(head_core) >= 8
                        and entry_core.startswith(head_core))):
                home = k
                lent.add(k)
                break
        if home is not None:
            e, _t, ok = out[home]
            out[home] = (e, c.start, ok)
            merged += 1
            continue
        at = next((k for k, (_, t, _) in enumerate(out)
                   if t is not None and t > c.start), len(out))
        out.insert(at, (TocEntry(c.title, None, "", depth), c.start, False))
        merged += 1
    return out, merged


def contents_confidence(placed, n_pages, missed, furn_unmerged):
    """Warnings for a contents parse that deserves a human eye.

    Every bad parse found on this shelf had its signature sitting in the
    build log, printed and never compared: seven entries for a 288-page
    book, chapters four and five printed with one to three nowhere, the
    margins describing chapters the nav never mentioned. Each check here is
    a comparison the log was already in a position to make. Warnings only —
    nothing downstream reads them.
    """
    warns = []
    n = len(placed)
    if n and n_pages >= 60 and n * 40 < n_pages:
        warns.append(f"contents parse looks sparse: {n} entr"
                     f"{'y' if n == 1 else 'ies'} for {n_pages} pages")
    if n and missed >= max(3, n // 4):
        warns.append(f"{missed} of {n} contents entries never placed "
                     f"in the book")
    if furn_unmerged >= 3:
        warns.append(f"running heads describe {furn_unmerged} chapter(s) "
                     f"the contents never placed")
    nums = set()
    for e, _, _ in placed:
        t = e.title.strip()
        m = re.match(r"chapter\s+([A-Za-z0-9-]+)", t, re.I)
        v = _chapter_head_number(m.group(1)) if m else None
        if v is None:
            m = re.match(r"(\d{1,3})\b(?!\s*[-\u2013])", t)
            v = int(m.group(1)) if m else None
        if v is not None and v < 200:
            nums.add(v)
    if len(nums) >= 2:
        gaps = sorted(set(range(1, max(nums))) - nums)
        if len(gaps) >= 2:
            shown = ", ".join(map(str, gaps[:6]))
            warns.append(f"chapters {min(nums)}\u2013{max(nums)} are printed "
                         f"but {len(gaps)} number(s) never parsed: {shown}"
                         + (", \u2026" if len(gaps) > 6 else ""))
    return warns


def nav_from_placed(placed, pos_of, kept, bodies, toc_pages,
                    make_link, make_section) -> tuple:
    """The printed contents as a nav tree: links, nested sections, and the
    hunt for folio-less lines. Extracted from build_epub so the invariants
    the 2026-08-18 review found broken here are testable: every id unique
    (C1/C2 — one monotonic counter, no frozen list lengths), the title
    hunt always moving forward from the last thing placed, folio-resolved
    or hunted alike (C3), and an unresolvable grouping degrading to a bare
    heading.

    make_link(href, title, uid) / make_section(title, href|None) keep
    ebooklib out of the function's signature — tests pass plain tuples.
    Returns (links, verified, missed, resolved), where resolved is the
    placed list with every title hunt settled — the after-the-fact truth
    of what emitted and where, which is what the furniture check needs:
    measured against the pre-hunt list, a wholly folio-less contents
    looks entirely unplaced and every chapter looks missing."""
    links: list = []
    resolved: list = []
    verified = missed = 0
    next_id = iter(range(10 ** 9))
    empties: list = []
    # Titles are hunted in reading order, starting past the contents —
    # a book's half-title repeats the chapter's words before the book
    # has begun, and would otherwise win chapter one every time.
    hunt_from = max(toc_pages) if toc_pages else -1
    stack: list = [(-1, links)]
    stripped_cache: dict = {}

    def stripped(t):
        if t not in stripped_cache:
            stripped_cache[t] = _strip_tags(bodies[t])
        return stripped_cache[t]

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
                               e.title, stripped(t)[:300])),
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
                                   e.title, stripped(t)[:400])),
                              None)
        if target is not None and target in pos_of:
            # forward from the last thing placed means EVERY placement,
            # folio-resolved included — leaving hunt_from behind on
            # resolved entries let a later hunt match backward into an
            # already-passed chapter (C3)
            hunt_from = max(hunt_from, target)
        resolved.append((e, target if target in pos_of else None, ok))
        if e.depth < 2:
            # a grouping: BOOK SEVEN as printed, or a chapter promoted
            # because subsections follow it. A chapter that nests
            # children is still a destination, so it keeps its link
            # when it resolved; a grouping that never resolves is a
            # heading and no more.
            sect: list = []
            while stack and stack[-1][0] >= e.depth:
                stack.pop()
            href = (f"page_{target:04d}.xhtml"
                    if target is not None and target in pos_of else None)
            head = make_section(e.title, href)
            slot = [(head, sect)]
            stack[-1][1].append(slot)
            # A grouping that ends up with no children must not ship as
            # one: ebooklib writes an empty <ol/> for it, and an <ol> with
            # no <li> is invalid in a nav document. Recorded here so the
            # pass below can turn a childless grouping back into the plain
            # link it always was.
            empties.append((slot, e.title, href))
            stack.append((e.depth, sect))
            continue
        if target is None or target not in pos_of:
            missed += 1
            continue
        # one monotonic counter for every id: the old ids froze len(links)
        # inside sections and reused page numbers across same-rank heads,
        # shipping duplicate ids in nav.xhtml (C1/C2)
        stack[-1][1].append(make_link(f"page_{target:04d}.xhtml", e.title,
                                      f"toc{next(next_id)}-{pos_of[target]}"))
    # Childless groupings become links again, or vanish if they never
    # resolved to a page — either way no empty <ol> reaches the reader.
    for slot, title, href in empties:
        if slot[0][1]:
            continue
        slot[0] = (("link", make_link(href, title, f"toc{next(next_id)}-g"))
                   if href else None)
    def _flatten(seq):
        out = []
        for item in seq:
            if isinstance(item, list):          # a grouping's slot
                item = item[0]
                if item is None:                # never resolved: drop it
                    continue
                if item[0] == "link":           # pruned to a plain link
                    out.append(item[1])
                else:
                    out.append((item[0], _flatten(item[1])))
            else:
                out.append(item)
        return out
    return _flatten(links), verified, missed, resolved


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
    start = notes_section_start(bodies)
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


_ENDS_A_SENTENCE = re.compile(r"[.!?…:;][”’\"')\]]*\s*$")


def _carried_over_tail(below: str, before: str) -> int:
    """Where a page's own notes begin, past the end of a note the page
    before could not finish; 0 if the foot block does not open that way.

    A long footnote runs over the page, and its last lines are the first
    thing under the next page's rule, with no number in front of them. A
    foot block has to open with a foot, so that page was not a foot block,
    and none of its calls or notes were linked: twenty-seven pages of Kill
    Boxes, every note on them, and the same in the other books of its
    series.

    The tail is one or two paragraphs with no number. What makes it a tail
    and not prose under a divider is the page before: its last note stops
    without ending its sentence ("…the Statue of Liberty be-"), or the
    paragraph itself starts in the middle of one.
    """
    paras = list(re.finditer(r"<p[^>]*>.*?</p>", below, re.S))
    k = next((j for j, m in enumerate(paras) if FOOT_SHAPE.match(m.group(0))),
             None)
    if k is None or not 1 <= k <= 2:
        return 0
    if _strip_tags(below[:paras[0].start()]).strip() \
            or re.search(r"<(?:h[1-6]|table|ul|ol|figure)\b",
                         below[:paras[k].start()]):
        return 0
    first = _strip_tags(paras[0].group(0)).strip()
    if not first:
        return 0
    last = [m.group(0) for m in re.finditer(r"<p[^>]*>.*?</p>",
                                            before.split("<hr/>", 1)[-1], re.S)]
    unfinished = bool(last) and "<hr/>" in before \
        and not _ENDS_A_SENTENCE.search(_strip_tags(last[-1]).strip())
    if not (unfinished or first[0].islower()):
        return 0
    return paras[k].start()


# How much of what follows the rule the layout must have called footnotes
# for its word to stand in for the block's shape.
FOOT_LABELLED = 0.9


def _opens_with_a_foot(below: str) -> bool:
    first = re.search(r"<(p|ol|ul|table|figure|blockquote|h[1-6])[^>]*>", below)
    return first is not None and first.group(1) == "p" \
        and bool(FOOT_SHAPE.match(below[first.start():]))


def link_footnotes(bodies: list[str], allow_numbered: bool = False,
                   foot_labelled: "list | None" = None) -> dict:
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

    One shape the guards refused is not unusual at all: a note of two
    paragraphs, or one that ends in a list. Its second paragraph has no
    number, so the block was no longer "feet nearly throughout", and no
    note on the page was linked. `foot_labelled` is the layout's own word,
    page by page: the share of the words from the first block it called a
    footnote to the end of the page that it called footnotes. Where that
    is nearly all of them and the block opens with a foot, the paragraphs
    with no number are the notes going on, and the block stands.
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
        tail = ""
        if allow_numbered and not _looks_like_foot_block(below):
            cut = _carried_over_tail(below, bodies[pi - 1] if pi else "")
            if cut:
                tail, below = below[:cut], below[cut:]
                stats["carried"] = stats.get("carried", 0) + 1
        labelled = bool(foot_labelled) and pi < len(foot_labelled) \
            and foot_labelled[pi] >= FOOT_LABELLED \
            and _opens_with_a_foot(below)
        if allow_numbered and (_looks_like_foot_block(below) or labelled):
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

        bodies[pi] = above + rule + tail + below
    return stats


def _best_body_groups(bodies, notes_start, n_notes):
    """The body's chapters, read the way the notes section says there are.

    Splitting eagerly finds chapters whose notes are one or two citations,
    which a cautious rule folds into their neighbour — one book presented
    four groups against seven and was refused outright, losing all
    thirty-nine of its links. Splitting eagerly also cuts a chapter in two
    where it merely cites its first note twice, and the marker stranded on
    the near side of that cut is then linked to nothing.

    Neither reading is right in general, so both are offered and the notes
    section arbitrates: it knows how many chapters it has. The reading
    whose group count is closest wins, and where they tie — as they do when
    an eager split strands a marker into a group too small to stand — the
    one that leaves fewest markers behind does, because a marker outside
    every group is a marker that cannot link.
    """
    best = None
    # The readings without a headless chapter come first and keep any tie,
    # so a book whose groups already match its notes is read as before.
    for headless in (False, True):
        for climbed in (1, 3):
            raw = find_body_marker_groups(bodies, notes_start, climbed,
                                          headless)
            for smallest in (1, 2):
                # A chapter may cite a single note, and then its body group
                # is one marker long. Discarding every such group by default
                # stranded those markers; keeping them by default invents a
                # chapter out of any stray numeral. It is another thing the
                # notes section can settle, so it is offered as a reading
                # too.
                groups = [g for g in raw if len(g) >= smallest]
                rank = (abs(len(groups) - n_notes),
                        -sum(len(g) for g in groups))
                if best is None or rank < best[0]:
                    best = (rank, groups)
    return best[1]


def _fit_groups(body_groups, note_groups):
    """The one way these two lists of groups line up, or None.

    The counts differ when one side sees a chapter boundary the other
    cannot. A chapter whose notes are a single citation is the case that
    surfaced it: the notes side has an explicit head to open the group,
    while the body side has only the numbers, and it needs an ascending
    run before a return to 1 counts as a new chapter — otherwise an
    author re-citing note 2 would open one. So that chapter's lone marker
    is absorbed by its neighbour and the body comes up one group short.

    Trimming the tail, which is all this used to do, assumes the missing
    group is at the END. Here it was at the FRONT — a preface with one
    note — and trimming shifted all nine pairs, which the agreement test
    then caught and refused: 810 markers left plain on a book whose notes
    were fully parsed and sitting right there.

    Every contiguous fit is offered to that same test instead. The one
    the code already tried is offered FIRST and kept if it passes, so no
    book that pairs today can stop pairing; the others are consulted only
    after it fails, and one is taken only if it is the only one that
    passes. Two candidates agreeing means the numbers cannot separate
    them, and a confident link into the wrong chapter's citations is the
    one outcome worse than no link at all.
    """
    k = min(len(body_groups), len(note_groups))
    body_longer = len(body_groups) > len(note_groups)
    longer = body_groups if body_longer else note_groups

    def paired(off):
        win = longer[off:off + k]
        if body_longer:
            return win, note_groups
        # The notes side keeps a group this side lost, and the markers it
        # lost sit at the front of the first group that remains — as a
        # repeated 1, since the absorbed chapter's numbering started over
        # there too. Pairing them anyway pointed a preface's only
        # citation at the introduction's first source. They are left
        # plain instead: those notes have no marker this code can name.
        bg = list(body_groups)
        if off and bg:
            bg[0] = bg[0][_leading_repeats(bg[0]):]
        return bg, win

    if _paired_groups_agree(*paired(0), min_judged=2):
        return paired(0)
    rest = [paired(off) for off in range(1, len(longer) - k + 1)
            if _paired_groups_agree(*paired(off), min_judged=2)]
    return rest[0] if len(rest) == 1 else None


def _leading_repeats(group) -> int:
    """Where a body group's own numbering starts, past any absorbed one.

    A chapter whose notes are a single citation gives the body splitter
    nothing to split on — it needs an ascending run before a return to 1
    counts as a new chapter, or an author re-citing note 2 would open one
    — so that marker is absorbed by the next chapter and shows up as a
    repeated 1 at the front.

    Only a LEADING run of 1s counts. A 1 further along is the author
    citing note 1 again, and that marker does belong to this chapter.
    """
    k = 0
    while k + 1 < len(group) and group[k][1] == 1 and group[k + 1][1] == 1:
        k += 1
    return k


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


from .typography import (  # noqa: F401  (moved; kept importable from here)
    _ELISIONS, _requote, normalize_typography, _c1_repair_table,
    C1_REPAIR, _XML_NAME, _A_TAG, _AN_ATTR, _KEPT_IN_AN_ADDRESS,
    _swallowed, mend_swallowed_links, repair_c1_controls,
    unusual_characters,
)


from .citations import (  # noqa: F401  (moved; kept importable from here)
    BIBLIOGRAPHY_HEAD, AD_ENTRY, YEAR_ALONE, BARE_YEAR_ENTRY, _PARTICLES,
    _norm_title, _surname_forms, _outside_markup, parse_bibliography,
    link_citations,
)


from .index_links import (  # noqa: F401  (moved; kept importable from here)
    INDEX_HEAD, INDEX_NOT_PAGES, INDEX_REF, _trailing_runs, link_index,
)




def _box_iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = ((a[2] - a[0]) * (a[3] - a[1])
             + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _match_layout_labels(items: list, blocks: list, width: int,
                         height: int) -> int:
    """Assign layout labels to cached items by greedy best-IoU, one block
    per item, nothing under 0.5. An item that matches no block cleanly
    stays unlabeled — a wrong label misleads every consumer downstream,
    an absent one merely declines to help."""
    cands = []
    for bi, (lab, bb) in enumerate(blocks):
        nb = (bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height)
        for ii, it in enumerate(items):
            if not it.get("box") or it.get("lab"):
                continue
            iou = _box_iou(nb, tuple(it["box"]))
            if iou >= 0.5:
                cands.append((iou, ii, bi, lab))
    cands.sort(key=lambda c: -c[0])
    used_i, used_b, n = set(), set(), 0
    for iou, ii, bi, lab in cands:
        if ii in used_i or bi in used_b:
            continue
        items[ii]["lab"] = lab
        used_i.add(ii)
        used_b.add(bi)
        n += 1
    return n


def relabel_cache(cache_path: str, pdf_path: str, backend_name: str = "surya",
                  langs: "list[str] | None" = None) -> dict:
    """Backfill layout labels into an existing cache without re-reading a
    word.

    Recognition is the slow half of a build and the settled half: the words
    in the cache do not improve by being read again by the same pinned
    model. Labels are the cheap half — one layout call per page instead of
    layout plus recognition; the layout head rides the same served
    foundation model, so the inference server must be free, and it is
    auto-started the way a build starts it. Pages are
    regenerated through the same extraction path the original build used,
    so their cache keys reproduce exactly; a page
    whose key misses is reported and skipped, never guessed at. The text is
    left byte-identical, which keeps every dispute record and decisions
    file valid."""
    from collections import Counter as _Counter
    langs = langs or ["en"]
    tmp = tempfile.mkdtemp(prefix="relabel_")
    try:
        page_paths = load_pages_from_pdf(
            pdf_path, tmp, former_keys=OCRCache.filed_by_bytes(cache_path),
            turns_path=cache_path + TURNS_SUFFIX)
        cache = OCRCache(cache_path, backend_name, langs)
        if not cache.entries:
            print(f"  [!] cache is cold or tagged differently: {cache_path}")
            return {"pages": 0}
        # Guided layout, surya's default and what a build's fallback uses,
        # with the schema repaired so llama.cpp can compile it. This ran
        # unguided from August to dodge the grammar error. It cost nothing
        # that can be seen: on Eagle Forgotten's last 22 pages the two
        # return identical boxes — including none at all on the 17 dense
        # back-matter pages, which neither mode can box.
        llamacpp_safe_schemas()
        from surya.layout import LayoutPredictor
        lp = LayoutPredictor()
        st = _Counter()
        # The served model cold-starts, and a call against a waking server
        # returns nothing — indistinguishable from a blank page. Warm up
        # against a mid-book page (guaranteed ink) with patience, so the
        # dry-streak guard below judges a live server only.
        probe_path = page_paths[len(page_paths) // 2]
        for _ in range(12):
            if lp([Image.open(probe_path)])[0].bboxes:
                break
            time.sleep(8)
        else:
            print("  [!] layout server never came up after 96s; aborting "
                  "with the cache untouched")
            return dict(st)
        dry_streak = 0
        for pp in page_paths:
            entry = cache._entry(pp)
            if entry is None:
                st["page_miss"] += 1
                continue
            raw = entry["items"] if isinstance(entry, dict) else entry
            boxed = [it for it in raw if it.get("box")]
            if boxed and all(it.get("lab") for it in boxed):
                st["already_labeled"] += 1
                continue
            img = Image.open(pp)
            result = lp([img])[0]
            blocks = [(b.label, list(b.bbox)) for b in result.bboxes]
            # a page with cached blocks but zero layout blocks is the shape
            # of a dead inference server retrying forever, not of a book;
            # three in a row and this run stops rather than spins
            if boxed and not blocks:
                dry_streak += 1
                if dry_streak >= 3:
                    print("  [!] layout returned nothing three pages "
                          "running — inference server down? aborting; "
                          "cache untouched beyond what was already saved")
                    break
            else:
                dry_streak = 0
            st["items_labeled"] += _match_layout_labels(
                raw, blocks, img.width, img.height)
            st["items_unmatched"] += sum(1 for it in raw
                                         if it.get("box") and not it.get("lab"))
            st["pages"] += 1
            # stamped current: these pages have been through the layout
            # model, so a later build must not read them again
            if isinstance(entry, dict):
                entry["v"] = CACHE_VERSION
            cache.dirty = True
        if cache.dirty:
            backup = cache_path + ".prelabel"
            if not os.path.exists(backup):
                shutil.copy2(cache_path, backup)
            cache.save()
        print(f"  relabel: {st['items_labeled']} item(s) labeled across "
              f"{st['pages']} page(s), {st['items_unmatched']} unmatched "
              f"(left unlabeled), {st['page_miss']} page key(s) not in cache, "
              f"{st['already_labeled']} already labeled")
        return dict(st)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def rekey_cache(cache_path: str, pdf_path: str, backend_name: str = "surya",
                langs: "list[str] | None" = None) -> dict:
    """Refile every reading in a cache under the name its page has now.

    A build does this for itself, page by page, the first time it meets a
    cache written before pages were named for their pixels (see page_key).
    This does a whole book at once without building it, which is the way to
    bring a shelf of caches over while the encoder that wrote their old
    names is still the one installed: once it changes, a reading that was
    never refiled can only be read again.

    Nothing is read and no reading is altered. The entries after are
    checked to be the entries before, one for one, and the file is left
    untouched if they are not.
    """
    from collections import Counter as _Counter

    def fingerprints(entries: dict) -> "_Counter":
        return _Counter(hashlib.sha256(
            json.dumps(e, sort_keys=True).encode("utf-8")).hexdigest()
            for e in entries.values())

    cache = OCRCache(cache_path, backend_name, langs or ["en"])
    st = {"entries": len(cache.entries), "pages": 0, "found": 0,
          "refiled": 0, "missing": 0, "left": len(cache.by_bytes)}
    if not cache.entries:
        print(f"  [!] cache is cold or tagged differently: {cache_path}")
        return st
    if not cache.by_bytes:
        print(f"  rekey: all {st['entries']} reading(s) in {cache_path} are "
              f"already filed under their pages' present names")
        return st
    before = fingerprints(cache.entries)
    tmp = tempfile.mkdtemp(prefix="rekey_")
    try:
        page_paths = load_pages_from_pdf(pdf_path, tmp, former_keys=True,
                                         turns_path=cache_path + TURNS_SUFFIX)
        st["pages"] = len(page_paths)
        for pp in page_paths:
            if cache._entry(pp) is None:
                st["missing"] += 1
            else:
                st["found"] += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    st["refiled"], st["left"] = cache.refiled, len(cache.by_bytes)
    if fingerprints(cache.entries) != before:
        print(f"  [!] rekey: the readings after are not the readings before; "
              f"{cache_path} is left as it was")
        st["refiled"] = 0
        return st
    cache.save()
    print(f"  rekey: {st['found']} of {st['pages']} page(s) have a reading; "
          f"{st['refiled']} refiled under the page's pixels"
          + (f"; {st['missing']} page(s) of this PDF are not in the cache"
             if st["missing"] else "")
          + (f"; {st['left']} reading(s) match no page of this PDF and stay "
             f"under their old names" if st["left"] else ""))
    return st


PAGE_HYPHEN = re.compile(r"([A-Za-z]+)-((?:\s*</[^>]+>)*\s*)$")
PAGE_CONT = re.compile(r"^((?:\s*<[^>]+>)*\s*)([a-z][A-Za-z'’]*[.,;:!?'’\"]*)")


def join_page_break_hyphens(bodies: list, dropped: "set[int]") -> dict:
    """A word the printer split across a page turn is one word.

    Surya rejoins line-break hyphenation within a page, but a page is the
    edge of its world: eagle shipped sixty words split at page edges —
    Rus- | sia — where search cannot match and speech stumbles. When a
    kept page ends mid-word on an ASCII hyphen and the next kept page
    opens with a lowercase fragment, and the joined form is a word — the
    dictionary's or this book's own — the continuation moves up to
    complete it. The page-break anchor stays where it was.

    Refusals are the point: an em-dash never matches, a capitalized
    continuation is left alone, a heading is never raided for its first
    word, and a printed compound breaking at the page edge (self- |
    control) fails the join gate and keeps its hyphen, exactly as set."""
    from .adjudicate import _is_word, fold_word
    kept = [i for i in range(len(bodies)) if i not in dropped]
    # the whole-book vocabulary is only evidence for judging a join, so a
    # book with no page ending mid-word — common in modern typesetting —
    # pays nothing for it (E4)
    if not any(PAGE_HYPHEN.search(bodies[i]) for i in kept):
        return {"joined": 0, "refused": 0}
    vocab = set()
    for b in bodies:
        for w in re.findall(r"[A-Za-z][A-Za-z'’]*", _strip_tags(b)):
            vocab.add(fold_word(w))
    joined = refused = 0
    for a, b in zip(kept, kept[1:]):
        m = PAGE_HYPHEN.search(bodies[a])
        if not m:
            continue
        c = PAGE_CONT.match(bodies[b])
        if not c or re.search(r"<(h\d|figure)", c.group(1)):
            continue
        frag = c.group(2)
        word2 = re.match(r"[a-zA-Z'’]+", frag).group(0)
        f = fold_word(m.group(1) + word2)
        if not (_is_word(f) or f in vocab):
            refused += 1
            continue
        bodies[a] = bodies[a][:m.start()] + m.group(1) + frag + m.group(2)
        bodies[b] = c.group(1) + bodies[b][c.end():].lstrip()
        joined += 1
    return {"joined": joined, "refused": refused}


def _decision_rounds(decisions_path) -> list:
    """The decisions files to apply, in order: none, one, or several.

    A file named here brings its earlier rounds with it: the ones
    `codicology decisions file` set aside beside it as book.decisions.r1.json
    and so on (review.decision_rounds). Naming only the round in hand is
    what the Calibre plugin does, and it used to mean the rounds before it
    were not applied at all."""
    from .review import decision_rounds
    if not decisions_path:
        return []
    if isinstance(decisions_path, (str, bytes, os.PathLike)):
        decisions_path = [decisions_path]
    out: list = []
    for given in (p for p in decisions_path if p):
        for path in decision_rounds(os.fspath(given)) or [given]:
            if not any(_same_file(path, seen) for seen in out):
                out.append(path)
    return out


def apply_reviewer_decisions(bodies: list, decisions_path: str,
                             kept_pages: "int | None" = None) -> dict:
    """A review sheet's exported decisions, reapplied at rebuild time —
    before the linkers run, so a corrected word can still earn its anchor.

    Same contract as `codicology apply`, and the same code
    (review.apply_page_decisions): the page's text only, the word each
    decision is addressed to, deletions before swaps, and a decision that
    cannot be placed with certainty is stale — reported and left as read,
    never guessed at."""
    from .adjudicate import tokens_as
    from .review import (apply_page_decisions, coerce_page, pages_guard,
                         retired_rows)
    try:
        with open(decisions_path) as fh:
            dec = json.load(fh)
        rows = dec.get("decisions", [])
    except (OSError, ValueError) as exc:
        print(f"    [!] decisions file unreadable, ignored: {exc}")
        return {"applied": 0, "stale": 0}
    if retired_rows(dec):
        print(f"    {len(retired_rows(dec))} decision(s) in "
              f"{os.path.basename(os.fspath(decisions_path))} are retired "
              f"and left alone")
    refusal = pages_guard(dec.get("page_files"),
                          kept_pages if kept_pages is not None
                          else len(bodies))
    if refusal:
        print(f"    refused: {refusal}")
        return {"applied": 0, "stale": len(rows)}
    applied = stale = 0
    by_page: dict = {}
    for d in rows:
        i = coerce_page(d)
        if i is None or not (0 <= i < len(bodies)):
            stale += 1
            print(f"    stale decision: page {d.get('page')!r} is not a "
                  f"page of this build — {d.get('old')!r} skipped")
            continue
        by_page.setdefault(i, []).append(d)
    for i in sorted(by_page):
        # read the page the way the file's rows were addressed
        with tokens_as(dec.get("tokens")):
            bodies[i], outcomes = apply_page_decisions(bodies[i], by_page[i])
        for d, why in zip(by_page[i], outcomes):
            if why is None:
                applied += 1
            else:
                stale += 1
                print(f"    stale decision: p{i} {d.get('old')!r} -> "
                      f"{d.get('new')!r} — {why}; kept as read")
    return {"applied": applied, "stale": stale}


# A raised figure standing directly on a title in italics, an abbreviation
# in capitals or a Roman numeral is how an edition or a volume is written:
# SEHRE², SIG³, IG II². In the body of a book whose notes are numbered it is
# also exactly how a note call is written, so it is only taken for the
# edition where it does not continue its chapter's calls: the calls climb,
# and a 2 after call 14 has not climbed. Ste. Croix cites editions this way
# ninety-odd times in a body of 1,275 calls; each would have opened the
# second or third note of whatever section it stood in.
#
# One that happens to continue the calls (PL² met just after call 1) is
# told another way: a title carries its edition wherever it is cited, so
# a figure once shown to be a title's edition is that title's edition
# everywhere in the book. Shown, not merely met twice: "in the EU¹" may
# open two chapters, and both are calls.
_EDITION_MARK = re.compile(
    r"(?:<i>([^<]{1,40})</i>|(?<![A-Za-z])([A-Z]{2,}))?"
    r"(?:\s*(?<![A-Za-z])([IVXL]{1,6}))?$")


def _title_under(body: str, pos: int) -> "str | None":
    """The title, abbreviation or volume the raised figure at `pos` stands
    directly on, or None where it stands on anything else."""
    m = _EDITION_MARK.search(body[max(0, pos - 60):pos])
    if not m or not any(m.groups()):
        return None
    return " ".join(g.strip() for g in m.groups() if g)


def _titles_with_editions(bodies: list[str], body_groups: list,
                          note_groups: list) -> set:
    """(title, figure) pairs the order of the calls shows to be editions:
    met somewhere on that title at or below the highest call its chapter
    had made."""
    shown: set = set()
    for bg, ng in zip(body_groups, note_groups):
        answered = {n for (_, n, _) in ng}
        top = 0
        for (pi, n, pos) in bg:
            if n not in answered:
                continue
            title = _title_under(bodies[pi], pos)
            if title is not None and n <= top:
                shown.add((title, n))
                continue
            top = max(top, n)
    return shown


def _an_editions_number(body: str, pos: int, n: int, top: int,
                        editions: set = frozenset()) -> bool:
    """Whether the raised figure at `pos` is an edition's or a volume's and
    not a call: it stands on a title, and either is no higher than the
    highest call its chapter has made so far or is that title's edition
    elsewhere in the book (`editions`, from _titles_with_editions)."""
    title = _title_under(body, pos)
    return title is not None and (n <= top or (title, n) in editions)


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
        body_groups = _best_body_groups(bodies, notes_start, len(note_groups))
    if len(body_groups) != len(note_groups):
        if abs(len(body_groups) - len(note_groups)) > 2:
            stats["misaligned"] = True
            return stats
        stats["misaligned"] = True
        fit = _fit_groups(body_groups, note_groups)
        if fit is None:
            return stats
        body_groups, note_groups = fit

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

    editions = _titles_with_editions(bodies, body_groups, note_groups)
    # rewrite from the last page backward so match offsets stay valid
    edits: dict[int, list[tuple[int, int, str]]] = {}
    for g, (bg, ng) in enumerate(zip(body_groups, note_groups)):
        stats["groups"] += 1
        notes_by_n: dict[int, tuple[int, int]] = {}
        for (pi, n, pos) in ng:
            notes_by_n.setdefault(n, (pi, pos))
        seen_backref: set[int] = set()
        top = 0                 # the highest call this chapter has made
        for (pi, n, pos) in bg:
            target = notes_by_n.get(n)
            if target is None or pi in dropped:
                stats["unlinked"] += 1
                continue
            if _an_editions_number(bodies[pi], pos, n, top, editions):
                stats["unlinked"] += 1
                continue
            top = max(top, n)
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

class ReadPages(NamedTuple):
    """Everything the reading half of a build produces: the text of every
    page, the figures already added to the book, and the counters the
    report speaks from. build_epub's assembly half consumes exactly this,
    and nothing else crosses the boundary (S4)."""
    bodies: list
    page_figures: list
    figure_data: dict
    page_furniture: list
    furniture_known: list
    n_figures: int
    n_blank_figures: int
    n_label_rules: int
    n_reattached: int
    n_phantoms: int
    n_forms: int
    n_hand_figures: int
    label_heads: list
    toc_label_pages: set
    n_doubled: int = 0
    foot_labelled: "list | None" = None


def _read_pages(page_paths, backend, cache, book, strip_furniture) -> ReadPages:
    """The OCR-reading half of a build, extracted from build_epub so the
    two jobs that shared one 790-line function can be reasoned about — and
    driven by tests — separately. Figures are added to the passed book as
    they are met, exactly as before; the assembly half receives the rest
    as a bundle."""
    from ebooklib import epub
    warmed = False           # the backend has proved it can read
    _read_pages_resiliently.silent.clear()      # a run belongs to one book
    bodies: list[str] = []
    page_figures: list[list[str]] = []  # figure files each page put in the book
    figure_data: dict[str, bytes] = {}  # their bytes, for spotting repeats
    page_furniture: list[list[str]] = []   # running head/foot text, per page
    furniture_known: list[bool] = []       # False = cached before heads were kept
    n_figures = 0
    n_blank_figures = 0
    n_hand_figures = 0
    n_label_rules = 0
    n_reattached = 0
    n_phantoms = 0
    n_forms = 0
    n_doubled = 0
    n_unvouched = 0
    foot_labelled: list = []    # per page: how much of its foot the layout
                                # itself called footnotes (see link_footnotes)
    marker_seq_state: dict = {"last": None}
    label_heads: list = []      # (page, rank, text) from SectionHeader
    toc_label_pages: set = set()
    total = len(page_paths)

    nonlocal_page_i = [0]      # the page emit_figure is currently working on

    from concurrent.futures import ThreadPoolExecutor
    encoders = ThreadPoolExecutor(max_workers=FIGURE_ENCODERS)
    encoding: list = []         # (stand-in name, the encoding under way)

    def settle_figures() -> None:
        """Give every figure its file, in the order the pages met them, and
        its real name wherever a page used the stand-in."""
        names = {}
        for k, (stand_in, job) in enumerate(encoding):
            data, ext, mime = job.result()
            name = names[stand_in] = f"images/fig_{k:04d}.{ext}"
            figure_data[name] = data
            book.add_item(epub.EpubImage(
                uid=f"fig{k}",
                file_name=name,
                media_type=mime,
                content=data,
            ))
        encoders.shutdown()
        if not names:
            return
        named = re.compile(r'(?<=<img src=")images/fig_\d{4}\.encoding(?=")')
        bodies[:] = [named.sub(lambda m: names[m.group(0)], b) for b in bodies]
        page_figures[:] = [[names[n] for n in figs] for figs in page_figures]

    def emit_figure(item: PageItem, caption: PageItem | None, caption_first: bool) -> str:
        """Store the crop in the book and wrap it, with its caption, in one <figure>."""
        nonlocal n_figures, n_blank_figures, n_hand_figures
        if figure_is_a_hand(item.figure):
            n_hand_figures += 1
            item.figure.close()
            return f"<p>{caption.html}</p>" if caption is not None else ""
        if not figure_has_content(item.figure):
            # A blank leaf the layout pass boxed for want of anything else to
            # call it. Its caption, if any, is still worth keeping: a plate's
            # caption sometimes sits on the facing blank.
            n_blank_figures += 1
            item.figure.close()
            return f"<p>{caption.html}</p>" if caption is not None else ""
        art = better_figure(page_paths[nonlocal_page_i[0]], item.box, item.figure)
        enc = _encodable(art)
        if enc is not art and art is not item.figure:
            art.close()
        art = enc

        def encode(art=art, crop=item.figure):
            try:
                return encode_figure(art)
            finally:
                for im in (art, crop):
                    try:
                        im.close()
                    except Exception:
                        pass

        # Encoded on another thread while the pages go on being assembled:
        # Pillow lets go of the interpreter while it compresses, and this
        # was 54 of a 104 second rebuild, one figure at a time. Whether a
        # figure ends as PNG or JPEG is not known until it is done, so the
        # page is written with a stand-in for the file's name and
        # settle_figures puts the real one in. No more are left waiting
        # than the encoders can hold: a plate is tens of megabytes open.
        while sum(1 for _, job in encoding if not job.done()) \
                >= 2 * FIGURE_ENCODERS:
            next(job for _, job in encoding if not job.done()).result()
        name = f"images/fig_{n_figures:04d}.encoding"
        encoding.append((name, encoders.submit(encode)))
        current_figures.append(name)
        n_figures += 1

        # A page shown as itself says so, in the words a page nothing was
        # read from already uses: it is what a reader who cannot see the
        # picture is told, and what verify knows such a page by.
        alt = f"Page {nonlocal_page_i[0] + 1}, as printed" \
            if item.whole_page else "Figure"
        img_tag = f'<img src="{name}" alt="{alt}" style="max-width:100%"/>'
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
            if not warmed:
                # The first page this run must read for itself. Before its
                # answer is believed — or cached — make the backend prove it
                # can read at all, because a silent empty is how a cold
                # server blanks a book.
                why = warm_backend(backend)
                if why is not None:
                    sys.exit(
                        f"the {backend.name} backend is not reading: it "
                        f"{why} on a synthetic page of known words.\n"
                        f"{len(unseen)} page(s) of this book are not in the "
                        f"cache and would be read live — an unserved backend "
                        f"returns empty pages rather than an error, so they "
                        f"would ship blank.\nCheck the inference server "
                        f"(`codicology doctor --full`) and run again; "
                        f"nothing has been written.")
                warmed = True
            opened = [Image.open(p).convert("RGB") for p in unseen]
            fresh = _read_pages_resiliently(unseen, opened, backend, cache)
            for im in opened:
                im.close()

        for path, hit in zip(chunk_paths, remembered):
            nonlocal_page_i[0] = len(bodies)   # the page these figures sit on
            items = hit if hit is not None else fresh[path]
            items = refold_cached_fragments(items)
            items = demote_mislabelled_heads(items, os.path.basename(path))
            items, _nf = route_form_blocks(items)
            n_forms += _nf
            items, _counts = drop_counting_tables(items, path)
            for _n in _counts:
                print(f"    [!] {os.path.basename(path)}: a table of {_n} "
                      f"numbers that only counts, "
                      + ("and neither the page's text layer nor a classical "
                         "reader finds the count on the page"
                         if witness.witness_available() else
                         "which the page's text layer does not hold, and no "
                         "classical reader is installed to ask")
                      + " — left out rather than printed as invention")
            items, _mended = mend_tables(items, path, switches.SECOND_OPINION)
            items, _eqs, _differ = mend_equations(items, path, switches.SECOND_OPINION,
                                                  switches.THIRD_OPINION)
            if switches.SECOND_OPINION is not None:
                _page = os.path.splitext(os.path.basename(path))[0]
                switches.SECOND_OPINION.tables += len(_mended)
                _own = sum(1 for _e in _eqs if len(_e) > 2
                           and _e[2].startswith("the third reader's own"))
                switches.SECOND_OPINION.equations += len(_eqs) - _own
                switches.SECOND_OPINION.by_third += _own
                switches.SECOND_OPINION.differ += len(_differ)
                switches.SECOND_OPINION.changed += [(_page, "equation", *_e)
                                           for _e in _eqs]
                switches.SECOND_OPINION.changed += [
                    (_page, "table", f"held {_w} of the {_p} figures the PDF "
                     f"prints in its place", f"holds {_n}"
                     + (f"; also prints {', '.join(_o)}, which the PDF does "
                        f"not" if _o else ""))
                    for _p, _w, _n, _o in _mended]
                switches.SECOND_OPINION.listed += [(_page, *_d) for _d in _differ]
            for _was, _now, *_why in _eqs:
                print(f"    [+] {os.path.basename(path)}: an equation "
                      + (f"is set by a vote of three readers ({_why[0]})"
                         if _why else
                         "the second reader reads otherwise is set as it "
                         "reads it")
                      + f": {_was[:60]} → {_now[:60]}")
            for _one, _other, *_why in _differ:
                print(f"    [?] {os.path.basename(path)}: the two readers "
                      f"differ on an equation"
                      + (f" ({_why[0][:120]})" if _why else
                         " and nothing on the page says which is right")
                      + f" — the first is kept; worth a look:\n"
                      f"          first:  {_one[:110]}\n"
                      f"          second: {_other[:110]}")
            for _printed, _was, _now, _odd in _mended:
                print(f"    [+] {os.path.basename(path)}: a table held {_was} "
                      f"of the {_printed} figures the PDF prints in its "
                      f"place; the second reader's holds {_now} and is "
                      f"printed instead"
                      + (f" (it also prints {', '.join(_odd)}, which the PDF "
                         f"does not: worth a look)" if _odd else ""))
            for _printed, _held in tables_short_of_figures(items, path):
                print(f"    [?] {os.path.basename(path)}: a table holds "
                      f"{_held} of the {_printed} figures the PDF prints in "
                      f"its place — left as read; worth a look")
            _unsure = unvouched_reading(items, path)
            _ink = _page_core_ink(path) if _unsure is not None else None
            if _unsure is not None and _ink is not None \
                    and _ink < PHANTOM_CORE_INK:
                # Words nobody vouches for on paper that holds no ink: the
                # ghost of the leaf behind, read as a page (Citadel of Sin,
                # a blank verso that came back as thirty-eight words). There
                # is nothing to show a picture of.
                _n = sum(len(_strip_tags(it.html).split()) for it in items
                         if it.html and not it.is_furniture)
                items = [it for it in items if it.is_furniture]
                n_unvouched += 1
                print(f"    [!] {os.path.basename(path)}: {_n} words the "
                      f"recogniser scored {_unsure[0]:.2f}, no witness reads "
                      f"more than {_unsure[1]:.0%} of them, and the paper "
                      f"holds no ink — left empty rather than printed as "
                      f"invention")
            elif _unsure is not None:
                _shown = _page_picture(path)
                if _shown is not None:
                    _n = sum(len(_strip_tags(it.html).split()) for it in items
                             if it.html and not it.is_furniture)
                    _kept = sum(len(_strip_tags(it.html).split())
                                for it in _unsure[2])
                    items = [it for it in items if it.is_furniture] \
                        + [_shown] + _unsure[2]
                    n_unvouched += 1
                    print(f"    [!] {os.path.basename(path)}: {_n} words the "
                          f"recogniser scored {_unsure[0]:.2f}, and no "
                          f"witness reads more than {_unsure[1]:.0%} of them "
                          f"on the page — shown as a picture of the page "
                          f"rather than printed as invention"
                          + (f", with the {_kept} words of {len(_unsure[2])} "
                             f"block(s) a witness does read printed beneath it"
                             if _unsure[2] else ""))
            # Furniture is set aside, never merely deleted: its text is the
            # folio audit's evidence, already recognised at full resolution.
            page_furniture.append([_strip_tags(it.html) for it in items
                                   if it.is_furniture and it.html])
            furniture_known.append((hit is None)
                                   or (cache is not None and cache.knows_furniture(path)))
            page_no_for_labels = len(bodies)
            for it in items:
                if it.label == "SectionHeader" and it.html \
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
            _first = next((k for k, it in enumerate(items)
                           if it.label == "Footnote"), None)
            _after = sum(len(_strip_tags(it.html).split())
                         for it in items[_first:] if it.html) \
                if _first is not None else 0
            foot_labelled.append(
                sum(len(_strip_tags(it.html).split()) for it in items[_first:]
                    if it.html and it.label == "Footnote") / _after
                if _after else 0.0)
            _ra, _sp = reattach_orphan_markers(items, furn_numbers,
                                               marker_seq_state)
            n_reattached += _ra
            n_phantoms += _sp
            if strip_furniture:
                items = strip_running_heads(items)
            items, _dbl, _held = drop_doubled_boxes(items, path)
            n_doubled += _dbl
            items = recover_placeholder_figures(items, path)
            items = recover_orphan_artwork(items, path, _held)
            items = recover_vector_artwork(items, path, _held)
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
                parts.append(item.html)
                i += 1
            bodies.append("".join(parts))
            page_figures.append(current_figures)
        done = min(start + len(chunk_paths), total)
        note = f" ({cache.hits} from cache)" if cache and cache.hits else ""
        print(f"    OCR'd {done}/{total} pages…{note}")
        progress.emit(event="progress", phase="ocr", done=done, total=total,
                      note=f"{cache.hits} from cache"
                           if cache and cache.hits else "")

    settle_figures()
    if cache:
        if cache.refiled:
            print(f"  OCR cache: {cache.refiled} reading(s) refiled under "
                  f"their pages' pixels; a later build finds them without "
                  f"the old encoding")
        cache.save()
    return ReadPages(bodies, page_figures, figure_data, page_furniture,
                     furniture_known, n_figures, n_blank_figures,
                     n_label_rules, n_reattached, n_phantoms, n_forms,
                     n_hand_figures, label_heads, toc_label_pages, n_doubled,
                     foot_labelled)


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
    serve_stale_cache: bool = False,
    cache_obj: "OCRCache | None" = None,
) -> None:
    try:
        from ebooklib import epub
    except ImportError:
        sys.exit("EPUB output needs ebooklib: pip install ebooklib")
    _restrict_page_list_to_pagebreaks()

    print(f"  OCR backend: {backend.name} (batch size {backend.batch_size})")
    progress.emit(event="phase", phase="ocr",
                  message=f"OCR backend: {backend.name}")
    cache = cache_obj if cache_obj is not None else (
        OCRCache(ocr_cache, backend.name, backend.langs,
                 serve_stale=serve_stale_cache) if ocr_cache else None)
    if cache and cache.entries:
        print(f"  OCR cache: {len(cache.entries)} pages remembered in {ocr_cache}")
        n_old = sum(1 for e in cache.entries.values()
                    if not isinstance(e, dict) or e.get("v", 0) < CACHE_VERSION)
        if n_old:
            print(f"    {n_old} of them predate layout labels"
                  + (" — served anyway (--keep-stale-cache); the label "
                     "consumers stay dark on those pages"
                     if serve_stale_cache else
                     " and will be read again, which is what puts the "
                     "labels there"))

    # A page printed sideways is turned once it has been read as it lies,
    # its head and page number kept upright (see TURN_MIN_CONFIDENCE).
    turn_sideways_bodies(page_paths, cache, backend)

    if switches.CACHE_ONLY:
        # Said before anything is done, and of every page at once: a
        # rebuild that must not read should not get as far as page 206 to
        # find that out.
        unread = [k for k, path_ in enumerate(page_paths)
                  if cache is None or cache.get(path_) is None]
        if unread:
            named = ", ".join(f"page_{k:04d}" for k in unread[:12]) \
                + (f" and {len(unread) - 12} more" if len(unread) > 12 else "")
            sys.exit(f"--cache-only: {len(unread)} of this book's "
                     f"{len(page_paths)} page(s) have no reading in the "
                     f"cache ({named}). Nothing was read and nothing was "
                     f"built. Build once without the flag to read them.")
        print("  --cache-only: every page has a reading in the cache; "
              "nothing will be read")

    book = epub.EpubBook()
    book.set_title(title)
    book.set_language(backend.langs[0] if backend.langs else "en")
    if switches.BUILD_NOTE:
        book.add_metadata("OPF", "meta", "", {"name": "codicology:build",
                                              "content": switches.BUILD_NOTE})

    # Load each batch from disk rather than holding every page in memory: a
    # 300-page book at full resolution is several GB of decoded pixels. Figures
    # go straight into the book for the same reason.
    rp = _read_pages(page_paths, backend, cache, book, strip_furniture)
    (bodies, page_figures, figure_data, page_furniture, furniture_known,
     n_figures, n_blank_figures, n_label_rules, n_reattached, n_phantoms,
     n_forms, n_hand_figures, label_heads, toc_label_pages,
     n_doubled, foot_labelled) = rp
    # Before anything reads the pages as text: what an unclosed address
    # swallowed is text again from here on.
    n_swallowed = mend_swallowed_links(bodies)
    if n_swallowed:
        print(f"    gave back the text {n_swallowed} link(s) had taken into "
              f"an address the recogniser never closed")
    if n_figures:
        print(f"    extracted {n_figures} figures")
    if n_blank_figures:
        print(f"    skipped {n_blank_figures} blank-page image(s)")
    if n_hand_figures:
        print(f"    skipped {n_hand_figures} figure(s) that were the "
              f"reader's hand, not a picture")
    if n_label_rules:
        print(f"    drew {n_label_rules} footnote rule(s) the layout labels "
              f"assert but the page never showed")
    if n_reattached:
        print(f"    reattached {n_reattached} superscript marker(s) the "
              f"layout had orphaned into standalone blocks")
    if n_phantoms:
        print(f"    suppressed {n_phantoms} phantom digit block(s) the "
              f"layout invented over prose — verified against the ink")
    if switches.SECOND_OPINION is not None:
        if switches.SECOND_OPINION.summary():
            print("    " + switches.SECOND_OPINION.summary())
        if switches.THIRD_OPINION is not None and switches.THIRD_OPINION.summary():
            print("    " + switches.THIRD_OPINION.summary())
        # also when it was asked nothing: an earlier build's account would
        # otherwise stay beside a book it no longer describes
        _sheet = switches.SECOND_OPINION.write_sheet()
        if _sheet:
            print(f"    second reader: its changes and the differences it "
                  f"did not settle are written out in full: {_sheet}")
    if n_doubled:
        print(f"    left out {n_doubled} picture(s) of a ruled box that a "
              f"block of the page already holds — a table printed as a "
              f"table is not printed again as a picture of one")
    if n_forms:
        print(f"    kept {n_forms} form block(s) out of the body — a "
              f"document facsimile's fields are not prose; their text "
              f"stays in the PDF layer")

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

    # The same reconciliation, for lines of type rather than pictures: a head
    # the layout pass labelled furniture on most pages it sometimes labels a
    # heading, and that copy lands inside the sentence running across the page
    # break. Here, beside the picture pass and for the same reason — before
    # anything downstream counts or links what a page holds.
    if strip_furniture:
        missed = strip_missed_running_heads(bodies, page_furniture)
        if missed:
            gone = {(pg, t[:90]) for pg, t in missed}
            label_heads = [h for h in label_heads if (h[0], h[2]) not in gone]
            shapes = sorted({head_shape(t) for _, t in missed})
            print(f"    dropped {len(missed)} running head(s) the layout "
                  f"labelled as content, on the book's own evidence that it "
                  f"prints the same line as furniture elsewhere"
                  + (f": {len(shapes)} distinct head(s)" if len(shapes) > 1
                     else f": {missed[0][1]!r}"))

    signatures = strip_signature_marks(bodies)
    if signatures:
        print(f"    took out {len(signatures)} signature mark(s) the printer "
              f"set at the foot of a gathering's first page: "
              + ", ".join(f"{mark} (page {i + 1})" for i, mark in signatures))
    # Subscripts and powers are set as such only where the page has no
    # born-digital text. Where it has, the layer is the better reading of
    # them: the recogniser reads a printed Y₁ as "Y_L" and e_D as "e_p", the
    # layer holds "Y1" and "eD", and the reconciler below takes the layer's
    # word. Set as <sub> first, the tag stood between the words and the
    # layer was no longer asked: 127 subscripts in the three economics
    # textbooks went back to the recogniser's letters on the first shelf
    # run of this rule. A flat "Y1" with the right figure is the better of
    # the two.
    bodies = [normalize_entry_heads(normalize_math(
        b, scripts=not (i < len(page_paths)
                        and os.path.exists(page_paths[i] + ".native"))))
        for i, b in enumerate(bodies)]
    n_promoted = normalize_note_heads(bodies)
    if n_promoted:
        print(f"    promoted {n_promoted} bold note head(s) to headings")
    n_reconciled = reconcile_native_text(bodies, page_paths)
    if n_reconciled:
        print(f"    reconciled {n_reconciled} word(s) against the book's "
              f"own born-digital text")
    if reconcile_native_text.put_back:
        print(f"    put back {sum(len(w.split()) for _, _, w, _ in reconcile_native_text.put_back)} "
              f"word(s) in {len(reconcile_native_text.put_back)} place(s) "
              f"where the PDF's own text prints them in the line and the "
              f"reading had passed over them:")
    for _pi, _before, _words, _after in reconcile_native_text.put_back:
        print(f"      page {_pi + 1}: “{_before} [{_words}] {_after}”")
    # After the layer has had its say and not before. A PDF's text layer
    # is flat: it holds "km2" for km², and a word it holds differently
    # from ours is taken from it. Set first, "km²" was one word the layer
    # spelled another way and came back "km2": the first whole-shelf run
    # of this rule showed it in four born-digital books.
    bodies = [set_unit_powers(b) for b in bodies]
    for _pi, _before, _figure, _after in reconcile_native_text.unplaced:
        print(f"    [?] page {_pi + 1}: the PDF's own text prints "
              f"“{_figure}” between “{_before}” and “{_after}”, and the "
              f"reading has no such figure there — left as read; worth "
              f"a look")
    # Straight after reconciliation, because that is where they come in, and
    # before typography, because two of them ARE quote marks.
    n_c1 = repair_c1_controls(bodies)
    if n_c1:
        print(f"    repaired {n_c1} punctuation mark(s) the text layer "
              f"delivered as control characters")
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
    n_folio_sup = strip_folio_superscripts(bodies, folios)
    if n_folio_sup:
        print(f"    dropped {n_folio_sup} page number(s) the layout had set "
              f"as a note marker")
        texts = [_strip_tags(b) for b in bodies]

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
                  f"({len(page_paths) - len(dropped)} kept)")
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
                 if i not in dropped and "<img" not in body
                 and _reads_as_empty(_strip_tags(body))}
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
                n = witness._classical_word_count(page_paths[i], min_len=3)
                if n is None:
                    # Nothing can tell an empty leaf from a page we failed to
                    # read, so this page is about to be deleted on our own
                    # word alone. Counted, and reported at the end.
                    switches.UNWITNESSED["blank"] += 1
                elif n >= 10:
                    second_opinion.add(i)
            if second_opinion:
                blank -= second_opinion
                for i in sorted(second_opinion):
                    print(f"    [!] page {i} read as empty, but a classical "
                          f"reader finds text on it — kept, not dropped")
        # And the inverse suspicion: a page with only a few words on
        # visually blank paper. The empty test above cannot see these —
        # they carry text, just not the book's.
        phantom = _phantom_blank_pages(
            page_paths,
            [_strip_tags(b) for b in bodies],
            ["<img" in b for b in bodies],
            skip=frozenset(dropped | blank))
        for i in sorted(phantom):
            w = len(re.findall(r"[A-Za-z]{2,}", _strip_tags(bodies[i])))
            print(f"    [~] page {i} carries {w} word(s) on visually blank "
                  f"paper and the classical reader finds none — "
                  f"show-through, dropped as blank")
        blank |= phantom
        if blank:
            orphans = {name for i in blank for name in page_figures[i]}
            book.items = [it for it in book.items
                          if getattr(it, "file_name", None) not in orphans]
            dropped |= blank
            print(f"    dropped {len(blank)} blank page(s) "
                  f"({len(page_paths) - len(dropped)} kept)")

    # The joiner runs FIRST: decisions were recorded against the shipped
    # book, whose page-turn words were already whole, so the rebuild must
    # reproduce the joins before the corrections can find their sites —
    # the other order stales every decision on a joined word (B1).
    jst = join_page_break_hyphens(bodies, dropped)
    if jst["joined"] or jst["refused"]:
        print(f"    page-break hyphens: {jst['joined']} split word(s) "
              f"rejoined across page turns"
              + (f", {jst['refused']} left as printed (compound or unknown)"
                 if jst["refused"] else ""))

    # Review happens in rounds, each reviewed against the book the
    # previous round produced, so the files apply in the order given — a
    # later round's "occurrence 1" counts among the text the earlier round
    # left. Merging rounds into one file would sort that history away.
    for rnd, path_ in enumerate(_decision_rounds(decisions_path), 1):
        dst = apply_reviewer_decisions(
            bodies, path_, kept_pages=len(bodies) - len(dropped))
        if dst["applied"] or dst["stale"]:
            print(f"    decisions (round {rnd}): {dst['applied']} reviewer "
                  f"correction(s) reapplied"
                  + (f", {dst['stale']} stale" if dst["stale"] else ""))

    # One placement of the printed contents serves every consumer below —
    # link_notes' chapter scoping and the nav alike. Two independent calls
    # could silently diverge as bodies mutate between them (C6).
    placed_and_pages = (_place_toc_entries(bodies, folios, dropped,
                                           book_title=title)
                        if folios is not None else ([], set()))

    if link_notes_flag:
        # chapter starts, where the printed contents can supply them: a leaf
        # titled "5 THE CASE OF…" placed at page P starts chapter 5 there.
        # Books number their chapters in Arabic or in Roman and both readings
        # land on the same scale, so "I" and "1" alike open chapter one. That
        # also covers the OCR reading a printed 1 as an I, which is what the
        # narrower rule here used to be for.
        chapter_starts = []
        if folios is not None:
            placed_ch, _tocp = placed_and_pages
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
        n_note_heads = count_notes_heads(bodies)
        # Only the strict heads gate the numbered footnote pass. Widening it
        # to READ References as well cost three books every same-page link
        # they had — 35, 14 and 4 — because a book can print numbered notes
        # at the foot of the page AND close with a References bibliography,
        # and the wider word calls that a notes section. The measurement
        # that let it through was taken against already-linked copies, where
        # the pass had nothing left to claim and returned zero either way;
        # the shelf rebuild is what caught it.
        fstats = link_footnotes(bodies, allow_numbered=(n_note_heads == 0),
                                foot_labelled=foot_labelled)
        if fstats["linked"] or fstats["skipped"]:
            print(f"    footnotes: {fstats['linked']} same-page notes linked"
                  + (f" ({fstats['numbered']} numbered)"
                     if fstats["numbered"] else "")
                  + f", {fstats['skipped']} left unsure")
        if n_note_heads >= 2:
            # Several "Notes" heads SUGGEST chapter-endnotes, where each
            # section closes the chapter it follows and position alone
            # scopes them. It is a guess at the layout and not a reading of
            # it: a back-of-book section whose title happens to appear twice
            # looks exactly the same from here.
            #
            # So both layouts are read and the book decides between them, by
            # how many of its own markers each one can account for. Asking
            # only whether the chapter layout found ANY section was the
            # narrow version of this test, and a book that prints "NOTES"
            # twice at the back of one continuous section answered yes with
            # two sections and 41 links, which silently vetoed the 875 the
            # other layout binds — 36 chapters' worth, aligned and agreeing.
            #
            # Counting links alone will not do it. Read as chapter-endnotes,
            # a back-of-book section binds every marker in the book to its
            # FIRST group — every chapter's note 1 to chapter one's note 1 —
            # so it can tie on links while most of them are wrong. What
            # separates the two readings is how many scopes each can offer:
            # one section swallowing the whole book against a group per
            # chapter, each of which link_notes refused to pair until its
            # numbers agreed with the body's.
            #
            # So the section reading is adopted only when it parsed groups,
            # did not refuse for misalignment, accounts for at least as many
            # markers, and tells more scopes apart than the chapter reading.
            # Anything less and the chapter reading stands, which assumes
            # least: position scopes it, and nothing is inferred.
            before = list(bodies)
            cstats = link_chapter_notes(bodies, dropped)
            chapter_result = list(bodies)
            trial = list(before)
            nstats = link_notes(trial, dropped=dropped,
                                chapter_starts=chapter_starts or None)
            if (nstats["groups"] and not nstats["misaligned"]
                    and nstats["linked"] >= cstats["linked"]
                    and nstats["groups"] > cstats["sections"]):
                bodies[:] = trial
                print(f"    notes: {nstats['linked']} markers linked across "
                      f"{nstats['groups']} chapters, {nstats['unlinked']} "
                      f"left plain"
                      + (f" (the {n_note_heads} Notes heads are one section, "
                         f"not {cstats['sections']})" if cstats["sections"]
                         else " (chapter-section layout did not apply)"))
            elif cstats["sections"]:
                bodies[:] = chapter_result
                print(f"    notes: {cstats['linked']} markers linked across "
                      f"{cstats['sections']} chapter sections, "
                      f"{cstats['unlinked']} left plain")
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
        fused = strip_fused_folios(bodies, {i: n for i, n in numbers.items()
                                            if i not in dropped})
        if fused:
            print(f"    took {len(fused)} page number(s) off the end of "
                  f"the page's last line, where the layout had run them "
                  f"in: " + ", ".join(f"{n} (page {i + 1})" for i, n in fused))
        read_at = {f.index for f in folios
                   if f.confident and f.number is not None}
        anchored = restored = 0
        for i in sorted(numbers):
            if i in dropped or i >= len(bodies):
                continue
            bodies[i] = (f'<span epub:type="pagebreak" role="doc-pagebreak" '
                         f'id="pgb-{i:04d}" '
                         f'aria-label="{folio_label(numbers[i])}"></span>'
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
            # Every outcome is said. An index whose heading was lost reads,
            # to the linker, exactly like a book with no index, and saying
            # nothing on that path is how Lost Worlds shipped 1,720 plain
            # references from a build whose log looked clean.
            if not ist["found"]:
                print("    index: no heading reading INDEX — nothing linked")
            elif ist["skipped_paragraph_index"]:
                print("    index: declares paragraph references, not pages — "
                      "left untouched as printed")
            elif ist["linked"]:
                print(f"    index: {ist['linked']} page reference(s) linked "
                      f"across {ist['pages']} index page(s)"
                      + (f", {ist['unknown_folio']} named folios outside the "
                         f"map left plain" if ist["unknown_folio"] else ""))
            elif not ist["reverted"]:
                print("    index: found, but "
                      + (f"none of its {ist['unknown_folio']} page "
                         f"reference(s) names a folio in the map"
                         if ist["unknown_folio"] else
                         "no page references were read in it")
                      + " — nothing linked")
            if ist["reverted"]:
                print(f"    [!] index: {ist['reverted']} page(s) reverted — "
                      f"a link would have changed the page's words")
    elif link_index_flag:
        print("    index: needs --check-folios for the folio map — nothing "
              "linked")

    # The book's own running heads, which name the chapter a reader is
    # inside and assert it on every page of it. Computed whether or not a
    # printed contents parsed: where one did, this cross-checks it; where
    # none did, it is a better skeleton than the layout's isolated labels,
    # because a span repeated twenty times is stronger evidence than a
    # single block that happened to be tagged a heading.
    furn_chapters, furn_stats = chapters_from_furniture(page_furniture)
    # Two sources describe the same skeleton, and the heading outline should
    # take whichever it can get. Running heads name the chapter a reader is
    # inside; the printed contents names it too, and on the books where the
    # heads are a document designation or a publisher's URL — every field
    # manual here — the contents is the only one that speaks. Measured: ten
    # books have contents hierarchy and no usable running heads, and their
    # outlines went uncorrected.
    outline_chapters = []
    if folios is not None:
        placed_ch, _ = placed_and_pages
        anchors = sorted({tgt: e.title for e, tgt, _ok in placed_ch
                          if e.depth < 2 and tgt is not None}.items())
        outline_chapters = [
            Chapter(title, pg,
                    (anchors[k + 1][0] - 1) if k + 1 < len(anchors)
                    else len(bodies) - 1)
            for k, (pg, title) in enumerate(anchors)]
    # The contents leads, as it does for the nav: it is the book's own
    # declaration, where the running heads are an inference from the
    # margins — and a weak one on some books. One manual's heads yielded
    # "May 2" and the bare word "Chapter" as chapter names, which matched
    # no heading on any page AND, merely by existing, kept the contents
    # from being consulted at all.
    n_promoted = 0
    if outline_chapters:
        n_promoted += promote_chapter_headings(bodies, outline_chapters)
    # Then the heads, even when the contents already spoke. The contents
    # names the divisions it ranks highest, and in a book built of Books or
    # Parts those are the Parts — its forty-four chapters sit a level below
    # the cut, so consulting the contents alone lifted two of them and left
    # the rest at h3 under their own chapter numbers. The heads name those
    # chapters directly. Letting the contents' mere existence silence them
    # cost more than it ever saved: a title that matches no heading on any
    # page promotes nothing, so a second source can only add.
    if furn_chapters:
        n_promoted += promote_chapter_headings(bodies, furn_chapters)
    if n_promoted:
        print(f"    lifted {n_promoted} chapter title(s) to the top of "
              f"the heading outline")

    # The last thing done to the words of a page: see write_display_math.
    mst = write_display_math(bodies)
    if mst["mathml"] or mst["text"]:
        print(f"    display equations: {mst['mathml']} set as MathML"
              + (f", {mst['text']} as their plain text"
                 + ("" if _pandoc() else " (pandoc is not installed)")
                 if mst["text"] else ""))

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

        # A page's printed number is a mark on it and not its content: with
        # the page list on, a page nothing was read from still had its mark,
        # was taken for a page with something on it, and lost the picture a
        # page without a number would have been given. The Knights of
        # Industry's handwritten note was shown only until the book was
        # built with page numbers.
        mark = _PAGE_MARKS.match(body)
        if not body[mark.end():].strip():
            body = body[:mark.end()] + _page_as_itself(book, page_paths[i], i)
        if not body.strip():
            # A page can legitimately end up with nothing on it — a blank leaf
            # whose only "content" was the blank-page image just dropped. An
            # empty body is not merely ugly: ebooklib parses every page when it
            # writes the spine, and an empty document raises there, taking the
            # whole build down at the last step.
            body = f'<p class="blank">[blank page {i + 1}]</p>'

        # No language of its own: ebooklib then writes the book's, which is
        # what --lang set. Every page used to say "en" whatever the book said.
        ch = epub.EpubHtml(title=f"Page {i + 1}", file_name=f"page_{i:04d}.xhtml")
        ch.content = f"<html><body>{body}</body></html>"
        if f'xmlns="{MATHML_NS}"' in body:
            # the package says which documents hold MathML (EPUB 3.3,
            # the "mathml" property); binding carries it to the chapter
            ch.properties.append("mathml")
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
        # one placement drives BOTH consumers — link_notes' chapter_starts
        # earlier and this nav — instead of two calls that could silently
        # diverge as bodies mutate between them (C6)
        placed, toc_pages = placed_and_pages
        mk_link = lambda href, title, uid: epub.Link(href, title, uid)
        mk_sect = lambda title, href: (epub.Section(title, href=href)
                                       if href else epub.Section(title))
        links, verified, missed, resolved = nav_from_placed(
            placed, pos_of, kept, bodies, toc_pages,
            make_link=mk_link, make_section=mk_sect)
        # The margins as a check on the parse, not only a fallback for its
        # absence — judged against the RESOLVED placements, after every
        # title hunt has settled, because before it a folio-less contents
        # looks entirely unplaced. Local copy: the shared placement drives
        # the note linker too, and the merge is a nav decision. Merging
        # rebuilds the nav once more from the settled targets; the second
        # pass has nothing left to hunt.
        missing_ch = (furniture_absent_from_contents(resolved, furn_chapters,
                                                     pos_of)
                      if furn_chapters else [])
        merged_placed, n_merged = merge_missing_furniture(
            resolved, missing_ch, len(furn_chapters))
        if n_merged:
            links, verified, missed, _ = nav_from_placed(
                merged_placed, pos_of, kept, bodies, toc_pages,
                make_link=mk_link, make_section=mk_sect)
        entries = [(i, t) for i, t, _ in find_numbered_entries(bodies) if i in pos_of]
        if entries:
            sect = [epub.Link(f"page_{i:04d}.xhtml", t, f"entry{i}") for i, t in entries]
            links.append((epub.Section("Entries"), sect))
        if links:
            toc_built = links
            print(f"    contents: {len(placed)} printed lines, "
                  f"{verified} title-verified, {missed} unplaced"
                  + (f", {len(entries)} numbered entries" if entries else ""))
    if toc_built and furn_chapters:
        # The printed contents is the book's own declaration and outranks an
        # inference from its margins — so a small disagreement is reported
        # and left alone. A parse that missed half of what the heads
        # describe has failed, and those chapters were merged in above.
        print(f"    running heads independently describe "
              f"{len(furn_chapters)} chapter(s)")
        if n_merged:
            print(f"    [!] the printed contents missed {n_merged} of them; "
                  f"added to the nav at the pages their running heads span")
    if toc_built:
        for w in contents_confidence(placed, len(bodies), missed,
                                     len(missing_ch) - n_merged):
            print(f"    [!] {w}")
    if not toc_built and furn_chapters:
        links = [epub.Link(f"page_{c.start:04d}.xhtml", c.title,
                           f"rh{k}-{c.start}")
                 for k, c in enumerate(furn_chapters) if c.start in pos_of]
        if len(links) >= 3:
            toc_built = links
            print(f"    contents: no printed contents page; built "
                  f"{len(links)} chapter(s) from the book's own running "
                  f"heads ({furn_stats['with_head']} pages carried one)")
    elif not toc_built and furn_stats.get("refused"):
        print(f"    [~] running heads did not describe chapters: "
              f"{furn_stats['refused']}")

    if not toc_built and len(label_heads) >= 4:
        # The layout model NAMED the book's headings; after the running-head
        # strip and the digit filter (markers masquerade as SectionHeader),
        # what survives is the book's own skeleton — measured 107 distinct
        # real heads on one book. Top rank becomes chapters, deeper ranks
        # nest beneath the chapter they follow.
        top = min(r for _, r, _ in label_heads)
        outline = []
        lh_id = iter(range(10 ** 9))
        for pg, r, t in label_heads:
            if pg not in pos_of:
                continue
            # one monotonic counter: pg repeats when a page carries two
            # heads and len(outline) freezes for non-top ranks — the old
            # pair shipped duplicate ids (C1)
            link = epub.Link(f"page_{pg:04d}.xhtml", t,
                             f"lh{next(lh_id)}-{pg}")
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
    if switches.UNWITNESSED["fabrication"] or switches.UNWITNESSED["blank"]:
        print(f"  [!] built without a witness: "
              f"{switches.UNWITNESSED['fabrication']} page(s) passed the fabrication "
              f"check unexamined, {switches.UNWITNESSED['blank']} page(s) were "
              f"deleted as blank on this pipeline's word alone. Install "
              f"tesseract and rebuild to have either checked.")
    restored = len(_read_pages_resiliently.restored)
    if restored:
        # One page the reader could not manage is a page; a hundred of them
        # is a failed run wearing a finished run's clothes. The restores are
        # a publisher's text layer laid down as one paragraph each — no
        # headings, no note markers, no reference or index entries — so a
        # book mostly made of them has lost its structure whatever its word
        # count says. None of them was cached, so a rebuild will read those
        # pages again; what this line exists to say is that it MUST.
        share = restored / max(1, len(page_paths))
        print(f"  [!] {restored} page(s) could not be read and were restored "
              f"from their own text layer — flat text, no structure."
              + (f" That is {share:.0%} of the book: the recognition server "
                 f"was failing, not the pages. Nothing was cached for them; "
                 f"rebuild once it is healthy." if share >= 0.1 else
                 " Nothing was cached for them; a rebuild will retry."))


# A page that is kept, and came through with nothing on it, was printed as
# "[blank page N]". Most such pages are blank. One on the shelf is not: a
# handwritten note reproduced in The Knights of Industry, which no reader
# could make words of, and which shipped as a blank page where the facsimile
# was the whole point of the leaf. What could not be read can still be shown.
#
# The floor is the ink of a page of text and not of that note. Of 348 pages
# with forty words or more, sampled from 75 books, the least inked measures
# 0.011; of the shelf's 171 kept pages with under five words, the part
# titles and dedications reach 0.013 and the genuinely empty ones 0.003. The
# note measures 0.138, which is the median of a full page of prose.
UNREAD_PAGE_INK = 0.02
_PAGE_MARKS = re.compile(r'(?:\s*<span epub:type="pagebreak"[^>]*>\s*</span>)*\s*')


def _page_as_itself(book, page_path: str, i: int) -> str:
    """The body for a page nothing was read from: a picture of the page, if
    there is as much ink on it as a page of text carries; else nothing."""
    from ebooklib import epub
    ink = _page_core_ink(page_path)
    if ink is None or ink < UNREAD_PAGE_INK:
        return ""
    try:
        with Image.open(page_path) as im:
            data, ext, mime = encode_figure(_encodable(im))
    except Exception:
        return ""
    name = f"images/unread_{i:04d}.{ext}"
    book.add_item(epub.EpubImage(uid=f"unread{i}", file_name=name,
                                 media_type=mime, content=data))
    print(f"    [~] page {i + 1}: nothing was read from it and it is not "
          f"blank — shown as a picture of the page")
    return (f'<figure><img src="{name}" alt="Page {i + 1}, as printed" '
            f'style="max-width:100%"/></figure>')


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


def turn_sideways_bodies(page_paths: list, cache: "OCRCache | None",
                         backend) -> list:
    """Turn the body of each page noted sideways (see TURN_MIN_CONFIDENCE),
    from its first reading as it lies, and put the turned sheet in its
    place in `page_paths`. Returns the pages turned, by index.

    The first reading is the cache's, or one made now; with --cache-only
    and none in the cache the page is left, and refused with the rest."""
    turned = []
    for k, path in enumerate(page_paths):
        try:
            with open(path + ".source.json", encoding="utf-8") as fh:
                src = json.load(fh)
        except (OSError, ValueError):
            continue
        how = src.get("sideways")
        if not how or src.get("turned"):
            continue
        items = cache.get(path) if cache is not None else None
        if items is None:
            if switches.CACHE_ONLY or backend is None:
                continue
            if warm_backend(backend) is not None:
                continue
            with Image.open(path) as im:
                img = im.convert("RGB")
            items = _read_pages_resiliently([path], [img], backend, cache).get(path)
            img.close()
            if items is None:
                continue
        furniture = [it.box for it in items if it.is_furniture and it.box]
        with Image.open(path) as im:
            sheet = _body_turned(im.convert("RGB"), how["degrees"], furniture)
        target = os.path.splitext(path)[0] + ".png"
        for old in (path + ".key", path + ".key1"):
            if os.path.exists(old):
                os.remove(old)
        if target != path and os.path.exists(path):
            os.remove(path)
        with open(target + ".key", "w", encoding="ascii") as fh:
            fh.write(_pixel_key(sheet))
        sheet.save(target, "PNG", dpi=(300, 300), compress_level=1)
        for side in (".source.json", ".layer.txt"):
            if target != path and os.path.exists(path + side):
                os.replace(path + side, target + side)
        # The turned sheet is the page and nothing else, as a facsimile is,
        # and it no longer stands where the PDF draws it: nothing may be
        # asked of the PDF by position on it.
        src.update(turned={**how, "furniture": len(furniture)}, facsimile=True)
        with open(target + ".source.json", "w", encoding="utf-8") as fh:
            json.dump(src, fh)
        page_paths[k] = target
        turned.append(k)
    if turned:
        print(f"    turned upright, printed sideways: page(s) "
              + ", ".join(f"{k + 1}" for k in turned)
              + " (the body turned; head and page number kept as printed)")
    return turned


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


from .guards import (  # noqa: F401  (moved; kept importable from here)
    DEGENERATE_REP6, _repeated_six_gram_share, _page_ink, LOOP_MAX_EXCESS,
    LOOP_LENGTH_BAND, _six_word_runs, _repeats_are_on_the_page,
    _page_witness, _looks_fabricated, COUNT_MIN_INTEGERS, COUNT_MIN_STEPS,
    COUNT_MAX_SAID, COUNT_MIN_WITNESSED, _TABLE, _TABLE_CELL,
    _BARE_INTEGER, _STANDALONE_INTEGER, _count_in_table,
    _count_is_on_the_page, drop_counting_tables, LAYER_MIN_OVERLAP,
    LAYER_MIN_WORDS, LAYER_MIN_ALIGNED, _layer_alignment,
    _contradicts_layer, UNSURE_CONFIDENCE, UNSURE_MIN_WORDS,
    VOUCHED_SHARE, BLOCK_VOUCHED, BLOCK_MIN_WORDS, _WITNESS_WORD,
    _share_witnessed, unvouched_reading, _page_picture,
    _native_layer_body, _layer_can_testify, PHANTOM_MIN_WORDS,
    PHANTOM_CORE_INK, _reads_as_empty, _page_core_ink,
    _phantom_blank_pages, _reading_agrees,
)




def warm_backend(backend) -> "str | None":
    """Prove the recogniser is serving before any of its answers are kept.

    Surya does not raise when its inference server is absent, still waking,
    or speaking a different protocol — it returns an EMPTY result, which is
    indistinguishable from a blank page. One run measured 1,536 such errors
    and blanked 191 of 197 pages; a single cold page is enough to lose a
    plate. So the first time a page must be read live, one synthetic page
    carrying known words goes through first: a backend that cannot read
    that cannot read anything, and the run stops rather than writing
    emptiness into a book.

    Only surya is probed. It is the backend with this failure mode, the
    others raise like ordinary code, and probing would cost a test double a
    scripted page it was never written to supply.

    Returns None when the backend answers, or what came back instead.
    """
    if getattr(backend, "name", "") != "surya":
        return None
    from .doctor import _EXPECTED, _MIN_MATCHED, _test_page
    try:
        items = backend.run_items([_test_page()])[0]
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"
    text = " ".join(_strip_tags(it.html or "") for it in items)
    if len(_EXPECTED & set(re.findall(r"[a-z]+", text.lower()))) >= _MIN_MATCHED:
        return None
    seen = " ".join(text.split())
    return f"read {seen[:60]!r}" if seen else "read nothing at all"


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
        """Whether the read produced anything the book is made of.

        A picture counts. A plate, a map, a frontispiece has no words on it,
        and judging such a page by its text alone calls a perfectly good
        read a failure: it is retried on every build, never cached (an
        "empty" read of an inked page is deliberately withheld), and re-read
        live for ever after — so the one time the server is slow, the page
        reads empty, ships no figure, and is dropped as blank. Two full-page
        plates left a book that way.
        """
        return any((it.figure is not None
                    or (it.html and _strip_tags(it.html).strip()))
                   for it in items if not it.is_furniture)

    def has_words(items):
        return any(it.html and _strip_tags(it.html).strip()
                   for it in items if not it.is_furniture)

    def unread_blocks(k=0):
        counts = getattr(backend, "last_unread_blocks", None) or []
        return counts[k] if k < len(counts) else 0

    out: dict[str, list[PageItem]] = {}
    batch = backend.run_items(images)
    unread = [unread_blocks(k) for k in range(len(images))]
    for k, (path, img, items) in enumerate(zip(paths, images, batch)):
        # Whether what we end up with came from the READER, or from the
        # page's own layer because the reader failed. A restore is the right
        # thing to ship and the wrong thing to keep: see the cache write.
        provisional = False
        # The page came back without its words although words were sent to
        # be read: the recogniser failed, whatever else it returned.
        silent = False
        if unread[k] and not has_words(items):
            retry = backend.run_items([img])[0]
            if has_words(retry):
                items = retry
            else:
                # Its pictures, and none of its text. A picture counts as a
                # reading below, so such a page used to be kept and cached
                # as one: Where There Is No Doctor shipped 64 pages as
                # their drawings alone, 19,866 words gone, and every rebuild
                # replayed them. The pictures are still worth having. The
                # text comes from the page's own layer where there is one,
                # and nothing is cached, so the page is read again.
                silent = provisional = True
                restored = _native_layer_body(path)
                if restored:
                    items = list(items) + restored
                print(f"    [!] {os.path.basename(path)}: "
                      f"{unread[k]} block(s) sent to be read came back "
                      f"with nothing, twice — "
                      + ("its text restored from the page's own "
                         "born-digital layer" if restored else
                         "left without its text")
                      + "; not cached")
        elif not has_text(items) and figure_has_content(img):
            retry = backend.run_items([img])[0]
            if has_text(retry):
                items = retry
            else:
                # Nothing at all from a page with ink on it is the other
                # way a recogniser that has stopped looks: not even the
                # layout came back.
                silent = True
                # Twice empty on an inked page. On a scan that stays empty —
                # honest, and rereadable later since nothing is cached. On a
                # born-digital page the book's own text is at hand, and a
                # blank leaf distinguishes itself: its layer holds only the
                # watermark and folio, which the restorer strips, leaving
                # nothing to restore. A page that had real words gets them.
                restored = _native_layer_body(path)
                if restored:
                    items, provisional = restored, True
                    print(f"    [!] {os.path.basename(path)}: read empty "
                          f"twice — restored from the page's own "
                          f"born-digital text")

        # …and the opposite failure: text where the page has none to give.
        # Invented prose attributed to the author is worse than a page left
        # unread, so this one refuses rather than guesses. A second reading
        # decides: a real page reads back the same, a made-up one usually
        # does not — and where it does, it is still looping, which shows.
        why = _looks_fabricated(items, img, path) \
            or _contradicts_layer(items, path)
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
                n_tess = witness._classical_word_count(path)
                if n_tess is None:
                    # The only witness this page had, and it is not here. The
                    # page passes unexamined — which is the honest outcome,
                    # but it must be counted and said, not simply skipped.
                    switches.UNWITNESSED["fabrication"] += 1
                elif n_tess < 5:
                    second = backend.run_items([img])[0]
                    if _reading_agrees(items, second) < 0.5:
                        why = ("read paragraphs where a classical OCR reads "
                               "nothing, and told a different story twice")
                        restored = _native_layer_body(path)
                        items = restored or []
                        provisional = bool(restored)
                        print(f"    [!] {os.path.basename(path)}: {why} — "
                              + ("restored from the page's own born-digital "
                                 "text" if restored else
                                 "left unread rather than filled with "
                                 "invention"))
                        if not restored:
                            _read_pages_resiliently.refused.append(path)
        elif why is not None:
            second = backend.run_items([img])[0]
            why2 = _looks_fabricated(second, img, path) \
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
                provisional = bool(items)
            if why is not None:
                restored = bool(items)
                print(f"    [!] {os.path.basename(path)}: {why} — "
                      + ("restored from the page's own born-digital text"
                         if restored else
                         "left unread rather than filled with invention"))
                if not restored:
                    _read_pages_resiliently.refused.append(path)

        out[path] = items
        # Such a page is a hard page or a recogniser that has stopped, and
        # only the recogniser can say which. It is asked at the first one
        # of a run, and again at every third, before the book goes on being
        # made of what it did not read.
        if silent:
            run = _read_pages_resiliently.silent
            run.append(path)
            if len(run) == 1 or len(run) % SILENT_RUN == 0:
                why = warm_backend(backend)
                if why is not None:
                    if cache is not None:
                        cache.save()
                    first = os.path.basename(run[0])
                    sys.exit(
                        f"the {backend.name} backend has stopped reading: "
                        + (f"{first} came back without its text"
                           if len(run) == 1 else
                           f"{len(run)} pages running came back without "
                           f"their text, from {first} on")
                        + f", and asked for a synthetic page of known words "
                        f"it {why}.\nNo book was written. Every page read "
                        f"before that is in the cache, and none of these "
                        f"is, so running this again once the backend is "
                        f"healthy carries on from {first}.")
        elif has_words(items):
            _read_pages_resiliently.silent.clear()
        if provisional:
            # The guard above already refuses to cache an EMPTY read, for
            # exactly this reason — but a page restored from its own layer
            # is not empty, so it walked past that guard into the cache.
            # Two books were built that way on one day: the server behind
            # recognition stopped answering at page 12 of one and page 92
            # of the other, every page after it was restored, and the
            # restores were kept as though they were readings. A layer
            # restore is one paragraph of running text, so those books lost
            # their headings, their note markers, their reference and index
            # entries, and kept their running heads in the body — and no
            # rebuild could recover, because the rebuild replayed the cache.
            # A restore ships in THIS build and is never kept.
            _read_pages_resiliently.restored.append(path)
        elif cache is not None:
            if has_text(items) or not figure_has_content(img):
                cache.put(path, items)
    return out


_read_pages_resiliently.refused = []
_read_pages_resiliently.restored = []
_read_pages_resiliently.silent = []     # the run of pages now without text


# A build used to finish whatever the recogniser did. Where There Is No
# Doctor was built on a day the server behind recognition stopped answering
# after page 119 of 503: 318 pages were laid down from the PDF's own text as
# one paragraph each, 64 more kept their drawings and lost their words, the
# build said so in one line at the end, and the book was installed.
#
# The first build to run under this was the repair of that book. Two hours
# and forty minutes in, at page 492, the server stopped again. The build
# stopped with it, and run again it read the eight pages that were left.
#
# It asked for the proof only after three such pages running, at first. The
# same night a rebuild of The Knights of Industry met a server that was
# refusing connections from its first live page. That book has two pages
# its cache has never held, so two is all it read: one came back empty and
# was dropped as blank, the page count fell by one, and the book's seven
# reviewed corrections were refused as recorded against another count. The
# proof is asked for at the first such page now.
SILENT_RUN = 3          # in a run of such pages, how often the proof is asked


# How one word can be set two ways: the publisher's curly quotes and primes
# where the recogniser read straight ones, a true minus for a hyphen, a
# ligature, an ellipsis as one character. None of them changes which word
# it is, so none of them needs the pairing proven.
_SETTING = str.maketrans({
    "‘": "'", "’": "'", "‚": "'", "‛": "'", "′": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"', "″": "''",
    "–": "-", "—": "-", "‒": "-", "−": "-", "…": "...",
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "ﬅ": "st", "ﬆ": "st",
})
# What some layers write where they broke a line: "Air\x02borne".
_CONTROL = re.compile(r"[\x00-\x1f]")
# What a layer holds for a glyph set in a symbol font: a private-use code
# point, or a control character. The publisher's bullet is U+F0B7 in the
# layer of Principles of Macroeconomics and 0x83 in FM 3-06, and no reader
# has a glyph for either; the recogniser read the bullet off the page. The
# layer says which words a page prints. It does not get to say that one of
# them is a character nobody can print.
#
# One kind of control character is not that. A layer that stores its
# punctuation as WinAnsi bytes holds 0x92 for the apostrophe and 0x97 for
# the dash, and repair_c1_controls puts those back once the page is
# assembled: FM 21-10 sets "unit's" and "LEADERS'" that way throughout, and
# refusing those words left the recogniser's straight apostrophe where the
# book has a curly one. What that repair will mend is as good as printed.
_NO_GLYPH = re.compile("[\x7f-\x9f\ue000-\uf8ff\U000f0000-\U0010fffd]")


def _no_glyph(word: str) -> bool:
    """Whether a word of the layer holds a character nobody can print."""
    for m in _NO_GLYPH.finditer(word):
        meant = C1_REPAIR.get(m.group(0))
        if meant is None or not unicodedata.category(meant).startswith("P"):
            return True
    return False


# The recogniser's own markup is setting as well. What it takes for
# mathematics it writes the way TeX would: a command and the braces of its
# argument (\text{rate}), the mark of a subscript or a power (S_3, r^n), a
# backslash before a sign TeX keeps for itself (\$, \%). The page prints
# none of it, and inside a page the layer's plain word is taken over it like
# any other. At a page's edge it was a different word and stayed: three
# economics textbooks ended 27 pages on "S_3." or "\$446,280.99" where the
# rest of the page reads S3, and three display equations there kept the
# closing brace of a \text{ the layer had already unwrapped. Those 28
# words are all this admits. Of the shelf's 134 books with born-digital
# pages, 59 have markup at the edge of any block; built as far as the
# reconciler, they refuse 395 words at a page's edge, and the other 367
# are still refused: a URL the layer breaks in two, a rule to write on, a
# folio.
_MARKUP = re.compile(r"\\[A-Za-z]+|[\\_^{}]")


def _same_word(ours: str, layer: str) -> bool:
    """Whether two readings are one word set differently — see _SETTING and
    _MARKUP — allowing the first letter's case, the sentence boundary the
    reconciler already concedes to the layer."""
    b = layer.translate(_SETTING)
    for a in (ours, _MARKUP.sub("", ours)):
        a = a.translate(_SETTING)
        if a and a[1:] == b[1:] and a[:1].lower() == b[:1].lower():
            return True
    return False


# At a page's edge the pairing of our word with the layer's is unproven,
# and only the same word set differently is taken (_same_word). That kept
# folios and running heads from being pasted over headings. It also kept
# the recogniser's slips with the marks around a word: a quote it dropped
# ("People" where the page prints "‘People", Work, Sex & Power p. 8), one
# it added ("employment.'"), the doubled dash a bibliography sets for "the
# same author" read as one ("—" for "——", Nomads, Empires, States). A word
# whose letters are the layer's, and which differs only in quotes and
# dashes standing at its ends, is the same word: the layer says which marks
# the page prints. Inside a word a mark counts, so "well-known" is never
# "wellknown"; a word of nothing but dashes is the same as another such.
_EDGE_MARKS = "‘’‚‛'\"“”„‟′″«»—–‒−-"


def _same_but_its_marks(ours: str, layer: str) -> bool:
    a, b = ours.translate(_SETTING), layer.translate(_SETTING)
    if a == b:
        return False
    ca, cb = a.strip(_EDGE_MARKS), b.strip(_EDGE_MARKS)
    if ca or cb:
        return ca == cb
    dashes = "—–‒−-"
    return bool(ours) and bool(layer) and all(c in dashes for c in ours + layer)


# Words the publisher prints and the reading passed over.
#
# A recogniser that writes a page out word by word does what a copyist
# does. It writes a doubled word once ("to say that that the world" came
# out with one that), takes the second of two words that look alike for the
# first ("1.4 billion toe to 2.5 billion" lost its unit, "as well as fora
# for teaching" its fora), and now and then drops down a line from one
# word to another like it: Kill Boxes lost eleven words between "prevents"
# and "pretends". Where There Is No Doctor gives a dose as "children 9 to
# 14 years: 1 ½ tablets" and it was read as "½ tablets". On a born-digital
# page the layer has every one of these, and the reconciler, which only
# ever exchanged one word for another, left them out.
#
# Most of what the layer has and we lack is not that. The layer's order is
# the typesetter's: it reads a table across its columns, puts a running
# head after the last line of the text, and sets the page reference from
# the next column in mid-sentence ("the first day: soak 102 joint in cold
# water"). Of 263 runs of the layer's words with nothing of ours against
# them, on the 26,728 born-digital pages of 134 books that differ from
# their layer at all, 38 are words the reading passed over. Four things
# tell them, and each was needed:
#
#   the words round the gap agree, three on each side (a running head's
#     "oil price shocks" lined up with a caption's "two oil shocks");
#   each word is a word, or a figure (the layer runs words together where
#     a heading is set tight: "changehorizontal");
#   the page does not have it already: the layer has each word more often
#     than the page does (a chapter's number read after its title);
#   it stands in the line: in the PDF each word follows the one before it
#     on the same line of type, or opens the line below.
#
# All 38 were looked at on the page: 59 words in 28 books, two of them
# whole lines, 13 a doubled word the book prints (some of them the book's
# own slips, which are the book's). What the last test alone refuses stays
# on the list a build prints, as before.
PUT_BACK_ANCHOR = 3         # agreeing words wanted on each side of the gap
PUT_BACK_LONGEST = 40       # words; nothing longer than twelve was found
_VULGAR_SIGNS = "½¼¾⅓⅔⅛⅜⅝⅞"
_PRINTED_RUN = re.compile(r"[^\W_]+|[%s]" % _VULGAR_SIGNS)


def _printed_runs(words) -> list:
    """The letters and figures of some words, each run apart, lower case:
    what is counted when asking whether a page has a word."""
    return [r.lower() for r in _PRINTED_RUN.findall(" ".join(words))]


def _layer_geometry(page_path: str) -> "list | None":
    """A typeset page's words in the PDF's own order, each with where it
    stands: (word, left, bottom, right, top) in points. None for a scan, or
    with no PDF to ask. Read a character at a time, so that a box is the
    box of the character it is filed under."""
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile") or src.get("scan"):
            return None
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
        textpage = pdfium.PdfDocument(src["pdf"])[src["page"]].get_textpage()
        words, cur = [], None
        for i in range(textpage.count_chars()):
            ch = chr(pdfium_c.FPDFText_GetUnicode(textpage.raw, i))
            if ch.isspace():
                if cur:
                    words.append(tuple(cur))
                cur = None
                continue
            if ch == "\x00":
                continue
            left, bottom, right, top = textpage.get_charbox(i)
            if cur is None:
                cur = [ch, left, bottom, right, top]
            else:
                cur[0] += ch
                cur[1], cur[2] = min(cur[1], left), min(cur[2], bottom)
                cur[3], cur[4] = max(cur[3], right), max(cur[4], top)
        if cur:
            words.append(tuple(cur))
        return words
    except Exception:
        return None


def _follows(a: tuple, b: tuple) -> bool:
    """Whether word b comes straight after word a in a line of type: beside
    it, a word space off, or opening the line below it."""
    h = max(a[4] - a[2], b[4] - b[2], 1e-3)
    if min(a[4], b[4]) - max(a[2], b[2]) >= 0.5 * min(a[4] - a[2], b[4] - b[2]):
        # the same line: a word space is a third of the type's height and
        # a justified line's twice that; the gutter of a table is more
        return -0.6 * h <= b[1] - a[3] <= 1.6 * h
    # the next line down, and back towards the margin it started from
    return b[4] <= a[2] + 0.3 * h and a[2] - b[4] <= 1.6 * h and b[1] < a[1]


def _stands_in_the_line(geometry: list, layer_words: list, b0: int,
                        b1: int) -> bool:
    """Whether layer_words[b0:b1] is set in the line between the word before
    it and the word after: found in one place in the PDF, each word
    following the last."""
    printed = [g[0] for g in geometry]
    for lead in (2, 1):
        if b0 - lead < 0 or b1 + lead > len(layer_words):
            continue
        run = layer_words[b0 - lead:b1 + lead]
        at = [i for i in range(len(printed) - len(run) + 1)
              if printed[i:i + len(run)] == run]
        if len(at) == 1:
            first = at[0] + lead - 1
            return all(_follows(geometry[i], geometry[i + 1])
                       for i in range(first, first + (b1 - b0) + 1))
    return False


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

    And words the layer prints where we have none are put back, where they
    are words the reading passed over and not the layer's own order: see
    PUT_BACK_ANCHOR. They are listed in reconcile_native_text.put_back and
    are not counted in what this returns.

    "Bounded" is enforced, not assumed. The aligner pairs a page's first
    words with the layer's first words whatever they are, and the layer's
    order is the typesetter's, not the reader's: InDesign writes a drop
    folio first and a chapter opening's heading last. On Lost Worlds' first
    index page that paired our "INDEX" with the layer's "248", the heading
    shipped as <h1>248</h1>, and --link-index, which finds the index by that
    heading, linked none of its 1,720 references. Across the 53 born-digital
    books on the shelf, 329 substitutions at a page's first or last words
    reached a book; 231 set the same word the publisher's way — curly
    quotes, en dashes, a true minus, primes — and most of the other 98 were
    somebody else's words: folios and running heads pasted over index heads
    and a caption's "Figure 8", "barter-free" turned "barter-technology".
    Where one side is open, only the same word set differently is taken
    (_same_word).

    A control character marks where the layer broke a line. Usually that is
    inside the very word it is paired with ("Air\x02borne"), the run is
    sound, and the rest of it is taken as before: 438 of the shelf's 612
    such runs. The other 174, 138 of them in one book's notes, were URLs the
    layer holds as two words where we read one; conceding the clean half
    paired our whole address with its first half and shipped "www.scmp.com/;"
    where the page prints the full path. A garbled word that is not the word
    it is paired with refuses its whole run.
    """
    import difflib

    changed = 0
    reconcile_native_text.unplaced.clear()
    reconcile_native_text.put_back.clear()
    vocabulary: "set | None" = None      # the book's own words, if asked for
    for pi, body in enumerate(bodies):
        if pi >= len(page_paths) or not body:
            continue
        base = page_paths[pi]
        if not os.path.exists(base + ".native"):
            continue
        try:
            with open(base + ".layer.txt", encoding="utf-8") as fh:
                layer_words = fh.read().split()
        except (OSError, Image.DecompressionBombError):
            # this pipeline stores pages losslessly, so one unusually
            # high-resolution page can trip Pillow's bomb guard — which is
            # an Exception, not an OSError, and used to crash the whole
            # folio audit instead of skipping the page (A6)
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
        puts: list[tuple[int, list]] = []    # (the word before, the words)
        ops = sm.get_opcodes()
        on_layer = on_page = geometry = None

        def agreed(n: int, step: int) -> int:
            """How many words in a row agree on that side of op n: equal,
            or the same word set the publisher's way."""
            count = 0
            n += step
            while 0 <= n < len(ops) and count < PUT_BACK_ANCHOR:
                kind, x0, x1, y0, y1 = ops[n]
                if kind != "equal" and not (
                        kind == "replace" and x1 - x0 == y1 - y0
                        and all(_same_word(a, b) for a, b in
                                zip(ours[x0:x1], layer_words[y0:y1]))):
                    break
                count += x1 - x0
                n += step
            return count

        for n, (tag, a0, a1, b0, b1) in enumerate(ops):
            if tag == "insert" and 0 < a0 < len(ours) \
                    and toks[a0 - 1][0] == toks[a0][0]:
                words = layer_words[b0:b1]
                printed = _printed_runs(words)
                passed_over = (
                    b1 - b0 <= PUT_BACK_LONGEST and bool(printed)
                    and not any(_CONTROL.search(w) or _no_glyph(w)
                                for w in words)
                    and agreed(n, -1) >= PUT_BACK_ANCHOR
                    and agreed(n, +1) >= PUT_BACK_ANCHOR)
                if passed_over:
                    from .adjudicate import _is_word, fold_word
                    for run in printed:
                        if run.isdigit() or run in _VULGAR_SIGNS \
                                or _is_word(fold_word(run)):
                            continue
                        if vocabulary is None:
                            vocabulary = {
                                fold_word(w) for b in bodies for w in
                                re.findall(r"[^\W\d_]+", _strip_tags(b))}
                        if fold_word(run) not in vocabulary:
                            passed_over = False
                            break
                if passed_over:
                    if on_layer is None:
                        on_layer = Counter(_printed_runs(layer_words))
                        on_page = Counter(_printed_runs(ours))
                    wanted = Counter(printed)
                    passed_over = all(on_layer[r] - on_page[r] >= c
                                      for r, c in wanted.items())
                if passed_over:
                    if geometry is None:
                        geometry = _layer_geometry(base) or []
                    passed_over = bool(geometry) and _stands_in_the_line(
                        geometry, layer_words, b0, b1)
                if passed_over:
                    puts.append((a0 - 1, words))
                    on_page.update(printed)
                    reconcile_native_text.put_back.append(
                        (pi, " ".join(ours[max(0, a0 - 3):a0]),
                         " ".join(words), " ".join(ours[a0:a0 + 3])))
                elif b1 - b0 <= 2 and any(
                        ch.isdigit() or ch in _VULGAR_SIGNS
                        for w in words for ch in w):
                    # A figure the publisher prints between two words we
                    # read one after the other, and not one that can be put
                    # back on the tests above. It is said and not acted on:
                    # see reconcile_native_text.unplaced.
                    reconcile_native_text.unplaced.append(
                        (pi, " ".join(ours[max(0, a0 - 2):a0]),
                         " ".join(words), " ".join(ours[a0:a0 + 2])))
            if tag != "replace" or (a1 - a0) != (b1 - b0) or (a1 - a0) > 3:
                continue
            run = list(zip(range(a0, a1), layer_words[b0:b1]))
            # Some layers carry control characters at line-break points
            # ("Cana\x02da"); deferring to those would corrupt a word we
            # read correctly. Authority does not extend to garbage — and
            # where the garbled word is not even the word it is paired with,
            # not to the rest of its run either (see the docstring).
            if any(_CONTROL.search(cand)
                   and not any(_same_word(ours[ti], w) for w in (
                       _CONTROL.sub("", cand), _CONTROL.sub("-", cand)))
                   for ti, cand in run):
                continue
            # The first or last run on the page has agreement on one side
            # only, so the pairing itself is unproven there.
            open_side = a0 == 0 or a1 == len(ours)
            for ti, cand in run:
                if _CONTROL.search(cand):
                    continue
                if _no_glyph(cand) and not _no_glyph(ours[ti]):
                    continue
                # The layer is authority on WHICH words are on the page, not
                # on how they are set. A book that prints a heading in small
                # capitals stores it lowercase and lets the font do the
                # capitals; Surya reads the printed glyphs. Deferring here
                # took the source characters over the rendered page and
                # downcased twenty-six section headings in one book —
                # "CONCLUSION" eight times over — and turned an economist
                # named McKinnon into Mckinnon. Measured across both
                # born-digital books available: every case-only substitution
                # was damage and none was a correction.
                # A difference confined to the FIRST letter is not that: it
                # is a sentence boundary, and the layer knows where those
                # are better than a recogniser reading one block at a time.
                # One book had a figure reference swallowed, leaving the
                # recogniser to capitalise the "illustrates" that followed;
                # the publisher's lower case is right there. Anything deeper
                # than the first letter — CONCLUSION, McKinnon, AAs — is the
                # page's own setting and stays ours.
                # A one-letter word has nothing after its first letter, so it
                # cannot show which of the two this is, and conceding it
                # broke a heading down the middle: FINDING A JOB came back
                # as "FINDING a JOB". With no evidence, the page keeps what
                # it prints.
                mine = ours[ti]
                if mine.lower() == cand.lower() and (
                        len(mine) == 1 or mine[1:] != cand[1:]):
                    continue
                if open_side and not (_same_word(mine, cand)
                                      or _same_but_its_marks(mine, cand)):
                    continue
                if mine != cand:
                    subs.append((ti, cand))
        if not subs and not puts:
            continue
        # A word exchanged takes the place of its word; words put back go
        # in after the word before them. Applied from the back, so that
        # recorded offsets stay valid.
        edits = [(toks[ti][0], toks[ti][1], toks[ti][2], html.escape(word))
                 for ti, word in subs]
        edits += [(toks[ti][0], toks[ti][2], toks[ti][2],
                   " " + html.escape(" ".join(words))) for ti, words in puts]
        for k, a, z, new in sorted(edits, reverse=True):
            parts[k] = parts[k][:a] + new + parts[k][z:]
        changed += len(subs)
        bodies[pi] = "".join(parts)
    return changed


# Figures the layer prints inside a run of text where we read none. Where
# There Is No Doctor gives a dose as "children 9 to 14 years: 1 ½ tablets",
# and the recogniser read "½ tablets". The layer has the 1, and the
# reconciler only ever exchanges one word for another, so it stood.
#
# That one is put back now (see PUT_BACK_ANCHOR). What is not: across the
# 26,044 born-digital pages of 133 books, 111 such words sit between two
# words we read together, and most are the layer reading a table across its
# columns, or a page reference from the next column landing in mid-sentence
# ("soak 102 joint in cold water"). Twenty of the 111 hold a figure, in
# seven books. A figure is the word a reader acts on, so those that cannot
# be put back are listed when the book is built, for someone to look at:
# three of them are that table's page references, which the page does lack.
reconcile_native_text.unplaced = []
reconcile_native_text.put_back = []


def _collect_page_items(page_paths: list[str], backend: "OCRBackend",
                        ocr_cache: str | None,
                        cache_obj: "OCRCache | None" = None
                        ) -> list[list[PageItem]]:
    """Every page's items, through the same cache the EPUB build fills.

    The instance is shared when the caller already loaded it: this cache
    embeds every figure as base64, and re-parsing the whole gzip for the
    --epub + --pdf-text-layer combo — the module docstring's own example —
    was the single largest avoidable cost in that invocation (E6)."""
    cache = cache_obj if cache_obj is not None else (
        OCRCache(ocr_cache, backend.name, getattr(backend, "langs", ["en"]))
        if ocr_cache else None)
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


from .searchable_pdf import (  # noqa: F401  (moved; kept importable from here)
    _LAYER_MIN_FONT, _force_text_into, _tesseract_word_boxes,
    build_searchable_pdf,
)




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
    serve_stale_cache: bool = False,
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

    if output_pdf and pdf_text_layer:
        # A checkpoint, not the deliverable: the geometry above is the most
        # expensive unrepeatable work in the run — with the dewarp ladder
        # and its witnesses, twenty-odd minutes on a real book — and the
        # prepared pages live in a temp directory that dies with the
        # process. One killed build re-paid the whole ladder to resume.
        # Written pages-only here, the facsimile plus the page map already
        # on disk make a crashed run resumable with --pages-from, geometry
        # never repaid; the text-layered version replaces this file at the
        # end of the run as before.
        print(f"\n[+] Checkpointing pages-only PDF → {output_pdf}")
        with open(output_pdf, "wb") as fh:
            fh.write(img2pdf.convert(page_paths))
        print(f"  {len(page_paths)} prepared page(s) safe; the text layer "
              f"replaces this file when the run completes")

    shared_cache = (OCRCache(ocr_cache, backend.name, backend.langs,
                             serve_stale=serve_stale_cache)
                    if (ocr_cache and backend is not None) else None)
    if output_epub and backend is not None:
        print(f"\n[+] Building EPUB → {output_epub}")
        build_epub(page_paths, output_epub, backend, title, embed_images, dedupe,
                   review_sheet, ocr_cache, page_ids=kept_ids,
                   check_folios=check_folios, drop_blank=drop_blank,
                   cover_spec=cover_spec, link_notes_flag=link_notes_flag,
                   link_citations_flag=link_citations_flag,
                   link_index_flag=link_index_flag,
                   typography_flag=typography_flag,
                   decisions_path=decisions_path,
                   serve_stale_cache=serve_stale_cache,
                   cache_obj=shared_cache)

    if output_pdf and pdf_text_layer:
        # After the EPUB, so a shared cache is already warm — and the reads
        # carry geometry only if made by a version that keeps it, so a stale
        # cache yields an honest count of unplaced blocks, not wrong ones.
        if backend is None:
            sys.exit("--pdf-text-layer needs an OCR backend")
        print(f"\n[+] Saving searchable PDF → {output_pdf}")
        items_pp = _collect_page_items(page_paths, backend, ocr_cache,
                                       cache_obj=shared_cache)
        # The layer's rule is that no block is ever dropped — but that rule
        # protects imperfectly PLACED text, and these pages' text is not
        # imperfectly placed, it is fabricated: show-through read as words.
        # Searchable phantoms lead a searcher to blank paper. Same two
        # witnesses as the EPUB gate; the caches make the re-ask free.
        pdf_texts = [" ".join(_strip_tags(it.html or "") for it in pg)
                     for pg in items_pp]
        ghost = _phantom_blank_pages(
            page_paths, pdf_texts,
            [any(it.figure is not None for it in pg) for pg in items_pp])
        # A reading that is nothing but the recogniser's bracketed asides
        # is not book text either — searchable "[Faded text block...]" only
        # ever leads a searcher to blank paper.
        ghost |= {i for i, t in enumerate(pdf_texts)
                  if t.strip() and _reads_as_empty(t)}
        if ghost:
            for i in ghost:
                items_pp[i] = []
            print(f"  [~] {len(ghost)} page(s) read only show-through; "
                  f"their text is left out of the layer")
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


def bind_book(epub_path: str) -> "dict | None":
    """The built book's pages bound into chapters, and a line to say so.

    The build writes a file to a page, because a page is what it reads,
    checks and corrects. A reader does not want one: every printed page
    then starts a new screen, half of them in the middle of a sentence. A
    book the binding will not answer for is left as it was built, which is
    a book that reads worse and is wrong in nothing."""
    from . import binding
    try:
        st = binding.bind_epub(epub_path)
    except binding.CannotBind as exc:
        print(f"\n[!] Left a file to a page: {exc}. The book is complete; "
              f"it starts a new screen at every printed page.")
        return None
    except Exception as exc:    # the book is built, and is worth more than
        # the way its files are cut: whatever went wrong in the binding,
        # the book as built is still there (nothing is replaced until the
        # bound one is whole) and the run ends as a run that built a book
        print(f"\n[!] Left a file to a page: the binding failed "
              f"({type(exc).__name__}: {exc}). The book is complete; it "
              f"starts a new screen at every printed page.")
        return None
    print(f"\n[+] Bound {st['pages']} page(s) into {st['files']} file(s) by "
          f"chapter: {st['joined']} paragraph(s) that ran over a page are "
          f"one paragraph"
          + (f", {st['words']} of them broken inside a word"
             if st.get("words") else "")
          + (f"; the notes or figures of {st['held']} page(s) wait for the "
             f"paragraph to end" if st["held"] else "")
          + ". Every page's start and printed number are marked where "
            "they were.")
    return st


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
    dewarp: bool = True,
    no_warp: bool = False,
    split_spreads: bool = True,
    rotate: int = 0,
    pages_from: str | None = None,
    dewarp_scans: bool = False,
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
    serve_stale_cache: bool = False,
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
            page_paths = load_pages_from_pdf(
                pages_from, pages_dir,
                former_keys=bool(ocr_cache)
                and OCRCache.filed_by_bytes(ocr_cache),
                turns_path=(ocr_cache + TURNS_SUFFIX) if ocr_cache else None)
            if dewarp_scans:
                dewarp_scan_pages(page_paths, pages_dir)
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
                    decisions_path=decisions_path,
                    serve_stale_cache=serve_stale_cache)
            return

        if from_images:
            image_paths = collect_image_paths(from_images)
            print(f"\n[1/2] Reading {len(image_paths)} photographs from {from_images}…")
            page_paths, page_ids = pages_from_images(
                image_paths, pages_dir, min_area_ratio, rotate, no_warp,
                enhance, deskew, split_spreads, motion_threshold,
                dewarp=dewarp,
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
                    decisions_path=decisions_path,
                    serve_stale_cache=serve_stale_cache)
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
        n_rejected = n_split = n_rectified = n_no_model = 0
        refused: list[tuple[str, int, int]] = []
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
            parts, info = capture_pages(img, rectify=not no_warp and dewarp, enhance=enhance,
                                        deskew=deskew, split_spreads=split_spreads,
                                        workdir=pages_dir, min_area_ratio=min_area_ratio)
            n_rectified += info["rectified"]
            n_split += info["split"]
            if info["kept_photo"] == "no rectifier":
                n_no_model += 1
            elif info["kept_photo"]:
                refused.append((f"run {run_ordinal}", *info["kept_photo"]))
            for part_idx, part in enumerate(parts):
                out = os.path.join(pages_dir, f"page_{len(page_paths):04d}.jpg")
                Image.fromarray(cv2.cvtColor(part, cv2.COLOR_BGR2RGB)).save(
                    out, "JPEG", quality=92, dpi=(300, 300)
                )
                page_paths.append(out)
                page_ids.append(page_id(run_ordinal, part_idx))
            os.remove(path)  # raw frame is no longer needed

        print(f"  {len(page_paths)} pages kept (discarded {n_rejected} mid-turn "
              f"fragments); {n_rectified} frame(s) flattened by the rectifier")
        _report_kept_photos(n_no_model, refused)
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
                    decisions_path=decisions_path,
                    serve_stale_cache=serve_stale_cache)

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
    parser.add_argument("--page-files", action="store_true",
                        help="Leave the EPUB one file to a printed page, as "
                             "it is built. By default the pages are then "
                             "bound into chapters: a paragraph that runs "
                             "over the page is one paragraph, the notes at "
                             "the foot of the page wait for it to end, and "
                             "a reader turns the screen where the text "
                             "does and not where the paper ran out. Every "
                             "page's start and printed number stay marked "
                             "either way, and `codicology unbind` takes a "
                             "bound book apart again")
    parser.add_argument("--pdf-text-layer", action="store_true",
                        help="Give the --pdf output a positioned, invisible text "
                             "layer from the OCR, so it searches and copies like a "
                             "born-digital file. Needs a cache written with geometry "
                             "(re-OCR with a fresh cache for books read before)")
    parser.add_argument("--batch-size", type=int, default=None, metavar="N",
                        help="how many pages are read at once (default: 4). "
                             "This also sizes the inference server's slots, "
                             "because a slot with no page in it is only "
                             "memory. 4 is the measured knee on an M1 Pro — "
                             "nine percent per page on the way up, nothing "
                             "past it; a larger GPU may keep climbing, a "
                             "smaller one may not reach it. Measure before "
                             "trusting a number: run each setting twice, "
                             "early and late, or a session's own drift will "
                             "pose as a result")
    parser.add_argument("--ocr", choices=sorted(OCR_BACKENDS), default="surya",
                        help="OCR backend used for --epub. Surya is the one "
                             "there is: everything this pipeline does beyond "
                             "reading characters depends on the layout it "
                             "returns")
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
    parser.add_argument("--cache-only", action="store_true",
                        help="Read no page on this run: build from what "
                             "--ocr-cache holds, and stop before doing "
                             "anything if a page is missing from it. A "
                             "rebuild made this way is the cache and the "
                             "code and nothing else, so two of them can be "
                             "compared")
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
    parser.add_argument("--keep-stale-cache", action="store_true",
                        help="serve cached pages that predate layout labels "
                             "instead of reading them again. Faster, and the "
                             "label consumers stay dark on those pages")
    parser.add_argument("--apply-decisions", metavar="JSON", default=None,
                        action="append",
                        help="review-sheet decisions to reapply at build time, "
                             "before the linkers run — the reviewer's "
                             "corrections survive rebuilds; sites that no "
                             "longer exist are reported stale, never guessed. "
                             "Repeat the flag for successive review rounds; "
                             "they apply in the order given")
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
    parser.add_argument("--second-reader", action="store_true",
                        help="Ask a second reader about what the first reads "
                             "worst: tables of figures and equations set on "
                             "their own lines. On a typeset page it is asked "
                             "only where the PDF's own text does not bear the "
                             "first reading out, and its table or equation is "
                             "printed only where that text prefers it. On a "
                             "scan nothing can say which reader is right, so "
                             "its equation is taken only where the two differ "
                             "in figures alone and it has no fewer of them, "
                             "and every other difference is listed for a "
                             "look, in the log and in a .second.txt beside "
                             "the cache. Where a third reader is installed "
                             "too (third.py), a scan's equation on which the "
                             "two differ is put to it and the reading two of "
                             "the three have at every point is printed. "
                             "Needs the model's two .gguf "
                             "files in ~/.cache/codicology/second/ (see "
                             "second.py); its readings are cached beside "
                             "--ocr-cache")
    parser.add_argument("--without-witness", action="store_true",
                        help="Run even though tesseract is not installed. Two "
                             "checks depend on it and both then pass without "
                             "examining anything: whether a page with no text "
                             "layer was invented, and whether a page about to "
                             "be deleted as blank was merely one this pipeline "
                             "failed to read. The run says at the end how many "
                             "pages went unwitnessed")
    parser.add_argument("--no-warp", action="store_true",
                        help="Keep captures as photographed: no rectification of "
                             "perspective or curl. Spreads are still split where "
                             "a page can be seen")
    parser.add_argument("--no-enhance", action="store_true",
                        help="Skip contrast/sharpness enhancement")
    parser.add_argument("--no-deskew", action="store_true",
                        help="Skip straightening each page against its own text lines")
    parser.add_argument("--dewarp-scans", action="store_true",
                        help="Run the page rectifier and deskew on pages loaded "
                             "with --pages-from. Off by default: pages this "
                             "script wrote have been rectified once and would "
                             "drift if flattened again, and changing a scan's "
                             "pixels invalidates its OCR cache. Opt in for a "
                             "third-party scan that visibly leans or curls")
    parser.add_argument("--no-dewarp", action="store_true",
                        help="Kept for old command lines; the rectifier now "
                             "handles perspective and curl together, so this "
                             "is the same as --no-warp")
    parser.add_argument("--no-split-spreads", action="store_true",
                        help="Keep two-page spreads as a single page. By default a landscape "
                             "crop is split at the gutter into two pages")
    parser.add_argument("--rotate", type=int, default=0, choices=[0, 90, 180, 270],
                        help="Rotate frames before processing, for footage that comes in "
                             "sideways (default: 0)")

    args = parser.parse_args(argv)
    if args.batch_size is not None:
        if args.batch_size < 1:
            parser.error("--batch-size must be at least 1")
        set_pages_in_flight(args.batch_size)
    if args.progress_json:
        progress.enable()
    if args.cache_only and not args.ocr_cache:
        parser.error("--cache-only needs --ocr-cache: the cache is what it "
                     "builds from")
    record = None
    try:
        switches.CACHE_ONLY = bool(args.cache_only)
        switches.READ_THIS_RUN["pages"] = 0
        if args.ocr_cache and args.epub:
            record = _BuildRecord(args, argv)
            switches.BUILD_NOTE = record.note()
            record.open()
        _convert(args, parser)
        if record:
            record.write()
    except SystemExit as exc:
        # sys.exit(message) is this pipeline's idiom for a refusal; hand the
        # message to whoever is driving before the status code ends us.
        if exc.code not in (None, 0):
            progress.emit(event="error", message=str(exc.code))
            if record:
                print(f"\n[stopped] {exc.code}")     # into the kept log
        raise
    except Exception as exc:
        progress.emit(event="error", message=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if record:
            record.close()
        switches.CACHE_ONLY = False
        switches.BUILD_NOTE = None


# What built a book, kept where the next person to wonder will look: beside
# the cache it was built from. Twelve of fifty installed books turned out,
# on 11 November, to be older than the code that was thought to have built
# them, and which passes each had been built with had to be read off the
# book: a page list meant --check-folios, an id beginning "bib-" meant
# --link-citations. The log of a build went wherever its caller sent it, or
# nowhere.
#
#   <cache stem>.build.log    what the build said, as it said it
#   <cache stem>.build.json   when, by what code, with what options, from
#                             what PDF, how many pages it had to read, and
#                             what came out
#
# Both are of the LAST build from that cache. The book itself carries the
# one line that matters most, in its package: the version, the commit and
# the options (<meta name="codicology:build">).

class _Tee:
    """What is written, written twice."""

    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, text):
        self.fh.write(text)
        return self.stream.write(text)

    def flush(self):
        self.fh.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _code_version() -> str:
    """The package's version and, where it runs from a checkout, the
    commit, with a + where the checkout holds changes not committed."""
    import subprocess
    from . import __version__
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        said = subprocess.run(
            ["git", "-C", here, "describe", "--always", "--dirty=+",
             "--abbrev=7"], capture_output=True, text=True, timeout=10)
        if said.returncode == 0 and said.stdout.strip():
            return f"{__version__} {said.stdout.strip()}"
    except (OSError, subprocess.SubprocessError):
        pass
    return __version__


# The options that are where a thing is and not what is done to it.
_PLACES = {"video", "pdf", "epub", "ocr_cache", "pages_from", "from_images",
           "review_sheet", "apply_decisions", "page_map", "patch", "title",
           "progress_json"}


class _BuildRecord:
    def __init__(self, args, argv):
        stem = args.ocr_cache
        for end in (".ocr.gz", ".gz", ".json"):
            if stem.endswith(end):
                stem = stem[:-len(end)]
                break
        self.args, self.argv = args, list(argv or sys.argv[1:])
        self.log_path, self.path = stem + ".build.log", stem + ".build.json"
        self.began = time.time()
        self.fh = self.stdout = None
        self.code = _code_version()
        # every option that is not as it would be if nothing were said
        self.options = sorted(
            "--" + k.replace("_", "-") + ("" if v is True else f"={v}")
            for k, v in vars(args).items()
            if k not in _PLACES and v not in (None, False, [], "")
            and v != self._unsaid(k))

    @staticmethod
    def _unsaid(key):
        # the few options whose silence is not None or False
        return {"ocr": "surya", "lang": "en", "stride": 1, "rotate": 0,
                "cover": "auto", "min_area": 0.15, "min_still_frames": 3}.get(
                    key, None)

    def note(self) -> str:
        return f"{self.code}; {' '.join(self.options)}".rstrip("; ")

    def open(self) -> None:
        try:
            self.fh = open(self.log_path, "w", encoding="utf-8")
        except OSError:
            self.fh = None
            return
        self.fh.write(f"codicology {self.code}\n"
                      f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                      f"convert {' '.join(self.argv)}\n\n")
        self.stdout, sys.stdout = sys.stdout, _Tee(sys.stdout, self.fh)

    def close(self) -> None:
        if self.stdout is not None:
            sys.stdout, self.stdout = self.stdout, None
        if self.fh is not None:
            self.fh.close()
            self.fh = None

    def write(self) -> None:
        a = self.args
        source = a.pages_from or a.from_images or a.video
        out = {"built": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "codicology": self.code, "options": self.options,
               "argv": self.argv,
               "source": os.path.abspath(source) if source else None,
               "cache": os.path.abspath(a.ocr_cache),
               "decisions": a.apply_decisions,
               "pages_sent_to_be_read": switches.READ_THIS_RUN["pages"],
               "cache_only": bool(a.cache_only),
               "seconds": round(time.time() - self.began),
               "log": os.path.abspath(self.log_path)}
        if source and os.path.isfile(source):
            out["source_bytes"] = os.path.getsize(source)
        if a.epub and os.path.exists(a.epub):
            from . import binding
            with open(a.epub, "rb") as fh:
                out["epub"] = {"path": os.path.abspath(a.epub),
                               "sha256": hashlib.sha256(fh.read()).hexdigest()}
            try:
                import zipfile
                with zipfile.ZipFile(a.epub) as z:
                    out["epub"]["bound"] = binding.is_bound(z.namelist())
                out["epub"]["pages"] = len(binding.page_bodies(a.epub))
            except Exception:
                pass
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1, ensure_ascii=False)
            print(f"\n  build record: {self.path}")
        except OSError as exc:
            print(f"\n  [~] no build record written: {exc}")


def _convert(args, parser) -> None:
    # The witness is checked here, before any work, because finding out after
    # forty minutes of reading that nothing was witnessed is finding out too
    # late. It is the tesseract BINARY, not the pytesseract package, which is
    # a different thing that only the tesseract OCR backend uses — so no pip
    # install satisfies this and the dependency list cannot enforce it.
    if not witness.witness_available():
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
        switches.WITNESS_WAIVED = True
        print("[!] Running WITHOUT a witness. Pages with no text layer will "
              "pass the fabrication check unexamined, and pages deleted as "
              "blank will be deleted on this pipeline's word alone.")

    if args.second_reader:
        from . import second as _second
        if not _second.available():
            sys.exit("--second-reader was asked for and there is none to "
                     f"ask: put the model's two .gguf files in "
                     f"{_second.DEFAULT_DIR} (or name them in "
                     f"${_second.MODEL_ENV} and ${_second.MMPROJ_ENV}), and "
                     f"have llama-server on the PATH.")
        _unasked = "--cache-only: no page is read on this run" \
            if args.cache_only else None
        _reader2 = _second.SecondReader()
        _reader2.down = _unasked
        _opinion = SecondOpinion(
            _reader2,
            _second.SecondCache(_second.SecondCache.beside(args.ocr_cache))
            if args.ocr_cache else None)
        switches.SECOND_OPINION = _opinion
        import atexit
        atexit.register(_opinion.close)
        from . import third as _third
        if _third.available():
            # Not asked for by name: it has no say but in what the second
            # reader raises, so it comes with it where it is installed.
            _reader3 = _third.ThirdReader()
            _reader3.down = _unasked
            _vote = ThirdOpinion(
                _reader3,
                _second.SecondCache(_second.SecondCache.beside(
                    args.ocr_cache, ".third.gz"), tag=_third.NAME)
                if args.ocr_cache else None)
            switches.THIRD_OPINION = _vote
            atexit.register(_vote.close)

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
        # The output PDF is assembled from the page images, and writing it
        # over the PDF they are being taken from replaces the book with a
        # copy of itself that may be worse: a born-digital source comes
        # back as pictures of its pages.
        for out_path in (output_pdf, args.epub):
            if out_path and args.pages_from and _same_file(out_path,
                                                           args.pages_from):
                sys.exit(f"{out_path} is the --pages-from source itself: "
                         f"writing the output there would overwrite the "
                         f"book being read. Give the output another name.")
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
        dewarp=not args.no_dewarp,
        dewarp_scans=args.dewarp_scans,
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
        serve_stale_cache=args.keep_stale_cache,
    )
    if args.epub and not args.page_files and os.path.exists(args.epub):
        bind_book(args.epub)
    progress.emit(event="result", epub=args.epub, pdf=output_pdf)


if __name__ == "__main__":
    main()
