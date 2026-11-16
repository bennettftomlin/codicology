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
import glob
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
import time
import unicodedata
from typing import NamedTuple

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

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


from .page_review import (  # noqa: F401  (moved; kept importable from here)
    SHINGLE_WORDS, DUPE_SIMILARITY, DUPE_WINDOW, MIN_SHINGLES,
    DUPE_BORDERLINE, REVIEW_THUMB_WIDTH, _shingles, PageVerdict,
    review_pages, _thumb_data_uri, PageFlag, FLAG_HAND, FLAG_SPARSE,
    FLAG_ASPECT, TURN_SHORTER, TURN_CONTAINED, TURN_WINDOW,
    find_page_turns, flag_pages, write_review_sheet,
)




from .cache import (  # noqa: F401  (moved; kept importable from here)
    CACHE_CHECKPOINT_PAGES, CACHE_VERSION, PAGE_KEYING, _PAGE_KEY,
    _pixel_key, _key_beside, _bytes_key, page_key, _as_formerly_written,
    former_page_key, OCRCache,
)




from .model import Block, Page, PageItem  # noqa: F401  (PageItem moved; kept importable from here)


from .backends import (  # noqa: F401  (moved; kept importable from here)
    OCRBackend, FIGURE_LABELS, BOX_MIN_AREA, BOX_MAX_AREA, BOX_MAX_INK,
    find_ruled_boxes, CAPTION_LABELS, FURNITURE_LABELS, _crop_block,
    _BLOCK_TAGS, _FOLD_INTO, _NO_SEAM_AFTER, _needs_seam_space,
    _fold_loose_inline, _STANDS_ALONE, _wrap_loose_inline, _to_xhtml,
    llamacpp_safe_schemas, set_pages_in_flight, SuryaBackend,
    OCR_BACKENDS, load_backend, reflow_paragraphs,
)


from .contents import labelled_contents_pages, labelled_heads
from .notes import footnote_share
from .furniture import furniture_texts
from .furniture import (  # noqa: F401  (moved; kept importable from here)
    RUNNING_HEAD, RUNNING_FOOT, FURNITURE_PAGES, strip_running_heads,
    head_shape, _LEAF_BLOCK, _FIRST_LEAF, _LAST_LEAF, _INLINE,
    _FIRST_FUSED, _LAST_FUSED, _OPENS_HEADING, _ENDS_HEADING, _HEAD_SITES,
    HEAD_CADENCE, _head_site, _head_key, _head_runs, _worded_head,
    strip_missed_running_heads, Chapter, head_title, _same_head,
    CHAPTER_RUN_MIN_MEAN, _NOT_A_CHAPTER_NAME, chapters_from_furniture,
)




from .markup import (  # noqa: F401  (moved; kept importable from here)
    RUNNING_HEAD_MAX_CHARS, BODY_MARKER, _roman_to_int, _SETTING,
)


from .figures import (  # noqa: F401  (moved; kept importable from here)
    PLACEHOLDER_IMG, _picture_signature, find_image_furniture,
    DOUBLED_BOX_OVERLAP, BOX_WORDS_READ, _box_holds_only_type, _BOX_WORD,
    _typeset_boxes_already_read, TABLE_MIN_FIGURES, TABLE_FIGURES_HELD,
    tables_short_of_figures, drop_doubled_boxes, _lies_inside,
    recover_placeholder_figures, recover_orphan_artwork, TABLE_RULE_REACH,
    recover_vector_artwork, _looks_like_same_picture, _render_region,
    _encodable, better_figure, FIGURE_PNG_OUT_OF_REACH, FIGURE_ENCODERS,
    encode_figure, _same_file, FIGURE_MIN_DETAIL, FIGURE_MAX_SKIN,
    FIGURE_HAND_INK, figure_is_a_hand, figure_has_content,
)




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


from .pdf_frame import (  # noqa: F401  (moved; kept importable from here)
    _page_frame, _pdf_box_to_normalized, _normalized_to_pdf,
)


from .pdf_frame import (  # noqa: F401  (moved; kept importable from here)
    native_orientation,
)


from .reading import (  # noqa: F401  (moved; kept importable from here)
    demote_mislabelled_heads, _CLOSED_EARLY, _OPENS_ON_A_BLOCK,
    refold_cached_fragments, route_form_blocks, MARKER_BLOCK_MAX_H,
    reattach_orphan_markers, ReadPages, _read_pages, turn_sideways_bodies,
    warm_backend, _read_pages_resiliently, SILENT_RUN,
    _collect_page_items, page_texts,
)




from .folios import (  # noqa: F401  (moved; kept importable from here)
    Folio, ROMAN_FOLIO_BASE, _ROMAN_TENS, _ROMAN_ONES, folio_label,
    parse_folio, folios_from_furniture, _TAKES_A_NUMBER,
    strip_fused_folios, strip_folio_superscripts, read_folios, FolioAudit,
    audit_folios, fill_folio_gaps,
)


from .contents import (  # noqa: F401  (moved; kept importable from here)
    _DESIGNATION, _NUMBER_WORD, _CHAPTER_MARK, _bare_title,
    _names_chapter, promote_chapter_headings, _APPARATUS_WORD,
    APPARATUS_ROW, NOT_A_DIVISION, PART_HEAD, TocEntry, _DESIGNATION_ONLY,
    _FOLIO_ONLY, _line_text, _fold_contents_lines, _NAMES_ITSELF_CONTENTS,
    _COLUMN_HEADING, _blank_lead_runs, _leads_indented_rows,
    _peer_banners, parse_printed_toc, _close_group_on_outdent,
    _lift_trailing_apparatus, _infer_row_depths, _depths_from_typography,
    _contents_folio, _folio_series, folio_resolver, _title_on_page,
    _title_names_this_page, _place_toc_entries, BOLD_HEAD_BLOCK,
    CHAPTER_HEAD, _WORD_VALUE, _chapter_head_number,
    normalize_chapter_heads, furniture_absent_from_contents,
    merge_missing_furniture, contents_confidence, nav_from_placed,
    promote_missing_chapter_heads, chapter_head_contents,
)




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


def _ocr_one(image: Image.Image, backend: "OCRBackend") -> str:
    try:
        return _strip_tags(backend.run([image])[0]).strip()
    except Exception:
        return ""


from .pdf_pages import (  # noqa: F401  (moved; kept importable from here)
    _union_fraction, PAGE_PICTURE_MIN, _is_picture_of_itself,
    _scanned_pages, _scans_are_spreads, PDF_POOL_MIN_PAGES,
    PDF_POOL_WORKERS, PDF_WORKERS_ENV, _PDF_WORKER, TURN_MIN_CONFIDENCE,
    TURN_SPREAD_ASPECT, TURN_CHECK_LONGEST, TURN_FURNITURE_PAD, _TURNS,
    _which_way_up, _halves, _sideways, _body_turned, TURNS_SUFFIX,
    load_turns, save_turns, _note_sideways, _load_pdf_page,
    _load_pdf_share, _load_pdf_pages_in_workers, _pdf_pool_workers,
    load_pages_from_pdf,
)




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


from .equations import (  # noqa: F401  (moved; kept importable from here)
    VULGAR, _RAISED, _LOWERED, _FRACTION, _MATH_FORMS, _set_fraction,
    _math_as_text, _SUPERSCRIPT, _SCRIPT, _scripts_as_html,
    normalize_math, _UNIT_POWER, set_unit_powers, MATHML_NS, PANDOC_ENV,
    _SEPARATOR, _pandoc, _mathml_from_pandoc, write_display_math,
)




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


from .notes import (  # noqa: F401  (moved; kept importable from here)
    NOTE_ENTRY, _NOTES_WORDS, NOTES_HEAD, NOTES_SECTION_HEAD, GROUP_NAMES,
    GROUP_HEAD, GROUP_HEAD_TITLED, _SECTION_NUMERAL, GROUP_HEAD_SECTION,
    _notes_follow, notes_head_on, count_notes_heads, notes_section_start,
    _bare_numbered, parse_notes_section, _lists_inside_notes,
    _ungrouped_notes, _entry_anchor, link_chapter_notes,
    find_body_marker_groups, NOTE_ENTRY_PLAIN, NOTE_ENTRY_LI,
    NOTE_ENTRY_BARE, NOTE_ENTRY_BR, NOTE_ENTRY_BR_SUP, NOTE_ENTRY_BR_BARE,
    _br_run_notes, GROUP_HEAD_NUMBERED, GROUP_HEAD_NUMBERED_B,
    GROUP_HEAD_ROMAN, GROUP_HEAD_ROMAN_P, normalize_note_heads,
    FOOTNOTE_SYMBOLS, FOOT_MARK, FOOT_SHAPE, FOOT_ENTRY,
    _looks_like_foot_block, _foot_block_without_a_rule, _ENDS_A_SENTENCE,
    _carried_over_tail, FOOT_LABELLED, _opens_with_a_foot, link_footnotes,
    _best_body_groups, _fit_groups, _leading_repeats,
    _paired_groups_agree, _EDITION_MARK, _title_under,
    _titles_with_editions, _an_editions_number, link_notes,
)




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


from .pdf_frame import (  # noqa: F401  (moved; kept importable from here)
    _box_iou,
)


from .cache_tools import (  # noqa: F401  (moved; kept importable from here)
    _match_layout_labels, relabel_cache, rekey_cache,
)




from .typography import (  # noqa: F401  (moved; kept importable from here)
    PAGE_HYPHEN, PAGE_CONT, join_page_break_hyphens,
)


from .decisions import (  # noqa: F401  (moved; kept importable from here)
    _decision_rounds, apply_reviewer_decisions,
)




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
    (bodies, page_figures, figure_data,
     n_figures, n_blank_figures, n_label_rules, n_reattached, n_phantoms,
     n_forms, n_hand_figures,
     n_doubled, pages) = rp
    # Running heads and feet are read from the pages, where they were set
    # aside as blocks; a page cached before heads were kept says so.
    page_furniture = furniture_texts(pages)
    furniture_known = [p.furniture_known for p in pages]
    # What the layout labelled a heading or a printed contents, as the pages
    # hold it when read: the contents falls back on these.
    label_heads = labelled_heads(pages)
    toc_label_pages = labelled_contents_pages(pages)
    # How much of each page's foot the layout itself called footnotes.
    foot_labelled = footnote_share(pages)
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


from .photos import (  # noqa: F401  (moved; kept importable from here)
    IMAGE_SUFFIXES, _HEIF_READY, read_image, collect_image_paths,
    group_burst, _prepare_page_image, load_patches, _patch_frames,
    apply_patches, pages_from_images, _report_kept_photos,
    dewarp_scan_pages, POSITIONAL, page_id, resolve_selector,
    load_page_map,
)


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


from .reconcile import (  # noqa: F401  (moved; kept importable from here)
    _CONTROL, _NO_GLYPH, _no_glyph, _MARKUP, _same_word, _EDGE_MARKS,
    _same_but_its_marks, PUT_BACK_ANCHOR, PUT_BACK_LONGEST, _VULGAR_SIGNS,
    _PRINTED_RUN, _printed_runs, _layer_geometry, _follows,
    _stands_in_the_line, reconcile_native_text,
)


from .searchable_pdf import (  # noqa: F401  (moved; kept importable from here)
    _LAYER_MIN_FONT, _force_text_into, _tesseract_word_boxes,
    build_searchable_pdf,
)


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
