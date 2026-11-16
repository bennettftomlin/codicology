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


from .assemble import (  # noqa: F401  (moved; kept importable from here)
    _SIGNATURE, SIGNATURES_WANTED, _signature_rank, strip_signature_marks,
    ENTRY_HEAD, ENTRY_HEAD_BARE, _HEAD_FORMS, normalize_entry_heads,
    _load_cover, find_numbered_entries, SELF_MARKED_NUM, SELF_MARKED_BUL,
    LIST_OPEN, normalize_list_markers, _restrict_page_list_to_pagebreaks,
    build_epub, UNREAD_PAGE_INK, _PAGE_MARKS, _page_as_itself,
)




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


from .equations import (  # noqa: F401  (moved; kept importable from here)
    VULGAR, _RAISED, _LOWERED, _FRACTION, _MATH_FORMS, _set_fraction,
    _math_as_text, _SUPERSCRIPT, _SCRIPT, _scripts_as_html,
    normalize_math, _UNIT_POWER, set_unit_powers, MATHML_NS, PANDOC_ENV,
    _SEPARATOR, _pandoc, _mathml_from_pandoc, write_display_math,
)


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
