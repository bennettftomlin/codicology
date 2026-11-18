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
    _FIGURE, _figures, _typeset_text_in, typeset_body_size_text_in,
    BODY_SIZE_SHARE, _share_held, SecondOpinion,
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


from .backends import (  # noqa: F401  (moved; kept importable from here)
    _ocr_one,
)




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


from .convert import (  # noqa: F401  (moved; kept importable from here)
    _finish, bind_book, process_video, main, _Tee, _code_version, _PLACES,
    _BuildRecord, _convert,
)




if __name__ == "__main__":
    main()
