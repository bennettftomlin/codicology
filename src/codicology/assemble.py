"""The assembly half of a build: the pages the reading half hands over turned
into a book, pass by pass (links mended, furniture and signature marks set
aside, the text layer reconciled, typography, chapters and lists, page
numbers, duplicates and blank leaves, decisions, notes, citations, the
index, the contents, equations), and written as an EPUB 3 package with its
page list and nav.
"""
from __future__ import annotations

import cv2
import html
import io
import os
import re
import sys
from PIL import Image
from . import progress, switches, witness
from .backends import OCRBackend
from .cache import CACHE_VERSION, OCRCache
from .citations import link_citations
from .contents import _place_toc_entries, chapter_head_contents, contents_confidence, furniture_absent_from_contents, labelled_contents_pages, labelled_heads, merge_missing_furniture, nav_from_placed, normalize_chapter_heads, promote_chapter_headings, promote_missing_chapter_heads
from .decisions import _decision_rounds, apply_reviewer_decisions
from .equations import MATHML_NS, _pandoc, normalize_math, set_unit_powers, write_display_math
from .figures import _encodable, encode_figure, find_image_furniture
from .folios import Folio, audit_folios, fill_folio_gaps, folio_label, folios_from_furniture, read_folios, strip_folio_superscripts, strip_fused_folios
from .furniture import Chapter, chapters_from_furniture, furniture_texts, head_shape, restore_lone_furniture, strip_missed_running_heads
from .guards import _page_core_ink, _phantom_blank_pages, _reads_as_empty
from .index_links import link_index
from .markup import _roman_to_int, _strip_tags
from .model import PageItem
from .notes import count_notes_heads, footnote_share, link_chapter_notes, link_footnotes, link_notes, normalize_note_heads
from .page_review import DUPE_SIMILARITY, flag_pages, review_pages, write_review_sheet
from .photos import read_image, resolve_selector
from .reading import _read_pages, _read_pages_resiliently, turn_sideways_bodies
from .reconcile import reconcile_native_text
from .typography import close_short_links, join_page_break_hyphens, mend_swallowed_links, normalize_typography, repair_c1_controls, unusual_characters


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
    # A line the layout called furniture that the book never repeats is a
    # caption, a note or an imprint at the page's edge: back into the page.
    for _i, _t in restore_lone_furniture(pages, bodies):
        print(f"    [~] page {_i + 1}: {_t[:70]!r} was labelled page furniture "
              f"but is printed on no other page — put back")
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
    n_short = close_short_links(bodies)
    if n_short:
        print(f"    closed {n_short} link(s) over the rest of the address "
              f"their text was cut from")
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
        #
        # And a PDF's copy is the whole page, so the two must be alike each
        # way. The score asks how much of the SMALLER page the other holds,
        # and a title page is held whole by the copyright page after it,
        # whose cataloguing entry quotes it word for word: The Modern
        # Corporation lost its title page at 1.00. Each page must be nearly
        # all of the other.
        from_pdf = any(os.path.exists(p + ".source.json") for p in page_paths)
        floor = 0.90 if from_pdf else DUPE_SIMILARITY
        dropped = {v.index for v in verdicts
                   if v.status == "duplicate" and v.index not in native
                   and v.score >= floor
                   and (not from_pdf or v.alike >= floor)}
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
                  + (f" ({fstats['table']} in tables)"
                     if fstats["table"] else "")
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
