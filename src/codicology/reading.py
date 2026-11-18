"""The reading half of a build: each page read (from the cache where it was
read before, by the recogniser otherwise, with an empty reading of an inked
page read again and never cached), guarded, its figures taken, and handed
to the assembly half as a model.Page of blocks with the string rendered
from them. A two-page sheet's sideways page is turned here once it has
been read as it lies.
"""
from __future__ import annotations

import json
import os
import re
import sys
from PIL import Image
from typing import TYPE_CHECKING, NamedTuple
from . import progress, switches, witness
from .backends import _to_xhtml
from .cache import OCRCache, _pixel_key
from .figures import FIGURE_ENCODERS, _encodable, better_figure, drop_doubled_boxes, encode_figure, figure_has_content, figure_is_a_hand, recover_orphan_artwork, recover_placeholder_figures, recover_vector_artwork, tables_short_of_figures
from .furniture import strip_running_heads
from .guards import LAYER_MIN_WORDS, PHANTOM_CORE_INK, _contradicts_layer, _layer_can_testify, _looks_fabricated, _native_layer_body, _page_core_ink, _page_picture, _reading_agrees, drop_counting_tables, unvouched_reading
from .markup import _strip_tags
from .model import Block, Page, PageItem
from .opinions import mend_equations, mend_tables, typeset_body_size_text_in, vote_scanned_tables
from .pdf_pages import _body_turned

if TYPE_CHECKING:
    from .backends import OCRBackend


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
                      f"line of text — set aside, unless the book never "
                      f"repeats it: {t[:70]!r}")
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


# A superscript marker the layout orphaned into its own block stands exactly
# one text line tall. Measured on eagle's 274 specimens: markers 0.019-0.020
# of page height, the shortest multi-line text block 0.038 — this sits in
# the gap. A display chapter numeral is several times taller.
MARKER_BLOCK_MAX_H = 0.03


def reattach_orphan_markers(items: list, furn_numbers: set,
                            seq_state: "dict | None" = None,
                            body_text_in=None) -> tuple:
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

    On a typeset page the PDF's own text can say more. A misfit whose box
    holds exactly that number in the PDF, set at the page's body size, is
    neither a phantom nor a raised call: it is a figure the page prints on
    its own line, such as a page reference in a table column (Where There Is
    No Doctor p. 48 lost six of them). It is left where it stands.
    `body_text_in` gives that text, from typeset_body_size_text_in.

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
        if body_text_in is not None \
                and re.sub(r"\s+", "", body_text_in(it.box)) == t:
            continue    # printed on its own line at body size: keep it
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


class ReadPages(NamedTuple):
    """Everything the reading half of a build produces: the text of every
    page, the figures already added to the book, and the counters the
    report speaks from. build_epub's assembly half consumes exactly this,
    and nothing else crosses the boundary (S4)."""
    bodies: list
    page_figures: list
    figure_data: dict
    n_figures: int
    n_blank_figures: int
    n_label_rules: int
    n_reattached: int
    n_phantoms: int
    n_forms: int
    n_hand_figures: int
    n_doubled: int = 0
    # The same pages before they were joined: their blocks and furniture
    # (model.Page). bodies[i] is pages[i].render().
    pages: "list | None" = None


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
    n_figures = 0
    n_blank_figures = 0
    n_hand_figures = 0
    n_label_rules = 0
    n_reattached = 0
    n_phantoms = 0
    n_forms = 0
    n_doubled = 0
    n_unvouched = 0
    marker_seq_state: dict = {"last": None}
    pages: list[Page] = []      # each page before it is joined (model.Page)
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
        for page in pages:
            for b in page.blocks:
                b.html = named.sub(lambda m: names[m.group(0)], b.html)
        bodies[:] = [page.render() for page in pages]
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
            items, _voted, _ties = vote_scanned_tables(
                items, path, switches.SECOND_OPINION, switches.THIRD_OPINION)
            if switches.SECOND_OPINION is not None and (_voted or _ties):
                _pg = os.path.splitext(os.path.basename(path))[0]
                switches.SECOND_OPINION.figures += len(_voted)
                switches.SECOND_OPINION.changed += [
                    (_pg, "table figure", _a, _b,
                     "the second and third readers read it so")
                    for _a, _b in _voted]
                switches.SECOND_OPINION.ties += [(_pg, *_t) for _t in _ties]
            for _a, _b in _voted:
                print(f"    [+] {os.path.basename(path)}: a scanned table's "
                      f"figure is set by a vote of three readers: {_a} → {_b}")
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
            _unsure = unvouched_reading(items, path, questioned=bool(_counts))
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
                    _why = (f"{_n} words read beside the table it invented, "
                            f"and no witness reads more than "
                            f"{_unsure[1]:.0%} of them" if _counts and _n else
                            "nothing left beside the table it invented"
                            if _counts else
                            f"{_n} words the recogniser scored "
                            f"{_unsure[0]:.2f}, and no witness reads more "
                            f"than {_unsure[1]:.0%} of them on the page")
                    print(f"    [!] {os.path.basename(path)}: {_why} — shown "
                          f"as a picture of the page rather than printed as "
                          f"invention"
                          + (f", with the {_kept} words of {len(_unsure[2])} "
                             f"block(s) a witness does read printed beneath it"
                             if _unsure[2] else ""))
            # Furniture is set aside, never merely deleted: its text is the
            # folio audit's evidence, already recognised at full resolution.
            page = Page(index=len(bodies), path=path,
                        furniture=[Block.of(it) for it in items if it.is_furniture],
                        furniture_known=(hit is None) or (
                            cache is not None and cache.knows_furniture(path)))
            furn_numbers = {m for it in items if it.is_furniture and it.html
                            for m in re.findall(r"\d{1,3}",
                                                _strip_tags(it.html))}
            items = [it for it in items if not it.is_furniture]
            _lone = any(it.html and it.figure is None and re.fullmatch(
                r"\d{1,3}", _strip_tags(it.html).strip()) for it in items)
            _ra, _sp = reattach_orphan_markers(
                items, furn_numbers, marker_seq_state,
                typeset_body_size_text_in(path) if _lone else None)
            n_reattached += _ra
            n_phantoms += _sp
            if strip_furniture:
                items = strip_running_heads(items)
            items, _dbl, _held = drop_doubled_boxes(items, path)
            n_doubled += _dbl
            items = recover_placeholder_figures(items, path)
            items = recover_orphan_artwork(items, path, _held)
            items = recover_vector_artwork(items, path, _held)
            current_figures: list[str] = []

            def figure_block(fig: PageItem, cap: "PageItem | None",
                             caption_first: bool) -> None:
                """The figure, with its caption, as one block; its caption
                alone where the figure is not worth keeping; nothing where
                neither is."""
                html_ = emit_figure(fig, cap, caption_first=caption_first)
                if html_.startswith("<figure>"):
                    page.blocks.append(Block.of(
                        fig, html=html_, caption=cap.html if cap is not None else None))
                elif html_:
                    page.blocks.append(Block.of(cap, html=html_))

            i = 0
            while i < len(items):
                item = items[i]
                nxt = items[i + 1] if i + 1 < len(items) else None

                # Books caption above as well as below, so bind a caption on
                # either side of the image rather than assuming one convention.
                if item.is_caption and nxt is not None and nxt.figure is not None:
                    figure_block(nxt, item, caption_first=True)
                    i += 2
                    continue

                if item.figure is not None:
                    if nxt is not None and nxt.is_caption:
                        figure_block(item, nxt, caption_first=False)
                        i += 2
                    else:
                        figure_block(item, None, caption_first=False)
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
                        and "<hr/>" not in page.render() \
                        and not any(it.label == "Footnote" for it in items[:i]):
                    page.blocks.append(Block(kind="rule", html="<hr/>"))
                    n_label_rules += 1
                page.blocks.append(Block.of(item))
                i += 1
            pages.append(page)
            bodies.append(page.render())
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
    return ReadPages(bodies, page_figures, figure_data,
                     n_figures, n_blank_figures,
                     n_label_rules, n_reattached, n_phantoms, n_forms,
                     n_hand_figures, n_doubled,
                     pages)


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
        # What of the reading to keep when the page itself cannot be read
        # and its own text stands in: kept with that outcome, the page is
        # served the same way by every later build (see the cache write).
        keep: "list | None" = None
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
                        keep = []
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
                keep = []
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
        if keep is not None and cache is not None:
            # The reader read this page twice and both times told a story
            # the page does not hold: it looped on itself, or read nothing
            # like the page's own text. A recogniser gives a page the same
            # reading every time, so asking again at every build only re-ran
            # this, and the book could never be built from its cache: four
            # on the shelf, for one such page each. A recogniser that
            # answers with text is reading, so the outcome is kept — the
            # page's own text standing in, or nothing where it has none —
            # and served by every later build without asking.
            #
            # Not so a page that came back EMPTY. That is also what a
            # recogniser failing on and off looks like, and one that passed
            # its test page between two failures once left a book 64 pages
            # of drawings without their text, replayed by every rebuild.
            # Those stay unkept, and are read again.
            cache.put(path, keep, outcome="layer")
            print(f"    [~] {os.path.basename(path)}: remembered as a page the "
                  f"reader cannot give, so later builds need not ask again")
        elif provisional:
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
