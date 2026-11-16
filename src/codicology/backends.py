"""The OCR backend: Surya, run through the local inference server, and what
turns its answer into a page's blocks: the layout's labels, ruled boxes
taken for drawings, each block's markup tidied into XHTML, a paragraph
reflowed where the page broke it.
"""
from __future__ import annotations

import cv2
import html
import numpy as np
import os
import statistics
import sys
from PIL import Image
from . import switches
from .markup import _strip_tags
from .model import PageItem


# ══ OCR backends ══════════════════════════════════════════════════════════════

class OCRBackend:
    """Takes a batch of PIL page images, returns one result per page."""

    name = "base"
    batch_size = 1

    def __init__(self, langs: list[str]) -> None:
        self.langs = langs

    def run(self, images: list[Image.Image]) -> list[str]:
        """Plain text, one string per page."""
        raise NotImplementedError

    def run_html(self, images: list[Image.Image]) -> list[str]:
        """XHTML body fragment per page. Backends that recover layout override this."""
        return [
            "".join(f"<p>{html.escape(p)}</p>" for p in reflow_paragraphs(text))
            for text in self.run(images)
        ]

    def run_items(self, images: list[Image.Image]) -> list[list[PageItem]]:
        """
        Page contents in reading order.

        Backends that locate figures override this so illustrations survive into
        the EPUB; the rest degrade to a single markup item per page.
        """
        return [[PageItem(html=body)] for body in self.run_html(images)]


# The blocks whose pixels are their content. These are the labels Surya does
# not read (surya.recognition.SKIP_CANON_LABELS, less BlankPage): each comes
# back skipped, with no HTML, so a label missing here is a block that leaves
# no figure, no text and nothing in the cache to show it was there. Diagram
# was missing. tests/test_surya_labels.py holds this set, and every other
# label named in this package, against the table the installed Surya ships.
FIGURE_LABELS = frozenset({"Picture", "Figure", "Diagram"})


# A ruled box smaller than this fraction of the page is an initial or an
# ornament; larger than the ceiling it is the page's own border.
BOX_MIN_AREA, BOX_MAX_AREA = 0.015, 0.35


# How empty a ruled box's interior must be to read as a drawing. A chemical
# structure is sparse strokes on white; a boxed paragraph or a filled table
# carries far more ink than this.
BOX_MAX_INK = 0.20


def find_ruled_boxes(image: Image.Image,
                     claimed: "list[tuple[float, float, float, float]] | None" = None
                     ) -> list[tuple[int, int, int, int]]:
    """
    Rectangles drawn on the page that the layout model never made blocks of.

    TiHKAL sets every chemical structure inside a ruled box with body text
    wrapped around it, and the layout pass swallows the whole region into the
    text block: no Picture, no garbage, nothing — fifty-five structure drawings
    silently absent from the book. The box border itself is the reliable
    signal: a closed dark rectangle of plausible size whose interior is mostly
    white. That is a contour question, not a layout question.

    `claimed` lists regions (x0, y0, x1, y1) already emitted as figures, so a
    framed photograph is not added twice.
    """
    g = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = g.shape[:2]
    ink = (g < 200).astype(np.uint8)
    contours, _ = cv2.findContours(ink, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    page_area = float(w * h)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh / page_area
        if not BOX_MIN_AREA <= area <= BOX_MAX_AREA:
            continue
        if not 0.25 <= bw / max(1, bh) <= 6.0:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) != 4:
            continue
        # the border must actually be drawn on all four sides
        t = 3
        sides = [ink[y:y + t, x:x + bw], ink[y + bh - t:y + bh, x:x + bw],
                 ink[y:y + bh, x:x + t], ink[y:y + bh, x + bw - t:x + bw]]
        # 0.3, not higher: the rule is one drawn line and anti-aliasing puts
        # some of it just outside the contour's bounding strip — a real box on
        # a real page measured 0.496 on its weakest side and was rejected at
        # 0.5. Prose never forms a four-cornered contour of this size at all,
        # so the corner and hollowness tests carry the discrimination.
        if min(float(sd.mean()) for sd in sides if sd.size) < 0.3:
            continue
        interior = ink[y + t * 2:y + bh - t * 2, x + t * 2:x + bw - t * 2]
        if interior.size == 0 or float(interior.mean()) > BOX_MAX_INK:
            continue
        if any(not (x1 <= x or x0 >= x + bw or y1 <= y or y0 >= y + bh)
               for x0, y0, x1, y1 in (claimed or [])):
            continue
        boxes.append((x, y, x + bw, y + bh))
    # nested rectangles (a rule drawn double) collapse to the outermost
    boxes.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept: list[tuple[int, int, int, int]] = []
    for b in boxes:
        if not any(b[0] >= k[0] - 6 and b[1] >= k[1] - 6
                   and b[2] <= k[2] + 6 and b[3] <= k[3] + 6 for k in kept):
            kept.append(b)
    return sorted(kept, key=lambda b: b[1])


CAPTION_LABELS = frozenset({"Caption"})


FURNITURE_LABELS = frozenset({"PageHeader", "PageFooter"})


def _crop_block(image: Image.Image, block, page) -> Image.Image | None:
    """
    Cut a detected block's polygon out of the page scan.

    Taken exactly, with no padding: blocks sit only a few pixels apart, so even a
    small margin pulls the running header or the caption into the figure.
    """
    poly = getattr(block, "polygon", None)
    if not poly:
        return None
    xs = [float(p[0]) for p in poly]
    ys = [float(p[1]) for p in poly]

    # Block coordinates are in the frame the model saw, which need not match the
    # scan we hold, so rescale via the reported page box.
    box = getattr(page, "image_bbox", None) or [0, 0, image.width, image.height]
    sx = image.width / max(1e-6, float(box[2]) - float(box[0]))
    sy = image.height / max(1e-6, float(box[3]) - float(box[1]))

    x0 = int(max(0, min(xs) * sx))
    y0 = int(max(0, min(ys) * sy))
    x1 = int(min(image.width, max(xs) * sx))
    y1 = int(min(image.height, max(ys) * sy))
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    # crop() is lazy; copy so the result survives closing the source image.
    return image.crop((x0, y0, x1, y1)).copy()


# Elements that end a block. Anything else between two of them is inline
# content that belongs to the block before it, however the model punctuated it.
_BLOCK_TAGS = frozenset({"p", "div", "ul", "ol", "li", "table", "thead", "tbody",
                         "tr", "td", "th", "figure", "figcaption", "blockquote",
                         "pre", "hr", "h1", "h2", "h3", "h4", "h5", "h6"})


# and the ones worth folding INTO. A rule or a table has no running text.
_FOLD_INTO = frozenset({"p", "li", "blockquote", "figcaption",
                        "h1", "h2", "h3", "h4", "h5", "h6"})


# Punctuation that already binds to what follows it: no book writes
# "strategic ( continued)", and the fold should not be the first to.
_NO_SEAM_AFTER = "([{“‘'\"/-–—"


def _needs_seam_space(text: str) -> bool:
    return bool(text) and not text[-1].isspace() and text[-1] not in _NO_SEAM_AFTER


def _fold_loose_inline(div) -> None:
    """Put inline content the model left outside its block back inside it.

    One book's notes section arrives from the layout pass as

        <p>53.</p><i>Ibid.</i>, p. 51.<p>54.</p><i>Ibid.</i>, p. 53.

    — each entry's NUMBER closed as a paragraph of its own and the entry's
    text left loose beside it. The text is not lost, but it belongs to no
    block, which is not valid XHTML and is invisible to every pattern that
    reads a note entry: they all anchor the number to a block's opening, so
    nine entries were unreadable and the markers pointing at them stayed
    plain.

    The repair is structural and says nothing about notes: inline content
    between two blocks belongs to the block before it. A rule, a table or
    another paragraph ends the run, so nothing is ever pulled across a
    boundary the model actually drew.
    """
    # A snapshot, because folding MOVES elements out of div: iterating the
    # live tree skips the element after every fold, and a fragment holding
    # two entries had only its first one repaired.
    for el in list(div):
        if el.tag not in _FOLD_INTO or el.getparent() is not div:
            continue
        while True:
            nxt = el.getnext()
            tail = el.tail or ""
            if not tail.strip() and (nxt is None or nxt.tag in _BLOCK_TAGS):
                break
            # A closed block leaves a word boundary behind, not nothing —
            # the same reasoning _strip_tags is built on. Without it the
            # number runs into its own note ("53.Ibid.") and the entry
            # patterns, which want whitespace after the period, still miss it.
            if len(el):
                seam = (el[-1].tail or "")
                if seam and _needs_seam_space(seam):
                    el[-1].tail = seam + " "
                elif not seam and not (el[-1].text or "").endswith(" "):
                    el[-1].tail = " "
            elif _needs_seam_space(el.text or ""):
                el.text += " "
            if tail:
                if len(el):
                    el[-1].tail = (el[-1].tail or "") + tail
                else:
                    el.text = (el.text or "") + tail
                el.tail = None
            if nxt is not None and nxt.tag not in _BLOCK_TAGS:
                el.append(nxt)          # its own tail travels with it


# What may stand in a page's body without a paragraph round it, beside the
# blocks themselves: an equation set on its own line, and a picture.
_STANDS_ALONE = _BLOCK_TAGS | {"math", "img", "svg", "section", "aside",
                               "dl", "caption"}


def _wrap_loose_inline(div) -> None:
    """Give running text that stands outside every block a paragraph.

    A block the layout calls a heading often comes back from the recogniser
    as nothing but `<b>Keep Your Hands Clean</b>`: 871 of the 921 fragments
    in the shelf's caches that hold no block at all. Each went into the
    page as it came, and nothing in a page separates one such fragment
    from the next, so the index of Where There Is No Doctor read
    "Consumption (See Tuberculosis)Contact dermatitis, 204, 421Contraception"
    and a dosage ran straight on from the warning before it. 259 pages in
    40 books carried them.

    What _fold_loose_inline could not give to a block before it is a block
    of its own. Nothing is said about what kind: it is a paragraph, bold or
    italic as it was read.
    """
    from lxml import etree

    nodes = []
    if div.text:
        nodes.append(div.text)
        div.text = None
    for child in list(div):
        tail, child.tail = child.tail, None
        div.remove(child)
        nodes.append(child)
        if tail:
            nodes.append(tail)
    para = None
    for node in nodes:
        if isinstance(node, str):
            if para is None:
                if not node.strip():
                    continue            # white space between two blocks
                para = etree.SubElement(div, "p")
                para.text = node.lstrip()
            elif len(para):
                para[-1].tail = (para[-1].tail or "") + node
            else:
                para.text = (para.text or "") + node
        elif node.tag in _STANDS_ALONE and not (
                node.tag == "math" and node.get("display") != "block"):
            para = None
            div.append(node)
        else:
            if para is None:
                para = etree.SubElement(div, "p")
            para.append(node)
    for para in list(div):
        # a line break or an empty tag with no words round it is not a
        # paragraph, and a placeholder picture is looked for bare
        if para.tag == "p" and para.get("id") is None \
                and not "".join(para.itertext()).strip():
            at = div.index(para)
            for inner in reversed(list(para)):
                if inner.tag != "br":
                    div.insert(at, inner)
            if not len(para) or all(k.tag == "br" for k in para):
                div.remove(para)


def _to_xhtml(fragment: str) -> str:
    """Reparse model-generated HTML into well-formed XHTML for the EPUB."""
    try:
        from lxml import etree
        from lxml import html as lhtml

        div = lhtml.fragment_fromstring(fragment, create_parent="div")
    except Exception:
        return f"<p>{html.escape(_strip_tags(fragment))}</p>"

    _fold_loose_inline(div)
    _wrap_loose_inline(div)
    return "".join(etree.tostring(child, method="xml", encoding="unicode")
                   for child in div)


# Surya's llama.cpp backend sizes the server's KV cache as parallel-slots x
# 12k tokens, and matches its client workers to the same number — but its
# default of 8 slots is double the 4-page batches this pipeline ever sends.
# The surplus is pure memory, and buys nothing. Measured on an M1 Pro,
# each configuration run twice in mirrored order so drift over a session
# could not pose as a result — pages per minute against slots:
#
#     2 slots  2.44      4 slots  2.89      (8 slots ties 4, at +1GB)
#     3 slots  2.66
#
# Four is the knee, and it is a real one: nine percent per slot on the way
# up, nothing at all past it. Fewer is measurably slower — dropping to 2
# to save a gigabyte would cost a fifth of the throughput of every book.
#
# The measurement was run because the reason first written here was wrong.
# It blamed this on memory pressure from grammar-initialisation failures
# when two OCR stacks overlapped. Those failures have nothing to do with
# slots: they are surya's layout schema being refused by llama.cpp, which
# llamacpp_safe_schemas below repairs. Sized to the work, for the reason
# the work actually supports; set before surya reads its settings.
os.environ.setdefault("SURYA_INFERENCE_PARALLEL", "4")


def llamacpp_safe_schemas() -> int:
    r"""Rewrite `\d` to `[0-9]` in surya's guided-decoding schemas, in place.

    Surya 0.22 constrains its layout (and table) output with a JSON schema
    whose box pattern is `^\d{1,4} \d{1,4} \d{1,4} \d{1,4}$`. llama.cpp turns
    a schema into a grammar, and its pattern parser has no `\d` (build 10450,
    and master as of 2026-09-29), so every guided layout request came back
    HTTP 400 "failed to parse grammar" — four to a page, one call and three
    retries — with no boxes at all.

    That was once written off here as a harmless startup probe. It is not:
    layout is surya's fallback for a page whose full-page read fails —
    loops, errors, times out — and the fallback reads block by block inside
    the boxes layout returns. With none returned it read nothing, so every
    page that needed the fallback came back empty, on every build since
    surya 0.22.1 went in on 2026-07-30. The empty-read retry above caught
    each one the logs recorded. Three hard scanned pages from the shelf,
    forced through the fallback, read 0 words with the stock pattern and
    75, 470 and 107 with this.

    The dicts are shared by reference — surya.layout and surya.table_rec
    import the same objects from surya.inference.prompts — so an in-place
    edit reaches every caller, and it must happen before a predictor is
    built. Idempotent. Returns the number of patterns rewritten.
    """
    try:
        from surya.inference import prompts
    except ImportError:
        return 0        # an older surya: no guided schemas to repair
    n = 0
    for name in ("LAYOUT_JSON_SCHEMA", "TABLE_REC_JSON_SCHEMA"):
        schema = getattr(prompts, name, None) or {}
        for prop in schema.get("items", {}).get("properties", {}).values():
            pat = prop.get("pattern")
            if pat and "\\d" in pat:
                prop["pattern"] = pat.replace("\\d", "[0-9]")
                n += 1
    return n


def set_pages_in_flight(n: int) -> None:
    """Set how many pages are read at once — both halves of it.

    The batch the pipeline sends and the slots the server holds are one
    number wearing two hats: a slot with no page in it is only memory, and
    a page with no slot waits. Changing one alone is how a machine ends up
    paying for capacity it cannot use, so this changes both, and does it
    before surya reads its settings — the environment is consulted once, at
    import, and the import is lazy for exactly this reason."""
    os.environ["SURYA_INFERENCE_PARALLEL"] = str(n)
    SuryaBackend.batch_size = n


class SuryaBackend(OCRBackend):
    """
    Surya through surya.recognition.RecognitionPredictor, called as
    rec(images): detection and layout run inside it, and the page comes back
    as layout blocks carrying HTML, ordered by `reading_order`.

    That is the API of the release this package pins (pyproject.toml), and
    the only one read here. The branches for surya.ocr.run_ocr, which
    returned flat text lines, could not run at the pin and are gone; when
    the pin moves and the shape with it, this class is where to look.

    0.2x delegates inference to llama.cpp and needs a `llama-server` binary
    on PATH (`brew install llama.cpp`), or LLAMA_CPP_BINARY set.
    """

    name = "surya"
    batch_size = 4
    # Recognition at this pin is called with the images and nothing else, so
    # the language list changes no word of the reading. OCRCache leaves it
    # out of the tag for such an engine.
    language_blind = True

    _NOT_THERE = (
        "Could not find Surya's RecognitionPredictor. Install the "
        "release this package pins: pip install 'codicology[surya]'.\n"
        "If Surya is installed and this still fails, its API has "
        "moved — check the installed version with\n"
        "  python -c \"import importlib.metadata as m; "
        "print(m.version('surya-ocr'))\"\n"
        "then adjust SuryaBackend.\n")

    def __init__(self, langs: list[str]) -> None:
        super().__init__(langs)
        self._shape: str | None = None
        self._rec = None
        # Whether Surya is there is asked now, before minutes of work. The
        # predictor itself, which starts the recognition server and loads
        # its model, waits for the first page that has to be read: a
        # rebuild from a full cache used to start a server it never spoke
        # to, and two such builds at once could find each other's.
        import importlib.util
        if "surya.recognition" not in sys.modules:
            try:
                found = importlib.util.find_spec("surya.recognition")
            except (ImportError, ValueError) as exc:
                raise RuntimeError(self._NOT_THERE
                                   + f"Underlying import error: {exc}") from exc
            if found is None:
                raise RuntimeError(self._NOT_THERE
                                   + "Underlying import error: no module "
                                     "named surya.recognition")

    def _load(self) -> None:
        if switches.CACHE_ONLY:
            # Not an exception a retry could catch: the promise of the flag
            # is that nothing is read, whatever asked.
            sys.exit("--cache-only: a page was sent to be read. Nothing "
                     "more was done. Build without the flag to read it.")
        llamacpp_safe_schemas()     # before any predictor: the fallback's layout
        try:
            from surya.recognition import RecognitionPredictor
        except ImportError as exc:
            raise RuntimeError(self._NOT_THERE
                               + f"Underlying import error: {exc}") from exc
        self._rec = RecognitionPredictor()
        self._shape = "predictor"

    def _predict(self, images: list[Image.Image]):
        if self._rec is None:
            self._load()
        switches.READ_THIS_RUN["pages"] += len(images)
        return self._rec(images)

    def _page_fragments(self, images: list[Image.Image]) -> list[list[str]]:
        """Per page, the usable HTML fragments in reading order."""
        pages: list[list[str]] = []
        for pred in self._predict(images):
            blocks = getattr(pred, "blocks", None) or []
            ordered = sorted(blocks, key=lambda b: getattr(b, "reading_order", 0))
            pages.append([
                b.html for b in ordered
                if getattr(b, "html", "") and not getattr(b, "skipped", False)
                and not getattr(b, "error", False)
            ])
        return pages

    def run(self, images: list[Image.Image]) -> list[str]:
        return [
            "\n\n".join(filter(None, (_strip_tags(f) for f in frags)))
            for frags in self._page_fragments(images)
        ]

    def run_html(self, images: list[Image.Image]) -> list[str]:
        return ["".join(_to_xhtml(f) for f in frags) for frags in self._page_fragments(images)]

    def run_items(self, images: list[Image.Image]) -> list[list[PageItem]]:
        pages: list[list[PageItem]] = []
        # Per page of this call: the blocks that were sent to be read and
        # came back with nothing, in error or simply empty. A picture is
        # never sent, so a page that is all picture counts none. When the
        # server behind recognition stops answering, this is the only trace
        # it leaves: the layout still comes back, the pictures are still
        # cut out, and every block of text is quietly missing.
        self.last_unread_blocks: list[int] = []
        for image, pred in zip(images, self._predict(images)):
            blocks = getattr(pred, "blocks", None) or []
            unread = sum(1 for b in blocks
                         if not getattr(b, "skipped", False)
                         and (getattr(b, "error", False)
                              or not getattr(b, "html", "")))
            self.last_unread_blocks.append(unread)

            items: list[PageItem] = []
            tops: list[float] = []          # each item's top edge, for box placement
            claimed: list[tuple[float, float, float, float]] = []
            box = getattr(pred, "image_bbox", None) or [0, 0, image.width, image.height]
            sx = image.width / max(1e-6, float(box[2]) - float(box[0]))
            sy = image.height / max(1e-6, float(box[3]) - float(box[1]))

            def block_top(b) -> float:
                poly = getattr(b, "polygon", None)
                return min(float(q[1]) for q in poly) * sy if poly else 0.0

            def block_box(b) -> tuple | None:
                poly = getattr(b, "polygon", None)
                if not poly:
                    return None
                xs = [float(q[0]) * sx for q in poly]
                ys = [float(q[1]) * sy for q in poly]
                return (min(xs) / image.width, min(ys) / image.height,
                        max(xs) / image.width, max(ys) / image.height)

            for b in sorted(blocks, key=lambda b: getattr(b, "reading_order", 0)):
                label = getattr(b, "label", "")
                if label in FURNITURE_LABELS:
                    # A running head belongs to the paper, not to the book — it
                    # never reaches the body. But its text is the folio audit's
                    # raw material, read here at full resolution with the whole
                    # page for context, so it rides along flagged instead of
                    # being discarded and re-derived later at lower quality.
                    frag = getattr(b, "html", "")
                    if frag and not getattr(b, "skipped", False) \
                            and not getattr(b, "error", False):
                        items.append(PageItem(html=_to_xhtml(frag),
                                              is_furniture=True,
                                              box=block_box(b),
                                              conf=getattr(b, "confidence", None),
                                              label=label or None))
                        tops.append(block_top(b))
                    continue
                if label in FIGURE_LABELS:
                    # Figures come back with skipped=True and no HTML — there is
                    # no text to read — so the pixels are the only content.
                    crop = _crop_block(image, b, pred)
                    if crop is not None:
                        items.append(PageItem(figure=crop, box=block_box(b),
                                              conf=getattr(b, "confidence", None),
                                              label=label or None))
                        tops.append(block_top(b))
                        poly = getattr(b, "polygon", None)
                        if poly:
                            xs = [float(q[0]) * sx for q in poly]
                            ys = [float(q[1]) * sy for q in poly]
                            claimed.append((min(xs), min(ys), max(xs), max(ys)))
                    continue
                frag = getattr(b, "html", "")
                if not frag or getattr(b, "skipped", False) or getattr(b, "error", False):
                    continue
                items.append(PageItem(html=_to_xhtml(frag),
                                      is_caption=label in CAPTION_LABELS,
                                      box=block_box(b),
                                      conf=getattr(b, "confidence", None),
                                      label=label or None))
                tops.append(block_top(b))

            # Drawings the layout never segmented — a boxed structure with text
            # wrapped around it leaves no block of its own, so the border is
            # found directly and the crop slotted in by where it sits on the
            # page, after whatever text stands above it.
            for (x0, y0, x1, y1) in find_ruled_boxes(image, claimed):
                pad = 4
                crop = image.crop((max(0, x0 - pad), max(0, y0 - pad),
                                   min(image.width, x1 + pad),
                                   min(image.height, y1 + pad)))
                at = next((k for k, t in enumerate(tops) if t > y0), len(items))
                items.insert(at, PageItem(figure=crop, box=(
                    x0 / image.width, y0 / image.height,
                    x1 / image.width, y1 / image.height)))
                tops.insert(at, float(y0))
            pages.append(items)
        return pages


OCR_BACKENDS: dict[str, type[OCRBackend]] = {
    "surya": SuryaBackend,
}


def load_backend(name: str, langs: list[str]) -> OCRBackend:
    if name not in OCR_BACKENDS:
        sys.exit(f"There is no OCR backend called {name!r}. Surya is the "
                 f"one this pipeline reads with: everything it does beyond "
                 f"reading characters depends on the layout Surya returns.")
    try:
        return OCR_BACKENDS[name](langs)
    except ImportError as exc:
        sys.exit(f"OCR backend '{name}' is not installed: {exc}\nSee the header of this file for install commands.")
    except RuntimeError as exc:
        sys.exit(str(exc))


# ── Text reflow for EPUB ──────────────────────────────────────────────────────

def reflow_paragraphs(text: str) -> list[str]:
    """Join OCR'd lines back into paragraphs so the EPUB reflows properly."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []

    # A line noticeably shorter than the page's typical line usually ends a
    # paragraph; full-width lines are mid-paragraph wraps.
    typical = statistics.median(len(ln) for ln in lines)
    paragraphs: list[str] = []
    buf = ""

    for line in lines:
        if buf.endswith("-"):
            buf = buf[:-1] + line
        elif buf:
            buf += " " + line
        else:
            buf = line
        if len(line) < 0.75 * typical:
            paragraphs.append(buf)
            buf = ""

    if buf:
        paragraphs.append(buf)
    return paragraphs
