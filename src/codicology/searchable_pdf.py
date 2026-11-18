"""The searchable PDF: each page's words set as invisible text over its
picture, placed by tesseract's word boxes where it read the same words and
by the layout's blocks otherwise.
"""
from __future__ import annotations

import html
import re
from PIL import Image
from .markup import _strip_tags
from .model import PageItem


# The layer is written with one TextWriter a page in a real font embedded
# with its Unicode map. The Base-14 "helv" of insert_text has an 8-bit
# encoding and set every character outside it as a dot: "don’t" came back
# "don·t", Greek and Cyrillic as dots, and none of it could be found. The
# Font("helv") file has Greek, Cyrillic, curly quotes and ligatures; a
# character it lacks still keeps its Unicode, so it is found by search.
_RTL = re.compile(r"[\u0590-\u08ff\ufb1d-\ufdff\ufe70-\ufeff]")


def _rtl(text: str) -> bool:
    """Whether text is in a right-to-left script: written front to back it
    is found by search only when the writer is told so. A block is so when
    most of its letters are; told so, the writer reverses all of it, and a
    Hebrew word in an English paragraph would turn the English round."""
    letters = sum(ch.isalpha() for ch in text)
    return letters > 0 and len(_RTL.findall(text)) * 2 > letters


class _Measure:
    """Widths in a font at size 1, character by character, remembered: the
    font's own text_length asks MuPDF once a character every time, and was
    most of the time the layer took."""

    def __init__(self, font):
        self.font, self.seen = font, {}

    def __call__(self, text: str) -> float:
        seen, total = self.seen, 0.0
        for ch in text:
            w = seen.get(ch)
            if w is None:
                w = seen[ch] = self.font.text_length(ch, fontsize=1)
            total += w
        return total


def _lines_in(measure, text: str, width: float, height: float, fs: float,
              line_h: float) -> "list[str] | None":
    """The text broken into lines that fill a box at size fs, word by word,
    or None where it does not fit: a word wider than the box (broken across
    lines it could not be found) or more lines than the box holds."""
    space = measure(" ")
    lines, cur, cur_w = [], [], 0.0
    for word in text.split():
        w = measure(word)
        if w * fs > width:
            return None
        if cur and (cur_w + space + w) * fs > width:
            lines.append(" ".join(cur))
            cur, cur_w = [word], w
        else:
            cur_w = cur_w + space + w if cur else w
            cur.append(word)
    if cur:
        lines.append(" ".join(cur))
    return lines if len(lines) * line_h * fs <= height else None


# Invisible text may be set as small as it likes — nobody reads it, and search
# does not care about size. The floor exists only so a font size stays a
# positive number the writer will accept.
_LAYER_MIN_FONT = 0.4


def _force_text_into(writer, font, rect, text: str) -> None:
    """
    Write text inside rect whatever it takes, as a last resort.

    Sized from the box's area rather than tried and retried, so the fit is
    arithmetic instead of hopeful: every character goes in. The layout is
    nothing like the printed page's, which costs a highlight that covers the
    block loosely — the alternative on this path is losing the words.
    """
    import math

    fs = math.sqrt(max(1.0, abs(rect.get_area())) / (0.9 * max(1, len(text))))
    fs = max(0.02, min(fs, max(0.02, rect.height)))
    per_line = max(1, int(rect.width / (fs * 0.62)))
    y = rect.y0 + fs
    for i in range(0, len(text), per_line):
        line = text[i:i + per_line]
        writer.append((rect.x0, min(y, rect.y1 + fs)), line, font=font,
                      fontsize=fs, right_to_left=_rtl(line))
        y += fs * 1.2


def _tesseract_word_boxes(page_path: str) -> "list[dict] | None":
    """
    Word-level geometry from a classical OCR, or None if unavailable.

    Surya reports geometry per layout block only, so a paragraph is one
    rectangle and selection drifts inside it. Tesseract reads words WITH
    their boxes; its words are not trusted as text — surya's reading stays
    the truth — but where both engines read the same word inside the same
    block, tesseract's box places it. Measured on six pages: words sitting
    on their own ink went from 70.6% (block placement) to 92.9%.
    """
    import shutil
    import subprocess
    if not shutil.which("tesseract"):
        return None
    try:
        r = subprocess.run(["tesseract", page_path, "stdout", "tsv"],
                           capture_output=True, text=True, timeout=180)
    except Exception:
        return None
    out = []
    for line in r.stdout.splitlines()[1:]:
        c = line.split("\t")
        if len(c) == 12 and c[0] == "5" and c[11].strip():
            try:
                x, y, w, h, conf = map(float, (c[6], c[7], c[8], c[9], c[10]))
            except ValueError:
                continue
            if conf >= 30:
                t = re.sub(r"^[^\w]+|[^\w]+$", "", c[11].lower())
                if t:
                    out.append({"t": t, "x": x, "y": y, "w": w, "h": h,
                                "used": False})
    return out


def build_searchable_pdf(page_paths: list[str],
                         items_per_page: list[list[PageItem]],
                         out_path: str) -> tuple[int, int, int]:
    """
    The cleaned page images with the OCR's text laid invisibly over each
    block, so the PDF searches, selects and copies like a born-digital file
    while showing exactly the pixels the camera kept.

    Placement is per layout block — the finest geometry surya records; it
    reports no line or character boxes, so a highlight covers the block that
    holds the hit rather than the exact words.

    No block's text is ever dropped. Refusing to place something is right
    where a wrong answer would assert a falsehood — a note link, a folio —
    but this layer has no reader: text nobody sees cannot mislead anyone.
    The costs are lopsided the other way. A block laid out imperfectly puts
    a highlight rectangle slightly wrong; a block left out makes the words
    unfindable, and silently, since nothing in the file says they are gone.
    So a block that will not fit at a natural size is shrunk, and one that
    will not fit at any size is written in a forced layout of its own.
    Returns (blocks placed, blocks forced).
    """
    import math
    import unicodedata
    import pymupdf

    doc = pymupdf.open()
    font = pymupdf.Font("helv")
    measure = _Measure(font)
    ascent, line_h = font.ascender, font.ascender - font.descender
    placed = forced = word_placed = 0
    for path, items in zip(page_paths, items_per_page):
        with Image.open(path) as im:
            w_px, h_px = im.size
            dpi = (im.info.get("dpi") or (300, 300))[0] or 300
        scale_pt = 72.0 / dpi
        w_pt, h_pt = w_px * scale_pt, h_px * scale_pt
        page = doc.new_page(width=w_pt, height=h_pt)
        page.insert_image(page.rect, filename=path)
        writer = pymupdf.TextWriter(page.rect)
        tess = _tesseract_word_boxes(path)
        for it in items:
            if not it.html or it.box is None:
                continue
            text = unicodedata.normalize(
                "NFKC", html.unescape(_strip_tags(it.html))).strip()
            if not text:
                continue
            if tess:
                # Words both engines read take tesseract's box. Words only
                # surya read are interpolated from their matched neighbours
                # — between the end of the previous placed word and the start
                # of the next is almost always where the word actually sits —
                # and every word is inserted in READING order, because the
                # content stream's order is what search and copy follow: a
                # layer whose words are shuffled breaks phrase search across
                # every matched/unmatched seam.
                bx0, by0 = it.box[0] * w_px, it.box[1] * h_px
                bx1, by1 = it.box[2] * w_px, it.box[3] * h_px
                words = text.split()
                wrects: list = [None] * len(words)
                for wi, word in enumerate(words):
                    nw = re.sub(r"^[^\w]+|[^\w]+$", "", word.lower())
                    if len(nw) < 2:
                        continue
                    for cand in tess:
                        if cand["used"] or cand["t"] != nw:
                            continue
                        cx = cand["x"] + cand["w"] / 2
                        cy = cand["y"] + cand["h"] / 2
                        if bx0 - 20 <= cx <= bx1 + 20 \
                                and by0 - 20 <= cy <= by1 + 20:
                            cand["used"] = True
                            wrects[wi] = pymupdf.Rect(
                                cand["x"] * scale_pt, cand["y"] * scale_pt,
                                (cand["x"] + cand["w"]) * scale_pt,
                                (cand["y"] + cand["h"]) * scale_pt)
                            break
                anchors = [wi for wi, r in enumerate(wrects) if r is not None]
                if anchors:
                    heights = [wrects[wi].height for wi in anchors]
                    line_h = sum(heights) / len(heights)
                    cw = sum(wrects[wi].width / max(1, len(words[wi]))
                             for wi in anchors) / len(anchors)
                    blk_l, blk_r = it.box[0] * w_pt, it.box[2] * w_pt
                    est = lambda w: cw * (len(w) + 0.5)
                    for wi, word in enumerate(words):
                        if wrects[wi] is not None:
                            continue
                        prev = next((j for j in reversed(anchors) if j < wi), None)
                        nxt = next((j for j in anchors if j > wi), None)
                        if prev is not None and nxt is not None:
                            a, b = wrects[prev], wrects[nxt]
                            same_line = abs((a.y0 + a.y1) - (b.y0 + b.y1)) / 2 \
                                < line_h * 0.6
                            if same_line:
                                # share the gap between the two anchors
                                run = [k for k in range(prev + 1, nxt)
                                       if wrects[k] is None]
                                slot = (b.x0 - a.x1) / max(1, len(run))
                                off = run.index(wi)
                                x = a.x1 + slot * off
                                wrects[wi] = pymupdf.Rect(
                                    x, a.y0, x + max(slot, est(word)), a.y1)
                            else:
                                # a line break sits in the run: fill toward
                                # the block edge, then before the next anchor
                                x = a.x1 + cw * 0.4
                                if x + est(word) > blk_r:
                                    x = max(blk_l, b.x0 - est(word))
                                    a = b
                                wrects[wi] = pymupdf.Rect(
                                    x, a.y0, x + est(word), a.y1)
                        elif prev is not None:
                            a = wrects[prev]
                            x = a.x1 + cw * 0.4
                            if x + est(word) > blk_r:
                                x, a = blk_l, pymupdf.Rect(
                                    blk_l, a.y0 + line_h * 1.1,
                                    blk_r, a.y1 + line_h * 1.1)
                            wrects[wi] = pymupdf.Rect(x, a.y0,
                                                      x + est(word), a.y1)
                        else:
                            b = wrects[nxt]
                            x = max(blk_l, b.x0 - est(word) - cw * 0.4)
                            wrects[wi] = pymupdf.Rect(x, b.y0,
                                                      x + est(word), b.y1)
                        # synthesized boxes stay honest about their origin:
                        # inside the block, near the anchors, never beyond
                        wrects[wi] = wrects[wi] & pymupdf.Rect(
                            blk_l, it.box[1] * h_pt, blk_r, it.box[3] * h_pt) \
                            or wrects[wi]
                    for wi, word in enumerate(words):
                        r = wrects[wi]
                        unit = measure(word)
                        fs = max(1.0, min(r.height * 1.05,
                                          r.width / max(0.1, unit)))
                        # the trailing space is what tells a reader of the
                        # file that the next word is another word
                        writer.append((r.x0, r.y1 - 0.15 * fs), word + " ",
                                      font=font, fontsize=fs,
                                      right_to_left=_rtl(word))
                    word_placed += len(anchors)
                    placed += 1
                    continue
            x0, y0, x1, y1 = it.box
            rect = pymupdf.Rect(x0 * w_pt, y0 * h_pt, x1 * w_pt, y1 * h_pt)
            rect.normalize()
            if rect.width < 1 or rect.height < 1:
                # Degenerate geometry — a sliver of a polygon. Give it a
                # minimal box where the block claimed to be, so the words
                # still enter the file.
                rect = pymupdf.Rect(
                    min(rect.x0, w_pt - 24), min(rect.y0, h_pt - 8),
                    min(w_pt, max(rect.x1, rect.x0 + 24)),
                    min(h_pt, max(rect.y1, rect.y0 + 8)))
            # Start at the size that would just fill the box rather than
            # walking down from an arbitrary maximum: a dense block lands in
            # a step or two instead of eight, and a sparse one keeps a size
            # close to the ink it sits under.
            fs = min(rect.height * 0.9,
                     math.sqrt(max(1.0, abs(rect.get_area()))
                               / (0.62 * max(1, len(text)))))
            fs = max(_LAYER_MIN_FONT, min(fs, 72.0))
            ok = False
            rtl = _rtl(text)
            while True:
                lines = _lines_in(measure, text, rect.width, rect.height, fs,
                                  line_h)
                if lines is not None:
                    for k, line in enumerate(lines):
                        writer.append(
                            (rect.x0, rect.y0 + (ascent + k * line_h) * fs),
                            line + " ", font=font, fontsize=fs,
                            right_to_left=rtl)
                    ok = True
                    break
                if fs <= _LAYER_MIN_FONT:
                    break
                fs = max(_LAYER_MIN_FONT, fs * 0.75)
            if ok:
                placed += 1
            else:
                _force_text_into(writer, font, rect, text)
                forced += 1
        # invisible (render mode 3), in the order appended: reading order
        writer.write_text(page, render_mode=3)
    doc.set_metadata({"producer": "video_to_book"})
    doc.save(out_path, deflate=True, garbage=3)
    doc.close()
    return placed, forced, word_placed
