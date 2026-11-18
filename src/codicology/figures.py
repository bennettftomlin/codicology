"""A page's pictures: the figure taken from the PDF at its own size where it
can be confirmed, a render or the page's crop otherwise; a picture that is
only a hand, a blank leaf, a repeated ornament, or type the page already
has, left out; artwork the layout missed recovered; and each encoded as
the smaller of PNG and JPEG.
"""
from __future__ import annotations

import cv2
import glob
import io
import json
import numpy as np
import os
import re
from PIL import Image, ImageFilter
from collections import Counter
from .capture import _skin_mask
from .furniture import FURNITURE_PAGES
from .markup import _strip_tags
from .model import PageItem
from .opinions import _figures, _typeset_text_in
from .pdf_frame import _box_iou, _normalized_to_pdf, _page_frame, _pdf_box_to_normalized, native_orientation


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


ORPHAN_TEXT_SHARE = 0.4     # of a stored picture's place that read as text (white between lines counts against it)
# |correlation| under which the stored picture is not what is drawn, at any
# quarter turn: Disturbing Times stores its photographs of two opening pages
# on their side (0.67 turned upright, 0.00 as stored); the map tiles hidden
# under How the West Came to Rule's p. 115 reach 0.47 at best.
ORPHAN_UNRELATED = 0.5
# …and a drawing stored as a negative comes out clearly against the page as it
# lies, however much of it is labels drawn on top: Where There Is No Doctor's
# bottles and back-pain figure at -0.36 and -0.49, its calendars at -0.8 and
# below. The hidden tiles lie within 0.12 of nothing either way.
ORPHAN_NEGATIVE = -0.3


def _touches(a, b, reach: float = 0.01) -> bool:
    """Whether two boxes meet or overlap, within `reach` of the page."""
    return (a[0] <= b[2] + reach and b[0] <= a[2] + reach
            and a[1] <= b[3] + reach and b[1] <= a[3] + reach)


def _unrelated(art: "Image.Image", ref: "Image.Image") -> bool:
    """Whether a stored picture is not what the page draws in its place: no
    match at any quarter turn, and not a negative of it."""
    if _correlation(art, ref) <= ORPHAN_NEGATIVE:
        return False
    return max(abs(_correlation(art.rotate(turn, expand=True), ref))
               for turn in (0, 90, 180, 270)) < ORPHAN_UNRELATED


def _correlation(a: "Image.Image", b: "Image.Image") -> float:
    import numpy as np
    try:
        x = np.asarray(a.convert("L").resize((96, 96)), dtype=np.float32).ravel()
        y = np.asarray(b.convert("L").resize((96, 96)), dtype=np.float32).ravel()
    except Exception:
        return 1.0
    if x.std() < 1e-3 and y.std() < 1e-3:
        return 1.0 if abs(x.mean() - y.mean()) < 8 else 0.0
    if x.std() < 1e-3 or y.std() < 1e-3:
        return 0.0              # a blank tile against a drawing, or the reverse
    return float(np.corrcoef(x, y)[0, 1])


def _read_as_text(items: list, box) -> float:
    """How much of BOX the page's text blocks cover, as a share of it."""
    x0, y0, x1, y1 = box
    area = max(1e-9, (x1 - x0) * (y1 - y0))
    got = 0.0
    for it in items:
        if it.figure is not None or not it.box or it.is_furniture \
                or not (it.html and _strip_tags(it.html).strip()):
            continue
        a0, b0, a1, b1 = it.box
        w, h = min(x1, a1) - max(x0, a0), min(y1, b1) - max(y0, b0)
        if w > 0 and h > 0:
            got += w * h
    return min(1.0, got / area)


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

    decided = []                # (placed, art, hidden)
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
        # The picture is stored there and not shown: what the page draws in
        # its place is prose the reading already has. How the West Came to
        # Rule keeps a map's tiles under a page of text on p. 115, and six
        # crops of that text were printed as pictures. Unrelated, not merely
        # unlike: a drawing stored as a negative, or on its side, is the
        # same drawing (see _unrelated); and a drawing stored in bands keeps
        # every band if one of them is kept (below): FM 5-103's title
        # drawing has a band its lettering made read as text.
        hidden = (art is not None and ref is not None and _unrelated(art, ref)
                  and (_read_as_text(items, placed) >= ORPHAN_TEXT_SHARE
                       or not figure_has_content(ref)))
        if art is None or ref is None or not _looks_like_same_picture(art, ref):
            art = ref               # the page as drawn, if the stored pixels
        decided.append((placed, art, hidden))   # cannot be confirmed
    kept = [p for p, a, h in decided if not h and a is not None]
    out = list(items)
    for placed, art, hidden in decided:
        if hidden and not any(_touches(placed, k) for k in kept):
            continue
        if art is None:
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
