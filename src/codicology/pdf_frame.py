"""One frame for a PDF page: what the page draws (its crop box), measured from
where the PDF's own coordinates start, and boxes converted between page
fractions and PDF points in it. Every place that asks the PDF about a
region of a page goes through these, and so does the question of how an
image the page draws is turned or flipped where it is placed.
"""
from __future__ import annotations

from PIL import Image


# Where on a PDF page a box in fractions of the page lies. A page's
# fractions are of what is drawn when it is rendered, the crop box; the
# PDF's own coordinates (an object's bounds, a character's box, a region
# of text asked for) are of its user space, whose origin is the media box's
# corner. The two agree only where the crop box starts at that corner, and
# on 8,844 pages of 42 books on the shelf it does not: a publisher's PDF
# whose trim is set 21 to 60 points in from the media box, on pages about
# 400 points wide. Every conversion used to divide by the page's size and
# stop there, so on those pages the table check asked the PDF for the text
# of a region a tenth of the page away from the table (121 tables in 30
# books measured in the wrong place, and the table Burning Up lost on page
# 69 not mended for it), and boxes found in the PDF were placed as far off
# on the page. There is one frame now and every conversion goes through it.
# (No page on the shelf is rotated; a /Rotate would need this to turn too.)

def _page_frame(page) -> "tuple[float, float, float, float]":
    """(x, y, width, height) of what the page draws, in its user space."""
    pw, ph = page.get_size()
    try:
        x0, y0, _, _ = page.get_bbox()
    except Exception:
        x0 = y0 = 0.0
    return float(x0), float(y0), float(pw), float(ph)


def _pdf_box_to_normalized(b, frame):
    """A PDF-space rect (bottom-up y) as this pipeline's normalized box
    (top-down y), in the page's frame (_page_frame). The y-flip convention
    lives here and in _normalized_to_pdf and nowhere else (R6)."""
    x0, y0, pw, ph = frame
    return ((b[0] - x0) / pw, 1 - (b[3] - y0) / ph,
            (b[2] - x0) / pw, 1 - (b[1] - y0) / ph)


def _normalized_to_pdf(box, frame) -> dict:
    """A normalized box as the left, bottom, right and top of the PDF's
    user space, as get_text_bounded takes them."""
    x0, y0, pw, ph = frame
    return {"left": x0 + box[0] * pw, "bottom": y0 + (1 - box[3]) * ph,
            "right": x0 + box[2] * pw, "top": y0 + (1 - box[1]) * ph}


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
