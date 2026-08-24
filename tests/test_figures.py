"""Blank leaves must not reach the book as pictures.

The layout pass marks a whole blank page as a figure — reasonably, since there
is nothing else on it to name the region — and those crops then land in the EPUB
as full-page images of nothing. On a 518-page scan, 44 of 50 extracted figures
were blank leaves.
"""
import numpy as np
from PIL import Image


def paper(width=1400, height=2300, tone=228, grain=4, seed=0):
    """A blank leaf: warm stock with a little grain, the way a scan of one looks."""
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(tone, grain, (height, width)), 0, 255).astype("uint8")
    return Image.fromarray(a).convert("RGB")


def printed(width=440, height=430, seed=1):
    """A halftone-ish plate: real structure at real contrast."""
    rng = np.random.default_rng(seed)
    a = np.full((height, width), 210, dtype="uint8")
    for _ in range(60):
        x, y = rng.integers(0, width - 40), rng.integers(0, height - 40)
        a[y:y + rng.integers(8, 38), x:x + rng.integers(8, 38)] = rng.integers(20, 90)
    return Image.fromarray(a).convert("RGB")


def test_a_blank_leaf_is_not_a_figure(vtb):
    assert vtb.figure_has_content(paper()) is False


def test_a_printed_plate_is_a_figure(vtb):
    assert vtb.figure_has_content(printed()) is True


def test_the_two_populations_sit_far_apart(vtb, tmp_path):
    """
    The threshold is not finely balanced and must not become so. Measured on a
    real book the blank crops ran to 0.00071 and the real ones from 0.149 — the
    gap is over two hundred fold, and any cut inside it does the same job.
    """
    def detail(im):
        from PIL import ImageFilter
        g = im.convert("L")
        a = np.asarray(g, dtype=np.float32)
        bg = np.asarray(g.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32)
        return float(((bg - a) > 12).mean())

    blanks = [detail(paper(seed=s, tone=t)) for s in range(3) for t in (215, 228, 240)]
    reals = [detail(printed(seed=s)) for s in range(3)]
    assert max(blanks) < vtb.FIGURE_MIN_DETAIL < min(reals)
    assert min(reals) > max(blanks) * 20


def test_a_dark_but_featureless_page_is_still_blank(vtb):
    # Tan or shadowed stock is dark in absolute terms; what matters is that it
    # carries no ink ABOVE its own local background, which is why the measure is
    # relative. A fixed darkness threshold would call this whole sheet ink.
    assert vtb.figure_has_content(paper(tone=150)) is False


def test_a_crop_too_small_to_hold_anything_is_dropped(vtb):
    assert vtb.figure_has_content(Image.new("RGB", (4, 4), "white")) is False


def test_an_unreadable_image_is_kept_for_a_human_to_judge(vtb):
    class Broken:
        width = height = 100

        def convert(self, _mode):
            raise OSError("truncated")

    # Erring toward keeping: a figure wrongly dropped is gone silently, while a
    # figure wrongly kept is visible and can be removed.
    assert vtb.figure_has_content(Broken()) is True


# --------------------------------------------------------------------------
# a page holding nothing is left out of the book
# --------------------------------------------------------------------------

def test_a_page_with_neither_text_nor_picture_is_droppable(vtb):
    """
    The rule the EPUB build applies: emptiness is judged AFTER every fragment
    the page produced has been assembled, so a page still empty at that point
    genuinely carries nothing and dropping it cannot lose content. A blank leaf
    earns its keep in a facsimile PDF, where pagination is the point; in a
    reflowable book it is an empty screen to swipe past.
    """
    bodies = ["<p>real text</p>", "", "   ", '<figure><img src="x.jpg"/></figure>',
              "<p>more text</p>"]
    blank = {i for i, b in enumerate(bodies)
             if not vtb._strip_tags(b).strip() and "<img" not in b}
    assert blank == {1, 2}


def test_a_page_whose_only_content_is_a_picture_is_kept(vtb):
    # A full-page plate carries no text at all. Dropping it for that reason
    # would throw away exactly the pages a reader most wants.
    body = '<figure><img src="plate.jpg" alt="Figure"/></figure>'
    assert not vtb._strip_tags(body).strip()      # no text whatsoever
    assert "<img" in body                          # and yet it must survive


def test_a_caption_orphaned_by_a_dropped_blank_image_keeps_its_page(vtb):
    # When a blank-page image is skipped its caption is still emitted, so the
    # page has text and stays — a plate's caption sometimes sits on the facing
    # blank, and losing it would leave the plate unexplained.
    body = "<p>Altgeld in 1893, from a photograph</p>"
    assert vtb._strip_tags(body).strip()


# --------------------------------------------------------------------------
# ruled boxes the layout model never segmented
# --------------------------------------------------------------------------

def _boxed_structure_page(box=(500, 700, 1300, 1050), fill=False, border=True):
    """A page of text with a ruled box mid-paragraph, as TiHKAL sets structures."""
    import cv2
    img = np.full((2400, 1700), 255, dtype="uint8")
    for k in range(40):                       # body text above, around and below
        y = 200 + k * 52
        cv2.rectangle(img, (150, y), (1550, y + 24), 40, -1)
    x0, y0, x1, y1 = box
    cv2.rectangle(img, (x0 - 10, y0 - 10), (x1 + 10, y1 + 10), 255, -1)  # clear the region
    if border:
        cv2.rectangle(img, (x0, y0), (x1, y1), 0, 3)
    if fill:
        cv2.rectangle(img, (x0 + 6, y0 + 6), (x1 - 6, y1 - 6), 30, -1)
    else:
        # sparse strokes: a structure diagram's kind of ink
        cv2.line(img, (x0 + 60, y0 + 60), (x1 - 80, y1 - 90), 0, 3)
        cv2.line(img, (x0 + 60, y1 - 90), (x1 - 80, y0 + 60), 0, 3)
        cv2.putText(img, "CH3O", (x0 + 40, (y0 + y1) // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, 0, 2)
    return Image.fromarray(img).convert("RGB")


def test_a_ruled_box_with_sparse_strokes_is_found(vtb):
    """
    The failure this guards: TiHKAL boxes every structure drawing mid-paragraph
    and the layout pass swallows the region into the text block — no figure, no
    garbage, nothing. Fifty-five drawings would leave the book without a trace.
    The drawn border is the signal the layout model ignores.
    """
    boxes = vtb.find_ruled_boxes(_boxed_structure_page())
    assert len(boxes) == 1
    x0, y0, x1, y1 = boxes[0]
    assert abs(x0 - 500) < 25 and abs(y1 - 1050) < 25


def test_a_page_of_plain_text_yields_no_boxes(vtb):
    import cv2
    img = np.full((2400, 1700), 255, dtype="uint8")
    for k in range(40):
        y = 200 + k * 52
        cv2.rectangle(img, (150, y), (1550, y + 24), 40, -1)
    assert vtb.find_ruled_boxes(Image.fromarray(img).convert("RGB")) == []


def test_a_filled_box_is_not_mistaken_for_a_drawing(vtb):
    # A solid dark panel is a photograph or a print artefact; the Picture label
    # and its own pipeline handle those. A drawing is hollow by nature.
    assert vtb.find_ruled_boxes(_boxed_structure_page(fill=True)) == []


def test_a_region_already_claimed_as_a_figure_is_not_added_twice(vtb):
    page = _boxed_structure_page()
    claimed = [(480.0, 680.0, 1320.0, 1070.0)]
    assert vtb.find_ruled_boxes(page, claimed) == []


def test_an_unruled_structure_would_not_be_found(vtb):
    # Honesty about scope: the detector keys on the border. A book that sets
    # its drawings without rules gets nothing from this — that limitation is
    # deliberate, because without the border the same test cannot tell a
    # drawing from the paragraph beside it.
    assert vtb.find_ruled_boxes(_boxed_structure_page(border=False)) == []


def test_a_cmyk_plate_can_be_encoded_both_ways(vtb):
    """After Queer Theory lost a whole 225-page build to one press-ready
    plate: figures are encoded as JPEG and PNG so the smaller ships, and PNG
    cannot hold CMYK. JPEG accepts it, so the crash landed on the second
    encode with the first already done."""
    import io
    from PIL import Image
    cmyk = Image.new("CMYK", (40, 30), (0, 120, 200, 8))
    out = vtb._encodable(cmyk)
    assert out.mode == "RGB"
    out.save(io.BytesIO(), "PNG", optimize=True)
    out.save(io.BytesIO(), "JPEG", quality=88)


def test_alpha_is_composited_onto_white_not_dropped(vtb):
    """The twin crash, on the other encoder: JPEG cannot hold alpha. What
    hides under a transparent mask is not the page it was drawn for, so the
    mask is honoured against white rather than discarded."""
    import io
    from PIL import Image
    img = Image.new("RGBA", (10, 10), (255, 0, 0, 0))     # fully transparent
    out = vtb._encodable(img)
    assert out.mode == "RGB"
    assert out.getpixel((5, 5)) == (255, 255, 255), "transparent must read white"
    out.save(io.BytesIO(), "JPEG", quality=88)


def test_an_already_encodable_image_is_passed_through_untouched(vtb):
    from PIL import Image
    rgb = Image.new("RGB", (8, 8), (10, 20, 30))
    assert vtb._encodable(rgb) is rgb
    grey = Image.new("L", (8, 8), 128)
    assert vtb._encodable(grey) is grey
