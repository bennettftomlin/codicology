"""A picture that recurs down the book is decoration, not a figure.

An Introduction to Cooperation and Mutualism carried the same ornamental band
eight times — 566x5895, printed down the margin beside the folio numbers,
2.5MB of a 7MB file. The artwork recovery was right that it was a picture and
wrong that it was a figure: it is a running head that happens to be drawn
instead of typeset, and the pipeline has always stripped those.

The rule this file guards is recurrence judged on the PICTURE. The tempting
cheap version — match on width and height — would be a catastrophe in a field
manual, where every diagram is drawn to one column width; FM 3-25.150 alone
carries 365 figures that would collapse into a handful of size classes. The
last test here is that book in miniature, and it fails loudly if anyone ever
swaps the fingerprint for a measurement.
"""
import io

import pytest
from PIL import Image, ImageDraw


def png(w, h, draw=None, seed=0):
    img = Image.new("RGB", (w, h), "white")
    if draw is not None:
        d = ImageDraw.Draw(img)
        draw(d, w, h)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def band(d, w, h):
    for k in range(0, h, 40):
        d.rectangle([0, k, w, k + 20], fill=(20, 60, 140))


def diagram(n):
    """A distinct drawing, at the one width every figure in a manual uses."""
    def go(d, w, h):
        d.rectangle([10, 10, w - 10, h - 10], outline="black", width=3)
        for k in range(n + 1):
            d.line([15, 20 + k * 12, w - 15, 20 + k * 12], fill="black", width=4)
        d.text((20, h - 30), f"Figure {n}", fill="black")
    return go


def test_a_band_repeated_down_the_book_is_furniture(vtb):
    art = png(60, 600, band)
    page_figures = [["images/fig_0000.png"], [], ["images/fig_0001.png"],
                    [], ["images/fig_0002.png"], ["images/fig_0003.png"]]
    data = {f"images/fig_{i:04d}.png": art for i in range(4)}
    doomed = vtb.find_image_furniture(page_figures, data)
    assert doomed == set(data)


def test_a_unique_picture_is_never_furniture(vtb):
    page_figures = [[f"images/fig_{i:04d}.png"] for i in range(6)]
    data = {f"images/fig_{i:04d}.png": png(400, 300, diagram(i))
            for i in range(6)}
    assert vtb.find_image_furniture(page_figures, data) == set()


def test_a_picture_shown_twice_is_content(vtb):
    """A chart reproduced for reference in a later chapter is the book saying
    something, not the page being decorated. Two is not a pattern."""
    art = png(400, 300, diagram(1))
    page_figures = [["images/fig_0000.png"], [], [], ["images/fig_0001.png"]]
    data = {"images/fig_0000.png": art, "images/fig_0001.png": art}
    assert vtb.find_image_furniture(page_figures, data) == set()


def test_three_copies_on_one_page_are_not_furniture(vtb):
    """Furniture recurs across PAGES. Three copies of a symbol in one table
    is that page's content, however often it repeats within it."""
    art = png(30, 30, band)
    page_figures = [["images/fig_0000.png", "images/fig_0001.png",
                     "images/fig_0002.png"]]
    data = {f"images/fig_{i:04d}.png": art for i in range(3)}
    assert vtb.find_image_furniture(page_figures, data) == set()


def test_distinct_diagrams_of_identical_size_all_survive(vtb):
    """FM 3-25.150 in miniature: 20 different figures, every one drawn to the
    same column width and height. Matching on dimensions would delete
    nineteen of them. Nothing here may be touched."""
    page_figures, data = [], {}
    for i in range(20):
        name = f"images/fig_{i:04d}.png"
        page_figures.append([name])
        data[name] = png(400, 300, diagram(i))
    assert len({len(v) for v in data.values()}) > 1, "fixture is too uniform"
    assert vtb.find_image_furniture(page_figures, data) == set(), \
        "identical dimensions were mistaken for identical pictures"


def test_the_ornament_goes_and_the_figures_on_its_pages_stay(vtb):
    """The mixed page is the real case: a chapter opening carries both the
    decorative band and a genuine illustration."""
    art = png(60, 600, band)
    page_figures = [["images/fig_0000.png", "images/fig_0001.png"],
                    ["images/fig_0002.png"],
                    ["images/fig_0003.png", "images/fig_0004.png"]]
    data = {
        "images/fig_0000.png": art,                  # ornament
        "images/fig_0001.png": png(400, 300, diagram(1)),
        "images/fig_0002.png": art,                  # ornament
        "images/fig_0003.png": art,                  # ornament
        "images/fig_0004.png": png(400, 300, diagram(2)),
    }
    doomed = vtb.find_image_furniture(page_figures, data)
    assert doomed == {"images/fig_0000.png", "images/fig_0002.png",
                      "images/fig_0003.png"}


def test_unreadable_bytes_are_left_alone_not_guessed_at(vtb):
    page_figures = [["images/fig_0000.png"], ["images/fig_0001.png"],
                    ["images/fig_0002.png"]]
    data = {f"images/fig_{i:04d}.png": b"not an image" for i in range(3)}
    assert vtb.find_image_furniture(page_figures, data) == set()


def test_a_missing_figure_does_not_crash_the_pass(vtb):
    page_figures = [["images/gone.png"], ["images/fig_0000.png"]]
    assert vtb.find_image_furniture(page_figures, {}) == set()


def test_signature_separates_pictures_that_share_a_shape(vtb):
    a = vtb._picture_signature(png(400, 300, diagram(3)))
    b = vtb._picture_signature(png(400, 300, diagram(9)))
    same = vtb._picture_signature(png(400, 300, diagram(3)))
    assert a is not None and a != b and a == same


def test_signature_refuses_what_it_cannot_read(vtb):
    assert vtb._picture_signature(b"") is None
    assert vtb._picture_signature(b"\x00\x01\x02") is None
