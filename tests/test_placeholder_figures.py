"""Pictures the layout declined to call pictures.

Surya sometimes returns a block whose whole content is an <img> with no
source — it has seen artwork and labelled the region text. The caption
survives and the artwork does not, so the book shows "Figure 6.19" with
nothing under it. The block's own box says where the picture is, so it can
be cut from the page afterwards, which also means caches already written can
be repaired without re-reading a single page.
"""
from PIL import Image, ImageDraw


def _page(tmp_path, size=(1000, 1400)):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    # a distinctive block of "artwork" in the middle third
    d.rectangle([150, 500, 850, 700], outline="black", width=3)
    for k in range(6):
        d.line([(160, 520 + k * 28), (840, 520 + k * 28)], fill="black", width=2)
    p = str(tmp_path / "page.png")
    im.save(p)
    return p


def test_a_placeholder_block_becomes_the_picture_it_marks(vtb, tmp_path):
    p = _page(tmp_path)
    items = [
        vtb.PageItem(html="<p>Figure 6.19</p>", box=(0.16, 0.33, 0.24, 0.34)),
        vtb.PageItem(html="<img/>", box=(0.15, 0.357, 0.85, 0.5)),
    ]
    out = vtb.recover_placeholder_figures(items, p)
    assert len(out) == 2
    assert out[0].html == "<p>Figure 6.19</p>"           # caption untouched
    assert out[1].figure is not None, "the picture was not recovered"
    assert out[1].html == ""
    # it cut the region the block claimed, not the whole page
    assert 650 <= out[1].figure.width <= 720, out[1].figure.size
    assert 180 <= out[1].figure.height <= 220, out[1].figure.size


def test_a_placeholder_among_real_words_only_loses_the_broken_tag(vtb, tmp_path):
    """An empty <img> renders as a broken image in every reader."""
    p = _page(tmp_path)
    items = [vtb.PageItem(html="<p>See <img/> the table above.</p>",
                          box=(0.1, 0.3, 0.9, 0.4))]
    out = vtb.recover_placeholder_figures(items, p)
    assert out[0].figure is None
    assert "<img" not in out[0].html
    assert "the table above" in out[0].html


def test_a_real_image_tag_is_left_alone(vtb, tmp_path):
    p = _page(tmp_path)
    items = [vtb.PageItem(html='<p><img src="images/fig_0001.jpg"/></p>',
                          box=(0.1, 0.3, 0.9, 0.4))]
    out = vtb.recover_placeholder_figures(items, p)
    assert out[0] is items[0]


def test_items_without_a_box_are_left_alone(vtb, tmp_path):
    """A cache written before geometry cannot say where the picture was."""
    p = _page(tmp_path)
    items = [vtb.PageItem(html="<img/>", box=None)]
    out = vtb.recover_placeholder_figures(items, p)
    assert out[0].figure is None and out[0] is items[0]


def test_a_page_with_no_placeholders_is_returned_untouched(vtb, tmp_path):
    p = _page(tmp_path)
    items = [vtb.PageItem(html="<p>ordinary prose</p>", box=(0.1, 0.2, 0.9, 0.3))]
    assert vtb.recover_placeholder_figures(items, p) is items
