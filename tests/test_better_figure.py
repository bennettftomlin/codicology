"""Taking a figure from the PDF instead of from the page raster.

A figure cropped out of the rendered page is a crop of a resample — one
manual's line art was upscaled 3.12x before anything saw it, and JPEG rang
every hard edge that upscaling made. Where the PDF holds the artwork, the
artwork itself is better. What must never happen is passing through the
WRONG picture, so every substitution is confirmed against the crop it
replaces.
"""
import os
import pytest
from PIL import Image, ImageDraw

pdfium = pytest.importorskip("pypdfium2")

FM = ("/Users/Bennett1/claude_knowledge/assets/books/"
      "FM21-76_SurvivalManual.pdf")


def _drawing(size, seed=0):
    im = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(im)
    for k in range(6):
        y = 5 + k * (size[1] // 8) + seed
        d.line([(4, y), (size[0] - 4, y + seed)], fill="black", width=2)
    d.ellipse([size[0] // 4, size[1] // 4, size[0] // 2, size[1] // 2], outline="black")
    return im


def test_same_picture_at_different_sizes_is_recognised(vtb):
    big = _drawing((600, 400))
    small = big.resize((150, 100))
    assert vtb._looks_like_same_picture(big, small)


def test_a_different_picture_is_rejected(vtb):
    assert not vtb._looks_like_same_picture(_drawing((400, 300), 0),
                                            _drawing((400, 300), 11).rotate(90, expand=True))


def test_a_mask_lost_to_black_is_rejected(vtb):
    """An extracted image whose transparency became black must not pass."""
    art = _drawing((400, 300))
    blacked = Image.new("RGB", (400, 300), "black")
    assert not vtb._looks_like_same_picture(blacked, art)


def test_without_a_source_sidecar_the_crop_is_kept(vtb, tmp_path):
    crop = _drawing((200, 150))
    p = str(tmp_path / "page.jpg")
    crop.save(p)
    out = vtb.better_figure(p, (0.1, 0.1, 0.5, 0.5), crop)
    assert out is crop


def test_a_missing_box_keeps_the_crop(vtb, tmp_path):
    crop = _drawing((200, 150))
    p = str(tmp_path / "page.jpg"); crop.save(p)
    assert vtb.better_figure(p, None, crop) is crop


@pytest.mark.skipif(not os.path.exists(FM), reason="source book absent")
def test_a_real_figure_comes_back_at_its_native_size(vtb, tmp_path):
    """
    FM 21-76 page 41 holds one 450x292 drawing that our 300dpi raster blows
    up to 1406x911. The figure we ship should be the drawing, not the
    enlargement.
    """
    pages_dir = tmp_path / "pages"; pages_dir.mkdir()
    src = pdfium.PdfDocument(FM)
    one = pdfium.PdfDocument.new()
    one.import_pages(src, [40])
    slice_path = str(tmp_path / "slice.pdf")
    one.save(slice_path)
    paths = vtb.load_pages_from_pdf(slice_path, str(pages_dir))
    assert os.path.exists(paths[0] + ".source.json")
    # the drawing's true box, as the layout pass would report it
    pg = pdfium.PdfDocument(slice_path)[0]
    pw, ph = pg.get_size()
    img_obj = next(o for o in pg.get_objects() if isinstance(o, pdfium.PdfImage))
    b = img_obj.get_bounds()
    box = (b[0] / pw, 1 - b[3] / ph, b[2] / pw, 1 - b[1] / ph)
    native_w = img_obj.get_metadata().width
    with Image.open(paths[0]) as page_img:
        page_img.load()
        x0, y0 = int(box[0] * page_img.width), int(box[1] * page_img.height)
        x1, y1 = int(box[2] * page_img.width), int(box[3] * page_img.height)
        crop = page_img.crop((x0, y0, x1, y1)).copy()
    assert crop.width > native_w * 2, "the raster really is an enlargement"
    better = vtb.better_figure(paths[0], box, crop)
    assert better.width == native_w, (
        f"expected the drawing at its native {native_w}px, got {better.size}")


def test_a_facsimile_page_keeps_its_crop_rather_than_being_re_rendered(vtb, tmp_path):
    """
    Citadel's pages are photographs wrapped in a PDF: the page IS the image.
    Re-rendering a region of one only resamples pixels the crop already holds
    exactly, so the crop must win — a re-render here is a loss dressed up as
    an improvement, and the picture-comparison guard would wave it through.
    """
    import json
    crop = _drawing((800, 600))
    p = str(tmp_path / "page.jpg"); crop.save(p)
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": str(tmp_path / "nonexistent.pdf"), "page": 0,
                   "facsimile": True}, fh)
    assert vtb.better_figure(p, (0.1, 0.1, 0.6, 0.6), crop) is crop


def test_line_art_ships_lossless_and_photographs_ship_as_jpeg(vtb, tmp_path):
    """
    An EPUB figure is a final artifact, so size counts — and for line art the
    lossless encoding is also the smaller one, because JPEG spends its bits
    ringing hard black edges. A photograph inverts that by more than four to
    one. Measuring both and keeping the smaller sorts them with no classifier.
    """
    import io
    from PIL import Image
    line = _drawing((900, 600))
    jb = io.BytesIO(); line.save(jb, "JPEG", quality=88)
    pb = io.BytesIO(); line.save(pb, "PNG", optimize=True)
    assert len(pb.getvalue()) < len(jb.getvalue()), "line art: PNG must win"

    rng = __import__("numpy").random.default_rng(0)
    noise = Image.fromarray(
        rng.integers(0, 255, (600, 900, 3), dtype="uint8"), "RGB")
    jb2 = io.BytesIO(); noise.save(jb2, "JPEG", quality=88)
    pb2 = io.BytesIO(); noise.save(pb2, "PNG", optimize=True)
    assert len(jb2.getvalue()) < len(pb2.getvalue()), "photographic: JPEG must win"
