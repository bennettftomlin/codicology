"""A page's reading is filed under what the page shows.

It was filed under the SHA-256 of the page's file, and for a page rendered
from a PDF that file is a PNG this pipeline wrote: the name depended on
Pillow's encoder. A warm rebuild spent half its time encoding PNGs only to
hash them, and a new encoder would have orphaned every rendered page's
reading on the shelf.

Now a PNG is named for its pixels and everything else for its bytes, and a
cache written under the old names is brought over the first time each page
is asked for — which needs the old name to be reproducible exactly, so the
recipe is pinned here too.
"""
import gzip
import hashlib
import io
import json
import os

import pytest
from PIL import Image

from codicology import pipeline as vtb
from test_page_loading import _png, _printed, make_book, typeset_page

pytest.importorskip("pypdfium2")


def _picture(shade=40):
    im = Image.new("RGB", (120, 160), "white")
    for x in range(20, 100):
        for y in range(30, 40):
            im.putpixel((x, y), (shade, shade, shade))
    return im


def _stored(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _old_cache(path, pages):
    """A cache as every build before this wrote it."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"tag": "surya|en", "pages": pages}, fh)


def _entry(text):
    return {"v": vtb.CACHE_VERSION, "items": [{"html": f"<p>{text}</p>"}]}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


# ── the name ─────────────────────────────────────────────────────────────────

def test_a_png_is_named_for_its_pixels_not_its_encoding(tmp_path):
    quick, small = str(tmp_path / "quick.png"), str(tmp_path / "small.png")
    _picture().save(quick, "PNG", compress_level=1)
    _picture().save(small, "PNG", optimize=True, dpi=(300, 300))
    assert open(quick, "rb").read() != open(small, "rb").read()
    assert vtb.page_key(quick) == vtb.page_key(small)

    other = str(tmp_path / "other.png")
    _picture(shade=41).save(other, "PNG")
    assert vtb.page_key(other) != vtb.page_key(quick)


def test_the_same_pixels_in_another_shape_are_another_page(tmp_path):
    a, b = str(tmp_path / "a.png"), str(tmp_path / "b.png")
    Image.new("L", (30, 20), 255).save(a)
    Image.new("L", (20, 30), 255).save(b)
    assert vtb.page_key(a) != vtb.page_key(b)


def test_a_jpeg_is_named_for_its_bytes(tmp_path):
    """Those bytes were given to us, and a JPEG does not decode to the same
    pixels on every machine."""
    page = str(tmp_path / "page.jpg")
    _picture().save(page, "JPEG", quality=90)
    assert vtb.page_key(page) == _sha(open(page, "rb").read())


def test_a_png_that_is_not_a_picture_has_only_its_bytes(tmp_path):
    page = tmp_path / "page.png"
    page.write_bytes(b"\x89PNG not really")
    assert vtb.page_key(str(page)) == _sha(page.read_bytes())


def test_the_loaders_sidecar_is_the_name_and_a_damaged_one_is_not(tmp_path):
    page = str(tmp_path / "page.png")
    _picture().save(page, "PNG")
    shown = vtb.page_key(page)
    with open(page + ".key", "w") as fh:
        fh.write("ab" * 32 + "\n")
    assert vtb.page_key(page) == "ab" * 32
    with open(page + ".key", "w") as fh:
        fh.write("not a key")
    assert vtb.page_key(page) == shown


# ── the loader ───────────────────────────────────────────────────────────────

def _leaf(n):
    return [("text", (50, 60 + 14 * k), f"Leaf {n}, line {k}, set in type.")
            for k in range(30)]


@pytest.fixture
def book(tmp_path):
    """Three pages, no two alike: pages that show the same thing share a
    name, and a reading, as they always did."""
    return make_book(tmp_path / "book.pdf", [_leaf(n) for n in range(3)])


def _as_every_earlier_build_wrote(pdf, i):
    """The loader's own expression until now, kept here word for word: the
    bytes of this file were the page's name in every cache on the shelf."""
    import pypdfium2 as pdfium
    buf = io.BytesIO()
    pdfium.PdfDocument(pdf)[i].render(
        scale=300 / 72, draw_annots=False).to_pil().convert("RGB").save(
        buf, "PNG", dpi=(300, 300))
    return buf.getvalue()


def test_a_rendered_page_carries_its_name_and_the_same_pixels(book, tmp_path):
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(book, str(d))
    for i, page in enumerate(pages):
        with Image.open(page) as im, \
                Image.open(io.BytesIO(_as_every_earlier_build_wrote(book, i))) as was:
            assert im.tobytes() == was.tobytes() and im.size == was.size
            assert im.info["dpi"] == pytest.approx((300, 300), abs=0.01)
            assert open(page + ".key").read() == vtb._pixel_key(im)
        assert not os.path.exists(page + ".key1"), "not asked for, not paid for"


def test_the_old_name_is_the_old_files_hash(book, tmp_path):
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(book, str(d), former_keys=True)
    for i, page in enumerate(pages):
        was = _sha(_as_every_earlier_build_wrote(book, i))
        assert open(page + ".key1").read() == was
        os.remove(page + ".key1")       # and worked out again when wanted
        assert vtb.former_page_key(page) == was
        assert vtb.page_key(page) != was


def test_a_page_lifted_out_as_a_png_is_named_for_its_pixels(tmp_path):
    """A facsimile page that is not a JPEG is decoded by pdfium and encoded
    again by Pillow on the way out, so its bytes are Pillow's too."""
    leaf = [("image", (0, 0, 466, 704), _png((1000, 1511), _printed))]
    pdf = make_book(tmp_path / "scan.pdf", [leaf])
    d = tmp_path / "pages"; d.mkdir()
    (page,) = vtb.load_pages_from_pdf(pdf, str(d))
    assert json.load(open(page + ".source.json"))["facsimile"] is True
    with Image.open(page) as im:
        assert im.size == (1000, 1511)
        assert open(page + ".key").read() == vtb._pixel_key(im)
    # its old name was the file it still is
    assert vtb.former_page_key(page) == _sha(open(page, "rb").read())


# ── a cache from before ──────────────────────────────────────────────────────

def test_an_old_cache_serves_its_pages_and_is_refiled(book, tmp_path,
                                                      monkeypatch):
    path = str(tmp_path / "c.ocr.gz")
    _old_cache(path, {_sha(_as_every_earlier_build_wrote(book, i)):
                      _entry(f"page {i}") for i in range(3)})
    assert vtb.OCRCache.filed_by_bytes(path) is True

    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(book, str(d))
    cache = vtb.OCRCache(path, "surya", ["en"])
    assert [cache.get(p)[0].html for p in pages] == \
           [f"<p>page {i}</p>" for i in range(3)]
    assert cache.refiled == 3 and cache.knows_furniture(pages[0])
    cache.save()

    stored = _stored(path)
    assert stored["keying"] == vtb.PAGE_KEYING and stored["by_bytes"] == []
    assert set(stored["pages"]) == {vtb.page_key(p) for p in pages}
    assert vtb.OCRCache.filed_by_bytes(path) is False

    # the next build never asks what a page used to be called
    monkeypatch.setattr(vtb, "former_page_key",
                        lambda p: pytest.fail("the old name was asked for"))
    again = vtb.OCRCache(path, "surya", ["fr"])
    assert all(again.get(p) is not None for p in pages)
    assert again.refiled == 0 and not again.dirty


def test_a_page_the_old_cache_never_read_is_a_miss(book, tmp_path):
    path = str(tmp_path / "c.ocr.gz")
    _old_cache(path, {_sha(_as_every_earlier_build_wrote(book, 0)):
                      _entry("page 0")})
    d = tmp_path / "pages"; d.mkdir()
    pages = vtb.load_pages_from_pdf(book, str(d))
    cache = vtb.OCRCache(path, "surya", ["en"])
    assert cache.get(pages[1]) is None
    assert cache.get(pages[0]) is not None and cache.refiled == 1


def test_a_photograph_keeps_the_name_it_had(tmp_path):
    """Pages that came as JPEGs were always named for their bytes: found
    where they are, and no longer counted among the ones to bring over."""
    page = str(tmp_path / "page_0000.jpg")
    _picture().save(page, "JPEG")
    path = str(tmp_path / "c.ocr.gz")
    _old_cache(path, {_sha(open(page, "rb").read()): _entry("a photograph")})
    cache = vtb.OCRCache(path, "surya", ["en"])
    assert cache.by_bytes
    assert cache.get(page)[0].html == "<p>a photograph</p>"
    assert cache.refiled == 0 and not cache.by_bytes


def test_a_new_reading_is_filed_under_the_new_name(book, tmp_path):
    d = tmp_path / "pages"; d.mkdir()
    (page, *_rest) = vtb.load_pages_from_pdf(book, str(d))
    path = str(tmp_path / "c.ocr.gz")
    cache = vtb.OCRCache(path, "surya", ["en"])
    cache.put(page, [vtb.PageItem(html="<p>read today</p>")])
    cache.save()
    stored = _stored(path)
    assert list(stored["pages"]) == [open(page + ".key").read()]
    assert stored["by_bytes"] == []


def test_filed_by_bytes_is_false_for_no_cache_and_for_a_broken_one(tmp_path):
    assert vtb.OCRCache.filed_by_bytes(str(tmp_path / "absent.ocr.gz")) is False
    broken = tmp_path / "broken.ocr.gz"
    broken.write_bytes(b"not gzip")
    assert vtb.OCRCache.filed_by_bytes(str(broken)) is False


# ── a whole cache at once ────────────────────────────────────────────────────

def test_rekey_brings_a_cache_over_without_touching_a_reading(book, tmp_path,
                                                             capsys):
    path = str(tmp_path / "c.ocr.gz")
    old = {_sha(_as_every_earlier_build_wrote(book, i)): _entry(f"page {i}")
           for i in range(3)}
    old["f" * 64] = _entry("a page this PDF no longer has")
    _old_cache(path, old)

    st = vtb.rekey_cache(path, book)
    assert (st["pages"], st["found"], st["refiled"], st["missing"],
            st["left"]) == (3, 3, 3, 0, 1)
    said = capsys.readouterr().out
    assert "3 refiled" in said and "1 reading(s) match no page" in said

    stored = _stored(path)
    assert stored["by_bytes"] == ["f" * 64]         # kept, and still marked
    assert sorted(json.dumps(e, sort_keys=True) for e in stored["pages"].values()) \
        == sorted(json.dumps(e, sort_keys=True) for e in old.values())

    # a second pass has nothing to do, and a build finds every page
    before = open(path, "rb").read()
    assert vtb.rekey_cache(path, book)["refiled"] == 0
    assert open(path, "rb").read() == before


def test_rekey_reports_a_cache_that_is_already_over(book, tmp_path, capsys):
    path = str(tmp_path / "c.ocr.gz")
    _old_cache(path, {_sha(_as_every_earlier_build_wrote(book, 0)):
                      _entry("page 0")})
    vtb.rekey_cache(path, book)
    capsys.readouterr()
    st = vtb.rekey_cache(path, book)
    assert st["refiled"] == 0 and st["left"] == 0
    assert "already filed" in capsys.readouterr().out


def test_the_rekey_verb(book, tmp_path, capsys):
    from codicology import cli
    path = str(tmp_path / "c.ocr.gz")
    _old_cache(path, {_sha(_as_every_earlier_build_wrote(book, 1)):
                      _entry("page 1")})
    assert cli.main(["rekey", path, book]) == 0
    assert "1 refiled" in capsys.readouterr().out
    assert _stored(path)["keying"] == vtb.PAGE_KEYING


def test_two_pages_that_show_the_same_thing_share_one_reading(tmp_path):
    """As they did under their bytes: 305 pages of one shelf book are held
    by 293 readings."""
    pdf = make_book(tmp_path / "twins.pdf", [typeset_page(), typeset_page()])
    path = str(tmp_path / "c.ocr.gz")
    _old_cache(path, {_sha(_as_every_earlier_build_wrote(pdf, 0)):
                      _entry("both pages")})
    d = tmp_path / "pages"; d.mkdir()
    first, second = vtb.load_pages_from_pdf(pdf, str(d))
    cache = vtb.OCRCache(path, "surya", ["en"])
    assert cache.get(first)[0].html == cache.get(second)[0].html \
        == "<p>both pages</p>"
    assert cache.refiled == 1 and len(cache.entries) == 1
