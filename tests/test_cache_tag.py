"""The language list is not part of a reading it does not steer.

The cache tag was "engine|languages", and a cache whose tag does not match
loads nothing and is overwritten by the next save. Surya at this pin is
called with the page images alone, so "en,de" typed for one book — and the
Calibre plugin keeps that field as a default for every later book — threw
away an hour's reading to change one line of metadata.
"""
import gzip
import json

from codicology import pipeline as vtb


def _png(tmp_path, name=b"a"):
    p = tmp_path / ("p%s.png" % name.decode())
    p.write_bytes(b"\x89PNG" + name)
    return str(p)


def _stored(path):
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def _write(path, tag, pages):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        json.dump({"tag": tag, "pages": pages}, fh)


def _entry(text):
    return {"v": vtb.CACHE_VERSION, "items": [{"html": f"<p>{text}</p>"}]}


def test_another_language_list_reads_the_same_surya_cache(tmp_path):
    path, page = str(tmp_path / "c.ocr.gz"), _png(tmp_path)
    first = vtb.OCRCache(path, "surya", ["en"])
    first.put(page, [vtb.PageItem(html="<p>read once</p>")])
    first.save()

    again = vtb.OCRCache(path, "surya", ["en", "de"])
    got = again.get(page)
    assert got is not None and got[0].html == "<p>read once</p>"


def test_a_cache_written_with_the_old_tag_still_loads(tmp_path):
    """Every cache on the shelf says "surya|en"."""
    path, page = str(tmp_path / "c.ocr.gz"), _png(tmp_path)
    key = vtb.OCRCache(path, "surya", ["en"])._key(page)
    for old in ("surya|en", "surya|en,de", "surya|"):
        _write(path, old, {key: _entry("from the shelf")})
        c = vtb.OCRCache(path, "surya", ["fr"])
        assert c.get(page)[0].html == "<p>from the shelf</p>", old


def test_saving_does_not_lose_the_pages_already_there(tmp_path):
    """The loss itself: a second language list, one new page, a save."""
    path = str(tmp_path / "c.ocr.gz")
    old, new = _png(tmp_path, b"a"), _png(tmp_path, b"b")
    key = vtb.OCRCache(path, "surya", ["en"])._key(old)
    _write(path, "surya|en", {key: _entry("an hour's reading")})

    c = vtb.OCRCache(path, "surya", ["en", "de"])
    c.put(new, [vtb.PageItem(html="<p>one more</p>")])
    c.save()

    stored = _stored(path)
    assert stored["tag"] == "surya"
    assert len(stored["pages"]) == 2 and key in stored["pages"]
    assert vtb.OCRCache(path, "surya", ["en"]).get(old) is not None


def test_another_engines_cache_is_not_served(tmp_path):
    path, page = str(tmp_path / "c.ocr.gz"), _png(tmp_path)
    key = vtb.OCRCache(path, "surya", ["en"])._key(page)
    for other in ("tesseract|en", "suryax|en", "", None):
        _write(path, other, {key: _entry("someone else's")})
        assert vtb.OCRCache(path, "surya", ["en"]).get(page) is None, other


def test_an_engine_the_languages_steer_keeps_them_in_its_tag(tmp_path):
    """Only an engine that declares itself blind to them drops them."""
    path, page = str(tmp_path / "c.ocr.gz"), _png(tmp_path)
    c = vtb.OCRCache(path, "an-engine-that-takes-languages", ["en"])
    c.put(page, [vtb.PageItem(html="<p>in English</p>")])
    c.save()
    assert _stored(path)["tag"] == "an-engine-that-takes-languages|en"
    assert vtb.OCRCache(path, "an-engine-that-takes-languages",
                        ["de"]).get(page) is None
    assert vtb.OCRCache(path, "an-engine-that-takes-languages",
                        ["en"]).get(page) is not None
