"""A cache entry too old to carry what the pipeline now needs is not a hit.

Labels arrive fused with recognition, so a page read before the pipeline
recorded them builds silently without the footnote rules, the reattached
markers, and the heading hierarchy. Such a page is read again. What makes
that converge — rather than re-read the same page every build forever — is
the version stamp: a page whose ink genuinely yields no labelled block is
stamped current all the same, and is a hit ever after.
"""
from codicology import pipeline as vtb


def _png(tmp_path, name, byte):
    p = tmp_path / name
    p.write_bytes(b"\x89PNG" + byte)
    return str(p)


def _cache(tmp_path, **kw):
    return vtb.OCRCache(str(tmp_path / "c.ocr.gz"), "surya", ["en"], **kw)


def test_stale_entry_is_reread(vtb, tmp_path):
    page = _png(tmp_path, "p.png", b"a")
    c = _cache(tmp_path)
    c.entries[c._key(page)] = {"v": 3, "items": [{"html": "<p>old</p>"}]}
    assert c.get(page) is None, "a pre-label read must not be served"
    assert c.stale == 1


def test_a_reread_page_is_never_reread_again(vtb, tmp_path):
    """Even when the page's ink yields no label at all — otherwise a plain
    page of prose would be re-read on every build for the rest of time."""
    page = _png(tmp_path, "p.png", b"a")
    c = _cache(tmp_path)
    c.put(page, [vtb.PageItem(html="<p>read fresh</p>", label=None)])
    got = c.get(page)
    assert got is not None and got[0].html == "<p>read fresh</p>"
    assert c.stale == 0


def test_current_entries_are_served(vtb, tmp_path):
    page = _png(tmp_path, "p.png", b"a")
    c = _cache(tmp_path)
    c.entries[c._key(page)] = {"v": vtb.CACHE_VERSION,
                               "items": [{"html": "<p>x</p>", "lab": "Text"}]}
    got = c.get(page)
    assert got and got[0].label == "Text"


def test_the_ancient_listform_entry_is_a_miss(vtb, tmp_path):
    """Pre-v3 entries are bare lists — older still, and equally not served."""
    page = _png(tmp_path, "p.png", b"a")
    c = _cache(tmp_path)
    c.entries[c._key(page)] = [{"html": "<p>ancient</p>"}]
    assert c.get(page) is None


def test_the_escape_hatch_serves_stale_pages(vtb, tmp_path):
    """--keep-stale-cache: an unlabeled build beats an hour of waiting,
    when the person asking knows that is the trade."""
    page = _png(tmp_path, "p.png", b"a")
    c = _cache(tmp_path, serve_stale=True)
    c.entries[c._key(page)] = {"v": 3, "items": [{"html": "<p>old</p>"}]}
    got = c.get(page)
    assert got and got[0].html == "<p>old</p>"
    assert got[0].label is None


def test_a_missing_page_is_still_just_a_miss(vtb, tmp_path):
    c = _cache(tmp_path)
    assert c.get(_png(tmp_path, "p.png", b"a")) is None
    assert c.stale == 0, "never read is not the same as read too long ago"
