"""Pages that read empty every time, named so a cache-only build can go on.

The cache never keeps an empty reading, because an empty can be a server's
failure; so a cover, a scanner's stamp or a back board that reads empty is
never in the cache, and --cache-only refused seven books of the shelf for
one or two such pages each."""
from codicology import switches
from codicology.cache import OCRCache


def test_a_named_page_the_cache_never_read_is_taken_as_empty(tmp_path, monkeypatch):
    cache = OCRCache(str(tmp_path / "c.ocr.gz"), "surya", ["en"])
    page = tmp_path / "page_0005.png"
    page.write_bytes(b"\x89PNG not really")
    monkeypatch.setattr(switches, "TAKEN_AS_EMPTY", [])
    monkeypatch.setattr(switches, "READ_AS_EMPTY", frozenset())
    assert cache.get(str(page)) is None
    monkeypatch.setattr(switches, "READ_AS_EMPTY", frozenset({"page_0005"}))
    assert cache.get(str(page)) == []
    assert switches.TAKEN_AS_EMPTY == ["page_0005"]


def test_an_unnamed_page_is_still_missing(tmp_path, monkeypatch):
    cache = OCRCache(str(tmp_path / "c.ocr.gz"), "surya", ["en"])
    page = tmp_path / "page_0006.png"
    page.write_bytes(b"\x89PNG not really")
    monkeypatch.setattr(switches, "READ_AS_EMPTY", frozenset({"page_0005"}))
    monkeypatch.setattr(switches, "TAKEN_AS_EMPTY", [])
    assert cache.get(str(page)) is None and switches.TAKEN_AS_EMPTY == []


def test_a_page_taken_as_empty_knows_it_has_no_running_head(tmp_path, monkeypatch):
    """Otherwise the folio pass takes it for a page cached before heads were
    kept and sends it to be read."""
    cache = OCRCache(str(tmp_path / "c.ocr.gz"), "surya", ["en"])
    page = tmp_path / "page_0005.png"
    page.write_bytes(b"\x89PNG not really")
    monkeypatch.setattr(switches, "READ_AS_EMPTY", frozenset())
    assert not cache.knows_furniture(str(page))
    monkeypatch.setattr(switches, "READ_AS_EMPTY", frozenset({"page_0005"}))
    assert cache.knows_furniture(str(page))
