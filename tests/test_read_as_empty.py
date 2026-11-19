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


def test_a_named_page_with_ink_is_shown_as_its_own_picture(tmp_path, monkeypatch):
    """A cover or a scanner's stamp: read live it ships its pictures, so
    taken unread it is the one picture of the page; blank paper is empty."""
    from PIL import Image, ImageDraw
    cache = OCRCache(str(tmp_path / "c.ocr.gz"), "surya", ["en"])
    inked, blank = tmp_path / "page_0001.png", tmp_path / "page_0002.png"
    im = Image.new("L", (400, 600), 255)
    d = ImageDraw.Draw(im)
    for y in range(150, 450, 12):
        d.line([(80, y), (320, y)], fill=0, width=2)
    im.save(inked)
    Image.new("L", (400, 600), 255).save(blank)
    monkeypatch.setattr(switches, "READ_AS_EMPTY", frozenset({"page_0001", "page_0002"}))
    monkeypatch.setattr(switches, "TAKEN_AS_EMPTY", [])
    got = cache.get(str(inked))
    assert len(got) == 1 and got[0].whole_page and got[0].figure is not None
    assert cache.get(str(blank)) == []


def test_pages_dropped_past_every_decision_do_not_refuse_them(tmp_path):
    """The Knights of Industry: its back board, the last leaf, dropped, and
    all seven of its corrections refused for the one page fewer."""
    import json
    from codicology.decisions import apply_reviewer_decisions
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"page_files": 3, "tokens": 2, "decisions": [
        {"page": 0, "old": "navy", "new": "navvy", "before": ["a"], "after": ["a"]}]}))
    bodies = ["<p>a navy a</p>", "<p>b</p>", ""]
    got = apply_reviewer_decisions(bodies, str(path), kept_pages=2, dropped={2})
    assert got["applied"] == 1 and "navvy" in bodies[0]
    bodies = ["<p>a navy a</p>", "", "<p>c</p>"]
    got = apply_reviewer_decisions(bodies, str(path), kept_pages=2, dropped={0})
    assert got["applied"] == 0      # a page dropped before it: refused, as before
