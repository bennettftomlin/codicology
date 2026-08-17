"""Running heads survive into the cache and feed the folio audit for free.

The audit's raw material is the head text the main OCR pass already recognised
at full resolution with the whole page for context — the best reading the
pipeline will ever have. Discarding it and re-deriving it later from a
downscaled second layout pass was measured misreading small type ("18" for 16)
and cost hours per book. What is pinned here: furniture rides the cache flagged,
never reaches a page's body or the turn detector, and an old cache — written
before heads were kept — is told apart from a book that genuinely has none.
"""
import pytest


def item(vtb, html="", furniture=False, caption=False):
    return vtb.PageItem(html=html, is_caption=caption, is_furniture=furniture)


@pytest.fixture
def cache(vtb, tmp_path):
    return vtb.OCRCache(str(tmp_path / "c.gz"), "surya", ["en"])


@pytest.fixture
def page(tmp_path):
    p = tmp_path / "page_0000.jpg"
    p.write_bytes(b"not really a jpeg, but bytes to hash")
    return str(p)


def test_furniture_flag_survives_the_cache_round_trip(vtb, cache, page):
    cache.put(page, [item(vtb, "<p>60 - Citadel of Sin</p>", furniture=True),
                     item(vtb, "<p>body text</p>")])
    back = cache.get(page)
    assert [it.is_furniture for it in back] == [True, False]
    assert cache.knows_furniture(page)


def test_an_old_cache_entry_is_not_mistaken_for_a_headless_page(vtb, cache, page):
    """
    The distinction the version marker exists for: a page whose entry predates
    furniture-keeping has UNKNOWN heads, not absent ones. Treating the two the
    same would audit an old cache as a book with no folios anywhere and report
    nothing wrong — silence indistinguishable from a clean bill.

    The marker now settles that by re-reading rather than by flagging: an
    entry below the current version is not served at all, so the page comes
    back from the engine with its heads — and its labels — present.
    """
    cache.entries[cache._key(page)] = [{"html": "<p>body</p>", "fig": None,
                                        "cap": False}]          # v1 shape: a bare list
    assert cache.get(page) is None                              # re-read, not served
    assert not cache.knows_furniture(page)                      # and not trusted for heads

    cache.put(page, [item(vtb, "<p>body</p>")])
    assert cache.knows_furniture(page)                          # absence now means absence
    assert cache.get(page) is not None                          # and it is a hit again


def test_the_escape_hatch_still_reads_an_old_entry(vtb, tmp_path, page):
    """--keep-stale-cache trades the labels for the hours: the old reading
    is served, and still carries no claim about heads."""
    cache = vtb.OCRCache(str(tmp_path / "c.gz"), "surya", ["en"],
                         serve_stale=True)
    cache.entries[cache._key(page)] = [{"html": "<p>body</p>", "fig": None,
                                        "cap": False}]
    assert cache.get(page) is not None
    assert not cache.knows_furniture(page)


def test_folios_come_straight_from_furniture_text(vtb):
    folios = vtb.folios_from_furniture([
        ["60 • Citadel of Sin"],          # verso head
        ["Chapter 7 • 61"],               # recto head
        ["55"],                           # a chapter opening's bare-number foot
        ["Hell at midday"],               # furniture with no number
        [],                               # no furniture at all
    ])
    assert [(f.number, f.confident) for f in folios] == [
        (60, True), (61, True), (55, True), (None, False), (None, False)]


def test_bare_numeral_is_accepted_only_because_layout_called_it_furniture(vtb):
    # The strict parser still refuses it; the furniture path may accept it.
    assert vtb.parse_folio("55") is None
    assert vtb.folios_from_furniture([["55"]])[0].number == 55
    # and a year-sized numeral stays refused even as furniture
    assert vtb.folios_from_furniture([["1893"]])[0].number is None


def test_furniture_never_reaches_page_text_used_for_turn_detection(vtb, cache, page):
    """
    'EAGLE FORGOTTEN' repeats on every verso of that book. Let it into the text
    used for containment matching and any short page starts sharing "content"
    with every neighbour — the turn detector would condemn real pages on the
    strength of the paper's own furniture.
    """
    cache.put(page, [item(vtb, "<p>EAGLE FORGOTTEN</p>", furniture=True),
                     item(vtb, "<p>the actual words of the page</p>")])
    cache.save()          # page_texts opens the cache from its path, not this object

    class NoBackend:
        name, batch_size, langs = "surya", 4, ["en"]

        def run_items(self, images):
            raise AssertionError("cached page must not be re-read")

    texts = vtb.page_texts([page], NoBackend(), cache.path)
    assert "EAGLE FORGOTTEN" not in texts[0]
    assert "actual words" in texts[0]


def test_legacy_reread_upgrades_the_cache_entry(vtb, cache, page):
    """
    The re-read is paid once. Afterward the entry is current: known, with the
    head recorded — or recorded absent, which is then trusted — and the page
    is served from cache ever after.
    """
    cache.entries[cache._key(page)] = [{"html": "<p>body</p>", "fig": None,
                                        "cap": False}]
    assert not cache.knows_furniture(page)
    assert cache.get(page) is None, "the stale entry sends the page back to OCR"
    cache.put(page, [vtb.PageItem(html="<p>body</p>"),
                     vtb.PageItem(html="<p>Chapter 11 • 91</p>",
                                  is_furniture=True)])
    assert cache.knows_furniture(page)
    back = cache.get(page)
    assert any(it.is_furniture and "91" in it.html for it in back)


def test_an_empty_read_of_an_inked_page_is_retried_and_never_cached(vtb, tmp_path):
    """
    The failure that blanked 191 of 197 pages: the inference server degraded
    (1,536 errors in one run), every page came back empty, and the empties
    were CACHED — poisoning all future builds. The guard: an empty result
    from a page that visibly has ink is retried once, solo; and nothing empty
    reaches the cache unless the page truly is blank.

    The first version of this test exercised a code path the fix was not in
    and asserted nearly nothing. This one drives the helper itself.
    """
    import numpy as np
    from PIL import Image

    inked = np.full((1300, 900), 236, dtype="uint8")
    for k in range(16):
        inked[150 + k*60:176 + k*60, 90:770] = 35
    inked_img = Image.fromarray(inked).convert("RGB")
    blank_img = Image.fromarray(np.full((1300, 900), 236, dtype="uint8")).convert("RGB")

    p_inked, p_blank = str(tmp_path / "a.jpg"), str(tmp_path / "b.jpg")
    inked_img.save(p_inked); blank_img.save(p_blank)

    calls = {"batch": 0, "solo": 0}

    class Flaky:
        name, batch_size, langs = "surya", 4, ["en"]

        def run_items(self, images):
            if len(images) > 1:
                calls["batch"] += 1
                return [[] for _ in images]          # server down: all empty
            calls["solo"] += 1
            return [[vtb.PageItem(html="<p>recovered</p>")]]

    cache = vtb.OCRCache(str(tmp_path / "c.gz"), "surya", ["en"])
    out = vtb._read_pages_resiliently([p_inked, p_blank],
                                      [inked_img, blank_img], Flaky(), cache)

    # the inked page was retried solo and recovered
    assert calls["solo"] == 1
    assert any("recovered" in it.html for it in out[p_inked])
    # the recovered result is cached; the blank page is cached as blank
    assert cache.get(p_inked) is not None
    assert cache.get(p_blank) is not None


def test_a_still_empty_inked_page_is_not_poisoned_into_the_cache(vtb, tmp_path):
    import numpy as np
    from PIL import Image
    inked = np.full((1300, 900), 236, dtype="uint8")
    for k in range(16):
        inked[150 + k*60:176 + k*60, 90:770] = 35
    img = Image.fromarray(inked).convert("RGB")
    p = str(tmp_path / "a.jpg"); img.save(p)

    class Dead:
        name, batch_size, langs = "surya", 4, ["en"]
        def run_items(self, images):
            return [[] for _ in images]              # down for good

    cache = vtb.OCRCache(str(tmp_path / "c.gz"), "surya", ["en"])
    out = vtb._read_pages_resiliently([p], [img], Dead(), cache)
    assert out[p] == []                              # honest empty result
    assert cache.get(p) is None                      # but never cached


def test_decorated_folios_are_undressed_in_furniture(vtb):
    # A 1964 book parenthesizes its folios — "(56)" — and the scan sometimes
    # double-strikes the paren: "((54". The layout pass already called these
    # furniture, so stripping decoration cannot promote prose to a folio.
    folios = vtb.folios_from_furniture([
        ["THE INVISIBLE GOVERNMENT", "(56)"],
        ["THE INVISIBLE GOVERNMENT", "((54"],
        ["THE INVISIBLE GOVERNMENT"],          # head only, no number
    ])
    assert [(f.number, f.confident) for f in folios] == [
        (56, True), (54, True), (None, False)]


def test_labels_round_trip_and_old_entries_stay_readable(vtb, tmp_path):
    """The layout model's block labels survive the cache, and an entry
    written before labels were kept reads back with label None — which every
    consumer must treat as "unknown", never as "Text"."""
    from PIL import Image
    page = str(tmp_path / "p.png")
    Image.new("RGB", (60, 90), "white").save(page)
    path = str(tmp_path / "c.ocr.gz")
    c = vtb.OCRCache(path, "surya", ["en"])
    c.put(page, [vtb.PageItem(html="<p>1. src</p>", label="Footnote"),
                 vtb.PageItem(html="<p>body</p>")])
    c.save()
    c2 = vtb.OCRCache(path, "surya", ["en"])
    got = c2.get(page)
    assert got[0].label == "Footnote"
    assert got[1].label is None
    # a pre-label entry: strip the key as an old writer would never have set it
    key = c2._key(page)
    entry = c2.entries[key]
    for it in (entry["items"] if isinstance(entry, dict) else entry):
        it.pop("lab", None)
    old = c2.get(page)
    assert old[0].label is None and old[0].html == "<p>1. src</p>"
