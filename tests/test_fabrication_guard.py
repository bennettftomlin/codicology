"""The guard against invented pages.

A generative recogniser, handed a page it cannot read, does not always return
nothing — sometimes it writes. Measured across 47 reads of known-blank,
known-sparse and known-good pages: fabrications either loop (every six-word
window repeating) or carry more text than the page has ink to have held. The
thresholds here come from that measurement, and the cases below are the ones
that actually occurred, including the two that defeat the obvious guards —
a fabrication that repeated verbatim across three reads, and a real page of
six words whose ink is indistinguishable from a blank one.
"""
import numpy as np
import pytest
from PIL import Image


def _page(ink_rows=0, size=(600, 900)):
    """A page image with a controllable amount of ink on it."""
    img = Image.new("RGB", size, "white")
    px = img.load()
    for r in range(ink_rows):
        y = 120 + r * 20
        for x in range(80, 520):
            for dy in range(10):
                if y + dy < size[1]:
                    px[x, y + dy] = (25, 25, 25)
    return img


class FakeBackend:
    """Returns scripted readings, one per call."""
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def __init__(self, vtb, scripts):
        self.vtb, self.scripts, self.calls = vtb, list(scripts), 0

    def run_items(self, images):
        out = []
        for _ in images:
            text = self.scripts[min(self.calls, len(self.scripts) - 1)]
            self.calls += 1
            out.append([self.vtb.PageItem(html=f"<p>{text}</p>")] if text else [])
        return out


def test_a_looping_reading_is_refused_even_when_it_repeats_verbatim(vtb, tmp_path):
    """
    The case that defeats re-reading: a blank Citadel verso produced the same
    587 characters on all three reads. Agreement says "stable", so only the
    looping itself can condemn it.
    """
    loop = "The design step is then to determine the support strength " * 12
    p = str(tmp_path / "blank.png"); _page(ink_rows=0).save(p)
    be = FakeBackend(vtb, [loop])            # every read identical
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [_page(ink_rows=0)], be, None)
    assert got[p] == []
    assert vtb._read_pages_resiliently.refused == [p]


def test_a_page_that_loops_differently_on_each_read_is_refused(vtb, tmp_path):
    """
    The other observed shape (blank_p45): the first read loops one way, the
    retry loops a different way. A different loop is still a loop — the
    rescue requires the second reading to be CLEAN, not merely different.
    """
    p = str(tmp_path / "blank.png"); _page(ink_rows=0).save(p)
    be = FakeBackend(vtb, [
        "a very different set of paper was used in the study " * 14,
        "the conversion of a liquid into a solid is called conversion " * 14,
    ])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [_page(ink_rows=0)], be, None)
    assert got[p] == []
    assert vtb._read_pages_resiliently.refused == [p]


def test_a_genuinely_sparse_page_is_kept(vtb, tmp_path):
    """
    "BOOK ONE / THE STRUGGLE" is six words on a nearly empty page — the exact
    shape an ink threshold would condemn. Every such page measured read back
    identically, and none may be thrown away.
    """
    img = _page(ink_rows=1)
    p = str(tmp_path / "parttitle.png"); img.save(p)
    be = FakeBackend(vtb, ["BOOK ONE THE STRUGGLE"])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert vtb._strip_tags(got[p][0].html) == "BOOK ONE THE STRUGGLE"
    assert vtb._read_pages_resiliently.refused == []


def test_a_dense_but_legitimate_page_is_kept(vtb, tmp_path):
    """A chemical index repeats tokens without looping; it must survive."""
    # entries that differ only by their numbers — the shape that looks like a
    # loop the moment digits are dropped from the tokens
    index = " ".join(f"{n}-MeO-DMT see under 4-MeO-MIPT #{n} Page {300 + n}"
                     for n in range(40))
    img = _page(ink_rows=30)
    p = str(tmp_path / "index.png"); img.save(p)
    be = FakeBackend(vtb, [index])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_a_slow_but_honest_second_look_can_rescue_a_page(vtb, tmp_path):
    """First read loops, second reads cleanly — keep the good one."""
    img = _page(ink_rows=30)
    p = str(tmp_path / "page.png"); img.save(p)
    be = FakeBackend(vtb, ["the same words over and over " * 20,
                           "Altgeld returned to Chicago that spring and began "
                           "at once to prepare his answer to the newspapers."])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert "Altgeld" in vtb._strip_tags(got[p][0].html)
    assert vtb._read_pages_resiliently.refused == []


def test_ink_ignores_the_photographs_dark_border(vtb):
    """
    Counting the desk and the book's shadow put a blank leaf at 0.11, close
    to a printed page. The measurement must look at the paper only.
    """
    img = Image.new("RGB", (600, 900), "white")
    px = img.load()
    for x in range(600):                       # heavy dark border
        for y in list(range(40)) + list(range(860, 900)):
            px[x, y] = (0, 0, 0)
    assert vtb._page_ink(img) < 0.02


def test_a_short_invention_on_a_near_blank_page_is_a_known_blind_spot(vtb, tmp_path):
    """
    Recorded rather than hidden: a few words invented on a nearly empty page
    are not distinguishable from a few words genuinely printed there. Real
    pages measured at 0.003 ink carried 21 to 51 characters ("BOOK ONE / THE
    STRUGGLE", a dedication), and a fabrication of that size sits inside the
    same range. The guard catches the damaging cases — the 587-, 1,274- and
    17,442-character inventions — and lets this one through.
    """
    img = _page(ink_rows=0)
    p = str(tmp_path / "blank.png"); img.save(p)
    be = FakeBackend(vtb, ["G. B. DELAMORE March 26, 1941"])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p], "short text on a blank page is currently kept"
    assert vtb._read_pages_resiliently.refused == []


@pytest.mark.parametrize("chars,ink_rows,kept,note", [
    (230, 1, True,  "a chapter's last two lines on a nearly blank page"),
    (254, 1, True,  "another carry-over stub"),
    (131, 1, True,  "a contents fragment"),
    (80,  1, True,  "a title page — purged by the first version of the guard"),
    (2666, 13, True, "TiHKAL's epilogue: dense text, faint page"),
    (2900, 2, True,  "a born-digital page: thin type, very low ink — the case "
                     "that made 276 false positives of an ink-rate rule"),
])
def test_real_pages_survive_regardless_of_their_ink(vtb, tmp_path, chars,
                                                    ink_rows, kept, note):
    """
    Every one of these is real text, and earlier versions of the guard threw
    some of them away — two books' title pages under a "text on a blank page"
    rule, and most of a born-digital book under a characters-per-ink rule.
    Ink takes no part in condemnation: only looping does.
    """
    img = _page(ink_rows=ink_rows)
    p = str(tmp_path / "sparse.png"); img.save(p)
    # varied prose, not a repeated sentence — a repeated fixture is itself a
    # loop, and the loop rule (correctly) fires on it before the rate rule
    # is ever consulted
    text = " ".join(f"word{i} came before word{i + 1} in sentence {i // 9}"
                    for i in range(200))[:chars]
    be = FakeBackend(vtb, [text])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert bool(got[p]) == kept, note


def _with_layer(tmp_path, img, layer_text):
    p = str(tmp_path / "page.png"); img.save(p)
    with open(p + ".layer.txt", "w") as fh:
        fh.write(layer_text)
    return p


def test_a_reading_contradicting_the_pages_own_layer_is_refused(vtb, tmp_path):
    """
    The two escapes that beat the loop rule: a fabricated project-management
    table on a page about medical students, and ninety-one invented words on
    a part-title whose layer says only "Epilogue". Neither loops; both share
    almost nothing with what the page says about itself.
    """
    img = _page(ink_rows=20)
    layer = ("Medical students also reported systematic sharing between older "
             "and younger students as part of a larger structure of mentoring "
             "across levels one student reported we have a different class "
             "each year so a student from the second year will choose someone "
             "from the first year a freshman who is just coming in to pass "
             "on materials tips exams and advice about professors courses "
             "readings libraries copies practices traditions obligations "
             "networks favors relationships responsibilities and expectations")
    fab = " ".join(f"Project {w} row{i} status{i} lead{i} date{i}"
                   for i, w in enumerate(
                       "Details Status Timeline Resources Progress Completion "
                       "Financials Alpha Beta Gamma Delta".split()))
    p = _with_layer(tmp_path, img, layer)
    be = FakeBackend(vtb, [fab, fab])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] == []
    assert vtb._read_pages_resiliently.refused == [p]


def test_a_clean_retry_rescues_a_layer_contradicted_page(vtb, tmp_path):
    """Shadow Libraries p259 read correctly in isolation — the retry wins."""
    img = _page(ink_rows=20)
    layer = ("Medical students also reported systematic sharing between older "
             "and younger students as part of a larger structure of mentoring "
             "across levels one student reported we have a different class "
             "each year so a student from the second year will choose someone "
             "from the first year a freshman who is just coming in to pass "
             "on materials tips exams and advice about professors courses "
             "readings libraries copies practices traditions obligations "
             "networks favors relationships responsibilities and expectations")
    fab = " ".join(f"Project {w} row{i} status{i} lead{i} date{i}"
                   for i, w in enumerate(
                       "Details Status Timeline Resources Progress Completion "
                       "Financials Alpha Beta Gamma Delta".split()))
    real = ("Medical students also reported systematic sharing between older "
            "and younger students, as part of a larger structure of mentoring "
            "across levels. One student reported that a student from the "
            "second year will choose a student from the first year to pass "
            "on materials, tips and exams for every course they had shared.")
    p = _with_layer(tmp_path, img, layer)
    be = FakeBackend(vtb, [fab, real])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert "Medical students" in vtb._strip_tags(got[p][0].html)
    assert vtb._read_pages_resiliently.refused == []


def test_a_garbled_scanner_layer_does_not_condemn_a_good_reading(vtb, tmp_path):
    """
    A Google-scan layer mangles edges ("blufif", "haubehg") but shares its
    prose vocabulary. Our better reading must not be refused for the layer's
    own errors.
    """
    img = _page(ink_rows=20)
    real = ("They had scattered and destroyed the Eries whose home was on the "
            "south shore of the lake that bears their name and the shrewd "
            "Frenchman assured him that he had missed his way entirely there")
    garbled = real.replace("shrewd", "shrew").replace("Eries", "fries") \
                  .replace("destroyed", "stroyed").replace("lake", "hke")
    p = _with_layer(tmp_path, img, garbled)
    be = FakeBackend(vtb, [real])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_short_readings_are_not_judged_against_the_layer(vtb, tmp_path):
    """ "Epilogue" against a layer of "384 TIHKAL Epilogue" must pass —
    six words have no vocabulary to compare."""
    img = _page(ink_rows=1)
    p = _with_layer(tmp_path, img, "384 TIHKAL — The Continuation Epilogue")
    be = FakeBackend(vtb, ["Epilogue"])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_no_layer_sidecar_means_no_judgement(vtb, tmp_path):
    """Photograph sources have no layer; the witness rule must stay silent."""
    img = _page(ink_rows=20)
    p = str(tmp_path / "photo.png"); img.save(p)
    text = " ".join(f"word{i} follows word{i+1} in line {i//8}"
                    for i in range(80))
    be = FakeBackend(vtb, [text])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_classical_silence_plus_unstable_reads_refuses_a_photo_page(vtb, tmp_path, monkeypatch):
    """
    The photo-source blind spot: no layer to consult, the model reads
    paragraphs, a classical OCR reads nothing, and the two model readings
    tell different stories. That combination is fabrication.
    """
    img = _page(ink_rows=0)
    p = str(tmp_path / "photo.png"); img.save(p)
    monkeypatch.setattr(vtb, "_classical_word_count", lambda _: 0)
    a = " ".join(f"alpha{i} beta{i} gamma{i} delta{i}" for i in range(30))
    b = " ".join(f"omega{i} sigma{i} theta{i} lambda{i}" for i in range(30))
    be = FakeBackend(vtb, [a, b])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] == []
    assert vtb._read_pages_resiliently.refused == [p]


def test_classical_silence_with_stable_reads_keeps_the_page(vtb, tmp_path, monkeypatch):
    """
    Tesseract also goes silent on hard-but-real pages. If the model tells
    the same story twice, its reading stands — silence alone cannot convict.
    """
    img = _page(ink_rows=20)
    p = str(tmp_path / "photo.png"); img.save(p)
    monkeypatch.setattr(vtb, "_classical_word_count", lambda _: 0)
    text = " ".join(f"word{i} follows word{i+1} in line {i//8}" for i in range(80))
    be = FakeBackend(vtb, [text, text])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_classical_witness_not_consulted_when_a_layer_exists(vtb, tmp_path, monkeypatch):
    """Scanned sources have the better witness; the classical one stays out."""
    img = _page(ink_rows=20)
    layer = " ".join(f"word{i} follows word{i+1} in line {i//8}" for i in range(80))
    p = _with_layer(tmp_path, img, layer)
    def boom(_): raise AssertionError("classical witness consulted despite layer")
    monkeypatch.setattr(vtb, "_classical_word_count", boom)
    be = FakeBackend(vtb, [layer])
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p]


def test_missing_tesseract_binary_means_no_judgement(vtb, tmp_path, monkeypatch):
    img = _page(ink_rows=0)
    p = str(tmp_path / "photo.png"); img.save(p)
    monkeypatch.setattr(vtb, "_classical_word_count", lambda _: None)
    text = " ".join(f"word{i} and word{i+1} met in line {i//8}" for i in range(60))
    be = FakeBackend(vtb, [text])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_a_broken_three_word_layer_cannot_condemn_a_rich_real_reading(vtb, tmp_path, monkeypatch):
    """
    TiHKAL p353: a genuine 261-word aside over a layer that says only "how
    many zeros" — the scanner's OCR failed, not ours. The sparse layer must
    step aside; the classical witness (which reads the full page) decides
    nothing is wrong.
    """
    img = _page(ink_rows=20)
    p = _with_layer(tmp_path, img, "how many zeros")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda _: 250)
    text = " ".join(f"factoid{i} zillion{i} trillion{i} degrees{i}"
                    for i in range(30))
    be = FakeBackend(vtb, [text])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] and vtb._read_pages_resiliently.refused == []


def test_a_sparse_layer_page_falls_through_to_the_classical_witness(vtb, tmp_path, monkeypatch):
    """
    TiHKAL p415 again: a five-word layer ("384 TIHKAL Epilogue") cannot
    testify about vocabulary, but tesseract reads four words where the model
    reads ninety-one that change on every look. The fall-through refuses it.
    """
    img = _page(ink_rows=1)
    p = _with_layer(tmp_path, img, "384 TIHKAL — The Continuation Epilogue")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda _: 4)
    a = " ".join(f"continuation{i} paper{i} dense{i} list{i}" for i in range(25))
    b = " ".join(f"different{i} story{i} entirely{i} here{i}" for i in range(25))
    be = FakeBackend(vtb, [a, b])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] == []
    assert vtb._read_pages_resiliently.refused == [p]


def test_a_refused_native_page_is_restored_from_its_own_text(vtb, tmp_path):
    """
    Four textbook problem-set pages made the model loop on every read.
    On a scan they would stay honestly unread — but these pages carry the
    publisher's own text, and restoring from it is not a guess. Watermark
    and folio lines are furniture, not words, and stay out.
    """
    img = _page(ink_rows=20)
    p = str(tmp_path / "page_0000.jpg"); img.save(p)
    with open(p + ".layer.txt", "w") as fh:
        fh.write("Saylor URL: http://www.saylor.org/books Saylor.org\n99\n"
                 "1. Approximately what percentage of growth was due to "
                 "increases in quantities of factors of production?")
    open(p + ".native", "w").close()
    loop = "the elasticity of the demand of the elasticity " * 14
    be = FakeBackend(vtb, [loop, loop])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    text = vtb._strip_tags(got[p][0].html)
    assert "percentage of growth" in text
    assert "Saylor" not in text and not text.startswith("99")
    assert vtb._read_pages_resiliently.refused == []


def test_a_refused_scan_page_still_stays_unread(vtb, tmp_path):
    """No native authority, no restoration — the old honesty holds."""
    img = _page(ink_rows=0)
    p = str(tmp_path / "page_0000.jpg"); img.save(p)
    loop = "the county of the contrary is as follows " * 14
    be = FakeBackend(vtb, [loop, loop])
    vtb._read_pages_resiliently.refused = []
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] == []
    assert vtb._read_pages_resiliently.refused == [p]


def test_an_empty_read_of_a_native_page_restores_its_text(vtb, tmp_path):
    """p620: the dishwasher problem read empty twice and shipped as a hole.
    A native page's words are at hand; an inked page that reads empty twice
    gets them."""
    img = _page(ink_rows=6)
    p = str(tmp_path / "page_0000.jpg"); img.save(p)
    with open(p + ".layer.txt", "w") as fh:
        fh.write("Saylor URL: http://www.saylor.org/books Saylor.org\n621\n"
                 "4. At a minimum daily wage of $200 per day, how many "
                 "dishwashers will be employed at the restaurant?")
    open(p + ".native", "w").close()
    be = FakeBackend(vtb, ["", ""])
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert "dishwashers" in vtb._strip_tags(got[p][0].html)


def test_an_empty_read_of_a_blank_native_page_stays_blank(vtb, tmp_path):
    """A blank leaf's layer holds only watermark and folio; stripping them
    leaves nothing, and nothing is what the page gets."""
    img = _page(ink_rows=6)
    p = str(tmp_path / "page_0000.jpg"); img.save(p)
    with open(p + ".layer.txt", "w") as fh:
        fh.write("Saylor URL: http://www.saylor.org/books Saylor.org\n266")
    open(p + ".native", "w").close()
    be = FakeBackend(vtb, ["", ""])
    got = vtb._read_pages_resiliently([p], [img], be, None)
    assert got[p] == []


def test_a_page_read_as_empty_but_readable_is_not_dropped_as_blank(vtb, monkeypatch):
    """
    The silent counterpart of fabrication: an empty body is deleted with the
    word "blank" beside it, and both witnesses are deaf because they judge
    text we produced. A classical reader that finds a sentence on the page
    overrules the emptiness.
    """
    counts = {"/tmp/real.jpg": 40, "/tmp/leaf.jpg": 0, "/tmp/showthrough.jpg": 4}
    monkeypatch.setattr(vtb, "_classical_word_count",
                        lambda p, min_len=2: counts[p])
    paths = ["/tmp/real.jpg", "/tmp/leaf.jpg", "/tmp/showthrough.jpg"]
    blank = {0, 1, 2}
    keep = {i for i in blank
            if (n := vtb._classical_word_count(paths[i], min_len=3)) and n >= 10}
    assert keep == {0}, "only the page with real text survives the drop"


def test_a_picture_page_is_content_and_gets_cached(vtb, tmp_path):
    """A plate, a map, a frontispiece carries no words. Judging such a page
    by its text alone calls a good read a failure: an "empty" read of an
    inked page is deliberately withheld from the cache, so the page is
    re-read live on EVERY later build — and the one time the inference
    server is slow it comes back empty, ships no figure, and is dropped as
    blank. Two full-page plates left a book that way, and 87 pages across
    the shelf were being re-read live for this reason."""
    import numpy as np, cv2
    from PIL import Image
    img = np.full((400, 300, 3), 255, np.uint8)
    cv2.rectangle(img, (30, 40), (270, 360), (20,) * 3, -1)   # a big plate
    p = tmp_path / "plate.png"
    cv2.imwrite(str(p), img)

    class PlateBackend:
        batch_size = 1
        name = "fake"
        langs = ["en"]
        calls = 0

        def run_items(self, images):
            PlateBackend.calls += 1
            return [[vtb.PageItem(html="", figure=Image.open(str(p)))]]

    class Cache:
        def __init__(self):
            self.stored = {}

        def get(self, path):
            return None

        def put(self, path, items):
            self.stored[path] = items

    cache = Cache()
    out = vtb._read_pages_resiliently([str(p)], [Image.open(str(p))],
                                      PlateBackend(), cache)
    assert str(p) in cache.stored, "a picture page must be cached"
    assert PlateBackend.calls == 1, "and must not be retried as a failed read"
    assert out[str(p)], "the picture survives the read"
