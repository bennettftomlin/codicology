"""The guard against show-through read as text.

Two pages arrived on one book the day the geometry ladder changed its
pixels: a blank verso whose reading transcribed the ghost of the reverse
leaf, and one where the recogniser invented a fluent sentence outright.
Both measured a core ink fraction of exactly 0.00000 while the faintest
real sparse page measured 0.00597 — but a withdrawn ancestor ("text on a
near-blank page") once condemned two books' title pages, and the record
holds a real six-word page whose ink is indistinguishable from a blank
leaf's. So conviction here takes two agreeing witnesses and never touches
a page under ten words.
"""
import numpy as np
import pytest
from PIL import Image

import codicology.pipeline as vtb


def _paper(path, ink=False):
    """A photographed-page stand-in: plain paper, optionally with real ink.

    Strokes are 5px — comfortably under the 9px black-hat kernel, which
    cannot see features wider than itself (a lesson paid for once already
    with 12px synthetic bars an earlier probe called invisible).
    """
    a = np.full((900, 600), 235, dtype=np.uint8)
    rng = np.random.default_rng(7)
    a = np.clip(a + rng.integers(-4, 4, a.shape), 0, 255).astype(np.uint8)
    if ink:
        for row in range(200, 700, 40):
            a[row:row + 5, 150:450] = 30
    Image.fromarray(a).save(path)
    return str(path)


GHOST_TEXT = ("The proper text of the statement of the statement is "
              "repeated and are now not read in the paper and here")


@pytest.fixture
def fresh_caches():
    vtb._page_core_ink.cache_clear()
    vtb._classical_word_count.cache_clear()
    before = dict(vtb.UNWITNESSED)
    yield
    vtb.UNWITNESSED.update(before)


def test_ghost_on_blank_paper_is_convicted(tmp_path, monkeypatch, fresh_caches):
    p = _paper(tmp_path / "p.png")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: 0)
    assert vtb._phantom_blank_pages([p], [GHOST_TEXT], [False]) == {0}


def test_classical_reader_finding_text_acquits(tmp_path, monkeypatch, fresh_caches):
    p = _paper(tmp_path / "p.png")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: 7)
    assert vtb._phantom_blank_pages([p], [GHOST_TEXT], [False]) == set()


def test_real_ink_acquits_even_when_the_reader_is_silent(tmp_path, monkeypatch,
                                                         fresh_caches):
    """The measured page: 50 true words tesseract could not see. Ink must
    veto before the silent witness is even asked."""
    p = _paper(tmp_path / "p.png", ink=True)
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: 0)
    assert vtb._phantom_blank_pages([p], [GHOST_TEXT], [False]) == set()


def test_six_word_pages_are_never_questioned(tmp_path, monkeypatch, fresh_caches):
    """The dedication class that killed the withdrawn guard: sparse AND
    inkless AND unwitnessable — still kept, on principle."""
    p = _paper(tmp_path / "p.png")
    asked = []
    monkeypatch.setattr(vtb, "_classical_word_count",
                        lambda *a, **k: asked.append(1) or 0)
    out = vtb._phantom_blank_pages([p], ["For Mary and the old country"],
                                   [False])
    assert out == set() and not asked


def test_verbose_inventions_are_still_convicted(tmp_path, monkeypatch,
                                                fresh_caches):
    """The first draft capped suspicion at 40 words; the very next rebuild
    produced a 140-word invention on a blank verso. Length is evidence of
    nothing — no real page carries a hundred words with no measurable ink."""
    p = _paper(tmp_path / "p.png")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: 0)
    text = " ".join(["invented"] * 140)
    assert vtb._phantom_blank_pages([p], [text], [False]) == {0}


def test_a_wordy_inked_page_is_acquitted_by_its_ink(tmp_path, monkeypatch,
                                                    fresh_caches):
    p = _paper(tmp_path / "p.png", ink=True)
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: 0)
    text = " ".join(["word"] * 140)
    assert vtb._phantom_blank_pages([p], [text], [False]) == set()


def test_a_figure_page_is_content(tmp_path, monkeypatch, fresh_caches):
    p = _paper(tmp_path / "p.png")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: 0)
    assert vtb._phantom_blank_pages([p], [GHOST_TEXT], [True]) == set()


def test_no_witness_means_no_conviction_and_an_honest_count(tmp_path,
                                                            monkeypatch,
                                                            fresh_caches):
    p = _paper(tmp_path / "p.png")
    monkeypatch.setattr(vtb, "_classical_word_count", lambda *a, **k: None)
    before = vtb.UNWITNESSED["fabrication"]
    assert vtb._phantom_blank_pages([p], [GHOST_TEXT], [False]) == set()
    assert vtb.UNWITNESSED["fabrication"] == before + 1


def test_asides_alone_read_as_empty():
    aside = ("[Faded text block, likely bleed-through from the reverse "
             "side of the page] [Faded text block, likely bleed-through "
             "from the reverse side of the page]")
    assert vtb._reads_as_empty(aside)
    assert vtb._reads_as_empty("   ")
    assert not vtb._reads_as_empty("Chapter One [sic] began")
    # A bare arabic number is a leaked folio, not text; roman stays out —
    # "vi" may be a part divider's whole printed content.
    assert vtb._reads_as_empty("14")
    assert vtb._reads_as_empty(" 2 ")
    assert not vtb._reads_as_empty("vi")
    assert not vtb._reads_as_empty("14 men died")
