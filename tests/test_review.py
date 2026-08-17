"""The review sheet — where the dispute record meets the human eye.

The facts these tests pin down: ranking is by likelihood the shipped word
is wrong, so the ladder's singleton catches outrank abstains and systematic
repeats sink into collapsed groups; every row — including confirmed ones —
carries a free-text field because the reader's own insight outranks every
rung; and the sheet is a single self-contained file that phones home to
nothing.
"""
import json

from codicology import review


def _report(disputes):
    return {"pdf": "", "epub": "book.epub", "name": "book",
            "disputes": disputes}


def _d(page, surya, tess, rung=None, winner=None):
    return {"page": page, "surya": surya, "tesseract": tess,
            "rung": rung or ("abstain" if winner is None else rung),
            "winner": winner, "shipped": surya}


def test_singleton_catches_outrank_everything(vtb):
    tiers = review.build_tiers([
        _d(3, "property", "properly"),                       # open abstain
        _d(5, "beligerents", "belligerents", "dictionary",
           winner="belligerents"),                           # the catch
        _d(7, "exposure", "expdésure", "dictionary", winner="exposure"),
    ])
    assert [r["surya"] for r in tiers["catches"]] == ["beligerents"]
    assert [r["surya"] for r in tiers["confirmed"]] == ["exposure"]


def test_repeated_pairs_are_convention_not_damage(vtb):
    rows = [_d(p, "ff", "f", "dictionary", winner="f") for p in range(20)]
    rows.append(_d(30, "conscented", "consented", "dictionary",
                   winner="consented"))
    tiers = review.build_tiers(rows)
    assert len(tiers["systematic"]) == 20
    assert [r["surya"] for r in tiers["catches"]] == ["conscented"]


def test_broken_abstains_outrank_open_ones(vtb):
    tiers = review.build_tiers([
        _d(1, "property", "properly"),      # both words: open
        _d(2, "mcisaac", "mcisaas"),        # neither a word: broken
    ])
    assert [r["surya"] for r in tiers["broken"]] == ["mcisaac"]
    assert [r["surya"] for r in tiers["open"]] == ["property"]


def test_catches_sort_by_rung_reliability(vtb):
    tiers = review.build_tiers([
        _d(9, "designing", "deigning", "vision", winner="deigning"),
        _d(2, "parently", "apparently", "dictionary", winner="apparently"),
        _d(5, "Haisted", "Halsted", "lexicon", winner="Halsted"),
    ])
    assert [r["rung"] for r in tiers["catches"]] == [
        "lexicon", "vision", "dictionary"]


def test_every_row_is_human_overridable(vtb):
    disputes = [_d(1, "ff", "f", "dictionary", winner="f")] * 3 + [
        _d(2, "exposure", "expdésure", "dictionary", winner="exposure"),
        _d(3, "property", "properly"),
    ]
    sheet = review.render_sheet(_report(disputes))
    assert sheet.count('input class="own"') == len(disputes), \
        "confirmed and systematic rows must stay editable too"


def test_occurrences_count_repeats_within_a_page(vtb):
    occs = review.assign_occurrences([
        _d(4, "ff", "f"), _d(4, "ff", "f"), _d(4, "op", "of"),
        _d(5, "ff", "f"),
    ])
    assert occs == [0, 1, 0, 0]


def test_sheet_is_self_contained(vtb):
    sheet = review.render_sheet(_report([
        _d(1, "beligerents", "belligerents", "dictionary",
           winner="belligerents")]))
    assert "http://" not in sheet and "https://" not in sheet
    assert "src=" not in sheet.replace('src="data:image/png', "")


def test_sheet_survives_html_hostile_words(vtb):
    sheet = review.render_sheet(_report([
        _d(1, '<i>&"quoted"', "plain", "dictionary", winner="plain")]))
    assert "<i>&" not in sheet
    assert "&lt;i&gt;" in sheet


def test_export_distinguishes_human_from_ladder(vtb):
    sheet = review.render_sheet(_report([
        _d(1, "a", "b", "dictionary", winner="b")]))
    assert "'human'" in sheet and "'ladder'" in sheet
    assert ".decisions.json" in sheet


def test_missing_crop_never_blocks_the_row(vtb):
    sheet = review.render_sheet(_report([
        _d(1, "word", "worb", "dictionary", winner="word")]), crops={})
    assert "ink not located" in sheet


def test_rung_chips_carry_measured_accuracy(vtb):
    sheet = review.render_sheet(_report([
        _d(1, "Haisted", "Halsted", "lexicon", winner="Halsted")]))
    assert "lexicon 97.8%" in sheet


def test_fingerprint_is_stable_across_renders(vtb):
    rep = _report([_d(1, "a", "b", "dictionary", winner="b")])
    assert review._fingerprint(rep) == review._fingerprint(
        json.loads(json.dumps(rep)))


def test_locator_exact_beats_everything(vtb):
    words = [(0, 0, 9, 9, "property"), (30, 0, 9, 9, "properly")]
    assert review._locate(words, ["properly"]) == [words[1]]


def test_locator_finds_apparatus_inside_compounds(vtb):
    words = [(0, 0, 9, 9, "242ff"), (30, 0, 9, 9, "chicago")]
    assert review._locate(words, ["ff"]) == [words[0]]


def test_locator_accepts_a_lone_close_variant(vtb):
    words = [(0, 0, 9, 9, "propcrly"), (30, 0, 9, 9, "chicago")]
    assert review._locate(words, ["properly", "property"]) == [words[0]]


def test_locator_refuses_ambiguity(vtb):
    """Two near-misses: a wrong crop misleads where a missing one is honest."""
    words = [(0, 0, 9, 9, "propcrly"), (30, 0, 9, 9, "properiy")]
    assert review._locate(words, ["properly"]) == []


def test_repeated_abstains_sink_as_convention(vtb):
    """The apparatus guard turns ff/f into abstains by the dozen; identical
    abstains are convention, not eighty-six separate coin-flips."""
    rows = [_d(p, "ff", "f") for p in range(6)]
    rows.append(_d(50, "property", "properly"))
    tiers = review.build_tiers(rows)
    assert len(tiers["systematic"]) == 6
    assert [r["surya"] for r in tiers["open"]] == ["property"]
