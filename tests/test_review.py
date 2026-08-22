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
    assert "src=" not in sheet.replace('src="data:image/', "")


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
        _d(1, "Haisted", "Halsted", "lexicon", winner="Halsted"),
        _d(2, "parently", "apparently", "dictionary", winner="apparently")]))
    assert "lexicon 97.8%" in sheet
    assert "dictionary 92.3%" in sheet, "chips must carry battery v3"


def test_fingerprint_is_stable_across_renders(vtb):
    rep = _report([_d(1, "a", "b", "dictionary", winner="b")])
    assert review._fingerprint(rep) == review._fingerprint(
        json.loads(json.dumps(rep)))


def test_repeated_abstains_sink_as_convention(vtb):
    """The apparatus guard turns ff/f into abstains by the dozen; identical
    abstains are convention, not eighty-six separate coin-flips."""
    rows = [_d(p, "ff", "f") for p in range(6)]
    rows.append(_d(50, "property", "properly"))
    tiers = review.build_tiers(rows)
    assert len(tiers["systematic"]) == 6
    assert [r["surya"] for r in tiers["open"]] == ["property"]


def _epub(tmp_path, pages):
    import zipfile
    p = tmp_path / "book.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip",
                   zipfile.ZIP_STORED)
        for n, body in pages.items():
            z.writestr(f"EPUB/page_{n:04d}.xhtml",
                       f"<html><body>{body}</body></html>")
    return str(p)


def _decisions(tmp_path, rows):
    p = tmp_path / "d.json"
    p.write_text(json.dumps({"decisions": rows}))
    return str(p)


def test_apply_lands_at_the_recorded_occurrence(vtb, tmp_path):
    epub = _epub(tmp_path, {7: "<p>the ff and the ff again</p>"})
    dec = _decisions(tmp_path, [{"page": 7, "occurrence": 1, "old": "ff",
                                 "new": "6", "source": "human",
                                 "rung": "human"}])
    r = review.apply_decisions(epub, dec)
    assert len(r["applied"]) == 1 and not r["stale"]
    import zipfile
    got = zipfile.ZipFile(epub).read("EPUB/page_0007.xhtml").decode()
    assert "the ff and the 6 again" in got


def test_apply_never_touches_tags(vtb, tmp_path):
    """A correction to the word 'i' must not rewrite <i> markup."""
    epub = _epub(tmp_path, {1: "<p><i>emphasis</i> i said</p>"})
    dec = _decisions(tmp_path, [{"page": 1, "occurrence": 0, "old": "i",
                                 "new": "I", "source": "human",
                                 "rung": "human"}])
    review.apply_decisions(epub, dec)
    import zipfile
    got = zipfile.ZipFile(epub).read("EPUB/page_0001.xhtml").decode()
    assert "<i>emphasis</i> I said" in got


def test_stale_decisions_are_reported_never_guessed(vtb, tmp_path):
    epub = _epub(tmp_path, {2: "<p>a re-read changed this text</p>"})
    dec = _decisions(tmp_path, [{"page": 2, "occurrence": 0,
                                 "old": "vanished", "new": "word",
                                 "source": "human", "rung": "human"}])
    r = review.apply_decisions(epub, dec)
    assert not r["applied"] and len(r["stale"]) == 1
    import zipfile
    got = zipfile.ZipFile(epub).read("EPUB/page_0002.xhtml").decode()
    assert "changed this text" in got, "stale must leave the page alone"


def test_apply_preserves_the_epub_contract(vtb, tmp_path):
    """mimetype stays the first entry and stored; a backup is kept."""
    import os, zipfile
    epub = _epub(tmp_path, {1: "<p>word</p>"})
    dec = _decisions(tmp_path, [{"page": 1, "occurrence": 0, "old": "word",
                                 "new": "world", "source": "ladder",
                                 "rung": "dictionary"}])
    review.apply_decisions(epub, dec)
    z = zipfile.ZipFile(epub)
    first = z.infolist()[0]
    assert first.filename == "mimetype"
    assert first.compress_type == zipfile.ZIP_STORED
    assert os.path.exists(epub + ".preapply")


def test_apply_word_boundaries_protect_substrings(vtb, tmp_path):
    epub = _epub(tmp_path, {3: "<p>off ff offer</p>"})
    dec = _decisions(tmp_path, [{"page": 3, "occurrence": 0, "old": "ff",
                                 "new": "6", "source": "human",
                                 "rung": "human"}])
    review.apply_decisions(epub, dec)
    import zipfile
    got = zipfile.ZipFile(epub).read("EPUB/page_0003.xhtml").decode()
    assert "off 6 offer" in got


def test_rebuild_reapply_lands_before_linkers(vtb, tmp_path):
    dec = _decisions(tmp_path, [{"page": 1, "occurrence": 0,
                                 "old": "beligerents", "new": "belligerents",
                                 "source": "ladder", "rung": "dictionary"}])
    bodies = ["<p>front</p>", "<p>the beligerents met</p>"]
    st = vtb.apply_reviewer_decisions(bodies, dec)
    assert st == {"applied": 1, "stale": 0}
    assert bodies[1] == "<p>the belligerents met</p>"


def test_rebuild_reapply_reports_stale_sites(vtb, tmp_path):
    dec = _decisions(tmp_path, [{"page": 0, "occurrence": 0,
                                 "old": "vanished", "new": "word",
                                 "source": "human", "rung": "human"}])
    bodies = ["<p>a re-read changed this</p>"]
    st = vtb.apply_reviewer_decisions(bodies, dec)
    assert st == {"applied": 0, "stale": 1}
    assert "changed this" in bodies[0]


def test_rebuild_reapply_tolerates_quote_shape(vtb, tmp_path):
    """The reviewer read a typography-restored book; a rebuild's text may
    carry the other quote shape. Both variants are tried before stale."""
    dec = _decisions(tmp_path, [{"page": 0, "occurrence": 0,
                                 "old": "don't", "new": "won't",
                                 "source": "human", "rung": "human"}])
    bodies = ["<p>they don’t know</p>"]
    st = vtb.apply_reviewer_decisions(bodies, dec)
    assert st["applied"] == 1
    assert "won't" in bodies[0]


def test_rebuild_reapply_highest_occurrence_first(vtb, tmp_path):
    dec = _decisions(tmp_path, [
        {"page": 0, "occurrence": 0, "old": "ff", "new": "6",
         "source": "human", "rung": "human"},
        {"page": 0, "occurrence": 1, "old": "ff", "new": "7",
         "source": "human", "rung": "human"},
    ])
    bodies = ["<p>see ff and ff</p>"]
    st = vtb.apply_reviewer_decisions(bodies, dec)
    assert st["applied"] == 2
    assert bodies[0] == "<p>see 6 and 7</p>"


def test_apply_reaches_words_stored_as_entities(vtb, tmp_path):
    """The reviewer corrects the printed &c.; the page stores &amp;c.
    The escaped retry must land it, and the replacement must arrive
    escaped — a raw ampersand would corrupt the xhtml."""
    epub = _epub(tmp_path, {4: "<p>letters, &amp;c. sent to H&amp;M</p>"})
    dec = _decisions(tmp_path, [
        {"page": 4, "occurrence": 0, "old": "&c.", "new": "etc.",
         "source": "human", "rung": "human"},
        {"page": 4, "occurrence": 0, "old": "sent", "new": "s&nt",
         "source": "human", "rung": "human"}])
    r = review.apply_decisions(epub, dec)
    assert len(r["applied"]) == 2 and not r["stale"]
    import zipfile
    got = zipfile.ZipFile(epub).read("EPUB/page_0004.xhtml").decode()
    assert "letters, etc. s&amp;nt to H&amp;M" in got
