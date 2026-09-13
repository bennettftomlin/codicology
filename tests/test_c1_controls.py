"""Punctuation a mis-decoded text layer delivered as control characters.

A PDF writes its punctuation in WinAnsi: 0x95 is a bullet, 0x97 an em dash,
0x92 a right single quote. Decoded as Latin-1 instead, those bytes land in
the C1 block, where they are not punctuation and not anything -- they render
as nothing at all, match no search, and read as nothing aloud. FM 21-10 shipped
608 of them, which is every bullet in every list the manual prints.

This is the only character repair the pipeline makes without being asked, and
the reason it is allowed to is that the mapping is not a judgement: cp1252 is
what the bytes meant, so reading them back through it undoes the fault rather
than correcting the text. The five bytes cp1252 leaves unassigned have no
answer, so they are left alone -- unusual_characters reports those to a human,
which is the right treatment for a character nobody can decode.
"""
import pytest


def test_an_em_dash_the_layer_delivered_as_a_control_comes_back(vtb):
    bodies = [f"<p>principles of, 1-3{chr(0x97)}4</p>"]
    assert vtb.repair_c1_controls(bodies) == 1
    assert bodies == ["<p>principles of, 1-3—4</p>"]


def test_every_bullet_in_a_manuals_lists_comes_back(vtb):
    bodies = [f"<p>{chr(0x95)} boil water</p>", f"<p>{chr(0x95)} treat it</p>"]
    assert vtb.repair_c1_controls(bodies) == 2
    assert bodies == ["<p>• boil water</p>", "<p>• treat it</p>"]


def test_quote_marks_come_back_before_typography_can_see_them(vtb):
    # 0x91-0x94 are the curly quotes; they must be real quotes by the time
    # normalize_typography runs or it has nothing to work on.
    raw = f"{chr(0x93)}water{chr(0x94)} and the soldier{chr(0x92)}s canteen"
    bodies = [f"<p>{raw}</p>"]
    assert vtb.repair_c1_controls(bodies) == 3
    assert bodies == ["<p>“water” and the soldier’s canteen</p>"]


def test_a_symbol_fonts_dingbat_is_not_read_as_cp1252_punctuation(vtb):
    # FM 3-06 sets its second-level list marker at 0x83, an index into a
    # symbol font. cp1252 calls that byte a florin sign, which is visible,
    # searchable, and not what the page prints -- so with no witness in the
    # book it stays a control character and unusual_characters reports it.
    bodies = [f"<p>{chr(0x83)} Toxic industrial material</p>",
              "<p>• Industrial areas</p>"]
    assert vtb.repair_c1_controls(bodies) == 0
    assert bodies[0] == f"<p>{chr(0x83)} Toxic industrial material</p>"


def test_a_letter_the_book_prints_elsewhere_is_restored_on_that_evidence(vtb):
    # The other half of the rule: a book that genuinely lost a letter to the
    # same fault says so, because the letter survived somewhere.
    bodies = [f"<p>c{chr(0x9C)}ur</p>", "<p>le cœur du problème</p>"]
    assert vtb.repair_c1_controls(bodies) == 1
    assert bodies[0] == "<p>cœur</p>"


def test_punctuation_needs_no_witness_because_the_encoding_is_the_evidence(vtb):
    # FM 21-10 prints no curly quote that survived, and its quotes are still
    # restored: for punctuation cp1252 is not a guess about the book.
    bodies = [f"<p>Use {chr(0x93)}safety{chr(0x94)} Stoddard solvent.</p>"]
    assert vtb.repair_c1_controls(bodies) == 2
    assert bodies == ["<p>Use “safety” Stoddard solvent.</p>"]


def test_the_bytes_cp1252_never_assigned_are_left_exactly_as_they_are(vtb):
    for c in (0x81, 0x8D, 0x8F, 0x90, 0x9D):
        bodies = [f"<p>a{chr(c)}b</p>"]
        assert vtb.repair_c1_controls(bodies) == 0
        assert bodies == [f"<p>a{chr(c)}b</p>"]


def test_a_book_with_no_control_characters_is_not_rewritten(vtb):
    bodies = ["<p>ordinary prose — with real punctuation • already</p>"]
    before = list(bodies)
    assert vtb.repair_c1_controls(bodies) == 0
    assert bodies == before


def test_the_repair_reports_what_unusual_characters_was_only_able_to_warn_about(vtb):
    # The detector has always seen these; it says of itself that "nothing is
    # ever changed on this evidence". After the repair there is nothing left
    # for it to report.
    bodies = [f"<p>a{chr(0x97)}b{chr(0x95)}c</p>"]
    assert [ch for ch, _ in vtb.unusual_characters(bodies)]
    vtb.repair_c1_controls(bodies)
    assert [ch for ch, _ in vtb.unusual_characters(bodies)
            if 0x80 <= ord(ch) <= 0x9F] == []


def test_what_the_repair_declines_is_still_handed_to_the_human(vtb):
    bodies = [f"<p>a{chr(0x83)}b{chr(0x90)}c</p>"]
    vtb.repair_c1_controls(bodies)
    left = [ch for ch, _ in vtb.unusual_characters(bodies)
            if 0x80 <= ord(ch) <= 0x9F]
    assert sorted(left) == [chr(0x83), chr(0x90)]


def test_markup_is_not_disturbed(vtb):
    bodies = [f'<p class="x"><a href="page_0001.xhtml#pgb-0001">1{chr(0x97)}3</a></p>']
    assert vtb.repair_c1_controls(bodies) == 1
    assert bodies[0].startswith('<p class="x"><a href="page_0001.xhtml#pgb-0001">')
    assert bodies[0].endswith("</a></p>")
