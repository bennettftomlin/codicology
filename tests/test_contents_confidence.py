"""The parser knowing when its own parse is bad.

Every bad contents parse found on this shelf had its signature sitting in
the build log, printed on adjacent lines and never compared: seven nav
entries for a 288-page book beside running heads describing its chapters
plainly; chapters four and five printed while one to three appeared
nowhere. These tests pin the comparisons, and the one case where the
margins are allowed to win: a parse that missed at least three chapters
and at least half of what the heads independently describe.
"""
import pytest


def _placed(vtb, rows):
    """rows: (title, target_page_or_None)"""
    return [(vtb.TocEntry(t, None, "", 2), pg, pg is not None)
            for t, pg in rows]


def _chapters(vtb, spans):
    return [vtb.Chapter(t, a, b) for t, a, b in spans]


# ---------------------------------------------------------------- confidence

def test_a_sparse_parse_announces_itself(vtb):
    placed = _placed(vtb, [(f"Entry {i}", i * 3) for i in range(7)])
    warns = vtb.contents_confidence(placed, 288, 0, 0)
    assert any("sparse" in w for w in warns), warns


def test_a_normal_parse_stays_quiet(vtb):
    placed = _placed(vtb, [(f"{i}. Chapter {i}", i * 8) for i in range(1, 13)])
    assert vtb.contents_confidence(placed, 300, 0, 0) == []


def test_widespread_placement_failure_announces_itself(vtb):
    placed = _placed(vtb, [(f"Entry {i}", None) for i in range(12)])
    warns = vtb.contents_confidence(placed, 300, 9, 0)
    assert any("never placed" in w for w in warns), warns


def test_a_gap_in_the_printed_chapter_numbers_announces_itself(vtb):
    """The book that prompted this printed chapters 4 and 5 while 1 to 3
    appeared nowhere in the parse."""
    placed = _placed(vtb, [("4 Case Studies of Media Content, 2011", 87),
                           ("5 Impacts of Media Coverage", 131)])
    warns = vtb.contents_confidence(placed, 60, 0, 0)
    assert any("never parsed" in w for w in warns), warns


def test_spelled_out_numbers_join_the_sequence(vtb):
    placed = _placed(vtb, [("CHAPTER ONE. Beginnings", 4),
                           ("CHAPTER TWO. The Middle", 30),
                           ("CHAPTER THREE. The End", 60)])
    assert vtb.contents_confidence(placed, 90, 0, 0) == []


def test_section_numbers_and_years_are_not_chapter_numbers(vtb):
    """"2-15. Steel Helmet" numbers a section, and "1640–49" is a date; a
    gap conjured from either would cry wolf."""
    placed = _placed(vtb, [("2-15. Steel Helmet", 20),
                           ("8-4. Interrelationship", 40),
                           ("The Crisis, 1640–49", 60)])
    warns = vtb.contents_confidence(placed, 30, 0, 0)
    assert not any("never parsed" in w for w in warns), warns


def test_unmerged_disagreement_announces_itself(vtb):
    placed = _placed(vtb, [("Something", 5)])
    warns = vtb.contents_confidence(placed, 30, 0, 4)
    assert any("running heads describe 4" in w for w in warns), warns


# ---------------------------------------------------------------- the merge

def _pos_of(n):
    return {i: i for i in range(n)}


def test_a_plainly_failed_parse_gets_the_margins_chapters(vtb):
    """Seven entries shipped for a 288-page book while the running heads
    named its chapters plainly. At least three missing and at least half:
    the margins win."""
    placed = _placed(vtb, [("Acknowledgements", 3), ("Index", 280)])
    furn = _chapters(vtb, [("THE COLLAPSE", 10, 25), ("THE OTHER", 26, 44),
                           ("UPSIDE DOWN", 45, 60), ("THIRD ESTATE", 61, 80)])
    missing = vtb.furniture_absent_from_contents(placed, furn, _pos_of(300))
    assert len(missing) == 4
    merged, n = vtb.merge_missing_furniture(placed, missing, len(furn))
    assert n == 4
    targets = [t for _, t, _ in merged]
    assert targets == sorted(targets, key=lambda x: (x is None, x)), targets


def test_a_healthy_parse_is_left_entirely_alone(vtb):
    """Eagle's shape: the contents placed richly, the heads agree."""
    placed = _placed(vtb, [(f"CHAPTER {i}", i * 10) for i in range(1, 9)])
    furn = _chapters(vtb, [(f"TITLE {i}", i * 10, i * 10 + 9)
                           for i in range(1, 9)])
    missing = vtb.furniture_absent_from_contents(placed, furn, _pos_of(100))
    assert missing == []


def test_a_title_covered_chapter_is_not_duplicated(vtb):
    """A placed entry that names the chapter covers it even when it landed
    outside the verso window — adding the margins' copy would duplicate an
    entry the reader already has."""
    placed = _placed(vtb, [("The Collapse of the Middle Ages", 4)])
    furn = _chapters(vtb, [("THE COLLAPSE OF THE MIDDLE AGES", 10, 30)])
    assert vtb.furniture_absent_from_contents(placed, furn, _pos_of(50)) == []


def test_even_a_small_shortfall_is_made_good(vtb):
    """The gate that once stood here — at least three missing and half the
    book's — only ever kept proven chapters out of thin navs. Per-chapter
    evidence decides now: a sustained head run nothing in the contents
    accounts for goes in, however rich the rest of the parse."""
    placed = _placed(vtb, [(f"{i}. Chapter", i * 10) for i in range(1, 11)])
    furn = _chapters(vtb, [("A REAL DIVISION", 45, 52)])
    missing = vtb.furniture_absent_from_contents(placed, furn, _pos_of(120))
    merged, n = vtb.merge_missing_furniture(placed, missing, len(furn))
    assert n == 1
    assert any(e.title == "A REAL DIVISION" for e, _, _ in merged)


def test_a_bare_letter_head_is_not_a_nav_label(vtb):
    """One manual's appendix heads are the letters A and B — real
    destinations, meaningless as labels. They stay out; the glossary
    beside them goes in."""
    placed = _placed(vtb, [("Chapter One", 5)])
    furn = _chapters(vtb, [("A", 40, 47), ("B", 48, 55),
                           ("GLOSSARY", 56, 62)])
    missing = vtb.furniture_absent_from_contents(placed, furn, _pos_of(70))
    merged, n = vtb.merge_missing_furniture(placed, missing, len(furn))
    assert n == 1
    titles = [e.title for e, _, _ in merged]
    assert "GLOSSARY" in titles and "A" not in titles


def test_a_head_starting_after_a_verso_opening_is_covered(vtb):
    """The head names its chapter from the page after the opening, two
    leaves back on a verso-title book. A placed entry at the true opening
    covers a head run starting two pages later."""
    placed = _placed(vtb, [("1. The Chapter", 10)])
    furn = _chapters(vtb, [("THE CHAPTER HEAD TEXT DIFFERS", 12, 30)])
    assert vtb.furniture_absent_from_contents(placed, furn, _pos_of(40)) == []


def test_an_unplaced_leaf_does_not_cover_its_chapter(vtb):
    """An unplaced leaf is skipped from the nav entirely; counting it as
    cover blocked the rescue in exactly the case rescue helps."""
    placed = [(vtb.TocEntry("The Gathering Storm", None, "", 2), None, False)]
    furn = _chapters(vtb, [("THE GATHERING STORM", 20, 34)])
    missing = vtb.furniture_absent_from_contents(placed, furn, _pos_of(40))
    assert len(missing) == 1


def test_an_unplaced_grouping_still_covers(vtb):
    """A grouping emits as a heading even unplaced, so it covers."""
    placed = [(vtb.TocEntry("The Gathering Storm", None, "", 0), None, False)]
    furn = _chapters(vtb, [("THE GATHERING STORM", 20, 34)])
    assert vtb.furniture_absent_from_contents(placed, furn, _pos_of(40)) == []


def test_a_head_that_is_the_subtitle_covers_its_chapter(vtb):
    """One book's margins run the post-colon half of the title; matching
    only the front half called five of its chapters unaccounted for."""
    placed = _placed(
        vtb, [("CHAPTER ONE. “The King’s in His Castle”: "
               "The Collapse of the Middle Ages", 40)])
    furn = _chapters(vtb, [("THE COLLAPSE OF THE MIDDLE AGES", 90, 110)])
    assert vtb.furniture_absent_from_contents(placed, furn, _pos_of(120)) == []
