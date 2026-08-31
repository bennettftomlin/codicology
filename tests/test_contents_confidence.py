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
    """An entry that failed to place still declares its chapter; adding the
    margins' copy would duplicate it, not rescue it."""
    placed = _placed(vtb, [("The Collapse of the Middle Ages", None)])
    furn = _chapters(vtb, [("THE COLLAPSE OF THE MIDDLE AGES", 10, 30)])
    assert vtb.furniture_absent_from_contents(placed, furn, _pos_of(50)) == []


def test_a_small_disagreement_does_not_merge(vtb):
    """Two chapters short of a rich contents is placement's problem, not
    the parse having failed; the contents outranks the margins there."""
    placed = _placed(vtb, [(f"{i}. Chapter", i * 10) for i in range(1, 11)])
    furn = _chapters(vtb, [("EXTRA ONE", 45, 52), ("EXTRA TWO", 85, 92)])
    missing = vtb.furniture_absent_from_contents(placed, furn, _pos_of(120))
    merged, n = vtb.merge_missing_furniture(placed, missing, len(furn))
    assert n == 0 and merged == placed
