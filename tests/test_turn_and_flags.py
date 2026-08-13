"""
Page-turn fragments, and pages that need a human.

find_page_turns exists because three *image* measures were tried first and all
three failed: a page eighty percent hidden behind the next leaf has an ordinary
shape and more dark pixels than the median (the covering leaf's shadow, and the
desk), while the genuinely low-ink pages are the title page and the chapter
openings — the worst possible things to drop. So the verdict is made on text,
and only when a short capture's words are already on a page *close by*.

All the text here is written by hand and every page image is synthetic. Nothing
in this module reads the real book, opens a video, or loads an OCR model.
"""
import glob
import os

import pytest


# --------------------------------------------------------------------------
# Deterministic page text. The ordinary pages share no vocabulary with any of
# the fragments below, so a match is always a match the function really found.
# --------------------------------------------------------------------------

_ORDINARY = (
    "harbour ledger notary consulate steamer tobacco veranda monsoon quinine "
    "survey rubber wharf clerk sealed cable envelope schooner rainfall pepper "
    "customs manifest tonnage barrel lantern rigging"
).split()


def ordinary(k: int, count: int = 110) -> str:
    """A page of ordinary length, distinct per k, sharing nothing with the fragments."""
    return " ".join(_ORDINARY[(k * 7 + t) % len(_ORDINARY)] for t in range(count))


def pages(n: int, replace: dict[int, str] | None = None) -> list[str]:
    """n ordinary pages with the given indices replaced."""
    texts = [ordinary(k) for k in range(n)]
    for i, t in (replace or {}).items():
        texts[i] = t
    return texts


# A full, ordinary page of the book.
CHICAGO_PAGE = (
    "In Chicago the winter came early that year and the lake froze solid along "
    "the breakwater below the water works. The orgies and the gambling of the "
    "north side had moved indoors, into the rooms above the shuttered shops on "
    "Clark Street, and the buildings behind them fell into disrepair while the "
    "aldermen argued over who should pay to light the lanes. A reporter counted "
    "eleven such houses in a single block, and wrote that the police had learned "
    "to walk on the other side of the road. The city paid for the lamps in the "
    "end, the houses stayed exactly where they stood, and nobody in the ward was "
    "surprised by any part of it."
)

# The same page caught with the next leaf across it: a sliver of text, every
# visible word cut off mid-word by the covering page's edge.
CHICAGO_FRAGMENT = "In Chicag the wint orgi a the gambl disrepai the alderm breakwat shutte"

# Low-ink pages that must survive. The title page says little and repeats
# nothing near it; the preface four leaves later happens to name the same people.
TITLE_PAGE = (
    "THE CITADEL OF SIN\n\n"
    "A Chronicle of Marchetti and Vansittart\n\n"
    "Harroway Blackwood Press\nLondon MCMXLVII"
)

PREFACE_PAGE = (
    "The chronicle that follows was begun by Marchetti in the year the Harroway "
    "papers were opened, and finished by Vansittart, who had by then left London "
    "for good. Blackwood, whose press printed the first edition in MCMXLVII, "
    "thought the title too strong: a citadel, he wrote, is a thing a man can walk "
    "into and out of again, and the men in these pages never once walked out. He "
    "was overruled, as editors are, and the jacket went to the printer unchanged. "
    "Everything set down here was told to one of us by somebody who was in the "
    "room at the time, and where two accounts of a room disagreed we have said so "
    "in a note rather than choose between them."
)

CHAPTER_OPENING = (
    "CHAPTER FOUR\n\nThe Long Walk Home\n\n"
    "Salvatore left the courthouse an hour before the verdict was read."
)


def fake_paths(n: int) -> list[str]:
    """Paths to files that do not exist. Nothing here may be opened."""
    return [f"/nowhere/at/all/page_{i:04d}.jpg" for i in range(n)]


# --------------------------------------------------------------------------
# find_page_turns
# --------------------------------------------------------------------------

def test_no_pages_yields_no_turns(vtb):
    assert vtb.find_page_turns([], []) == []


def test_without_text_nothing_can_be_called_a_turn(vtb, page_files):
    """The verdict is made on words; with no words there is no verdict to make."""
    assert vtb.find_page_turns(page_files(5), None) == []


def test_fragment_cut_mid_word_still_matches_the_page_it_covers(vtb, page_files):
    """
    The decisive case. Every legible word on the fragment is a truncation —
    "Chicag", "orgi", "disrepai" — so as *words* it overlaps its own source page
    by exactly zero, which is what the first attempt scored. Matching character
    runs instead, it is contained by that page and can be dropped safely.
    """
    source_words = {w.strip(".,:;").lower() for w in CHICAGO_PAGE.split()}
    visible = [w.lower() for w in CHICAGO_FRAGMENT.split() if len(w) >= 4]
    assert visible, "the fragment must have something legible on it"
    assert not (source_words & set(visible)), (
        "this test is only meaningful while no legible word on the fragment is a "
        "whole word of its source page"
    )

    texts = pages(5, {1: CHICAGO_PAGE, 2: CHICAGO_FRAGMENT})
    turns = vtb.find_page_turns(page_files(5), texts)

    assert [i for i, _ in turns] == [2]


def test_turn_reason_names_the_page_the_fragment_repeats(vtb, page_files):
    texts = pages(5, {1: CHICAGO_PAGE, 2: CHICAGO_FRAGMENT})
    (index, why), = vtb.find_page_turns(page_files(5), texts)

    assert index == 2
    assert "page 1" in why           # the page it was found to repeat
    assert "13 words" in why         # what it says, against its neighbours


def test_turns_are_decided_without_ever_opening_the_page_images(vtb):
    """
    Geometry cannot tell a covered page from an uncovered one — dark-pixel count,
    proportions and text spread all failed on this very fragment. The verdict is
    text alone, so it must stand up with no image to look at.
    """
    texts = pages(5, {1: CHICAGO_PAGE, 2: CHICAGO_FRAGMENT})
    assert [i for i, _ in vtb.find_page_turns(fake_paths(5), texts)] == [2]


def test_title_page_repeated_four_pages_away_is_not_a_turn(vtb, page_files):
    """
    The real false positive that fixed the window at 2: a title page is short,
    and a preface several leaves later names both authors and the press. On
    characters, its every stub is inside that preface — but a leaf in flight
    shows the page beside it and no other, so distance is what saves it.
    """
    texts = pages(9, {2: TITLE_PAGE, 6: PREFACE_PAGE})
    assert vtb.find_page_turns(page_files(9), texts) == []


def test_the_same_repeat_one_page_away_is_a_turn(vtb, page_files):
    """Same two texts, moved next to each other: only the distance changed."""
    texts = pages(9, {2: TITLE_PAGE, 3: PREFACE_PAGE})
    assert [i for i, _ in vtb.find_page_turns(page_files(9), texts)] == [2]


def test_chapter_opening_that_repeats_nothing_is_kept(vtb, page_files):
    """Low ink is not evidence. Short and unrepeated is just a chapter opening."""
    texts = pages(7, {3: CHAPTER_OPENING})
    assert vtb.find_page_turns(page_files(7), texts) == []


def test_full_length_page_is_never_a_turn_even_when_it_repeats_its_neighbour(vtb, page_files):
    """A page that says as much as its neighbours has not been caught in flight."""
    texts = pages(6, {3: CHICAGO_PAGE, 4: CHICAGO_PAGE})
    assert vtb.find_page_turns(page_files(6), texts) == []


def test_short_pages_among_short_pages_are_left_alone(vtb, page_files):
    """
    "Shorter than its neighbours" says nothing where no neighbour is substantial:
    front matter is all short and half of it repeats the rest.
    """
    texts = [TITLE_PAGE, TITLE_PAGE, CHAPTER_OPENING, TITLE_PAGE, CHAPTER_OPENING]
    assert vtb.find_page_turns(page_files(5), texts) == []


# --------------------------------------------------------------------------
# flag_pages
# --------------------------------------------------------------------------

def written(folder: str) -> list[str]:
    return sorted(glob.glob(os.path.join(folder, "*.jpg")))


def test_hand_across_the_body_is_flagged_for_a_human(vtb, make_page, write_images):
    folder = write_images([make_page(), make_page(hand=True), make_page()])
    paths = written(folder)
    flags = vtb.flag_pages(paths, [ordinary(k) for k in range(3)],
                           page_ids=["p0", "p1", "p2"])

    hands = [f for f in flags if f.kind == "hand"]
    assert [f.index for f in hands] == [1]
    assert "p1" in hands[0].action


def test_page_with_no_text_is_flagged_blank(vtb, make_page, write_images):
    folder = write_images([make_page(), make_page(), make_page()])
    flags = vtb.flag_pages(written(folder), [ordinary(0), "", ordinary(2)],
                           page_ids=["p0", "p1", "p2"])

    blanks = [f for f in flags if f.kind == "blank"]
    assert [f.index for f in blanks] == [1]
    assert "p1" in blanks[0].action


def test_every_flag_says_what_to_do_about_which_page(vtb, make_page, write_images):
    """A flag a reader cannot act on is noise; each one names its own page."""
    folder = write_images([make_page(), make_page(hand=True), make_page()])
    ids = ["v1-p0", "v1-p1", "v1-p2"]
    flags = vtb.flag_pages(written(folder), [ordinary(0), "", ordinary(2)],
                           page_ids=ids)

    assert flags, "the scene has a hand and an empty page in it"
    for f in flags:
        assert isinstance(f, vtb.PageFlag)
        assert f.detail
        assert ids[f.index] in f.action


def test_flags_fall_back_to_the_page_number_when_there_are_no_ids(vtb, make_page,
                                                                  write_images):
    folder = write_images([make_page(), make_page(hand=True), make_page()])
    flags = vtb.flag_pages(written(folder), [ordinary(k) for k in range(3)])

    hands = [f for f in flags if f.kind == "hand"]
    assert [f.index for f in hands] == [1]
    assert "1" in hands[0].action


def test_clean_pages_raise_nothing(vtb, make_page, write_images):
    """The flags have to stay quiet on good pages or nobody reads the sheet."""
    folder = write_images([make_page() for _ in range(4)])
    assert vtb.flag_pages(written(folder), [ordinary(k) for k in range(4)]) == []


def test_turn_fragments_reach_the_reader_as_a_flag_with_a_drop_action(vtb, page_files):
    texts = pages(5, {1: CHICAGO_PAGE, 2: CHICAGO_FRAGMENT})
    flags = vtb.flag_pages(page_files(5), texts,
                           page_ids=[f"p{i}" for i in range(5)])

    turns = [f for f in flags if f.kind == "turn"]
    assert [f.index for f in turns] == [2]
    assert "--drop-pages p2" in turns[0].action
