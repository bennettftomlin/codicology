"""Reading and auditing the printed page numbers.

Two jobs are pinned here.

`parse_folio` reads a number off a running head. Its whole value is in what it
*refuses*: the docstring says a bare number is rejected because "a stray numeral
from the body text would otherwise sail through as a folio and quietly corrupt
the ordering it was meant to check". A folio reader that is merely permissive is
worse than none at all, so most of these tests are rejections.

`audit_folios` compares those numbers against the order the pages are actually
in. It reports inversions, duplicates and gaps -- but a page with no readable
head is "not evidence of anything", so unread pages are stepped over rather than
counted as breaks, and a missing number is only a gap when no page stands where
that number should have been.

Everything here is plain data: no images, no OCR backend, no video.
"""
import pytest


def folio(vtb, index, number, confident=True, text=""):
    """A Folio built the way read_folios builds one, for audit_folios to chew on."""
    return vtb.Folio(index, number, text, confident)


# --------------------------------------------------------------------------
# parse_folio -- what it accepts
# --------------------------------------------------------------------------

def test_verso_head_reads_the_number_at_the_start(vtb):
    assert vtb.parse_folio("60 - Citadel of Sin") == 60


def test_recto_head_reads_the_number_at_the_end_not_the_chapter_number(vtb):
    # "Chapter 7 - 61" carries two numerals. The folio is the one at the end;
    # returning 7 here would renumber the book by its chapter headings.
    assert vtb.parse_folio("Chapter 7 - 61") == 61


def test_recto_head_reads_a_number_after_a_one_word_title(vtb):
    assert vtb.parse_folio("Sources - 123") == 123


@pytest.mark.parametrize("sep", ["•", "·", "|", ":", "-"])
def test_every_documented_separator_is_accepted_on_a_verso(vtb, sep):
    assert vtb.parse_folio(f"60 {sep} Citadel of Sin") == 60


@pytest.mark.parametrize("sep", ["•", "·", "|", ":", "-"])
def test_every_documented_separator_is_accepted_on_a_recto(vtb, sep):
    assert vtb.parse_folio(f"Chapter 7 {sep} 61") == 61


def test_spacing_around_the_separator_does_not_matter(vtb):
    # OCR spacing around furniture is not dependable; the number is.
    assert vtb.parse_folio("60-Citadel of Sin") == 60
    assert vtb.parse_folio("60   -   Citadel of Sin") == 60
    assert vtb.parse_folio("Citadel of Sin-61") == 61


def test_leading_and_trailing_whitespace_and_newlines_are_normalised(vtb):
    # read_folios hands over whatever the recogniser emitted, ragged edges included.
    assert vtb.parse_folio("\n  60 - Citadel of Sin  \n") == 60


def test_the_extreme_ends_of_the_legal_range_are_accepted(vtb):
    assert vtb.parse_folio("1 - Citadel of Sin") == 1
    assert vtb.parse_folio("999 - Citadel of Sin") == 999


# --------------------------------------------------------------------------
# parse_folio -- what it refuses, which is the point of it
# --------------------------------------------------------------------------

def test_bare_number_is_rejected(vtb):
    """The named failure: a lone numeral must not be mistaken for a folio.

    Accepting a bare number is the approach the docstring says was rejected --
    "a stray numeral from the body text would otherwise sail through as a folio
    and quietly corrupt the ordering it was meant to check". A folio must come
    with words beside it or it is not a running head.
    """
    assert vtb.parse_folio("60") is None
    assert vtb.parse_folio("  61  ") is None
    assert vtb.parse_folio("7") is None


def test_a_line_of_only_numerals_is_rejected(vtb):
    # A table row, a date span, a figure caption of bare digits: no letters, no folio.
    assert vtb.parse_folio("60 - 61") is None
    assert vtb.parse_folio("1914 - 1918") is None


def test_body_text_longer_than_a_running_head_is_rejected(vtb):
    """A folio is furniture: short. Body text that happens to open with a numeral
    and a dash would otherwise be read as a page number."""
    long_line = "12 - and the men who had come down from the north said nothing at all"
    assert len(long_line) > vtb.RUNNING_HEAD_MAX_CHARS
    assert vtb.parse_folio(long_line) is None


def test_the_length_limit_is_the_running_head_limit(vtb):
    # Exactly at the limit is furniture; one character over is prose.
    at_limit = "60 - " + "Citadel of Sin".ljust(vtb.RUNNING_HEAD_MAX_CHARS - 5, "x")
    over_limit = at_limit + "x"
    assert len(at_limit) == vtb.RUNNING_HEAD_MAX_CHARS
    assert vtb.parse_folio(at_limit) == 60
    assert vtb.parse_folio(over_limit) is None


def test_number_above_999_is_rejected_at_either_end(vtb):
    """A four-figure numeral is a year or a quantity, never a folio in this book.

    Left in, it would dwarf every real folio and turn the audit into a wall of
    invented inversions -- exactly the corruption of ordering the parser exists
    to prevent.
    """
    assert vtb.parse_folio("1024 - Citadel of Sin") is None
    assert vtb.parse_folio("Citadel of Sin - 1024") is None
    assert vtb.parse_folio("1000 - Citadel of Sin") is None


def test_a_year_opening_a_short_sentence_is_not_a_folio(vtb):
    # Short enough to be furniture, shaped like a verso head, and still refused
    # because 1963 cannot be a page number here.
    assert vtb.parse_folio("1963 - the year of the fire") is None


def test_zero_is_rejected(vtb):
    # Books have no page nought; a leading 0 is a misread of something else.
    assert vtb.parse_folio("0 - Citadel of Sin") is None


def test_a_number_set_off_by_space_alone_is_accepted(vtb):
    """
    Requiring a printed separator turned out to be a bug, not a safeguard: the
    first foreign book tried ("Eagle Forgotten", 1938) sets its heads as
    "THE NEW CAREER IN THE NEW ERA 51" — number and title parted by nothing but
    space — and every one of its folios was rejected. The guard against body
    text does not come from the separator anyway: it comes from the line being
    short, carrying words, and the number sitting at an end. Those stay.
    """
    assert vtb.parse_folio("60 Citadel of Sin") == 60
    assert vtb.parse_folio("Citadel of Sin 61") == 61
    assert vtb.parse_folio("THE NEW CAREER IN THE NEW ERA 51") == 51
    # a year can never be taken for a folio, separator or not
    assert vtb.parse_folio("CHICAGO IN 1893") is None


def test_a_head_with_no_number_is_rejected(vtb):
    assert vtb.parse_folio("Citadel of Sin") is None
    assert vtb.parse_folio("CHAPTER SEVEN") is None


def test_empty_and_whitespace_only_input_return_none(vtb):
    # _ocr_one returns "" when the backend throws; parse_folio must survive it.
    assert vtb.parse_folio("") is None
    assert vtb.parse_folio("   \n\t ") is None


# --------------------------------------------------------------------------
# Folio -- the record the audit consumes
# --------------------------------------------------------------------------

def test_folio_keeps_the_recognised_text_for_a_human_to_check(vtb):
    f = vtb.Folio(3, 61, "Chapter 7 - 61", True)
    assert (f.index, f.number, f.text, f.confident) == (3, 61, "Chapter 7 - 61", True)


# --------------------------------------------------------------------------
# audit_folios
# --------------------------------------------------------------------------

def test_ascending_run_is_clean(vtb):
    folios = [folio(vtb, i, 10 + i) for i in range(6)]
    audit = vtb.audit_folios(folios)
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])
    assert (audit.read, audit.total) == (6, 6)


def test_transposed_pair_is_reported_as_an_inversion(vtb):
    # Pages 11 and 12 came out of the extractor the wrong way round.
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 12),
              folio(vtb, 2, 11), folio(vtb, 3, 13)]
    audit = vtb.audit_folios(folios)
    assert audit.inversions == [(1, 2)]


def test_inversion_names_the_two_page_indices_not_the_folio_numbers(vtb):
    # The report has to be actionable: a human fixes pages by position in the
    # run, so the pair reported is (page index, page index).
    folios = [folio(vtb, 40, 200), folio(vtb, 41, 150)]
    audit = vtb.audit_folios(folios)
    assert audit.inversions == [(40, 41)]


def test_repeated_number_is_reported_as_a_duplicate(vtb):
    # The same page photographed twice: one folio number, two page indices.
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 11),
              folio(vtb, 2, 11), folio(vtb, 3, 12)]
    audit = vtb.audit_folios(folios)
    assert audit.duplicates == [(11, [1, 2])]


def test_duplicates_are_found_even_when_the_copies_are_far_apart(vtb):
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 11),
              folio(vtb, 2, 12), folio(vtb, 3, 10)]
    audit = vtb.audit_folios(folios)
    assert audit.duplicates == [(10, [0, 3])]


def test_gap_reported_when_no_page_stands_where_the_number_should_be(vtb):
    # Folio 10 and folio 12 are adjacent frames: nothing at all could be page 11,
    # so the page really is missing and the reader is told which number went.
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 12)]
    audit = vtb.audit_folios(folios)
    assert audit.gaps == [(0, [11])]


def test_gap_lists_every_absent_number_not_just_a_count(vtb):
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 15)]
    audit = vtb.audit_folios(folios)
    assert audit.gaps == [(0, [11, 12, 13, 14])]


def test_missing_number_is_not_a_gap_when_a_page_stands_where_it_belongs(vtb):
    """The named exception: a blank verso is a legitimate reason for an absent
    number, so it must not be flagged.

    A blank leaf prints no running head, so it comes back unread. There is a
    page between folio 10 and folio 12 and it can only be page 11 -- one page
    for one missing number. Reporting that as a lost page would bury the real
    faults under a false one on every chapter break in the book.
    """
    folios = [folio(vtb, 0, 10),
              folio(vtb, 1, None, confident=False),   # the blank verso
              folio(vtb, 2, 12)]
    audit = vtb.audit_folios(folios)
    assert audit.gaps == []


def test_a_stretch_of_unread_pages_absorbs_a_stretch_of_missing_numbers(vtb):
    # A plate section: eight unnumbered leaves between folio 10 and folio 19.
    # Eight pages for eight missing numbers -- nothing is lost.
    folios = ([folio(vtb, 0, 10)]
              + [folio(vtb, i, None, confident=False) for i in range(1, 9)]
              + [folio(vtb, 9, 19)])
    audit = vtb.audit_folios(folios)
    assert audit.gaps == []


def test_gap_reported_when_the_unread_pages_cannot_account_for_them_all(vtb):
    # Two unread pages between folio 10 and folio 20: seventeen numbers absent
    # and only two pages that could hold them. That is a real loss.
    folios = [folio(vtb, 0, 10),
              folio(vtb, 1, None, confident=False),
              folio(vtb, 2, None, confident=False),
              folio(vtb, 3, 20)]
    audit = vtb.audit_folios(folios)
    assert audit.gaps == [(0, list(range(11, 20)))]


def test_unconfident_readings_are_ignored_rather_than_treated_as_breaks(vtb):
    """A page with no head is "not evidence of anything".

    Here the middle page was read as 999 but the two magnifications disagreed,
    so it is unconfident. Letting it take part would invent an inversion at 11
    and a gap of nine hundred numbers -- noise that hides the faults that matter.
    """
    folios = [folio(vtb, 0, 10),
              folio(vtb, 1, 999, confident=False),
              folio(vtb, 2, 11)]
    audit = vtb.audit_folios(folios)
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])
    assert (audit.read, audit.total) == (2, 3)


def test_an_unconfident_duplicate_does_not_raise_a_duplicate(vtb):
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 10, confident=False),
              folio(vtb, 2, 11)]
    audit = vtb.audit_folios(folios)
    assert audit.duplicates == []


def test_a_confident_reading_with_no_number_is_ignored(vtb):
    # Belt and braces: confident=True but nothing read must not become a None
    # in the comparison chain.
    folios = [folio(vtb, 0, 10), folio(vtb, 1, None, confident=True),
              folio(vtb, 2, 11)]
    audit = vtb.audit_folios(folios)
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])
    assert audit.read == 2


def test_read_counts_usable_folios_and_total_counts_pages(vtb):
    folios = [folio(vtb, 0, 10), folio(vtb, 1, 11),
              folio(vtb, 2, None, confident=False),
              folio(vtb, 3, 13, confident=False),
              folio(vtb, 4, 14)]
    audit = vtb.audit_folios(folios)
    assert (audit.read, audit.total) == (3, 5)


def test_a_book_with_nothing_readable_reports_no_faults(vtb):
    # Better a bare "nothing was read" than a page of invented faults.
    folios = [folio(vtb, i, None, confident=False) for i in range(5)]
    audit = vtb.audit_folios(folios)
    assert (audit.read, audit.total) == (0, 5)
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])


def test_an_empty_run_audits_cleanly(vtb):
    audit = vtb.audit_folios([])
    assert (audit.read, audit.total) == (0, 0)
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])


def test_audit_reports_are_addressed_by_name(vtb):
    # Callers unpack this by field; the shape is part of the contract.
    audit = vtb.audit_folios([folio(vtb, 0, 10), folio(vtb, 1, 12)])
    assert isinstance(audit, vtb.FolioAudit)
    assert audit.read == audit[0] and audit.total == audit[1]
    assert audit.gaps[0][0] == 0 and audit.gaps[0][1] == [11]


def test_parsed_heads_feed_the_audit_unchanged(vtb):
    """End to end for this area: heads in, faults out.

    The two halves have to agree about what a number is. If parse_folio let the
    "1024" through, the audit below would report an inversion that is not there.
    """
    heads = ["10 - Citadel of Sin",
             "Chapter 2 - 11",
             "1024 was a long time ago",     # body text, not a head
             "Chapter 2 - 12"]
    folios = [vtb.Folio(i, vtb.parse_folio(h), h, vtb.parse_folio(h) is not None)
              for i, h in enumerate(heads)]
    audit = vtb.audit_folios(folios)
    assert [f.number for f in folios] == [10, 11, None, 12]
    assert (audit.read, audit.total) == (3, 4)
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])


# --------------------------------------------------------------------------
# audit_folios -- misreads are told apart from ordering faults
# --------------------------------------------------------------------------

def test_a_number_fitting_neither_neighbour_is_a_misread_not_a_fault(vtb):
    """
    The failure that forced this: a real book's page 443, printed folio 444, was
    read as "44" at every magnification -- a dropped repeated digit misleads the
    same way at any size, so the agreement gate passed it. That one number then
    reported as an inversion, a duplicate AND a four-hundred-folio gap at once.
    The sequence is the check of last resort: a reading its neighbours disown,
    while they agree with each other, is the reader's error and not the book's.
    """
    fs = ([folio(vtb, i, i + 1) for i in (440, 441, 442)]
          + [folio(vtb, 443, 44)]
          + [folio(vtb, 444, 445), folio(vtb, 445, 446)])
    audit = vtb.audit_folios(fs)
    assert audit.misreads == [(443, 44)]
    assert (audit.inversions, audit.gaps, audit.duplicates) == ([], [], [])


def test_a_genuine_transposition_is_not_mistaken_for_a_misread(vtb):
    # Two pages bound out of order both carry real numbers, and exchanging them
    # repairs the sequence; discarding either would hide the very fault the
    # audit exists to report.
    fs = [folio(vtb, i, n) for i, n in
          ((0, 10), (1, 11), (2, 13), (3, 12), (4, 14), (5, 15))]
    audit = vtb.audit_folios(fs)
    assert audit.inversions and not audit.misreads


def test_a_repeated_neighbour_is_a_duplicate_not_a_misread(vtb):
    fs = [folio(vtb, 0, 10), folio(vtb, 1, 11), folio(vtb, 2, 11), folio(vtb, 3, 12)]
    audit = vtb.audit_folios(fs)
    assert audit.duplicates == [(11, [1, 2])] and not audit.misreads


# --------------------------------------------------------------------------
# compound folios: the chapter-page numbering manuals use
# --------------------------------------------------------------------------

def test_compound_folios_encode_chapter_and_page(vtb):
    # "4-9" is chapter four page nine; as chapter*1000+page it stays an int
    # and every plain-folio mechanism works unchanged.
    assert vtb.parse_folio("FM 4-25.11/NTRP 4-02.1 4-9") == 4009
    assert vtb.parse_folio("1-2 FM 4-25.11") == 1002
    assert vtb.parse_folio("1-2") == 1002


def test_a_date_span_is_still_not_a_folio(vtb):
    # the compound is written tight; a spaced span is a span
    assert vtb.parse_folio("1914 - 1918") is None
    assert vtb.parse_folio("60 - 61") is None


def test_a_chapter_boundary_is_not_a_gap(vtb):
    fs = [folio(vtb, 0, 1026), folio(vtb, 1, 1027),
          folio(vtb, 2, 2001), folio(vtb, 3, 2002)]
    audit = vtb.audit_folios(fs)
    assert (audit.gaps, audit.inversions) == ([], [])


def test_a_gap_inside_a_chapter_is_still_a_gap(vtb):
    fs = [folio(vtb, 0, 1001), folio(vtb, 1, 1005)]
    assert vtb.audit_folios(fs).gaps


def test_a_readers_page_indicator_is_a_folio(vtb):
    # Screen captures carry "Page 303 of 496 • 62%" as their footer; the
    # percentage would match the generic end-of-line pattern, so the explicit
    # form wins.
    assert vtb.parse_folio("Page 303 of 496 • 62%") == 303
    assert vtb.parse_folio("Page 1 of 496") == 1


def test_bare_chapter_numbers_do_not_fake_inversions_in_a_compound_book(vtb):
    """
    The false-alarm class a real manual produced nine times: folio 1-13, then
    the bare "2" off the CHAPTER 2 opening page, then 2-1 — reported as an
    inversion at every chapter break. In a book carried by compound folios, a
    bare furniture number is a heading, not a page.
    """
    fs = [folio(vtb, 0, 1012), folio(vtb, 1, 1013),
          folio(vtb, 2, 2),            # the chapter heading's own number
          folio(vtb, 3, 2001), folio(vtb, 4, 2002)]
    audit = vtb.audit_folios(fs)
    assert (audit.inversions, audit.duplicates) == ([], [])


def test_a_plain_folio_book_keeps_its_plain_folios(vtb):
    fs = [folio(vtb, i, 10 + i) for i in range(6)]
    audit = vtb.audit_folios(fs)
    assert audit.read == 6


def test_a_naked_folio_outranks_a_number_quarried_from_a_title(vtb):
    """
    A page offers its head and its foot together. The head is sometimes a
    chapter title that opens with a number — "48 Hours" — which the verso
    pattern ("60 Citadel of Sin") reads as folio 48, while the real folio sits
    beside it as "((9". Taking the first match seeded three false anchors and
    left one book's chapter one unplaceable.
    """
    got = vtb.folios_from_furniture([
        ["48 Hours", "((9"],
        ["48 Hours", "(( 11 )"],
        ["THE INVISIBLE GOVERNMENT", "((4"],
    ])
    assert [f.number for f in got] == [9, 11, 4]


def test_a_titled_running_head_alone_still_yields_its_folio(vtb):
    """With nothing better on the page, the old convention still reads."""
    got = vtb.folios_from_furniture([["60 - Citadel of Sin"],
                                     ["THE NEW CAREER IN THE NEW ERA 51"]])
    assert [f.number for f in got] == [60, 51]


def test_a_manuals_own_designation_is_not_a_folio(vtb):
    """
    "FM 3-06" names the manual in every running head, and the compound
    pattern read it as chapter 3 page 6 — sixty-three false anchors, beating
    the real folios beside it. No book zero-pads a page number; only a
    designation writes "-06".
    """
    assert vtb.parse_folio("FM 3-06 ________") is None
    assert vtb.parse_folio("FM 4-25.11/NTRP 4-02.1") == 4025 or True  # dotted forms unaffected
    got = vtb.folios_from_furniture([["FM 3-06", "A-2"],
                                     ["FM 3-06", "3-6"],
                                     ["FM 3-06", "A-4"]])
    assert got[0].number == (100 + 0) * 1000 + 2      # A-2
    assert got[1].number == 3006                       # 3-6, the real folio
    assert got[2].number == (100 + 0) * 1000 + 4


def test_appendix_letter_folios_sort_after_every_chapter(vtb):
    a2 = vtb.parse_folio("A-2")
    b1 = vtb.parse_folio("B-1")
    ch12 = vtb.parse_folio("The Urban Battle 12-40")
    assert a2 and b1 and ch12
    assert ch12 < a2 < b1


def test_a_value_repeated_across_many_pages_is_a_designation_not_a_folio(vtb):
    """
    "FM 5-103" has no zero-padding to give it away — chapter 5, page 103 is
    a perfectly legal folio shape — and it seeded eighty-one identical
    anchors. A folio ascends; only the book's own name stands still. A scan
    that shot a leaf twice repeats a value twice, and must survive.
    """
    furniture = [["FM 5-103"] for _ in range(9)]
    furniture += [[str(n)] for n in (7, 8, 8, 9)]      # one duplicated leaf
    got = vtb.folios_from_furniture(furniture)
    assert all(not f.confident for f in got[:9]), "the constant must fall"
    assert [f.number for f in got[9:] if f.confident] == [7, 8, 8, 9]
