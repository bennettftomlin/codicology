"""The hole check must not call a re-scanned leaf a loss — and must not use
that as an excuse for anything else.

TiHKAL's scanner shot the xxii/xxiii spread twice. Dropping the second pair
was right, but by page index those pages are simply gone, and the check read
them as two holes and halted a batch of eighteen books over content that was
never lost. A gate that cries wolf on correct behavior gets ignored, which
costs more than the false alarm.

The exemption is deliberately narrow and deliberately loud: it needs three
quarters of the page to be present elsewhere, and it prints which pages took
it. The tests that matter most here are the ones in the other direction —
that a genuinely lost page is still a hole, however much its subject matter
resembles the rest of the book.
"""
from codicology import verify


def words(s):
    return s.split()


def test_a_page_kept_elsewhere_is_recognised(vtb):
    page = words("the horrors of the Inquisition with its lethal intolerance "
                 "of dissent called heresy are well documented and yet it was "
                 "during those dark years that the structure of alchemy was "
                 "established")
    kept = [verify.shingles(page)]
    assert verify.covered_elsewhere(page, kept) == 1.0


def test_a_rescan_with_a_few_misread_words_still_counts(vtb):
    """One scan reads "intrinsic", the other "intiinsic" — the same leaf."""
    a = words("there is no intrinsic good or evil in the objective world of "
              "academic scientific inquiry and of course that there is no "
              "meaning to the idea of a need for maintaining some sort of "
              "balance but still I would like to illustrate some rather "
              "incredible coincidences of timing")
    b = list(a)
    b[3] = "intiinsic"
    b[20] = "mainlaining"
    got = verify.covered_elsewhere(b, [verify.shingles(a)])
    assert got >= verify.RESCAN_COVERAGE, got


def test_a_genuinely_lost_page_is_still_a_hole(vtb):
    """Different content, same book, same vocabulary. This is the case the
    whole check exists for and no exemption may swallow it."""
    kept = [verify.shingles(words(
        "the horrors of the Inquisition with its lethal intolerance of "
        "dissent called heresy are well documented during those dark years"))]
    lost = words("phenethylamine dopamine and a tryptamine serotonin thus "
                 "there is encouragement to the neuroscientists to search for "
                 "some neurotransmitter mismanagement using the psychedelic "
                 "drug as chemically related probes")
    assert verify.covered_elsewhere(lost, kept) < verify.RESCAN_COVERAGE


def test_shared_phrasing_is_not_enough(vtb):
    """Two pages of a chemistry reference share a template — the failure that
    once deleted TiHKAL's compound #20 at 0.59 similarity. The bar sits well
    above anything a shared template produces."""
    kept = [verify.shingles(words(
        "SYNTHESIS a solution of 3 g of the aldehyde in 20 mL of glacial "
        "acetic acid was treated with 4 g of nitroethane and heated for 2 h"))]
    other = words("SYNTHESIS a solution of 5 g of the ketone in 30 mL of "
                  "glacial acetic acid was treated with 6 g of nitropropane "
                  "and stirred for 8 h")
    assert verify.covered_elsewhere(other, kept) < verify.RESCAN_COVERAGE


def test_an_empty_book_offers_no_cover(vtb):
    assert verify.covered_elsewhere(words("some words here at all"), []) == 0.0


def test_an_empty_page_claims_no_coverage(vtb):
    assert verify.covered_elsewhere([], [verify.shingles(words("a b c d e f g"))]) == 0.0


def test_coverage_is_measured_against_the_absent_page_not_the_book(vtb):
    """A short page wholly contained in a long one is fully covered; the long
    page is not covered by the short one. The direction matters — otherwise a
    lost page would be excused by any large page that happens to contain a
    similar run."""
    short = words("a solution of the aldehyde in glacial acetic acid was "
                  "treated with nitroethane and heated")
    long = short + words("and then cooled filtered washed with water dried "
                         "over magnesium sulfate and distilled under reduced "
                         "pressure to give a pale yellow oil which solidified")
    assert verify.covered_elsewhere(short, [verify.shingles(long)]) == 1.0
    assert verify.covered_elsewhere(long, [verify.shingles(short)]) < 1.0
