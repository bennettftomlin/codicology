"""The adjudication ladder, rung by rung — and its refusals.

The measured facts these tests encode: hyphenation and diacritics are policy,
not disputes (77 of 83 apparent disagreements on Eagle); the book's own
vocabulary adjudicates coinages an external dictionary has never heard of;
a non-word loses TO A WORD and two non-words abstain; and the record ships
the shipped reading regardless — adjudication reports, it never rewrites.
"""
from collections import Counter

from codicology import adjudicate as adj


def test_hyphenation_is_policy_not_dispute(vtb):
    ours = adj.tokens("The crys-\ntallized plan improved.")
    theirs = adj.tokens("The crystallized plan improved.")
    assert adj.align_disputes(ours, theirs) == []


def test_diacritics_fold_for_comparison_only(vtb):
    v = adj.adjudicate_pair("reëntry", "reentry", Counter())
    assert v["rung"] == "fold"
    assert v["winner"] == "reëntry", "the faithful form must ship"


def test_the_lexicon_rescues_a_coinage(vtb):
    lex = Counter({"kvothe": 212})
    v = adj.adjudicate_pair("Kvothe", "Kvo1he", lex)
    # "Kvo1he" carries a digit — not even a WORD token — but test the pure
    # letter case too:
    v2 = adj.adjudicate_pair("Kvothe", "Kvothc", lex)
    assert v2["rung"] == "lexicon" and v2["winner"] == "Kvothe"


def test_the_lexicon_works_in_both_directions(vtb):
    lex = Counter({"halsted": 14})
    v = adj.adjudicate_pair("Haisted", "Halsted", lex)
    assert v["rung"] == "lexicon" and v["winner"] == "Halsted"


def test_dictionary_defeats_a_non_word_both_ways(vtb):
    lex = Counter()
    v = adj.adjudicate_pair("prisoners", "prisoncrs", lex)
    assert v["rung"] == "dictionary" and v["winner"] == "prisoners"
    v = adj.adjudicate_pair("conscented", "consented", lex)
    assert v["rung"] == "dictionary" and v["winner"] == "consented"


def test_two_words_abstain_without_deeper_evidence(vtb):
    v = adj.adjudicate_pair("property", "properly", Counter())
    assert v["rung"] == "abstain" and v["winner"] is None


def test_two_non_words_abstain_never_coin_flip(vtb):
    v = adj.adjudicate_pair("mcisaac", "mcisaas", Counter())
    assert v["rung"] == "abstain"


def test_a_hapax_coinage_gets_no_lexicon_protection(vtb):
    lex = adj.build_lexicon([["kvothe"], ["denna"], ["denna"], ["denna"]])
    assert "denna" in lex and "kvothe" not in lex


def test_alignment_ignores_furniture_asymmetry(vtb):
    """Insertions and deletions are layout differences — one reader kept the
    running head, the other stripped it — never character disputes."""
    ours = adj.tokens("The strike ended in July.")
    theirs = adj.tokens("EAGLE FORGOTTEN The strike ended in July.")
    assert adj.align_disputes(ours, theirs) == []


def test_alignment_finds_the_real_replacement(vtb):
    ours = adj.tokens("the prisoners were moved")
    theirs = adj.tokens("the prisoncrs were moved")
    pairs = adj.align_disputes(ours, theirs)
    assert pairs == [("prisoners", "prisoncrs")]


def test_wade_giles_survives_the_lexicon_path(vtb):
    lex = adj.build_lexicon([["chü", "chingnien"]] * 4)
    v = adj.adjudicate_pair("ch'ü", "ch'ii", lex)
    assert v["rung"] == "lexicon" and v["winner"] == "ch'ü"
