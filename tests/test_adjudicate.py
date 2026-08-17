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
    """The third element is the witness-token index — geometry's handle."""
    ours = adj.tokens("the prisoners were moved")
    theirs = adj.tokens("the prisoncrs were moved")
    pairs = adj.align_disputes(ours, theirs)
    assert pairs == [("prisoners", "prisoncrs", 1)]


def test_wade_giles_survives_the_lexicon_path(vtb):
    lex = adj.build_lexicon([["chü", "chingnien"]] * 4)
    v = adj.adjudicate_pair("ch'ü", "ch'ii", lex)
    assert v["rung"] == "lexicon" and v["winner"] == "ch'ü"


def test_the_apparatus_is_lexical(vtb):
    """Eagle measured it: 92 of 105 verdicts against the shipped reading
    were the dictionary crowning 'f' over 'ff' (242ff.) and 'of' over 'op'
    (op. cit.) — fluency standardizing away a scholarly book's own
    apparatus. Apparatus tokens are words; those disputes now abstain and
    fall through to rungs that look at ink."""
    v = adj.adjudicate_pair("ff", "f", Counter())
    assert v["rung"] == "abstain"
    v = adj.adjudicate_pair("op", "of", Counter())
    assert v["rung"] == "abstain"


def test_single_letters_are_not_words_except_a_i_o(vtb):
    """The system list contains all 26 letters; running English does not."""
    assert not adj._is_word("x")
    assert adj._is_word("a") and adj._is_word("i") and adj._is_word("o")
    v = adj.adjudicate_pair("ax", "x", Counter())
    assert v["rung"] == "dictionary" and v["winner"] == "ax"


def _tsv(rows):
    head = "\t".join(["level"] * 12)
    return head + "\n" + "\n".join(
        "\t".join(map(str, [5, b, p, ln, wn, 0, x, y, w, h, 96, t]))
        for b, p, ln, wn, x, y, w, h, t in rows)


def test_tsv_tokens_carry_normalized_boxes(vtb):
    tsv = _tsv([(1, 1, 1, 1, 100, 50, 80, 20, "the"),
                (1, 1, 1, 2, 200, 50, 120, 20, "prisoners")])
    toks, boxes = adj._tsv_tokens(tsv, 1000, 1000)
    assert toks == ["the", "prisoners"]
    assert boxes[1] == [0.2, 0.05, 0.32, 0.07]


def test_tsv_hyphen_join_keeps_first_fragments_box(vtb):
    """The joined token must exist (fold_text parity) and its crop must
    show the line end where the dispute actually sits."""
    tsv = _tsv([(1, 1, 1, 1, 500, 50, 90, 20, "crys-"),
                (1, 1, 2, 1, 100, 90, 110, 20, "tallized"),
                (1, 1, 2, 2, 250, 90, 70, 20, "plan")])
    toks, boxes = adj._tsv_tokens(tsv, 1000, 1000)
    assert toks == ["crystallized", "plan"]
    assert boxes[0][0] == 0.5, "geometry points at the crys- fragment"


def test_tsv_tokens_match_the_plain_tokenizer(vtb):
    """Same words in, same token stream out — alignment depends on it."""
    tsv = _tsv([(1, 1, 1, 1, 0, 0, 9, 9, "don’t"),
                (1, 1, 1, 2, 20, 0, 9, 9, "stop—ever"),
                (1, 1, 1, 3, 40, 0, 9, 9, "242ff.")])
    toks, boxes = adj._tsv_tokens(tsv, 100, 100)
    assert toks == adj.tokens("don’t stop—ever 242ff.")
    assert len(toks) == len(boxes)
