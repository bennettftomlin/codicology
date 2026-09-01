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
    """Real TSV column order: level, page, block, par, line, word, then
    geometry, conf, text — the line key is (page, block, par, line)."""
    head = "\t".join(["level"] * 12)
    return head + "\n" + "\n".join(
        "\t".join(map(str, [5, 1, b, p, ln, wn, x, y, w, h, 96, t]))
        for b, p, ln, wn, x, y, w, h, t in rows)


def test_tsv_tokens_carry_normalized_boxes(vtb):
    tsv = _tsv([(1, 1, 1, 1, 100, 50, 80, 20, "the"),
                (1, 1, 1, 2, 200, 50, 120, 20, "prisoners")])
    toks, boxes, _ = adj._tsv_tokens(tsv, 1000, 1000)
    assert toks == ["the", "prisoners"]
    assert boxes[1] == [0.2, 0.05, 0.32, 0.07]


def test_tsv_hyphen_join_unions_both_fragments(vtb):
    """A crop that ends at "crys-" asks the reviewer to guess; the joined
    token's box spans both fragments so both lines land in the crop."""
    tsv = _tsv([(1, 1, 1, 1, 500, 50, 90, 20, "crys-"),
                (1, 1, 2, 1, 100, 90, 110, 20, "tallized"),
                (1, 1, 2, 2, 250, 90, 70, 20, "plan")])
    toks, boxes, _ = adj._tsv_tokens(tsv, 1000, 1000)
    assert toks == ["crystallized", "plan"]
    assert boxes[0] == [0.1, 0.05, 0.59, 0.11], \
        "the union spans line end to next line's fragment"


def test_line_edge_tokens_pull_context_across_the_wrap(vtb):
    """request-whether at a line end reads as a cliff without the next
    line's head; a line-initial word without the previous line's tail is
    a phrase missing its start. cbox carries the neighbor along."""
    tsv = _tsv([(1, 1, 1, 1, 100, 50, 80, 20, "she"),
                (1, 1, 1, 2, 200, 50, 120, 20, "request"),
                (1, 1, 2, 1, 100, 90, 90, 20, "whether"),
                (1, 1, 2, 2, 210, 90, 70, 20, "any")])
    toks, boxes, cboxes = adj._tsv_tokens(tsv, 1000, 1000)
    assert toks == ["she", "request", "whether", "any"]
    # "request" ends line 1: its crop box reaches down to "whether"
    assert cboxes[1] == [0.1, 0.05, 0.32, 0.11]
    # "whether" opens line 2: its crop box reaches up to "request"
    assert cboxes[2] == [0.1, 0.05, 0.32, 0.11]
    # mid-line tokens keep their own box
    assert cboxes[0] == boxes[0]


def test_tsv_tokens_match_the_plain_tokenizer(vtb):
    """Same words in, same token stream out — alignment depends on it."""
    tsv = _tsv([(1, 1, 1, 1, 0, 0, 9, 9, "don’t"),
                (1, 1, 1, 2, 20, 0, 9, 9, "stop—ever"),
                (1, 1, 1, 3, 40, 0, 9, 9, "242ff.")])
    toks, boxes, cboxes = adj._tsv_tokens(tsv, 100, 100)
    assert toks == adj.tokens("don’t stop—ever 242ff.")
    assert len(toks) == len(boxes) == len(cboxes)


def test_edge_context_refuses_distant_neighbors(vtb):
    """The entry after a paragraph's last line can be a footnote at the
    page foot; a crop spanning the page helps nobody."""
    tsv = _tsv([(1, 1, 1, 1, 100, 50, 80, 20, "paragraph"),
                (1, 1, 1, 2, 200, 50, 90, 20, "ends"),
                (2, 1, 1, 1, 100, 900, 70, 20, "footnote")])
    toks, boxes, cboxes = adj._tsv_tokens(tsv, 1000, 1000)
    assert toks == ["paragraph", "ends", "footnote"]
    assert cboxes[1] == boxes[1], "the footnote is not this line's wrap"


def test_a_numeral_cannot_be_outvoted(vtb):
    """Eagle's contents page: the lexicon crowned 'a' over XXXII, because
    common words outnumber every numeral. Numerals are apparatus; only
    ink evidence or a human settles a dispute touching one."""
    lex = Counter({"a": 500})
    v = adj.adjudicate_pair("XXXII", "a", lex)
    assert v["rung"] == "abstain" and v.get("note") == "roman numeral"
    v = adj.adjudicate_pair("xxxii", "xxxti", Counter())
    assert v["rung"] == "abstain", "front-matter folios count too"


def test_roman_syntax_keeps_real_words_out(vtb):
    assert adj._is_roman("XXXII") and adj._is_roman("xxxii")
    assert adj._is_roman("XVI.") , "contents entries carry punctuation"
    assert not adj._is_roman("CIVIL"), "every letter Roman, grammar not"
    assert not adj._is_roman("Mix"), "mixed case is prose"
    assert not adj._is_roman("I"), "one letter stays a pronoun"
    v = adj.adjudicate_pair("civil", "civii", Counter())
    assert v["rung"] == "dictionary" and v["winner"] == "civil"


def _wpage(*toks):
    return [list(toks), [[0, 0, 0.1, 0.1]] * len(toks),
            [[0, 0, 0.1, 0.1]] * len(toks)]


def test_witness_stitches_where_the_book_joined(vtb):
    """The builder shipped 'consulate' whole across a page turn; the
    witness, whose world ends at the page edge, read 'con-'. Stitched,
    the pages fold silent instead of crowning the fragment."""
    pages = {0: _wpage("the", "con-"), 1: _wpage("sulate", "closed")}
    n = adj._stitch_page_turns(pages, {0: "consulate", 1: "closed"})
    assert n == 1
    assert pages[0][0] == ["the", "consulate"]
    assert pages[1][0] == ["closed"]
    assert len(pages[1][1]) == 1 and len(pages[1][2]) == 1


def test_witness_keeps_fragments_where_the_book_refused(vtb):
    """self- | control failed the builder's gate and shipped as printed;
    the stitch condition fails symmetrically and both pages stay as
    read."""
    pages = {0: _wpage("of", "self-"), 1: _wpage("control", "it")}
    n = adj._stitch_page_turns(pages, {0: "self", 1: "it"})
    assert n == 0
    assert pages[0][0] == ["of", "self-"]
    assert pages[1][0] == ["control", "it"]


def test_stitch_reaches_across_a_dropped_blank(vtb):
    """The builder joins across dropped blanks; the witness's empty page
    is skipped and the fold-mirror condition stays the gate."""
    pages = {0: _wpage("con-"), 1: _wpage(), 2: _wpage("sulate", "was")}
    assert adj._stitch_page_turns(pages, {0: "consulate"}) == 1
    assert pages[0][0] == ["consulate"]
    assert pages[2][0] == ["was"]


def test_stitch_refuses_when_the_mirror_fails(vtb):
    """A gap plus a non-matching join: the fragment stays a fragment."""
    pages = {0: _wpage("con-"), 5: _wpage("gress")}
    assert adj._stitch_page_turns(pages, {0: "consulate"}) == 0
    assert pages[0][0] == ["con-"]


def test_stitch_reads_past_the_running_head(vtb):
    """The witness reads furniture too: the next page opens RUSSIAN PURGE
    before the continuation. The completing token is sought among the
    first few; nothing coincidental can complete the shipped word."""
    pages = {0: _wpage("the", "con-"),
             1: _wpage("RUSSIAN", "PURGE", "sulate", "was")}
    n = adj._stitch_page_turns(pages, {0: "consulate"})
    assert n == 1
    assert pages[0][0] == ["the", "consulate"]
    assert pages[1][0] == ["RUSSIAN", "PURGE", "was"]
    assert len(pages[1][1]) == 3


def test_a_seam_is_not_a_list_of_disputes(vtb):
    """When tesseract's segmentation walks a page's blocks in a different
    order than surya's, the streams re-sync at seams, and the alignment
    pairs words that were never looking at the same ink. Citadel of Sin
    produced runs of three to six such pairs — "nothing" against "a", "to"
    against "geere" — where a book both engines take in the same order
    never exceeds two."""
    ours = adj.tokens("alpha beta wanted nothing to gamma delta")
    theirs = adj.tokens("alpha beta wa a geere gamma delta")
    stats = Counter()
    assert adj.align_disputes(ours, theirs, stats) == []
    assert stats["seam"] == 3


def test_damage_survives_the_seam_rule(vtb):
    """A torn line garbles several words in a row and those ARE disputes:
    inside a suspect run, readings that could still be the same ink stay."""
    ours = adj.tokens("alpha the quick brown fox omega")
    theirs = adj.tokens("alpha thc qmck browu fux omega")
    stats = Counter()
    pairs = adj.align_disputes(ours, theirs, stats)
    assert [a for a, _, _ in pairs] == ["the", "quick", "brown", "fox"]
    assert stats["seam"] == 0


def test_short_blocks_are_never_second_guessed(vtb):
    """One or two words disagreeing amid agreement is the ordinary shape of
    a misread, and eagle's 'A' against 'Rasy' — a drop cap — must survive
    even sharing no letters."""
    ours = adj.tokens("chapter A the strike")
    theirs = adj.tokens("chapter Rasy the strike")
    stats = Counter()
    pairs = adj.align_disputes(ours, theirs, stats)
    assert pairs == [("A", "Rasy", 1)]
    assert stats["seam"] == 0


def test_a_displaced_word_is_not_a_dispute(vtb):
    """The seam's other half, in a block too short for the run rule: the
    witness offers a word surya read elsewhere on this very page, looking
    nothing like what surya read here. It was displaced, not misread."""
    ours = adj.tokens("although the well-cultured man spoke although again")
    theirs = adj.tokens("although the although man spoke although again")
    stats = Counter()
    assert adj.align_disputes(ours, theirs, stats) == []
    assert stats["seam"] == 1


def test_a_truncation_survives_a_page_that_repeats_the_word(vtb):
    """'be' against 'been' rhymes, so it is judged on the ink and kept even
    though the page says 'been' elsewhere."""
    ours = adj.tokens("it had been said he would be there")
    theirs = adj.tokens("it had been said he would been there")
    stats = Counter()
    pairs = adj.align_disputes(ours, theirs, stats)
    assert pairs == [("be", "been", 6)]
    assert stats["seam"] == 0


def test_a_folded_count_cannot_settle_a_case_shape_disagreement(vtb):
    """As If Already Free p120 printed "artificial intelligence (AI)". The
    book's twelve 'et al.' citations attested 'al' twelve times and 'ai'
    never, so the lexicon crowned the witness's 'Al' — a lowercase
    abbreviation vouching for a title-case word it never appears as.
    Capital-I and lowercase-l are the same ink in a serif face; the ink
    and the reader settle this, not a case-blind tally."""
    v = adj.adjudicate_pair("AI", "Al", Counter({"al": 12}))
    assert v["winner"] is None and v["rung"] == "abstain"


def test_the_apparatus_still_wins_on_its_own_attestation(vtb):
    """The guard must not cost the 214 verdicts the apparatus fix earned:
    242ff. and its kin are lowercase on both sides, so the lexicon rung
    keeps ruling exactly as measured."""
    v = adj.adjudicate_pair("ff", "f", Counter({"ff": 5}))
    assert v == {"rung": "lexicon", "winner": "ff", "count": 5}


def test_two_acronyms_are_still_settled_by_the_book(vtb):
    """The guard fires on DISAGREEMENT about case shape, not on capitals:
    when both readers see an acronym the fold is comparing like with like
    and the book's own counts remain evidence."""
    v = adj.adjudicate_pair("USA", "USB", Counter({"usa": 9}))
    assert v["winner"] == "USA" and v["rung"] == "lexicon"


def test_surya_only_runs_finds_the_unmatched_sentence(vtb):
    """Words tesseract has no trace of, contiguous, in reading order —
    the aligner's discarded insertions, recovered as testimony."""
    from codicology.adjudicate import surya_only_runs
    s = ("The morning was cold . Invented words nobody witnessed here "
         "at all . The evening was warm .").split()
    t = "The morning was cold . The evening was warm .".split()
    frac, only, runs = surya_only_runs(s, t)
    assert only == 6 and len(runs) == 1
    words, n_only = runs[0]
    assert n_only == 6
    # the span is VERBATIM — interior short words ride along, so the text
    # is findable in the layer and deletable from the body
    assert words == ["Invented", "words", "nobody", "witnessed",
                     "here", "at", "all"]


def test_surya_only_ignores_disputed_words(vtb):
    """A word tesseract read DIFFERENTLY is a dispute, not an invention —
    the ladder already judges it."""
    from codicology.adjudicate import surya_only_runs
    s = "alpha beta gamma delta epsilon".split()
    t = "alpha bete gamma delta epsilon".split()
    frac, only, runs = surya_only_runs(s, t, skip_folds={"beta"})
    assert only == 0 and runs == []


def test_short_scatter_is_not_a_run(vtb):
    from codicology.adjudicate import surya_only_runs
    s = "one stray word here another there and more filler words done".split()
    t = "one word here another there and more filler words done".split()
    frac, only, runs = surya_only_runs(s, t)
    assert runs == []          # single unmatched word, below the run floor


def test_page_similarity_is_overlap_over_smaller(vtb):
    from codicology.adjudicate import _page_similarity
    a = {"river", "channel", "dredge", "engineer"}
    b = {"river", "channel", "dredge"}
    assert _page_similarity(a, b) == 1.0          # b fully inside a
    assert _page_similarity(a, {"unrelated"}) == 0.0
    assert _page_similarity(set(), b) == 0.0


def test_run_verdict_bands(vtb):
    """0.8 greenlights, 0.3 sends word differences to the ladder, below
    it the crop shows unrelated ink; an empty read is silence, which ink
    must arbitrate; no crop at all stays advisory."""
    from codicology.adjudicate import _run_verdict
    run = "channel improvement required constant dredging work".split()
    assert _run_verdict(run, run) == "confirmed"
    assert _run_verdict(run, "channel improvement required constant "
                             "dredgmg work".split()) == "confirmed"
    assert _run_verdict(run, "channel improvement noise other".split()) \
        == "located"
    assert _run_verdict(run, "totally unrelated words here".split()) \
        == "advisory"
    assert _run_verdict(run, []) == "silent"
    assert _run_verdict(run, None) == "advisory"
