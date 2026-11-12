"""Native-text reconciliation: on born-digital pages the layer IS the page.

The substitutions must be surgical — isolated aligned words only — and the
authority must be scoped: a scan's layer (another OCR's guess) never
substitutes, and loosely aligned regions stay ours however wrong they look.
"""
import os

import pytest


def _native_page(tmp_path, layer):
    p = str(tmp_path / "page_0000.jpg")
    open(p, "w").close()
    with open(p + ".layer.txt", "w") as fh:
        fh.write(layer)
    open(p + ".native", "w").close()
    return p


def test_an_isolated_misread_defers_to_the_native_text(vtb, tmp_path):
    p = _native_page(tmp_path,
                     "The joint copublication was announced by the press "
                     "office later that same afternoon in the city.")
    bodies = ["<p>The joint copulication was announced by the press "
              "office later that same afternoon in the city.</p>"]
    n = vtb.reconcile_native_text(bodies, [p])
    assert n == 1
    assert "copublication" in bodies[0] and "copulication" not in bodies[0]
    assert bodies[0].startswith("<p>") and bodies[0].endswith("</p>")


def test_the_books_own_typo_is_restored_not_corrected(vtb, tmp_path):
    """Surya silently editing the author is still an error of fidelity."""
    p = _native_page(tmp_path,
                     "National competiveness was the subject of the annual "
                     "report released by the ministry that spring season.")
    bodies = ["<p>National competitiveness was the subject of the annual "
              "report released by the ministry that spring season.</p>"]
    assert vtb.reconcile_native_text(bodies, [p]) == 1
    assert "competiveness" in bodies[0]


def test_a_scan_layer_never_substitutes(vtb, tmp_path):
    """Same disagreement, no .native marker: the layer is another OCR's
    guess (a scan), and ours stays."""
    p = str(tmp_path / "page_0000.jpg")
    open(p, "w").close()
    with open(p + ".layer.txt", "w") as fh:
        fh.write("The shrew Frenchman assured him that he had missed his "
                 "way entirely and turned back toward the fort again.")
    bodies = ["<p>The shrewd Frenchman assured him that he had missed his "
              "way entirely and turned back toward the fort again.</p>"]
    assert vtb.reconcile_native_text(bodies, [p]) == 0
    assert "shrewd" in bodies[0]


def test_hyphen_splits_in_the_layer_do_not_break_rejoined_words(vtb, tmp_path):
    """The layer keeps line breaks ("de- veloped"); ours rejoins. A 1:2
    alignment is not a substitution."""
    p = _native_page(tmp_path,
                     "The region de- veloped rapidly after the war ended "
                     "and the new roads finally reached the interior towns.")
    bodies = ["<p>The region developed rapidly after the war ended "
              "and the new roads finally reached the interior towns.</p>"]
    assert vtb.reconcile_native_text(bodies, [p]) == 0
    assert "developed" in bodies[0]


def test_a_restructured_table_region_stays_ours(vtb, tmp_path):
    """Where surya rebuilt a table and alignment is loose, nothing moves."""
    p = _native_page(tmp_path,
                     "Price Quantity 10 200 20 150 30 100 40 50 the schedule "
                     "shows demand falling as the price rises steadily")
    bodies = ["<table><tr><td>Quantity demanded</td><td>at each price "
              "point</td></tr><tr><td>two hundred</td><td>one fifty</td>"
              "</tr></table>"]
    before = bodies[0]
    vtb.reconcile_native_text(bodies, [p])
    assert bodies[0] == before


def test_substituted_words_are_html_escaped(vtb, tmp_path):
    p = _native_page(tmp_path,
                     "Profits at Smith & Sons doubled after the merger was "
                     "approved by the board and the shareholders that year.")
    bodies = ["<p>Profits at Smith 8 Sons doubled after the merger was "
              "approved by the board and the shareholders that year.</p>"]
    assert vtb.reconcile_native_text(bodies, [p]) == 1
    assert "&amp;" in bodies[0]


def test_layer_words_with_control_characters_never_substitute(vtb, tmp_path):
    """MIT's layer writes "Cana\x02da" at line breaks. Authority does not
    extend to garbage: the clean reading we made stays."""
    p = _native_page(tmp_path,
                     "The Cana\x02da study found that academic sharing was "
                     "widespread among the students at every large campus.")
    bodies = ["<p>The Canada study found that academic sharing was "
              "widespread among the students at every large campus.</p>"]
    assert vtb.reconcile_native_text(bodies, [p]) == 0
    assert "Canada" in bodies[0] and "\x02" not in bodies[0]


def test_a_born_digital_page_is_never_dropped_as_a_duplicate(vtb, tmp_path):
    """
    Macro p471: two consecutive problem-set pages sharing their template
    ("Suppose the multiplier is 1.5 and the economy's real GDP is $5,000
    billion...") scored 0.569 and one was dropped with 290 words on it.
    A publisher's page is a page; only a capture can be a duplicate.
    """
    from codicology import pipeline as v
    shared = ("Suppose the multiplier is 1.5 and the economy's real GDP is "
              "5,000 billion. In which direction will the aggregate demand "
              "curve shift and by how much? Explain using a graph why the "
              "change in real GDP is likely to be smaller than the shift in "
              "the aggregate demand curve. ")
    a = "Suppose a country decreases government purchases by 100 billion. " + shared
    b = "Suppose a country repeals an investment tax credit. " + shared * 2
    verdicts = v.review_pages([a, b])
    assert any(x.status == "duplicate" for x in verdicts), \
        "fixture must actually trip the detector for this test to mean anything"
    # with native markers present, the build must keep both
    p0 = str(tmp_path / "page_0000.jpg"); open(p0, "w").close()
    p1 = str(tmp_path / "page_0001.jpg"); open(p1, "w").close()
    for p in (p0, p1):
        open(p + ".native", "w").close()
    import os
    native = {i for i, p in enumerate([p0, p1]) if os.path.exists(p + ".native")}
    kept = {x.index for x in verdicts if x.status == "duplicate"} - native
    assert kept == set(), "no born-digital page may be dropped"


def test_a_pdf_needs_near_identity_before_a_page_is_deleted(vtb):
    """
    A chemical reference lost compound #20 because its entry is laid out like
    compound #24's — 0.59 similar, and deleted. A PDF that really does hold a
    page twice holds it identically (measured 1.00 on the same book's
    duplicated leaves), so near-identity is the honest bar there; a camera
    shooting one page twice yields two readings and needs the looser one.
    """
    entry20 = ("#20. 4-HO-DPT; TRYPTAMINE, N,N-DIPROPYL-4-HYDROXY. DOSAGE: "
               "unknown. DURATION: unknown. QUALITATIVE COMMENTS: the synthesis "
               "follows the general method described for the parent compound "
               "and the product was obtained as an off-white solid. ")
    entry24 = ("#24. 4-HO-pyr-T; TRYPTAMINE, 4-HYDROXY-N,N-TETRAMETHYLENE. "
               "DOSAGE: unknown. DURATION: unknown. QUALITATIVE COMMENTS: the "
               "synthesis follows the general method described for the parent "
               "compound and the product was obtained as an off-white solid. ")
    v = vtb.review_pages([entry20, entry24])
    dup = [x for x in v if x.status == "duplicate"]
    assert dup, "fixture must trip the detector at the camera's bar"
    assert dup[0].score < 0.90, (
        f"score {dup[0].score:.2f}: a PDF bar of 0.90 must spare this page")
    same = (" ".join(f"In the {n}th year the survey party followed the ridge "
                     f"north of the creek and camped near the ford" 
                     for n in range(14)))
    v2 = vtb.review_pages([same, same])
    d2 = [x for x in v2 if x.status == "duplicate"]
    assert d2 and d2[0].score >= 0.90, (
        f"a genuinely duplicated leaf must still be caught: {d2}")


def test_a_page_covered_by_several_images_is_a_scan(vtb):
    """Whether a picture of the whole page sits behind the text is a question
    about coverage, and a scan need not answer it with a single image: stored
    as two tiles or as MRC layers, its largest piece can sit well under the
    threshold while the page is completely covered. One book's cover measured
    0.71 as its largest image and 1.0 as the union, and was being handed to
    the reconciler as a publisher's typesetting."""
    halves = [(0, 0, 10, 5), (0, 5, 10, 10)]
    assert max((b[2] - b[0]) * (b[3] - b[1]) for b in halves) / 100 < 0.9
    assert vtb._union_fraction(halves, 10, 10) >= 0.9


def test_overlapping_images_are_counted_once(vtb):
    """Summing the areas would make two overlapping halves cover the page
    twice, and would call any busy born-digital page a scan."""
    assert vtb._union_fraction([(0, 0, 10, 6), (0, 4, 10, 10)], 10, 10) == 1.0
    assert vtb._union_fraction([(0, 0, 4, 4), (0, 0, 4, 4)], 10, 10) == 0.16


def test_a_plate_with_room_for_its_caption_is_still_born_digital(vtb):
    """The threshold's original job: a photographic plate covers at most about
    nine tenths of its page, and the page around it is the publisher's own
    text."""
    assert vtb._union_fraction([(0, 0, 9, 9)], 10, 10) < 0.9


def test_an_image_hanging_over_the_edge_cannot_over_cover(vtb):
    assert vtb._union_fraction([(-5, -5, 15, 15)], 10, 10) == 1.0


def test_a_page_with_no_images_is_covered_by_nothing(vtb):
    assert vtb._union_fraction([], 10, 10) == 0.0


def _native(tmp_path, body, layer):
    page = tmp_path / "page_0001.png"
    page.write_bytes(b"")
    (tmp_path / "page_0001.png.native").write_bytes(b"")
    (tmp_path / "page_0001.png.layer.txt").write_text(layer, encoding="utf-8")
    bodies = [body]
    return bodies, [str(page)]


def test_the_layer_does_not_get_to_recase_a_heading(vtb, tmp_path):
    """A book that prints a heading in capitals may store it lowercase and
    let the font draw the capitals; the recogniser reads the printed glyphs.
    Deferring here downcased twenty-six section headings in one book and
    turned an economist named McKinnon into Mckinnon. The layer is authority
    on WHICH words are on the page, not on how they are set."""
    words = ("but the layer stores it in lower case and the font is asked "
             "to draw the capitals so this line").split()
    body = "<h2>MARKETISING SHELTER FINANCE</h2><p>" + " ".join(words) + "</p>"
    layer = "marketising shelter finance " + " ".join(words)
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert "MARKETISING SHELTER FINANCE" in bodies[0]
    assert "marketising shelter finance" not in bodies[0]


def test_a_real_spelling_difference_is_still_taken(vtb, tmp_path):
    """The guard must not shut the door on the substitutions that matter:
    where the letters themselves differ, the publisher's text still wins."""
    tail = ("and the rest of this line agrees exactly so that the run is "
            "bounded on both sides by agreement").split()
    body = "<p>the county of cook " + " ".join(tail) + "</p>"
    layer = "the country of cook " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert "the country of cook" in bodies[0]


def test_a_difference_in_the_first_letter_alone_is_the_layer_s_to_call(vtb,
                                                                      tmp_path):
    """The case guard must not swallow sentence boundaries. One book had a
    figure reference dropped, leaving the recogniser to capitalise the word
    after it; the publisher's lower case is right there, and the difference
    is confined to the first letter — unlike a heading in capitals, which
    differs all the way through."""
    tail = ("the alternatives for closing a recessionary gap and the rest of "
            "the line agrees so the run is bounded").split()
    body = "<p>Policy? Illustrates " + " ".join(tail) + "</p>"
    layer = "Policy? illustrates " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert "Policy? illustrates" in bodies[0]


def test_a_capitalised_abbreviation_is_not_a_sentence_boundary(vtb, tmp_path):
    """AAs differs from Aas beyond the first letter, so it is the page's own
    setting and stays ours."""
    tail = ("on terrain with natural screens and a developed road network "
            "so that the run is bounded by agreement").split()
    body = "<p>Designate AAs " + " ".join(tail) + "</p>"
    layer = "Designate Aas " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert "Designate AAs" in bodies[0]


def test_a_one_letter_word_keeps_the_case_the_page_prints(vtb, tmp_path):
    """A single letter has nothing after its first letter, so it cannot show
    whether it is a sentence opening or a word inside a heading. Conceding
    it broke a heading down the middle: FINDING A JOB came back as
    "FINDING a JOB"."""
    tail = ("I chose the site because it was the largest in the city and the "
            "rest of this line agrees exactly").split()
    body = "<p>FINDING A JOB " + " ".join(tail) + "</p>"
    layer = "finding a job " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert "FINDING A JOB" in bodies[0]


def test_a_drop_folio_is_not_pasted_over_the_heading_it_was_paired_with(
        vtb, tmp_path):
    """InDesign writes a drop folio first and a chapter opening's heading
    last, so the aligner paired Lost Worlds' "INDEX" with the layer's "248"
    — nothing agreed on the left to prove it — and the index shipped headed
    <h1>248</h1>, which is why --link-index never found it. The same word
    set the publisher's way needs no proof: the apostrophe beside it is
    still the layer's to set."""
    entries = ("Protection Society, 208 adventure fiction, 44, 47, 48, 73, "
               "76, 208 British politics, and, 48").split()
    body = "<h1>INDEX</h1><p>Aborigines' " + " ".join(entries) + "</p>"
    layer = "248 Aborigines’ " + " ".join(entries) + " INDEX"
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert "<h1>INDEX</h1>" in bodies[0]
    assert "248" not in bodies[0]
    assert "Aborigines’" in bodies[0]


def test_a_folio_is_not_pasted_over_the_last_word_on_the_page(vtb, tmp_path):
    """The other open end. A Works Cited page broke mid-entry after
    "epitomiz'd ,", the layer ended "epitomiz’d, 224", and the folio went
    into the bibliography."""
    lead = ("Bacon, Francis (1651) Sylva Sylvarum: or A Natural History in "
            "Ten Centuries, Verulam, Viscount St. Albans:").split()
    body = "<p>" + " ".join(lead) + " <i>epitomiz'd</i> ,</p>"
    layer = " ".join(lead) + " epitomiz’d, 224"
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert "224" not in bodies[0]


def test_the_same_word_set_differently_is_still_taken_at_the_edge(vtb,
                                                                 tmp_path):
    """Most substitutions at a page's edges were the publisher's curly
    quotes and dashes — 231 of the 329 that reached the shelf — and those
    are the same word, so the open side costs them nothing."""
    lead = ("As Walter Prescott Webb noted in his study of the great plains, "
            "the term").split()
    body = "<p>" + " ".join(lead) + " 'frontier'</p>"
    layer = " ".join(lead) + " ‘frontier’"
    bodies, paths = _native(tmp_path, body, layer)
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert "‘frontier’" in bodies[0]


LEAD = ("We see in the supply schedule that the quantity supplied falls at "
        "each price, and the supply curve thus shifts from S1 to").split()


@pytest.mark.parametrize("ours, layer", [
    ("S_3.", "S3."),                            # a subscript
    ("(MB_E).", "(MBE)."),
    ("P_0=P_n(1+r)^n", "P0=Pn(1+r)n"),          # and a power
    (r"\$446,280.99", "$446,280.99"),           # a sign TeX keeps for itself
    (r"MUA\$1=MUO\$1", "MUA$1=MUO$1"),
])
def test_the_recognisers_markup_is_setting_too(vtb, tmp_path, ours, layer):
    """What the recogniser takes for mathematics it writes as TeX, and the
    page prints none of that. Inside a page the layer's plain word replaces
    it like any other; at a page's last word it was a different word, and
    three economics textbooks ended 27 pages on S_3. where every other
    mention on the page reads S3."""
    body = "<p>" + " ".join(LEAD) + f" {ours}</p>"
    bodies, paths = _native(tmp_path, body, " ".join(LEAD) + f" {layer}")
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert bodies[0].endswith(f" {layer}</p>")


def test_an_equation_that_ends_the_page_is_not_left_half_unwrapped(vtb,
                                                                   tmp_path):
    """The words of a display equation are taken from the layer one by one,
    and the last of them was at the page's edge: \text{ went, and its
    closing brace stayed."""
    body = ("<p>" + " ".join(LEAD) + " S3.</p>"
            r'<math display="block">\text{Exports (X)-imports (M)=net '
            r"exports (Xn)}</math>")
    layer = " ".join(LEAD) + " S3. Exports (X)-imports (M)=net exports (Xn)"
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0].endswith('<math display="block">Exports (X)-imports '
                              "(M)=net exports (Xn)</math>")


def test_a_whole_run_of_markup_at_the_edge_is_taken_together(vtb, tmp_path):
    body = ("<p>" + " ".join(LEAD) + " S3.</p>"
            r'<math display="block">\text{Inflation rate}=(2.073-2.016)/'
            r"2.016=0.028=2.8\%</math>")
    layer = (" ".join(LEAD)
             + " S3. Inflation rate=(2.073-2.016)/2.016=0.028=2.8%")
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0].endswith('<math display="block">Inflation rate='
                              "(2.073-2.016)/2.016=0.028=2.8%</math>")


@pytest.mark.parametrize("ours, layer", [
    ("Y_p.", "YP."),            # not the same letters under the markup
    ("x_2", "y2"),
    ("_____", "224"),           # a rule to write on, and a folio
    ("{", "224"),
    ("r^n", "248"),
])
def test_markup_does_not_make_another_word_the_same_word(vtb, tmp_path,
                                                         ours, layer):
    body = "<p>" + " ".join(LEAD) + f" {ours}</p>"
    bodies, paths = _native(tmp_path, body, " ".join(LEAD) + f" {layer}")
    assert vtb.reconcile_native_text(bodies, paths) == 0
    assert bodies[0].endswith(f" {ours}</p>")


@pytest.mark.parametrize("glyph", ["", "", "\x83", "\x90"])
def test_a_bullet_the_layer_holds_as_a_symbol_font_code_stays_a_bullet(
        vtb, tmp_path, glyph):
    """The layer's word for a bullet set in a symbol font is a private-use
    code point or a control character, which no reader can show. Three
    books rebuilt in November carried 755 of them where the recogniser had
    read the bullet off the page."""
    lead = "These features are designed to give you a sense of what we mean".split()
    tail = "Start Up chapter introductions set the stage for each chapter".split()
    body = "<p>" + " ".join(lead) + "</p><ul><li>• " + " ".join(tail) + "</li></ul>"
    layer = " ".join(lead) + f" {glyph} " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    assert vtb.reconcile_native_text(bodies, paths) == 0
    assert "<li>• Start" in bodies[0] and glyph not in bodies[0]


@pytest.mark.parametrize("byte, ours, printed", [
    ("\x92", "unit's", "unit\u2019s"),         # WinAnsi apostrophe
    ("\x97", "offense-defense", "offense\u2014defense"),   # and em dash
])
def test_winansi_punctuation_in_the_layer_is_not_such_a_glyph(
        vtb, tmp_path, byte, ours, printed):
    """FM 21-10's layer holds its apostrophes as the byte 0x92, which the
    page is given back as a curly apostrophe once it is assembled. Refused
    with the symbol-font codes, those words kept the recogniser's straight
    one: twenty-two pages of the manual."""
    lead = "The commander is responsible for the health of every member of the".split()
    tail = "and for the measures that keep it fit to fight in the field".split()
    theirs = printed.replace("\u2019", byte).replace("\u2014", byte)
    body = "<p>" + " ".join(lead) + f" {ours} " + " ".join(tail) + "</p>"
    bodies, paths = _native(tmp_path, body,
                            " ".join(lead) + f" {theirs} " + " ".join(tail))
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert vtb.repair_c1_controls(bodies) == 1
    assert f" {printed} " in bodies[0]


def test_a_word_beside_such_a_glyph_is_still_the_layers_to_correct(vtb,
                                                                   tmp_path):
    lead = "These features are designed to give you a sense of what we mean".split()
    tail = "chapter introductions set the stage for each chapter in turn".split()
    body = "<p>" + " ".join(lead) + "</p><ul><li>• Startt " + " ".join(tail) + "</li></ul>"
    layer = " ".join(lead) + "  Start " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert "<li>• Start chapter" in bodies[0]


def test_a_url_the_layer_breaks_in_two_is_not_cut_to_its_first_half(
        vtb, tmp_path):
    """The layer held a URL as two words, the second carrying its line-break
    mark; we read one word and a semicolon. Conceding the clean half paired
    our whole address with its first half, and 138 citations in one book's
    notes shipped as "www.scmp.com/;"."""
    lead = "South China Morning Post, October 25, 2017,".split()
    tail = "Chris Buckley and Steven Myers, on the leader's words".split()
    url = ("www.scmp.com/news/china/policies-politics/article/2116836/"
           "xi-jinping")
    body = "<p>" + " ".join(lead) + f" {url} ; " + " ".join(tail) + "</p>"
    layer = (" ".join(lead) + " www.scmp.com/ news/china/policies-politics/"
             "article/2116836/xi-\x02jinping; " + " ".join(tail))
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert url in bodies[0]


def test_a_line_break_inside_the_paired_word_does_not_cost_its_run(vtb,
                                                                   tmp_path):
    """Most garbled runs are sound — 438 of the shelf's 612: the mark sits
    inside the very word it is paired with, so the publisher's apostrophe
    beside it is still taken and the clean reading of the word is kept."""
    lead = "The operation was planned by the 82nd".split()
    tail = "staff over three weeks in the late summer of that year".split()
    body = ("<p>" + " ".join(lead) + " Airborne Division's "
            + " ".join(tail) + "</p>")
    layer = " ".join(lead) + " Air\x02borne Division’s " + " ".join(tail)
    bodies, paths = _native(tmp_path, body, layer)
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert "Airborne Division’s" in bodies[0]
    assert "\x02" not in bodies[0]


# ── a figure the layer has and the reading lacks ─────────────────────────────

def test_a_figure_the_reading_dropped_is_reported_and_not_put_in(vtb, tmp_path):
    """Where There Is No Doctor, page 413: "children 9 to 14 years: 1 ½
    tablets" was read as "½ tablets". The layer has the 1. Words the layer
    has and we lack are mostly a table read across its columns, so with no
    PDF to say where the figure stands (as here) nothing is inserted, and
    a figure in mid-sentence is listed for a person. With one, it is put
    back: tests/test_put_back.py."""
    lead = ("Give medicine each day, starting the day entering a malaria area "
            "until 28 days after leaving the area.").split()
    body = "<p>" + " ".join(lead) + "</p><p>children 9 to 14 years: ½ tablets (150 mg.)</p>"
    layer = " ".join(lead) + " children 9 to 14 years: 1 ½ tablets (150 mg.)"
    bodies, paths = _native(tmp_path, body, layer)
    assert vtb.reconcile_native_text(bodies, paths) == 0
    assert "years: ½ tablets" in bodies[0]
    assert vtb.reconcile_native_text.unplaced == [(0, "14 years:", "1", "½ tablets")]


def test_a_word_the_reading_lacks_is_not_reported(vtb, tmp_path):
    lead = ("Give medicine each day, starting the day entering a malaria area "
            "until 28 days after leaving the area.").split()
    body = "<p>" + " ".join(lead) + "</p><p>adults: tablets each day</p>"
    layer = " ".join(lead) + " adults: two tablets each day"
    bodies, paths = _native(tmp_path, body, layer)
    vtb.reconcile_native_text(bodies, paths)
    assert vtb.reconcile_native_text.unplaced == []

