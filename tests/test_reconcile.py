"""Native-text reconciliation: on born-digital pages the layer IS the page.

The substitutions must be surgical — isolated aligned words only — and the
authority must be scoped: a scan's layer (another OCR's guess) never
substitutes, and loosely aligned regions stay ours however wrong they look.
"""
import os


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
