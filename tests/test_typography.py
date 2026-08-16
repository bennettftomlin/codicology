"""Typography restoration: put back what the recogniser flattened, and only
that.

The boundary this file patrols: restoration versus modernisation. The printed
page had directional quotes and single spaces — the OCR flattened them, so
putting them back recovers the page. The same page set its ellipses ". . ."
with spaces because that is how its printing house worked — "normalising"
those would replace the book's typography with ours, and stays refused. No
letter is ever touched, which is the invariant every test here re-asserts.
"""


def test_double_quotes_take_direction_from_context(vtb):
    bodies = ['<p>He said, "the strike is over," and left. "Not yet."</p>']
    st = vtb.normalize_typography(bodies)
    assert bodies[0] == ('<p>He said, “the strike is over,” and left. '
                        '“Not yet.”</p>')
    assert st["quotes"] == 4


def test_apostrophes_in_contractions_and_possessives(vtb):
    bodies = ["<p>Altgeld's critics don't say it wasn't the workers' "
              "fault.</p>"]
    vtb.normalize_typography(bodies)
    assert bodies[0] == ("<p>Altgeld’s critics don’t say it wasn’t the "
                        "workers’ fault.</p>")


def test_wade_giles_romanisation_keeps_its_glottal_marks(vtb):
    """ch'ü and ts'ung carry the mark between letters — always ’."""
    bodies = ["<p>Ch'en Tu-hsiu and the ch'ü of Hsin ch'ing-nien.</p>"]
    vtb.normalize_typography(bodies)
    assert "Ch’en" in bodies[0] and "ch’ü" in bodies[0]
    assert "ch’ing-nien" in bodies[0]


def test_elisions_open_with_an_apostrophe_not_a_quote(vtb):
    bodies = ["<p>'Tis said the '90s roared, 'til they didn't.</p>"]
    vtb.normalize_typography(bodies)
    assert "’Tis" in bodies[0]
    assert "the ’90s" in bodies[0]
    assert "’til" in bodies[0]
    assert "‘" not in bodies[0]


def test_a_single_quotation_still_opens(vtb):
    bodies = ["<p>The sign read 'no entry' plainly.</p>"]
    vtb.normalize_typography(bodies)
    assert "‘no entry’" in bodies[0]


def test_double_spaces_collapse_and_ellipses_do_not(vtb):
    """". . ." is the book's own typography; the double space is nobody's."""
    bodies = ["<p>It ended.  Badly.  The record shows . . . nothing.</p>"]
    st = vtb.normalize_typography(bodies)
    assert "It ended. Badly." in bodies[0]
    assert ". . ." in bodies[0], "the book's spaced ellipsis was modernised"
    assert st["spaces"] == 1


def test_markup_is_never_entered(vtb):
    bodies = ['<p title="don\'t touch">The \'90s.</p>'
              '<img src="x.jpg" alt="a \'quote\'"/>']
    vtb.normalize_typography(bodies)
    assert 'title="don\'t touch"' in bodies[0]
    assert 'alt="a \'quote\'"' in bodies[0]
    assert "’90s" in bodies[0]


def test_letters_are_identical_before_and_after(vtb):
    import re
    bodies = ['<p>He said, "the \'90s weren\'t Altgeld\'s fault — '
              "'tis true.\"  Twice.</p>"]
    before = re.findall(r"[A-Za-z]+", vtb._strip_tags(bodies[0]))
    vtb.normalize_typography(bodies)
    after = re.findall(r"[A-Za-z]+", vtb._strip_tags(bodies[0]))
    assert before == after


def test_already_directional_text_is_stable(vtb):
    body = "<p>He said, “done” — and Altgeld’s answer stood.</p>"
    bodies = [body]
    st = vtb.normalize_typography(bodies)
    assert bodies[0] == body
    assert st["quotes"] == 0


def test_unusual_characters_are_reported_not_changed(vtb):
    bodies = ["<p>A broken liga¬ture and a stray � mark.</p>"]
    odd = dict(vtb.unusual_characters(bodies))
    assert odd.get("¬") == 1 and odd.get("�") == 1
    assert "liga¬ture" in bodies[0], "reporting must never rewrite"


def test_clean_text_reports_nothing(vtb):
    assert vtb.unusual_characters(["<p>Perfectly ordinary prose.</p>"]) == []


def test_a_born_digital_page_keeps_the_publishers_punctuation(vtb, tmp_path):
    """Its text is the publisher's typesetting, not a flattening to repair —
    even when it contains marks the heuristic would love to change."""
    scan = tmp_path / "p0.png"; scan.write_bytes(b"")
    native = tmp_path / "p1.png"; native.write_bytes(b"")
    (tmp_path / "p1.png.native").write_bytes(b"")
    bodies = ['<p>He said "yes" plainly.</p>',
              '<p>The publisher set "this" deliberately.</p>']
    st = vtb.normalize_typography(bodies, [str(scan), str(native)])
    assert "“yes”" in bodies[0], "the scanned page was not restored"
    assert '"this"' in bodies[1], "the publisher's own marks were overwritten"
    assert st["native_skipped"] == 1
