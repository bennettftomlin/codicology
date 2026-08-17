"""Words split across page turns rejoin; everything doubtful stays printed.

Eagle measured the need: sixty pages ending mid-word, Rus- | sia in
Beginnings among them, unfindable by search and unreadable by speech.
The join is gated on lexicality — the dictionary's words or the book's
own — and every refusal class here is deliberate.
"""
from codicology import pipeline as vtb


def _run(bodies, dropped=frozenset()):
    stats = vtb.join_page_break_hyphens(bodies, set(dropped))
    return bodies, stats


def test_the_split_word_rejoins_with_its_punctuation(vtb_=None):
    bodies, st = _run(["<p>he fled to Rus-</p>", "<p>sia, and beyond</p>"])
    assert st == {"joined": 1, "refused": 0}
    assert bodies[0] == "<p>he fled to Russia,</p>"
    assert bodies[1] == "<p>and beyond</p>"


def test_a_printed_compound_keeps_its_hyphen(vtb_=None):
    bodies, st = _run(["<p>an act of self-</p>", "<p>control it was</p>"])
    assert st == {"joined": 0, "refused": 1}
    assert bodies[0].endswith("self-</p>")
    assert bodies[1].startswith("<p>control")


def test_an_em_dash_is_authorial_and_never_joins(vtb_=None):
    bodies, st = _run(["<p>a plea for liberty—</p>", "<p>to the end</p>"])
    assert st == {"joined": 0, "refused": 0}
    assert bodies[0].endswith("liberty—</p>")


def test_a_capitalized_continuation_is_left_alone(vtb_=None):
    bodies, st = _run(["<p>the New-</p>", "<p>York custom</p>"])
    assert st == {"joined": 0, "refused": 0}


def test_a_heading_is_never_raided(vtb_=None):
    bodies, st = _run(["<p>the situa-</p>", "<h2>tion room</h2><p>text</p>"])
    assert st == {"joined": 0, "refused": 0}
    assert "<h2>tion room</h2>" in bodies[1]


def test_the_books_own_vocabulary_rescues_a_proper_noun(vtb_=None):
    bodies, st = _run(["<p>Governor Alt-</p>", "<p>geld spoke</p>",
                       "<p>Altgeld was heard again</p>"])
    assert st["joined"] == 1
    assert bodies[0] == "<p>Governor Altgeld</p>"


def test_joins_reach_across_a_dropped_blank(vtb_=None):
    bodies, st = _run(["<p>to Rus-</p>", "", "<p>sia at last</p>"],
                      dropped={1})
    assert st["joined"] == 1
    assert bodies[0] == "<p>to Russia</p>"
    assert bodies[2] == "<p>at last</p>"


def test_anchors_at_the_page_top_survive(vtb_=None):
    bodies, st = _run([
        "<p>to Rus-</p>",
        '<span epub:type="pagebreak" id="pgb-7"/><p>sia at last</p>'])
    assert st["joined"] == 1
    assert bodies[1].startswith('<span epub:type="pagebreak" id="pgb-7"/>')
    assert "sia" not in bodies[1]
