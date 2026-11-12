"""Words the publisher prints and the reading passed over.

A recogniser that writes a page out word by word errs as a copyist does: a
doubled word is written once, the second of two like words is taken for the
first, the eye drops a line. On a born-digital page the PDF's own text has
what was dropped, and it is put back where four things hold: the words
round the gap agree, each word is a word or a figure, the page does not
have it already, and in the PDF it stands in the line. Most of what the
layer has and the page lacks is none of that: a table read across its
columns, a running head, a page reference from the next column.

Each case here is one found on a shelf of 134 books and looked at on the
page; the pages are set again in type so the PDF can be asked where its
words stand.
"""
import json

import pytest

pymupdf = pytest.importorskip("pymupdf")


def _typeset(tmp_path, lines, body, name="page_0000", **source):
    """A page a publisher set: `lines` are (x, y, text). Returns what the
    reconciler is given, with the sidecars the loader writes."""
    import pypdfium2 as pdfium
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=648)
    for x, y, text in lines:
        page.insert_text((x, y), text, fontsize=9)
    pdf = str(tmp_path / f"{name}.pdf")
    doc.save(pdf)
    p = str(tmp_path / f"{name}.png")
    open(p, "w").close()
    with open(p + ".layer.txt", "w", encoding="utf-8") as fh:
        fh.write(pdfium.PdfDocument(pdf)[0].get_textpage().get_text_bounded().strip())
    open(p + ".native", "w").close()
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": pdf, "page": 0, "facsimile": False, "scan": False, **source}, fh)
    return [body], [p]


def _set(*texts, x=72, y=100):
    return [(x, y + 13 * k, t) for k, t in enumerate(texts)]


# ── put back ─────────────────────────────────────────────────────────────────

def test_the_figure_of_a_dose(vtb, tmp_path):
    """Where There Is No Doctor, p. 413: "1 ½ tablets" was read "½ tablets"."""
    bodies, paths = _typeset(tmp_path, _set(
        "Give medicine each day, starting the day entering a malaria area.",
        "adults: 2 tablets (200 mg.)",
        "children 9 to 14 years: 1 ½ tablets (150 mg.)",
        "children 3 to 8 years: 1 tablet (100 mg.)"),
        "<p>Give medicine each day, starting the day entering a malaria area.</p>"
        "<p>adults: 2 tablets (200 mg.)</p><p>children 9 to 14 years: ½ tablets (150 mg.)</p>"
        "<p>children 3 to 8 years: 1 tablet (100 mg.)</p>")
    assert vtb.reconcile_native_text(bodies, paths) == 0     # no word was exchanged
    assert "<p>children 9 to 14 years: 1 ½ tablets (150 mg.)</p>" in bodies[0]
    assert vtb.reconcile_native_text.put_back == [(0, "to 14 years:", "1", "½ tablets (150")]
    assert vtb.reconcile_native_text.unplaced == []


@pytest.mark.parametrize("printed, read", [
    # a doubled word, written once: the book's grammar, or the book's slip
    ("It is no exaggeration to say that that the world is now in a state of crisis.",
     "It is no exaggeration to say that the world is now in a state of crisis."),
    ("They tried to organize workers but had had no significant impact on wages.",
     "They tried to organize workers but had no significant impact on wages."),
    # the second of two like words taken for the first
    ("Final consumption rose from 1.4 billion toe to 2.5 billion toe in those years.",
     "Final consumption rose from 1.4 billion to 2.5 billion toe in those years."),
    ("They are recruiting grounds for activists, as well as fora for teaching and learning.",
     "They are recruiting grounds for activists, as well as for teaching and learning."),
    # a small word gone
    ("Sharing face to face is widespread but by no means a dominant practice there.",
     "Sharing face to face is widespread but no means a dominant practice there."),
    ("He found that the Admiral had improperly driven home his Government vehicle on occasion.",
     "He found that the Admiral had improperly driven his Government vehicle on occasion."),
    # the year of a book in a note
    ("See Paul Preston, Franco, Madrid: Temas de hoy, 1996, p. 521; and Casanova, pp. 119-29.",
     "See Paul Preston, Franco, Madrid: Temas de hoy, p. 521; and Casanova, pp. 119-29."),
])
def test_a_word_passed_over_in_a_line(vtb, tmp_path, printed, read):
    bodies, paths = _typeset(tmp_path, _set(printed), f"<p>{read}</p>")
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == f"<p>{printed}</p>"
    assert len(vtb.reconcile_native_text.put_back) == 1


def test_a_line_the_eye_dropped(vtb, tmp_path):
    """Kill Boxes, p. 96: eleven words gone between "prevents" and "pretends"."""
    bodies, paths = _typeset(tmp_path, _set(
        "the pain, like the interrogation, is a vehicle of self-betrayal.",
        "Torture systematically prevents the prisoner from being",
        "an agent of anything and simultaneously pretends that he is the",
        "agent of some things."),
        "<p>the pain, like the interrogation, is a vehicle of self-betrayal. Torture "
        "systematically pretends that he is the agent of some things.</p>")
    vtb.reconcile_native_text(bodies, paths)
    assert ("Torture systematically prevents the prisoner from being an agent of anything "
            "and simultaneously pretends that he is") in bodies[0]
    assert [w for _, _, w, _ in vtb.reconcile_native_text.put_back] == [
        "prevents the prisoner from being an agent of anything and simultaneously"]


def test_a_word_of_the_books_own_counts_as_a_word(vtb, tmp_path):
    printed = "The miners called this their own Zwangsarbeit and said so in the pit."
    read = "The miners called this their own and said so in the pit."
    bodies, paths = _typeset(tmp_path, _set(printed), f"<p>{read}</p>")
    vtb.reconcile_native_text(bodies, paths)
    assert vtb.reconcile_native_text.put_back == []          # no dictionary has it
    bodies, paths = _typeset(tmp_path, _set(printed), f"<p>{read}</p>")
    bodies.append("<p>Zwangsarbeit was abolished in the following year.</p>")
    paths.append(str(tmp_path / "page_0001.png"))
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == f"<p>{printed}</p>"


def test_a_word_exchanged_and_a_word_put_back_in_one_paragraph(vtb, tmp_path):
    printed = ("The joint copublication was announced by the press office, and it is no "
               "exaggeration to say that that the world noticed.")
    bodies, paths = _typeset(tmp_path, _set(printed[:71], printed[72:]),
                             "<p>" + printed.replace("copublication", "copulication")
                             .replace("that that", "that") + "</p>")
    assert vtb.reconcile_native_text(bodies, paths) == 1
    assert bodies[0] == f"<p>{printed}</p>"


def test_agreement_is_of_words_and_not_of_how_they_are_set(vtb, tmp_path, monkeypatch):
    """Tweets and the Streets, p. 95. The recogniser's straight quotation
    mark beside the gap is the publisher's curly one read plainly: the
    words on that side agree, though only one of them is letter for letter
    the same."""
    layer = ("They called it the ‘jump to the streets’ (salto a la calle) and overcame "
             "their isolation.")
    p = str(tmp_path / "page_0000.png")
    open(p, "w").close()
    with open(p + ".layer.txt", "w", encoding="utf-8") as fh:
        fh.write(layer)
    open(p + ".native", "w").close()
    # one line of type, a word space between its words
    monkeypatch.setattr(vtb, "_layer_geometry", lambda path: [
        (w, 72 + 30 * k, 498, 72 + 30 * k + 27, 507) for k, w in enumerate(layer.split())])
    bodies = ["<p>They called it the 'jump to the streets' (salto la calle) and overcame "
              "their isolation.</p>"]
    vtb.reconcile_native_text(bodies, [p])
    assert "(salto a la calle)" in bodies[0]
    assert [w for _, _, w, _ in vtb.reconcile_native_text.put_back] == ["a"]


# ── left as read ─────────────────────────────────────────────────────────────

def test_a_page_reference_in_the_next_column_is_not_part_of_the_sentence(vtb, tmp_path):
    """Where There Is No Doctor, p. 94: the layer has "soak 102 joint in cold
    water". The page does lack the 102; it is not where the layer has it."""
    lines = _set("8. stiff, sore muscles and joints: hot compresses",
                 "9. strains and sprains the first day: soak",
                 "joint in cold water; then use hot soaks")
    # set, as that page is, a row at a time: the reference comes after the
    # first line of its row and before the second
    lines.insert(2, (330, lines[1][1], "102"))
    body = ("<p>8. stiff, sore muscles and joints: hot compresses</p><p>9. strains and sprains "
            "the first day: soak joint in cold water; then use hot soaks</p>")
    bodies, paths = _typeset(tmp_path, lines, body)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body and vtb.reconcile_native_text.put_back == []
    assert vtb.reconcile_native_text.unplaced == [(0, "day: soak", "102", "joint in")]


def test_a_running_head_lined_up_with_a_caption(vtb, tmp_path):
    """Burning Up, p. 111: the head "crises and oil price shocks" met the
    caption "two oil shocks" at two words, one on each side of "price"."""
    body = ("<p>Energy use fell twice in the decade and rose again after each fall.</p>"
            "<p>Consumption through two oil shocks</p>")
    bodies, paths = _typeset(tmp_path, _set(
        "Energy use fell twice in the decade and rose again after each fall.",
        "crises and oil price shocks 97"), body)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body and vtb.reconcile_native_text.put_back == []


def test_what_the_page_has_already_is_only_somewhere_else(vtb, tmp_path):
    """A chapter's number set beside its title and read after it."""
    body = "<p>The contents list Chapter Offensive operations and the preparations for battle 4</p>"
    bodies, paths = _typeset(tmp_path, _set(
        "The contents list Chapter 4 Offensive operations and the preparations for battle"), body)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body and vtb.reconcile_native_text.put_back == []


def test_words_the_layer_ran_together_are_not_a_word(vtb, tmp_path):
    body = "<p>The slope of the line equals the vertical change over the whole of its run.</p>"
    bodies, paths = _typeset(tmp_path, _set(
        "The slope of the line equals the vertical changehorizontal change over the whole of its run."), body)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body and vtb.reconcile_native_text.put_back == []


def test_a_gap_between_two_blocks_belongs_to_neither(vtb, tmp_path):
    printed = "Sharing face to face is widespread but by no means a dominant practice there."
    body = "<p>Sharing face to face is widespread but</p><p>no means a dominant practice there.</p>"
    bodies, paths = _typeset(tmp_path, _set(printed), body)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body and vtb.reconcile_native_text.put_back == []


@pytest.mark.parametrize("source", [{"scan": True}, {"facsimile": True}])
def test_a_page_that_is_a_picture_of_itself_is_not_asked(vtb, tmp_path, source):
    printed = "It is no exaggeration to say that that the world is now in a state of crisis."
    body = "<p>" + printed.replace("that that", "that") + "</p>"
    bodies, paths = _typeset(tmp_path, _set(printed), body, **source)
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body


def test_with_no_pdf_to_ask_nothing_is_put_back(vtb, tmp_path):
    import os
    printed = "It is no exaggeration to say that that the world is now in a state of crisis."
    body = "<p>" + printed.replace("that that", "that") + "</p>"
    bodies, paths = _typeset(tmp_path, _set(printed), body)
    os.remove(paths[0] + ".source.json")
    vtb.reconcile_native_text(bodies, paths)
    assert bodies[0] == body and vtb.reconcile_native_text.put_back == []


def test_one_more_on_the_layer_is_one_put_back(vtb, tmp_path):
    """Two gaps that the same word would fill, and the layer has it only
    once more than the page: the second is not a word the page lacks."""
    bodies, paths = _typeset(tmp_path, _set(
        "The clerk said that that the ledger was closed for the year.",
        "The master said that the books were all in order again."),
        "<p>The clerk said that the ledger was closed for the year.</p>"
        "<p>The master said that the books were all in order again.</p>")
    vtb.reconcile_native_text(bodies, paths)
    assert len(vtb.reconcile_native_text.put_back) == 1
    assert bodies[0].count("that that") == 1 and "master said that the" in bodies[0]


# ── where a word stands ──────────────────────────────────────────────────────

def test_what_follows_what_in_a_line_of_type(vtb):
    word = lambda left, right, base: ("w", left, base - 2, right, base + 7)
    a = word(100, 130, 500)
    assert vtb._follows(a, word(133, 150, 500))              # a word space on
    assert vtb._follows(a, word(140, 160, 500))              # a justified line's
    assert not vtb._follows(a, word(170, 190, 500))          # the next column
    assert not vtb._follows(a, word(60, 90, 500))            # before it
    assert vtb._follows(a, word(72, 95, 487))                # opening the next line
    assert not vtb._follows(a, word(100, 120, 487))          # under it, not back at the margin
    assert not vtb._follows(a, word(72, 95, 513))            # the line above
    assert not vtb._follows(a, word(72, 95, 440))            # further down the page
