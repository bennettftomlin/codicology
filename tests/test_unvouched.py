"""Words neither the recogniser nor a witness will stand behind.

The Knights of Industry reproduces seven handwritten letters in facsimile.
Each came back as fluent text that is not on the page, in English for a
German hand, with no picture of the letter. The recogniser scored those
pages 0.75 to 0.84 where a page of print scores 0.99, and neither the PDF's
text layer nor the classical reader found a tenth of the words. A page like
that is shown as a picture of itself.
"""
import zipfile

import pytest
from PIL import Image, ImageDraw

from codicology import pipeline as vtb

INVENTED = ("may should inspire all times growing and taking the Birch's - "
            "this ware, she serve & there, I say - sought them, Birch - thing, "
            "there is thereing arose Caroline, George & George arose therever. "
            "I should think it will be there when I will thing should, in "
            "which I should I should should I will there should I should I")
PRINTED = ("The firm has stood for some years in Trinity Square, and its "
           "letters go out under two names. The clerk who writes them is paid "
           "by the week and keeps no copy of what he sends to the country "
           "houses that answer the advertisements in the German papers.")


def _blocks(text, conf):
    half = len(text) // 2
    return [vtb.PageItem(html=f"<p>{text[:half]}</p>", label="Text", conf=conf),
            vtb.PageItem(html=f"<p>{text[half:]}</p>", label="Text", conf=conf)]


def _leaf(tmp_path, name="page_0000.png"):
    path = tmp_path / name
    im = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(im)
    for row in range(70, 740, 34):
        for x in range(50, 540, 22):
            d.line((x, row + 14, x + 9, row, x + 18, row + 16), fill=(20, 20, 20),
                   width=4)
    im.save(path)
    return str(path)


@pytest.fixture
def witness(monkeypatch):
    """What the classical reader makes of the page, set by the test."""
    said = {"text": ""}
    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    monkeypatch.setattr(vtb, "_classical_text", lambda path: said["text"])
    monkeypatch.setattr(vtb, "_classical_word_count",
                        lambda path, min_len=2: len(said["text"].split()))
    return said


def test_low_confidence_and_no_witness_is_unvouched(tmp_path, witness):
    conf, share, borne_out = vtb.unvouched_reading(_blocks(INVENTED, 0.78),
                                                   _leaf(tmp_path))
    assert conf == pytest.approx(0.78) and share == 0.0 and borne_out == []


def test_a_confident_reading_is_not_questioned(tmp_path, witness):
    """Print the classical reader cannot read at all is still print: the
    question is asked only of a page the recogniser itself is unsure of."""
    assert vtb.unvouched_reading(_blocks(PRINTED, 0.99), _leaf(tmp_path)) is None


def test_an_unsure_reading_the_classical_reader_shares_stands(tmp_path, witness):
    """KGB, pages 172, 398 and 408: scored 0.86 to 0.92, and the witness
    reads 97 words in a hundred of them."""
    witness["text"] = PRINTED.replace("Trinity", "Trinitv")
    assert vtb.unvouched_reading(_blocks(PRINTED, 0.87), _leaf(tmp_path)) is None


def test_the_layer_can_vouch_where_the_classical_reader_cannot(tmp_path, witness):
    """A map: the classical reader finds none of its spaced-out labels and
    the layer holds them."""
    page = _leaf(tmp_path)
    with open(page + ".layer.txt", "w") as fh:
        fh.write(PRINTED)
    assert vtb.unvouched_reading(_blocks(PRINTED, 0.90), page) is None


def test_a_few_words_are_too_few_to_ask_about(tmp_path, witness):
    """A half-title scores low and no witness reads display type. Thirty-five
    such pages on the shelf are covers and part titles, read correctly."""
    assert vtb.unvouched_reading(
        _blocks("THE INVISIBLE GOVERNMENT DAVID WISE", 0.80), _leaf(tmp_path)) is None


def test_a_reading_that_carries_no_confidence_is_not_judged(tmp_path, witness):
    assert vtb.unvouched_reading(_blocks(INVENTED, None), _leaf(tmp_path)) is None


def test_with_nobody_to_ask_the_page_is_passed_and_counted(tmp_path, monkeypatch):
    monkeypatch.setattr(vtb, "witness_available", lambda: False)
    before = vtb.UNWITNESSED["fabrication"]
    assert vtb.unvouched_reading(_blocks(INVENTED, 0.78), _leaf(tmp_path)) is None
    assert vtb.UNWITNESSED["fabrication"] == before + 1
    vtb.UNWITNESSED["fabrication"] = before


def test_a_plate_with_a_printed_caption_is_still_unvouched(tmp_path, witness):
    """Studies in the History of Accounting, plate V: the caption is print
    and both witnesses read it, which made 0.55 of an invented page."""
    caption = ("BALANCE SHEET OF THE DATINI BRANCH IN BARCELONA 1399 see page "
               "one hundred and thirty four of this volume for the text")
    witness["text"] = caption
    said = INVENTED + " " + caption
    assert vtb.unvouched_reading(_blocks(said, 0.84), _leaf(tmp_path)) is not None


def test_what_a_witness_reads_word_for_word_is_kept(tmp_path, witness):
    """The same plate, as the reader blocked it: the hand, and the caption
    under it. The caption is print, both witnesses have it, and it was read
    right. It is the one block of the page that is still printed."""
    caption = vtb.PageItem(
        html="<p>V. LAST PAGE OF BALANCE SHEET OF FRANCESCO DATINI AND CO. IN "
             "BARCELONA, 1399 (see p. 143)</p>", label="Caption",
        is_caption=True, conf=0.97)
    witness["text"] = ("ApePNor prteo Boi Oearte wenDiute V. LAST PAGE OF "
                       "BALANCE SHEET OF FRANCESCO DATINI AND CO. IN "
                       "BARCELONA, 1399 (see p. 143)")
    items = _blocks(INVENTED, 0.80) + [caption]
    conf, share, borne_out = vtb.unvouched_reading(items, _leaf(tmp_path))
    assert share < vtb.VOUCHED_SHARE and borne_out == [caption]


def test_a_block_the_witness_has_most_of_is_not_enough(tmp_path, witness):
    """Lloyd's, page 43: a damaged page read four words in five right and
    the fifth supplied ("the Hinsearch traders", "the Soligarie"). No
    paragraph of it is one a witness reads through, and none is kept."""
    half = PRINTED.split()
    witness["text"] = " ".join(w for k, w in enumerate(half) if k % 5)
    conf, share, borne_out = vtb.unvouched_reading(
        _blocks(PRINTED, 0.85) + _blocks(INVENTED, 0.80) + _blocks(INVENTED, 0.80),
        _leaf(tmp_path))
    assert borne_out == []


def test_a_word_or_two_the_witness_happens_to_have_is_not_a_block(tmp_path, witness):
    witness["text"] = "Birch Caroline George"
    items = _blocks(INVENTED, 0.78) + [
        vtb.PageItem(html="<p>Caroline George</p>", label="Text", conf=0.8)]
    assert vtb.unvouched_reading(items, _leaf(tmp_path))[2] == []


# ── through a build ──────────────────────────────────────────────────────────

class _Reads:
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def run_items(self, images):
        return [_blocks(PRINTED, 0.995) if k == 0 else _blocks(INVENTED, 0.78)
                for k, _ in enumerate(images)]


def test_a_built_book_shows_the_letter_and_not_the_invention(tmp_path, witness,
                                                             capsys):
    pages = [_leaf(tmp_path, "page_0000.png"), _leaf(tmp_path, "page_0001.png")]
    # the classical reader makes something of the letterhead and nothing of
    # the hand: not silent, which an older guard already acts on, and not
    # the words that were read either
    witness["text"] = ("Lehnert Spedition Auskunft Incasso Trinity Square "
                       "Tower Hill London den Oktober Herrn Wagner")
    out = tmp_path / "book.epub"
    vtb.build_epub(pages, str(out), _Reads(), "Letters", False,
                   dedupe=False, drop_blank=False)
    z = zipfile.ZipFile(out)
    first = z.read("EPUB/page_0000.xhtml").decode()
    second = z.read("EPUB/page_0001.xhtml").decode()
    assert "Trinity Square" in first and "<img" not in first, \
        "a confident page is printed as read, whatever the witness says"
    assert "I should" not in second and second.count("<img") == 1
    # marked the way a page nothing was read from is marked, for a reader
    # who cannot see the picture and for verify
    assert 'alt="Page 2, as printed"' in second
    said = capsys.readouterr().out
    assert "shown as a picture of the page rather than printed as invention" in said


class _ReadsAPlate(_Reads):
    """A page of print, then a manuscript plate with its printed caption."""
    CAPTION = ("V. LAST PAGE OF BALANCE SHEET OF FRANCESCO DATINI AND CO. IN "
               "BARCELONA, 1399 (see p. 143)")

    def run_items(self, images):
        plate = _blocks(INVENTED, 0.78) + [vtb.PageItem(
            html=f"<p>{self.CAPTION}</p>", label="Caption", is_caption=True,
            conf=0.97, box=(0.1, 0.9, 0.9, 0.95))]
        return [_blocks(PRINTED, 0.995) if k == 0 else plate
                for k, _ in enumerate(images)]


def test_a_built_plate_keeps_its_printed_caption_under_the_picture(
        tmp_path, witness, capsys):
    pages = [_leaf(tmp_path, "page_0000.png"), _leaf(tmp_path, "page_0001.png")]
    witness["text"] = "ApePNor prteo Boi Oearte wenDiute " + _ReadsAPlate.CAPTION
    out = tmp_path / "book.epub"
    vtb.build_epub(pages, str(out), _ReadsAPlate(), "Plates", False,
                   dedupe=False, drop_blank=False)
    plate = zipfile.ZipFile(out).read("EPUB/page_0001.xhtml").decode()
    assert "I should" not in plate and plate.count("<img") == 1
    assert 'alt="Page 2, as printed"' in plate
    assert plate.index("<img") < plate.index("FRANCESCO DATINI")
    assert "<figcaption>" in plate, "a caption goes under its picture as one"
    said = capsys.readouterr().out
    assert "block(s) a witness does read printed beneath it" in said


def test_words_nobody_vouches_for_on_blank_paper_are_not_a_picture_either(
        tmp_path, witness, capsys):
    """Citadel of Sin, a blank verso: thirty-eight words read off the ghost
    of the leaf behind it. A picture of the page would be a picture of
    nothing."""
    printed = _leaf(tmp_path, "page_0000.png")
    blank = tmp_path / "page_0001.png"
    Image.new("RGB", (600, 800), (238, 236, 230)).save(blank)
    # what the classical reader makes of the ghost: something, and not this
    witness["text"] = ("Lehnert Spedition Auskunft Incasso Trinity Square "
                       "Tower Hill London den Oktober Herrn Wagner")
    out = tmp_path / "book.epub"
    vtb.build_epub([printed, str(blank)], str(out), _Reads(), "Letters", False,
                   dedupe=False, drop_blank=False)
    verso = zipfile.ZipFile(out).read("EPUB/page_0001.xhtml").decode()
    assert "I should" not in verso and "<img" not in verso
    assert "the paper holds no ink" in capsys.readouterr().out

