"""A page the build chose to show as itself is not a page that went missing.

A scan's text layer is another reader's attempt at the page. Where this
pipeline's own reading of a page could not be stood behind (a letter in
facsimile, a ledger in a fourteenth-century hand) the page is shown as a
picture, and the other reader's attempt at it then looked like a loss: the
source "has text here and we have none". Studies in the History of
Accounting and The Knights of Industry each went from "no holes found" to
"LOOK AT THIS" over pages that were shown on purpose.

The exemption is as narrow as the reason for it. It needs the page to be
marked as shown, by the build, and the source to be a scan. A born-digital
page shown as a picture is still a hole: there the layer is the page's own
text. And a page that merely has a picture on it and no words is still a
hole anywhere.
"""
import io
import zipfile

import fitz
import pytest
from PIL import Image

from codicology import verify

LAYER = ("the entries listed include an item of eighty pounds set aside as a "
         "reserve for unpaid expenses and taxes and another item representing "
         "the balance of profit and loss at first there was a gap between "
         "total debits and credits but it was reduced by making several "
         "adjustments in the books of the company")
PROSE = ("the firm kept its books in the italian manner with a journal and a "
         "ledger and a separate book for the merchandise that went by sea")

XHTML = ('<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/'
         '1999/xhtml"><head><title>p</title></head><body>{}</body></html>')
SHOWN = ('<figure><img src="images/fig_0000.jpg" alt="Page {}, as printed" '
         'style="max-width:100%"/></figure>')
PLAIN = '<figure><img src="images/fig_0000.jpg" alt="Figure" style="max-width:100%"/></figure>'


def _pdf(tmp_path, n, scan):
    """n pages with a text layer; a scan also has a picture of each page."""
    doc = fitz.open()
    buf = io.BytesIO()
    Image.new("RGB", (120, 160), (235, 232, 225)).save(buf, "PNG")
    for k in range(n):
        page = doc.new_page(width=400, height=560)
        if scan:
            page.insert_image(page.rect, stream=buf.getvalue())
        page.insert_textbox(fitz.Rect(30, 30, 370, 530),
                            LAYER if k == 3 else f"{PROSE} leaf {k}",
                            fontsize=9, render_mode=3 if scan else 0)
    path = str(tmp_path / ("scan.pdf" if scan else "typeset.pdf"))
    doc.save(path)
    return path


def _epub(tmp_path, n, page_3):
    path = str(tmp_path / "book.epub")
    with zipfile.ZipFile(path, "w") as z:
        for k in range(n):
            body = page_3 if k == 3 else f"<p>{PROSE} leaf {k}</p>"
            z.writestr(f"EPUB/page_{k:04d}.xhtml", XHTML.format(body))
    return path


@pytest.fixture(autouse=True)
def every_word_is_a_word(monkeypatch):
    monkeypatch.setattr(verify, "_WORDS", verify._Everything())


def test_a_page_shown_as_itself_on_a_scan_is_not_a_hole(tmp_path, capsys):
    pdf = _pdf(tmp_path, 8, scan=True)
    assert verify.main(_epub(tmp_path, 8, SHOWN.format(4)), pdf) == 0
    said = capsys.readouterr().out
    assert "source looks scan" in said
    assert "pages where the source has text and we have none: 0" in said
    assert "pages shown as pictures of themselves" in said and "p3 (" in said
    assert "VERDICT: no holes found" in said


def test_a_page_with_only_a_picture_on_it_is_still_a_hole(tmp_path, capsys):
    """The mark is the build's own. A figure standing alone on a page whose
    source has words is the loss this check exists to catch."""
    pdf = _pdf(tmp_path, 8, scan=True)
    assert verify.main(_epub(tmp_path, 8, PLAIN), pdf) == 1
    said = capsys.readouterr().out
    assert "pages where the source has text and we have none: 1" in said
    assert "shown as pictures of themselves" not in said


def test_on_a_born_digital_source_a_shown_page_is_still_a_hole(tmp_path, capsys):
    """There the layer is the page's own text, and it is not in the book."""
    pdf = _pdf(tmp_path, 8, scan=False)
    assert verify.main(_epub(tmp_path, 8, SHOWN.format(4)), pdf) == 1
    said = capsys.readouterr().out
    assert "source looks born-digital" in said
    assert "pages where the source has text and we have none: 1" in said


def test_a_shown_page_whose_words_are_in_the_book_is_not_listed(tmp_path, capsys):
    pdf = _pdf(tmp_path, 8, scan=True)
    body = SHOWN.format(4) + f"<p>{LAYER}</p>"
    assert verify.main(_epub(tmp_path, 8, body), pdf) == 0
    assert "shown as pictures of themselves" not in capsys.readouterr().out
