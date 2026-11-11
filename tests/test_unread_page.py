"""What could not be read can still be shown.

A page that is kept and came through with nothing on it was printed as
"[blank page N]". The Knights of Industry reproduces a handwritten note in
facsimile; no reader could make words of it, and it shipped as a blank page
where the facsimile was the whole point of the leaf. Such a page is now a
picture of itself, when it carries as much ink as a page of text does.
"""
import zipfile

from PIL import Image, ImageDraw

from codicology import pipeline as vtb


class _Silent:
    """Reads the first page and returns nothing for the rest."""
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def run_items(self, images):
        return [[vtb.PageItem(html="<p>The firm has stood for some years.</p>",
                              label="Text")] if k == 0 and len(images) > 1 else []
                for k, _ in enumerate(images)]


def _written(path):
    """A leaf covered in a running hand."""
    im = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(im)
    for row in range(70, 740, 34):
        for x in range(50, 540, 22):
            d.line((x, row + 14, x + 9, row, x + 18, row + 16), fill=(20, 20, 20),
                   width=4)
    im.save(path)
    return str(path)


def _printed(path):
    im = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(im)
    for row in range(80, 700, 40):
        d.rectangle((60, row, 540, row + 14), fill=(30, 30, 30))
    im.save(path)
    return str(path)


def _part_title(path):
    """Two short lines in the middle of an empty page."""
    im = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((250, 300, 350, 308), fill=(40, 40, 40))
    d.rectangle((220, 330, 380, 338), fill=(40, 40, 40))
    im.save(path)
    return str(path)


def test_the_floor_sits_between_a_part_title_and_a_page_of_writing(tmp_path):
    # strokes, as letters and handwriting are; the measure is of pressed
    # ink against paper, and a solid block of grey is not that
    assert vtb._page_core_ink(_written(tmp_path / "w.png")) > 2 * vtb.UNREAD_PAGE_INK
    assert vtb._page_core_ink(_part_title(tmp_path / "t.png")) < vtb.UNREAD_PAGE_INK / 2
    blank = tmp_path / "b.png"
    Image.new("RGB", (600, 800), "white").save(blank)
    assert (vtb._page_core_ink(str(blank)) or 0) < vtb.UNREAD_PAGE_INK / 2


def test_an_unread_page_of_writing_ships_as_a_picture_of_itself(tmp_path,
                                                                capsys):
    paths = [_printed(tmp_path / "page_0000.png"),
             _written(tmp_path / "page_0001.png"),
             _part_title(tmp_path / "page_0002.png")]
    out = tmp_path / "book.epub"
    vtb.build_epub(paths, str(out), _Silent(), "Knights", False, dedupe=False,
                   drop_blank=False)
    z = zipfile.ZipFile(out)
    note = z.read("EPUB/page_0001.xhtml").decode()
    assert 'alt="Page 2, as printed"' in note and "blank page" not in note
    (name,) = [n for n in z.namelist() if "/images/unread_0001" in n]
    with Image.open(zipfile.Path(z, name).open("rb")) as im:
        assert im.size == (600, 800)
    assert "page 2: nothing was read from it and it is not blank" in \
        capsys.readouterr().out
    # a page with next to no ink is still what it was
    quiet = z.read("EPUB/page_0002.xhtml").decode()
    assert "[blank page 3]" in quiet and "<img" not in quiet
    assert not [n for n in z.namelist() if "/images/unread_0002" in n]


def test_a_page_that_was_read_is_never_replaced_by_its_picture(tmp_path):
    paths = [_printed(tmp_path / "page_0000.png"),
             _written(tmp_path / "page_0001.png")]
    out = tmp_path / "book.epub"
    vtb.build_epub(paths, str(out), _Silent(), "Knights", False, dedupe=False,
                   drop_blank=False)
    first = zipfile.ZipFile(out).read("EPUB/page_0000.xhtml").decode()
    assert "The firm has stood" in first and "as printed" not in first


def test_a_printed_page_number_does_not_make_an_unread_page_look_read(
        tmp_path, monkeypatch):
    """With the page list on, the page's number is set at the head of its
    body before the body is asked whether it has anything on it. It had its
    number, so it was not empty, so the note was not shown: the one page
    this rule was written for went blank when its book was rebuilt with
    page numbers."""
    paths = [_printed(tmp_path / "page_0000.png"),
             _written(tmp_path / "page_0001.png"),
             _part_title(tmp_path / "page_0002.png")]
    monkeypatch.setattr(vtb, "fill_folio_gaps",
                        lambda folios: ({0: 11, 1: 12, 2: 13}, [], []))
    out = tmp_path / "book.epub"
    vtb.build_epub(paths, str(out), _Silent(), "Knights", False, dedupe=False,
                   drop_blank=False, check_folios=True)
    z = zipfile.ZipFile(out)
    note = z.read("EPUB/page_0001.xhtml").decode()
    assert 'aria-label="12"' in note, "the page keeps its number"
    assert 'alt="Page 2, as printed"' in note
    assert note.index("pagebreak") < note.index("<img")
    # and a page with a number and nothing else on it is still that
    quiet = z.read("EPUB/page_0002.xhtml").decode()
    assert 'aria-label="13"' in quiet and "<img" not in quiet

