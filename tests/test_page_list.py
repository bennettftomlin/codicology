"""The printed page numbers, restored and written where a reader can use them.

Two separate things are tested here, and the second is the one that bites.

Interpolation: a page that reads no folio has usually not lost one — chapter
openings set their number as display type, part-titles and blank versos print
none at all. The number still exists in the publisher's pagination. It may be
restored only where the book's own arithmetic supplies it, never guessed.

Purity: ebooklib decides what belongs in a page-list by asking whether an
element has both an epub:type and an id, never which epub:type — so every note
marker qualified, and one book shipped a page-list of 567 footnote markers
labelled 1, 2, 3. Nothing renders it, which is exactly why it went unnoticed
across twenty books. The test that matters asserts the OUTCOME: a page-list
holds page numbers and nothing else, however the harvesting is arranged.
"""
import re
import zipfile

import pytest

pytest.importorskip("ebooklib")


def F(vtb, index, number, confident=True):
    return vtb.Folio(index, number, str(number or ""), confident)


def test_a_gap_the_arithmetic_closes_is_filled(vtb):
    # p10 is folio 5, p12 is folio 7: p11 can only be folio 6
    folios = [F(vtb, 10, 5), F(vtb, 11, None, False), F(vtb, 12, 7)]
    numbers, refused, _ = vtb.fill_folio_gaps(folios)
    assert numbers[11] == 6
    assert refused == []


def test_a_run_of_several_pages_is_filled_not_feared(vtb):
    """Two unread pages between anchors three apart is still arithmetic."""
    folios = [F(vtb, 10, 5), F(vtb, 11, None, False),
              F(vtb, 12, None, False), F(vtb, 13, 8)]
    numbers, refused, _ = vtb.fill_folio_gaps(folios)
    assert (numbers[11], numbers[12]) == (6, 7)
    assert refused == []


def test_a_long_run_fills_completely(vtb):
    folios = ([F(vtb, 0, 1)]
              + [F(vtb, i, None, False) for i in range(1, 11)]
              + [F(vtb, 11, 12)])
    numbers, refused, _ = vtb.fill_folio_gaps(folios)
    assert [numbers[i] for i in range(1, 11)] == list(range(2, 12))
    assert refused == []


def test_a_gap_that_does_not_close_is_refused(vtb):
    """Three pages apart but five numbers apart: leaves are unaccounted for,
    and which page holds which number is not knowable."""
    folios = [F(vtb, 10, 5), F(vtb, 11, None, False),
              F(vtb, 12, None, False), F(vtb, 13, 10)]
    numbers, refused, _ = vtb.fill_folio_gaps(folios)
    assert 11 not in numbers and 12 not in numbers
    assert refused == [(10, 13, 5, 10)]


def test_nothing_is_extrapolated_past_the_outermost_anchor(vtb):
    """Front matter is commonly Roman; running Arabic backwards off the first
    anchor would invent a numbering the book does not use."""
    folios = ([F(vtb, i, None, False) for i in range(0, 3)]
              + [F(vtb, 3, 1), F(vtb, 4, 2)]
              + [F(vtb, 5, None, False)])
    numbers, refused, _ = vtb.fill_folio_gaps(folios)
    assert set(numbers) == {3, 4}
    assert refused == []


def test_a_numbering_restart_is_refused_not_smoothed(vtb):
    """Folio 1 after folio 240 is a restart or a misread; either way no
    number in between is knowable. The order filter cuts the reading that
    breaks the run, and the page between stays unfilled — what matters is
    that nothing was invented."""
    folios = [F(vtb, 10, 240), F(vtb, 11, None, False), F(vtb, 12, 1)]
    numbers, refused, distrusted = vtb.fill_folio_gaps(folios)
    assert 11 not in numbers
    assert distrusted == [12] or refused


def test_a_disbelieved_reading_is_not_used_as_an_anchor(vtb):
    """The audit demotes a misread to unconfident; it must not anchor a fill."""
    folios = [F(vtb, 10, 5), F(vtb, 11, 87, False), F(vtb, 12, 7)]
    numbers, _, _ = vtb.fill_folio_gaps(folios)
    assert numbers[11] == 6, "the disbelieved 87 was treated as an anchor"


def test_mutually_supporting_misreads_are_distrusted_and_repaired(vtb):
    """The Invisible Government shipped a page-list running 215, 16, 16, 218:
    two adjacent misreads that supported each other, so neither looked wrong
    alone. Order is the judge — folios strictly increase through a book — and
    once the false pair is cut, the same pages interpolate back to the 216
    and 217 the printer actually bound there."""
    folios = [F(vtb, 0, 213), F(vtb, 1, 214), F(vtb, 2, 215),
              F(vtb, 3, 16), F(vtb, 4, 16),
              F(vtb, 5, 218), F(vtb, 6, 219), F(vtb, 7, 220)]
    numbers, refused, distrusted = vtb.fill_folio_gaps(folios)
    assert distrusted == [3, 4]
    assert (numbers[3], numbers[4]) == (216, 217)
    assert refused == []
    vals = [numbers[i] for i in sorted(numbers)]
    assert vals == sorted(vals), "page-list would not be monotonic"


def test_a_genuine_early_folio_is_not_distrusted_for_the_pairs_sins(vtb):
    """The real folio 16 lives early in the book inside the increasing run;
    only the impostors sitting out of order are cut."""
    folios = [F(vtb, 0, 15), F(vtb, 1, 16), F(vtb, 2, 17),
              F(vtb, 3, 215), F(vtb, 4, 16), F(vtb, 5, 217)]
    numbers, _, distrusted = vtb.fill_folio_gaps(folios)
    assert numbers[1] == 16
    assert distrusted == [4]
    assert numbers[4] == 216, "the cut page did not interpolate back"


def _tiny_book(vtb, tmp_path, bodies):
    """Build a real EPUB through the library, so the page-list is the one a
    reading system would actually see."""
    from ebooklib import epub
    vtb._restrict_page_list_to_pagebreaks()
    book = epub.EpubBook()
    book.set_identifier("t"); book.set_title("T"); book.set_language("en")
    chapters = []
    for i, b in enumerate(bodies):
        c = epub.EpubHtml(title=f"p{i}", file_name=f"page_{i:04d}.xhtml",
                          lang="en")
        c.content = ('<html xmlns="http://www.w3.org/1999/xhtml" '
                     'xmlns:epub="http://www.idpf.org/2007/ops">'
                     f"<body>{b}</body></html>")
        book.add_item(c); chapters.append(c)
    book.toc = tuple(chapters)
    book.spine = chapters
    book.add_item(epub.EpubNcx()); book.add_item(epub.EpubNav())
    out = str(tmp_path / "t.epub")
    epub.write_epub(out, book)
    return out


def _page_list(path):
    z = zipfile.ZipFile(path)
    nav = next(n for n in z.namelist() if n.endswith("nav.xhtml"))
    h = z.read(nav).decode("utf-8")
    parts = re.split(r"(<nav[^>]*>)", h)
    for i in range(1, len(parts), 2):
        if "page-list" in parts[i]:
            return re.findall(r'<a[^>]*href="([^"]+)"[^>]*>([^<]*)</a>',
                              parts[i + 1])
    return []


def test_note_markers_are_not_filed_as_printed_pages(vtb, tmp_path):
    """The defect itself: a noteref carries epub:type and an id, and was
    counted as a page. 568 links became 567 phantom pages."""
    bodies = ['<p>Prose.<sup><a epub:type="noteref" id="ref-g0-1" '
              'href="page_0001.xhtml#note-g0-1">1</a></sup></p>',
              '<p id="note-g0-1"><a href="page_0000.xhtml#ref-g0-1">1.</a> '
              'A source.</p>']
    entries = _page_list(_tiny_book(vtb, tmp_path, bodies))
    assert entries == [], f"note markers leaked into the page-list: {entries}"


def test_real_pagebreaks_are_listed_with_their_printed_numbers(vtb, tmp_path):
    bodies = ['<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-0000" '
              'aria-label="7"></span><p>Chapter opens.</p>',
              '<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-0001" '
              'aria-label="8"></span><p>And continues.</p>']
    entries = _page_list(_tiny_book(vtb, tmp_path, bodies))
    assert [t for _, t in entries] == ["7", "8"]
    assert entries[0][0].endswith("page_0000.xhtml#pgb-0000")


def test_a_page_list_holds_page_numbers_and_nothing_else(vtb, tmp_path):
    """The outcome guard. If note entries ever take an epub:type of their own
    for popup support, this is what refuses to let the list rot again."""
    bodies = ['<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-0000" '
              'aria-label="7"></span>'
              '<p>Prose.<sup><a epub:type="noteref" id="ref-g0-1" '
              'href="page_0001.xhtml#note-g0-1">1</a></sup></p>',
              '<p id="note-g0-1" epub:type="endnote">'
              '<a href="page_0000.xhtml#ref-g0-1">1.</a> A source.</p>']
    entries = _page_list(_tiny_book(vtb, tmp_path, bodies))
    assert [t for _, t in entries] == ["7"]
    assert not any("ref-" in h or "note-" in h for h, _ in entries)


def test_the_pagebreak_marker_shows_nothing_to_the_reader(vtb):
    """It carries its number in aria-label, not as text: a visible "7" halfway
    down a paragraph would be the pipeline printing on the page."""
    span = ('<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-0000" '
            'aria-label="7"></span><p>Chapter opens.</p>')
    assert vtb._strip_tags(span).strip() == "Chapter opens."
