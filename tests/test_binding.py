"""Pages bound into chapters, and taken apart again.

One file to a page makes a reading system start a new screen wherever the
paper ran out, which on one shelf is inside a sentence at 18,663 of 37,359
page turns. Bound, a paragraph that runs over the page is one paragraph,
and what stood at the foot of the page waits until it has ended. Everything
that checks a book still reads pages, so the binding has to come apart
exactly: what unbind() gives back is what bind() was given.
"""
import re
import zipfile

import pytest

from codicology import binding as B

FOLIO = '<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-%04d" aria-label="%d"></span>'


def _back(pages):
    return B.unbind(B.bind(pages))


# ── a paragraph that runs over the page ──────────────────────────────────────

def test_a_sentence_that_runs_over_the_page_is_one_paragraph():
    pages = [(7, "<p>It was a bright cold day in April, and the clocks were</p>"),
             (8, (FOLIO % (8, 4)) + "<p>striking thirteen.</p><p>Winston Smith slipped in.</p>")]
    flow = B.bind(pages)
    assert flow == ('<span class="pg" id="page-0007"></span>'
                    "<p>It was a bright cold day in April, and the clocks were "
                    '<span class="pg pg-joined" id="page-0008"></span>'
                    + (FOLIO % (8, 4)) +
                    "striking thirteen.</p><p>Winston Smith slipped in.</p>")
    assert B.unbind(flow) == pages


def test_a_paragraph_that_ends_with_its_page_is_left_alone():
    pages = [(1, "<p>The door closed.</p>"), (2, "<p>nobody spoke for a while.</p>")]
    assert "pg-joined" not in B.bind(pages)
    assert _back(pages) == pages


def test_the_mark_of_a_note_after_the_stop_is_not_the_sentence_running_on():
    pages = [(1, '<p>Senior transfers were exempted.<sup><a href="#fn-p1-n32">32</a></sup></p>'),
             (2, "<p>The rationale for restriction was put plainly.</p>")]
    assert "pg-joined" not in B.bind(pages)
    # nor with another note earlier in the paragraph: the marks at its end
    # are taken off, and not everything from the first mark on
    pages = [(1, '<p>It needs no subscription as do mailing services like riseup<sup>5</sup> or aktivix.<sup>6</sup></p>'),
             (2, "<p>As we have seen, Twitter has proved attractive.</p>")]
    assert "pg-joined" not in B.bind(pages)


def test_a_signature_mark_at_the_foot_is_not_the_start_of_a_sentence():
    pages = [(1, "<p>and a great deal of time is</p><p>B B</p>"),
             (2, "<p>saved by carefully indexing the contents.</p>")]
    assert "pg-joined" not in B.bind(pages)


@pytest.mark.parametrize("a, b, joins", [
    # taken up by a capital: prose that stops on a word
    ("the Qianlong Emperor banned all trade with the barbarians of the western ocean except in",
     "Canton on the Pearl River.", True),
    ("as the anthropologists of the period had shifted their attention during the 1920s and",
     "1930s from the Pacific to Africa.", True),
    # a short line and a capital: a title page, a line of verse
    ("John Newsinger", "First published 2018 by Pluto Press", False),
    ("Burst with words they knew", "We deserved to feel", False),
    # a caption closes nothing and opens nothing
    ("Figure 14.1 Components of gross private domestic investment by year and by sector, 1990 to 2008",
     "continued to rise.", False),
    ("the share of industry in the consumption of fossil fuels in the later years became a major sector fuel",
     "Table 4 Fossil fuel consumption, 1950–2015", False),
    ("Source: Clinton Fernandez", "is 73% owned by American-based investors.", False),
    # a running head the build let through, in the middle of a sentence
    ("Two groups, under a corporal, comprise a light-machine-gun", "COMPOSITION OF THE FORCE", False),
    # the mark of a note read as a plain figure after the stop; a decimal is not one
    ("does not require a special subscription as do activist mailing services like riseup5 or aktivix.6",
     "As we have seen, with its rapidity Twitter has proved attractive.", False),
    ("Nader was also the Green candidate in 2000 and reached an all-time high for the Greens, with 2.9",
     "million votes, 2.7%.", True),
    # an entry of an index; prose that stops on a figure
    ("social media 118, 119", "social networks continued", False),
    ("the payment of dividends to the common stock in no way cut into the $16,000,000",
     "accumulated by withholding dividends on the cumulative preferred;", True),
    # a first line and no more at the foot of the page
    ("Some of those in prison had succeeded in earning huge", "profits for the Soviet Union.", True),
    # a credit under a picture, a line of the colophon: a label and names
    ("Credit: AP Images/Ng Han Guan", "people. In March 2019, a chemical factory exploded.", False),
    ("Cover design: Hatim Eujayl", "the", False),
    # which a speaker's first words at the foot of the page are not
    ("Mathews: No, no. You heard him swear in the police", "court, and you said so yourself.", True),
])
def test_what_takes_up_a_paragraph_and_what_does_not(a, b, joins):
    pages = [(1, f"<p>{a}</p>"), (2, f"<p>{b}</p>")]
    assert ("pg-joined" in B.bind(pages)) is joins
    assert _back(pages) == pages


def test_a_compound_broken_at_its_hyphen_keeps_it_and_is_given_no_space():
    pages = [(1, "<p>a printed compound such as self-</p>"), (2, "<p>control keeps its hyphen.</p>")]
    assert ('such as self-control<span class="pg pg-joined" id="page-0002" data-back="7"></span>'
            " keeps its hyphen.</p>") in B.bind(pages)
    assert _back(pages) == pages


def test_a_word_broken_at_the_foot_of_the_page_is_one_word():
    """The build mends these where the page ends on the word; where notes or
    a figure follow the paragraph, the word was left in two."""
    feet = '<hr/><p id="fn-p0001-n1"><a href="#fnref-p0001-n1">1</a> Moody, 1904.</p>'
    pages = [(1, "<p>estimates of the 1900 con-</p>" + feet),
             (2, (FOLIO % (2, 344)) + "<p>centration, industry by industry.</p>")]
    flow = B.bind(pages)
    # the word is whole, and the page (and its printed number) begins after it
    assert ('1900 concentration,<span class="pg pg-joined pg-hyphen" id="page-0002" data-back="11"></span>'
            + (FOLIO % (2, 344)) + " industry by industry.</p>") in flow
    assert B.unbind(flow) == pages


def test_the_books_own_words_count_as_words():
    pages = [(1, "<p>what the miners called a Zwangs-</p>"), (2, "<p>arbeit of their own.</p>")]
    assert "pg-hyphen" not in B.bind(pages)
    words = B.book_words(["<p>Zwangsarbeit was abolished.</p>"])
    assert "Zwangsarbeit<span" in B.bind(pages, words)
    assert B.unbind(B.bind(pages, words)) == pages


def test_a_dash_with_a_space_before_it_has_one_after():
    pages = [(1, "<p>capital accumulated on the basis of social forms of production –</p>"),
             (2, "<p>feudalism, slavery – that were foreign to it.</p>")]
    assert "production – <span" in B.bind(pages)
    close = [(1, "<p>the immediate practical aftermath of arrival and of finding a place—</p>"),
             (2, "<p>accommodation, a job—posed many challenges.</p>")]
    assert "place—<span" in B.bind(close)


def test_lines_are_not_a_sentence():
    """An index is set as lines; the last entry of one page and the first of
    the next are two entries."""
    pages = [(1, "<p>policy ineffectiveness. See Rational<br/>Expectations Hypothesis<br/>politicians 84–5</p>"),
             (2, "<p>politics, rational choice theory in. See<br/>Public Choice</p>")]
    assert "pg-joined" not in B.bind(pages)


def test_a_paragraph_that_runs_over_three_pages():
    pages = [(1, "<p>It began on one page and</p>"),
             (2, "<p>went on through the whole of the next and</p>"),
             (3, "<p>ended on a third.</p><p>Then another.</p>")]
    flow = B.bind(pages)
    assert flow.count("<p>") == 2 and flow.count("pg-joined") == 2
    assert B.unbind(flow) == pages


def test_a_heading_a_list_or_a_blank_leaf_is_never_run_into():
    for last in ("<h2>Chapter Two</h2>", "<ul><li>one and</li></ul>",
                 '<p class="blank">[blank page 3]</p>'):
        pages = [(1, last), (2, "<p>and so it went on.</p>")]
        assert "pg-joined" not in B.bind(pages)
        assert _back(pages) == pages


# ── what stood at the foot of the page ───────────────────────────────────────

FEET = ('<hr/><p id="fn-p0001-n1"><a href="#fnref-p0001-n1">1</a> See the '
        'report of the committee.</p>')


def test_footnotes_wait_until_the_paragraph_has_ended():
    pages = [(1, '<p>The committee met<sup><a id="fnref-p0001-n1" href="#fn-p0001-n1">1</a></sup> and adjourned without</p>' + FEET),
             (2, "<p>deciding anything.</p><p>It met again in May.</p>")]
    flow = B.bind(pages)
    assert flow.index("deciding anything.</p>") < flow.index('class="pg-held"') \
        < flow.index("See the report") < flow.index("It met again")
    assert B.unbind(flow) == pages


def test_footnotes_stay_put_where_the_page_ends_its_paragraph():
    pages = [(1, "<p>The committee met and adjourned.</p>" + FEET),
             (2, "<p>It met again in May.</p>")]
    flow = B.bind(pages)
    assert "pg-held" not in flow and flow.index("See the report") < flow.index("page-0002")
    assert B.unbind(flow) == pages


def test_a_rule_between_two_sections_is_not_the_foot_of_the_page():
    pages = [(1, "<p>So ended the first part.</p><hr/><p>The second part began in the autumn and</p>"),
             (2, "<p>ran until the spring.</p>")]
    flow = B.bind(pages)
    assert "pg-held" not in flow and "pg-joined" in flow
    assert B.unbind(flow) == pages


def test_a_year_after_a_rule_is_the_next_section_and_not_a_note():
    pages = [(1, "<p>The committee met in the spring.</p><hr/><p>1944 was a hard year for the town, and it</p>"),
             (2, "<p>ended harder than it began.</p>")]
    flow = B.bind(pages)
    assert "pg-held" not in flow
    assert flow.index("1944 was a hard year") < flow.index("ended harder")
    assert "and it <span" in flow and B.unbind(flow) == pages
    # a note is numbered, and may be numbered into the hundreds
    noted = [(1, "<p>The committee met in the spring and then it</p><hr/><p>112. See the minutes.</p>"),
             (2, "<p>adjourned until autumn.</p>")]
    assert 'class="pg-held"' in B.bind(noted)


def test_a_figure_at_the_foot_and_one_at_the_head_are_stepped_round():
    fig = '<figure><img src="images/fig_%04d.png" alt="Figure"/></figure>'
    pages = [(1, "<p>By 2015 the office vacancy rate in Shenyang was nearly</p>" + fig % 1),
             (2, (FOLIO % (2, 10)) + fig % 2 + "<p>thirty per cent.</p><p>Next.</p>")]
    flow = B.bind(pages)
    assert flow.index("thirty per cent.</p>") < flow.index("fig_0001") < flow.index("fig_0002") \
        < flow.index("Next.")
    assert 'class="pg-held" data-page="0001"' in flow and 'class="pg-led" data-page="0002"' in flow
    assert B.unbind(flow) == pages


def test_a_figure_is_stepped_round_only_for_a_sentence_plainly_unfinished():
    fig = '<figure><img src="images/fig_0001.png" alt="Figure"/></figure>'
    pages = [(1, "<p>the Qianlong Emperor banned all trade with the barbarians of the western ocean except in</p>" + fig),
             (2, "<p>Canton was far from the capital.</p>")]
    assert "pg-joined" not in B.bind(pages)


# ── taken apart again ────────────────────────────────────────────────────────

def test_every_page_begins_at_its_own_mark_and_keeps_its_number():
    pages = [(4, (FOLIO % (4, 1)) + "<h1>One</h1><p>It began.</p>"),
             (5, (FOLIO % (5, 2)) + "<p>It went on.</p>"),
             (9, "<p>It ended.</p>")]
    flow = B.bind(pages)
    for page, _ in pages:
        assert f'id="page-{page:04d}"' in flow
    assert flow.count('epub:type="pagebreak"') == 2
    assert B.unbind(flow) == pages


def test_the_marks_are_read_however_the_file_was_written_out():
    """A library that writes the file may close an empty element in place."""
    flow = B.bind([(1, "<p>It was a bright cold day in April, and the clocks were</p>"),
                   (2, "<p>striking thirteen.</p>")])
    written = flow.replace("></span>", "/>")
    assert [p for p, _ in B.unbind(written)] == [1, 2]
    assert B.unbind(written)[0][1] == "<p>It was a bright cold day in April, and the clocks were</p>"


def test_the_pieces_of_a_page_at_its_top_level():
    body = ('<p>one <b>two</b></p><hr/><ul><li>a<ul><li>b</li></ul></li></ul>'
            'loose<img src="x.png"/><div><div>deep</div></div>')
    assert [t for t, _ in B.blocks(body)] == ["p", "hr", "ul", "", "img", "div"]
    assert "".join(h for _, h in B.blocks(body)) == body


# ── a whole book, written either way ─────────────────────────────────────────

XHTML = ('<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/'
         '1999/xhtml"><head><title>t</title></head><body>{}</body></html>')
PAGES = [(0, "<h1>A Book</h1>"),
         (1, "<p>It was a bright cold day in April, and the clocks were</p>"),
         (2, "<p>striking thirteen.</p>"),
         (5, "<h1>Two</h1><p>Winston Smith slipped in.</p>")]


def _epub(tmp_path, files):
    path = str(tmp_path / "book.epub")
    with zipfile.ZipFile(path, "w") as z:
        for name, body in files.items():
            z.writestr(name, XHTML.format(body))
    return path


def test_a_books_pages_are_the_same_whichever_way_it_was_written(tmp_path):
    by_page = _epub(tmp_path / "a" if (tmp_path / "a").mkdir() is None else tmp_path,
                    {f"EPUB/page_{i:04d}.xhtml": b for i, b in PAGES})
    (tmp_path / "b").mkdir()
    bound = _epub(tmp_path / "b", {"EPUB/ch_0000.xhtml": B.bind(PAGES[:3]),
                                   "EPUB/ch_0005.xhtml": B.bind(PAGES[3:]),
                                   "EPUB/nav.xhtml": "<nav/>"})
    assert B.page_bodies(by_page) == B.page_bodies(bound) == dict(PAGES)
    with zipfile.ZipFile(bound) as z:
        assert B.is_bound(z.namelist()) and not B.is_bound(["EPUB/page_0001.xhtml"])


def test_files_begin_where_chapters_do_and_never_inside_a_sentence():
    files = B.plan_files(PAGES, starts={0, 2, 5})
    # page 2 is the middle of a sentence: no file begins there
    assert [[p for p, _ in f] for f in files] == [[0, 1, 2], [5]]


def test_a_chapter_too_long_for_one_file_is_cut_where_a_page_ends_its_paragraph():
    pages = [(i, "<p>A page that ends its paragraph.</p>") for i in range(7)]
    pages[3] = (3, "<p>A page that does not end its paragraph and</p>")
    pages[4] = (4, "<p>runs on to the next.</p>")
    files = B.plan_files(pages, starts={0}, most=3)
    assert [[p for p, _ in f] for f in files] == [[0, 1, 2], [3, 4, 5], [6]]
    assert files[1][0][0] == 3


def test_links_follow_their_pages_into_the_files_they_are_bound_in():
    file_of = {12: "ch_0010.xhtml", 257: "ch_0250.xhtml"}
    assert B.relink('<a href="page_0257.xhtml#bib-0257-2380">Meulemann</a> '
                    '<a href="page_0012.xhtml">xii</a> <a href="#fn-p1-n1">1</a> '
                    '<a href="page_0999.xhtml#x">gone</a>', file_of) == (
        '<a href="ch_0250.xhtml#bib-0257-2380">Meulemann</a> '
        '<a href="ch_0010.xhtml#page-0012">xii</a> <a href="#fn-p1-n1">1</a> '
        '<a href="page_0999.xhtml#x">gone</a>')


# ── a book as the build writes it, bound and taken apart ─────────────────────

BOOK = [
    (0, "<h1>A Book</h1>"),
    (1, "<h1>One</h1><p>It was a bright cold day in April, and the clocks were</p>"
        '<hr/><p id="fn-p0001-n1"><a href="#fnref-p0001-n1">1</a> As the clocks of the period did.</p>'),
    (2, (FOLIO % (2, 4)) + "<p>striking thirteen. See <a href=\"page_0004.xhtml#bib-1\">Orwell</a> "
        'and <a href="page_0001.xhtml">the start</a>.</p>'),
    (3, (FOLIO % (3, 5)) + "<h1>Two</h1><p>Winston Smith slipped in.</p>"),
    (4, (FOLIO % (4, 6)) + '<p id="bib-1">Orwell, George. 1949.</p>'),
]


def _built(tmp_path, pages=BOOK, toc=((1, "One"), (3, "Two"))):
    """A book written the way build_epub writes one."""
    from ebooklib import epub
    from codicology import pipeline as P
    P._restrict_page_list_to_pagebreaks()
    book = epub.EpubBook()
    book.set_identifier("test")
    book.set_title("A Book")
    book.set_language("en")
    files = []
    for i, body in pages:
        ch = epub.EpubHtml(title=f"Page {i + 1}", file_name=f"page_{i:04d}.xhtml")
        ch.content = f"<html><body>{body}</body></html>"
        book.add_item(ch)
        files.append(ch)
    book.toc = tuple(epub.Link(f"page_{i:04d}.xhtml", t, f"c{i}") for i, t in toc) \
        if toc else tuple(files)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + files
    path = str(tmp_path / "book.epub")
    epub.write_epub(path, book)
    return path


def _members(path):
    with zipfile.ZipFile(path) as z:
        return {n: z.read(n).decode("utf-8") for n in z.namelist()
                if n.endswith((".xhtml", ".opf", ".ncx"))}


def test_a_built_book_is_bound_by_its_chapters_and_loses_nothing(tmp_path):
    path = _built(tmp_path)
    before = B.page_bodies(path)
    st = B.bind_epub(path)
    assert st == {"pages": 5, "files": 3, "joined": 1, "held": 1, "led": 0, "words": 0, "places": 0}
    got = _members(path)
    assert sorted(n for n in got if "ch_" in n or "page_" in n) == [
        "EPUB/ch_0000.xhtml", "EPUB/ch_0001.xhtml", "EPUB/ch_0003.xhtml"]
    # the pages are the pages they were, but for where their links point
    after = B.page_bodies(path)
    assert set(after) == set(before)
    for i in before:
        a = re.sub(r'href="[^"]*"', "", B._norm(before[i]))
        assert re.sub(r'href="[^"]*"', "", B._norm(after[i])) == a
    # the chapter's file carries the chapter's name; the sentence is whole
    one = got["EPUB/ch_0001.xhtml"]
    assert "<title>One</title>" in one
    assert re.search(r'the clocks were <span class="pg pg-joined" id="page-0002"></span>'
                     r'<span epub:type="pagebreak"[^>]*aria-label="4"\s*/>striking thirteen', one)
    assert one.index("striking thirteen") < one.index("As the clocks of the period")
    # every file of the book is one a strict reader will open
    import xml.etree.ElementTree as ET
    for name, text in got.items():
        ET.fromstring(text.encode("utf-8"))


def test_links_and_the_page_list_lead_where_they_led(tmp_path):
    path = _built(tmp_path)
    B.bind_epub(path)
    got = _members(path)
    one = got["EPUB/ch_0001.xhtml"]
    assert 'href="ch_0003.xhtml#bib-1"' in one       # to a thing on a page
    assert 'href="ch_0001.xhtml"' in one             # to the page a file opens with
    nav = got["EPUB/nav.xhtml"]
    assert 'href="ch_0001.xhtml">One<' in nav and 'href="ch_0003.xhtml">Two<' in nav
    assert 'href="ch_0001.xhtml#pgb-0002">4<' in nav
    assert 'href="ch_0003.xhtml#pgb-0004">6<' in nav
    assert 'src="ch_0003.xhtml"' in got["EPUB/toc.ncx"]
    opf = got["EPUB/content.opf"]
    assert opf.count(".xhtml") == 4 and "page_" not in opf      # three files and the contents
    assert opf.count("<itemref") == 4
    # and every name a link points at is in the file it points at
    for text in got.values():
        for f, frag in re.findall(r'(?:href|src)="(ch_\d{4}\.xhtml)(?:#([^"]*))?"', text):
            assert not frag or f'id="{frag}"' in got["EPUB/" + f]


def test_a_bound_book_comes_apart_into_the_book_it_was(tmp_path):
    path = _built(tmp_path)
    before, files = B.page_bodies(path), _members(path)
    bound = str(tmp_path / "bound.epub")
    B.bind_epub(path, out=bound)
    assert B.page_bodies(path) == before             # the original is untouched
    back = str(tmp_path / "back.epub")
    assert B.unbind_epub(bound, out=back) == {"pages": 5, "files": 3}
    assert B.same_pages(sorted(B.page_bodies(back).items()), sorted(before.items()))
    again = _members(back)
    assert sorted(again) == sorted(files)
    assert re.findall(r'href="[^"]*"', again["EPUB/nav.xhtml"]) == \
        re.findall(r'href="[^"]*"', files["EPUB/nav.xhtml"])
    assert again["EPUB/content.opf"].count("<itemref") == files["EPUB/content.opf"].count("<itemref")


def test_a_book_with_no_chapters_named_is_cut_where_pages_end_their_paragraphs(tmp_path):
    pages = [(i, f"<p>Page {i} ends its paragraph.</p>") for i in range(7)]
    path = _built(tmp_path, pages, toc=None)         # the contents is the list of its pages
    st = B.bind_epub(path, most=3)
    assert (st["pages"], st["files"]) == (7, 3)
    nav = _members(path)["EPUB/nav.xhtml"]
    assert 'href="ch_0000.xhtml#page-0001"' in nav and 'href="ch_0003.xhtml"' in nav


def test_a_book_is_not_bound_twice_and_one_never_built_here_is_refused(tmp_path):
    path = _built(tmp_path)
    B.bind_epub(path)
    with pytest.raises(B.CannotBind, match="already bound"):
        B.bind_epub(path)
    with pytest.raises(B.CannotBind, match="not bound"):
        B.unbind_epub(_built(tmp_path / "x" if (tmp_path / "x").mkdir() is None else tmp_path))


def test_a_page_that_would_make_a_broken_file_leaves_the_book_as_it_was(tmp_path):
    """The recogniser once wrote a link whose address never closed; the page
    was a file no strict reader opens, and bound in it would have taken a
    chapter with it."""
    path = _built(tmp_path)
    with zipfile.ZipFile(path) as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w") as z:
        for info, data in members:
            if info.filename.endswith("page_0003.xhtml"):
                data = data.replace(b"<p>Winston", b'<p><a 04="" href="x">y</a>Winston')
            z.writestr(info, data)
    was = _members(path)
    with pytest.raises(B.CannotBind, match="well-formed"):
        B.bind_epub(path)
    assert _members(path) == was


def test_one_name_on_two_pages_of_a_file_is_refused(tmp_path):
    pages = [(0, '<p id="x">One page ends here.</p>'), (1, '<p id="x">And so does this.</p>')]
    with pytest.raises(B.CannotBind, match="one name on two things"):
        B.bind_epub(_built(tmp_path, pages, toc=None))


# ── the checks go on reading pages ───────────────────────────────────────────

def test_the_checks_read_the_same_pages_from_a_bound_book(tmp_path):
    from codicology import adjudicate, compare
    path = _built(tmp_path)
    texts, words = adjudicate.epub_page_texts(path), compare.epub_pages(path)
    B.bind_epub(path)
    assert {i: t.split() for i, t in adjudicate.epub_page_texts(path).items()} == \
        {i: t.split() for i, t in texts.items()}
    assert compare.epub_pages(path) == words


def test_a_correction_lands_on_its_page_in_a_bound_book(tmp_path):
    import html
    import json
    from codicology import adjudicate as adj, review
    path = _built(tmp_path)

    def fix(page, old, new):
        """The decision a review sheet exports, addressed as it addresses
        one: by the word's place among the words of its page."""
        toks = adj.tokens(html.unescape(re.sub(r"<[^>]+>", " ", dict(BOOK)[page])))
        return {"page": page, "old": old, "new": new, "source": "human",
                "rung": "human", **adj.site_address(toks, toks.index(old))}

    # recorded against the book a file to a page, applied to it bound
    rows = [fix(2, "thirteen", "fourteen"), fix(1, "bright", "dull")]
    B.bind_epub(path)
    dec = tmp_path / "d.json"
    dec.write_text(json.dumps({"page_files": 5, "decisions": rows}))
    r = review.apply_decisions(path, str(dec))
    assert len(r["applied"]) == 2 and not r["stale"]
    pages = B.page_bodies(path)
    assert "striking fourteen." in pages[2] and "thirteen" not in pages[2]
    assert "a dull cold day" in pages[1]
    one = _members(path)["EPUB/ch_0001.xhtml"]
    # still one sentence, still one file, and nothing else moved
    assert re.search(r"the clocks were <span[^>]*pg-joined[^>]*></span><span[^>]*/>striking fourteen", one)
    assert one.count('id="page-0002"') == 1 and "<title>One</title>" in one
    import xml.etree.ElementTree as ET
    ET.fromstring(one.encode("utf-8"))


# ── the commands ─────────────────────────────────────────────────────────────

def test_convert_binds_the_book_it_built_unless_asked_not_to(tmp_path, monkeypatch, capsys):
    from codicology import pipeline as P
    built = _built(tmp_path)
    (tmp_path / "scans").mkdir()

    def stands_in_for_the_build(**kw):
        import shutil
        shutil.copy(built, kw["output_epub"])
    monkeypatch.setattr(P, "process_video", stands_in_for_the_build)

    out = str(tmp_path / "bound.epub")
    P.main(["--from-images", str(tmp_path / "scans"), "--epub", out])
    assert "Bound 5 page(s) into 3 file(s) by chapter: 1 paragraph(s)" in capsys.readouterr().out
    with zipfile.ZipFile(out) as z:
        assert B.is_bound(z.namelist())

    out = str(tmp_path / "pages.epub")
    P.main(["--from-images", str(tmp_path / "scans"), "--epub", out, "--page-files"])
    assert "Bound" not in capsys.readouterr().out
    with zipfile.ZipFile(out) as z:
        assert not B.is_bound(z.namelist())


def test_a_book_that_cannot_be_bound_is_still_a_book(tmp_path, capsys):
    from codicology import pipeline as P
    pages = [(0, '<p id="x">One page ends here.</p>'), (1, '<p id="x">And so does this.</p>')]
    path = _built(tmp_path, pages, toc=None)
    was = _members(path)
    assert P.bind_book(path) is None
    assert "Left a file to a page: pages 0–1 have one name on two things (x)" in capsys.readouterr().out
    assert _members(path) == was


def test_the_bind_and_unbind_verbs(tmp_path, capsys):
    from codicology import cli
    path = _built(tmp_path)
    before = B.page_bodies(path)
    assert cli.main(["bind", path]) == 0
    assert "bound 5 page(s) into 3 file(s)" in capsys.readouterr().out
    assert cli.main(["bind", path]) == 1
    assert "not bound: it is already bound" in capsys.readouterr().out
    back = str(tmp_path / "back.epub")
    assert cli.main(["unbind", path, "--out", back]) == 0
    assert "3 file(s) taken apart into 5 page(s)" in capsys.readouterr().out
    assert B.same_pages(sorted(B.page_bodies(back).items()), sorted(before.items()))


# ── where the reader had got to ──────────────────────────────────────────────

def _with_bookmarks(path, raw):
    with zipfile.ZipFile(path, "a") as z:
        z.writestr(B.BOOKMARKS, raw)
    return path


def _last_read(path):
    import base64
    import json
    with zipfile.ZipFile(path) as z:
        raw = z.read(B.BOOKMARKS).decode()
    return json.loads(base64.b64decode(raw.partition(":")[2]))


def test_the_readers_place_goes_with_its_page(tmp_path):
    """Calibre keeps where the reader stopped as a file of the spine and
    steps into it; bound, that file is another file."""
    import base64
    import json
    path = _built(tmp_path)
    # the spine is the contents, then the pages: page 2 is its fourth file,
    # and the reader was 20 letters into its paragraph
    rows = [{"pos": "epubcfi(/8/2/4/4/1:20)", "pos_type": "epubcfi", "type": "last-read",
             "timestamp": "2026-11-10T21:49:56+00:00"},
            {"pos": "epubcfi(/10/2/4/2/1:3)", "pos_type": "epubcfi", "type": "bookmark", "title": "Two"}]
    _with_bookmarks(path, "encoding=json+base64:\n" + base64.encodebytes(json.dumps(rows).encode()).decode())
    with zipfile.ZipFile(path) as z:
        assert [B.page_at(z, r["pos"]) for r in rows] == [2, 3]
    assert B.bind_epub(path)["places"] == 2
    got = _last_read(path)
    # page 2 begins inside the paragraph that is the third thing in the
    # chapter's file (a mark, a heading, the paragraph), at its first child;
    # page 3 opens the next file
    assert [r["pos"] for r in got] == ["epubcfi(/6/2/4/6/2)", "epubcfi(/8/2/4/2)"]
    assert got[0]["timestamp"] == rows[0]["timestamp"] and got[1]["title"] == "Two"
    with zipfile.ZipFile(path) as z:
        assert [B.page_at(z, r["pos"]) for r in got] == [2, 3]
        # a place further down the bound file is on the page that is there
        assert B.page_at(z, "epubcfi(/6/2/4/6/5:12)") == 2
        assert B.page_at(z, "epubcfi(/6/2/4/6/1:12)") == 1
        assert B.page_at(z, "epubcfi(/6/2/4/4/1:2)") == 1
    # and taken apart, the reader is at the top of the page they were on
    B.unbind_epub(path)
    assert [r["pos"] for r in _last_read(path)] == ["epubcfi(/8/2/4)", "epubcfi(/10/2/4)"]


def test_a_place_in_the_contents_stays_where_it_is(tmp_path):
    import base64
    import json
    rows = [{"pos": "epubcfi(/2/2/4/2[id]/4/6/2/1:3)", "pos_type": "epubcfi", "type": "last-read"}]
    path = _with_bookmarks(_built(tmp_path), "encoding=json+base64:\n"
                           + base64.encodebytes(json.dumps(rows).encode()).decode())
    assert B.bind_epub(path)["places"] == 1
    assert _last_read(path) == rows


def test_a_long_chapter_is_cut_by_its_weight_as_well_as_its_pages():
    pages = [(i, "<p>" + "A page that ends its paragraph. " * 40 + "</p>") for i in range(6)]
    assert [len(f) for f in B.plan_files(pages, starts=set(), weight=3000)] == [3, 3]


def test_the_place_as_calibre_once_wrote_it(tmp_path):
    path = _with_bookmarks(_built(tmp_path), "calibre_current_page_bookmark*|!|?|*3*|!|?|*/2/4/4/1:126\n")
    assert B.bind_epub(path)["places"] == 1
    with zipfile.ZipFile(path) as z:
        assert z.read(B.BOOKMARKS).decode() == "calibre_current_page_bookmark*|!|?|*2*|!|?|*/2/4/6/2\n"


def test_a_place_in_the_library_is_carried_from_one_copy_to_the_other(tmp_path):
    path = _built(tmp_path)
    bound = str(tmp_path / "bound.epub")
    B.bind_epub(path, out=bound)
    assert B.carried(path, bound, ["epubcfi(/8/2/4/4/1:20)", "epubcfi(/400/2/4)", "nonsense"]) == \
        ["epubcfi(/6/2/4/6/2)", None, None]


# ── found by a second reading of the code ────────────────────────────────────

def test_a_name_is_looked_for_in_the_file_its_link_names(tmp_path):
    """A name need only be the one of its kind in its file. Two chapters
    that each have an "x" are bound without complaint; taken apart, a link
    to the second chapter's must not come out as a link to the first's."""
    pages = [(0, '<h1>One</h1><p id="x">The first chapter ends here.</p>'),
             (1, '<p>It has a second page, and a <a href="page_0003.xhtml#x">link that led nowhere</a>.</p>'),
             (2, '<h1>Two</h1><p id="x">The second chapter ends here.</p>'),
             (3, '<p>And a <a href="page_0002.xhtml#x">link to its own first page</a>.</p>')]
    path = _built(tmp_path, pages, toc=((0, "One"), (2, "Two")))
    B.bind_epub(path)
    assert 'href="ch_0002.xhtml#x"' in _members(path)["EPUB/ch_0002.xhtml"]
    B.unbind_epub(path)
    got = B.page_bodies(path)
    assert 'href="page_0002.xhtml#x"' in got[3]
    assert 'href="page_0000.xhtml' not in got[1] and 'href="page_0000.xhtml' not in got[3]


def test_a_paragraph_over_three_pages_with_notes_on_each_through_a_whole_book(tmp_path):
    def foot(n):
        return f'<hr/><p id="fn-p{n:04d}-n1"><a href="#fnref-p{n:04d}-n1">1</a> A note to page {n}.</p>'
    pages = [(0, "<h1>One</h1><p>It began on one page and</p>" + foot(0)),
             (1, "<p>went on through the whole of the next and</p>" + foot(1)),
             (2, "<p>ended on a third.</p><p>Then another.</p>" + foot(2))]
    path = _built(tmp_path, pages, toc=((0, "One"),))
    before = B.page_bodies(path)
    st = B.bind_epub(path)
    assert (st["files"], st["joined"], st["held"]) == (1, 2, 2)
    one = _members(path)["EPUB/ch_0000.xhtml"]
    order = [one.index(x) for x in ("ended on a third.", "A note to page 0.", "A note to page 1.",
                                    "Then another.", "A note to page 2.")]
    assert order == sorted(order)
    assert B.same_pages(sorted(B.page_bodies(path).items()), sorted(before.items()))


def _damage(path, member, change):
    with zipfile.ZipFile(path) as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    with zipfile.ZipFile(path, "w") as z:
        for info, data in members:
            if info.filename.endswith(member):
                data = change(data)
                if data is None:
                    continue
            z.writestr(info, data)


@pytest.mark.parametrize("change, why", [
    (lambda data: data.replace(b"Winston", b"Winst\xffn"), "not text as it should be"),
    (lambda data: None, "is named in its package and is not in it"),
])
def test_a_damaged_book_is_refused_and_left_exactly_as_it_was(tmp_path, change, why, capsys):
    import os
    from codicology import cli, pipeline as P
    path = _built(tmp_path)
    _damage(path, "page_0003.xhtml", change)
    was = open(path, "rb").read()
    with pytest.raises(B.CannotBind, match=why):
        B.bind_epub(path)
    assert cli.main(["bind", path]) == 1 and "not bound:" in capsys.readouterr().out
    assert P.bind_book(path) is None and "Left a file to a page" in capsys.readouterr().out
    assert open(path, "rb").read() == was and not os.path.exists(path + ".tmp")


def test_a_failure_nobody_foresaw_still_leaves_the_built_book(tmp_path, monkeypatch, capsys):
    from codicology import pipeline as P
    path = _built(tmp_path)
    was = open(path, "rb").read()
    monkeypatch.setattr(B, "plan_files", lambda *a, **k: 1 / 0)
    assert P.bind_book(path) is None
    assert "the binding failed (ZeroDivisionError" in capsys.readouterr().out
    assert open(path, "rb").read() == was


def test_what_the_package_says_a_page_holds_it_says_of_the_file(tmp_path):
    path = _built(tmp_path)
    _damage(path, "content.opf", lambda data: data.replace(
        b'<item href="page_0002.xhtml"', b'<item properties="mathml" href="page_0002.xhtml"'))
    B.bind_epub(path)
    opf = _members(path)["EPUB/content.opf"]
    item = re.search(r'<item [^>]*href="ch_0001.xhtml"[^>]*/>', opf).group(0)
    assert 'properties="mathml"' in item and 'media-type="application/xhtml+xml"' in item
    assert opf.count('properties="mathml"') == 1
    import xml.etree.ElementTree as ET
    ET.fromstring(opf.encode("utf-8"))


def test_bookmarks_that_are_not_as_calibre_writes_them_do_not_stop_a_bind(tmp_path):
    import base64
    import json
    rows = [None, {"type": "last-read", "title": "no place in it"},
            {"pos": 5, "pos_type": "epubcfi", "type": "bookmark"},
            {"pos": "epubcfi(/8/2/4/4/1:20)", "pos_type": "epubcfi", "type": "last-read"},
            {"pos": "epubcfi(/400/2/4)", "pos_type": "epubcfi", "type": "bookmark", "title": "nowhere"}]
    path = _with_bookmarks(_built(tmp_path), "encoding=json+base64:\n"
                           + base64.encodebytes(json.dumps(rows).encode()).decode())
    assert B.bind_epub(path)["places"] == 1
    # the one place that could be told is carried, the one that could not is
    # taken out, and every row that is not a place goes back as it came
    assert _last_read(path) == rows[:3] + [dict(rows[3], pos="epubcfi(/6/2/4/6/2)")]

    for raw in ("encoding=json+base64:\nnot base64 at all", "\x00\x01 nothing Calibre wrote"):
        again = _with_bookmarks(_built(tmp_path / str(len(raw)) if (tmp_path / str(len(raw))).mkdir() is None
                                       else tmp_path), raw)
        assert B.bind_epub(again)["places"] == 0
        with zipfile.ZipFile(again) as z:
            assert z.read(B.BOOKMARKS).decode() == raw and B.is_bound(z.namelist())


def test_the_original_is_kept_and_a_stale_decision_reported_on_a_bound_book(tmp_path, capsys):
    import html
    import json
    import os
    from codicology import adjudicate as adj, review
    path = _built(tmp_path)
    B.bind_epub(path)
    bound = open(path, "rb").read()
    toks = adj.tokens(html.unescape(re.sub(r"<[^>]+>", " ", dict(BOOK)[2])))
    good = {"page": 2, "old": "thirteen", "new": "fourteen", "source": "human", "rung": "human",
            **adj.site_address(toks, toks.index("thirteen"))}
    gone = dict(good, old="twelve", new="eleven")
    off = dict(good, page=40)
    dec = tmp_path / "d.json"
    dec.write_text(json.dumps({"page_files": 5, "decisions": [good, gone, off]}))
    r = review.apply_decisions(path, str(dec))
    assert len(r["applied"]) == 1 and len(r["stale"]) == 2
    said = capsys.readouterr().out
    assert "page 40 is not a page of this book" in said and "applied 1 correction(s), 2 stale" in said
    assert open(path + ".preapply", "rb").read() == bound
    assert "striking fourteen." in B.page_bodies(path)[2]
    # recorded against another count of pages, nothing is touched
    dec.write_text(json.dumps({"page_files": 9, "decisions": [good]}))
    after = open(path, "rb").read()
    assert review.apply_decisions(path, str(dec))["refused"] and open(path, "rb").read() == after


# ── a page's mark never stands inside a word ─────────────────────────────────

INSIDE = re.compile(r'[^\s>]<span class="pg[^>]*></span>(?:<span[^>]*epub:type="pagebreak"[^>]*>(?:</span>)?)*[^\s<]')


@pytest.mark.parametrize("a, b", [
    ("estimates of the 1900 con-", "centration, industry by industry."),
    ("estimates of the 1900 con-", "centration"),                          # the paragraph ends with the word
    ("estimates of the 1900 con-", 'centration.<sup><a href="#fn-p2-n3">3</a></sup> Moody says so.'),
    ("a printed compound such as self-", "control keeps its hyphen."),
    ("a printed compound such as self-", "<i>control</i> in italic."),       # nothing of the word to take: left
    ("the choice was and/", "or both of them, he said."),
    ("for the letter itself see www.stuff.co.nz/letters-to-the-", "editor/9652661 (accessed 26 November)."),
    ("R&amp;D spending in pharma-", "ceuticals rose &amp; rose."),
])
def test_where_the_page_turned_inside_a_word_it_begins_after_the_word(a, b):
    for head in ("", FOLIO % (2, 9)):
        pages = [(1, f"<p>{a}</p>"), (2, f"{head}<p>{b}</p>")]
        flow = B.bind(pages)
        assert "pg-joined" in flow
        if not b.startswith("<"):
            assert not INSIDE.search(flow), flow
        assert B.unbind(flow) == pages


def test_a_dash_between_two_words_is_between_two_words_already():
    pages = [(1, "<p>the aftermath of arrival and of finding a place—</p>"),
             (2, "<p>accommodation, a job—posed many challenges.</p>")]
    assert 'place—<span class="pg pg-joined" id="page-0002"></span>accommodation,' in B.bind(pages)
    assert _back(pages) == pages


def test_a_word_broken_at_each_of_two_turns():
    pages = [(1, "<p>It began with a long con-</p>"),
             (2, (FOLIO % (2, 5)) + "<p>centration of the whole manu-</p>"),
             (3, (FOLIO % (3, 6)) + "<p>facturing interest. Then it ended.</p>")]
    flow = B.bind(pages)
    assert "long concentration<span" in flow and "whole manufacturing<span" in flow
    assert not INSIDE.search(flow)
    assert B.unbind(flow) == pages


def test_a_book_bound_before_the_mark_was_moved_is_still_read():
    """As books were bound on 11 and 12 November 2026: the mark at the
    break, inside the word."""
    pages = [(1, "<p>estimates of the 1900 con-</p>"),
             (2, (FOLIO % (2, 344)) + "<p>centration, industry by industry.</p>")]
    old = ('<span class="pg" id="page-0001"></span><p>estimates of the 1900 con'
           '<span class="pg pg-joined pg-hyphen" id="page-0002"></span>' + (FOLIO % (2, 344))
           + "centration, industry by industry.</p>")
    assert B.unbind(old) == pages
    assert B.bind(B.unbind(old)) != old and B.unbind(B.bind(B.unbind(old))) == pages


def _as_bound_before(path):
    """Put a bound book's in-word marks back where they stood before."""
    with zipfile.ZipFile(path) as z:
        members = [(i, z.read(i.filename)) for i in z.infolist()]
    moved = 0
    with zipfile.ZipFile(path, "w") as z:
        for info, data in members:
            if B.BOUND_FILE.match(info.filename):
                text = data.decode("utf-8")

                def back(m):
                    n = int(m.group(3))
                    return (m.group(1)[:-n] + f'<span class="pg pg-joined pg-hyphen" id="page-{m.group(2)}"></span>'
                            + m.group(4) + m.group(1)[-n:])
                text, k = re.subn(r'([^<>]+)<span class="pg pg-joined pg-hyphen" id="page-(\d{4})" data-back="(\d+)"></span>'
                                  r'((?:<span[^>]*epub:type="pagebreak"[^>]*/>)?)', back, text)
                moved += k
                data = text.encode("utf-8")
            z.writestr(info, data)
    return moved


BROKEN = [(0, "<h1>One</h1><p>Estimates of the 1900 con-</p>"
              '<hr/><p id="fn-p0000-n1"><a href="#fnref-p0000-n1">1</a> Moody, 1904.</p>'),
          (1, (FOLIO % (1, 344)) + "<p>centration, industry by industry, were made later.</p>"),
          (2, (FOLIO % (2, 345)) + "<h1>Two</h1><p>Another chapter, with no word broken.</p>")]


def test_a_bound_book_is_bound_again_by_the_rules_as_they_now_are(tmp_path, capsys):
    from codicology import cli
    path = _built(tmp_path, BROKEN, toc=((0, "One"), (2, "Two")))
    B.bind_epub(path)
    new = _members(path)
    assert _as_bound_before(path) == 1
    old = _members(path)
    assert INSIDE.search(old["EPUB/ch_0000.xhtml"]) and old != new
    pages = B.page_bodies(path)
    assert B.rebind_epub(path) == {"files": 2, "changed": 1, "pages": 3}
    again = _members(path)
    assert not INSIDE.search(again["EPUB/ch_0000.xhtml"])
    assert "1900 concentration,<span" in again["EPUB/ch_0000.xhtml"]
    # the pages are the pages they were, and nothing else in the book moved
    assert B.page_bodies(path) == pages
    assert {n: t for n, t in again.items() if "ch_0000" not in n} == {n: t for n, t in old.items() if "ch_0000" not in n}
    assert again == new
    # a second time changes nothing; the verb says what it did
    assert B.rebind_epub(path)["changed"] == 0
    assert cli.main(["rebind", path]) == 0
    assert "0 of 2 file(s) bound again differently (3 pages, all as they were)" in capsys.readouterr().out
    unbound = _built(tmp_path / "u" if (tmp_path / "u").mkdir() is None else tmp_path)
    assert cli.main(["rebind", unbound]) == 1 and "not bound again: it is not bound" in capsys.readouterr().out


def test_the_printed_page_number_is_found_after_the_word(tmp_path):
    path = _built(tmp_path, BROKEN, toc=((0, "One"), (2, "Two")))
    B.bind_epub(path)
    one = _members(path)["EPUB/ch_0000.xhtml"]
    assert re.search(r'concentration,<span class="pg pg-joined pg-hyphen" id="page-0001" data-back="11"></span>'
                     r'<span epub:type="pagebreak"[^>]*id="pgb-0001"[^>]*aria-label="344"\s*/> industry', one)
    assert 'href="ch_0000.xhtml#pgb-0001">344<' in _members(path)["EPUB/nav.xhtml"]
    with zipfile.ZipFile(path) as z:
        assert B.page_places(z)[1] == "epubcfi(/4/2/4/6/2)"
