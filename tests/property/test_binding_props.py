"""Pages bound into one flow and taken apart again (binding.bind, unbind,
plan_files), checked on generated pages.

The rules, from binding.py:
  - unbind() of what bind() made gives back the pages it was given (the
    build refuses to bind a book where it does not: same_pages);
  - every page's start is marked, and a page's mark never stands inside a
    word: where the paper ran out inside one, the page begins after it;
  - a paragraph that runs over the page is one paragraph, with the turn
    marked inside it; taken apart, it is the two paragraphs it was;
  - the bound file is well-formed (the build refuses a book where it is
    not) and has one name on one thing;
  - plan_files never begins a file inside a paragraph bind() would join.

Pages are made from a small model of what the build writes: a printed page
number at the head, paragraphs (prose that may stop mid-sentence, a word
broken at the foot, a compound broken at its hyphen, a dash), headings,
lists, figures and tables, a blank leaf, loose text, a rule and footnotes at
the foot; and edge cases (an empty page, a page that is only a figure, a
page that opens with a heading).
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import pytest
from hypothesis import assume, given, note, strategies as st

from codicology import binding as B

FOLIO = ('<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-%04d" '
         'aria-label="%d"></span>')

LOWER = ["the", "of", "and", "clocks", "were", "striking", "river", "kept",
         "a", "in", "said", "books", "café", "naïve", "λόγος", "café",
         "it’s", "o’clock", "1904", "years,", "him;", "x&amp;y"]
UPPER = ["The", "It", "Winston", "Canton", "Ἀθῆναι", "Moody", "April", "“So"]
BROKEN = [("con-", "centration"), ("manu-", "facturing"), ("pharma-", "ceuticals"),
          ("indus-", "try"), ("self-", "control"), ("and/", "or"),
          ("place—", "accommodation"), ("1914–", "18"), ("letters-to-the-", "editor"),
          ("Ab-", "solute"), ("x-", "ray")]


@st.composite
def prose(draw, n_min=1, n_max=20, start=None, end=None):
    """Words of running text, with now and then a little markup round one."""
    n = draw(st.integers(n_min, n_max))
    ws = []
    for k in range(n):
        w = draw(st.sampled_from(LOWER if k or start != "upper" else UPPER))
        r = draw(st.integers(0, 14))
        if r == 0:
            w = f"<i>{w}</i>"
        elif r == 1:
            w = f"<b>{w}</b>"
        elif r == 2 and k:
            w = f'{w}<sup><a href="#fn-{draw(st.integers(1, 99))}">{draw(st.integers(1, 99))}</a></sup>'
        ws.append(w)
    if start == "upper" and ws:
        ws[0] = draw(st.sampled_from(UPPER))
    if end == "stop":
        ws.append(draw(st.sampled_from(["end.", "home!", "so?", "him.”", "it:"])))
    return " ".join(ws)


@st.composite
def page_body(draw, idx, rest=None):
    """One page as the build writes it, and the rest of the word it broke at
    its foot, if it did. `rest`: the rest of the word the page before broke."""
    carry = None
    parts = []
    if draw(st.booleans()):
        parts.append(FOLIO % (idx, idx + 1))
    shape = draw(st.sampled_from(["prose"] * 6 + ["empty", "figure only", "heading",
                                                   "list", "blank", "loose", "verse"]))
    if shape == "empty":
        return "".join(parts), None
    if shape == "figure only":
        return "".join(parts) + f'<figure><img src="images/fig_{idx:04d}.png" alt="Figure"/></figure>', None
    if shape == "blank":
        return "".join(parts) + f'<p class="blank">[blank page {idx + 1}]</p>', None
    if shape == "loose":
        return "".join(parts) + draw(prose(1, 5)), None
    if shape == "verse":
        return "".join(parts) + "<p>" + draw(prose(2, 5)) + "<br/>" + draw(prose(2, 5)) + "</p>", None
    # the page's first block: often the rest of a paragraph from the page before
    if shape == "heading":
        parts.append(f"<h2>{draw(prose(1, 4, start='upper'))}</h2>")
    if draw(st.integers(0, 3)) == 0:
        parts.append(f'<figure><img src="images/fig_{idx:04d}a.png" alt="Figure"/></figure>')
    first = "rest of word" if rest else draw(st.sampled_from(["lower", "upper", "rest of word"]))
    if first == "rest of word":
        rest = rest or draw(st.sampled_from([b for _, b in BROKEN]))
        if draw(st.integers(0, 5)) == 0:
            rest = f"<i>{rest}</i>"         # an italic word, broken at the turn
        parts.append("<p>" + rest + draw(st.sampled_from(["", ",", "."]))
                     + draw(st.sampled_from(["", " " + draw(prose(0, 12))])) + "</p>")
    else:
        parts.append("<p>" + draw(prose(1, 16, start=first, end=draw(
            st.sampled_from([None, "stop"])))) + "</p>")
    for _ in range(draw(st.integers(0, 2))):
        kind = draw(st.sampled_from(["p", "p", "h2", "ul", "table"]))
        if kind == "p":
            parts.append("<p>" + draw(prose(3, 16, start="upper", end="stop")) + "</p>")
        elif kind == "h2":
            parts.append(f"<h2>{draw(prose(1, 3, start='upper'))}</h2>")
        elif kind == "ul":
            parts.append("<ul>" + "".join(f"<li>{draw(prose(1, 4))}</li>"
                                          for _ in range(draw(st.integers(1, 3)))) + "</ul>")
        else:
            parts.append(f"<table><tr><td>{draw(prose(1, 3))}</td></tr></table>")
    # the page's last paragraph: often one that stops mid-sentence
    last = draw(st.sampled_from(["mid", "mid", "broken", "stop", "none"]))
    if last == "mid":
        parts.append("<p>" + draw(prose(3, 20, start="upper")) + "</p>")
    elif last == "broken":
        a, carry = draw(st.sampled_from(BROKEN))
        if draw(st.integers(0, 5)) == 0:
            a = f"<i>{a}</i>"
        parts.append("<p>" + draw(prose(3, 16, start="upper")) + " " + a + "</p>")
    elif last == "stop":
        parts.append("<p>" + draw(prose(3, 16, start="upper", end="stop")) + "</p>")
    if draw(st.integers(0, 3)) == 0:
        parts.append(f'<figure><img src="images/fig_{idx:04d}b.png" alt="Figure"/></figure>')
    if draw(st.integers(0, 3)) == 0:
        n = draw(st.integers(1, 9))
        parts.append(f'<hr/><p id="fn-p{idx:04d}-n{n}"><a href="#fnref-p{idx:04d}-n{n}">{n}</a> '
                     + draw(prose(2, 8, start="upper", end="stop")) + "</p>")
    return "".join(parts), carry


@st.composite
def book(draw, min_pages=1, max_pages=6):
    n = draw(st.integers(min_pages, max_pages))
    start = draw(st.integers(0, 50))
    gaps = draw(st.lists(st.integers(1, 3), min_size=n, max_size=n))
    idx, out, carry = start, [], None
    for g in gaps:
        body, carry = draw(page_body(idx, carry))
        out.append((idx, body))
        idx += g
    return out


NS = ('<body xmlns="http://www.w3.org/1999/xhtml" '
      'xmlns:epub="http://www.idpf.org/2007/ops">%s</body>')


def well_formed(markup: str) -> bool:
    try:
        ET.fromstring(NS % markup)
        return True
    except ET.ParseError:
        return False


# a page's mark, with the printed page number the build may set after it
MARKS = re.compile(r'<span class="pg[^"]*" id="page-\d{4}"[^>]*></span>'
                   r'(?:<span[^>]*epub:type="pagebreak"[^>]*></span>)*')


def text_round_marks(flow: str) -> "list[tuple[str, str]]":
    """For each page mark: the text just before it and just after it, with
    every tag taken out (a word can run through <i>…</i>)."""
    def text(s):
        s = re.sub(r"</?(?:p|div|h\d|ul|ol|li|table|tr|td|figure|hr|br)\b[^>]*>", " ", s)
        return re.sub(r"<[^>]*>", "", s)
    out = []
    for m in MARKS.finditer(flow):
        before = text(flow[:m.start()])
        after = text(MARKS.sub("", flow[m.end():]))
        out.append((before[-12:], after[:12]))
    return out


def letter(ch: str) -> bool:
    return ch.isalnum() or ch in "́̈"


# ── the rules ────────────────────────────────────────────────────────────────

@given(book(), st.booleans())
def test_the_pages_come_apart_as_they_went_in(pages, with_words):
    words = B.book_words(b for _, b in pages) if with_words else None
    flow = B.bind(pages, words)
    note(f"flow={flow!r}")
    back = B.unbind(flow)
    assert [p for p, _ in back] == [p for p, _ in pages]
    assert B.same_pages(back, pages), back


@given(book())
def test_the_pages_come_apart_exactly(pages):
    """Not only the same pages to same_pages' eye: the same strings, but for
    white space between blocks, which a page's markup does not need."""
    flow = B.bind(pages, B.book_words(b for _, b in pages))
    back = B.unbind(flow)
    norm = lambda s: re.sub(r">\s+<", "><", s).strip()     # noqa: E731
    assert [(p, norm(b)) for p, b in back] == [(p, norm(b)) for p, b in pages]


@given(book())
def test_a_bound_flow_is_well_formed_and_names_each_thing_once(pages):
    for _, body in pages:
        assume(well_formed(body))
    flow = B.bind(pages, B.book_words(b for _, b in pages))
    note(f"flow={flow!r}")
    assert well_formed(flow)
    ids = re.findall(r'\bid="([^"]*)"', flow)
    assert len(ids) == len(set(ids))
    for page, _ in pages:
        assert ids.count("page-%04d" % page) == 1


@given(book(min_pages=2))
def test_a_page_mark_never_stands_inside_a_word(pages):
    flow = B.bind(pages, B.book_words(b for _, b in pages))
    note(f"flow={flow!r}")
    for before, after in text_round_marks(flow):
        assert not (before and after and letter(before[-1]) and letter(after[0])), \
            (before, after)


@pytest.mark.xfail(strict=False, reason="documented exception: a word whose "
                   "rest opens in markup (self- | <i>control</i>) is left, mark "
                   "and all, at the break")
@given(book(min_pages=2))
def test_a_compound_broken_at_its_hyphen_is_marked_after_the_whole_word(pages):
    """ "self-" | "control": the mark after "control", not after the hyphen."""
    flow = B.bind(pages, B.book_words(b for _, b in pages))
    note(f"flow={flow!r}")
    for before, after in text_round_marks(flow):
        if 'pg-joined' not in flow:
            continue
        assert not (len(before) > 1 and before[-1] in "-/" and letter(before[-2])
                    and after and letter(after[0])), (before, after)


@given(st.integers(0, 40), st.data())
def test_a_paragraph_run_over_the_page_is_one_and_comes_apart_as_two(start, data):
    """A paragraph of prose that stops mid-sentence on a small word, and a
    page that takes it up with a small one: one paragraph bound, with the
    turn marked in it, and the two paragraphs again taken apart."""
    a = data.draw(prose(B.PROSE_WORDS, 20, start="upper")) + " " + \
        data.draw(st.sampled_from(["the", "and", "were", "of"]))
    b = data.draw(st.sampled_from(["the", "striking", "river"])) + " " + \
        data.draw(prose(0, 10, end="stop"))
    head1 = data.draw(st.sampled_from(["", "<h2>One</h2>", "<p>Before it.</p>"]))
    tail2 = data.draw(st.sampled_from(["", "<p>Then another.</p>"]))
    folio = data.draw(st.sampled_from(["", FOLIO % (start + 1, 9)]))
    pages = [(start, f"{head1}<p>{a}</p>"), (start + 1, f"{folio}<p>{b}</p>{tail2}")]
    flow = B.bind(pages)
    note(f"flow={flow!r}")
    assert flow.count("<p>") == pages[0][1].count("<p>") + pages[1][1].count("<p>") - 1
    assert 'class="pg pg-joined" id="page-%04d"' % (start + 1) in flow
    joined = re.search(r"<p>((?:(?!</p>).)*pg-joined(?:(?!</p>).)*)</p>", flow).group(1)
    words = lambda s: re.sub(r"<[^>]*>", " ", s).split()     # noqa: E731
    assert words(joined) == words(a) + words(b)
    assert B.unbind(flow) == pages


@given(book(min_pages=2, max_pages=10), st.integers(1, 4), st.data())
def test_a_file_never_begins_inside_a_paragraph_bind_would_join(pages, most, data):
    starts = set(data.draw(st.lists(st.sampled_from([p for p, _ in pages]), max_size=3)))
    files = B.plan_files(pages, starts, most=most)
    assert [pg for f in files for pg in f] == pages
    whole = B.bind(pages)
    for f in files[1:]:
        first = f[0][0]
        assert 'class="pg" id="page-%04d"' % first in whole, (first, whole)


@given(book(min_pages=2, max_pages=8), st.data())
def test_bound_by_files_the_pages_still_come_apart(pages, data):
    """As bind_epub does it: the pages planned into files, each file bound
    and checked; then page_bodies' union of every file's pages."""
    files = B.plan_files(pages, set(), most=data.draw(st.integers(1, 5)))
    words = B.book_words(b for _, b in pages)
    back = [pg for f in files for pg in B.unbind(B.bind(f, words))]
    assert B.same_pages(back, pages)


@given(book())
def test_binding_what_was_taken_apart_binds_it_the_same(pages):
    """rebind_epub: a bound flow taken apart and bound again is the same
    flow, so a book bound by today's rules is left as it is."""
    words = B.book_words(b for _, b in pages)
    flow = B.bind(pages, words)
    assert B.bind(B.unbind(flow), words) == flow


@pytest.mark.xfail(strict=False, reason="finding: a word broken at the foot "
                   "is not made whole where either half stands in markup "
                   "(<i>manu-</i> | <i>facturing</i>); see test_binding_findings.py")
@given(st.data())
def test_a_word_broken_at_the_foot_is_one_word_however_it_is_set(data):
    """ "manu-" | "facturing" is "manufacturing", with the page's mark after
    it, whether or not the word is set in italic."""
    first, rest = data.draw(st.sampled_from([("manu", "facturing"), ("con", "centration"),
                                             ("pharma", "ceuticals")]))
    wrap_a, wrap_b = data.draw(st.sampled_from([("", ""), ("i", "i"), ("i", ""), ("", "i"),
                                                ("b", "b")]))
    a = f"{first}-" if not wrap_a else f"<{wrap_a}>{first}-</{wrap_a}>"
    b = rest if not wrap_b else f"<{wrap_b}>{rest}</{wrap_b}>"
    pages = [(1, f"<p>It was a matter for the whole of the {a}</p>"),
             (2, f"<p>{b} interest of the town.</p>")]
    flow = B.bind(pages)
    note(f"flow={flow!r}")
    assert B.unbind(flow) == pages
    text = re.sub(r"<[^>]*>", "", flow)
    assert first + rest in text, text


# ── a whole book, written and bound ──────────────────────────────────────────

def _write(tmp_path, pages, toc):
    """A book written the way build_epub writes one (as tests/test_binding)."""
    from ebooklib import epub
    from codicology import pipeline as P
    P._restrict_page_list_to_pagebreaks()
    bk = epub.EpubBook()
    bk.set_identifier("test")
    bk.set_title("A Book")
    bk.set_language("en")
    files = []
    for i, body in pages:
        ch = epub.EpubHtml(title=f"Page {i + 1}", file_name=f"page_{i:04d}.xhtml")
        ch.content = f"<html><body>{body}</body></html>"
        bk.add_item(ch)
        files.append(ch)
    bk.toc = tuple(epub.Link(f"page_{i:04d}.xhtml", f"Chapter {i}", f"c{i}") for i in toc) \
        if toc else tuple(files)
    bk.add_item(epub.EpubNcx())
    bk.add_item(epub.EpubNav())
    bk.spine = ["nav"] + files
    path = str(tmp_path / "book.epub")
    epub.write_epub(path, bk)
    return path


@given(book(min_pages=1, max_pages=7), st.data())
def test_a_book_bound_and_unbound_has_the_pages_it_had(tmp_path_factory, pages, data):
    """bind_epub: "page_bodies() of the result is page_bodies() of what it
    was given"; and unbind_epub gives the book back a file to a page."""
    for _, body in pages:
        # (ebooklib will not write a page with no element in it)
        assume(well_formed(body) and re.search(r"<(?:p|h2|figure|ul|table)\b", body))
    # a link to another page, which binding points at the page's new file
    if len(pages) > 1:
        to = data.draw(st.sampled_from([p for p, _ in pages]))
        k = data.draw(st.integers(0, len(pages) - 1))
        pages[k] = (pages[k][0], pages[k][1] + f'<p>See <a href="page_{to:04d}.xhtml">there</a>.</p>')
    toc = sorted(set(data.draw(st.lists(st.sampled_from([p for p, _ in pages]), max_size=3))))
    tmp = tmp_path_factory.mktemp("book")
    path = _write(tmp, pages, toc)
    before = B.page_bodies(path)
    bound = str(tmp / "bound.epub")
    try:
        B.bind_epub(path, out=bound, most=data.draw(st.integers(1, 4)))
    except B.CannotBind as exc:
        raise AssertionError(f"refused a book built as the build writes one: {exc}")
    # the same pages, but for links between them, which now name the files
    # the pages are bound in (and unbind_epub names the pages again)
    unlinked = lambda d: sorted((k, re.sub(r'href="(?:page|ch)_\d{4}\.xhtml[^"]*"', "", v))  # noqa: E731
                                for k, v in d.items())
    after = B.page_bodies(bound)
    assert B.same_pages(unlinked(after), unlinked(before))
    back = str(tmp / "back.epub")
    B.unbind_epub(bound, out=back)
    assert B.same_pages(sorted(B.page_bodies(back).items()), sorted(before.items()))
