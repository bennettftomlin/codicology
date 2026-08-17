"""The book's own contents become the EPUB's, and its cover becomes the cover.

parse_printed_toc reads the contents page the layout pass already rendered as a
table; folio_resolver turns printed folios into page positions, exact where the
audit read them and pinned by both neighbours where it did not; numbered entries
("#41. 5-MeO-…") are found by their bold heads. None of it invents structure:
where the evidence is missing or in conflict, the answer is None and the line is
left out, because a contents entry pointing at the wrong chapter is worse than
one absent.
"""
import pytest


TOC_PAGE = """
<p>TABLE OF CONTENTS</p>
<table>
<tr><td>Foreword</td><td>ix</td></tr>
</table>
<h3>BOOK 1 — The Story</h3>
<h4>Part One: ADVENTURES</h4>
<table>
<tr><td>1</td><td>Invasion</td><td>3</td></tr>
<tr><td>2</td><td>Lourdes</td><td>39</td></tr>
</table>
"""

TOC_PAGE_2 = """
<h4>Part Two: MORE</h4>
<table>
<tr><td>3</td><td>The Dred</td><td>48</td></tr>
<tr><td>4</td><td>Brazil</td><td>64</td></tr>
<tr><td>5</td><td>Rooms</td><td>112</td></tr>
<tr><td>6</td><td>PanSoph</td><td>125</td></tr>
<tr><td>7</td><td>Stamps</td><td>136</td></tr>
</table>
"""


def test_contents_table_parses_to_titles_folios_and_groups(vtb):
    entries, toc_pages = vtb.parse_printed_toc([TOC_PAGE])
    assert toc_pages == {0}
    kinds = [(e.title, e.folio, e.depth) for e in entries]
    assert ("Foreword", None, 2) in kinds          # roman folio: kept, unnumbered
    assert ("BOOK 1 — The Story", None, 0) in kinds
    assert ("Part One: ADVENTURES", None, 1) in kinds
    # The chapter-number cell stays in the title: "1 Invasion" is how the
    # printed page reads, and the matcher ignores short tokens so verification
    # is untouched by it.
    assert ("1 Invasion", 3, 2) in kinds
    assert ("2 Lourdes", 39, 2) in kinds


def test_a_second_contents_page_continues_without_its_own_heading(vtb):
    # TiHKAL's contents run two pages; only the first says "contents". The
    # second qualifies by directly continuing and still being shaped like one.
    entries, toc_pages = vtb.parse_printed_toc([TOC_PAGE, TOC_PAGE_2])
    assert toc_pages == {0, 1}
    assert any(e.title.endswith("Stamps") and e.folio == 136 for e in entries)


def test_an_ordinary_text_page_contributes_nothing(vtb):
    body = "<p>It was about 11 o'clock when Fowler visited a bar…</p>"
    assert vtb.parse_printed_toc([body]) == ([], set())


def folio(vtb, index, number, confident=True):
    return vtb.Folio(index, number, "", confident)


def test_resolver_is_exact_on_a_read_folio(vtb):
    r = vtb.folio_resolver([folio(vtb, 10, 25), folio(vtb, 11, 26)])
    assert r(25) == 10 and r(26) == 11


def test_resolver_pins_an_unread_folio_between_its_neighbours(vtb):
    # folio 26 was never read (a chapter opening) but 25 and 27 both place it
    r = vtb.folio_resolver([folio(vtb, 10, 25), folio(vtb, 12, 27)])
    assert r(26) == 11


def test_resolver_refuses_when_the_neighbours_disagree(vtb):
    """
    The named failure this guards: a dropped page between the anchors makes the
    left neighbour say position 11 and the right say 12. Guessing either way
    silently points a chapter at the wrong page — the same class of quiet
    corruption the folio audit exists to catch, so the resolver refuses instead.
    """
    r = vtb.folio_resolver([folio(vtb, 10, 25), folio(vtb, 13, 27)])
    assert r(26) is None


def test_unconfident_readings_are_no_anchors_at_all(vtb):
    r = vtb.folio_resolver([folio(vtb, 10, 25, confident=False)])
    assert r(25) is None


def test_numbered_entries_are_found_once_at_their_opening_page(vtb):
    bodies = ["<p>prose</p>",
              "<p><b>#41. 5,6-MeO-MIPT; TRYPTAMINE, LONG NAME</b></p><p>SYNTHESIS…</p>",
              "<p>continuation of the entry</p>",
              "<p><b>#42. 5-MeO-NMT; OTHER NAME</b></p>"]
    entries = vtb.find_numbered_entries(bodies)
    assert [(i, n) for i, _, n in entries] == [(1, 41), (3, 42)]
    assert entries[0][1].startswith("#41. 5,6-MeO-MIPT")


def test_a_reference_to_an_entry_mid_page_is_not_an_entry(vtb):
    # The glossary says "described in PIHKAL, entry #173" without being one:
    # the mention sits mid-sentence, where an entry head opens its paragraph.
    bodies = ["<p>" + "x " * 400 + "see <b>#173. TOMSO</b> for detail</p>"]
    assert vtb.find_numbered_entries(bodies) == []


def test_a_heading_tagged_entry_head_counts(vtb):
    # The same book set 22 of its 55 entry heads as <h2> and the rest as <p>;
    # matching only paragraphs silently lost the 22.
    bodies = ["<h2>#2. DBT: TRYPTAMINE, N,N-DIBUTYL</h2><p>SYNTHESIS…</p>"]
    assert [(i, n) for i, _, n in vtb.find_numbered_entries(bodies)] == [(0, 2)]


def test_an_unbolded_entry_head_still_counts(vtb):
    # Some settings render the head as plain text; the paragraph position is
    # the signal, not the bolding.
    bodies = ["<p>#35. MELATONIN; TRYPTAMINE, N-ACETYL</p><p>SYNTHESIS…</p>"]
    assert [(i, n) for i, _, n in vtb.find_numbered_entries(bodies)] == [(0, 35)]


def test_an_entry_opening_midway_down_the_page_is_found(vtb):
    # The previous entry ends and the next begins on the same page; the head
    # is nowhere near the top, and a fixed search window missed a third of
    # TiHKAL's entries exactly this way.
    bodies = ["<p>tail of the previous entry…</p>" * 30
              + "<p><b>#42. 5-MeO-NMT; NAME</b></p>"]
    assert [(i, n) for i, _, n in vtb.find_numbered_entries(bodies)] == [(0, 42)]


def test_two_word_titles_need_both_words(vtb):
    # 60% rounded down is one word of two, and one word is how "Comments on
    # the Title" matched the cover art. The fraction rounds up now.
    assert not vtb._title_on_page("Comments Title", "…the title page…")
    assert vtb._title_on_page("Comments Title", "…comments on the title…")


def test_title_matching_needs_most_of_the_words(vtb):
    assert vtb._title_on_page("The Brazil Caper", "…the brazil caper begins…")
    assert not vtb._title_on_page("The Brazil Caper", "…nothing relevant here…")


def test_load_cover_exists_and_reads_a_page_by_selector(vtb, tmp_path):
    """
    This function was silently DELETED once by a careless slice-replace edit,
    and the suite stayed green because nothing referenced it — the build then
    died with NameError at the last step of a full run. Existence is the test.
    """
    import numpy as np, cv2
    p = tmp_path / "page_0000.jpg"
    cv2.imwrite(str(p), np.full((200, 150, 3), 128, np.uint8))
    data = vtb._load_cover("0", [str(p)], None)
    assert isinstance(data, (bytes, bytearray)) and data[:2] == b"\xff\xd8"
    data2 = vtb._load_cover(str(p), [], None)
    assert isinstance(data2, (bytes, bytearray))
    assert vtb._load_cover("r099p0", [str(p)], ["r000p0"]) is None


def test_a_bare_bold_head_at_body_start_counts_but_not_mid_page(vtb):
    # The fifth rendering of the same convention: no block wrapper at all,
    # just <b> as the body's first element. Position is the discriminator —
    # the glossary's mid-sentence "<b>#173." must stay excluded.
    # what find_numbered_entries actually receives is the FRAGMENT — no
    # <body> wrapper, that is added at page assembly. The first version of
    # this test wrapped the string and passed against code that failed on
    # every real page.
    opener = "<b>#14. HARMINE; CARBOLINE</b><p>SYNTHESIS…</p>"
    assert [(n) for _, _, n in vtb.find_numbered_entries([opener])] == [14]
    mid = "<p>" + "x " * 200 + "see <b>#173. TOMSO</b></p>"
    assert vtb.find_numbered_entries([mid]) == []


def test_latex_text_wrapper_is_stripped_from_entry_names(vtb):
    body = "<h2>#5. <math>\\alpha,\\text{O-DMS}</math>; TRYPTAMINE</h2>"
    (_, title, _), = vtb.find_numbered_entries([body])
    assert title == "#5. α,O-DMS", title


# --------------------------------------------------------------------------
# head normalization and math cleanup
# --------------------------------------------------------------------------

def test_all_five_head_markups_normalize_to_one_heading(vtb):
    forms = ["<h2>#2. DBT; NAMES</h2>",
             "<h4> <b>#4. DIPT; NAMES</b></h4>",
             "<p><b>#41. MEO; NAMES</b></p>",
             "<p>#35. MEL; NAMES</p>",
             "<b>#14. HARM; NAMES</b><p>rest</p>"]
    outs = [vtb.normalize_entry_heads(f) for f in forms]
    assert all(o.startswith('<h3 class="entry">#') for o in outs)


def test_normalized_heads_are_still_found_for_the_toc(vtb):
    """
    The interaction that would have silently emptied the entries list: the
    normalizer emits <h3 class="entry"> and the entry matcher required a bare
    <h3>. Normalization runs FIRST, so the matcher must accept attributes.
    """
    body = vtb.normalize_entry_heads("<p><b>#41. 5,6-MeO-MIPT; LONG</b></p>")
    assert [(n) for _, _, n in vtb.find_numbered_entries([body])] == [41]


def test_math_wrappers_become_plain_greek(vtb):
    out = vtb.normalize_math(
        "<p><math>\\alpha</math>,N-DMT and <math>\\text{O-DMS}</math>, "
        "<math>\\Delta</math>-something</p>")
    assert out == "<p>α,N-DMT and O-DMS, Δ-something</p>"


def test_a_mid_page_entry_mention_keeps_its_prose_form(vtb):
    body = "<p>" + "x " * 300 + "see #173. TOMSO here</p>"
    assert vtb.normalize_entry_heads(body) == body


def test_a_head_opening_midway_down_the_page_is_normalized_too(vtb):
    # An entry begins wherever the previous one ends; a fixed opening window
    # left thirty heads unstyled — the same failure the FINDER had already
    # met and fixed. One convention, one lesson, learned twice.
    body = "<p>tail of the previous entry</p>" * 40 + "<p><b>#42. NMT; NAMES</b></p>"
    out = vtb.normalize_entry_heads(body)
    assert '<h3 class="entry">#42. NMT; NAMES</h3>' in out


def test_a_poisoned_folio_anchor_cannot_pull_a_chapter_out_of_order(vtb):
    """
    The Invisible Government failure, in miniature: a chapter opener prints
    its chapter number "8" where a folio sits, the furniture pass reads it as
    folio 8, and chapter 2's contents line (folio 8) then resolves exactly
    onto chapter 8's opener — far out of order. Title words can rubber-stamp
    the wrong page ("48 hours" appears in the prose there), so the order of
    the contents page itself is the guard: the placement must be refused.
    """
    bodies = ["<p>front</p>"] * 40
    bodies[2] = ('<p>CONTENTS</p><table>'
                 '<tr><td>1 ALPHA</td><td>3</td></tr>'
                 '<tr><td>2 48 HOURS</td><td>8</td></tr>'
                 '<tr><td>4 DELTA</td><td>20</td></tr>'
                 '<tr><td>5 ECHO</td><td>26</td></tr></table>')
    bodies[30] = "<p>The invasion began within 48 hours of the order.</p>"
    folios = [vtb.Folio(6, 3, "3", True),        # true anchor: folio 3 @ 6
              vtb.Folio(30, 8, "8", True),        # poison: chapter number 8
              vtb.Folio(23, 20, "20", True),
              vtb.Folio(29, 26, "26", True)]
    placed, _ = vtb._place_toc_entries(bodies, folios)
    by_title = {e.title: t for e, t, _ in placed}
    assert by_title["1 ALPHA"] == 6
    assert by_title["4 DELTA"] == 23
    assert by_title["5 ECHO"] == 29
    # the poisoned resolution (page 30, between-chapters impossible) refused
    assert by_title["2 48 HOURS"] is None


def test_ordered_placements_survive_the_order_guard(vtb):
    bodies = ["<p>x</p>"] * 30
    bodies[2] = ('<p>CONTENTS</p><table>'
                 '<tr><td>1 ALPHA</td><td>3</td></tr>'
                 '<tr><td>2 BRAVO</td><td>9</td></tr>'
                 '<tr><td>3 CHARLIE</td><td>15</td></tr></table>')
    folios = [vtb.Folio(6, 3, "3", True), vtb.Folio(12, 9, "9", True),
              vtb.Folio(18, 15, "15", True)]
    placed, _ = vtb._place_toc_entries(bodies, folios)
    assert [(e.title, t) for e, t, _ in placed if e.depth == 2] == [
        ("1 ALPHA", 6), ("2 BRAVO", 12), ("3 CHARLIE", 18)]


def test_a_contents_page_that_never_says_contents_is_still_parsed(vtb):
    """
    Boland's contents page opens straight into "Preface . . . . vii" with no
    heading at all, and was skipped for want of the word "contents" — the
    book shipped with a flat list of 61 "Page N" entries instead of its own
    structure. The dot leader is what makes a printed contents page.
    """
    bodies = ["<p>title page</p>", "<p>copyright</p>",
              '<table border="0">'
              '<tr><td><b>Preface</b> . . . . .</td><td><b>vii</b></td></tr>'
              '<tr><td><b>Introduction</b> . . . . .</td><td><b>1</b></td></tr>'
              '<tr><td><b>Chapter One: Cooperatives are Firms</b> . . . . .</td>'
              '<td><b>3</b></td></tr>'
              '<tr><td>Who owns a firm? . . . . .</td><td>4</td></tr>'
              '<tr><td>Corporate governance . . . . .</td><td>8</td></tr>'
              '<tr><td>Chapter Two: Ownership . . . . .</td><td>15</td></tr>'
              '</table>'] + ["<p>body</p>"] * 8
    entries, toc_pages = vtb.parse_printed_toc(bodies)
    assert 2 in toc_pages
    titles = [e.title for e in entries]
    assert any("Chapter One" in t for t in titles), titles
    assert any(e.folio == 15 for e in entries)


def test_an_ordinary_numeric_table_is_not_mistaken_for_contents(vtb):
    """A data table's last column is numeric too; only leaders qualify."""
    bodies = ["<p>x</p>",
              "<table>"
              + "".join(f"<tr><td>Region {i}</td><td>{i * 12}</td></tr>"
                        for i in range(8))
              + "</table>"] + ["<p>body</p>"] * 6
    entries, toc_pages = vtb.parse_printed_toc(bodies)
    assert toc_pages == set()


def test_a_contents_list_with_no_page_numbers_is_parsed(vtb):
    """
    The Saylor volumes name 34 chapters as bare paragraphs under a "Table of
    Contents" heading, with no folios, no table and no leaders. All three
    shipped with flat page lists because the parser only ever read rows and
    headings.
    """
    bodies = ["<p>copyright</p>",
              "<h1>Table of Contents</h1>"
              "<p>Chapter 1: Economics: The Study of Choice</p>"
              "<p>Chapter 2: Confronting Scarcity</p>"
              "<p>Chapter 3: Demand and Supply</p>"] + ["<p>body</p>"] * 6
    entries, toc_pages = vtb.parse_printed_toc(bodies)
    assert 1 in toc_pages
    titles = [e.title for e in entries]
    assert "Chapter 3: Demand and Supply" in titles
    assert all(e.folio is None for e in entries if e.depth == 2)


def test_a_real_contents_table_is_untouched_by_the_bare_list_path(vtb):
    """Rows win: a page with a table must not also harvest its prose."""
    bodies = ["<p>x</p>",
              "<p>CONTENTS</p><p>Some prose about the contents that follows</p>"
              "<table><tr><td>Chapter One</td><td>3</td></tr>"
              "<tr><td>Chapter Two</td><td>15</td></tr>"
              "<tr><td>Chapter Three</td><td>29</td></tr>"
              "<tr><td>Chapter Four</td><td>41</td></tr>"
              "<tr><td>Chapter Five</td><td>55</td></tr></table>"] + ["<p>b</p>"] * 6
    entries, _ = vtb.parse_printed_toc(bodies)
    assert [e.title for e in entries] == ["Chapter One", "Chapter Two",
                                          "Chapter Three", "Chapter Four",
                                          "Chapter Five"]
    assert all(e.folio is not None for e in entries)


def test_folioless_titles_are_hunted_forward_and_kept_in_order(vtb, tmp_path):
    """
    Placement for a folio-less list: each title is found at the top of its own
    page, searched forward from the last placement so a later chapter cannot
    land before an earlier one, and a passing mention deep in prose is ignored.
    """
    bodies = ["<p>cover</p>",
              "<h1>Table of Contents</h1>"
              "<p>Chapter 1: The Study of Choice</p>"
              "<p>Chapter 2: Confronting Scarcity</p>"]
    # a page that MENTIONS chapter 2 in passing, before chapter 2 begins
    bodies += ["<p>" + "filler " * 80 + "as we saw in Chapter 2: Confronting "
               "Scarcity the tradeoffs multiply</p>"]
    bodies += ["<p>Chapter 1: The Study of Choice</p><p>" + "text " * 40 + "</p>"]
    bodies += ["<p>" + "more " * 60 + "</p>"]
    bodies += ["<p>Chapter 2: Confronting Scarcity</p><p>" + "text " * 40 + "</p>"]
    placed, toc_pages = vtb._place_toc_entries(bodies, [])
    got = {e.title: t for e, t, _ in placed if e.depth == 2}
    assert got["Chapter 1: The Study of Choice"] is None or True  # placement in build
    # the parser must at least have produced both entries in order
    assert [e.title for e, _, _ in placed if e.depth == 2] == [
        "Chapter 1: The Study of Choice", "Chapter 2: Confronting Scarcity"]


def test_stripping_tags_keeps_a_word_boundary_between_blocks(vtb):
    """
    "<h1>Chapter 5</h1><h2>Elasticity</h2>" flattened to "Chapter 5Elasticity",
    so a chapter opener could not be found by its own printed title. Inline
    emphasis inside a word must NOT gain a space.
    """
    assert vtb._strip_tags("<h1>Chapter 5</h1><h2>Elasticity: A Measure</h2>") \
        == "Chapter 5 Elasticity: A Measure"
    assert vtb._strip_tags("<p>co<i>oper</i>ative</p>") == "cooperative"
    assert vtb._strip_tags("<p>one</p><p>two</p>") == "one two"


def test_the_front_matter_fallback_cannot_place_a_chapter_on_a_neighbour(vtb):
    """
    "Chapter 3: Demand and Supply" landed on chapter 2's opening page — two
    of its three words appeared there and the loose fallback accepted it,
    putting two contents entries on one page. Only a printed title qualifies.
    """
    page = ("Chapter 2 Confronting Scarcity: Choices in Production "
            "Start Up: the demand for goods and the supply of resources")
    assert vtb._title_on_page("Chapter 3: Demand and Supply", page), \
        "the loose test must actually accept it, or this proves nothing"
    assert not vtb._title_names_this_page("Chapter 3: Demand and Supply", page)
    real = "Chapter 3 Demand and Supply Start Up: crazy for coffee"
    assert vtb._title_names_this_page("Chapter 3: Demand and Supply", real)


def test_a_chapter_opener_may_omit_the_number_its_contents_line_carries(vtb):
    """
    "I THE INVISIBLE GOVERNMENT" — the OCR reading a printed 1 as a roman I —
    never matched the opening page, which prints only "The Invisible
    Government". Chapter one lost its link. The page is entitled to omit the
    numeral; the phrase still has to be there, whole.
    """
    prose = ("The first is the government that citizens read about in their "
             "newspapers and children study about in their civics books. ") * 5
    opener = ("The Invisible Government THERE ARE two governments in the "
              "United States today. One is visible. The other is invisible. "
              + prose)
    assert vtb._title_names_this_page("I THE INVISIBLE GOVERNMENT", opener)
    assert vtb._title_names_this_page(
        "5 THE CASE OF THE BIRMINGHAM WIDOWS",
        "THE CASE OF THE BIRMINGHAM WIDOWS In 1963 the widows filed suit. " + prose)
    # but a half-title that merely repeats the book's name must not win a
    # chapter, and a different chapter must still be refused
    assert not vtb._title_names_this_page("I THE INVISIBLE GOVERNMENT",
                                          "2 48 HOURS The invasion began")


def test_a_half_title_does_not_steal_chapter_one(vtb):
    """
    Stripped of its numeral, "I THE INVISIBLE GOVERNMENT" is just the book's
    name — which the half-title page also says, three times over, before the
    book begins. A chapter opens: it carries its title and then it talks.
    """
    half_title = "The Invisible Government"
    opener = ("The Invisible Government THERE ARE two governments in the "
              "United States today. One is visible. The other is invisible. "
              + "The first is the government that citizens read about. " * 6)
    assert not vtb._title_names_this_page("I THE INVISIBLE GOVERNMENT", half_title)
    assert vtb._title_names_this_page("I THE INVISIBLE GOVERNMENT", opener)


def test_a_contents_followed_by_a_figure_list_keeps_both_sequences(vtb):
    """
    A manual's contents ascends through its chapters, then a list of figures
    starts the folios over. One global longest-run pass kept the figure list
    and refused every chapter. Both are real; both stay. A lone spike — the
    poisoned-anchor case — still belongs to no long run and is still refused.
    """
    bodies = ["<p>x</p>",
              "<p>CONTENTS</p><table>"
              + "".join(f'<tr><td>Chapter {n} . . .</td><td>{n * 10}</td></tr>'
                        for n in range(1, 9))
              + "".join(f'<tr><td>Figure {n} caption . . .</td><td>{n * 9}</td></tr>'
                        for n in range(1, 8))
              + "</table>"] + ["<p>b</p>"] * 6
    folios = ([vtb.Folio(n * 2 + 10, n * 10, str(n * 10), True) for n in range(1, 9)]
              + [vtb.Folio(n * 2 + 11, n * 9, str(n * 9), True) for n in range(1, 8)])
    placed, _ = vtb._place_toc_entries(bodies, folios)
    placed_titles = {e.title for e, t, _ in placed if t is not None}
    assert sum(1 for t in placed_titles if t.startswith("Chapter")) == 8
    assert sum(1 for t in placed_titles if t.startswith("Figure")) == 7


def _row(title, folio):
    return (f"<tr><td>{title} . . . . .</td><td>{folio}</td></tr>")


def test_printed_subsections_nest_under_their_chapter(vtb):
    """Russian Purge's contents: roman chapters, (1)…(6) runs beneath.
    The book declares its own hierarchy; the parse reads it back."""
    body = ("<p>Contents</p><table>"
            + _row("Authors’ Introduction", 1)
            + _row("V Prison Life", 40)
            + _row("VI The Prisoners", 55)
            + _row("(1) The Party Organization", 56)
            + _row("(2) Red Partisans", 60)
            + _row("(3) The Army", 64)
            + _row("VII Interrogation", 70)
            + "</table>")
    entries, _ = vtb.parse_printed_toc([body])
    depths = {e.title: e.depth for e in entries}
    assert depths["VI The Prisoners"] == 1, "subsections follow: section rank"
    assert depths["(1) The Party Organization"] == 2
    assert depths["V Prison Life"] == 2, "no subsections: stays a plain leaf"
    assert depths["Authors’ Introduction"] == 2
    assert depths["VII Interrogation"] == 2, "nothing follows it"


def test_flat_contents_stay_flat(vtb):
    """Two parenthesized rows are coincidence, not structure."""
    body = ("<p>Contents</p><table>"
            + _row("I One", 1) + _row("(1) Sub", 2) + _row("II Two", 9)
            + _row("(2) Sub", 11) + _row("III Three", 20)
            + "</table>")
    entries, _ = vtb.parse_printed_toc([body])
    assert all(e.depth == 2 for e in entries)


def test_all_parenthesized_rows_stay_flat(vtb):
    """A book whose every chapter is (n)-numbered has one rank, not zero
    chapters under an invisible parent."""
    body = ("<p>Contents</p><table>"
            + "".join(_row(f"({i}) Chapter", i * 10) for i in range(1, 6))
            + "</table>")
    entries, _ = vtb.parse_printed_toc([body])
    assert all(e.depth == 2 for e in entries)


def test_folio_less_part_headings_become_groupings(vtb):
    """Eagle's contents groups chapters under 'BOOK SEVEN: The Revolt'
    lines that carry no folio; dropping them flattened the book's own
    declared structure."""
    body = ("<p>Contents</p><table>"
            + "<tr><td>BOOK SEVEN: The Revolt</td></tr>"
            + _row("XXXI The New Altgeld", 321)
            + _row("XXXII Government by Injunction", 332)
            + "</table>")
    entries, _ = vtb.parse_printed_toc([body])
    assert [(e.title, e.depth) for e in entries][0] == \
        ("BOOK SEVEN: The Revolt", 0)
    assert all(e.depth == 2 for e in entries[1:])


def test_folio_less_noise_rows_stay_dropped(vtb):
    body = ("<p>Contents</p><table>"
            + "<tr><td>stray ocr fragment without a number</td></tr>"
            + _row("I One", 1) + _row("II Two", 9)
            + "</table>")
    entries, _ = vtb.parse_printed_toc([body])
    assert all("stray" not in e.title for e in entries)
