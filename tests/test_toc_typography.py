"""The hierarchy a contents page declares by how it is SET.

Numbered prefixes — (1), (a) — are one convention and the parser has read
them for a while. The commoner one is typographic: the chapter is bold, and
what belongs under it is indented. Both survive recognition into the markup
and both were being discarded at parse time, so one book shipped forty
entries flat, its chapters level with their own subsections.
"""


def contents(rows, head=None):
    """A contents page as the recogniser hands it over: a table of rows,
    each a title cell and a folio cell."""
    out = [f"<h2>{head}</h2>"] if head else []
    out.append("<table>")
    for title, folio, bold, indent in rows:
        cell = f"<b>{title}</b>" if bold else title
        out.append(f"<tr><td>{' ' * indent}{cell}</td><td>{folio}</td></tr>")
    out.append("</table>")
    return ["<p>front matter</p>", "<h1>Contents</h1>" + "".join(out)]


def book_rows():
    return [("Introduction", 1, True, 0),
            ("Setting the Scene", 1, False, 4),
            ("Six Points", 5, False, 4),
            ("1 Currents of Queer", 9, True, 0),
            ("Community and Subversion", 9, False, 4),
            ("Phenomenally Queer", 16, False, 4),
            ("2 The Universal Alternative", 40, True, 0),
            ("Gay Politics in America", 41, False, 4),
            ("Wedded to Subversion", 52, False, 4)]


def test_bold_marks_the_chapter_and_plain_rows_nest_under_it(vtb):
    entries, _ = vtb.parse_printed_toc(contents(book_rows()))
    got = [(e.depth, e.title) for e in entries]
    assert (1, "Introduction") in got
    assert (1, "1 Currents of Queer") in got
    assert (2, "Phenomenally Queer") in got
    assert (1, "Phenomenally Queer") not in got


def test_indentation_speaks_when_nothing_is_bold(vtb):
    """A contents set without bold still indents what it subordinates."""
    rows = [(t, f, False, i) for t, f, _, i in book_rows()]
    entries, _ = vtb.parse_printed_toc(contents(rows))
    depths = {e.title: e.depth for e in entries}
    assert depths["1 Currents of Queer"] == 1
    assert depths["Phenomenally Queer"] == 2


def test_a_uniformly_bold_contents_stays_flat(vtb):
    """Bold everywhere says nothing about rank; inventing a hierarchy from
    it would nest arbitrary rows under their neighbours."""
    rows = [(t, f, True, 0) for t, f, _, i in book_rows()]
    entries, _ = vtb.parse_printed_toc(contents(rows))
    assert all(e.depth == 2 for e in entries)


def test_a_chapter_with_no_subsections_is_still_a_chapter(vtb):
    """Rank belongs to the level, not to what happens to follow. Requiring
    a subsection after each head left one book's Chapter Five sitting
    inside Chapter Four — set identically to it, but followed only by its
    own References, which the apparatus test had set aside. A chapter with
    nothing under it is a top-level entry, which is what the book prints."""
    rows = book_rows() + [("3 A Chapter Alone", 60, True, 0)]
    entries, _ = vtb.parse_printed_toc(contents(rows))
    depths = {e.title: e.depth for e in entries}
    assert depths["3 A Chapter Alone"] == 1
    assert depths["Phenomenally Queer"] == 2, "its subsections still nest"

def test_the_book_reprinting_its_title_does_not_adopt_the_chapters(vtb):
    """A contents running onto a second page reprints the book's title at
    the head of it. Taken as a division, that title adopted every chapter
    below it — chapters, notes and index all hung under 'AFTER QUEER
    THEORY' in one book's nav."""
    entries, _ = vtb.parse_printed_toc(
        contents(book_rows(), head="AFTER QUEER THEORY"),
        book_title="After Queer Theory")
    assert not any(e.title == "AFTER QUEER THEORY" for e in entries)


def test_a_division_that_names_no_division_is_still_a_grouping(vtb):
    """The test is identity with the book's title, not a vocabulary of
    division words: one book heads its parts 'HOW THE NORTHWEST WAS LOST
    TO FRANCE', which contains no such word and is unmistakably a part.
    Requiring BOOK/PART/VOLUME cost that book its structure."""
    half = book_rows()[:5], book_rows()[5:]
    page = (["<p>front</p>", "<h1>Contents</h1>"
             + "<h2>HOW THE NORTHWEST WAS LOST TO FRANCE</h2>"
             + contents(half[0])[1].split("</h1>", 1)[1]
             + "<h2>THE WINNING OF THE NORTHWEST</h2>"
             + contents(half[1])[1].split("</h1>", 1)[1]])
    entries, _ = vtb.parse_printed_toc(
        page, book_title="The Conquest of the Old Northwest")
    parts = [e for e in entries if e.title.startswith("HOW THE NORTHWEST")]
    assert parts and parts[0].depth == 0, [(e.depth, e.title) for e in entries]


def test_a_named_part_division_is_still_a_grouping(vtb):
    entries, _ = vtb.parse_printed_toc(
        contents(book_rows(), head="PART ONE: THE EARLY YEARS"),
        book_title="Some Other Book")
    parts = [e for e in entries if e.title.startswith("PART ONE")]
    assert parts and parts[0].depth == 0


def test_the_apparatus_is_not_a_subsection(vtb):
    """A book that lists chapters and no subsections prints bold chapters
    and a plain tail of Notes, References and Index. The last chapter is
    then the only one a plain row follows — and it adopted all three. Three
    books flattened into '7 Conclusion' parenting the back matter."""
    rows = [("Acknowledgments", 7, False, 0),
            ("1 Introduction", 1, True, 0),
            ("2 Working in the Call Centre", 20, True, 0),
            ("3 Management", 40, True, 0),
            ("4 Moments of Resistance", 60, True, 0),
            ("5 Precarious Organisation", 80, True, 0),
            ("6 Conclusion", 100, True, 0),
            ("Notes", 110, False, 0),
            ("References", 120, False, 0),
            ("Index", 130, False, 0)]
    entries, _ = vtb.parse_printed_toc(contents(rows))
    assert all(e.depth == 2 for e in entries), \
        [(e.depth, e.title) for e in entries if e.depth < 2]


def test_back_matter_is_lifted_out_of_the_last_chapter(vtb):
    """A contents ends with its apparatus set at the same indent as a
    chapter's subsections, so all of it disappeared inside whichever
    chapter came last — fourteen entries across seven books, including an
    index filed under a chapter it has nothing to do with."""
    rows = book_rows() + [("Notes", 200, False, 0),
                          ("Index", 210, False, 0)]
    entries, _ = vtb.parse_printed_toc(contents(rows))
    depths = {e.title: e.depth for e in entries}
    assert depths["1 Currents of Queer"] == 1
    assert depths["Phenomenally Queer"] == 2, "real subsections still nest"
    # …and the book's own back matter is not one of that chapter's sections
    assert depths["Notes"] == 1 and depths["Index"] == 1


def _banner_page(rows):
    """A contents page where some rows carry no folio at all."""
    out = ["<table>"]
    for title, folio, indent in rows:
        cell = f"<td>{' ' * indent}{title}</td>"
        out.append(f"<tr>{cell}<td>{folio}</td></tr>" if folio
                   else f"<tr>{cell}<td/></tr>")
    out.append("</table>")
    return ["<h1>Contents</h1>" + "".join(out)]


def test_a_part_named_for_all_it_contains_is_still_a_part(vtb):
    """Length was standing in for structure. A banner named after every
    country in it runs past any word count worth setting, and four of them
    were thrown away — taking the book's whole declared shape with them.
    What marks it is the rows indented underneath."""
    page = _banner_page([
        ("Introduction", "1", 0),
        ("Part I: YUGOSLAVIA, GREECE, POLAND AND LATVIA – Between the blocs",
         "", 0),
        ("2. Yugoslavia – Balancing Powers", "25", 4),
        ("3. Greece – Allies at War with the Resistance", "38", 4),
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert any(t.startswith("Part I:") for t in titles)


def test_a_book_that_indents_its_banners_instead_still_gets_them(vtb):
    """One book sets its chapters flush and indents the part lines, so
    nothing is ever set deeper than a banner. The series is the evidence
    there: Part I, Part II, Part III, each printed without a page number."""
    page = _banner_page([
        ("Part I. Poverty finance and the antinomies of colonialism", "", 11),
        ("1. A colonial problem", "23", 0),
        ("Part II. Making markets for poverty finance", "", 11),
        ("4. Commercialising community", "85", 0),
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert any(t.startswith("Part I.") for t in titles)
    assert any(t.startswith("Part II.") for t in titles)


def test_a_lone_long_line_is_not_promoted_on_its_word_alone(vtb):
    """PART_HEAD matches anything opening with Section. Without rows set
    under it and without company of its own kind, a long folio-less line
    stays out — which is what keeps the relaxation from admitting prose."""
    page = _banner_page([
        ("Introduction", "1", 0),
        ("Section rates were renegotiated annually by the standing committee",
         "", 0),
        ("1. A colonial problem", "23", 0),
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert not any(t.startswith("Section rates") for t in titles)


def test_a_part_ends_where_the_contents_stops_setting_lines_inside_it(vtb):
    """Reading the parts is only half of it. A book's Conclusion is set
    flush with its Introduction while every chapter inside a part is
    indented under it — so once the parts were read, the Conclusion, being
    merely the next line after the last of them, was filed inside Part IV.
    A line cannot belong to a group whose members are set further in."""
    page = _banner_page([
        ("Introduction", "1", 0),
        ("Part I: BETWEEN THE BLOCS AND EVERYTHING THEY CONTAINED", "", 0),
        ("2. Yugoslavia", "25", 4),
        ("3. Greece", "38", 4),
        ("Conclusion", "207", 0),
    ])
    ents = vtb.parse_printed_toc(page)[0]
    by = {e.title: e.depth for e in ents}
    assert by["Conclusion"] == min(by.values()), by


def test_a_chapter_set_level_with_its_part_still_belongs_to_it(vtb):
    """The mirror case, and the reason the test is indent and not position:
    where a book indents the banner instead of the chapters, the chapters
    are not outdented relative to anything and the group stays open."""
    page = _banner_page([
        ("Part I. Poverty finance and the antinomies of colonialism", "", 11),
        ("1. A colonial problem", "23", 0),
        ("2. Nascent neoliberalism", "45", 0),
        ("Part II. Making markets for poverty finance", "", 11),
        ("4. Commercialising community", "85", 0),
    ])
    ents = vtb.parse_printed_toc(page)[0]
    top = min(e.depth for e in ents)
    assert [e.title for e in ents if e.depth == top] == [
        "Part I. Poverty finance and the antinomies of colonialism",
        "Part II. Making markets for poverty finance"]


def _plain_page(html):
    """A contents page set without a table: headings and paragraphs only."""
    return ["<h1>Contents</h1>" + html]


def test_a_contents_page_without_a_table_is_read_from_its_headings_too(vtb):
    """One book sets CHAPTER ONE through SEVEN as headings and the rest as
    paragraphs. The table-less reader looked only at paragraphs, so half the
    book's designations were never seen at all."""
    page = _plain_page(
        "<h2>CHAPTER ONE</h2><p>The Collapse of the Middle Ages</p>"
        "<p>CHAPTER TWO</p><p>Martin Luther and the Common People</p>")
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert titles == ["CHAPTER ONE. The Collapse of the Middle Ages",
                      "CHAPTER TWO. Martin Luther and the Common People"], titles


def test_a_line_broken_inside_a_paragraph_keeps_its_space(vtb):
    """text_content() closes a <br/> up, so a title set over two lines came
    out as one run-together word and then matched no heading in the book."""
    page = _plain_page(
        "<p>CHAPTER ONE</p>"
        "<p>“All’s Right with the World”:<br/>The Collapse of the Middle Ages</p>")
    t = vtb.parse_printed_toc(page)[0][0].title
    assert "World”: The Collapse" in t, t


def test_a_title_finished_on_the_next_line_is_put_back_together(vtb):
    """A colon or a comma ending a contents line is the book saying the title
    continues, and on a page with no rows it is the only such evidence. It is
    also the one thing that keeps a year range attached to its chapter."""
    page = _plain_page(
        "<p>CHAPTER SEVEN</p>"
        "<p>The Rise of the Working Classes: Trade Unions and Socialism,</p>"
        "<p>1871–1914</p>")
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert titles == ["CHAPTER SEVEN. The Rise of the Working Classes: "
                      "Trade Unions and Socialism, 1871–1914"], titles


def test_a_bare_folio_on_a_table_less_page_is_not_a_title(vtb):
    """The page numbers sit on lines of their own. They finish a title broken
    before its year range and are otherwise dropped."""
    page = _plain_page("<p>Acknowledgements</p><p>vii</p>"
                       "<p>Introduction</p><p>viii</p>")
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert titles == ["Acknowledgements", "Introduction"], titles


def test_a_designation_is_not_swallowed_by_the_one_before_it(vtb):
    """Two designations in a row must not merge into each other."""
    page = _plain_page("<p>CHAPTER ONE</p><p>CHAPTER TWO</p><p>A Real Title</p>")
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert titles == ["CHAPTER ONE", "CHAPTER TWO. A Real Title"], titles


def test_a_page_that_merely_mentions_contents_is_not_one(vtb):
    """"The top should be opened and the contents allowed to melt slowly" is
    a manual talking about a thermos. Read as a contents page, it harvested
    prose about canteens into the book's structure. A label stands on its
    own; a common noun is preceded by an article."""
    prose = ("<p>Conventional thermos bottles will keep liquids hot for about "
             "24 hours. If they freeze, thaw them carefully to prevent "
             "bursting. The top should be opened and the contents allowed to "
             "melt slowly before drinking any of it at all.</p>") * 2
    assert vtb.parse_printed_toc([prose])[0] == []


def test_a_page_headed_table_of_contents_still_counts(vtb):
    """The article rule must not catch the "of" in "table of contents"."""
    page = ["<p>FIRST AID TABLE OF CONTENTS</p>"
            "<p>Chapter One: Fundamental Criteria for First Aid</p>"
            "<p>Chapter Two: Basic Measures for First Aid</p>"]
    assert vtb.parse_printed_toc(page)[1] == {0}


def _cells_page(rows):
    """A contents page whose rows are given as explicit cell lists, so a
    blank leading cell can be expressed."""
    out = ["<h1>Contents</h1><table>"]
    for cells in rows:
        out.append("<tr>" + "".join("<td/>" if c is None else f"<td>{c}</td>"
                                    for c in cells) + "</tr>")
    return ["".join(out) + "</table>"]


def test_a_chapter_numbered_in_its_own_cell_is_not_dropped(vtb):
    """One book sets three chapters as a number cell plus a title cell with
    no page against them, and the other two as number-and-title in one cell
    with a page. Only the latter survived: the number is not a folio and the
    title is not a page, so the row parsed as neither."""
    page = _cells_page([
        ["<b>1</b>", "<b>A Brief History of Migration</b>"],
        [None, "The Global Economy"],
        ["<b>2</b>", "<b>Methods and Perspectives</b>"],
        [None, "Methods"],
        [None, "Alternative Perspectives"],
        ["Introduction", "1"],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert "1 A Brief History of Migration" in titles, titles
    assert "2 Methods and Perspectives" in titles, titles


def test_a_blank_leading_cell_marks_what_belongs_under_a_chapter(vtb):
    """The book says a row is subordinate by leaving the number cell empty.
    Read as neither title nor folio, those rows were dropped, taking every
    subsection of three chapters with them."""
    page = _cells_page([
        ["<b>1</b>", "<b>A Brief History of Migration</b>"],
        [None, "The Political and Economic Contexts"],
        [None, "The Global Economy"],
        [None, "Asylum and Immigration"],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert "The Global Economy" in titles, titles


def test_a_lone_cell_with_nothing_in_front_of_it_is_not_an_entry(vtb):
    """The blank leading cell is the whole guard. Keeping any folio-less
    lone cell instead pulled in one book's copyright page — LIBRARY OF
    CONGRESS, Two Copies Received, JAN 27 1908."""
    page = _cells_page([
        ["<b>1</b>", "<b>A Brief History of Migration</b>"],
        [None, "The Global Economy"],
        ["LIBRARY OF CONGRESS"],
        ["Two Copies Received"],
        ["JAN 27 1908"],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert "LIBRARY OF CONGRESS" not in titles, titles
    assert "Two Copies Received" not in titles, titles


def test_the_tables_own_column_heading_is_not_an_entry(vtb):
    """"Page" sits in a row like any other and, with a blank cell in front
    of it, looks exactly like a subordinate entry."""
    page = _cells_page([
        [None, "Page"],
        ["<b>1</b>", "<b>A Brief History of Migration</b>"],
        [None, "The Global Economy"],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert "Page" not in titles, titles


def test_a_title_broken_inside_a_table_cell_keeps_its_space(vtb):
    """text_content() closes a <br/> up inside a cell too: "Migrant
    Communities" and "in the United Kingdom" arrived as "Communitiesin"."""
    page = _cells_page([
        ["<b>5</b>", "<b>Impacts on Migrant Communities<br/>in the United Kingdom</b>"],
        [None, "Media Images and Public Understanding"],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert any("Communities in the United Kingdom" in t for t in titles), titles


def test_a_chapters_byline_is_not_one_of_its_subsections(vtb):
    """An edited collection names who wrote each chapter, set the same way a
    subsection is — number cell empty, title cell filled — so the row alone
    cannot say which it is. What tells them apart is company: sections come
    in runs, and a book that gives every chapter an author gives each of
    them exactly one line."""
    page = _cells_page([
        ["<b>1</b>", "<b>Introduction: Access from Above</b>", "1"],
        [None, "Joe Karaganis", None],
        ["<b>2</b>", "<b>The Genesis of Library Genesis</b>", "25"],
        [None, "Balázs Bodó", None],
        ["<b>3</b>", "<b>Shadow Libraries in India</b>", "49"],
        [None, "Lawrence Liang", None],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert "Joe Karaganis" not in titles, titles
    assert "Balázs Bodó" not in titles, titles


def test_subsections_two_deep_are_still_read(vtb):
    """The mirror case: the same shape, but running two or more deep, is a
    list of sections and must survive."""
    page = _cells_page([
        ["<b>1</b>", "<b>A Brief History of Migration</b>", None],
        [None, "The Political and Economic Contexts", None],
        [None, "The Global Economy", None],
        [None, "Asylum and Immigration", None],
    ])
    titles = [e.title for e in vtb.parse_printed_toc(page)[0]]
    assert "The Global Economy" in titles, titles
