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
