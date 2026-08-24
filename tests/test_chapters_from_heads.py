"""Chapters as the book's own running heads report them.

A running head is the one piece of structure a book repeats on every page,
and the recto's names the chapter the reader is inside. That is evidence the
printed contents cannot give — a span asserted twenty times over, rather
than one line whose page must be inferred from a folio — and it is available
on books whose contents page never parses at all.

What these tests pin is mostly the refusals. The stream is only chapters
when it behaves like chapters, and the shelf offers several ways for it not
to: a book whose head is only its own title, a manual whose head is a
document number, an edited volume naming a different contributoron every leaf.
"""


def heads(*rows):
    """One furniture line per page; None for a page carrying none."""
    return [[] if r is None else [r] for r in rows]


def book(title, *chapters):
    """A book set the ordinary way: verso carries the title, recto the
    chapter, and the chapter's own opening page carries neither."""
    pages = [None, None]
    for name, n in chapters:
        pages.append(None)                      # the opening page
        for i in range(n):
            pages += [f"{title} · {10 + i}", f"{name} · {11 + i}"]
    return heads(*pages)


def test_the_recto_heads_partition_the_book(vtb):
    ch, st = vtb.chapters_from_furniture(
        book("Anthropology's World", ("Editing Anthropology", 6),
             ("Diversity Is Our Business", 6), ("Being There", 6)))
    assert [c.title for c in ch] == ["Editing Anthropology",
                                     "Diversity Is Our Business",
                                     "Being There"]
    assert st["refused"] == ""
    assert all(c.end > c.start for c in ch)


def test_a_chapter_link_lands_on_its_opening_page(vtb):
    """The opening page prints no running head — that is the convention —
    so the first page NAMING a chapter is its second."""
    ch, _ = vtb.chapters_from_furniture(
        book("T", ("One", 6), ("Two", 6), ("Three", 6)))
    # the head naming 'One' sits on page 4; its unheaded opening is 3
    assert ch[0].start == 3


def test_one_misread_head_does_not_split_a_chapter(vtb):
    """A single page's TIHKAL is another's THIKAL. Exact matching turned an
    18-page chapter into a 12 and a 6 on a real book."""
    ch, _ = vtb.chapters_from_furniture(heads(
        None, "Invasion · 3", "Invasion · 4", "Invasion · 5",
        "Invasiom · 6", "Invasion · 7", "Invasion · 8", "Invasion · 9",
        "Lourdes · 10", "Lourdes · 11", "Lourdes · 12", "Lourdes · 13",
        "Places · 14", "Places · 15", "Places · 16", "Places · 17"))
    assert [c.title for c in ch] == ["Invasion", "Lourdes", "Places"]


def test_a_book_whose_head_is_only_its_title_yields_nothing(vtb):
    """Three textbooks on the shelf print a URL footer and nothing else.
    Refusing is the right answer, not inventing one chapter."""
    ch, st = vtb.chapters_from_furniture(heads(*["Saylor.org · 4"] * 40))
    assert ch == []
    assert "second head stream" in st["refused"] or "three chapters" in st["refused"]


def test_marginalia_changing_every_page_is_refused(vtb):
    """An edited volume names a different contributor on each leaf: 267
    runs across 268 pages, measured. Real chapters run 8-12 pages."""
    ch, st = vtb.chapters_from_furniture(
        heads(*[f"{n} · {i}" for i, n in enumerate(
            ["Karaganis", "Liang", "Mizukami", "Cruz", "Alkalimat",
             "Ekdale", "Rens", "Vadi", "Zhang", "Sundaram"] * 4)]))
    assert ch == []
    assert "marginalia" in st["refused"]


def test_a_head_that_owns_the_book_is_not_a_chapter(vtb):
    """A manual names itself on every page. Where that designation is not
    the most repeated head it survives the title filter, and one 'chapter'
    then spanned 247 of 258 pages."""
    rows = ["Index · 1", "Index · 2", "Index · 3"]
    rows += [f"FM 3-25.150 · {i}" for i in range(4, 44)]
    ch, st = vtb.chapters_from_furniture(heads(*rows))
    assert ch == []


def test_the_folio_and_its_separator_come_off_the_head(vtb):
    assert vtb.head_title("60 · Citadel of Sin") == "Citadel of Sin"
    assert vtb.head_title("THE NEW CAREER 51") == "THE NEW CAREER"
    assert vtb.head_title("Chapter 7 — 61") == "Chapter 7"
    assert vtb.head_title("&#x27;Shrooms 44") == "'Shrooms"


def test_a_chapter_title_is_lifted_to_the_top_of_the_outline(vtb):
    """h1 was measured holding title-page fragments while the chapters that
    structure the book sat at h2 or h3."""
    bodies = ["<p>front</p>",
              "<h2>Being There</h2><p>Prose.</p>",
              "<p>more</p>"]
    ch = [vtb.Chapter("Being There", 1, 2)]
    assert vtb.promote_chapter_headings(bodies, ch) == 1
    assert bodies[1].startswith("<h1>Being There</h1>")


def test_nothing_is_demoted_and_no_other_heading_moves(vtb):
    bodies = ["<h1>The Book Itself</h1>",
              "<h2>Being There</h2><h2>A Subsection</h2>"]
    ch = [vtb.Chapter("Being There", 1, 1)]
    vtb.promote_chapter_headings(bodies, ch)
    assert bodies[0] == "<h1>The Book Itself</h1>", "title page untouched"
    assert "<h2>A Subsection</h2>" in bodies[1], "only the chapter's own name"


def test_a_roman_folio_is_not_a_chapter(vtb):
    """Front matter paginates in roman, and a bare 'vi' left standing became
    a two-page chapter called vi on a real book."""
    assert vtb.head_title("vi") == ""
    assert vtb.head_title("xvii") == ""
    assert vtb.head_title("I Saw the Border") == "I Saw the Border"


def test_the_heading_is_found_before_the_first_head_page(vtb):
    """A chapter opens on a recto with no running head, the verso after it
    carries the book's title, and only the NEXT recto names the chapter — so
    the heading can sit two pages before the first page that names it."""
    bodies = ["<p>prior chapter</p>",
              "<h2>Being There: Social Life</h2><p>Opens here.</p>",
              "<p>verso</p>", "<p>recto</p>", "<p>more</p>"]
    ch = [vtb.Chapter("BEING THERE", 3, 4)]
    assert vtb.promote_chapter_headings(bodies, ch) == 1
    assert bodies[1].startswith("<h1>Being There: Social Life</h1>")


def test_the_search_never_crosses_into_the_previous_chapter(vtb):
    bodies = ["<p>x</p>", "<h2>Ethics and Encounters</h2>",
              "<p>y</p>", "<p>z</p>"]
    ch = [vtb.Chapter("SOMETHING ELSE", 0, 1),
          vtb.Chapter("ETHICS AND ENCOUNTERS", 2, 3)]
    assert vtb.promote_chapter_headings(bodies, ch) == 0
    assert "<h2>Ethics and Encounters</h2>" in bodies[1]


def test_a_running_head_abbreviates_its_chapter(vtb):
    """Heads shorten: "INTRODUCTION" names the page titled "Introduction:
    Going Inside", which scores 0.59 against it."""
    assert vtb._names_chapter("Introduction: Going Inside", "INTRODUCTION")
    assert not vtb._names_chapter("Border security", "Border")


def test_a_head_that_names_nothing_is_not_a_chapter(vtb):
    """A manual prints its issue date and the bare word "Chapter" where a
    book prints a title. Those pass every structural guard — long runs,
    evenly spread — while naming nothing, they match no heading on any
    page, and by existing at all they kept the printed contents from being
    consulted for the heading outline."""
    rows = []
    for name in ("May 2014", "Chapter", "May 2014", "Chapter"):
        rows += [f"{name} · {i}" for i in range(6)]
    ch, st = vtb.chapters_from_furniture(heads(*rows))
    assert ch == [], st


def test_real_titles_survive_the_same_guard(vtb):
    ch, _ = vtb.chapters_from_furniture(heads(
        *[f"The Brazil Caper · {i}" for i in range(6)],
        *[f"Places in the Mind · {i}" for i in range(6)],
        *[f"DMT is Everywhere · {i}" for i in range(6)]))
    assert [c.title for c in ch] == ["The Brazil Caper", "Places in the Mind",
                                     "DMT is Everywhere"]
