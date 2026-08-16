"""Citations bound to the bibliography, driven from the bibliography.

The design constraint everything here tests: the prose is never searched for
anything SHAPED like a citation, only for the specific names and years the
bibliography holds. "New York: Harper, 1933" was the failure of the shaped
approach — it is exactly (Capitalised Year) — and here it can never bind
because no author is called New. The junk is not filtered; it is never a
candidate.

The second constraint is the word-preservation rule. This is the only linker
that rewrites inside running sentences rather than replacing an isolated
superscript, so every edited page must read back word-for-word identical, and
a page that does not is reverted whole.
"""


def ad_book():
    return [
        "<p>Fintech claims were criticised (Bernards 2019a) and defended "
        "elsewhere (see Gould 2015; Clarke 2019). Politi (2019) reported "
        "the summit. Published in (New York: Harper and Brothers, 1933) "
        "regardless.</p>",
        "<h1>Bibliography</h1>"
        "<p>Bernards, N. (2019a) 'Tech'.</p>"
        "<p>Clarke, C. (2019) 'Lending'.</p>"
        "<p>Gould, H. (2015) 'Fintech'.</p>"
        "<p>Politi, J. (2019) 'Poverty'.</p>",
    ]


def test_author_date_binds_and_imprints_never_can(vtb):
    bodies = ad_book()
    st = vtb.link_citations(bodies, dropped=set())
    assert st["authordate"] == 4
    assert st["reverted"] == 0
    assert 'href="page_0001.xhtml#bib-' in bodies[0]
    assert "1933)</a>" not in bodies[0], "an imprint was bound as a citation"
    assert "Harper" not in bodies[0].split("<a")[1], "imprint inside an anchor"


def test_the_entries_gain_anchors(vtb):
    bodies = ad_book()
    vtb.link_citations(bodies, dropped=set())
    assert bodies[1].count('<p id="bib-') == 4


def test_page_suffix_and_see_prefix_bind(vtb):
    bodies = ["<p>Access matters (AFI 2010:1). Credit shifted (see Adams "
              "1971). Nothing else.</p>",
              "<h1>References</h1>"
              "<p>Adams, D. (1971) 'Credit'.</p>"
              "<p>AFI (2010) 'Principles'.</p>"]
    st = vtb.link_citations(bodies, dropped=set())
    assert st["authordate"] == 2


def test_an_ambiguous_key_binds_nothing(vtb):
    bodies = ["<p>As shown (Brown 2011) conclusively.</p>",
              "<h1>Bibliography</h1>"
              "<p>Brown, E. (2011) 'One'.</p>"
              "<p>Brown, M. (2011) 'Another'.</p>"]
    st = vtb.link_citations(bodies, dropped=set())
    assert st["authordate"] == 0
    assert "<a" not in bodies[0]


def test_words_are_preserved_exactly(vtb):
    bodies = ad_book()
    before = [vtb._strip_tags(b).split() for b in bodies]
    vtb.link_citations(bodies, dropped=set())
    assert [vtb._strip_tags(b).split() for b in bodies] == before


def test_a_book_with_no_bibliography_is_untouched(vtb):
    bodies = ["<p>Plain prose with Politi (2019) in it.</p>",
              "<p>More prose.</p>"]
    keep = list(bodies)
    st = vtb.link_citations(bodies, dropped=set())
    assert st == {"authordate": 0, "chicago": 0, "ambiguous": 0,
                  "reverted": 0, "entries": 0}
    assert bodies == keep


def test_a_year_in_a_dropped_page_is_left_alone(vtb):
    bodies = ad_book()
    st = vtb.link_citations(bodies, dropped={0})
    assert st["authordate"] == 0


# ── Chicago short-form, in notes ─────────────────────────────────────────────

def chicago_book():
    return [
        # a note paragraph (id says so) and a PROSE paragraph mentioning the
        # same title: only the note may bind
        '<p id="note-g0-1"><sup>1</sup> In Bromfield, <i>The Farm</i>, '
        "p. 51, the point is made.</p>"
        "<p>Everyone remembers reading <i>The Farm</i> at school.</p>",
        "<h1>BIBLIOGRAPHY</h1>"
        "<p>Bromfield, Louis, <i>The Farm</i> (New York: Harper and "
        "Brothers, 1933).</p>"
        "<p><i>Watching the World Go By</i> (Boston: Little, Brown and "
        "Co., 1933).</p>",
    ]


def test_a_note_citation_binds_and_a_prose_mention_does_not(vtb):
    bodies = chicago_book()
    st = vtb.link_citations(bodies, dropped=set())
    assert st["chicago"] == 1
    note, prose = bodies[0].split("</p>", 1)
    assert '<a href="page_0001.xhtml#bib-' in note
    assert "<a" not in prose, "a passing mention was bound as a citation"


def test_an_authorless_entry_inherits_the_surname_above(vtb):
    """Chicago sets a second work by the same hand with no author repeated.
    A note citing "Bromfield, Watching the World Go By" must reach it."""
    bodies = ['<p id="note-g0-1"><sup>1</sup> Bromfield, <i>Watching the '
              "World Go By</i>, p. 9.</p>"] + chicago_book()[1:]
    st = vtb.link_citations(bodies, dropped=set())
    assert st["chicago"] == 1


def test_a_shortened_title_still_reaches_its_entry(vtb):
    bodies = ['<p id="note-g0-1"><sup>1</sup> Hayes, <i>A Political and '
              "Social History</i>, p. 88.</p>",
              "<h1>BIBLIOGRAPHY</h1>"
              "<p>Hayes, C. J. H., <i>A Political and Social History of "
              "Modern Europe</i> (New York: Macmillan, 1924).</p>"]
    assert vtb.link_citations(bodies, dropped=set())["chicago"] == 1


def test_a_title_matching_two_entries_binds_neither(vtb):
    bodies = ['<p id="note-g0-1"><sup>1</sup> Smith, <i>History</i>, '
              "p. 12.</p>",
              "<h1>BIBLIOGRAPHY</h1>"
              "<p>Smith, A., <i>History of Trade</i> (London: X, 1901).</p>"
              "<p>Smith, B., <i>History of Labour</i> (London: Y, 1902).</p>"]
    st = vtb.link_citations(bodies, dropped=set())
    assert st["chicago"] == 0 and st["ambiguous"] >= 1


def test_the_surname_must_stand_near_the_title(vtb):
    """An italic title with no author nearby is a mention, not a short-form
    citation, even inside a note."""
    bodies = ['<p id="note-g0-1"><sup>1</sup> On this see <i>The Farm</i> '
              "generally.</p>"] + chicago_book()[1:]
    assert vtb.link_citations(bodies, dropped=set())["chicago"] == 0


def test_no_anchor_is_ever_nested(vtb):
    bodies = ['<p id="note-g0-1"><sup>1</sup> See <a href="x.xhtml">already '
              "linked Bromfield, <i>The Farm</i></a>, p. 51.</p>"] \
             + chicago_book()[1:]
    vtb.link_citations(bodies, dropped=set())
    assert bodies[0].count("<a ") == 1, "an anchor was written inside an anchor"


def test_a_field_manual_references_appendix_binds_nothing(vtb):
    """FM 3-06 parses a References appendix; with no author-date citations
    and no note short-forms, entries alone must produce zero links."""
    bodies = ["<p>Doctrine described plainly, no citations.</p>",
              "<h1>References</h1>"
              "<p>Glenn, R. (2002) <i>Urban Combat</i>.</p>"]
    st = vtb.link_citations(bodies, dropped=set())
    assert st["authordate"] == 0 and st["chicago"] == 0
    assert st["entries"] == 1
