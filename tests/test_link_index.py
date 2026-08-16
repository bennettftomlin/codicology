"""The index's page numbers go where they point — and nowhere else.

Every shape here was surveyed off the real shelf: table-cell indexes
(Chinese Communism), <br/>-packed paragraphs (Working the Phones),
newline-packed paragraphs with —Continued carryovers (Eagle), en-dash and
abbreviated ranges (When Protest: 219–20), See cross-references, names that
contain digits (9/11 Commission; 1848 revolutions), and the field manuals
whose indexes print the warning that their numbers are paragraphs, not
pages — a warning that is read and obeyed.
"""


MAP = {n: 100 + n for n in range(1, 300)}     # folio n lives on page 100+n


def test_a_plain_entry_links(vtb):
    bodies = ["<h1>INDEX</h1><p>Adventurism, Li Li-san attacked for, "
              "159-160</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 1
    assert '<a href="page_0259.xhtml#pgb-0259">159-160</a>' in bodies[0]


def test_a_number_list_links_each(vtb):
    bodies = ["<h1>INDEX</h1><p>Addams, Jane, 46, 115, 150</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 3
    for n in (46, 115, 150):
        assert f'#pgb-{100+n:04d}">{n}</a>' in bodies[0]


def test_semicolon_subentries_each_get_their_run(vtb):
    bodies = ["<h1>INDEX</h1><p>leadership, 87; on revolutionary surge in "
              "China, 110, 111</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 3


def test_en_dash_and_abbreviated_ranges(vtb):
    bodies = ["<h1>INDEX</h1><p>ag-gag laws 219–20, 122</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 2
    assert "#pgb-0319\">219–20</a>" in bodies[0]


def test_br_packed_and_no_comma_entries(vtb):
    bodies = ["<h1>INDEX</h1><p>absence 21<br/>Ackroyd, Stephen 119, 139"
              "<br/>affective dimension 44</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 4


def test_a_name_with_digits_is_not_a_reference(vtb):
    """"1848 revolutions, 12" cites one page. "9/11 Commission, 66" one."""
    bodies = ["<h1>INDEX</h1><p>1848 revolutions, 12</p>"
              "<p>9/11 Commission, 66</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 2
    assert ">12</a>" in bodies[0] and ">66</a>" in bodies[0]
    assert "1848</a>" not in bodies[0]
    assert ">11</a>" not in bodies[0]


def test_see_cross_references_pass_untouched(vtb):
    body = "<h1>INDEX</h1><p>AEPA. <i>See</i> Animal Enterprise Protection Act</p>"
    bodies = [body]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 0
    assert bodies[0] == body


def test_a_folio_outside_the_map_stays_plain(vtb):
    bodies = ["<h1>INDEX</h1><p>Zealotry, 999</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 0 and st["unknown_folio"] == 1
    assert "<a" not in bodies[0]


def test_a_paragraph_index_is_believed_and_skipped(vtb):
    bodies = ["<h1>INDEX</h1><p>Entries are by paragraph number unless "
              "figure (fig.) or page (pg.) is specified.</p>"
              "<p>Aachen, 1-15</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["skipped_paragraph_index"] is True
    assert st["linked"] == 0
    assert "<a" not in bodies[0]


def test_table_cells_link_like_paragraphs(vtb):
    bodies = ["<h1>INDEX</h1><table><tr><td>Cadets, 43</td>"
              "<td>leadership, 87</td></tr></table>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 2


def test_the_index_ends_at_the_next_heading(vtb):
    bodies = ["<h1>INDEX</h1><p>Cadets, 43</p>",
              "<h1>The Pluto Press Newsletter</h1><p>Sign up in 2024, "
              "call 44, 45</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 1, "numbers beyond the index were linked"


def test_prose_before_the_index_is_never_touched(vtb):
    bodies = ["<p>In 46 the legions left. 115 died.</p>",
              "<h1>INDEX</h1><p>Addams, Jane, 46</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 1
    assert "<a" not in bodies[0]


def test_dropped_target_pages_are_not_linked_to(vtb):
    bodies = ["<h1>INDEX</h1><p>Cadets, 43</p>"]
    st = vtb.link_index(bodies, {43: 143}, {143})
    assert st["linked"] == 0 and st["unknown_folio"] == 1


def test_words_are_preserved_exactly(vtb):
    bodies = ["<h1>INDEX</h1><p>Agrarian question, 65-68; Mao’s report on, "
              "in Hunan, 73-78, 84; in Conference, 91</p>"]
    before = vtb._strip_tags(bodies[0]).split()
    vtb.link_index(bodies, MAP, set())
    assert vtb._strip_tags(bodies[0]).split() == before


def test_continued_carryover_entries_still_link(vtb):
    bodies = ["<h1>INDEX</h1><p>Altgeld, John Peter—Continued\n"
              "Congress, 66; defeated, 68; association with Martin, 65-66</p>"]
    st = vtb.link_index(bodies, MAP, set())
    assert st["linked"] == 3
