"""Inline content the layout model left outside the block it belongs to.

Chinese Communism's notes section arrives as

    <p>53.</p><i>Ibid.</i>, p. 51.<p>54.</p><i>Ibid.</i>, p. 53.

— each entry's NUMBER closed as a paragraph of its own, the entry's text
left loose beside it. Nothing is lost to the reader, but the text belongs to
no block, which is not valid XHTML and is invisible to every pattern that
reads a note entry: they all anchor the number to a block's opening. Nine
entries were unreadable and the markers pointing at them stayed plain.

The repair is structural and knows nothing about notes — inline content
between two blocks belongs to the block before it — and the seam carries a
space, for the same reason _strip_tags puts one there: a closed block is a
word boundary, and without it the number runs into its own note.

Measured over 123,377 cached blocks: 137 change, in 11 books, and most of
those are running heads that never reach the body anyway.
"""
import pytest


def test_a_note_whose_number_closed_without_it_is_made_whole(vtb):
    out = vtb._to_xhtml('<p>53.</p><i>Ibid.</i>, p. 51.')
    assert out == '<p>53. <i>Ibid.</i>, p. 51.</p>'
    # and is now a note entry, which is the whole point
    assert vtb.NOTE_ENTRY_PLAIN.match(out)


def test_the_seam_is_a_word_boundary(vtb):
    assert vtb._to_xhtml("<p>Carbolines</p><i>beta</i>") \
        == "<p>Carbolines <i>beta</i></p>"


def test_but_not_where_the_punctuation_already_binds(vtb):
    # An index entry: no book writes "strategic ( continued)".
    assert vtb._to_xhtml("<p>strategic (</p><i>continued</i>)") \
        == "<p>strategic (<i>continued</i>)</p>"


def test_a_space_the_model_already_left_is_not_doubled(vtb):
    assert vtb._to_xhtml("<p>one </p><i>two</i>") == "<p>one <i>two</i></p>"


def test_loose_text_with_no_markup_of_its_own_is_taken_too(vtb):
    assert vtb._to_xhtml("<p>a</p>tail text") == "<p>a tail text</p>"


# ── what it must never cross ─────────────────────────────────────────────────

def test_two_paragraphs_stay_two_paragraphs(vtb):
    assert vtb._to_xhtml("<p>one</p><p>two</p>") == "<p>one</p><p>two</p>"


def test_a_rule_ends_the_run(vtb):
    # <p>…</p><hr/> is the commonest shape of all in the caches; folding a
    # footnote rule into the paragraph above it would erase the boundary the
    # footnote linker keys on.
    assert vtb._to_xhtml("<p>DEPARTMENT OF THE ARMY</p><hr/>") \
        == "<p>DEPARTMENT OF THE ARMY</p><hr/>"


def test_a_heading_is_not_swallowed_by_the_paragraph_before_it(vtb):
    assert vtb._to_xhtml("<p>prose</p><h2>CHAPTER TWO</h2>") \
        == "<p>prose</p><h2>CHAPTER TWO</h2>"


def test_a_table_is_left_where_the_model_put_it(vtb):
    out = vtb._to_xhtml("<p>above</p><table><tr><td>cell</td></tr></table>")
    assert out.startswith("<p>above</p><table>")


def test_nothing_is_folded_into_a_figure(vtb):
    out = vtb._to_xhtml('<figure><img src="f.jpg"/></figure>loose words')
    assert out.startswith('<figure><img src="f.jpg"/></figure>')


def test_a_whole_run_of_entries_is_separated_correctly(vtb):
    out = vtb._to_xhtml('<p>53.</p><i>Ibid.</i>, p. 51.'
                        '<p>54.</p><i>Ibid.</i>, p. 53.')
    assert out == ('<p>53. <i>Ibid.</i>, p. 51.</p>'
                   '<p>54. <i>Ibid.</i>, p. 53.</p>')


def test_the_entries_then_parse_as_a_group(vtb):
    notes = ("<h1>NOTES</h1><h2>Chapter One</h2>"
             + vtb._to_xhtml('<p>1.</p><i>Ibid.</i>, p. 51.')
             + vtb._to_xhtml('<p>2.</p><i>Ibid.</i>, p. 53.')
             + vtb._to_xhtml('<p>3.</p><i>Ibid.</i>, p. 55.'))
    bodies = ["<p>a<sup>1</sup>b<sup>2</sup>c<sup>3</sup></p>", notes]
    stats = vtb.link_notes(bodies, set())
    assert stats["linked"] == 3 and stats["unlinked"] == 0
