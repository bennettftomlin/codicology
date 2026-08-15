"""Same-page footnotes: symbols and numbers, bound only where the page itself
settles the match.

The symbol path has carried 249 live links in one book since before these
tests existed; they pin its behavior down now because the numbered path is
being added beside it, and "beside" is exactly where a regression would land.

The numbered path answers a book that footnotes with digits at the page foot.
Digits are everywhere in a book — folios, years, section numbers — so a
marker counts only as a bare <sup>, the block below the rule must be SHAPED
like feet (no heading, opens with a foot, nearly all its words inside
foot-shaped paragraphs), and the pass runs only when the caller says the
book's numbers are not already spoken for by an endnotes section. Every
guard costs recall somewhere; an unlinked marker is just the page as printed,
which is the cheap side to fail on.
"""


# ── the symbol path, as it has always behaved ────────────────────────────────

def test_one_asterisk_above_and_below_links(vtb):
    bodies = ["<p>A claim*</p><hr/><p>* The evidence.</p>"]
    stats = vtb.link_footnotes(bodies)
    assert stats["linked"] == 1
    assert 'epub:type="noteref"' in bodies[0]
    assert 'id="fn-p0-1"' in bodies[0] and 'href="#fn-p0-1"' in bodies[0]


def test_two_markers_for_one_foot_stay_plain(vtb):
    bodies = ["<p>One* and two*</p><hr/><p>* Which one?</p>"]
    stats = vtb.link_footnotes(bodies)
    assert stats["linked"] == 0 and stats["skipped"] == 1
    assert "noteref" not in bodies[0]


def test_dagger_and_asterisk_link_independently(vtb):
    bodies = ["<p>First* then second†</p><hr/>"
              "<p>* For the first.</p><p>† For the second.</p>"]
    stats = vtb.link_footnotes(bodies)
    assert stats["linked"] == 2


def test_a_page_without_a_rule_is_untouched(vtb):
    bodies = ["<p>A claim* with no foot.</p>"]
    assert vtb.link_footnotes(bodies)["linked"] == 0
    assert bodies[0] == "<p>A claim* with no foot.</p>"


# ── the numbered path ────────────────────────────────────────────────────────

def test_numbered_markers_link_to_their_feet(vtb):
    bodies = ["<p>A claim.<sup>1</sup> Another.<sup>2</sup></p><hr/>"
              "<p>1. The first source.</p><p>2. The second.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["linked"] == 2 and stats["numbered"] == 2
    assert 'id="fnref-p0-n1"' in bodies[0]
    assert 'href="#fn-p0-n2"' in bodies[0]
    assert 'id="fn-p0-n1"' in bodies[0]


def test_numbering_may_run_on_through_the_chapter(vtb):
    """Continuous numbering: page carries markers 17 and 18, not 1 and 2."""
    bodies = ["<p>Claim.<sup>17</sup> More.<sup>18</sup></p><hr/>"
              "<p>17. A source.</p><p>18. Another.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 2


def test_scope_is_the_page_not_the_book(vtb):
    """Marker 1 on each of two pages: each binds to its own foot."""
    bodies = ["<p>First page.<sup>1</sup></p><hr/><p>1. First foot.</p>",
              "<p>Second page.<sup>1</sup></p><hr/><p>1. Second foot.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 2
    assert 'href="#fn-p0-n1"' in bodies[0]
    assert 'href="#fn-p1-n1"' in bodies[1]


def test_a_number_in_prose_is_not_a_marker(vtb):
    """"1918 was the year" contains no marker; only a bare <sup> counts.
    And a year opening a paragraph is not a foot number — four digits is a
    date, not a footnote, so a block of such paragraphs is not feet."""
    bodies = ["<p>In 1918 the war ended.<sup>3</sup></p><hr/>"
              "<p>3. On the armistice of 1918.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 1
    assert 'id="fn-p0-n3"' in bodies[0]
    assert "1918" not in bodies[0].split("<hr/>")[0].replace(
        "In 1918 the war ended.", "")   # no id or href minted from the year
    blocks = ["<p>Prose.<sup>2</sup></p><hr/>"
              "<p>1918. A chronicle entry, not a footnote.</p>"]
    assert vtb.link_footnotes(blocks, allow_numbered=True)["numbered"] == 0


def test_numbered_pass_is_off_unless_the_caller_allows_it(vtb):
    """The book decides. A book with an endnotes section resolves its numbered
    markers there — build_epub passes allow_numbered only when no Notes head
    exists — so by default a rule page's numbers stay untouched for the
    endnote linker. This is The Invisible Government's shape: symbol feet and
    numbered endnote markers on the same rule pages."""
    bodies = ["<p>Claim* and endnoted.<sup>44</sup></p><hr/>"
              "<p>* The starred foot.</p><p>44. Looks like a foot too.</p>"]
    stats = vtb.link_footnotes(bodies)
    assert stats["linked"] == 1 and stats["numbered"] == 0
    assert "<sup>44</sup>" in bodies[0], \
        "the endnote linker's marker was consumed by the footnote pass"


def test_a_marker_with_no_answering_foot_stays_plain(vtb):
    bodies = ["<p>Claim.<sup>4</sup></p><hr/><p>Closing prose, no notes.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 0
    assert "<sup>4</sup>" in bodies[0]


def test_a_section_divider_rule_is_not_a_footnote_rule(vtb):
    """An hr followed by a heading and the bulk of the page divides sections;
    binding across it would link a marker to a numbered list item."""
    bodies = ["<p>Short opening.<sup>1</sup></p><hr/>"
              "<h2>THE NEXT SECTION</h2>"
              "<p>1. A numbered point in a list, not a footnote, followed by "
              "enough further prose that the lower half plainly outweighs the "
              "upper and reads as body text in every respect worth naming, "
              "continuing on at length as sections do.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 0
    assert "<sup>1</sup>" in bodies[0]


def test_the_same_number_twice_below_stays_plain(vtb):
    bodies = ["<p>Claim.<sup>5</sup></p><hr/>"
              "<p>5. One reading.</p><p>5. Another reading.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 0 and stats["skipped"] == 1


def test_symbols_and_numbers_share_a_page_without_collision(vtb):
    bodies = ["<p>Starred* and numbered.<sup>9</sup></p><hr/>"
              "<p>* The starred foot.</p><p>9. The numbered foot.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["linked"] == 2 and stats["numbered"] == 1


def test_a_symbol_linked_marker_is_not_reclaimed_by_the_number_pass(vtb):
    """After the symbol pass rewrites its marker, the anchor's digits must not
    read as a bare numbered marker to the second pass."""
    bodies = ["<p>Starred*</p><hr/><p>* Foot with a 7 in it.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["linked"] == 1 and stats["numbered"] == 0


def test_an_endnote_marker_is_left_for_the_endnote_linker(vtb):
    """A page with markers but no rule belongs to link_notes, and after
    link_footnotes declines it the endnote pass must still see bare sups."""
    bodies = ["<p>Endnoted claim.<sup>12</sup></p>"]
    vtb.link_footnotes(bodies)
    assert bodies[0] == "<p>Endnoted claim.<sup>12</sup></p>"


# ── the superscript rendering, and pages the layout gave no rule ─────────────

def test_a_foot_numbered_in_superscript_is_read(vtb):
    """When Protest Becomes Crime writes "<p><sup>2</sup> See: …" on some
    pages and "<p>1. According to …" on others. Only the second was
    understood, which cost it 52 of its 63 footnotes."""
    bodies = ["<p>Claim.<sup>1</sup> Another.<sup>2</sup></p><hr/>"
              "<p><sup>1</sup> First source.</p><p><sup>2</sup> Second.</p>"]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["numbered"] == 2
    assert 'href="#fn-p0-n2"' in bodies[0]
    assert "Second." in bodies[0]


def test_both_renderings_on_one_page(vtb):
    bodies = ["<p>A.<sup>1</sup> B.<sup>2</sup></p><hr/>"
              "<p>1. Plain form.</p><p><sup>2</sup> Superscript form.</p>"]
    assert vtb.link_footnotes(bodies, allow_numbered=True)["numbered"] == 2


def test_a_foot_block_under_no_rule_is_found(vtb):
    """Seven pages of that book carry a plain foot block with no rule at
    all, and the pass skipped them on its first line."""
    body = ("<p>" + "Body prose about the case. " * 12 + "Claim.<sup>1</sup></p>"
            "<p><sup>1</sup> The foot of the page.</p>")
    bodies = [body]
    stats = vtb.link_footnotes(bodies, allow_numbered=True)
    assert stats["ruleless"] == 1 and stats["numbered"] == 1


def test_a_ruleless_page_needs_a_page_above_the_feet(vtb):
    """A page that is nothing but numbered paragraphs is a list or a notes
    page, not a page with feet under it."""
    bodies = ["<p><sup>1</sup> One.</p><p><sup>2</sup> Two.</p>"
              "<p><sup>3</sup> Three.</p>"]
    assert vtb.link_footnotes(bodies, allow_numbered=True)["ruleless"] == 0


def test_a_ruleless_page_needs_the_feet_at_the_end(vtb):
    """A superscript-led paragraph in the middle, with prose after it, is
    not a foot block — the block is the last thing on its page."""
    body = ("<p>" + "Body prose. " * 12 + "Claim.<sup>1</sup></p>"
            "<p><sup>1</sup> Looks like a foot.</p>"
            "<p>" + "But the page carries on afterwards. " * 8 + "</p>")
    bodies = [body]
    assert vtb.link_footnotes(bodies, allow_numbered=True)["ruleless"] == 0


def test_the_ruleless_path_is_off_when_numbers_belong_to_endnotes(vtb):
    """A book with an endnotes section resolves its numbers there; without
    a rule there is not even a printed boundary to argue otherwise."""
    body = ("<p>" + "Body prose. " * 12 + "Claim.<sup>44</sup></p>"
            "<p><sup>44</sup> Would look like a foot.</p>")
    bodies = [body]
    assert vtb.link_footnotes(bodies)["ruleless"] == 0
    assert "<sup>44</sup>" in bodies[0]


def test_a_ruleless_page_with_no_superscript_feet_is_untouched(vtb):
    """Plain "1." paragraphs at the foot are not enough without a rule: an
    ordinary numbered list at the end of a page looks exactly the same."""
    body = ("<p>" + "Body prose. " * 12 + "Claim.<sup>1</sup></p>"
            "<p>1. Could be a list item.</p>")
    bodies = [body]
    assert vtb.link_footnotes(bodies, allow_numbered=True)["ruleless"] == 0
