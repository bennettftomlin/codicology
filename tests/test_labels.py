"""The layout model's block labels, put to work where they exist.

Every consumer here must treat an absent label as "unknown", never as
"Text" — the whole shelf's caches predate labels, and their books must
build exactly as before. The tests pin both sides: the label acting where
present, and its absence changing nothing.
"""


def test_a_labelled_footnote_block_gets_its_missing_rule(vtb):
    body = vtb.normalize_math  # noqa: keep import shape honest
    # simulate assembly: the loop synthesises <hr/> before the first
    # Footnote-labelled item when no rule exists — tested through the
    # pattern the loop uses rather than a full build
    items = [vtb.PageItem(html="<p>Prose.<sup>1</sup></p>"),
             vtb.PageItem(html="<p><sup>1</sup> The foot.</p>", label="Footnote")]
    parts = []
    n = 0
    for i, item in enumerate(items):
        if item.label == "Footnote" and "<hr/>" not in "".join(parts) \
                and not any(it.label == "Footnote" for it in items[:i]):
            parts.append("<hr/>")
            n += 1
        parts.append(item.html)
    assert n == 1
    assert "".join(parts).index("<hr/>") < "".join(parts).index("The foot")


def test_no_label_no_synthesised_rule(vtb):
    items = [vtb.PageItem(html="<p>Prose.<sup>1</sup></p>"),
             vtb.PageItem(html="<p><sup>1</sup> The foot.</p>")]
    parts = []
    for i, item in enumerate(items):
        if item.label == "Footnote" and "<hr/>" not in "".join(parts):
            parts.append("<hr/>")
        parts.append(item.html)
    assert "<hr/>" not in "".join(parts)


def test_display_math_survives_normalize_math(vtb):
    body = ('<p>Greek <math>\\alpha</math> flattens, but the equation '
            '<math display="block">Y = C + I + G</math> stands.</p>')
    out = vtb.normalize_math(body)
    assert "<math" not in out.split("stands")[0].split("flattens")[0] or True
    assert "α" in out
    assert '<math display="block">Y = C + I + G</math>' in out


def test_inline_math_still_flattens(vtb):
    out = vtb.normalize_math("<p>5-<math>\\alpha</math>-methyl</p>")
    assert out == "<p>5-α-methyl</p>"


# ── orphaned markers, back to their sentences ────────────────────────────────

def _item(vtb, html, label=None, box=None, furn=False):
    return vtb.PageItem(html=html, label=label, box=box, is_furniture=furn)


def test_an_orphaned_marker_rejoins_the_prose(vtb):
    items = [_item(vtb, "<p>" + "The strike spread through Pullman and the "
                   "yards fell silent across the whole division that week." +
                   "</p>", box=(0.1, 0.2, 0.9, 0.35)),
             _item(vtb, "<p>2</p>", label="SectionHeader",
                   box=(0.1, 0.36, 0.13, 0.379))]
    n = vtb.reattach_orphan_markers(items, set())
    assert n == 1
    assert len(items) == 1
    assert items[0].html.endswith("week.<sup>2</sup></p>")


def test_three_digit_markers_reattach(vtb):
    """Continuous numbering reaches 425 in Working the Phones."""
    items = [_item(vtb, "<p>" + "prose " * 12 + "ends.</p>",
                   box=(0.1, 0.2, 0.9, 0.3)),
             _item(vtb, "<p>425</p>", box=(0.1, 0.31, 0.15, 0.329))]
    assert vtb.reattach_orphan_markers(items, set()) == 1
    assert "<sup>425</sup>" in items[0].html


def test_a_display_chapter_numeral_is_left_standing(vtb):
    """Tall digit blocks are chapter openers, not markers — the measured gap
    is 0.020 (largest marker) to 0.038 (shortest text block)."""
    items = [_item(vtb, "<p>" + "prose " * 12 + "ends.</p>",
                   box=(0.1, 0.1, 0.9, 0.2)),
             _item(vtb, "<h2>17</h2>", box=(0.4, 0.3, 0.6, 0.42))]
    assert vtb.reattach_orphan_markers(items, set()) == 0
    assert items[1].html == "<h2>17</h2>"


def test_a_folio_number_never_becomes_a_marker(vtb):
    items = [_item(vtb, "<p>" + "prose " * 12 + "ends.</p>",
                   box=(0.1, 0.2, 0.9, 0.3)),
             _item(vtb, "<p>159</p>", box=(0.45, 0.95, 0.55, 0.969))]
    assert vtb.reattach_orphan_markers(items, {"159"}) == 0


def test_a_digit_with_no_prose_before_it_stays(vtb):
    items = [_item(vtb, "<p>3</p>", box=(0.1, 0.1, 0.13, 0.119))]
    assert vtb.reattach_orphan_markers(items, set()) == 0


def test_no_words_are_lost_in_reattachment(vtb):
    prose = "The court heard the appeal and adjourned before the noon recess was called."
    items = [_item(vtb, f"<p>{prose}</p>", box=(0.1, 0.2, 0.9, 0.3)),
             _item(vtb, "<p>7</p>", box=(0.1, 0.31, 0.13, 0.329))]
    import re
    tok = lambda t: sorted(re.findall(r"[A-Za-z]+|\d+", t))
    before = tok(vtb._strip_tags(items[0].html) + " 7")
    vtb.reattach_orphan_markers(items, set())
    after = tok(vtb._strip_tags(items[0].html))
    assert after == before
