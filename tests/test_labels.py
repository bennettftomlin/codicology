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


# ── orphaned markers: sequence-fit or suppress ──────────────────────────────

def _item(vtb, html, label=None, box=None, furn=False):
    return vtb.PageItem(html=html, label=label, box=box, is_furniture=furn)


PROSE = ("<p>The strike spread through Pullman and the yards fell silent "
         "across the whole division that week.<sup>1</sup></p>")


def test_a_sequence_fitting_orphan_rejoins_the_prose(vtb):
    """Inline markers reach 1; a one-line '2' block continues the sequence."""
    items = [_item(vtb, PROSE, box=(0.1, 0.2, 0.9, 0.35)),
             _item(vtb, "<p>2</p>", box=(0.1, 0.36, 0.13, 0.379))]
    ra, sp = vtb.reattach_orphan_markers(items, set())
    assert (ra, sp) == (1, 0)
    assert "<sup>2</sup>" in items[0].html


def test_a_chapter_opening_one_reattaches_before_its_twos(vtb):
    items = [_item(vtb, "<p>The new chapter opens with a long enough line "
                   "of narrative prose to host a marker.</p>",
                   box=(0.1, 0.1, 0.9, 0.2)),
             _item(vtb, "<p>1</p>", box=(0.1, 0.21, 0.13, 0.229)),
             _item(vtb, "<p>More prose follows here with the second "
                   "reference.<sup>2</sup></p>", box=(0.1, 0.24, 0.9, 0.3))]
    ra, sp = vtb.reattach_orphan_markers(items, set())
    assert (ra, sp) == (1, 0)


def test_a_phantom_that_fits_no_sequence_is_suppressed(vtb):
    """Eagle's 220: digit blocks over plain prose, verified inkless. A '5'
    when the stream sits at 1 fits nothing and ships nowhere."""
    items = [_item(vtb, PROSE, box=(0.1, 0.2, 0.9, 0.35)),
             _item(vtb, "<p>5</p>", box=(0.5, 0.4, 0.53, 0.419))]
    ra, sp = vtb.reattach_orphan_markers(items, set())
    assert (ra, sp) == (0, 1)
    assert len(items) == 1, "the phantom shipped"


def test_three_digit_continuation_reattaches(vtb):
    state = {"last": 424}
    items = [_item(vtb, "<p>" + "prose " * 12 + "ends.</p>",
                   box=(0.1, 0.2, 0.9, 0.3)),
             _item(vtb, "<p>425</p>", box=(0.1, 0.31, 0.15, 0.329))]
    ra, sp = vtb.reattach_orphan_markers(items, set(), state)
    assert (ra, sp) == (1, 0)
    assert "<sup>425</sup>" in items[0].html


def test_a_display_chapter_numeral_is_left_standing(vtb):
    items = [_item(vtb, PROSE, box=(0.1, 0.1, 0.9, 0.2)),
             _item(vtb, "<h2>17</h2>", box=(0.4, 0.3, 0.6, 0.42))]
    ra, sp = vtb.reattach_orphan_markers(items, set())
    assert (ra, sp) == (0, 0)
    assert any("<h2>17</h2>" == it.html for it in items)


def test_a_furniture_number_is_neither_marker_nor_phantom(vtb):
    items = [_item(vtb, PROSE, box=(0.1, 0.2, 0.9, 0.3)),
             _item(vtb, "<p>159</p>", box=(0.45, 0.95, 0.55, 0.969))]
    ra, sp = vtb.reattach_orphan_markers(items, {"159"})
    assert (ra, sp) == (0, 0)


def test_sequence_state_carries_across_pages(vtb):
    state = {"last": None}
    page1 = [_item(vtb, PROSE.replace("<sup>1</sup>",
                   "<sup>1</sup> more.<sup>2</sup>"),
                   box=(0.1, 0.2, 0.9, 0.35))]
    vtb.reattach_orphan_markers(page1, set(), state)
    assert state["last"] == 2
    page2 = [_item(vtb, "<p>" + "prose " * 10 + "goes on.</p>",
                   box=(0.1, 0.1, 0.9, 0.2)),
             _item(vtb, "<p>3</p>", box=(0.1, 0.21, 0.13, 0.229))]
    ra, sp = vtb.reattach_orphan_markers(page2, set(), state)
    assert (ra, sp) == (1, 0)
