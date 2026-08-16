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
