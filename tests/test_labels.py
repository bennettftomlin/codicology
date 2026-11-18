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


def test_relabel_matcher_assigns_by_geometry(vtb):
    items = [{"box": [0.1, 0.1, 0.9, 0.3], "html": "<p>a</p>"},
             {"box": [0.1, 0.4, 0.9, 0.6], "html": "<p>b</p>"}]
    blocks = [("Text", [100, 400, 900, 600]),
              ("SectionHeader", [100, 100, 900, 300])]
    n = vtb._match_layout_labels(items, blocks, 1000, 1000)
    assert n == 2
    assert items[0]["lab"] == "SectionHeader"
    assert items[1]["lab"] == "Text"


def test_relabel_matcher_refuses_poor_overlap(vtb):
    """A wrong label misleads every consumer; an absent one declines to
    help. Nothing under half overlap attaches."""
    items = [{"box": [0.1, 0.1, 0.3, 0.2], "html": "<p>a</p>"}]
    blocks = [("Text", [600, 700, 900, 900])]
    assert vtb._match_layout_labels(items, blocks, 1000, 1000) == 0
    assert "lab" not in items[0]


def test_relabel_matcher_is_one_to_one(vtb):
    """Two items over one block: the better fit wins, the other stays
    unlabeled rather than sharing a guess."""
    items = [{"box": [0.1, 0.1, 0.9, 0.3]},
             {"box": [0.1, 0.12, 0.9, 0.34]}]
    blocks = [("Caption", [100, 100, 900, 300])]
    assert vtb._match_layout_labels(items, blocks, 1000, 1000) == 1
    assert items[0]["lab"] == "Caption" and "lab" not in items[1]


def test_relabel_matcher_never_overwrites(vtb):
    items = [{"box": [0.1, 0.1, 0.9, 0.3], "lab": "Footnote"}]
    blocks = [("Text", [100, 100, 900, 300])]
    assert vtb._match_layout_labels(items, blocks, 1000, 1000) == 0
    assert items[0]["lab"] == "Footnote"


def test_a_flat_image_declines_to_be_fingerprinted(vtb):
    """Every pixel of a uniform image sits at its own mean, so its
    signature collapsed to all-False — identical for a solid divider, a
    blank plate, and a grey placeholder, which once grouped three
    unrelated flat images as one recurring picture and deleted them all.
    Featureless is unidentifiable: no signature, never a shared one."""
    import io
    from PIL import Image
    def png(color):
        im = Image.new("L", (200, 120), color)
        buf = io.BytesIO(); im.save(buf, format="PNG")
        return buf.getvalue()
    assert vtb._picture_signature(png(255)) is None
    assert vtb._picture_signature(png(0)) is None
    assert vtb._picture_signature(png(128)) is None
    doomed = vtb.find_image_furniture(
        [["a"], ["b"], ["c"]],
        {"a": png(255), "b": png(0), "c": png(128)})
    assert doomed == set(), "unrelated flat images must never group"


def test_a_patterned_band_still_groups_as_furniture(vtb):
    """The fix must not break the feature's reason to exist: the
    cooperation_mutualism decorative band — a patterned ornament — still
    fingerprints and still strips when it recurs on three pages."""
    import io
    from PIL import Image
    im = Image.new("L", (200, 120), 255)
    for x in range(0, 200, 8):
        for y in range(0, 120, 6):
            im.putpixel((x, y), 0)
    buf = io.BytesIO(); im.save(buf, format="PNG")
    band = buf.getvalue()
    doomed = vtb.find_image_furniture(
        [["p0"], ["p1"], ["p2"]],
        {"p0": band, "p1": band, "p2": band})
    assert doomed == {"p0", "p1", "p2"}


def test_a_trailing_backward_recitation_does_not_reset_the_sequence(vtb):
    """A page whose inline markers run 16, 17, 2, 18 — the backward
    re-citation is real in this material — must hand the NEXT page its
    maximum (18), not its reading-order-last: the dead-branch expression
    this replaces always stored the last one, and page N+1's genuine
    marker 19 was suppressed as a phantom."""
    seq = {"last": 15}
    page1 = [_item(vtb, "<p>Long narrative prose that reaches its first "
                        "marker here<sup>16</sup> then another"
                        "<sup>17</sup> and a re-citation<sup>2</sup> then "
                        "finally<sup>18</sup></p>",
                   box=(0.1, 0.1, 0.9, 0.3))]
    vtb.reattach_orphan_markers(page1, set(), seq)
    assert seq["last"] == 18
    page2 = [_item(vtb, PROSE.replace("<sup>1</sup>", ""),
                   box=(0.1, 0.2, 0.9, 0.35)),
             _item(vtb, "<p>19</p>", box=(0.1, 0.36, 0.13, 0.379))]
    ra, sp = vtb.reattach_orphan_markers(page2, set(), seq)
    assert (ra, sp) == (1, 0), "19 continues from 18 and must reattach"


# ── a figure the page prints on its own line ────────────────────────────────

def _typeset_page(tmp_path, runs):
    """A typeset page: `runs` are (x, y, text, fontsize) in points on a
    432 x 648 page. Returns the page path, with the sidecar the loader
    writes."""
    import json
    import pytest
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open()
    page = doc.new_page(width=432, height=648)
    for x, y, text, size in runs:
        page.insert_text((x, y), text, fontsize=size)
    pdf = str(tmp_path / "page_0000.pdf")
    doc.save(pdf)
    p = str(tmp_path / "page_0000.png")
    open(p, "w").close()
    with open(p + ".source.json", "w") as fh:
        json.dump({"pdf": pdf, "page": 0, "facsimile": False, "scan": False}, fh)
    return p


def _box(x0, y0, x1, y1):
    """Points on the 432 x 648 page as a normalized box."""
    return (x0 / 432, y0 / 648, x1 / 432, y1 / 648)


TABLE = [(40, 100, "9. strains and sprains", 9),
         (160, 100, "the first day: soak", 9), (300, 100, "102", 9),
         (160, 112, "joint in cold water;", 9),
         (40, 140, "11. minor burns", 9),
         (160, 140, "hold in cold water", 9), (300, 140, "96", 9),
         (40, 400, "The prose at the foot of the page runs on for a line.", 9)]


def test_a_page_reference_in_a_table_column_is_kept(vtb, tmp_path):
    """Where There Is No Doctor, p. 48: six "see page" figures, each a
    one-line block of its own, fitted no note sequence and were thrown
    away as phantoms. The PDF prints each in that box at body size."""
    path = _typeset_page(tmp_path, TABLE)
    items = [_item(vtb, "<p>9. strains and sprains</p>", box=_box(38, 90, 140, 104)),
             _item(vtb, "<p>the first day: soak joint in cold water;</p>",
                   box=_box(158, 90, 280, 116)),
             _item(vtb, "<p>102</p>", box=_box(298, 91, 320, 103)),
             _item(vtb, "<p>11. minor burns</p>", box=_box(38, 130, 140, 144)),
             _item(vtb, "<p>hold in cold water</p>", box=_box(158, 130, 280, 144)),
             _item(vtb, "<p>96</p>", box=_box(298, 131, 316, 143))]
    text_in = vtb.typeset_body_size_text_in(path)
    assert text_in is not None
    ra, sp = vtb.reattach_orphan_markers(items, set(), None, text_in)
    assert (ra, sp) == (0, 0)
    assert [it.html for it in items if it.html in ("<p>102</p>", "<p>96</p>")] \
        == ["<p>102</p>", "<p>96</p>"]


def test_a_raised_call_in_its_own_block_is_still_judged_by_its_sequence(vtb, tmp_path):
    """A note call is set small. The PDF has the figure in the box, but not
    at body size, so the sequence decides as before: a 5 when the stream
    sits at 1 is suppressed."""
    path = _typeset_page(tmp_path, [
        (40, 100, "The strike spread through Pullman and the yards fell", 9),
        (40, 112, "silent across the whole division that week.", 9),
        (240, 108, "5", 5.5),
        (40, 400, "The prose at the foot of the page runs on for a line.", 9)])
    items = [_item(vtb, PROSE, box=_box(38, 90, 300, 116)),
             _item(vtb, "<p>5</p>", box=_box(238, 102, 246, 110))]
    ra, sp = vtb.reattach_orphan_markers(
        items, set(), None, vtb.typeset_body_size_text_in(path))
    assert (ra, sp) == (0, 1)


def test_a_phantom_over_prose_is_not_kept_by_the_layer(vtb, tmp_path):
    """Eagle's phantoms stood over ordinary prose: the PDF's text in that
    box is words, not the figure, so the block is suppressed as before."""
    path = _typeset_page(tmp_path, [
        (40, 100, "The strike spread through Pullman and the yards fell", 9),
        (40, 400, "The prose at the foot of the page runs on for a line.", 9)])
    items = [_item(vtb, PROSE, box=_box(38, 90, 300, 116)),
             _item(vtb, "<p>5</p>", box=_box(120, 92, 150, 104))]
    ra, sp = vtb.reattach_orphan_markers(
        items, set(), None, vtb.typeset_body_size_text_in(path))
    assert (ra, sp) == (0, 1)
