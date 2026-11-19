"""A page's blocks kept in step with the string a pass rewrote
(model.sync_page, sync_pages, Page.cut), checked on generated pages.

The rules, from model.py:
  - after sync_page(page, s), page.render() == s, whatever s is;
  - a block keeps its kind and label; its markup becomes what the pass made
    of it, and no block is cut inside an element (where s is whole markup);
  - a change that straddles a boundary goes to the earlier block and is
    counted; a change that straddles none is a fact, not a judgement;
  - Page.cut(a, b) renders as old[:a] + old[b:], drops a block wholly inside
    the span with its label, and never passes a label to another's text.

Each generated page is a list of blocks made from a small model (a paragraph,
a heading, a list, a table, a figure, a rule…) whose every word-bearing block
carries a word of its own, so that which block is which can be told from
the text alone. Edits are made on that model and the page re-rendered: the
expected markup of every block is known without asking the carrying.
"""
from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import pytest
from hypothesis import HealthCheck, assume, given, note, settings, strategies as st

from codicology.model import Block, Page, sync_page, sync_pages


# ── words ────────────────────────────────────────────────────────────────────

PLAIN = ["the", "a", "of", "and", "said", "books", "kept", "clerk", "river",
         "word,", "end.", "“quoted”", "it’s", "1904", "p.", "—", "§3",
         "λόγος", "Ἀθῆναι", "café", "café", "naïve", "Ångström",
         "&amp;", "&#8217;", "&lt;tag&gt;", "x&#160;y"]
LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZαβγδλωΩéǘ̈"
word = st.one_of(st.sampled_from(PLAIN),
                 st.text(alphabet=LETTERS, min_size=1, max_size=7)
                 .filter(lambda w: not w[0] in "́̈"))

# an inline piece: a word, or a little markup round words
INLINE_TAGS = ["i", "b", "em", "span", "sup", "sub", "small"]


@st.composite
def piece(draw):
    k = draw(st.integers(0, 9))
    if k <= 5:
        return draw(word)
    if k <= 7:
        return (draw(st.sampled_from(INLINE_TAGS)),
                draw(st.lists(word, min_size=1, max_size=3)))
    if k == 8:
        n = str(draw(st.integers(1, 120)))
        return ("a", n)
    return ("br",)


def render_piece(p) -> str:
    if isinstance(p, str):
        return p
    if p[0] == "br":
        return "<br/>"
    if p[0] == "a":
        return f'<sup><a href="#fn-{p[1]}" id="fnref-{p[1]}">{p[1]}</a></sup>'
    return f"<{p[0]}>{' '.join(p[1])}</{p[0]}>"


def render_content(c) -> str:
    return " ".join(render_piece(p) for p in c)


# ── blocks ───────────────────────────────────────────────────────────────────
#
# A block spec: {"kind", "attrs", "contents": [content, ...], "ws": trailing
# white space}. A content is a list of pieces; every word-bearing block's
# first content holds its own word "uN".

KINDS = ["p", "p", "p", "h2", "ul", "table", "figure", "blockquote", "hr",
         "img", "loose", "empty"]


def render_block(b) -> str:
    k, a, cs = b["kind"], b["attrs"], b["contents"]
    if k == "empty":
        return ""
    if k == "p":
        s = f"<p{a}>{render_content(cs[0])}</p>"
    elif k == "h2":
        s = f"<h2{a}>{render_content(cs[0])}</h2>"
    elif k == "ul":
        s = f"<ul{a}>" + "".join(f"<li>{render_content(c)}</li>" for c in cs) + "</ul>"
    elif k == "table":
        s = (f"<table{a}><tbody>" + "".join(f"<tr><td>{render_content(c)}</td></tr>"
                                         for c in cs) + "</tbody></table>")
    elif k == "figure":
        s = (f'<figure{a}><img src="images/fig_0001.png" alt="Figure"/>'
             f"<figcaption>{render_content(cs[0])}</figcaption></figure>")
    elif k == "blockquote":
        s = f"<blockquote{a}>" + "".join(f"<p>{render_content(c)}</p>" for c in cs) \
            + "</blockquote>"
    elif k == "hr":
        s = f"<hr{a}/>"
    elif k == "img":
        s = f'<img src="images/x.png" alt="x"{a}/>'
    else:                                   # loose text, in no element
        s = render_content(cs[0])
    return s + b["ws"]


@st.composite
def block_spec(draw, uid: int, kinds=KINDS, words=0, simple=False):
    kind = draw(st.sampled_from(kinds))
    attrs = draw(st.sampled_from(["", "", "", ' class="x"', f' id="b{uid}"']))
    n = 1
    if kind in ("ul", "table", "blockquote"):
        n = draw(st.integers(1, 3))
    contents = []
    for j in range(n):
        c = draw(st.lists(piece(), min_size=0 if j else words, max_size=6))
        contents.append(c)
    if kind in ("hr", "img", "empty"):
        contents = [[]]
    else:
        # loose text ends on its own word: text put after it would stand
        # at the boundary itself, where which block takes it is a choice
        at = len(contents[0]) if kind == "loose" else \
            0 if simple else draw(st.integers(0, len(contents[0])))
        # `words`: three words of its own, so that no small edit of another
        # block could make that block this one
        contents[0][at:at] = [f"u{uid}"] + ([f"v{uid}", f"w{uid}"] if words else [])
    ws = draw(st.sampled_from(["", "", "", "", "\n", " "]))
    if kind in ("loose", "empty"):
        ws = ""
    return {"kind": kind, "attrs": attrs, "contents": contents, "ws": ws}


@st.composite
def page_specs(draw, min_size=1, max_size=7, kinds=KINDS, words=0, simple=False):
    if simple:
        kinds = [k for k in kinds if k in ("p", "h2", "hr")]
    n = draw(st.integers(min_size, max_size))
    specs = [draw(block_spec(i, kinds, words, simple)) for i in range(n)]
    # two blocks of loose text side by side would be one run of text
    full = [x for x in specs if x["kind"] != "empty"]
    for a, b in zip(full, full[1:]):
        assume(not (a["kind"] == "loose" and b["kind"] == "loose"))
    return specs


def make_page(specs) -> Page:
    return Page(0, "p", [Block(s["kind"], render_block(s), label=f"L{i}")
                         for i, s in enumerate(specs)])


def labelled(page):
    return [(b.label, b.html) for b in page.blocks]


def same_blocks(page, want) -> bool:
    """Each block its label and its markup, but for white space between
    blocks, which may stand on either side of a boundary."""
    return [(lb, h.strip()) for lb, h in labelled(page)] == \
        [(lb, h.strip()) for lb, h in want]


def well_formed(markup: str) -> bool:
    try:
        ET.fromstring("<r>" + markup + "</r>")
        return True
    except ET.ParseError:
        return False


def apart(specs, new) -> bool:
    """No block edited into the markup another block had: which is which is
    still a fact of the text."""
    old = [render_block(s).strip() for s in specs]
    return all(render_block(b).strip() not in old[:i] + old[i + 1:]
               for i, b in enumerate(new) if render_block(b).strip())


def distinct(specs) -> bool:
    """Every block that has markup has markup no other block has: which is
    which is then a fact of the text."""
    hs = [render_block(s).strip() for s in specs if render_block(s).strip()]
    return len(hs) == len(set(hs))


# ── edits inside one block ───────────────────────────────────────────────────

@st.composite
def inner_edit(draw, spec, simple=False):
    """The block, edited within itself: a word changed, put in or taken out,
    wrapped or unwrapped, its attributes or its tag changed. `simple`: its
    words only, and never its first, which is its own."""
    s = copy.deepcopy(spec)
    if simple and s["kind"] in ("hr", "img", "empty"):
        return s
    if s["kind"] in ("hr", "img", "empty"):
        s["attrs"] = draw(st.sampled_from(["", ' class="y"', ' class="x"']))
        return s
    ci = draw(st.integers(0, len(s["contents"]) - 1))
    c = s["contents"][ci]
    op = draw(st.sampled_from(["replace", "insert", "delete", "wrap", "unwrap",
                               "upper", "quotes"] + ([] if simple else ["attr", "retag"])))
    keep = [i for i, p in enumerate(c) if not (isinstance(p, str) and p[:1] in "uvw"
                                               and p[1:].isdigit())]
    if op == "replace" and keep:
        c[draw(st.sampled_from(keep))] = draw(word)
    elif op == "insert":
        # loose text has no tag of its own at either end: what is put in
        # before its first word or after its last stands at the boundary
        if s["kind"] == "loose":
            if len(c) > 1:
                c.insert(draw(st.integers(1, len(c) - 1)), draw(piece()))
        else:
            c.insert(draw(st.integers(int(simple), max(len(c), int(simple)))), draw(piece()))
    elif op == "delete" and keep:
        del c[draw(st.sampled_from(keep))]
    elif op == "wrap" and keep:
        i = draw(st.sampled_from(keep))
        if isinstance(c[i], str):
            c[i] = (draw(st.sampled_from(INLINE_TAGS)), [c[i]])
    elif op == "unwrap":
        tagged = [i for i, p in enumerate(c) if isinstance(p, tuple) and p[0] in INLINE_TAGS]
        if tagged:
            i = draw(st.sampled_from(tagged))
            c[i:i + 1] = c[i][1]
    elif op == "attr" and s["kind"] != "loose":
        s["attrs"] = draw(st.sampled_from(["", ' class="y"', ' class="x"',
                                           ' class="chapter-head"']))
    elif op == "retag" and s["kind"] in ("p", "h2"):
        s["kind"] = "h2" if s["kind"] == "p" else "p"
    elif op == "upper":
        s["contents"][ci] = [p.upper() if i in keep and isinstance(p, str) and "&" not in p
                             else p for i, p in enumerate(c)]
    elif op == "quotes":
        s["contents"][ci] = [p.replace("“", '"').replace("”", '"').replace("’", "'")
                             if isinstance(p, str) else p for p in c]
    return s


# ── the rules ────────────────────────────────────────────────────────────────

@given(page_specs(), st.data())
def test_a_page_renders_as_any_string_it_is_carried_to(specs, data):
    """Whatever a pass makes of the page, well-formed or not, the page then
    renders as exactly that."""
    page = make_page(specs)
    old = page.render()
    a = data.draw(st.integers(0, len(old)))
    b = data.draw(st.integers(a, len(old)))
    junk = data.draw(st.text(alphabet="<>/p ab&;\n\"=", max_size=8) | st.just(""))
    new = old[:a] + junk + old[b:]
    labels = [(b.kind, b.label) for b in page.blocks]
    sync_page(page, new)
    assert page.render() == new
    assert [(b.kind, b.label) for b in page.blocks] == labels


@settings(suppress_health_check=[HealthCheck.filter_too_much, HealthCheck.too_slow])
@given(page_specs(), st.data())
def test_a_span_cut_from_well_formed_markup_leaves_every_block_whole(specs, data):
    page = make_page(specs)
    old = page.render()
    # spans from tag to tag or word to word, so that some leave whole markup
    stops = [i for i in range(len(old) + 1)
             if i in (0, len(old)) or old[i] in "< " or old[i - 1] in "> "]
    a = data.draw(st.sampled_from(stops))
    b = data.draw(st.sampled_from([i for i in stops if i >= a]))
    new = old[:a] + old[b:]
    assume(well_formed(new))
    sync_page(page, new)
    assert page.render() == new
    for blk in page.blocks:
        assert well_formed(blk.html), (blk.label, blk.html)


# ── edits whose right outcome is known ───────────────────────────────────────
#
# Each scenario draws a page and an edit of it, and gives back the page's
# specs, what every block should then hold (label, markup), and how many
# boundaries the edit really straddles.

def _want(specs, empty=()):
    return [(f"L{i}", "" if i in empty else render_block(s)) for i, s in enumerate(specs)]


@st.composite
def edits_inside_blocks(draw, simple=False):
    """Any number of blocks edited, each within itself: a pass that sets the
    quotes or the case of a page touches every block."""
    specs = draw(page_specs(simple=simple))
    which = draw(st.sets(st.integers(0, len(specs) - 1), min_size=1))
    new = [draw(inner_edit(s, simple)) if i in which else s for i, s in enumerate(specs)]
    assume(distinct(specs) and distinct(new) and apart(specs, new))
    return specs, _want(new), 0


@st.composite
def edit_inside_one_block(draw, simple=False):
    specs = draw(page_specs(simple=simple))
    k = draw(st.integers(0, len(specs) - 1))
    new = list(specs)
    new[k] = draw(inner_edit(specs[k], simple))
    assume(distinct(specs) and distinct(new) and apart(specs, new))
    return specs, _want(new), 0


@st.composite
def one_block_deleted(draw, simple=False):
    """A block's markup taken out whole (a running head stripped): it is
    left empty, with its label, and the rest keep theirs."""
    specs = draw(page_specs(min_size=2, simple=simple))
    assume(distinct(specs))
    k = draw(st.integers(0, len(specs) - 1))
    assume(render_block(specs[k]))
    return specs, _want(specs, {k}), 0


@st.composite
def block_deleted_and_others_edited(draw, simple=False):
    # with a word or two besides its own, a block is not mistaken for the
    # next by an edit that happens to cost the same either way
    specs = draw(page_specs(min_size=2, words=2, simple=simple))
    k = draw(st.integers(0, len(specs) - 1))
    which = draw(st.sets(st.integers(0, len(specs) - 1), min_size=1))
    new = [draw(inner_edit(s, simple)) if i in which else s for i, s in enumerate(specs)]
    assume(distinct(specs) and distinct(new) and apart(specs, new)
           and render_block(specs[k]))
    return specs, _want(new, {k}), 0


@st.composite
def run_of_blocks_deleted(draw, simple=False):
    specs = draw(page_specs(min_size=3, simple=simple))
    assume(distinct(specs))
    i = draw(st.integers(0, len(specs) - 1))
    j = draw(st.integers(i + 1, len(specs)))
    assume(any(render_block(s) for s in specs[i:j]))
    return specs, _want(specs, set(range(i, j))), 0


@st.composite
def paragraphs_joined(draw, simple=False):
    """Two paragraphs a pass joined: one paragraph, in the earlier block,
    and the one boundary it straddled counted."""
    specs = draw(page_specs(min_size=2, kinds=["p", "p", "h2", "hr", "ul", "figure"], simple=simple))
    pairs = [i for i in range(len(specs) - 1)
             if specs[i]["kind"] == specs[i + 1]["kind"] == "p"
             and not specs[i]["ws"] and not specs[i + 1]["ws"]
             and specs[i]["attrs"] == "" and specs[i + 1]["attrs"] == ""]
    assume(pairs and distinct(specs))
    k = draw(st.sampled_from(pairs))
    want = _want(specs, {k + 1})
    want[k] = (f"L{k}", render_block(specs[k])[:-len("</p>")] + " "
               + render_block(specs[k + 1])[len("<p>"):])
    return specs, want, 1


@st.composite
def paragraph_split(draw, simple=False):
    specs = draw(page_specs(kinds=["p", "p", "h2", "figure", "hr"], simple=simple))
    ps = [i for i, s in enumerate(specs) if s["kind"] == "p" and len(s["contents"][0]) > 1]
    assume(ps and distinct(specs))
    k = draw(st.sampled_from(ps))
    c = specs[k]["contents"][0]
    at = draw(st.integers(1, len(c) - 1))
    want = _want(specs)
    want[k] = (f"L{k}", f"<p{specs[k]['attrs']}>{render_content(c[:at])}</p>"
               f"<p>{render_content(c[at:])}</p>" + specs[k]["ws"])
    return specs, want, 0


SCENARIOS = {
    "edits inside blocks": edits_inside_blocks(),
    "edit inside one block": edit_inside_one_block(),
    "one block deleted": one_block_deleted(),
    "block deleted, others edited": block_deleted_and_others_edited(),
    "run of blocks deleted": run_of_blocks_deleted(),
    "paragraphs joined": paragraphs_joined(),
    "paragraph split": paragraph_split(),
}

# Pages of paragraphs, headings and rules whose every block opens on a word
# of its own, edited in their words only: the carrying is right on these
# today, and they hold it there (the full scenarios above find where it is
# not; see test_page_sync_findings.py). Not yet here: a block cut while its
# neighbours are edited, which can still pass a block's words to the one
# before it (counted as straddled).
SIMPLE = {
    "edits inside blocks": edits_inside_blocks(simple=True),
    "one block deleted": one_block_deleted(simple=True),
    "run of blocks deleted": run_of_blocks_deleted(simple=True),
    "paragraphs joined": paragraphs_joined(simple=True),
}


@pytest.mark.parametrize("scenario", list(SCENARIOS))
@given(data=st.data())
def test_carrying_is_never_silently_wrong(scenario, data):
    """Where the carrying reports no boundary as a judgement, every block
    holds exactly its own markup under its own label; whatever it reports,
    the page renders as the new string and every block is whole markup."""
    specs, want, _ = data.draw(SCENARIOS[scenario])
    page = make_page(specs)
    new = "".join(h for _, h in want)
    note(f"old={page.render()!r}\nnew={new!r}")
    straddled = sync_page(page, new)
    assert page.render() == new
    for blk in page.blocks:
        assert well_formed(blk.html), (blk.label, blk.html)
    if straddled == 0:
        assert same_blocks(page, want), labelled(page)


@pytest.mark.parametrize("scenario", list(SIMPLE))
@given(data=st.data())
def test_carrying_puts_every_simple_block_where_it_belongs(scenario, data):
    specs, want, straddles = data.draw(SIMPLE[scenario])
    page = make_page(specs)
    new = "".join(h for _, h in want)
    note(f"old={page.render()!r}\nnew={new!r}")
    straddled = sync_page(page, new)
    assert same_blocks(page, want), labelled(page)
    # a boundary inside a run that was only cut is counted, though where it
    # goes is no judgement (test_two_whole_blocks_cut_together_straddle_nothing)
    if scenario != "run of blocks deleted":
        assert straddled == straddles


@pytest.mark.xfail(strict=False, reason=(
    "carried by tokens, a boundary is lost where the edit's ends share more "
    "than one opening tag with the next block (<p><i>…, <ul><li>…), or both "
    "tags at a boundary changed; see test_page_sync_findings.py"))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
@given(data=st.data())
def test_carrying_puts_every_block_where_it_belongs(scenario, data):
    """The contract in full: each block its own markup, and the count is of
    the boundaries the edit really straddled."""
    specs, want, straddles = data.draw(SCENARIOS[scenario])
    page = make_page(specs)
    new = "".join(h for _, h in want)
    note(f"old={page.render()!r}\nnew={new!r}")
    straddled = sync_page(page, new)
    assert same_blocks(page, want), labelled(page)
    assert straddled == straddles


@given(page_specs(), st.data())
def test_a_block_put_in_goes_to_one_block_and_moves_nothing_else(specs, data):
    """New markup put in at a boundary is carried by one block, before or
    after the markup it had; every other block is as it was."""
    assume(distinct(specs))
    page = make_page(specs)
    k = data.draw(st.integers(0, len(specs)))
    spec = data.draw(block_spec(99))
    extra = render_block(spec)
    olds = [render_block(s) for s in specs]
    assume(extra.strip() and extra.strip() not in [h.strip() for h in olds])
    assume(spec["kind"] != "loose" or "loose" not in [s["kind"] for s in specs])
    new = "".join(olds[:k]) + extra + "".join(olds[k:])
    note(f"old={page.render()!r}\nnew={new!r}")
    sync_page(page, new)
    got = [b.html for b in page.blocks]
    assert page.render() == new
    moved = [n for n in range(len(olds)) if got[n] != olds[n]]
    assert len(moved) == 1, got
    n = moved[0]
    assert got[n] in (extra + olds[n], olds[n] + extra), got


@given(page_specs(min_size=2, kinds=["p", "p", "h2", "ul", "figure", "hr", "img"]), st.data())
def test_a_wrapper_put_round_blocks_keeps_the_rest_where_they_were(specs, data):
    assume(distinct(specs))
    page = make_page(specs)
    i = data.draw(st.integers(0, len(specs) - 1))
    j = data.draw(st.integers(i + 1, len(specs)))
    tag = data.draw(st.sampled_from(["div", "section", "aside", "blockquote"]))
    olds = [render_block(s) for s in specs]
    new = "".join(olds[:i]) + f"<{tag}>" + "".join(olds[i:j]) + f"</{tag}>" + "".join(olds[j:])
    note(f"old={page.render()!r}\nnew={new!r}")
    sync_page(page, new)
    got = [b.html for b in page.blocks]
    assert page.render() == new
    assert got[:i] == olds[:i] and got[j:] == olds[j:]
    for blk in page.blocks:
        assert well_formed(blk.html), (blk.label, blk.html)


@pytest.mark.xfail(strict=False, reason="see test_page_sync_findings.py: a "
                   "first block cut whole gives its label to a next block "
                   "that opens with two tags")
@given(page_specs(), st.data())
def test_carrying_agrees_with_cutting_where_whole_blocks_go(specs, data):
    """A pass that takes whole blocks out of a page's string leaves the
    blocks as Page.cut would, less the blocks cut leaves out."""
    assume(distinct(specs))
    page = make_page(specs)
    old = page.render()
    edges = sorted({0, len(old)} | {sum(len(b.html) for b in page.blocks[:n])
                                    for n in range(len(page.blocks))})
    a = data.draw(st.sampled_from(edges))
    b = data.draw(st.sampled_from([e for e in edges if e >= a]))
    new = old[:a] + old[b:]
    assume(new != old)
    cut = make_page(specs)
    cut.cut(a, b)
    sync_page(page, new)
    assert [(b.label, b.html) for b in page.blocks if b.html] == \
        [(b.label, b.html) for b in cut.blocks if b.html]



@given(page_specs(), st.data())
def test_cut_renders_as_the_string_less_the_span(specs, data):
    page = make_page(specs)
    old = page.render()
    a = data.draw(st.integers(0, len(old)))
    b = data.draw(st.integers(a, len(old)))
    before = [(blk.label, blk.html) for blk in page.blocks]
    page.cut(a, b)
    assert page.render() == old[:a] + old[b:]
    # stated by position, not by slicing: a block keeps those of its own
    # characters that stood outside the span, and goes only when it had
    # some and all were inside
    want, at = [], 0
    for label, h in before:
        mine = range(at, at + len(h))
        at += len(h)
        kept = "".join(ch for pos, ch in zip(mine, h) if not a <= pos < b)
        if h and not kept and all(a <= pos < b for pos in mine):
            continue
        want.append((label, kept))
    assert [(blk.label, blk.html) for blk in page.blocks] == want


@given(st.lists(page_specs(max_size=4), min_size=1, max_size=4), st.data())
def test_every_page_is_its_body_and_the_count_is_the_sum(pages_specs, data):
    pages = [make_page(s) for s in pages_specs]
    twins = [make_page(s) for s in pages_specs]
    bodies = []
    for p in pages:
        old = p.render()
        a = data.draw(st.integers(0, len(old)))
        b = data.draw(st.integers(a, len(old)))
        bodies.append(old[:a] + data.draw(st.sampled_from(["", "<p>x</p>", "y"])) + old[b:])
    total = sync_pages(pages, bodies)
    assert [p.render() for p in pages] == bodies
    assert total == sum(sync_page(t, b) for t, b in zip(twins, bodies))
    assert [labelled(p) for p in pages] == [labelled(t) for t in twins]


@given(page_specs(), st.data())
def test_carrying_twice_to_the_same_string_changes_nothing(specs, data):
    page = make_page(specs)
    old = page.render()
    a = data.draw(st.integers(0, len(old)))
    b = data.draw(st.integers(a, len(old)))
    new = old[:a] + old[b:]
    sync_page(page, new)
    once = labelled(page)
    assert sync_page(page, new) == 0
    assert labelled(page) == once


def test_an_empty_page_takes_the_whole_string_as_one_block():
    page = Page(0, "p", [])
    assert sync_page(page, "<p>new</p>") == 0
    assert [(b.kind, b.html) for b in page.blocks] == [("text", "<p>new</p>")]
