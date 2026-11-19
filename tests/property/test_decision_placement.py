"""A reviewer's saved corrections, placed on any page.

The contract (review.apply_page_decisions, _locate; decisions.
apply_reviewer_decisions): a correction is addressed to one word of its
page — its ordinal among all the page's instances of the token, and the
words either side (adjudicate.site_address). It lands on that word and
nowhere else; a decision that cannot be placed with certainty is stale,
reported and left as read, never guessed at; deletions run before swaps;
every swap is located on the same text before any is written.

Decisions here are made the way a sheet makes them: from a generated
page's own token stream, through site_address.
"""
import html
import json
import os
import re
import tempfile
import unicodedata

import pytest
from hypothesis import assume, example, given, note
from hypothesis import strategies as st

from codicology import adjudicate as adj
from codicology import decisions as decisions_mod
from codicology import review

from .pages import FRESH, VOCAB, LATIN, page_body, reader_text

GENERATIONS = [None, adj.TOKENS]


def _toks(body):
    return adj.tokens(reader_text(body))


def _fix(toks, i, new, page=0):
    """The row a sheet exports for the i-th word of a page."""
    return {"page": page, "old": toks[i], "new": new, "source": "human",
            "rung": "human", **adj.site_address(toks, i)}


def _continues(ch):
    return unicodedata.category(ch)[0] in "LM"


def _whole(folded, at, n):
    """The token at folded[at:at+n] is a whole printed word: no letter or
    accent runs on from either end (the documented meaning of a piece of
    a longer word)."""
    return not ((at > 0 and _continues(folded[at - 1]))
                or (at + n < len(folded) and _continues(folded[at + n])))


def _editable(body):
    """The token indices a correction can certainly be written at."""
    folded, words = adj.page_words(body)
    return [k for k, (t, lo, hi, at) in enumerate(words)
            if lo is not None and _whole(folded, at, len(t))]


def _expected(body, edits):
    """The body with each (token index, new) written over that word's span
    — the word's own markup, nothing else."""
    words = adj.page_words(body)[1]
    out = body
    for k, new in sorted(edits, key=lambda e: -words[e[0]][1]):
        _, lo, hi, _ = words[k]
        out = out[:lo] + html.escape(new, quote=False) + out[hi:]
    return out


@st.composite
def page_and_site(draw, vocab=VOCAB, breaks=True):
    body = draw(page_body(vocab, breaks))
    ed = _editable(body)
    assume(ed)
    i = draw(st.sampled_from(ed))
    new = draw(st.sampled_from(FRESH))
    return body, i, new


# ── one correction ───────────────────────────────────────────────────────

@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_correction_lands_on_its_word_and_nowhere_else(gen, data):
    with adj.tokens_as(gen):
        body, i, new = data.draw(page_and_site())
        toks = _toks(body)
        row = _fix(toks, i, new)
        got, (why,) = review.apply_page_decisions(body, [row])
        assert why is None, why
        assert got == _expected(body, [(i, new)])
        assert _toks(got) == toks[:i] + adj.tokens(new) + toks[i + 1:]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_applying_the_same_correction_again_changes_nothing(gen, data):
    """`codicology apply` works in place, and a reviewer who goes back to
    the sheet exports the earlier decisions again with the new ones.

    Here the reviewer's word is one the file's tokeniser reads as a word;
    the next test covers one it cannot see."""
    with adj.tokens_as(gen):
        body, i, new = data.draw(page_and_site())
        assume(adj.tokens(new))
        row = _fix(_toks(body), i, new)
        once, (why,) = review.apply_page_decisions(body, [row])
        assume(why is None)
        twice, (why2,) = review.apply_page_decisions(once, [row])
        assert twice == once
        assert why2 is not None


INVISIBLE = {None: ["1", "8", "&c.", "ἄλλος", "λόγος"],   # Greek: not Latin-1
             adj.TOKENS: ["1", "8", "&c.", "0"]}


# fixed 2026-11-19
@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_correction_to_an_invisible_reading_is_not_applied_twice(gen, data):
    with adj.tokens_as(gen):
        new = data.draw(st.sampled_from(INVISIBLE[gen]))
        # "&c." shows the tokens a "c": that one is guarded as any visible
        # reading is (the same neighbours around `new`)
        assume(not adj.tokens(new))
        body, i, _ = data.draw(page_and_site())
        row = _fix(_toks(body), i, new)
        once, (why,) = review.apply_page_decisions(body, [row])
        assume(why is None)
        twice, _ = review.apply_page_decisions(once, [row])
        assert twice == once


# fixed 2026-11-19
@pytest.mark.parametrize("gen,new", [(adj.TOKENS, "1"), (None, "1"),
                                     (None, "λόγος")])
def test_l_to_1_applied_twice_leaves_the_second_l(gen, new):
    body = "<p>see l and the rest of it; see l and the end of it</p>"
    with adj.tokens_as(gen):
        row = _fix(_toks(body), 1, new)
        once, (why,) = review.apply_page_decisions(body, [row])
        assert why is None
        assert once == body.replace("l", new, 1)
        twice, (why2,) = review.apply_page_decisions(once, [row])
    assert twice == once, twice


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_an_edit_away_from_the_word_does_not_move_it(gen, data):
    """An edit outside the recorded neighbours — a word that was not an
    instance of the corrected one, now read as something else — leaves the
    correction landing on the same word."""
    with adj.tokens_as(gen):
        body, i, new = data.draw(page_and_site())
        toks = _toks(body)
        row = _fix(toks, i, new)
        words = adj.page_words(body)[1]
        far = [k for k in _editable(body)
               if abs(k - i) > adj.SITE_CONTEXT and toks[k] != toks[i]]
        assume(far)
        j = data.draw(st.sampled_from(far))
        other = data.draw(st.sampled_from([w for w in FRESH if w != new]))
        _, lo, hi, _ = words[j]
        edited = body[:lo] + html.escape(other, quote=False) + body[hi:]
        note(f"edited: {edited!r}")
        got, (why,) = review.apply_page_decisions(edited, [row])
        assert why is None, why
        # the corrected word's own markup, moved by the edit if it stood
        # before it
        _, wlo, whi, _ = words[i]
        shift = len(edited) - len(body) if hi <= wlo else 0
        want = (edited[:wlo + shift] + html.escape(new, quote=False)
                + edited[whi + shift:])
        assert got == want


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_word_whose_neighbours_have_changed_is_not_corrected(gen, data):
    """More than half of the recorded neighbours read otherwise now: that
    instance is no longer certainly the one the reviewer saw. It is never
    corrected — and where the page prints the word only there, the
    decision is stale."""
    with adj.tokens_as(gen):
        body, i, new = data.draw(page_and_site())
        toks = _toks(body)
        row = _fix(toks, i, new)
        n = len(row["before"]) + len(row["after"])
        assume(n >= 1)
        near = [k for k in range(i - len(row["before"]),
                                 i + 1 + len(row["after"])) if k != i]
        words = adj.page_words(body)[1]
        assume(all(words[k][1] is not None for k in near))
        m = data.draw(st.integers(n // 2 + 1, n))
        hit = data.draw(st.lists(st.sampled_from(near), min_size=m,
                                 max_size=m, unique=True))
        visible = [w for w in FRESH if w != new and adj.tokens(w) == [w]]
        changed = body
        for k in sorted(hit, key=lambda k: -words[k][1]):
            _, lo, hi, _ = words[k]
            changed = (changed[:lo]
                       + html.escape(data.draw(st.sampled_from(visible)),
                                     quote=False) + changed[hi:])
        got, (why,) = review.apply_page_decisions(changed, [row])
        if toks.count(toks[i]) == 1:
            assert why is not None and got == changed
        elif why is None and adj.tokens(new) == [new]:
            # it may have gone to another instance (see the finding on
            # half-matching instances), but never to this one; every edit
            # here is one word for one, so the word is still the i-th
            assert _toks(got)[i] == toks[i]


@pytest.mark.xfail(strict=True, reason=(
    "FINDING (design, ambiguous): when the corrected word is no longer at "
    "its site, _locate still accepts any other instance whose recorded "
    "neighbours half match (2*best >= n) and writes the correction there"))
@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_word_that_is_gone_is_never_placed_elsewhere(gen, data):
    """The page now reads something else where the corrected word stood
    (the build read it differently). The correction is stale — or, if the
    page reads the reviewer's own word there, already in — never written
    over another instance of the old word."""
    with adj.tokens_as(gen):
        body, i, new = data.draw(page_and_site())
        toks = _toks(body)
        row = _fix(toks, i, new)
        words = adj.page_words(body)[1]
        reading = data.draw(st.sampled_from(
            [w for w in FRESH if w != new] + [new]))
        _, lo, hi, _ = words[i]
        changed = body[:lo] + html.escape(reading, quote=False) + body[hi:]
        note(f"changed: {changed!r}")
        got, (why,) = review.apply_page_decisions(changed, [row])
        assert why is not None, f"placed elsewhere: {got!r}"
        assert got == changed


@pytest.mark.xfail(strict=True, reason="FINDING (design): as above")
def test_a_gone_word_is_stale_not_moved_to_a_half_matching_instance():
    body = "<p>see navy and the rest of it; see navy and the end of it</p>"
    row = _fix(_toks(body), 1, "navvy")
    changed = body.replace("navy", "nayy", 1)
    got, (why,) = review.apply_page_decisions(changed, [row])
    # actual: why is None and got reads "... see navvy and the end of it"
    assert why is not None and got == changed, got


# ── look-alike places ────────────────────────────────────────────────────

SENTINEL = "Qwertz"     # a word printed once, at the head of the page


@st.composite
def repeated_phrase_page(draw, vocab=VOCAB):
    """A page printing the same nine-word phrase twice (and maybe more
    around it), so the corrected word's whole address fits two places."""
    words = draw(st.lists(st.sampled_from(vocab), min_size=9, max_size=9))
    phrase = " ".join(words)
    lead = [SENTINEL] + draw(st.lists(st.sampled_from(vocab), max_size=6))
    mid = draw(st.lists(st.sampled_from(vocab), min_size=1, max_size=6))
    body = (f"<p>{' '.join(lead)}. {phrase}. "
            f"{' '.join(mid)}; {phrase}.</p>")
    return body, len(lead), len(lead) + 9 + len(mid)


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data(), second=st.booleans())
def test_identical_addresses_are_told_apart_by_the_ordinal(gen, data, second):
    with adj.tokens_as(gen):
        whole = [w for w in VOCAB if adj.tokens(w) == [w.translate(adj._TYPO)]]
        body, a, b = data.draw(repeated_phrase_page(whole))
        toks = _toks(body)
        assume(len(toks) == b + 9)   # every word one token, in this script
        i = (b if second else a) + 4
        assume(i in _editable(body))
        new = data.draw(st.sampled_from(FRESH))
        row = _fix(toks, i, new)
        assert row["before"] == toks[i - 4:i]
        got, (why,) = review.apply_page_decisions(body, [row])
        assert why is None, why
        assert got == _expected(body, [(i, new)])
        # once the page has lost a run, the ordinal counts a page that is
        # gone: the look-alikes must not be guessed between
        run = {"page": 0, "kind": "delete_run", "old": SENTINEL, "new": ""}
        got, why = review.apply_page_decisions(body, [run, row])
        assert why[0] is None
        assert why[1] is not None and "equally" in why[1], why
        assert SENTINEL not in got and new not in got


# ── deletions ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_deletion_removes_a_run_where_it_stands_or_nothing(gen, data):
    """A run of the page's own words is removed whole, from a place the
    page prints it, or reported stale — never a partial deletion.

    Pages here have no line-break hyphens: with them, see the finding
    pinned by test_a_deletion_never_takes_half_a_broken_word."""
    with adj.tokens_as(gen):
        body = data.draw(page_body(breaks=False))
        toks = _toks(body)
        assume(toks)
        j = data.draw(st.integers(0, len(toks) - 1))
        n = data.draw(st.integers(1, 4))
        run = toks[j:j + n]
        row = {"page": 0, "kind": "delete_run", "old": " ".join(run),
               "new": ""}
        got, (why,) = review.apply_page_decisions(body, [row])
        if why is not None:
            assert got == body
            return
        left = _toks(got)
        assert any(toks[p:p + len(run)] == run
                   and left == toks[:p] + toks[p + len(run):]
                   for p in range(len(toks))), (toks, left)


# fixed 2026-11-19
@pytest.mark.parametrize("gen", GENERATIONS)
def test_a_deletion_never_takes_half_a_broken_word(gen):
    body = "<p>the trans-\nport of the goods; a port of the town</p>"
    with adj.tokens_as(gen):
        row = {"page": 0, "kind": "delete_run", "old": "port of the",
               "new": ""}
        assert "port of the" in " ".join(_toks(body))
        got, (why,) = review.apply_page_decisions(body, [row])
    # actual: "<p>the trans-\n goods; a port of the town</p>"
    assert why is not None or got == "<p>the trans-\nport of the goods; a town</p>", got


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_run_of_plainly_stored_words_is_always_found(gen, data):
    with adj.tokens_as(gen):
        plain = [w for w in VOCAB if "ﬁ" not in w and adj.tokens(w) == [w]
                 and "’" not in w]
        body = data.draw(page_body(plain, breaks=False))
        assume("&" not in body)
        toks = _toks(body)
        assume(toks)
        j = data.draw(st.integers(0, len(toks) - 1))
        run = toks[j:j + data.draw(st.integers(1, 4))]
        row = {"page": 0, "kind": "delete_run", "old": " ".join(run),
               "new": ""}
        got, (why,) = review.apply_page_decisions(body, [row])
        assert why is None


@pytest.mark.xfail(strict=True, reason=(
    "FINDING (minor, stale not wrong): a surya-only run is recorded in "
    "tokens (fold_text: ﬁ -> fi, a line-break hyphen joined, entities "
    "unescaped) but _delete_from_xhtml matches the raw xhtml text, so a "
    "run holding a ligature, a hyphen-broken word or an entity-stored "
    "letter can never be deleted"))
@pytest.mark.parametrize("body", [
    "<p>the ﬁnal word of it</p>",
    "<p>the crys-\ntal word of it</p>",
    "<p>the caf&eacute; word of it</p>",
    "<p>the don&#8217;t word of it</p>",
    "<p>so the &amp; word of it</p>",
])
def test_a_run_holding_a_folded_word_can_be_deleted(body):
    toks = _toks(body)
    row = {"page": 0, "kind": "delete_run", "old": " ".join(toks[1:3]),
           "new": ""}
    got, (why,) = review.apply_page_decisions(body, [row])
    assert why is None, why


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_swap_after_a_deletion_lands_on_its_word_or_is_stale(gen, data):
    with adj.tokens_as(gen):
        body, i, new = data.draw(page_and_site(breaks=False))
        toks = _toks(body)
        j = data.draw(st.integers(0, len(toks) - 1))
        n = data.draw(st.integers(1, 3))
        run = {"page": 0, "kind": "delete_run",
               "old": " ".join(toks[j:j + n]), "new": ""}
        row = _fix(toks, i, new)
        got, why = review.apply_page_decisions(body, [row, run])
        if why[1] is not None:
            assume(False)
        after = next(g for g in (review._delete_from_xhtml(body, shape)
                                 for shape in review._shapes(run["old"]))
                     if g is not None)
        if why[0] is not None:
            assert got == after
            return
        # it landed: the deleted page with one word, an instance of the
        # corrected one, swapped for the reviewer's
        want = _toks(after)
        got_t = _toks(got)
        assert len(got_t) == len(want) - 1 + len(adj.tokens(new))
        diff = [k for k, (x, y) in enumerate(zip(want, got_t)) if x != y]
        if adj.tokens(new) == [new]:
            assert len(diff) == 1 and want[diff[0]] == toks[i]


# ── several corrections ──────────────────────────────────────────────────

@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_corrections_on_one_page_do_not_interfere(gen, data):
    """All located on the page as it was read, then written: each lands
    where it would have landed alone, neighbours included."""
    with adj.tokens_as(gen):
        body = data.draw(page_body())
        ed = _editable(body)
        assume(len(ed) >= 2)
        sites = data.draw(st.lists(st.sampled_from(ed), min_size=2,
                                   max_size=min(6, len(ed)), unique=True))
        toks = _toks(body)
        news = [data.draw(st.sampled_from(FRESH)) for _ in sites]
        rows = [_fix(toks, i, n) for i, n in zip(sites, news)]
        alone = [review.apply_page_decisions(body, [r])[1][0] for r in rows]
        assert alone == [None] * len(rows)
        order = data.draw(st.permutations(range(len(rows))))
        got, why = review.apply_page_decisions(body, [rows[k] for k in order])
        assert why == [None] * len(rows), why
        assert got == _expected(body, list(zip(sites, news)))


# ── rows from before words were addressed ────────────────────────────────

@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_bare_row_lands_only_when_the_file_names_every_instance(gen, data):
    with adj.tokens_as(gen):
        body = data.draw(page_body(breaks=False))
        toks = _toks(body)
        ed = _editable(body)
        assume(ed)
        old = toks[data.draw(st.sampled_from(ed))]
        where = [k for k, t in enumerate(toks) if t == old]
        assume(all(k in ed for k in where))
        named = data.draw(st.integers(1, len(where)))
        rows = [{"page": 0, "occurrence": o, "old": old,
                 "new": FRESH[o % len(FRESH)]} for o in range(named)]
        got, why = review.apply_page_decisions(body, rows)
        if named == len(where):
            assert why == [None] * named, why
            assert got == _expected(
                body, [(where[o], r["new"]) for o, r in enumerate(rows)])
        else:
            assert got == body
            assert all(w is not None for w in why)


# ── through the rebuild's door, with the file's generation ───────────────

def _apply_file(body, rows, gen):
    meta = {"decisions": rows}
    if gen is not None:
        meta["tokens"] = gen
    fd, path = tempfile.mkstemp(suffix=".decisions.json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(meta, fh)
        bodies = [body]
        st_ = decisions_mod.apply_reviewer_decisions(bodies, path)
    finally:
        os.unlink(path)
    return bodies[0], st_


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_a_file_is_read_in_the_generation_it_was_made_in(gen, data):
    """A file made under a generation places its rows under that
    generation; written through apply_reviewer_decisions, a row made from
    the page lands on its word. Under the old pattern a row naming a
    fragment of a non-Latin-1 word (pl for plēthos) is stale, and nothing
    is ever written inside a longer word."""
    body = data.draw(page_body())
    with adj.tokens_as(gen):
        folded, words = adj.page_words(body)
        toks = [w[0] for w in words]
        assume(toks)
        i = data.draw(st.integers(0, len(toks) - 1))
        new = data.draw(st.sampled_from(FRESH))
        row = _fix(toks, i, new)
        t, lo, hi, at = words[i]
        must = lo is not None and _whole(folded, at, len(t))
    got, counts = _apply_file(body, [row], gen)
    if must:
        assert counts == {"applied": 1, "stale": 0}
        with adj.tokens_as(gen):
            assert got == _expected(body, [(i, new)])
    else:
        assert counts == {"applied": 0, "stale": 1}
        assert got == body
    # under any script, the page's words are its old words with at most
    # the one changed
    with adj.tokens_as(adj.TOKENS):
        a, b = _toks(body), _toks(got)
    if got != body:
        assert len(a) == len(b)
        assert sum(x != y for x, y in zip(a, b)) == 1


@given(data=st.data())
def test_an_nfd_page_takes_a_correction_made_from_it(data):
    """The page's text decomposed (e + U+0304): a sheet made from that
    page addresses NFD tokens, and the correction lands on them."""
    body = unicodedata.normalize("NFD", data.draw(page_body(breaks=False)))
    with adj.tokens_as(adj.TOKENS):
        ed = _editable(body)
        assume(ed)
        i = data.draw(st.sampled_from(ed))
        new = data.draw(st.sampled_from(FRESH))
        row = _fix(_toks(body), i, new)
        got, (why,) = review.apply_page_decisions(body, [row])
        assert why is None, why
        assert got == _expected(body, [(i, new)])


# fixed 2026-11-19
@pytest.mark.parametrize("tag", ["i", "em", "b", "span"])
def test_a_deletion_that_empties_a_block_drops_it(tag):
    body = f"<p>the navy and</p>\n<p><{tag}>of the end</{tag}></p>"
    got, (why,) = review.apply_page_decisions(
        body, [{"page": 0, "kind": "delete_run", "old": "of the end",
                "new": ""}])
    assert why is None
    assert got.rstrip() == "<p>the navy and</p>", got


# ── what the page prints hard against the word ───────────────────────────

PRE = ["", "", "8", "(", "“", "18", "["]
SUF = ["", "", "8", ")", ".", ",", "”", "2)", "]"]


@pytest.mark.parametrize("gen", GENERATIONS)
@given(data=st.data())
def test_what_stands_against_the_word_is_not_written_twice(gen, data):
    """A reviewer types the printed thing: the corrected word with the
    marks and digits printed hard against it. Those are written once —
    the page then prints exactly what the reviewer typed, where the word
    stood (_fitted)."""
    with adj.tokens_as(gen):
        plain = [w for w in VOCAB if adj.tokens(w) == [w.translate(adj._TYPO)]
                 and "’" not in w]
        words = data.draw(st.lists(st.sampled_from(plain), min_size=1,
                                   max_size=12))
        k = data.draw(st.integers(0, len(words) - 1))
        pre = data.draw(st.sampled_from(PRE))
        suf = data.draw(st.sampled_from(SUF))
        w2 = data.draw(st.sampled_from([w for w in FRESH
                                        if adj.tokens(w) == [w]]))
        printed = [html.escape(w, quote=False) for w in words]
        printed[k] = html.escape(pre + words[k] + suf, quote=False)
        body = "<p>" + " ".join(printed) + "</p>"
        toks = _toks(body)
        assert toks == [w.translate(adj._TYPO) for w in words]
        row = _fix(toks, k, pre + w2 + suf)
        got, (why,) = review.apply_page_decisions(body, [row])
        assert why is None, why
        printed[k] = html.escape(pre + w2 + suf, quote=False)
        assert got == "<p>" + " ".join(printed) + "</p>"
