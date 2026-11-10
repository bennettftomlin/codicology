"""A correction lands on the word it was made for.

A dispute names one word of one page, and a page may print that word many
times. Until 2026-11-10 the review sheet numbered a dispute among the
page's DISPUTES sharing its token while the apply step counted every
instance, so wherever an undisputed instance stood earlier on the page
the correction landed on it instead: six installed books had the wrong
word rewritten ("overheating should by avoided").

What is pinned here: the adjudicator addresses a dispute by its place in
the page's own token stream (the ordinal among all instances, and the
neighbours either side); the appliers read a page through the very same
tokeniser, keeping each word's place in the markup; and a decision that
cannot be placed with certainty is stale — said aloud, never guessed at.
"""
import html
import json
import random
import re
import zipfile

from codicology import adjudicate as adj
from codicology import review

TITLE = "<head><title>to be or not to be</title></head>"


def _decisions(tmp_path, rows, **meta):
    p = tmp_path / "d.decisions.json"
    p.write_text(json.dumps({**meta, "decisions": rows}))
    return str(p)


def _epub(tmp_path, pages):
    p = tmp_path / "book.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        for n, body in pages.items():
            z.writestr(f"EPUB/page_{n:04d}.xhtml",
                       f"<html>{TITLE}<body>{body}</body></html>")
    return str(p)


def _read(body):
    """A page as the adjudicator reads it."""
    return adj.tokens(html.unescape(re.sub(r"<[^>]+>", " ", body)))


def _fix(body, k, new, page=0):
    """The decision a sheet exports for the k-th word of a page."""
    toks = _read(body)
    return {"page": page, "old": toks[k], "new": new, "source": "human",
            "rung": "human", **adj.site_address(toks, k)}


def _bare(page, occ, old, new):
    """A decision as sheets wrote them before words were addressed."""
    return {"page": page, "occurrence": occ, "old": old, "new": new,
            "source": "human", "rung": "human"}


def _both(vtb, tmp_path, body, rows):
    """The page after the decisions, having checked that the rebuild and
    `codicology apply` did exactly the same thing to it."""
    dec = _decisions(tmp_path, rows)
    bodies = [body]
    st = vtb.apply_reviewer_decisions(bodies, dec)
    epub = _epub(tmp_path, {0: body})
    r = review.apply_decisions(epub, dec)
    page = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert page == f"<html>{TITLE}<body>{bodies[0]}</body></html>", \
        "the two consumers must be equivalent, and the title is not text"
    assert st == {"applied": len(r["applied"]), "stale": len(r["stale"])}
    return bodies[0], st


# ── one tokeniser on both sides ──────────────────────────────────────────

_PIECES = ["be", "be", "The", "the", "café", "naïve", "Al-Khwarazmi",
           "don't", "don’t", "ﬁnal", "co-", "gs", "5", "μ", "x",
           " ", " ", " ", " ", "\n", "-\n", "- \n ", ".", ",", "'", "’", "“",
           "–", "&amp;", "&eacute;", "&#8217;", "&#xFB01;", "&nbsp;",
           "&fjlig;", "&notit;", "&amp", "&#12", "&", ";", "#",
           "<p>", "</p>", "<i>", "</i>", "<br/>", "<", ">"]


def _pages(n=600, seed=20261110):
    rng = random.Random(seed)
    for _ in range(n):
        yield "".join(rng.choice(_PIECES) for _ in range(rng.randint(1, 40)))


def test_the_span_tokeniser_reads_what_the_adjudicator_reads():
    """page_tokens exists so the applier counts words exactly as the report
    did. Any difference between the two readings is the old bug again."""
    for body in _pages():
        want = html.unescape(re.sub(r"<[^>]+>", " ", body))
        text, starts, ends, markup = adj.text_view(body)
        assert text == want, body
        assert len(starts) == len(ends) == len(markup) == len(text)
        assert adj._fold_map(text)[0] == adj.fold_text(text), body
        got = adj.page_tokens(body)
        assert [t for t, _, _ in got] == adj.tokens(want), body
        for tok, lo, hi in got:
            if lo is not None:
                assert _read(body[lo:hi]) == [tok], (body, tok)


def test_rewriting_a_word_changes_that_word_alone():
    seen = 0
    for body in _pages():
        toks = adj.page_tokens(body)
        for k, (tok, lo, hi) in enumerate(toks):
            if lo is None:
                continue
            seen += 1
            after = _read(body[:lo] + "Zq" + body[hi:])
            assert after == [t for t, _, _ in toks[:k]] + ["Zq"] + \
                [t for t, _, _ in toks[k + 1:]], (body, tok)
    assert seen > 1000


def test_a_word_split_by_markup_counts_but_cannot_be_edited():
    body = "<p>the pre-</p>\n<p>fix and &notit; the caf&eacute;</p>"
    toks = adj.page_tokens(body)
    assert [t for t, _, _ in toks] == ["the", "prefix", "and", "it", "the",
                                       "café"]
    assert toks[1][1:] == (None, None), "a word joined across two blocks"
    assert toks[3][1:] == (None, None), "half of one entity's expansion"
    assert body[toks[5][1]:toks[5][2]] == "caf&eacute;"


def test_the_address_counts_every_instance_on_the_page():
    toks = adj.tokens("to be or not to be, he said; let it be so")
    assert adj.site_address(toks, 10) == {
        "occurrence": 2, "before": ["he", "said", "let", "it"],
        "after": ["so"]}
    assert adj.site_address(toks, 0) == {
        "occurrence": 0, "before": [], "after": ["be", "or", "not", "to"]}


def test_the_aligner_says_which_of_our_words_is_disputed():
    ours = adj.tokens("to be or not to be, he said; let it be so")
    theirs = adj.tokens("lo be or not to be, he said; let it he so")
    assert adj.align_disputes(ours, theirs) == [("to", "lo", 0, 0),
                                                ("be", "he", 10, 10)]


# ── the whole loop ───────────────────────────────────────────────────────

def _tsv(words):
    rows = ["level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext"]
    rows += [f"5\t1\t1\t1\t1\t{k + 1}\t{10 + 30 * k}\t20\t25\t12\t95\t{w}"
             for k, w in enumerate(words)]
    return "\n".join(rows)


def _exported(sheet, typed):
    """What the sheet's export button writes, read off the data attributes
    its script reads: `typed` maps a row's shipped word to the reviewer's
    own reading."""
    out = []
    for m in re.finditer(r'<div class="row"[^>]*>', sheet):
        row = {k: html.unescape(v) for k, v in
               re.findall(r'data-([a-z]+)="([^"]*)"', m.group())}
        if row["old"] not in typed:
            continue
        d = {"page": int(row["page"]), "occurrence": int(row["occ"]),
             "old": row["old"], "new": typed[row["old"]],
             "source": "human", "rung": "human"}
        if "before" in row:
            d["before"] = row["before"].split()
            d["after"] = row["after"].split()
        out.append(d)
    return out


def test_the_whole_loop_corrects_the_disputed_word_not_the_first(
        vtb, tmp_path, monkeypatch):
    """adjudicate → report → sheet → decisions → both consumers. The
    witness disagrees about the THIRD "be" only; before the fix the
    decision said "occurrence 0" and the first one was rewritten."""
    from PIL import Image
    body = "<p>to be or not to be, he said; let it be so</p>"
    epub = _epub(tmp_path, {0: body})
    pdf = str(tmp_path / "book.pdf")
    Image.new("RGB", (600, 200), "white").save(pdf, "PDF")
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "read_tesseract_tsv", lambda png: _tsv(
        "to be or not to be, he said; let it he so".split()))
    report_path = str(tmp_path / "book-disputes.json")
    assert adj.main(epub, pdf, report=report_path) == 0
    report = json.load(open(report_path))
    (row,) = report["disputes"]
    assert (row["surya"], row["tesseract"]) == ("be", "he")
    assert row["occurrence"] == 2
    assert row["before"] == ["he", "said", "let", "it"]
    assert row["after"] == ["so"]

    report["name"] = "book"
    rows = _exported(review.render_sheet(report, {}), {"be": "he"})
    assert rows == [{"page": 0, "occurrence": 2, "old": "be", "new": "he",
                     "source": "human", "rung": "human",
                     "before": ["he", "said", "let", "it"], "after": ["so"]}]
    got, st = _both(vtb, tmp_path, body, rows)
    assert got == "<p>to be or not to be, he said; let it he so</p>"
    assert st == {"applied": 1, "stale": 0}


def test_the_sheet_script_exports_the_address():
    js = review._JS
    assert "if ('before' in row.dataset)" in js
    assert "d.before = row.dataset.before.split(' ').filter(Boolean)" in js
    assert "d.after = row.dataset.after.split(' ').filter(Boolean)" in js


def test_an_old_report_numbers_by_dispute_and_exports_no_neighbours():
    """A report written before disputes were addressed still renders — its
    rows go out bare, and the appliers know what a bare row is worth."""
    old = [{"page": 0, "surya": "be", "tesseract": "he", "rung": "abstain",
            "winner": None, "shipped": "be"}]
    assert review.assign_occurrences(old) == [0]
    sheet = review.render_sheet({"pdf": "", "epub": "b.epub", "name": "b",
                                 "disputes": old}, {})
    assert "data-before" not in sheet
    assert _exported(sheet, {"be": "he"}) == [_bare(0, 0, "be", "he")]


# ── placing an addressed decision ────────────────────────────────────────

def test_neighbouring_corrections_do_not_disturb_each_other(vtb, tmp_path):
    """Each row's neighbours were recorded on the shipped text — which the
    row beside it is about to change. All are placed before any is
    written."""
    body = "<p>it was the bcst of timcs, it was the worst of times</p>"
    rows = [_fix(body, 3, "best"), _fix(body, 5, "times"),
            _fix(body, 6, "It")]
    got, st = _both(vtb, tmp_path, body, rows)
    assert got == "<p>it was the best of times, It was the worst of times</p>"
    assert st == {"applied": 3, "stale": 0}


def test_identical_phrases_are_told_apart_by_the_ordinal(vtb, tmp_path):
    body = "<p>" + " ".join(["ibid., p. 4;"] * 6) + "</p>"
    got, st = _both(vtb, tmp_path, body, [_fix(body, 6, "Ibid")])
    assert st == {"applied": 1, "stale": 0}
    assert got == ("<p>" + " ".join(["ibid., p. 4;"] * 3 + ["Ibid., p. 4;"]
                                    + ["ibid., p. 4;"] * 2) + "</p>")


def test_changed_neighbours_make_a_decision_stale(vtb, tmp_path):
    row = _fix("<p>we may be here or we may be gone</p>", 6, "he")
    body = "<p>a different page that happens to say be once</p>"
    got, st = _both(vtb, tmp_path, body, [row])
    assert (got, st) == (body, {"applied": 0, "stale": 1})


def test_a_correction_survives_a_deletion_earlier_on_its_page(vtb, tmp_path):
    """The deletion takes a "be" with it, so the ordinal now names the
    wrong word. The neighbours still name the right one."""
    body = ("<p>STRAY HEAD to be</p><p>to be or not to be, he said; "
            "let it be so</p>")
    rows = [_fix(body, 14, "he"),
            {"page": 0, "occurrence": 0, "old": "STRAY HEAD to be",
             "new": "", "source": "human", "rung": "surya-only",
             "kind": "delete_run"}]
    assert rows[0]["occurrence"] == 3
    got, st = _both(vtb, tmp_path, body, rows)
    assert got == "<p>to be or not to be, he said; let it he so</p>"
    assert st == {"applied": 2, "stale": 0}


def test_after_a_deletion_look_alike_places_are_not_guessed_between(
        vtb, tmp_path):
    body = ("<p>STRAY ibid</p><p>" + " ".join(["ibid., p. 4;"] * 6) + "</p>")
    rows = [_fix(body, 8, "Ibid"),
            {"page": 0, "occurrence": 0, "old": "STRAY ibid", "new": "",
             "source": "human", "rung": "surya-only", "kind": "delete_run"}]
    got, st = _both(vtb, tmp_path, body, rows)
    assert st == {"applied": 1, "stale": 1}
    assert "Ibid" not in got and "STRAY" not in got


def test_a_word_split_by_markup_is_stale_not_half_rewritten(vtb, tmp_path):
    body = "<p>he was pre-</p>\n<p>fixed there</p>"
    assert _read(body)[2] == "prefixed"
    got, st = _both(vtb, tmp_path, body, [_fix(body, 2, "prefaced")])
    assert (got, st) == (body, {"applied": 0, "stale": 1})


def test_a_word_in_other_typography_is_found_and_replaced_whole(
        vtb, tmp_path):
    body = "<p>the ﬁnal caf&eacute; of hyphen-\nated don’t</p>"
    rows = [_fix(body, 1, "first"), _fix(body, 2, "cafés & bars"),
            _fix(body, 4, "hyphenated"), _fix(body, 5, "won’t")]
    assert [r["old"] for r in rows] == ["final", "café", "hyphenated",
                                        "don't"]
    got, st = _both(vtb, tmp_path, body, rows)
    assert got == "<p>the first cafés &amp; bars of hyphenated won’t</p>"
    assert st == {"applied": 4, "stale": 0}


def test_two_decisions_for_one_word_apply_once(vtb, tmp_path):
    body = "<p>to be or not to be</p>"
    got, st = _both(vtb, tmp_path, body, [_fix(body, 5, "he"),
                                          _fix(body, 5, "by")])
    assert got == "<p>to be or not to he</p>"
    assert st == {"applied": 1, "stale": 1}


def test_applying_twice_changes_nothing_more(vtb, tmp_path):
    """`codicology apply` works in place, and a reviewer who goes back to
    the sheet exports the earlier decisions again with the new ones."""
    body = ("<p>to be or not to be, he said; let it be so. Take 5 gs of "
            "it; ibid., p. 4; ibid., p. 4; ibid., p. 4; ibid., p. 4.</p>")
    toks = _read(body)
    rows = [_fix(body, 10, "he"),                     # the third "be"
            _fix(body, toks.index("gs"), "μgs"),      # adds what no token holds
            _fix(body, toks.index("ibid") + 2, "Ibid")]
    epub = _epub(tmp_path, {0: body})
    dec = _decisions(tmp_path, rows)
    first = review.apply_decisions(epub, dec)
    assert len(first["applied"]) == 3
    once = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert "let it he so" in once and "5 μgs of" in once
    assert once.count("Ibid") == 1
    again = review.apply_decisions(epub, dec)
    assert not again["applied"] and len(again["stale"]) == 3
    assert zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode() == once


# ── a decision from before words were addressed ──────────────────────────

def test_a_bare_decision_lands_only_where_it_cannot_be_wrong(
        vtb, tmp_path, capsys):
    """Its ordinal counted disputes; the page counts instances. They are
    sure to agree only when the file names every instance on the page."""
    body = "<p>to be or not to be, he said; let it be so. A wurd.</p>"
    # the dispute was the third "be", and the old sheet called it 0
    got, st = _both(vtb, tmp_path, body, [_bare(0, 0, "be", "he"),
                                          _bare(0, 0, "wurd", "word")])
    assert got == "<p>to be or not to be, he said; let it be so. A word.</p>"
    assert st == {"applied": 1, "stale": 1}
    out = capsys.readouterr().out
    assert "the page prints 'be' 3 times" in out
    assert "adjudicate again" in out

    # all three disputed, the last one corrected: 2 can only be the third
    got, st = _both(vtb, tmp_path, body, [_bare(0, 2, "be", "he")])
    assert got == "<p>to be or not to be, he said; let it he so. A wurd.</p>"
    assert st == {"applied": 1, "stale": 0}


def test_a_bare_decision_does_not_double_what_the_page_already_reads(
        vtb, tmp_path):
    body = "<p>a dose of 5 μgs daily</p>"
    assert _read(body)[3] == "gs", "the tokeniser never sees the μ"
    got, st = _both(vtb, tmp_path, body, [_bare(0, 0, "gs", "μgs")])
    assert (got, st) == (body, {"applied": 0, "stale": 1})


def test_a_piece_of_a_longer_word_is_never_rewritten(vtb, tmp_path):
    """The tokeniser stops at a letter outside its alphabet, so the page's
    "diabolē" is read as "diabol" — and a correction to that token would
    land inside the word, leaving "diaboléē". The page is left as read,
    whichever way the decision was written, and whether the accent is its
    own character or not."""
    for tail, read in (("ē", "diabol"), ("e\u0304", "diabole")):
        body = f"<p>the Greek diabol{tail}, which is slander</p>"
        assert _read(body)[2] == read
        for row in (_fix(body, 2, "diabolé"),
                    _bare(0, 0, read, "diabolé")):
            got, st = _both(vtb, tmp_path, body, [row])
            assert (got, st) == (body, {"applied": 0, "stale": 1})


def test_a_note_number_against_a_word_does_not_make_it_a_piece(
        vtb, tmp_path):
    """A digit is not the rest of a word: "Lloyds3" is a word and its
    note's number, and the word is still the reviewer's to correct."""
    body = "<p>the rooms of Lloyds3 in Tower Street</p>"
    assert _read(body)[3] == "Lloyds"
    got, st = _both(vtb, tmp_path, body, [_fix(body, 3, "Lloyd's")])
    assert got == "<p>the rooms of Lloyd's3 in Tower Street</p>"
    assert st == {"applied": 1, "stale": 0}


# ── a correction typed for more than the tokeniser's word ────────────────

def test_what_the_page_prints_against_a_word_is_not_written_twice(
        vtb, tmp_path):
    """The reviewer types the printed word. The tokeniser's word can be
    less — it drops digits and signs — and what the correction shares
    with the characters standing hard against the token is already on
    the page."""
    for body, k, new, want in (
            ("<p>MORTAR POSITION (8IMM AND 4.2-IN MORTARS)</p>", 2, "81MM",
             "<p>MORTAR POSITION (81MM AND 4.2-IN MORTARS)</p>"),
            ("<p>about 5O% of them</p>", 1, "50%",
             "<p>about 50% of them</p>"),
            ("<p>she said, “Let it he so,” and left</p>", 4, "be",
             "<p>she said, “Let it be so,” and left</p>"),
            ("<p>the cost (£5) was tbe same</p>", 3, "the",
             "<p>the cost (£5) was the same</p>")):
        toks = _read(body)
        for row in (_fix(body, k, new), _bare(0, 0, toks[k], new)):
            got, st = _both(vtb, tmp_path, body, [row])
            assert (got, st) == (want, {"applied": 1, "stale": 0}), row


def test_a_correction_that_repeats_its_surroundings_is_not_written(
        vtb, tmp_path, capsys):
    """Three installed books had a whole expression printed inside
    itself: the reviewer typed out a formula, a rotation, a latitude for
    the one piece of it the sheet showed, and the piece was replaced with
    the lot. Nothing sure can be made of such a row; the page is left as
    read and the row is reported."""
    for body, k, new, why in (
            ("<p>Anal: C<sub>15</sub>H<sub>23</sub>ClN<sub>2</sub>O. "
             "C,H,N.</p>", 3, "C15H23CIN2O", "repeats the words beside it"),
            ("<p>and an [α]_D + 41.8 (c 0.44, EtOH)</p>", 2, "[α]D",
             "already prints '[α]' against it"),
            ("<p>between 41^{circ} 0’ and 42^{circ} 2’ north</p>", 1, "0'",
             "already prints \"0'\" beside it")):
        toks = _read(body)
        rows = [_fix(body, k, new)]
        if toks.count(toks[k]) == 1:
            rows.append(_bare(0, 0, toks[k], new))
        for row in rows:
            got, st = _both(vtb, tmp_path, body, [row])
            assert (got, st) == (body, {"applied": 0, "stale": 1}), row
            assert why in capsys.readouterr().out


def test_a_word_typed_empty_is_still_removed(vtb, tmp_path):
    body = "<p>the the cat (sat) down</p>"
    got, st = _both(vtb, tmp_path, body, [_fix(body, 1, "")])
    assert got == "<p>the  cat (sat) down</p>"
    assert st == {"applied": 1, "stale": 0}


def test_a_hand_written_phrase_is_still_found(vtb, tmp_path):
    body = "<p>pens, ink, &amp;c. were sold; the old cat sat</p>"
    got, st = _both(vtb, tmp_path, body, [_bare(0, 0, "&c.", "etc."),
                                          _bare(0, 0, "old cat", "old dog")])
    assert got == "<p>pens, ink, etc. were sold; the old dog sat</p>"
    assert st == {"applied": 2, "stale": 0}


# ── every decision is accounted for ──────────────────────────────────────

def test_apply_counts_the_decisions_it_could_not_even_place(tmp_path, capsys):
    epub = _epub(tmp_path, {0: "<p>a wurd here</p>"})
    dec = _decisions(tmp_path, [_bare("seven", 0, "wurd", "word"),
                                _bare(9, 0, "wurd", "word"),
                                _bare(0, 0, "wurd", "word")])
    r = review.apply_decisions(epub, dec)
    assert len(r["applied"]) == 1 and len(r["stale"]) == 2
    out = capsys.readouterr().out
    assert "is not a number" in out and "not a page of this book" in out
    assert "applied 1 correction(s), 2 stale" in out


def test_apply_never_edits_the_title(tmp_path):
    """The <title> reads "to be or not to be" and is not the reviewer's
    text: the page's one "be" is the body's."""
    epub = _epub(tmp_path, {0: "<p>let it be so</p>"})
    r = review.apply_decisions(
        epub, _decisions(tmp_path, [_fix("<p>let it be so</p>", 2, "he")]))
    assert len(r["applied"]) == 1
    page = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert page == f"<html>{TITLE}<body><p>let it he so</p></body></html>"


def test_a_change_of_type_is_not_a_space_between_what_is_printed(
        vtb, tmp_path):
    # "[α]<sub>D</sub>" is one printed expression: the readers see "[α] D",
    # and a correction typed out whole for the D would print "[α]" twice
    body = "<p>an [α]<sub>D</sub> + 41.8° (c 0.44, EtOH)</p>"
    row = {"page": 0, "occurrence": 0, "old": "D", "new": "[α]D"}
    got, st = _both(vtb, tmp_path, body, [row])
    assert (got, st) == (body, {"applied": 0, "stale": 1})
    _, (why,) = review.apply_page_decisions(body, [row])
    assert why == "the page already reads '[α]D' there"
    # and through it the shared part is still trimmed, not refused
    body = "<p>a <b>£</b>5O note</p>"
    got, st = _both(vtb, tmp_path, body,
                    [{"page": 0, "occurrence": 0, "old": "O", "new": "£50"}])
    assert (got, st) == ("<p>a <b>£</b>50 note</p>",
                         {"applied": 1, "stale": 0})


def test_the_hard_view_lines_up_with_the_folded_text():
    from codicology import adjudicate as adj
    body = ("<p>the ﬁnal caf&eacute; of hyphen-\nated <i>don’t</i>s, "
            "H<sub>2</sub>O</p>\n<p>next</p>")
    folded, words = adj.page_words(body)
    hard = adj.hard_view(body)
    assert len(hard) == len(folded)
    assert [a == b or (a, b) == (" ", adj.STRUCK)
            for a, b in zip(folded, hard)] == [True] * len(folded)
    assert hard.count(adj.STRUCK) == 4          # <i> </i> <sub> </sub>
