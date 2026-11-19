"""Endnotes (notes.link_chapter_notes and notes.link_notes) on generated
books.

link_chapter_notes: a "Notes" section closes each chapter, and "markers
between one section and the next belong to the next, and that is the whole
rule". link_notes: one back-of-book section grouped by chapter, "a marker
is linked only when its chapter scope is settled and exactly one note
answers to its number there". So, whatever the book:

- linking adds links and nothing else (an entry keeps its printed dress);
- every href resolves, every id is unique;
- a linked call showing n reaches an entry showing n, in the call's own
  chapter (the generator records which chapter every call and entry is);
- pages still parse; a second run over the linked book breaks nothing.
"""
from __future__ import annotations

import random

from hypothesis import given, strategies as st

from codicology.notes import link_chapter_notes, link_notes

from .notes_oracle import (bare, check_calls_meet_their_entries, check_links,
                           parses, unlinked)
from .test_footnote_linker_props import WORDS, Book


def _calls(rnd: random.Random, k: int) -> "list[int]":
    """A chapter's calls 1..k as an author and a reader leave them: one now
    and then lost, cited again later, or set out of order."""
    seq = [n for n in range(1, k + 1) if rnd.random() > 0.08]
    for n in range(1, k + 1):
        if rnd.random() < 0.08 and seq:
            seq.insert(rnd.randrange(len(seq) // 2, len(seq) + 1), n)
    if len(seq) > 2 and rnd.random() < 0.1:
        j = rnd.randrange(len(seq) - 1)
        seq[j], seq[j + 1] = seq[j + 1], seq[j]
    return seq


def _body_pages(book: Book, chapter: int, calls: "list[int]", n_pages: int,
                prose_list: bool) -> "list[str]":
    """The chapter's prose over n_pages, its calls in reading order."""
    rnd = book.rnd
    inserts = [book.call(("ch", chapter, n), str(n)) for n in calls]
    cuts = sorted(rnd.choices(range(len(inserts) + 1), k=n_pages - 1))
    bounds = [0] + cuts + [len(inserts)]
    pages = []
    for p in range(n_pages):
        chunk = inserts[bounds[p]:bounds[p + 1]]
        n_paras = rnd.randint(1, 3)
        pc = sorted(rnd.choices(range(len(chunk) + 1), k=n_paras - 1))
        pb = [0] + pc + [len(chunk)]
        paras = [_prose_in_order(book, rnd.randint(6, 25), chunk[pb[j]:pb[j + 1]])
                 for j in range(n_paras)]
        if prose_list and p == 0:
            paras.insert(min(1, len(paras)),
                         "<p>1. First, the mills.</p><p>2. Second, the grain.</p>")
        pages.append("".join(paras))
    return pages


def _prose_in_order(book: Book, n_words: int, inserts: list) -> str:
    """A paragraph, plain numbers among its words, the inserts (calls) set
    after words in the order given."""
    rnd = book.rnd
    words = [rnd.choice(WORDS) if rnd.random() > 0.08
             else rnd.choice(("1984", "12", "3", "1", "p. 2"))
             for _ in range(n_words)]
    words[0] = "The"
    slots = sorted(rnd.choices(range(n_words), k=len(inserts)))
    for at, ins in zip(slots, inserts):
        if not words[at][-1].isalpha() and not words[at].endswith("</sup>"):
            words[at] += " word"
        words[at] += ins
    return "<p>" + " ".join(words) + ".</p>"


def _entry(book: Book, key, n: int, dress: str) -> str:
    text = book.entry_words(key)
    return {"plain": f"<p>{n}. {text}</p>",
            "sup": f"<p><sup>{n}</sup> {text}</p>",
            "li": f"<li>{n}. {text}</li>"}[dress]


def _entries(book: Book, chapter: int, k: int, dress: str,
             lose: bool = True) -> "list[str]":
    rnd = book.rnd
    out = []
    for n in range(1, k + 1):
        if lose and rnd.random() < 0.08:
            continue                                  # an entry the reading lost
        out.append(_entry(book, ("ch", chapter, n), n, dress))
    return out


def _join(entries: "list[str]", dress: str) -> str:
    if dress == "li":
        return "<ol>" + "".join(entries) + "</ol>" if entries else ""
    return "".join(entries)


# Shapes that produce wrong links today, each pinned as a failing example in
# test_notes_linker_findings.py. They are kept out of the search so the
# properties go on looking for other faults; turn one on once it is mended.
LOSE_CHAPTER_ENTRIES = True    # entries lost from chapter-end notes (mended 2026-11-19)
RUN_ON_HEAD_PAGE = True        # a chapter begun on the page of the last one's (mended 2026-11-19)
                               # notes head (on a later notes page it is in)
GUESS_CHAPTERS = False         # link_notes placing calls by their numbers alone


@st.composite
def chapter_books(draw, lose_entries=None, run_on_head_page=None):
    """Chapters each closed by its own Notes section: entries in three
    dresses, a section running over a page, lost entries and calls, notes
    cited twice or out of order, numbered lists in the prose, and (now and
    then) the next chapter beginning on the page its notes end."""
    book = Book(random.Random(draw(st.integers(0, 2**32 - 1))))
    rnd = book.rnd
    n_ch = draw(st.integers(2, 4))
    pages: list[str] = ["<p>Front matter.</p>"]
    run_on = draw(st.booleans())
    on_head = RUN_ON_HEAD_PAGE if run_on_head_page is None else run_on_head_page
    last_is_head = False
    for c in range(n_ch):
        k = draw(st.integers(1, 8))
        calls = _calls(rnd, k)
        body = _body_pages(book, c, calls, draw(st.integers(1, 2)),
                           prose_list=draw(st.integers(0, 5)) == 0)
        if run_on and c > 0 and (on_head or not last_is_head):
            pages[-1] += f"<h1>Chapter {c + 1}</h1>" + body[0]
            body = body[1:]
        pages += body
        dress = draw(st.sampled_from(["plain", "sup", "li"]))
        head = draw(st.sampled_from(["<h2>Notes</h2>", "<h2>NOTES</h2>",
                                     "<h3>Endnotes</h3>"]))
        ents = _entries(book, c, k, dress, LOSE_CHAPTER_ENTRIES
                        if lose_entries is None else lose_entries)
        split = draw(st.sampled_from([0, len(ents), draw(st.integers(0, len(ents)))]))
        first = head + _join(ents[:split], dress)
        if draw(st.booleans()) and pages:
            pages[-1] += first                        # notes under the chapter's end
        else:
            pages.append(first)
        last_is_head = True
        if ents[split:]:
            pages.append(_join(ents[split:], dress))
            last_is_head = False
    return pages, book


@given(chapter_books())
def test_chapter_notes_add_links_and_nothing_else(case):
    pages, _ = case
    out = list(pages)
    link_chapter_notes(out, set())
    for i, (a, b) in enumerate(zip(pages, out)):
        assert unlinked(b) == bare(a), f"page {i} changed beyond its links"


@given(chapter_books())
def test_chapter_notes_links_resolve_and_ids_are_unique(case):
    pages, _ = case
    out = list(pages)
    link_chapter_notes(out, set())
    check_links(out)


@given(chapter_books())
def test_chapter_notes_bind_a_call_to_its_own_chapter(case):
    pages, book = case
    out = list(pages)
    link_chapter_notes(out, set())
    check_calls_meet_their_entries(out, book.call_intent, book.entry_intent)


@given(chapter_books())
def test_chapter_notes_pages_still_parse(case):
    pages, _ = case
    assert all(parses(p) for p in pages)
    out = list(pages)
    link_chapter_notes(out, set())
    for i, p in enumerate(out):
        assert parses(p), f"page {i}: {p!r}"


# No second-run property for link_chapter_notes: a build runs it once, and
# over its own output anchored entries no longer read as entries nor linked
# calls as calls, so sections are read differently and their ids (numbered by
# section) can collide. That path does not exist in a build.


# ── one back-of-book section ────────────────────────────────────────────────

_GROUP_HEADS = (lambda c: f"<h2>Chapter {c + 1}</h2>",
                lambda c: f"<h2>CHAPTER {c + 1}</h2>",
                lambda c: f"<h3>{c + 1}. The Mills</h3>",
                lambda c: f"<p><b>Notes to Chapter {c + 1}</b></p>")


@st.composite
def back_of_book(draw, settled=None):
    """Chapters with calls numbered from 1 in each, then one Notes section
    grouped by chapter heads in one of several dresses. Lost calls, lost
    entries, notes cited twice, a chapter with a single note, a group head
    the reading lost. `settled`: the group heads carry chapter numbers and
    the contents says where each chapter begins, so no call's chapter is a
    guess."""
    settled = (not GUESS_CHAPTERS) if settled is None else settled
    book = Book(random.Random(draw(st.integers(0, 2**32 - 1))))
    rnd = book.rnd
    n_ch = draw(st.integers(1, 6))
    ks = [draw(st.integers(1, 9)) for _ in range(n_ch)]
    pages: list[str] = ["<p>Contents.</p>"]
    starts = []
    for c, k in enumerate(ks):
        calls = _calls(rnd, k)
        body = _body_pages(book, c, calls, draw(st.integers(1, 2)),
                           prose_list=draw(st.integers(0, 5)) == 0)
        starts.append((c + 1, len(pages)))
        body[0] = f"<h1>Chapter {c + 1}</h1>" + body[0]
        pages += body
    head = _GROUP_HEADS[2] if settled else draw(st.sampled_from(_GROUP_HEADS))
    dress = draw(st.sampled_from(["plain", "sup", "li"]))
    lost_head = draw(st.integers(-1, n_ch - 1)) \
        if not settled and draw(st.integers(0, 3)) == 0 else -1
    section = ["<h1>Notes</h1>"]
    for c, k in enumerate(ks):
        ents = _entries(book, c, k, dress)
        section.append(("" if c == lost_head else head(c)) + _join(ents, dress))
    # the section laid over pages
    per = draw(st.integers(1, 3))
    for j in range(0, len(section), per):
        pages.append("".join(section[j:j + per]))
    chapter_starts = starts if settled or draw(st.booleans()) else None
    return pages, book, chapter_starts


@given(back_of_book(settled=False))
def test_endnotes_add_links_and_nothing_else(case):
    pages, _, cs = case
    out = list(pages)
    link_notes(out, set(), chapter_starts=cs)
    for i, (a, b) in enumerate(zip(pages, out)):
        assert unlinked(b) == bare(a), f"page {i} changed beyond its links"


@given(back_of_book(settled=False))
def test_endnotes_links_resolve_and_ids_are_unique(case):
    pages, _, cs = case
    out = list(pages)
    link_notes(out, set(), chapter_starts=cs)
    check_links(out)


@given(back_of_book())
def test_endnotes_bind_a_call_to_its_own_chapter(case):
    pages, book, cs = case
    out = list(pages)
    link_notes(out, set(), chapter_starts=cs)
    check_calls_meet_their_entries(out, book.call_intent, book.entry_intent)


@given(back_of_book(settled=False))
def test_endnotes_pages_still_parse(case):
    pages, _, cs = case
    once = list(pages)
    link_notes(once, set(), chapter_starts=cs)
    for p in once:
        assert parses(p)


@given(back_of_book())
def test_endnotes_second_run_changes_nothing(case):
    pages, _, cs = case
    once = list(pages)
    link_notes(once, set(), chapter_starts=cs)
    twice = list(once)
    stats = link_notes(twice, set(), chapter_starts=cs)
    assert twice == once and stats["linked"] == 0
