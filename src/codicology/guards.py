"""The guards between what the recogniser said and the book: a reading that
loops, invents, counts where the page prints a table, contradicts the page's
own text layer, or has no witness to vouch for it; and a page dropped as
blank that only failed to be read.
"""
from __future__ import annotations

import cv2
import difflib
import functools
import html
import os
import re
from PIL import Image
from collections import Counter
from . import switches, witness
from .markup import _strip_tags
from .model import PageItem


# A generative recogniser has a failure mode a classical one cannot have: given
# a page it cannot read, it invents. Measured here on 47 reads, the fabrications
# announce themselves two ways — they loop (every six-word window repeating), or
# they appear on a page that has almost no ink to have produced them. Neither
# alone is enough. One fabrication repeated three reads verbatim, so re-reading
# cannot catch it; every genuinely sparse page (a part title, a chapter's last
# two lines) reads back identically, so ink cannot condemn one on its own.
# Looping is the one signal that survived testing. Two others were tried and
# withdrawn, each after condemning real pages: "text on a near-blank page"
# threw away two books' title pages, and "characters per unit of ink" flagged
# 276 of Shadow Libraries' 321 pages — ink varies more across typefaces and
# scan styles than fabrication varies from truth, so no fixed threshold on it
# transfers between books. Every damaging fabrication observed loops; the
# short ones that do not are indistinguishable from a real dedication page
# and are accepted as a known blind spot rather than guessed at.
DEGENERATE_REP6 = 0.4        # legitimate pages measured at most 0.14


def _repeated_six_gram_share(text: str) -> float:
    """How much of the text is six-word runs that occur more than once."""
    w = re.findall(r"[a-z0-9]{2,}", text.lower())
    if len(w) < 40:
        return 0.0
    six = [" ".join(w[i:i + 6]) for i in range(len(w) - 5)]
    counts: dict[str, int] = {}
    for g in six:
        counts[g] = counts.get(g, 0) + 1
    return sum(v for v in counts.values() if v > 1) / max(1, len(six))


def _page_ink(image) -> float:
    """Fraction of the page clearly darker than its paper, border excluded.

    The border matters: a photographed page carries the desk and the book's
    own shadow, and counting those put a blank leaf at 0.11 when the paper
    itself was at 0.006 — nearly the reading of a printed page.
    """
    import numpy as np
    a = np.asarray(image.convert("L"), dtype=np.float32)
    h, w = a.shape
    a = a[int(h * .06):int(h * .94), int(w * .06):int(w * .94)]
    if a.size == 0:
        return 1.0
    return float((a < np.percentile(a, 90) - 40).mean())


# …but paper repeats too. A field manual prints the same chlorine test twice
# on facing pages (ST 31-91B, 20-4 and 20-5: 0.40, shipped blank), and
# OpenStax exercise sets restate their question stems (up to 0.94, with the
# publisher's own layer at 0.83). Legitimate pages reach far past 0.14; the
# 47 reads behind that figure held none of these. What a loop does that
# paper does not is repeat MORE than the page: so a looping reading is let
# through when a witness that cannot loop — the page's own text layer where
# it can testify, the classical reader where it cannot — holds the same
# repeats, and the reading is as long as the witness's.
#
# Excess is the share of the reading's six-word runs that are copies beyond
# the witness's own count of that run. The length band catches what excess
# cannot: a loop that cut the page short and hid its few copies among
# repeats the page really prints. Measured on 3,270 accepted readings of
# scanned pages against their layers, with loops made the way surya makes
# them — a true start, then the latest stretch repeated — including the
# adversarial kind that stops the moment the guard fires: of 9,810 such
# loops, none passes both tests (excess alone lets through 6 at 0.15; every
# one of them was a truncated page). Loops run to a token budget measured
# excess 0.395 and up. Genuine readings sit at excess 0.023 (p99) and 0.94
# to 1.06 of their layer's length (p1-p99). The field manual's spread,
# against the classical reader: excess 0.099, length 1.00.
LOOP_MAX_EXCESS = 0.15


LOOP_LENGTH_BAND = (0.85, 1.2)


def _six_word_runs(text: str) -> "Counter[str]":
    w = re.findall(r"[a-z0-9]{2,}", text.lower())
    return Counter(" ".join(w[i:i + 6]) for i in range(len(w) - 5))


def _repeats_are_on_the_page(text: str, witness: str) -> bool:
    """Whether a witness that cannot loop repeats what this reading repeats."""
    ours, theirs = _six_word_runs(text), _six_word_runs(witness)
    total = sum(ours.values())
    if not total:
        return False
    excess = sum(max(0, c - max(theirs.get(g, 0), 1))
                 for g, c in ours.items() if c > 1) / total
    words = lambda s: len(re.findall(r"[a-z]{3,}", s.lower()))
    ratio = words(text) / max(1, words(witness))
    lo, hi = LOOP_LENGTH_BAND
    return excess < LOOP_MAX_EXCESS and lo <= ratio <= hi


def _page_witness(page_path: str) -> "str | None":
    """The page's own account of its words: its text layer where that can
    testify, else what the classical reader makes of it; None if neither."""
    if _layer_can_testify(page_path):
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            return fh.read()
    return witness._classical_text(page_path)


def _looks_fabricated(items, image, page_path: "str | None" = None
                      ) -> str | None:
    """Why this reading is suspect, or None if it looks like a real one."""
    text = " ".join(_strip_tags(it.html) for it in items
                    if it.html and not it.is_furniture).strip()
    if not text:
        return None                      # an empty read is the other guard's
    if _repeated_six_gram_share(text) >= DEGENERATE_REP6:
        witness = _page_witness(page_path) if page_path else None
        if witness and _repeats_are_on_the_page(text, witness):
            return None
        return "text loops on itself"
    return None


# A loop is not the only thing a generative reader makes of a page it cannot
# read. It also counts. Three tables on the shelf are a count and nothing
# else: page 49 of The History of Lloyd's (2,029 cells — 674 integers, every
# one the same as the last or one more, and 8 cells of anything else), the
# "Epilogue" part title of TiHKAL (539 integers, each beside the same three
# words), and an eighteen-by-eighteen grid on a facsimile plate in Studies
# in the History of Accounting (1 to 18, twice, in 361 cells). The loop
# guard cannot see them, because a counter makes every six-word run unique;
# the layer guard cannot, because none of the three pages has forty words of
# layer to compare against. The first is why Kindle refused to reflow the
# book it was in.
#
# The shape is most of the evidence. Of 2,338 tables in 159 caches (41,388
# pages), ten hold thirty or more bare integers with nine steps in ten
# standing still or rising by one. Seven are real — contents pages, numbered
# rows — and say at least as many different things as they have numbers
# (the least: 33 distinct cells beside 36 integers). The three inventions
# say 0, 2 and 5 things beside 36, 539 and 674.
#
# But paper counts too: an answer key, a hundred chart, a numbered blank.
# None is on this shelf to measure, so the shape only accuses, and the
# verdict belongs to a witness that cannot invent. Of the 48 real tables
# that count at all (twelve integers, eight steps in ten), the better of the
# page's two witnesses prints every value of the count for 46, and 0.96 and
# 0.47 of it for the other two — that last one a table with 106 other cells,
# nowhere near the shape. For the inventions: 0.00, 0.00 and 0.28. It has to
# be the better of the two, because either alone goes missing on real pages:
# a scan has no layer worth the name (six real tables under half, four of
# them at 0.00), and tesseract loses half the figures of a dense one (0.56,
# 0.59) that the layer holds whole.
COUNT_MIN_INTEGERS = 30


COUNT_MIN_STEPS = 0.9        # of consecutive integers: the same, or one more


COUNT_MAX_SAID = 0.1         # distinct other cells, per integer


COUNT_MIN_WITNESSED = 0.5    # of the count's values, on the better witness


_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)


_TABLE_CELL = re.compile(r"<t[dh]\b[^>]*?(?:/>|>(.*?)</t[dh]>)", re.S | re.I)


_BARE_INTEGER = re.compile(r"[0-9]{1,5}")


_STANDALONE_INTEGER = re.compile(r"(?<![0-9])[0-9]{1,5}(?![0-9])")


def _count_in_table(table_html: str) -> "list[int] | None":
    """The integers a table counts through, if counting is all it does."""
    cells = [" ".join(_strip_tags(c).split())
             for c in _TABLE_CELL.findall(table_html)]
    numbers = [int(c) for c in cells if _BARE_INTEGER.fullmatch(c)]
    if len(numbers) < COUNT_MIN_INTEGERS:
        return None
    steps = sum(1 for a, b in zip(numbers, numbers[1:]) if 0 <= b - a <= 1)
    if steps < COUNT_MIN_STEPS * (len(numbers) - 1):
        return None
    said = {c for c in cells if c and not _BARE_INTEGER.fullmatch(c)}
    if len(said) > COUNT_MAX_SAID * len(numbers):
        return None
    return numbers


def _count_is_on_the_page(numbers: "list[int]", page_path: str) -> bool:
    """Whether a witness that cannot invent prints the count this table makes.

    Both are asked — the page's text layer, whatever its size (a page of
    figures has no forty words, and does not need them to hold a number),
    then the classical reader — and either is enough.
    """
    count = set(numbers)

    def holds(witness: "str | None") -> bool:
        if not witness:
            return False
        seen = {int(n) for n in _STANDALONE_INTEGER.findall(witness)}
        return len(count & seen) >= COUNT_MIN_WITNESSED * len(count)

    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            if holds(fh.read()):
                return True
    except OSError:
        pass
    return holds(witness._classical_text(page_path))


def drop_counting_tables(items: "list[PageItem]",
                         page_path: "str | None" = None
                         ) -> "tuple[list[PageItem], list[int]]":
    """Items without the tables that only count, and the size of each dropped.

    Only the table goes. The rest of the page was read by the same pass and
    stands or falls by the page-level guards: TiHKAL's part title keeps the
    heading it really prints, and a page left with nothing is an empty page
    to everything downstream, which is what the artwork recovery and the
    blank-page witness exist to look at. Run where the book is assembled
    rather than where the page is read, so it judges a reading from the
    cache exactly as it judges a fresh one — the three that prompted it were
    all in caches. With no page to consult, or no witness that holds the
    count, the table is refused: the loop guard's rule.
    """
    out, dropped = [], []
    for it in items:
        if it.is_furniture or not it.html or "<table" not in it.html.lower():
            out.append(it)
            continue

        def judge(m: "re.Match") -> str:
            numbers = _count_in_table(m.group(0))
            if numbers is None or (
                    page_path and _count_is_on_the_page(numbers, page_path)):
                return m.group(0)
            dropped.append(len(numbers))
            return ""

        kept = _TABLE.sub(judge, it.html)
        if kept == it.html:
            out.append(it)
        elif it.figure is not None or _strip_tags(kept) \
                or re.search(r"<(?:img|svg|math)\b", kept, re.I):
            out.append(it._replace(html=kept))
    return out, dropped


# Below this share of our words appearing in the page's own text layer, the
# reading is talking about something the page is not. Fabricated pages measured
# 0.00-0.04 against their layers; genuine readings of the same pages, 0.5 and
# up even against badly garbled scanner layers, because prose shares its own
# vocabulary however badly either side mangles the edges. The word floor keeps
# the rule off pages too short to have a vocabulary worth comparing.
LAYER_MIN_OVERLAP = 0.2


LAYER_MIN_WORDS = 40


# …except where the layer is the one garbling. A scanner's OCR layer can
# mangle whole words ("INT~O~UCTION -ro UNARM~~ COhIIIAT" for "INTRODUCTION
# TO UNARMED COMBAT"), and a count of shared words cannot tell a mangled
# copy of the page from a different page: Kill or Get Killed's contents
# shared 0.17 of its words with such a layer, read correctly, and shipped
# blank. Letters taken in order can. A reading refused on words is let
# through when this share of its letters lines up, in order and in runs of
# four or more, with the layer's. Measured on 12 scanned books whose
# layers are other people's OCR (3,270 pages): of 5,056 readings of the
# wrong text that the word rule refuses — the next page, a page seven on,
# a page of another book — none align this far, and one aligns past 0.25.
# That contents page aligns 0.48. It is a rescue only: as a rule on its
# own, alignment would condemn real TiHKAL pages that the word rule passes
# (0.18, 0.35), so it never refuses anything.
LAYER_MIN_ALIGNED = 0.3


def _layer_alignment(text: str, layer: str, cap: int = 5000) -> float:
    """Share of the reading's letters that line up, in order, with the layer's."""
    ours = re.sub(r"[^a-z]", "", text.lower())[:cap]
    theirs = re.sub(r"[^a-z]", "", layer.lower())[:cap]
    if not ours:
        return 0.0
    blocks = difflib.SequenceMatcher(None, ours, theirs,
                                     autojunk=False).get_matching_blocks()
    return sum(b.size for b in blocks if b.size >= 4) / len(ours)


def _contradicts_layer(items, page_path: str) -> str | None:
    """Why this reading disagrees with the page's own text layer, or None."""
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            layer = fh.read()
    except OSError:
        return None                        # no layer, no witness — not guilt
    words = re.findall(r"[a-z]{3,}", " ".join(
        _strip_tags(it.html) for it in items if it.html).lower())
    # The floor counts words READ, not distinct ones: a fabricated table says
    # "Project" eighty times in a thousand words — plenty of text, almost no
    # vocabulary — and a floor on distinct words would wave exactly the
    # fabrications through.
    if len(words) < LAYER_MIN_WORDS:
        return None
    ours = set(words)
    theirs = set(re.findall(r"[a-z]{3,}", layer.lower()))
    # A witness must know enough to testify. One TiHKAL page carries a real
    # 261-word aside about the size of a trillion over a layer that says only
    # "how many zeros" — the scanner's own OCR failed there — and judging
    # vocabulary against three words condemns the truth. A sparse layer means
    # no verdict here; the classical witness takes such pages instead.
    if len(theirs) < LAYER_MIN_WORDS:
        return None
    if len(ours & theirs) / len(ours) < LAYER_MIN_OVERLAP:
        body = " ".join(_strip_tags(it.html) for it in items
                        if it.html and not it.is_furniture)
        if _layer_alignment(body, layer) >= LAYER_MIN_ALIGNED:
            return None
        return "reads nothing like the page's own text layer"
    return None


# The loop guard and the layer guard both need the reading to give itself
# away, or the page to carry forty words of layer. A page of handwriting
# does neither. The Knights of Industry reproduces seven letters in
# facsimile, and each came back as fluent text that is not on the page ("I
# should should I will there should / I should I / I should I"), in the wrong
# language for a German hand, with no picture of the letter; an advertisement
# set in three languages came back with twenty-five paragraphs about the
# State of Virginia. A badly scanned page does the same in a smaller way: the
# words that can be read are read and the rest of each line is supplied.
#
# The recogniser knows when it is doing this. It reports a confidence for
# every page, which this pipeline has kept since it began and never used:
# the median over 38,250 cached pages is 0.995, and the seven letters are the
# seven lowest pages of their book, 0.75 to 0.84. On its own that number
# accuses half-titles and library slips too, which are short, and a few
# pages of honest print. So it only raises the question, and only of a page
# with words enough to answer it; the answer is a witness's. Of the 38 pages
# on the shelf under 0.93 with thirty words or more, the better of the page's
# two witnesses finds under a third of the words on 24 (letters, manuscript
# plates, two damaged scans, an engineer's drawing sheets), and seven tenths
# or more on the 6 that are print. One manuscript plate reaches 0.55 on the
# strength of its printed caption, and the line is drawn above it.
#
# Such a page is shown as a picture of itself. Nothing on it is lost that
# way, and nothing is said that it does not say.
#
# What a witness does read on such a page is still printed, beneath the
# picture: a block of four words or more that a witness has nine in ten of.
# That is the printed caption under the manuscript plate, which is print and
# was read right (and, being a caption, goes under the picture as one), and
# the legible labels of an engineer's drawing; it is none of a letter in
# facsimile, and on the two damaged pages of Lloyd's it is no paragraph at
# all, since none of them came through whole.
UNSURE_CONFIDENCE = 0.93


UNSURE_MIN_WORDS = 30


VOUCHED_SHARE = 0.6


BLOCK_VOUCHED = 0.9


BLOCK_MIN_WORDS = 4


_WITNESS_WORD = re.compile(r"[a-z]{3,}")


def _share_witnessed(text: str, witness: "str | None") -> float:
    """How much of a reading, word for word, a witness also reads."""
    ours = Counter(_WITNESS_WORD.findall(text.lower()))
    total = sum(ours.values())
    if not total or not witness:
        return 0.0
    theirs = Counter(_WITNESS_WORD.findall(witness.lower()))
    return sum(min(n, theirs.get(w, 0)) for w, n in ours.items()) / total


def unvouched_reading(items, page_path: "str | None"
                      ) -> "tuple[float, float, list] | None":
    """The recogniser's confidence, the best witness's share, and the blocks
    a witness reads word for word, where a page's words are ones neither
    will stand behind as a whole; None where they are vouched for, or there
    is nothing to judge by."""
    said = [it for it in items if it.html and not it.is_furniture]
    confs = [it.conf for it in said if it.conf is not None]
    if not confs or not page_path:
        return None
    conf = sum(confs) / len(confs)
    if conf >= UNSURE_CONFIDENCE:
        return None
    text = " ".join(_strip_tags(it.html) for it in said)
    if len(text.split()) < UNSURE_MIN_WORDS:
        return None
    layer = None
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            layer = fh.read()
    except OSError:
        pass
    if not witness.witness_available():
        if not _layer_can_testify(page_path):
            switches.UNWITNESSED["fabrication"] += 1
            return None                 # nobody to ask: not guilt
        witnesses = [layer]
    else:
        witnesses = [layer, witness._classical_text(page_path)]
    best = max(_share_witnessed(text, w) for w in witnesses)
    if best >= VOUCHED_SHARE:
        return None
    borne_out = []
    for it in said:
        words = _strip_tags(it.html)
        if len(_WITNESS_WORD.findall(words.lower())) >= BLOCK_MIN_WORDS and max(
                _share_witnessed(words, w) for w in witnesses) >= BLOCK_VOUCHED:
            borne_out.append(it)
    return conf, best, borne_out


def _page_picture(page_path: str) -> "PageItem | None":
    """The whole page as the one picture on it."""
    try:
        with Image.open(page_path) as im:
            return PageItem(figure=im.convert("RGB").copy(), label="Picture",
                            box=(0.0, 0.0, 1.0, 1.0), whole_page=True)
    except Exception:
        return None


def _native_layer_body(page_path: str) -> "list[PageItem] | None":
    """
    The page rebuilt from its own born-digital text, or None where there is
    no such authority (a scan, a photograph, an empty layer).

    The last resort when every reading of a native page fails the guards:
    the publisher's text IS the page, so restoring from it is not a guess —
    unlike a scan, where a failed page must stay honestly unread. Leading
    watermark and folio lines are dropped by shape (a URL, a bare number):
    they are the page's furniture, not its words.
    """
    if not os.path.exists(page_path + ".native"):
        return None
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            layer = fh.read()
    except OSError:
        return None
    lines = [ln.strip() for ln in layer.splitlines() if ln.strip()]
    while lines and (re.fullmatch(r"[0-9ivxlc]+", lines[0].lower())
                     or "http" in lines[0].lower()):
        lines.pop(0)
    text = " ".join(lines)
    if len(text) < 40:
        return None
    return [PageItem(html=f"<p>{html.escape(text)}</p>")]


def _layer_can_testify(page_path: str) -> bool:
    """Whether the page has a text layer rich enough to judge a reading."""
    try:
        with open(page_path + ".layer.txt", encoding="utf-8") as fh:
            return len(re.findall(r"[a-z]{3,}", fh.read().lower())) \
                >= LAYER_MIN_WORDS
    except OSError:
        return False


# A handful of words on visually blank paper is show-through, not text.
# The envelope, measured on a photographed book: the two phantom pages that
# prompted this (one transcribed the ghost of the reverse leaf, one invented
# a fluent sentence outright) carry a core ink fraction of 0.00000 exactly;
# the faintest REAL sparse page measured 0.00597, and the blank-figure
# survey higher up tops out at 0.00071. The floor below sits between the
# populations, six-fold under the faintest real page.
#
# The floor honours a withdrawn ancestor: "text on a near-blank page" was
# tried in the fabrication guard and thrown out after condemning two books'
# title pages, and the measured record holds a real six-word page whose ink
# is indistinguishable from a blank leaf's. Pages that short are never
# questioned here — the dedication class stays out of reach on principle,
# an accepted blind spot. There is deliberately NO upper bound: the first
# draft capped suspicion at 40 words as a cost bound, and the very next
# rebuild produced a 140-word invention on a blank verso that sailed over
# it. No real page carries a hundred words on paper with no measurable
# ink, so length is evidence of nothing; the ink floor and the agreeing
# witness are the safety, and they cost a witness read only on pages whose
# paper is already blank.
PHANTOM_MIN_WORDS = 10


PHANTOM_CORE_INK = 0.001


def _reads_as_empty(text: str) -> bool:
    """No text at all — or nothing but the recogniser's own bracketed asides.

    On unreadable regions surya editorialises instead of transcribing:
    "[Faded text block, likely bleed-through from the reverse side of the
    page]". A page whose entire reading is such asides has, by the
    recogniser's own account, no text — so it belongs to the empty-page
    gate, whose classical-reader rescue still protects a page the asides
    were wrong about.
    """
    if not text.strip():
        return True
    # A body that is nothing but a bare arabic number is the page's folio
    # leaked into its content, not text; blank versos ship exactly this
    # when the recogniser reads only the corner digit. Roman numerals are
    # deliberately NOT included — "vi" may be a part divider's whole
    # printed content, and "civil" is a word. The empty gate's rescue
    # still protects a page this is wrong about.
    if re.fullmatch(r"[0-9]{1,4}", text.strip()):
        return True
    unbracketed = re.sub(r"\[[^][]*\]", " ", text)
    # Any letter or digit, in any script. The test was for an ASCII one, so
    # a page of Greek or Cyrillic read as empty: it became a candidate for
    # deletion as blank and its PDF text layer was set aside.
    return not any(ch.isalnum() for ch in unbracketed)


@functools.lru_cache(maxsize=None)
def _page_core_ink(page_path: str) -> "float | None":
    """Black-hat ink fraction over the inner 70% of the page.

    The crop dodges what lives in a photographed page's outer margins —
    thumbs, page-edge shadow, the desk. The black-hat threshold is a
    contrast gate: show-through ghosts sit far below 40 grey levels of
    local contrast at any scale, while pressed ink clears it easily, so
    the fraction separates the two populations regardless of resolution.
    """
    img = cv2.imread(page_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    core = img[int(h * .15):int(h * .85), int(w * .15):int(w * .85)]
    if core.size == 0:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    hat = cv2.morphologyEx(core, cv2.MORPH_BLACKHAT, k)
    return float((hat > 40).mean())


def _phantom_blank_pages(page_paths, texts, has_figure, skip=frozenset()):
    """Pages whose sparse reading is show-through, by two agreeing witnesses.

    Surya generates, so on a blank verso it can transcribe the previous
    leaf's ghost — or invent a sentence outright; both arrived on one book
    the day the geometry ladder changed its pixels. The empty-page gate in
    the EPUB assembly cannot see them: it questions pages with NO text,
    and these carry a little. Conviction here needs both independent
    witnesses to agree: the paper must hold essentially no ink, AND the
    classical reader must find nothing. Neither alone may convict —
    tesseract also goes silent on faint real pages (one measured page held
    50 true words it could not see), and ink alone would rule with no
    second opinion. A page with a figure is never questioned: a picture is
    content. With no witness installed nothing is convicted; the pass is
    counted unexamined, mirroring the empty-page gate's honesty.
    """
    out = set()
    for i, path in enumerate(page_paths):
        if i in skip or has_figure[i]:
            continue
        words = len(re.findall(r"[A-Za-z]{2,}", texts[i]))
        if words < PHANTOM_MIN_WORDS:
            continue
        ink = _page_core_ink(path)
        if ink is None or ink >= PHANTOM_CORE_INK:
            continue
        n = witness._classical_word_count(path, min_len=3)
        if n is None:
            switches.UNWITNESSED["fabrication"] += 1
            continue
        if n == 0:
            out.add(i)
    return out


def _reading_agrees(a_items, b_items) -> float:
    """Word-set agreement between two readings of one page."""
    def ws(items):
        return set(re.findall(r"[a-z]{3,}", " ".join(
            _strip_tags(it.html) for it in items if it.html).lower()))
    wa, wb = ws(a_items), ws(b_items)
    if not wa and not wb:
        return 1.0
    return len(wa & wb) / max(1, len(wa | wb))
