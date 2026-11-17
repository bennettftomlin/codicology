"""Born-digital pages held to their own text: where the PDF's text layer is
the publisher's typesetting, its words replace the recogniser's word by
word inside runs both agree on, and words the reading passed over are put
back where the layer shows them standing in the line.
"""
from __future__ import annotations

import html
import json
import os
import re
import unicodedata
from PIL import Image
from collections import Counter
from .markup import _SETTING, _strip_tags
from .typography import C1_REPAIR


# What some layers write where they broke a line: "Air\x02borne".
_CONTROL = re.compile(r"[\x00-\x1f]")


# What a layer holds for a glyph set in a symbol font: a private-use code
# point, or a control character. The publisher's bullet is U+F0B7 in the
# layer of Principles of Macroeconomics and 0x83 in FM 3-06, and no reader
# has a glyph for either; the recogniser read the bullet off the page. The
# layer says which words a page prints. It does not get to say that one of
# them is a character nobody can print.
#
# One kind of control character is not that. A layer that stores its
# punctuation as WinAnsi bytes holds 0x92 for the apostrophe and 0x97 for
# the dash, and repair_c1_controls puts those back once the page is
# assembled: FM 21-10 sets "unit's" and "LEADERS'" that way throughout, and
# refusing those words left the recogniser's straight apostrophe where the
# book has a curly one. What that repair will mend is as good as printed.
_NO_GLYPH = re.compile("[\x7f-\x9f\ue000-\uf8ff\U000f0000-\U0010fffd]")


def _no_glyph(word: str) -> bool:
    """Whether a word of the layer holds a character nobody can print."""
    for m in _NO_GLYPH.finditer(word):
        meant = C1_REPAIR.get(m.group(0))
        if meant is None or not unicodedata.category(meant).startswith("P"):
            return True
    return False


# The recogniser's own markup is setting as well. What it takes for
# mathematics it writes the way TeX would: a command and the braces of its
# argument (\text{rate}), the mark of a subscript or a power (S_3, r^n), a
# backslash before a sign TeX keeps for itself (\$, \%). The page prints
# none of it, and inside a page the layer's plain word is taken over it like
# any other. At a page's edge it was a different word and stayed: three
# economics textbooks ended 27 pages on "S_3." or "\$446,280.99" where the
# rest of the page reads S3, and three display equations there kept the
# closing brace of a \text{ the layer had already unwrapped. Those 28
# words are all this admits. Of the shelf's 134 books with born-digital
# pages, 59 have markup at the edge of any block; built as far as the
# reconciler, they refuse 395 words at a page's edge, and the other 367
# are still refused: a URL the layer breaks in two, a rule to write on, a
# folio.
_MARKUP = re.compile(r"\\[A-Za-z]+|[\\_^{}]")


def _same_word(ours: str, layer: str) -> bool:
    """Whether two readings are one word set differently — see _SETTING and
    _MARKUP — allowing the first letter's case, the sentence boundary the
    reconciler already concedes to the layer."""
    b = layer.translate(_SETTING)
    for a in (ours, _MARKUP.sub("", ours)):
        a = a.translate(_SETTING)
        if a and a[1:] == b[1:] and a[:1].lower() == b[:1].lower():
            return True
    return False


# At a page's edge the pairing of our word with the layer's is unproven,
# and only the same word set differently is taken (_same_word). That kept
# folios and running heads from being pasted over headings. It also kept
# the recogniser's slips with the marks around a word: a quote it dropped
# ("People" where the page prints "‘People", Work, Sex & Power p. 8), one
# it added ("employment.'"), the doubled dash a bibliography sets for "the
# same author" read as one ("—" for "——", Nomads, Empires, States). A word
# whose letters are the layer's, and which differs only in quotes and
# dashes standing at its ends, is the same word: the layer says which marks
# the page prints. Inside a word a mark counts, so "well-known" is never
# "wellknown"; a word of nothing but dashes is the same as another such.
_EDGE_MARKS = "‘’‚‛'\"“”„‟′″«»—–‒−-"


def _same_but_its_marks(ours: str, layer: str) -> bool:
    a, b = ours.translate(_SETTING), layer.translate(_SETTING)
    if a == b:
        return False
    ca, cb = a.strip(_EDGE_MARKS), b.strip(_EDGE_MARKS)
    if ca or cb:
        return ca == cb
    dashes = "—–‒−-"
    return bool(ours) and bool(layer) and all(c in dashes for c in ours + layer)


# And the recogniser's slips between glyphs that look alike: the letter o
# for a zero, l or I for a one. When Protest Becomes Crime ends a page on a
# note's address, "www.archivochile.com/tesis/03_tpo/03po0004.pdf", read as
# "o3_tpo/o3p00004" — the last word on the page, so the layer's word was
# refused, and the address shipped with letters for its figures. Two words
# of one length that differ only where one has a glyph and the other its
# look-alike are one word misread: a folio or a running head pasted over a
# heading is a different word, not the same one with an o for a 0.
_LOOK_ALIKE = [set("0oO"), set("1lI|"), set("5S"), set("8B")]
LOOK_ALIKE_SHORTEST = 6     # a short word has too few letters to prove it


def _same_but_look_alikes(ours: str, layer: str) -> bool:
    if len(ours) != len(layer) or len(ours) < LOOK_ALIKE_SHORTEST \
            or ours == layer:
        return False
    return all(a == b or any(a in g and b in g for g in _LOOK_ALIKE)
               for a, b in zip(ours, layer))


# Words the publisher prints and the reading passed over.
#
# A recogniser that writes a page out word by word does what a copyist
# does. It writes a doubled word once ("to say that that the world" came
# out with one that), takes the second of two words that look alike for the
# first ("1.4 billion toe to 2.5 billion" lost its unit, "as well as fora
# for teaching" its fora), and now and then drops down a line from one
# word to another like it: Kill Boxes lost eleven words between "prevents"
# and "pretends". Where There Is No Doctor gives a dose as "children 9 to
# 14 years: 1 ½ tablets" and it was read as "½ tablets". On a born-digital
# page the layer has every one of these, and the reconciler, which only
# ever exchanged one word for another, left them out.
#
# Most of what the layer has and we lack is not that. The layer's order is
# the typesetter's: it reads a table across its columns, puts a running
# head after the last line of the text, and sets the page reference from
# the next column in mid-sentence ("the first day: soak 102 joint in cold
# water"). Of 263 runs of the layer's words with nothing of ours against
# them, on the 26,728 born-digital pages of 134 books that differ from
# their layer at all, 38 are words the reading passed over. Four things
# tell them, and each was needed:
#
#   the words round the gap agree, three on each side (a running head's
#     "oil price shocks" lined up with a caption's "two oil shocks");
#   each word is a word, or a figure (the layer runs words together where
#     a heading is set tight: "changehorizontal");
#   the page does not have it already: the layer has each word more often
#     than the page does (a chapter's number read after its title);
#   it stands in the line: in the PDF each word follows the one before it
#     on the same line of type, or opens the line below.
#
# All 38 were looked at on the page: 59 words in 28 books, two of them
# whole lines, 13 a doubled word the book prints (some of them the book's
# own slips, which are the book's). What the last test alone refuses stays
# on the list a build prints, as before.
PUT_BACK_ANCHOR = 3         # agreeing words wanted on each side of the gap


PUT_BACK_LONGEST = 40       # words; nothing longer than twelve was found


_VULGAR_SIGNS = "½¼¾⅓⅔⅛⅜⅝⅞"


_PRINTED_RUN = re.compile(r"[^\W_]+|[%s]" % _VULGAR_SIGNS)


def _printed_runs(words) -> list:
    """The letters and figures of some words, each run apart, lower case:
    what is counted when asking whether a page has a word."""
    return [r.lower() for r in _PRINTED_RUN.findall(" ".join(words))]


def _layer_geometry(page_path: str) -> "list | None":
    """A typeset page's words in the PDF's own order, each with where it
    stands: (word, left, bottom, right, top) in points. None for a scan, or
    with no PDF to ask. Read a character at a time, so that a box is the
    box of the character it is filed under."""
    try:
        with open(page_path + ".source.json") as fh:
            src = json.load(fh)
        if src.get("facsimile") or src.get("scan"):
            return None
        import pypdfium2 as pdfium
        import pypdfium2.raw as pdfium_c
        textpage = pdfium.PdfDocument(src["pdf"])[src["page"]].get_textpage()
        words, cur = [], None
        for i in range(textpage.count_chars()):
            ch = chr(pdfium_c.FPDFText_GetUnicode(textpage.raw, i))
            if ch.isspace():
                if cur:
                    words.append(tuple(cur))
                cur = None
                continue
            if ch == "\x00":
                continue
            left, bottom, right, top = textpage.get_charbox(i)
            if cur is None:
                cur = [ch, left, bottom, right, top]
            else:
                cur[0] += ch
                cur[1], cur[2] = min(cur[1], left), min(cur[2], bottom)
                cur[3], cur[4] = max(cur[3], right), max(cur[4], top)
        if cur:
            words.append(tuple(cur))
        return words
    except Exception:
        return None


def _follows(a: tuple, b: tuple) -> bool:
    """Whether word b comes straight after word a in a line of type: beside
    it, a word space off, or opening the line below it."""
    h = max(a[4] - a[2], b[4] - b[2], 1e-3)
    if min(a[4], b[4]) - max(a[2], b[2]) >= 0.5 * min(a[4] - a[2], b[4] - b[2]):
        # the same line: a word space is a third of the type's height and
        # a justified line's twice that; the gutter of a table is more
        return -0.6 * h <= b[1] - a[3] <= 1.6 * h
    # the next line down, and back towards the margin it started from
    return b[4] <= a[2] + 0.3 * h and a[2] - b[4] <= 1.6 * h and b[1] < a[1]


def _stands_in_the_line(geometry: list, layer_words: list, b0: int,
                        b1: int) -> bool:
    """Whether layer_words[b0:b1] is set in the line between the word before
    it and the word after: found in one place in the PDF, each word
    following the last."""
    printed = [g[0] for g in geometry]
    for lead in (2, 1):
        if b0 - lead < 0 or b1 + lead > len(layer_words):
            continue
        run = layer_words[b0 - lead:b1 + lead]
        at = [i for i in range(len(printed) - len(run) + 1)
              if printed[i:i + len(run)] == run]
        if len(at) == 1:
            first = at[0] + lead - 1
            return all(_follows(geometry[i], geometry[i + 1])
                       for i in range(first, first + (b1 - b0) + 1))
    return False


def reconcile_native_text(bodies: list[str], page_paths: list[str]) -> int:
    """
    Where a page's text is born-digital, defer to it word by word.

    On a scan, the embedded layer is another OCR's guess and ours is usually
    better. On a born-digital page the layer is not a guess — it is the
    publisher's own typesetting, so every disagreement is by definition our
    error, including the flattering ones: a recogniser that reads
    "competitiveness" where the book printed "competiveness" has silently
    edited the author. Surya still supplies everything structural — layout,
    reading order, figures, furniture — but inside tightly aligned prose
    runs, the native spelling wins.

    Only the unambiguous case substitutes: a run of at most three words that
    differs, bounded by words both sides agree on, with the same number of
    words on each side. Everything else — tables surya restructured, split
    hyphens the layer kept, regions that align loosely — stays ours.

    And words the layer prints where we have none are put back, where they
    are words the reading passed over and not the layer's own order: see
    PUT_BACK_ANCHOR. They are listed in reconcile_native_text.put_back and
    are not counted in what this returns.

    "Bounded" is enforced, not assumed. The aligner pairs a page's first
    words with the layer's first words whatever they are, and the layer's
    order is the typesetter's, not the reader's: InDesign writes a drop
    folio first and a chapter opening's heading last. On Lost Worlds' first
    index page that paired our "INDEX" with the layer's "248", the heading
    shipped as <h1>248</h1>, and --link-index, which finds the index by that
    heading, linked none of its 1,720 references. Across the 53 born-digital
    books on the shelf, 329 substitutions at a page's first or last words
    reached a book; 231 set the same word the publisher's way — curly
    quotes, en dashes, a true minus, primes — and most of the other 98 were
    somebody else's words: folios and running heads pasted over index heads
    and a caption's "Figure 8", "barter-free" turned "barter-technology".
    Where one side is open, only the same word set differently is taken
    (_same_word).

    A control character marks where the layer broke a line. Usually that is
    inside the very word it is paired with ("Air\x02borne"), the run is
    sound, and the rest of it is taken as before: 438 of the shelf's 612
    such runs. The other 174, 138 of them in one book's notes, were URLs the
    layer holds as two words where we read one; conceding the clean half
    paired our whole address with its first half and shipped "www.scmp.com/;"
    where the page prints the full path. A garbled word that is not the word
    it is paired with refuses its whole run.
    """
    import difflib

    changed = 0
    reconcile_native_text.unplaced.clear()
    reconcile_native_text.put_back.clear()
    vocabulary: "set | None" = None      # the book's own words, if asked for
    for pi, body in enumerate(bodies):
        if pi >= len(page_paths) or not body:
            continue
        base = page_paths[pi]
        if not os.path.exists(base + ".native"):
            continue
        try:
            with open(base + ".layer.txt", encoding="utf-8") as fh:
                layer_words = fh.read().split()
        except (OSError, Image.DecompressionBombError):
            # this pipeline stores pages losslessly, so one unusually
            # high-resolution page can trip Pillow's bomb guard — which is
            # an Exception, not an OSError, and used to crash the whole
            # folio audit instead of skipping the page (A6)
            continue
        if len(layer_words) < 10:
            continue
        # split the html into tags and text nodes; tokens live in text nodes
        parts = re.split(r"(<[^>]+>)", body)
        toks: list[tuple[int, int, int, str]] = []   # (part, start, end, word)
        for k, part in enumerate(parts):
            if part.startswith("<"):
                continue
            for m in re.finditer(r"\S+", part):
                toks.append((k, m.start(), m.end(), html.unescape(m.group(0))))
        ours = [t[3] for t in toks]
        sm = difflib.SequenceMatcher(a=ours, b=layer_words, autojunk=False)
        subs: list[tuple[int, str]] = []
        puts: list[tuple[int, list]] = []    # (the word before, the words)
        ops = sm.get_opcodes()
        on_layer = on_page = geometry = None

        def agreed(n: int, step: int) -> int:
            """How many words in a row agree on that side of op n: equal,
            or the same word set the publisher's way."""
            count = 0
            n += step
            while 0 <= n < len(ops) and count < PUT_BACK_ANCHOR:
                kind, x0, x1, y0, y1 = ops[n]
                if kind != "equal" and not (
                        kind == "replace" and x1 - x0 == y1 - y0
                        and all(_same_word(a, b) for a, b in
                                zip(ours[x0:x1], layer_words[y0:y1]))):
                    break
                count += x1 - x0
                n += step
            return count

        for n, (tag, a0, a1, b0, b1) in enumerate(ops):
            if tag == "insert" and 0 < a0 < len(ours) \
                    and toks[a0 - 1][0] == toks[a0][0]:
                words = layer_words[b0:b1]
                printed = _printed_runs(words)
                passed_over = (
                    b1 - b0 <= PUT_BACK_LONGEST and bool(printed)
                    and not any(_CONTROL.search(w) or _no_glyph(w)
                                for w in words)
                    and agreed(n, -1) >= PUT_BACK_ANCHOR
                    and agreed(n, +1) >= PUT_BACK_ANCHOR)
                if passed_over:
                    from .adjudicate import _is_word, fold_word
                    for run in printed:
                        if run.isdigit() or run in _VULGAR_SIGNS \
                                or _is_word(fold_word(run)):
                            continue
                        if vocabulary is None:
                            vocabulary = {
                                fold_word(w) for b in bodies for w in
                                re.findall(r"[^\W\d_]+", _strip_tags(b))}
                        if fold_word(run) not in vocabulary:
                            passed_over = False
                            break
                if passed_over:
                    if on_layer is None:
                        on_layer = Counter(_printed_runs(layer_words))
                        on_page = Counter(_printed_runs(ours))
                    wanted = Counter(printed)
                    passed_over = all(on_layer[r] - on_page[r] >= c
                                      for r, c in wanted.items())
                if passed_over:
                    if geometry is None:
                        geometry = _layer_geometry(base) or []
                    passed_over = bool(geometry) and _stands_in_the_line(
                        geometry, layer_words, b0, b1)
                if passed_over:
                    puts.append((a0 - 1, words))
                    on_page.update(printed)
                    reconcile_native_text.put_back.append(
                        (pi, " ".join(ours[max(0, a0 - 3):a0]),
                         " ".join(words), " ".join(ours[a0:a0 + 3])))
                elif b1 - b0 <= 2 and any(
                        ch.isdigit() or ch in _VULGAR_SIGNS
                        for w in words for ch in w):
                    # A figure the publisher prints between two words we
                    # read one after the other, and not one that can be put
                    # back on the tests above. It is said and not acted on:
                    # see reconcile_native_text.unplaced.
                    reconcile_native_text.unplaced.append(
                        (pi, " ".join(ours[max(0, a0 - 2):a0]),
                         " ".join(words), " ".join(ours[a0:a0 + 2])))
            if tag != "replace" or (a1 - a0) != (b1 - b0) or (a1 - a0) > 3:
                continue
            run = list(zip(range(a0, a1), layer_words[b0:b1]))
            # Some layers carry control characters at line-break points
            # ("Cana\x02da"); deferring to those would corrupt a word we
            # read correctly. Authority does not extend to garbage — and
            # where the garbled word is not even the word it is paired with,
            # not to the rest of its run either (see the docstring).
            if any(_CONTROL.search(cand)
                   and not any(_same_word(ours[ti], w) for w in (
                       _CONTROL.sub("", cand), _CONTROL.sub("-", cand)))
                   for ti, cand in run):
                continue
            # The first or last run on the page has agreement on one side
            # only, so the pairing itself is unproven there.
            open_side = a0 == 0 or a1 == len(ours)
            for ti, cand in run:
                if _CONTROL.search(cand):
                    continue
                if _no_glyph(cand) and not _no_glyph(ours[ti]):
                    continue
                # The layer is authority on WHICH words are on the page, not
                # on how they are set. A book that prints a heading in small
                # capitals stores it lowercase and lets the font do the
                # capitals; Surya reads the printed glyphs. Deferring here
                # took the source characters over the rendered page and
                # downcased twenty-six section headings in one book —
                # "CONCLUSION" eight times over — and turned an economist
                # named McKinnon into Mckinnon. Measured across both
                # born-digital books available: every case-only substitution
                # was damage and none was a correction.
                # A difference confined to the FIRST letter is not that: it
                # is a sentence boundary, and the layer knows where those
                # are better than a recogniser reading one block at a time.
                # One book had a figure reference swallowed, leaving the
                # recogniser to capitalise the "illustrates" that followed;
                # the publisher's lower case is right there. Anything deeper
                # than the first letter — CONCLUSION, McKinnon, AAs — is the
                # page's own setting and stays ours.
                # A one-letter word has nothing after its first letter, so it
                # cannot show which of the two this is, and conceding it
                # broke a heading down the middle: FINDING A JOB came back
                # as "FINDING a JOB". With no evidence, the page keeps what
                # it prints.
                mine = ours[ti]
                if mine.lower() == cand.lower() and (
                        len(mine) == 1 or mine[1:] != cand[1:]):
                    continue
                if open_side and not (_same_word(mine, cand)
                                      or _same_but_its_marks(mine, cand)
                                      or _same_but_look_alikes(mine, cand)):
                    continue
                if mine != cand:
                    subs.append((ti, cand))
        if not subs and not puts:
            continue
        # A word exchanged takes the place of its word; words put back go
        # in after the word before them. Applied from the back, so that
        # recorded offsets stay valid.
        edits = [(toks[ti][0], toks[ti][1], toks[ti][2], html.escape(word))
                 for ti, word in subs]
        # A word that is the text of a link is the link's address too, as
        # the recogniser wrote it: correcting the text and not the address
        # left When Protest Becomes Crime showing the right address and
        # linking to the misread one.
        tags: dict[int, str] = {}
        for ti, word in subs:
            k = toks[ti][0]
            tag = tags.get(k - 1, parts[k - 1] if k else "")
            old_word = html.escape(ours[ti])
            if tag.startswith("<a ") and re.search(
                    r'href="[^"]*' + re.escape(old_word), tag):
                tags[k - 1] = tag.replace(old_word, html.escape(word))
        edits += [(k, 0, len(parts[k]), tag) for k, tag in tags.items()]
        edits += [(toks[ti][0], toks[ti][2], toks[ti][2],
                   " " + html.escape(" ".join(words))) for ti, words in puts]
        for k, a, z, new in sorted(edits, reverse=True):
            parts[k] = parts[k][:a] + new + parts[k][z:]
        changed += len(subs)
        bodies[pi] = "".join(parts)
    return changed


# Figures the layer prints inside a run of text where we read none. Where
# There Is No Doctor gives a dose as "children 9 to 14 years: 1 ½ tablets",
# and the recogniser read "½ tablets". The layer has the 1, and the
# reconciler only ever exchanges one word for another, so it stood.
#
# That one is put back now (see PUT_BACK_ANCHOR). What is not: across the
# 26,044 born-digital pages of 133 books, 111 such words sit between two
# words we read together, and most are the layer reading a table across its
# columns, or a page reference from the next column landing in mid-sentence
# ("soak 102 joint in cold water"). Twenty of the 111 hold a figure, in
# seven books. A figure is the word a reader acts on, so those that cannot
# be put back are listed when the book is built, for someone to look at:
# three of them are that table's page references, which the page does lack.
reconcile_native_text.unplaced = []


reconcile_native_text.put_back = []
