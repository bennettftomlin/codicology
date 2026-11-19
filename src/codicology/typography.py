"""Typography restored on a built page: directional quotes, the C1 control
characters a PDF's text layer leaves where a font had them, the text a
link's unclosed address swallowed, and the census of characters that
usually mean the recogniser stumbled.
"""
from __future__ import annotations

import html
import os
import re
import unicodedata
from collections import Counter
from .markup import _strip_tags


# ── typography restoration ───────────────────────────────────────────────────

# Words that legitimately OPEN with an apostrophe, where the mark elides
# letters rather than opening a quotation: "'tis", "'em", "the '90s". An
# opening single quote before these would be wrong far more often than right.
_ELISIONS = re.compile(r"(?:tis|twas|twere|em|cause|til|till|round|bout|"
                       r"neath|gainst|\d0s?)\b", re.I)


def _requote(text: str, before: str = " ") -> str:
    """Directional quotes for one run of plain text, letters untouched.

    The printed page had directional quotes; the recogniser flattened them
    to straight ones — two and a half thousand in one book. Putting them
    back is restoration, not modernisation, which is why this pass exists
    and the spelling-modernising kind never will. The rules are the
    typesetter's own: a quote after space or an opening bracket opens, any
    other closes; a straight apostrophe between letters, after a letter, or
    eliding ("'tis", "'90s") is ’.

    `before` is the text the run follows: a run that starts just after an
    inline tag goes on from the words before it ("<i>Ulysses</i>" closes
    after the title, and <i>Guardian</i>'s is a possessive). What a quote
    follows is read as already restored, so quotes that open together
    ("'Free trade,' …" inside double quotes) both open.
    """
    out = []
    for i, ch in enumerate(text):
        prev = out[-1] if out else (before[-1:] or " ")
        if ch == '"':
            out.append("“" if prev in " \t\n(—–[{‘“"
                       else "”")
        elif ch == "'":
            nxt = text[i + 1] if i + 1 < len(text) else " "
            if prev.isalnum():
                out.append("’")           # contraction or possessive
            elif _ELISIONS.match(text[i + 1:]):
                out.append("’")           # elision keeps the apostrophe
            elif prev in " \t\n(—–[{“" and (nxt.isalnum()
                                                          or nxt in "“\""):
                out.append("‘")
            else:
                out.append("’")
        else:
            out.append(ch)
    return "".join(out)


# A tag, its quoted attribute values allowed to hold ">".
_TAG_SPLIT = re.compile(r"""(<(?:[^>"']|"[^"]*"|'[^']*')*>)""")

# Tags that end a run of text the way a line break does: a quote after one
# of these starts afresh. Any other tag (<i>, <sup>, <a>…) sits inside a
# sentence, and the text goes on through it.
_BREAKS_TEXT = re.compile(r"</?(?:p|div|h[1-6]|li|ol|ul|td|th|tr|table|br|hr|"
                          r"blockquote|figure|figcaption|section|dd|dt|dl)\b",
                          re.I)


def normalize_typography(bodies: list[str],
                         page_paths: "list[str] | None" = None) -> dict:
    """
    Restore what the recogniser flattened: quote direction, and the double
    spaces no printed page contains. Deliberately NOT normalised: spaced
    ellipses (". . ." is how older houses set them — that is the book),
    hyphens standing where a dash might (a guess between minus, range and
    dash would be wrong somewhere), and anything involving letters, which
    belong to the OCR and its witnesses. Text inside tags is never touched.

    Neither is a born-digital page. Its text is the publisher's own
    typesetting — restored or reconciled from the layer — and its
    punctuation is therefore the real thing, not a flattening to repair. A
    heuristic that overwrote it would be guessing over an answer sheet.
    Only pages the OCR alone speaks for are touched.
    """
    stats = {"quotes": 0, "spaces": 0, "native_skipped": 0}
    for i, body in enumerate(bodies):
        if page_paths is not None and i < len(page_paths) \
                and os.path.exists(page_paths[i] + ".native"):
            stats["native_skipped"] += 1
            continue
        parts = _TAG_SPLIT.split(body)
        before = " "
        for j, part in enumerate(parts):
            if part.startswith("<"):
                if _BREAKS_TEXT.match(part):
                    before = " "
                continue
            n_q = part.count('"') + part.count("'")
            if n_q:
                part = _requote(part, before)
                stats["quotes"] += n_q
            collapsed = re.sub(r"(?<=\S)  +(?=\S)", " ", part)
            if collapsed != part:
                stats["spaces"] += 1
                part = collapsed
            parts[j] = part
            if part:
                before = part
        bodies[i] = "".join(parts)
    return stats


def _c1_repair_table() -> dict:
    """Where each C1 control character came from, read back through cp1252."""
    table = {}
    for c in range(0x80, 0xA0):
        try:
            table[chr(c)] = bytes([c]).decode("cp1252")
        except UnicodeDecodeError:
            pass          # 0x81 0x8D 0x8F 0x90 0x9D are unassigned there
    return table


C1_REPAIR = _c1_repair_table()


# An address the recogniser opened and never closed. It writes a printed
# web address as a link, and now and then leaves the quotation mark off the
# end of it; the parser that makes its markup well-formed then takes
# everything up to the next quotation mark on the page for the address.
# What follows the address is no longer text (one book lost a whole entry
# of its bibliography into an address nobody sees), and the pieces of the
# next address become attributes named "04", "12", "2016", which is not XML:
# a reading system that holds a book to its format shows an error where the
# page should be.
_XML_NAME = re.compile(r"[A-Za-z_][\w.\-]*(?::[A-Za-z_][\w.\-]*)?$")
# (a name may be anything here, "<" included: these are the names the
# parser made of whatever followed the quotation mark it took for the end)
_A_TAG = re.compile(r"<a\b((?:\s+[^\s=>/]+(?:=\"[^\"<>]*\")?)*)\s*(/?)>")
_AN_ATTR = re.compile(r"([^\s=>/]+)(?:=\"([^\"<>]*)\")?")
_KEPT_IN_AN_ADDRESS = re.compile(r"&lt;(/?(?:i|b|em|strong|sup|sub))&gt;")


def _swallowed(attrs: str) -> "tuple[str, bool] | None":
    """What an <a> tag's address took in besides the address: (the address
    and that text, as markup; whether the tag is one no XML allows), or
    None for a link whose address is an address. The two signs that are
    certain: an address holds no tags, and an attribute is named as XML
    names one. An address with spaces in it is not a sign: printers break
    long addresses over lines, and such a link is left alone."""
    pairs = _AN_ATTR.findall(attrs)
    href = next((v for k, v in pairs if k == "href"), None)
    if href is None:
        return None
    said = html.unescape(href)
    broken = not all(_XML_NAME.match(k) for k, _ in pairs)
    if not broken and "<" not in said:
        return None
    # what the address ran into when it reached the next tag that has a
    # quotation mark in it: the opening of that tag, which is not text
    said = re.sub(r"<[A-Za-z][^<>]*$", "", said)
    said = _KEPT_IN_AN_ADDRESS.sub(r"<\1>", html.escape(
        re.sub(r"</?(?!(?:i|b|em|strong|sup|sub)\b)[A-Za-z][^<>]*>", " ",
               said), quote=False))
    for tag in ("i", "b", "em", "strong", "sup", "sub"):
        if said.count(f"<{tag}>") != said.count(f"</{tag}>"):
            said = re.sub(rf"</?{tag}>", "", said)
    return said.strip(), broken


def mend_swallowed_links(bodies: list) -> int:
    """Give back the text a link's unclosed address took in.

    The address and what it swallowed are printed as the text they are,
    and the link is taken off: its address was never known, only where it
    began. Where the link's own words already say all that the address
    holds (the recogniser wrote the entry twice), nothing was lost and
    nothing is added. Returns how many were mended."""
    words = lambda t: re.findall(r"\w+", html.unescape(_strip_tags(t)).lower())
    n = 0
    for i, body in enumerate(bodies):
        if "<a" not in body:
            continue
        out, pos = [], 0
        for m in _A_TAG.finditer(body):
            if m.start() < pos:
                continue
            got = _swallowed(m.group(1))
            if got is None:
                continue
            said, broken = got
            end = -1 if m.group(2) else body.find("</a>", m.end())
            held = body[m.end():end] if end >= 0 else ""
            took = words(said.split(None, 1)[1]) if " " in said else []
            shown = set(words(held))
            told = bool(took) and sum(w in shown for w in took) >= 0.8 * len(took)
            # what the link held is in a paragraph already: the wrappers of
            # a block the address ran into are not put inside it
            held = re.sub(r"</?(?:p|div)\b[^<>]*>", " ", held)
            out.append(body[pos:m.start()]
                       + (held if told else (said + " " + held).strip()))
            pos = end + len("</a>") if end >= 0 else m.end()
            n += 1
        if pos:
            bodies[i] = "".join(out) + body[pos:]
    return n


# A link that covers only the first part of its address, the rest following
# it as plain text: "<a href=…abstract=2723611>https://papers</a>.ssrn.com/
# abstract=2723611". The page reads in full and the address is whole, but
# only "https://papers" can be pressed. A layer that breaks the address over
# a line holds it as two words, and the reconciler gave the link the first.
# 1,143 links in 62 books on the shelf. Where the link's text and the run
# hard against it spell the address, the link is closed after the run.
_SHORT_LINK = re.compile(r'<a href="([^"]+)">([^<]+)</a>([^\s<]+)')
_TRAILING = ".,;:’”'\")]"


def close_short_links(bodies: list[str]) -> int:
    n = 0
    for i, body in enumerate(bodies):
        if "</a>" not in body:
            continue

        def mend(m):
            nonlocal n
            href, text, rest = m.group(1), m.group(2), m.group(3)
            core = rest.rstrip(_TRAILING)
            # a closing bracket the address itself holds stays with it
            while len(core) < len(rest) and rest[len(core)] == ")" \
                    and core.count("(") > core.count(")"):
                core += ")"
            whole = html.unescape(text + core)
            target = html.unescape(href)
            if not core or not re.search(r"[/.=]", whole) or not (
                    target == whole or target.endswith("//" + whole)
                    or target.endswith(whole) and len(whole) > 12):
                return m.group(0)
            n += 1
            return f'<a href="{href}">{text}{core}</a>{rest[len(core):]}'

        bodies[i] = _SHORT_LINK.sub(mend, body)
    return n


def repair_c1_controls(bodies: list[str]) -> int:
    """Put back the punctuation a mis-decoded text layer turned into controls.

    A PDF writes its punctuation in WinAnsi, where 0x97 is an em dash and
    0x95 a bullet; decoded as Latin-1 instead they become C1 control
    characters, which are not punctuation and not anything — they render as
    nothing, match no search, and read as nothing aloud. One field manual
    carried 608 of them, every bullet in every list it prints.

    This is the one character repair the pipeline makes on its own, and it
    makes it because for punctuation the mapping is not a guess: cp1252 is
    what those bytes meant, and reading them back through it inverts the
    fault rather than correcting the text. The five bytes cp1252 leaves
    unassigned are left exactly as they are — see [unusual_characters],
    which reports them for a human instead.

    Punctuation is the whole of the safe case, and the limit is a real one.
    A PDF may also use a C1 byte as an index into a SYMBOL font, where it
    stands for a dingbat and cp1252 has nothing to say about it: one manual
    sets its second-level list marker at 0x83, which cp1252 calls a florin
    sign — visible, searchable, and not what the page prints. So anything
    that is not punctuation needs the book's own word for it, exactly as a
    missed running head does: restored only where the book prints that same
    character somewhere it arrived intact. A French book that lost its œ
    says so; a manual that never prints a florin does not.
    """
    seen = set()
    for body in bodies:
        seen.update(body)
    repair = {ch: tgt for ch, tgt in C1_REPAIR.items()
              if unicodedata.category(tgt).startswith("P") or tgt in seen}
    n = 0
    for i, body in enumerate(bodies):
        if not any(0x80 <= ord(ch) <= 0x9F for ch in body):
            continue
        out = []
        for ch in body:
            rep = repair.get(ch)
            out.append(rep if rep is not None else ch)
            n += rep is not None
        bodies[i] = "".join(out)
    return n


def unusual_characters(bodies: list[str]) -> "list[tuple[str, int]]":
    """Characters that usually mean the recogniser stumbled, with counts —
    the broken-ligature ¬, box-drawing strays, replacement characters.
    Reported for a human eye; nothing is ever changed on this evidence."""
    sus = Counter()
    for body in bodies:
        text = re.sub(r"<[^>]+>", "", body)
        for ch in text:
            o = ord(ch)
            if ch in "¬¦§¤°^~`|\\" or 0x2500 <= o <= 0x257F \
                    or ch == "�" or 0x0080 <= o <= 0x009F:
                sus[ch] += 1
    return sus.most_common(8)


PAGE_HYPHEN = re.compile(r"([A-Za-z]+)-((?:\s*</[^>]+>)*\s*)$")


PAGE_CONT = re.compile(r"^((?:\s*<[^>]+>)*\s*)([a-z][A-Za-z'’]*[.,;:!?'’\"]*)")


def join_page_break_hyphens(bodies: list, dropped: "set[int]") -> dict:
    """A word the printer split across a page turn is one word.

    Surya rejoins line-break hyphenation within a page, but a page is the
    edge of its world: eagle shipped sixty words split at page edges —
    Rus- | sia — where search cannot match and speech stumbles. When a
    kept page ends mid-word on an ASCII hyphen and the next kept page
    opens with a lowercase fragment, and the joined form is a word — the
    dictionary's or this book's own — the continuation moves up to
    complete it. The page-break anchor stays where it was.

    Refusals are the point: an em-dash never matches, a capitalized
    continuation is left alone, a heading is never raided for its first
    word, and a printed compound breaking at the page edge (self- |
    control) fails the join gate and keeps its hyphen, exactly as set."""
    from .adjudicate import _is_word, fold_word
    kept = [i for i in range(len(bodies)) if i not in dropped]
    # the whole-book vocabulary is only evidence for judging a join, so a
    # book with no page ending mid-word — common in modern typesetting —
    # pays nothing for it (E4)
    if not any(PAGE_HYPHEN.search(bodies[i]) for i in kept):
        return {"joined": 0, "refused": 0}
    vocab = set()
    for b in bodies:
        for w in re.findall(r"[A-Za-z][A-Za-z'’]*", _strip_tags(b)):
            vocab.add(fold_word(w))
    joined = refused = 0
    for a, b in zip(kept, kept[1:]):
        m = PAGE_HYPHEN.search(bodies[a])
        if not m:
            continue
        c = PAGE_CONT.match(bodies[b])
        if not c or re.search(r"<(h\d|figure)", c.group(1)):
            continue
        frag = c.group(2)
        word2 = re.match(r"[a-zA-Z'’]+", frag).group(0)
        f = fold_word(m.group(1) + word2)
        if not (_is_word(f) or f in vocab):
            refused += 1
            continue
        bodies[a] = bodies[a][:m.start()] + m.group(1) + frag + m.group(2)
        bodies[b] = c.group(1) + bodies[b][c.end():].lstrip()
        joined += 1
    return {"joined": joined, "refused": refused}
