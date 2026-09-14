"""A contents line whose folio is Roman has an address like any other.

Front matter paginates in Roman, and the folio audit has read Roman off a
running head for a while, carrying it as i - ROMAN_FOLIO_BASE so the whole
numbering space stays in order. The contents parser recognised the very same
numerals -- it had a branch for them -- and passed None, so every front
matter line arrived with no address and fell through to the title hunt.

The hunt only moves FORWARD, and its floor moves with every placement. FM
3-06 says "FIGURES ..... iv" and its figure list was placed on page 237 of
348; Burning Up's "Acronyms and abbreviations xiii" went unplaced, and the
four lines after it, and its notes section, inherited a floor 200 pages
deep. None of that was a hunting problem. The books had said where to look.
"""
import pytest


def contents_table(rows):
    out = ["<h1>Contents</h1><table>"]
    for title, folio in rows:
        out.append(f"<tr><td>{title}</td><td>{folio}</td></tr>")
    out.append("</table>")
    return ["<p>half title</p>", "".join(out)]


def contents_leaders(rows):
    lines = "".join(f"<p>{t} ..... {f}</p>" for t, f in rows)
    return ["<p>half title</p>", "<h1>Contents</h1>" + lines]


# ── reading the number ───────────────────────────────────────────────────────

def test_roman_is_read_into_the_encoding_the_audit_uses(vtb):
    assert vtb._contents_folio("iv") == 4 - vtb.ROMAN_FOLIO_BASE
    assert vtb._contents_folio("viii") == 8 - vtb.ROMAN_FOLIO_BASE
    assert vtb._contents_folio("XIII") == 13 - vtb.ROMAN_FOLIO_BASE


def test_arabic_is_unchanged(vtb):
    assert vtb._contents_folio("12") == 12
    assert vtb._contents_folio("1") == 1


def test_what_is_not_a_folio_is_refused(vtb):
    # strictness comes from _roman_to_int: it must re-render exactly, and it
    # stops at 99, which is what front matter is
    for junk in ("", "iiii", "vx", "m", "A-4", "page", "1997"):
        assert vtb._contents_folio(junk) is None, junk


def test_roman_sorts_below_every_arabic_folio(vtb):
    assert vtb._contents_folio("xcix") < vtb._contents_folio("1")


# ── resolving it against the book ────────────────────────────────────────────

def folios(vtb, pairs):
    return [vtb.Folio(i, n, "", True) for i, n in pairs]


def test_a_roman_folio_resolves_inside_the_roman_run(vtb):
    f = folios(vtb, [(1, -999), (12, -988), (16, 2), (23, 9)])   # i, xii, 2, 9
    resolve = vtb.folio_resolver(f)
    assert resolve(vtb._contents_folio("viii")) == 8
    assert resolve(vtb._contents_folio("xii")) == 12


def test_an_arabic_folio_still_resolves_inside_the_arabic_run(vtb):
    f = folios(vtb, [(1, -999), (12, -988), (16, 2), (23, 9)])
    assert vtb.folio_resolver(f)(5) == 19


def test_the_two_numberings_never_interpolate_into_each_other(vtb):
    """The distance from folio iv to folio 7 is not a number of pages.

    Before the series split, a Roman folio with a Roman anchor on one side
    and an Arabic one on the other produced two wildly different candidates,
    and the resolver -- rightly refusing to guess between them -- returned
    nothing at all.
    """
    arabic_only = folios(vtb, [(16, 2), (23, 9), (40, 26)])
    assert vtb.folio_resolver(arabic_only)(vtb._contents_folio("viii")) is None
    roman_only = folios(vtb, [(1, -999), (12, -988)])
    assert vtb.folio_resolver(roman_only)(5) is None


# ── through the parser, in both shapes a contents page comes in ──────────────

def test_a_table_row_with_a_roman_folio_carries_it(vtb):
    entries, _ = vtb.parse_printed_toc(contents_table(
        [("Figures", "viii"), ("Tables", "ix"), ("Introduction", "1")]))
    got = {e.title: e.folio for e in entries}
    assert got["Figures"] == 8 - vtb.ROMAN_FOLIO_BASE
    assert got["Tables"] == 9 - vtb.ROMAN_FOLIO_BASE
    assert got["Introduction"] == 1


def test_a_dot_leader_line_with_a_roman_folio_carries_it(vtb):
    entries, _ = vtb.parse_printed_toc(contents_leaders(
        [("FIGURES", "iv"), ("PREFACE", "viii"), ("URBAN OUTLOOK", "1")]))
    got = {e.title: e.folio for e in entries}
    assert got["FIGURES"] == 4 - vtb.ROMAN_FOLIO_BASE
    assert got["PREFACE"] == 8 - vtb.ROMAN_FOLIO_BASE
    assert got["URBAN OUTLOOK"] == 1


def test_the_printed_numerals_are_still_kept_verbatim(vtb):
    entries, _ = vtb.parse_printed_toc(contents_table([("Figures", "viii")]))
    assert [e.folio_text for e in entries if e.title == "Figures"] == ["viii"]
