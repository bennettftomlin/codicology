"""A reviewer's decisions applied to a build: each round of corrections in a
decisions file, addressed to a word by its ordinal on its page, set where
the page's words still say what was decided about; a stale one is said
and left.
"""
from __future__ import annotations

import json
import os
from .figures import _same_file


def _decision_rounds(decisions_path) -> list:
    """The decisions files to apply, in order: none, one, or several.

    A file named here brings its earlier rounds with it: the ones
    `codicology decisions file` set aside beside it as book.decisions.r1.json
    and so on (review.decision_rounds). Naming only the round in hand is
    what the Calibre plugin does, and it used to mean the rounds before it
    were not applied at all."""
    from .review import decision_rounds
    if not decisions_path:
        return []
    if isinstance(decisions_path, (str, bytes, os.PathLike)):
        decisions_path = [decisions_path]
    out: list = []
    for given in (p for p in decisions_path if p):
        for path in decision_rounds(os.fspath(given)) or [given]:
            if not any(_same_file(path, seen) for seen in out):
                out.append(path)
    return out


def apply_reviewer_decisions(bodies: list, decisions_path: str,
                             kept_pages: "int | None" = None) -> dict:
    """A review sheet's exported decisions, reapplied at rebuild time —
    before the linkers run, so a corrected word can still earn its anchor.

    Same contract as `codicology apply`, and the same code
    (review.apply_page_decisions): the page's text only, the word each
    decision is addressed to, deletions before swaps, and a decision that
    cannot be placed with certainty is stale — reported and left as read,
    never guessed at."""
    from .adjudicate import tokens_as
    from .review import (apply_page_decisions, coerce_page, pages_guard,
                         retired_rows)
    try:
        with open(decisions_path) as fh:
            dec = json.load(fh)
        rows = dec.get("decisions", [])
    except (OSError, ValueError) as exc:
        print(f"    [!] decisions file unreadable, ignored: {exc}")
        return {"applied": 0, "stale": 0}
    if retired_rows(dec):
        print(f"    {len(retired_rows(dec))} decision(s) in "
              f"{os.path.basename(os.fspath(decisions_path))} are retired "
              f"and left alone")
    refusal = pages_guard(dec.get("page_files"),
                          kept_pages if kept_pages is not None
                          else len(bodies))
    if refusal:
        print(f"    refused: {refusal}")
        return {"applied": 0, "stale": len(rows)}
    applied = stale = 0
    by_page: dict = {}
    for d in rows:
        i = coerce_page(d)
        if i is None or not (0 <= i < len(bodies)):
            stale += 1
            print(f"    stale decision: page {d.get('page')!r} is not a "
                  f"page of this build — {d.get('old')!r} skipped")
            continue
        by_page.setdefault(i, []).append(d)
    for i in sorted(by_page):
        # read the page the way the file's rows were addressed
        with tokens_as(dec.get("tokens")):
            bodies[i], outcomes = apply_page_decisions(bodies[i], by_page[i])
        for d, why in zip(by_page[i], outcomes):
            if why is None:
                applied += 1
            else:
                stale += 1
                print(f"    stale decision: p{i} {d.get('old')!r} -> "
                      f"{d.get('new')!r} — {why}; kept as read")
    return {"applied": applied, "stale": stale}
