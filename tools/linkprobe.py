"""What today's linkers would make of every book already on the shelf.

A linker change used to be judged by rebuilding the library — five and a
half hours — or by an argument. This takes about four seconds, because it
never rasterises or reads a PDF: it opens each installed EPUB, strips the
note and footnote links back to the bare markers the linkers were given,
runs the real dispatcher over them, and compares.

The comparison that matters is not "how many links" but WHICH BOOKS STILL
REPRODUCE. A book whose installed copy was built by current code must come
back with exactly the count it already has; the baseline records which books
those are, so a change that breaks reproduction on a book nobody rebuilt is
a regression, named, before it is committed.

That property is the whole point, and it is the one the two bad measurements
in this project's history both lacked. A footnote change was cleared by
measuring on already-LINKED copies, where the pass has nothing left to claim
and returns zero either way. A grouping change was cleared by a simulation
that compared shipped code against itself. Both reported "no change" because
neither could report anything else; both shipped regressions. A probe that
cannot reproduce a book it has already built is not measuring anything, and
this one says so out loud.

    python tools/linkprobe.py                 # measure, diff against baseline
    python tools/linkprobe.py --update        # accept: rewrite the baseline

The baseline diff IS the review of a linker change. Update it deliberately.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from codicology import pipeline as P              # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BASELINE = os.path.join(HERE, "link-baseline.json")
LIBRARY = os.path.expanduser("~/Calibre Library")
PAGE = re.compile(r"page_(\d+)\.xhtml$")

# Undo exactly what the linkers do, so they are handed what they were handed
# the first time. The catch-all at the end is not decoration: one book sets
# its entry numbers as <sup><a>1</a></sup> with no period, which no specific
# pattern here matches, and the first version of this probe read that book as
# losing all 905 of its links. It was the stripper that was wrong.
_UNDO = [
    (re.compile(r'<sup><a epub:type="noteref"[^>]*href="[^"]*#fn-[^"]*"[^>]*>'
                r'(.{1,4}?)</a></sup>', re.S), r"<sup>\1</sup>"),
    (re.compile(r'<(p|li)([^>]*) id="fn-[^"]*"><a href="[^"]*">(.{1,4}?)</a>'), r"<\1\2>\3"),
    (re.compile(r'<sup><a epub:type="noteref"[^>]*>(\d{1,3})</a></sup>'), r"<sup>\1</sup>"),
    (re.compile(r'<(p|li)([^>]*) id="note-[^"]*"><a href="[^"]*">(\d{1,3})\.</a>'), r"<\1\2>\3."),
    (re.compile(r'<a id="note-[^"]*" href="[^"]*">(\d{1,3})\.</a>'), r"\1."),
    (re.compile(r'<a [^>]*href="[^"]*#(?:note|ref|fn)-[^"]*"[^>]*>(.*?)</a>', re.S), r"\1"),
    (re.compile(r'\s+id="(?:note|ref|fn)-[^"]*"'), ""),
]


def unlink(body: str) -> str:
    for pat, rep in _UNDO:
        body = pat.sub(rep, body)
    return body


def dispatch(bodies: list[str], dropped: set) -> tuple[int, int]:
    """build_epub's own note dispatch, kept in step with it by hand.

    Deliberately a copy and not a call: build_epub's version is wound
    through figure extraction and progress reporting and cannot be reached
    with bodies alone. Nothing needs to police the copy, because the
    reproduction property already does — let this drift from build_epub's
    dispatch and books stop reproducing, which is the loudest thing this
    tool can say.
    """
    n_heads = sum(1 for b in bodies if P.NOTES_HEAD.search(b))
    foot = P.link_footnotes(bodies, allow_numbered=(n_heads == 0))
    if n_heads >= 2:
        before = list(bodies)
        chapter = P.link_chapter_notes(bodies, dropped)
        chapter_result = list(bodies)
        trial = list(before)
        back = P.link_notes(trial, dropped=dropped, chapter_starts=None)
        if (back["groups"] and not back["misaligned"]
                and back["linked"] >= chapter["linked"]
                and back["groups"] > chapter["sections"]):
            bodies[:] = trial
            return foot["linked"], back["linked"]
        bodies[:] = chapter_result
        return foot["linked"], (chapter["linked"] if chapter["sections"] else 0)
    back = P.link_notes(bodies, dropped=dropped, chapter_starts=None)
    return foot["linked"], (back["linked"] if back["groups"] else 0)


def measure(path: str) -> dict | None:
    z = zipfile.ZipFile(path)
    files = {int(m.group(1)): n for m, n in
             ((PAGE.search(n), n) for n in z.namelist()) if m}
    if len(files) < 20:
        return None
    total = max(files) + 1
    raw = []
    for i in range(total):
        if i not in files:
            raw.append("")
            continue
        t = z.read(files[i]).decode("utf-8", "replace")
        raw.append(re.sub(r"</body>.*$", "",
                          re.sub(r"^.*?<body[^>]*>", "", t, flags=re.S), flags=re.S))
    has_note = sum(len(re.findall(r'href="[^"]*#note-', b)) for b in raw)
    has_foot = sum(len(re.findall(r'href="[^"]*#fn-', b)) for b in raw)
    bodies = [unlink(b) for b in raw]
    if sum(len(P.BODY_MARKER.findall(b)) for b in bodies) < 5 and not has_foot:
        return None                     # nothing in this book to link
    dropped = {i for i in range(total) if i not in files}
    foot, note = dispatch(bodies, dropped)
    return {"installed_notes": has_note, "probe_notes": note,
            "installed_foot": has_foot, "probe_foot": foot}


def state(r: dict) -> str:
    if r["probe_notes"] == r["installed_notes"] and r["probe_foot"] == r["installed_foot"]:
        return "reproduces"
    if r["probe_notes"] >= r["installed_notes"] and r["probe_foot"] >= r["installed_foot"]:
        return "would gain"
    return "WOULD LOSE"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--library", default=LIBRARY)
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--update", action="store_true",
                    help="accept what was measured and rewrite the baseline")
    args = ap.parse_args()

    books = {}
    for p in sorted(glob.glob(os.path.join(args.library, "*", "*", "*.epub"))):
        m = re.search(r"\((\d+)\)/", p)
        if not m:
            continue
        try:
            r = measure(p)
        except Exception as e:                     # a book we cannot read is
            print(f"  [!] {m.group(1)}: {type(e).__name__}: {e}")   # news, not
            continue                                                # silence
        if r is not None:
            r["name"] = os.path.basename(p)[:46]
            books[m.group(1)] = r

    old = {}
    if os.path.exists(args.baseline):
        old = json.load(open(args.baseline)).get("books", {})

    moved, gain, lose = [], [], []
    for bid, r in sorted(books.items(), key=lambda kv: int(kv[0])):
        s = state(r)
        was = old.get(bid)
        if was and (was.get("probe_notes"), was.get("probe_foot")) \
                != (r["probe_notes"], r["probe_foot"]):
            moved.append((bid, r, was))
        (gain if s == "would gain" else lose if s == "WOULD LOSE" else []).append((bid, r))

    print(f"{len(books)} books with notes or footnotes to link · "
          f"{sum(1 for r in books.values() if state(r) == 'reproduces')} reproduce exactly")
    if moved:
        print("\nCHANGED SINCE THE BASELINE — this is the review:")
        for bid, r, was in moved:
            print(f"  {bid:<6}{r['name']:48s}"
                  f"notes {was.get('probe_notes')}->{r['probe_notes']}  "
                  f"foot {was.get('probe_foot')}->{r['probe_foot']}")
    elif old:
        print("no change against the baseline")
    for label, rows in (("WOULD GAIN IF REBUILT", gain), ("WOULD LOSE IF REBUILT", lose)):
        if rows:
            print(f"\n{label}:")
            for bid, r in rows:
                print(f"  {bid:<6}{r['name']:48s}"
                      f"notes {r['installed_notes']}->{r['probe_notes']}  "
                      f"foot {r['installed_foot']}->{r['probe_foot']}")
    if args.update:
        json.dump({"generated": datetime.date.today().isoformat(), "books": books},
                  open(args.baseline, "w"), indent=1, sort_keys=True)
        print(f"\nbaseline rewritten: {args.baseline}")
    return 1 if (moved and not args.update) else 0


if __name__ == "__main__":
    sys.exit(main())
