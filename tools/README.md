# tools

Two instruments for judging a change to the note linkers before it ships.
Neither reads a PDF or runs OCR; both work on EPUBs that already exist.

## `linkprobe.py` — what today's code would make of every book

Opens each installed EPUB, strips its note and footnote links back to the
bare markers the linkers were originally given, runs the real dispatcher
over them, and compares. About **four seconds for the whole shelf**, against
five and a half hours to rebuild it.

```
python tools/linkprobe.py            # measure, diff against the baseline
python tools/linkprobe.py --update   # accept what was measured, rewrite it
```

The number to watch is not links, it is **how many books still reproduce**.
A book whose installed copy was built by current code must come back with
exactly the count it already has. `link-baseline.json` records that, so a
change which breaks reproduction on a book nobody rebuilt is named before it
is committed, and the baseline diff is the review of the change.

That property is what this project's two bad measurements both lacked. A
footnote change was cleared by measuring on already-*linked* copies, where
the pass has nothing left to claim and returns zero either way. A grouping
change was cleared by a simulation that compared shipped code with itself.
Both said "no change" because neither could say anything else.

The dispatch in `linkprobe.py` is a hand copy of `build_epub`'s, which cannot
be called with bodies alone. Nothing polices the copy because reproduction
already does: let it drift and books stop reproducing.

## `linkcheck.py` — four properties, whatever produced the links

```
python tools/linkcheck.py BOOK.epub [BOOK.epub …]     # nonzero if any fails
```

| | |
|---|---|
| `digits` | a marker printed 46 reaches an entry printed 46 |
| `span` | one chapter's markers occupy one stretch of the book |
| `order` | chapters appear in the body in the order their notes appear |
| `unreached` | every note printed in the back is reached by some marker |

None of them knows how the linkers work; they are what a reader could check
by hand. `span` is the one that catches mis-grouping, and it earns its place
on history: this pipeline once shipped 190 links that all resolved, all
counted, and every one of which pointed at the wrong chapter's note. A count
cannot see that. `tests/test_linkcheck.py` performs that injury on a healthy
book and fails if these checks stay quiet.

Run it on a staged book before installing, beside the size-and-count gate.
