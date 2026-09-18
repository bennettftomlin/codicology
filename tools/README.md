# tools

Three instruments, none of which reads a PDF or runs OCR.

Two judge a change to the note linkers before it ships, working on
EPUBs that already exist. The third builds an EPUB from a source that
never went near a camera.

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

## `pga2epub.py` — born-digital HTML, made into a book

The other two tools judge EPUBs this pipeline built from photographs.
This one builds an EPUB from a source that never went near a camera:
the HTML editions at Project Gutenberg Australia.

```
python tools/pga2epub.py inspect SOURCE [--json]   # read it, build nothing
python tools/pga2epub.py build BOOK.json [-o OUT.epub]
```

`inspect` takes a URL or a local file and reports what a conversion
would have to deal with; `--json` prints a starter book file. `build`
takes that file, fetches the page and its illustrations, and writes an
EPUB 3 — one file per chapter, a navigation document, notes bound in
both directions, a stylesheet that forces no colour or typeface, and a
typographic cover made from one of the book's own plates.

### The encoding, which is the whole job

Those pages declare `charset=windows-1252` and lie. Their accented
letters are UTF-8 bytes that were decoded as cp1252 and then
entity-encoded, so an e-acute is stored as `&Atilde;&copy;`. Reversing
that recovers the text exactly.

**But not every book there is mojibake.** Of eight sampled, one was; the
other seven carried real em dashes, a real e-acute, a real pound sign,
and reversing any of them would have turned every such character into an
invalid byte. So the reversal is not unconditional: `mojibake_score`
counts the pairs that read as UTF-8 through cp1252, `inspect` prints the
verdict and the count it rests on, and a book file can overrule it.

Two kinds of damage survive the reversal, and the second is the one that
bites:

| | |
|---|---|
| **gone** | the source has the first half of a pair and nothing after it. The bytes are invalid UTF-8, `inspect` points at them, and `demojibake` leaves each as its own escape (`\xc3`) so a repair can name it as plain text |
| **wrong but valid** | in one book `&mdash;` (cp1252 0x97) stood where `&ndash;` (0x96) belonged, so a place name read as `×hningen` six times and decoded without a murmur |

Nothing can find the second kind but reading every non-ASCII character
in the book, which is why `inspect` prints all of them with their
context — about a hundred in a 90,000-word novel, a few minutes' work.

Repairs live in the book file, and each states how many places it must
match. A repair that matches a different number stops the build, so a
source revision cannot quietly change what a repair does.

### Structure, and what this will not guess

The markup is not consistent between books: one is a novel in
`div.chapter` blocks nested in `div#bookN` parts, the next is a short
story with no structural markup at all. Two readers, both exercised
against real books:

| | |
|---|---|
| `divs` | `div.chapter` blocks, optionally inside `div#book*` parts |
| `flow` | one heading level splits the body; `flow_starts` names the first real chapter, because that heading level is usually also the byline's |

`inspect` says which fits, and says `none` rather than picking one. The
answer then is to write a reader against the book in front of you, not
to widen one of these until it swallows everything.

### The checks, which run on every build

No flag turns them on and a failure deletes the output:

| | |
|---|---|
| `chapters` | every chapter, character for character, against the text captured off the parse tree before anything was cleaned or moved |
| `coverage` | every block of the source reaches a chapter, a part page, or an `expect_dropped` entry that names it |
| `links` | every internal href resolves to a file and, where it has a fragment, to an id |
| `digits` | the number a marker prints is the number of the note it reaches |
| `wellformed` | every XHTML, OPF and NCX file parses as XML |

`coverage` is the one that earns its place. A word-frequency comparison
passed a build of the first book while a paragraph and a half of one
chapter was missing — a stray `</div>` had closed that chapter early, so
the rest of it sat outside the chapter div, and the lost words were
ordinary ones the book used elsewhere. Counting could not see it.
`absorb_strays` now puts those blocks back; `coverage` is what would
have found them.
