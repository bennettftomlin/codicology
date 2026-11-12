# codicology

Turns scanned or photographed books into EPUBs, using Surya to read the
page and Tesseract as a witness to check it.

Input is a PDF, a folder of photographs, or video of someone turning
pages. Output is an EPUB, and optionally a searchable PDF beside it.

Codicology is the study of books as physical objects, and the pipeline
works the same way: it uses the page's geometry — ink density, printed
folios, running heads, footnote rules — and not only the recognised text.
That is what lets it strip running heads, keep figures, link notes to
their markers, and publish the printed page numbers.

## Install

```bash
pip install -e ".[surya]"
```

Surya pins a few of its dependencies and leaves the rest open
(`transformers` has a floor and no ceiling), so the same command a month
later is a different environment. `constraints.txt` records the versions
this was last built and tested with:

```bash
pip install -c constraints.txt -e ".[surya]"
```

`python tools/lock_env.py` rewrites it from an environment that works.

Then install Tesseract, which pip cannot do for you:

```bash
brew install tesseract
```

**Both are required.**

Ask the environment whether it is actually ready — `--full` reads one
synthetic page through the real backend, which is the only check that
catches a surya that imports cleanly but cannot serve inference:

```bash
codicology doctor --full
```

### Surya is the OCR backend

This pipeline was built around Surya and only Surya. Everything it does
beyond reading characters — figures, captions, running heads, headings,
reading order, block geometry — depends on the layout Surya returns.

### Tesseract is the witness, not a backend

Tesseract has no language model behind it, so it cannot invent text. That
makes its silence meaningful: if it reads nothing where Surya reported
paragraphs, something is wrong. Two checks depend on it entirely:

- whether a page with **no text layer to consult** was invented — on
  photographs, video, and image-only scans it is the *only* witness;
- whether a page about to be deleted as **blank** was merely one the
  pipeline failed to read. Nothing else can tell those apart, and that check
  applies to every source.

Without it, both checks pass without examining anything, so a run refuses to
start. `--without-witness` overrides that and reports at the end how many
pages went unexamined. Its word boxes also position the searchable PDF text
layer far more precisely than layout blocks can.

### A second reader, where one is installed

Surya is a generative model, and on some pages it writes what is likely
where it should write what is there: a figure dropped from a fraction, a
table that loses three of its six columns. Tesseract cannot check those. It
does not read mathematics, and it reads a table as loose numbers.

`--second-reader` asks a generative reader of another family,
[LightOnOCR-2](https://huggingface.co/lightonai/LightOnOCR-2-1B), served by
the same llama-server. It is not part of this package and nothing depends
on it: put the model's two `.gguf` files (the model and its projector,
about 1.1 GB at Q8, from
[ggml-org/LightOnOCR-2-1B-GGUF](https://huggingface.co/ggml-org/LightOnOCR-2-1B-GGUF))
in `~/.cache/codicology/second/` and `codicology doctor` reports it.

It has no layout to give, so it replaces nothing wholesale, and it can
invent too, so its word alone settles very little. What it is used for:

- **Tables.** On a typeset page, a table that holds under 98% of the
  figures the PDF prints in its place is put to the second reader, and its
  table is printed where it holds nearly all of them and clearly more than
  the first. On the 26 typeset pages of one shelf whose tables had lost the
  most, Surya's reading held 78% of the 1,662 figures the PDF prints and
  the second reader's 91%; six tables were replaced, none with a figure
  the PDF does not print.
- **Equations set on their own lines.** On a typeset page the PDF's own
  symbols choose between the two readings. On a scan nothing can, so the
  second reader's equation is taken only where the two differ in digits
  alone and it has no fewer of them (`2/100` for a printed `20/100`: what
  Surya does to a figure is drop it); every other difference is listed in
  the build log, and in a file beside the cache, with both readings, and
  the first is kept. Tried on 118 pages with 247 equations, that changed
  three, each rightly, and listed fourteen.
- **Disputed words**, in `adjudicate --second-reader`: see below.

On a scan that leaves most differences listed and not settled. A third
reader settles them where one is installed:
[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR), its two `.gguf` files
(about 1.4 GB at Q8, from
[ggml-org/GLM-OCR-GGUF](https://huggingface.co/ggml-org/GLM-OCR-GGUF)) in
`~/.cache/codicology/third/`. It reads regions, not pages, so it is shown
the one block the first two readers differ on and asked for the formula,
which takes under a second. The equation printed is then the reading that
has another reader on its side at every point where it differs: the
second's, the third's own, or the first's as it stood. On the fourteen
equations of Sraffa's *Production of Commodities* on which the first two
readers differ, that is the page in twelve (eleven changed, one kept), and
the other two stay as the first reader had them, both still wrong, because
two readers misread the same mark. A reading with fewer figures than the
first reader's is never printed on a vote. It has no flag of its own: it
is asked with `--second-reader`, and `codicology doctor` says whether it
is there.

It reads only the pages it is asked about, about twelve seconds each on an
M1 Pro, and its readings are kept beside the OCR cache, so a rebuild does
not read them again.

## One command, thirteen verbs

```bash
# build — from a PDF (the usual case), photographs, or video.
# Always pass --ocr-cache: a rebuild then costs minutes, not hours,
# and every fix means a rebuild.
codicology convert --pages-from book.pdf --epub book.epub \
    --ocr-cache book.ocr.gz --check-folios --link-notes --title "Book"

# check — after every build: is anything missing?
codicology verify book.epub book.pdf

# audit — word-by-word disagreement with the source's own text layer
codicology compare book.epub book.pdf

# dispute — re-read the book with the witness engines; emit the record
# of every word they disagreed on. Never changes the book
codicology adjudicate book.epub book.pdf --report book-disputes.json

# review — the record rendered for human eyes: the ink beside every
# disputed word, and a field for your own reading
codicology review book-disputes.json

# apply — your exported decisions, fed back into the book
codicology apply book.epub book-disputes.decisions.json
```

(`codicology doctor` is the seventh — it checks the environment, and is
documented under Install above. The last two look after an OCR cache
without reading a word of the book again: `codicology relabel CACHE PDF`
backfills layout labels into a cache written before they were kept, and
`codicology rekey CACHE PDF` refiles one written before a page's reading
was named for the page's pixels. A build does that refiling for itself the
first time it meets such a cache; `rekey` brings a whole cache over ahead
of time. `codicology decisions` looks after decisions files; it is
described under the review sheet below. `codicology bind`, `unbind` and
`rebind` change how a built book's files are cut and nothing else; see
"Pages, bound into chapters" below.)

The cache is what makes a rebuild cheap, and it holds hours of reading, so
it is keyed on the page and nothing else: a rendered page by its pixels, a
page lifted out of the PDF as a JPEG by its bytes. The encoder that writes
the working page files can change without a single page being read again.

`codicology convert -h` documents the full flag set: page dropping and
swapping by stable run ids, review sheets for human adjudication, cover
selection, OCR backend and language choice, and the geometry knobs for
video extraction.

## Books that only exist on paper

The PDF path is the usual case, but the pipeline began life digitizing
books nobody had scanned, and those inputs remain first-class.

**From photographs** — the best capture path for physical books:

```bash
codicology convert --from-images ./shots book.pdf --epub book.epub \
    --ocr-cache book.ocr.gz
```

Point it at a folder or glob and the shots are read in filename order.
Consecutive shots of the same page are grouped and the best one kept, so
interval or burst shooting works as-is — photograph every page twice and
let the pipeline choose. Stills carry far more detail than video frames,
which is what makes small type readable. iPhone HEIC files are handled
directly (via `pillow-heif`), alongside JPEG, PNG, TIFF, and WebP.

**From video** — film the book page by page:

```bash
codicology convert recording.mp4 book.pdf --pdf-text-layer \
    --epub book.epub --ocr-cache book.ocr.gz
```

Pages are told apart from half-finished turns by stillness: a page must be
held steady for a few frames to count, and the motion threshold is chosen
from the footage itself. Hold each page flat for a second or two at a
steady pace, use even lighting, and a dark background helps detection.

Both paths share the same cleanup: a document rectifier (UVDoc, served
through transformers; `pip install 'codicology[geometry]'`) flattens the
whole photographed sheet in one pass — perspective, the curl into the
gutter and the crease at the spine come out of a single learned map —
then a landscape sheet is split at its ink-free gutter, each page is
trimmed back to its paper and levelled against its own text lines.
Tesseract reads the flat sheet against the photograph, and a sheet that
lost most of the photograph's words is refused and named, never shipped
quietly. Each step has a `--no-*` off switch, and `--rotate` handles
sideways captures. Both also
share the same safety net — pages the duplicate pass was unsure about go
on an HTML review sheet (`--review-sheet`), and its verdicts feed
`--drop-pages` and `--swap` by stable run ids that survive re-extraction.
A page that came out blurred or occluded can be re-shot later and patched
in by name (`--patch r060p1=folio60.jpg`) without redoing the capture.

Either path also works with no OCR at all — point it at the capture and
name only the PDF, and the sheets are flattened, the spreads split, and a
corrected facsimile is saved in minutes, no text engine ever loaded. For
a book worth preserving before it is worth reading by machine, that is
the whole job:

```bash
codicology convert --from-images ./shots book.pdf --rotate 270
```

The `book.pdf` these paths produce is a facsimile of the physical book —
with `--pdf-text-layer`, one that searches and copies like a born-digital
file — and it feeds back into `--pages-from` for every later rebuild, so
the camera work is done exactly once. Pages loaded that way are left
geometrically untouched: they have been rectified already, and flattening
them again would drift. A third-party scan that visibly leans or curls
can opt in with `--dewarp-scans`.

## What every build does

No flags required for any of this.

- **Images.** Pages are stored losslessly. Figures are taken from the
  source PDF at their own resolution rather than cut out of a re-render,
  and ship as PNG or JPEG, whichever is smaller.
- **Page numbers.** Printed folios are read, checked against reading
  order, and published as a real EPUB page-list, so a citation to the
  print edition still resolves.
- **Notes, linked both ways — where the binding is certain.** Back-of-book
  endnotes, per-chapter endnotes, symbol footnotes and numbered same-page
  footnotes are told apart by the book's own layout. A marker that cannot
  be bound confidently is left as printed, on the view that no link beats
  a wrong one.
- **Structure.** Chapters come from the printed contents where it parses,
  and otherwise from the book's own running heads, which name the chapter
  on every page of it. The two are also checked against each other: a
  contents parse that missed half of what the heads describe has its
  missing chapters merged back in, and a parse that looks wrong — sparse
  for the page count, gaps in its chapter numbers, entries that never
  placed — says so in the build log. Chapter titles are lifted to the top
  of the heading outline so the EPUB has a usable document structure.
- **Bad pages are caught, not filled in.** A page whose OCR loops, or
  claims more words than its ink can account for, is re-read and then left
  empty rather than invented. On born-digital sources it is restored from
  the publisher's own text. A table that does nothing but count — the
  other thing a recogniser makes of a page it cannot read — is left out
  of the page, unless the page's text layer or tesseract finds the count
  printed there.
- **Layout labels are used, not just the text.** A one-line digit block
  the layout invented over prose is checked against the ink and dropped;
  an orphaned note marker that fits the page's sequence is re-attached as
  a superscript; a footnote region printed without its rule gets one.
- **Born-digital text is authority.** On a scan, the embedded text layer
  is somebody else's OCR — it is used as a witness, never as the answer.
  On a page the publisher set, a word the reading got wrong is exchanged
  for the publisher's, and a word it passed over is put back: a doubled
  word written once, the second of two like words taken for the first, a
  line the eye dropped, the "1" of "1 ½ tablets". Only where the PDF sets
  the word in the line between words both agree on, and the page does not
  have it somewhere else; each one is named in the build log.
- **Pages are bound into chapters.** See the next section.

## Pages, bound into chapters

The build works a page at a time, and so does everything that checks it:
page 212 of the book is page 212 of the PDF, which is what lets a reading
be held against its ink. Written out that way, one file to a page, a
reading system starts a new screen at every printed page, and a printed
page ends where the paper ran out. On one shelf of 152 books that is the
middle of a sentence at about half of all page turns.

So the last thing `convert` does is bind the pages into the files a reader
turns through:

- **A file is a chapter.** It begins where an entry of the book's contents
  does. A chapter longer than 60 pages, or a book with no contents, is cut
  where a page ends its paragraph, and never anywhere else.
- **A paragraph that runs over the page is one paragraph.** A page that
  stops without a full stop, taken up by a small letter, is joined to what
  takes it up. Taken up by a capital it is joined only where the paragraph
  is prose that stops on a small word or a comma ("…banned all trade
  except in | Canton"). Lines are not sentences: an index, verse, a
  caption, a running head that slipped through, a signature mark at the
  foot of the page are left as they stand.
- **What stood in the way waits.** The footnotes of the page the paragraph
  began on, a figure set at its foot or at the head of the next, come
  after the paragraph instead of in the middle of it.
- **A word the printer broke at the foot of the page is one word** where
  its two halves make a word of the dictionary's or of the book's own
  ("con-" | "centration"). A compound that happened to break at its hyphen
  ("self-" | "control") keeps it. Either way the page begins after the
  whole word: a page's mark, and the printed page number with it, never
  stands inside a word.

Nothing is read again and no word changes. Every page's start is marked in
the text (`id="page-0212"`), the printed page number stays exactly where
its page begins, in the middle of a paragraph if that is where it falls,
and the page list, the contents, the notes, the index and the citations
all lead where they led. Because the marks are there, a bound book comes
apart into the same pages: `verify`, `compare`, `adjudicate`, `review` and
`apply` read a book either way, a decisions file recorded against one
layout applies to the other, and the place Calibre's viewer remembers
inside the book is carried to the page it was on.

```bash
codicology bind book.epub      # a book built before this, or with --page-files
codicology unbind book.epub    # one file to a printed page again
codicology rebind book.epub    # a bound book, bound again by the rules as they now are
```

`convert --page-files` leaves the book as it is built. A book the binding
will not answer for (its pages would not come apart again exactly, or one
of them is a file no strict reader would open) is left a file to a page,
with a line in the log to say why.

## Research conveniences, behind flags

Each is off by default: a flag on the command line, a checkbox in the
Calibre plugin.

- `--link-notes` — footnotes and endnotes linked both directions (see
  above; the flag gates the whole family).
- `--link-citations` — in-text citations linked to the bibliography.
  The prose is searched only for the names and years the bibliography
  actually lists, so a stray "(New York, 1933)" is never a candidate.
  Author–date and Chicago note styles.
- `--link-index` — the index's printed page numbers become links, ranges
  and abbreviated forms (167–8) included, guarded by the book's own folio
  arithmetic.
- `--typography` — restores what the recogniser flattened: directional
  quotes, collapsed double spaces. Letters are never touched, spaced
  ellipses stay as the book set them, and born-digital pages are never
  overwritten.
- `--second-reader` — a second opinion on tables and equations, where a
  second reader is installed (see above). The one flag here that can change
  what a table or an equation says; each change is in the build log with
  what it was.

## Where the readers disagree

`codicology adjudicate` re-reads every page with the witness engines and
compares them to the shipped text. About 0.6% of words genuinely differ.
Those go through a ladder, in order:

1. **Fold.** Line-break hyphenation and diacritics are transcription
   policy, not disagreement, so they are normalised away first.
2. **The book's own vocabulary.** A word the engines agree on elsewhere in
   this book is a word of this book — which is how coinages,
   transliterations and proper nouns get settled without a dictionary
   voting for the standard spelling.
3. **The dictionary.** A non-word loses to a word. Scholarly apparatus
   (ff., op. cit., ibid.) counts as words, so a book's own conventions are
   not corrected into ordinary prose.
4. **Apple Vision**, where it is available, re-reads the page.
5. **Abstain** — recorded as unresolved rather than guessed. Measured against 2,545 disputes whose answer is known, from six
born-digital books read both clean and artificially degraded: the book's
own vocabulary is right 97.8% of the time, Vision 94.8%, the dictionary
92.3%. Adjudication only ever writes a report; it never edits the book.

With `--second-reader`, the second reader is one more reader of the same
ink, and it changes two things. A dispute the ladder left open is settled
as a count of readers: Vision or the second reader with a view makes two
against one, both agreeing makes three against one, and the two of them on
opposite sides leaves it open. And a verdict of the vocabulary or the
dictionary that the second reader reads the other way is withdrawn and
left open, because those two rule on what a word is and not on what the
page says. Measured on 136 disputes from 207 born-digital pages, between
Surya as it first read them and Tesseract: of the 47 the ladder left open,
the count of readers settled 31, none wrongly (two could not be judged),
where Vision alone settled 34 and four of them wrongly; and the second
reader contradicted ten verdicts of the vocabulary and the dictionary, of
which nine were wrong and the tenth could not be judged. On a scan of 122
pages with 313 disputes, the ladder had proposed twelve changes to the
book and with the second reader proposed five: the seven that went were
each a misreading of damaged type offered for a right word ("barm" for
harm, "caling" for calling), and the four real errors among the twelve
were all still proposed. It reads every page that has a disputed word,
which on that scan was 105 of them and took 22 minutes.

`codicology review` renders that record as a single self-contained HTML
sheet: the ink cropped beside every disputed word (geometry recorded at
adjudication time, following hyphenated words across the line break),
rows ranked by how likely the shipped reading is to be wrong, and
repeated pairs — the same disagreement recurring across a book, which is
usually a convention rather than damage — collapsed into one row. Every
row has a free-text field: you can enter a reading no engine produced,
and it wins.

Decisions exported from the sheet apply two ways: `codicology apply`
corrects the EPUB in place (the original stays beside it as `.preapply`),
or the file sits beside the OCR cache and the next rebuild picks it up
via `convert --apply-decisions` — applied before the note, citation and
index linkers run, so a corrected word can still earn its link. Either
way, corrections land in the page's text only, on the one word each
decision is addressed to: the sheet records which of the page's
instances of the word was in dispute and the words either side of it,
and the apply step finds it by those. A decision that cannot be placed
with certainty — the word is gone, its neighbours have changed, two
places fit equally — is reported as stale and left alone rather than
applied somewhere approximate.

Decisions exported from a sheet made before 2026-11-10 carry no
neighbours, and their `occurrence` counted only the page's disputed
words. They are applied where that cannot matter (the file accounts for
every instance of the word on the page) and reported stale where it can;
adjudicate the book again for a sheet whose decisions are addressed.

Review happens in rounds: correct, rebuild, adjudicate again, correct what
is left. The second round's decisions are addressed to the text the first
round left, and the cache still holds the reading as it was first made, so
a rebuild has to apply every round, in order. `codicology decisions file
exported.json book-disputes.decisions.json` keeps them apart: the same
sheet exported again replaces the round in hand, a file from a later sheet
becomes a new round, and the earlier ones stay beside it as
`….decisions.r1.json`, `….r2.json`. `convert --apply-decisions` given the
round in hand applies the earlier ones first. `codicology decisions list`
prints a file a row to a line, and `codicology decisions retire FILE N`
takes a row out of use without forgetting it: a retired row is never
applied, and no later round can bring it back.

## Checking a build

Run `codicology verify` afterwards. A run log tells you what happened; it
does not tell you what went missing. Every content loss this project has
had — a page cached empty, a page removed by the duplicate detector, a
text block swallowed by a drawn border — looked fine in the log and showed
up in `verify`.

`codicology compare` shows where our reading and the source's own text
layer disagree. On a born-digital book, a disagreement is our error. On a
scan, that layer is just another OCR pass and usually a worse one, so
matching word counts show completeness, not correctness.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
