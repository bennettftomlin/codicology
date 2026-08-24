# codicology

Turns scanned or photographed books into EPUBs, using Surya to read the
page and Tesseract as a second reader to check it.

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

`--ocr` also accepts `easyocr`, `paddleocr`, `tesseract`, and `gcv`.
**Treat these as unimplemented.** None has been run against a real book
here and none is tested. Each returns flat text and nothing else, so a
book built on one would come out with no figures, no running heads
stripped, no headings, no contents and no note links — and nothing
downstream would report it. They exist only because the interface they
sit behind is generic.

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

## One command, seven verbs

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
documented under Install above.)

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

Both paths share the same cleanup: the page is found in the frame,
perspective-corrected, deskewed against its own text lines, and a
landscape spread is split at the gutter into two pages (each step has a
`--no-*` off switch, and `--rotate` handles sideways captures). Both also
share the same safety net — pages the duplicate pass was unsure about go
on an HTML review sheet (`--review-sheet`), and its verdicts feed
`--drop-pages` and `--swap` by stable run ids that survive re-extraction.
A page that came out blurred or occluded can be re-shot later and patched
in by name (`--patch r060p1=folio60.jpg`) without redoing the capture.

The `book.pdf` these paths produce is a facsimile of the physical book —
with `--pdf-text-layer`, one that searches and copies like a born-digital
file — and it feeds back into `--pages-from` for every later rebuild, so
the camera work is done exactly once.

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
  on every page of it. Chapter titles are lifted to the top of the heading
  outline so the EPUB has a usable document structure.
- **Bad pages are caught, not filled in.** A page whose OCR loops, or
  claims more words than its ink can account for, is re-read and then left
  empty rather than invented. On born-digital sources it is restored from
  the publisher's own text.
- **Layout labels are used, not just the text.** A one-line digit block
  the layout invented over prose is checked against the ink and dropped;
  an orphaned note marker that fits the page's sequence is re-attached as
  a superscript; a footnote region printed without its rule gets one.
- **Born-digital text is authority.** On a scan, the embedded text layer
  is somebody else's OCR — it is used as a witness, never as the answer.

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
way, corrections land in text nodes only, at their recorded
occurrence, and a decision whose site no longer exists is reported as
stale and left alone rather than applied somewhere approximate.

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
