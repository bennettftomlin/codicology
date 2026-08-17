# codicology

An EPUB conversion tool that takes a variety of inputs and uses a combination
of Surya and Tesseract to OCR these source files into well-rendered EPUBs.

Codicology is the study of books as physical objects. This pipeline earns the
name: it reasons about ink density, page geometry, printed folios, running
heads, footnote rules, and gathering structure — not just the text.

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

`--ocr` also accepts `easyocr`, `paddleocr`, `tesseract`, and `gcv`. Those
are residue. **Not one has ever been run against a real book here, and none
is tested.** Each returns flat text and nothing else, so a book built on one
would arrive with no figures, no furniture stripped, no headings, no
contents and no note links — silently, because nothing downstream checks.
They remain only because the seam they sit behind is honest. Treat them as
unimplemented.

### Tesseract is the witness, not a backend

It has no generator, so it cannot invent. Its silence on a page the model
read paragraphs from is evidence, and two checks rest entirely on it:

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

## What it does without being asked

- Pages rendered from a PDF are stored losslessly; figures are passed
  through from the source PDF at their own resolution rather than cut from
  a re-render, and ship as PNG or JPEG, whichever is smaller.
- Printed page numbers are read and audited against reading order; the
  EPUB carries a real page-list, with numbers the printer never inked
  restored only where the book's own arithmetic supplies them.
- Footnotes and endnotes are linked both directions — but only where the
  binding is certain. Back-of-book endnotes, chapter endnotes, symbol
  footnotes, and numbered same-page footnotes are distinguished by the
  book's own layout, and a marker that cannot be bound with certainty is
  left as printed. An unlinked superscript is the page as it always was; a
  wrong link is misinformation wearing the book's authority.
- A page whose OCR loops, or claims more words than its ink can account
  for, is re-read and then left empty rather than filled with invention —
  on born-digital sources it is restored from the publisher's own text.
- Surya's layout labels are put to work: a one-line digit block the layout
  invented over prose is verified against the ink and suppressed; an
  orphaned note marker that fits the page's marker sequence is re-attached
  as the superscript it was; a footnote region printed without its rule
  gets one; a book with no printed contents page falls back to the labeled
  heading hierarchy.
- Where the source is born-digital its text is authority; where it is a
  scan, its text layer is somebody else's OCR and serves only as a witness.

## Research conveniences, behind flags

Each of these is opt-in on the command line and a checkbox in the Calibre
plugin:

- `--link-notes` — footnotes and endnotes linked both directions (see
  above; the flag gates the whole family).
- `--link-citations` — in-text citations bound to the bibliography,
  driven *from* the bibliography, so a junk match is structurally
  impossible; author–date and Chicago note styles.
- `--link-index` — the index's printed page numbers become links, ranges
  and abbreviated forms (167–8) included, guarded by the book's own folio
  arithmetic.
- `--typography` — restores what the recogniser flattened: directional
  quotes, collapsed double spaces. Letters are never touched, spaced
  ellipses stay as the book set them, and born-digital pages are never
  overwritten.

## Where the readers disagree

`codicology adjudicate` re-reads every page with the witness engines
beside the shipped text. The ~0.6% of words that genuinely differ go
through a ladder: hyphenation and diacritics fold away first — they are
policy, not disagreement; the book's own recurring vocabulary settles
coinages, transliterations and proper nouns; a non-word loses to a word,
with the scholarly apparatus (ff., op. cit., ibid.) counted as words so a
book's conventions cannot lose to fluency; Apple Vision rereads the page
where present; and what nothing settles is an abstain — recorded, never
guessed. Measured on 2,591 truth-known disputes: lexicon 97.8% right,
Vision 95.0%, dictionary 88.4%. The record never changes the book.

`codicology review` renders that record as a single self-contained HTML
sheet: the ink cropped beside every disputed word (geometry recorded at
adjudication time, following hyphenated words across the line break),
rows ranked by how likely the shipped reading is wrong, convention rows —
the same pair repeating across a book — collapsed. Every row carries a
free-text field, because the reader's own eye outranks every rung and may
supply a reading no engine produced.

Decisions exported from the sheet apply two ways: `codicology apply`
corrects the EPUB in place (the original stays beside it as `.preapply`),
or the file sits beside the OCR cache and the next rebuild picks it up
via `convert --apply-decisions` — applied before the note, citation and
index linkers run, so a corrected word can still earn its link. Either
way, corrections land in text nodes only, at their recorded occurrence,
and a decision whose site no longer exists is reported stale and left
alone: a correction applied to the wrong site is worse than the misread
it meant to fix.

## The discipline

Run `codicology verify` after every build. It answers the one question no
run log answers: is anything missing? Every content loss this pipeline has
ever shipped — a page cached empty, a page eaten by the duplicate detector,
a text block swallowed by a drawn border — was invisible in the run log and
visible there.

`codicology compare` shows where our reading and the source's layer
disagree. On born-digital books every disagreement is our error. On scans
the layer is just another OCR, usually worse; matching word counts prove
completeness, never agreement.

## Tests

```bash
pip install -e ".[dev]"
pytest
```
