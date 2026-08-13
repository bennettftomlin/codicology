# codicology

An EPUB conversion tool that takes a variety of inputs and uses a combination
of Surya and Tesseract to OCR these source files into well-rendered EPUBs.

Codicology is the study of books as physical objects. This pipeline earns the
name: it reasons about ink density, page geometry, printed folios, running
heads, footnote rules, and gathering structure — not just the text.

## Install

```bash
pip install -e .
# plus an OCR backend, e.g.:
pip install -e ".[surya]"
```

Tesseract is optional but recommended (`brew install tesseract`): it is not
an OCR backend here but a second opinion — it cannot hallucinate, so its
silence on a page the model read paragraphs from is evidence, and its word
boxes position the searchable PDF text layer far more precisely than layout
blocks can.

## One command, three verbs

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
```

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
- Where the source is born-digital its text is authority; where it is a
  scan, its text layer is somebody else's OCR and serves only as a witness.

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
