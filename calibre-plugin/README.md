# Codicology OCR — a Calibre plugin

Select a book with a PDF, press **OCR PDF**, and get back a well-made EPUB
attached to the same record: OCR'd text, the book's printed page numbers,
notes linked both directions, figures at their own resolution — and
`codicology verify` run automatically afterwards to say whether anything
went missing.

> **Phase 1.** Real conversions, one book at a time, with live progress,
> working cancel, and the verify verdict surfaced. The environment is
> supplied by hand (below); the wizard that builds one arrives in Phase 2,
> and batch queueing with it.

## Why it needs a separate install

The plugin is a thin client — Qt and process plumbing, ~14 KB installed.
The [codicology](../) pipeline it drives is not bundled and cannot be: it
needs torch and OpenCV, and Calibre embeds Python 3.14, which those have
no wheels for. So codicology lives in its own environment and the plugin
drives it as a subprocess with a scrubbed environment (Calibre points
`SSL_CERT_DIR`, `OPENSSL_*` and `FONTCONFIG_*` into its own bundle; a
child that inherits those fails far from the cause).

## Pointing the plugin at codicology

```bash
python3.12 -m venv ~/codicology-env          # any Python 3.10–3.13
~/codicology-env/bin/pip install -e '/path/to/codicology[surya]'
brew install tesseract llama.cpp             # the witness, and inference
~/codicology-env/bin/codicology doctor --full   # proves it end to end
```

Then Preferences → Plugins → Codicology OCR → Customize, set the path to
`~/codicology-env/bin/codicology`, and press **Check environment**. If the
path is left empty the plugin probes `~/.local/bin`, `/opt/homebrew/bin`,
`/usr/local/bin` and `~/.pyenv/shims`.

## What a run looks like

1. Select a book that has a PDF format → **OCR PDF**.
2. Pre-flight: page count and size, whether an OCR cache already exists
   (a rebuild then costs minutes, not hours), doctor's warnings if any,
   and the options — languages, note linking, folio checking, embedded
   scans, blank pages, plus a free-text field for any other convert flag.
3. The job runs in Calibre's jobs list: page-by-page progress, a live
   log, and Cancel that actually kills the OCR process tree.
4. On success the EPUB is added to the book and verify's verdict is
   shown — quietly when clean, loudly with the list of holes when not.

## Layout

```
plugin/      contents become the ZIP root — Calibre wants __init__.py and
             plugin-import-name-*.txt at the top level
spike/       automated checks run under calibre-debug; not shipped
build.py     writes codicology-ocr.zip; --install adds it to Calibre
make_icon.py regenerates images/icon.png without needing Pillow
```

`runner.py` imports nothing from calibre, so the process-handling half is
testable outside a running Calibre — which is what the spike checks do.

## Developing

```bash
python3 calibre-plugin/build.py --install
```

Then restart Calibre; a plugin added while it is running is not picked up.
Neither `calibre-customize` nor `calibre-debug` is on `PATH` on macOS —
they live in `/Applications/calibre.app/Contents/MacOS/`.

Checks (35 runner/plugin assertions, including a live conversion when the
machine has an environment):

```bash
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_runner.py
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_plugin.py
```

## The contract with codicology

The plugin depends on three pipeline features, all part of its interface
rather than incidental output:

| What | Why |
|---|---|
| `convert --progress-json` | One JSON object per line on stderr; the human log stays on stdout. Scraping log text would break silently whenever a print was reworded. |
| `doctor [--full] --json` | Quick mode is the gate before every run: binaries, packages, tesseract's languages, in under a second. `--full` OCRs a synthetic page — the only check that catches a surya that imports cleanly but cannot serve inference (0.2x without llama-server). |
| `verify` | Exit 0 is clean, 1 is "LOOK AT THIS", anything else means the check itself failed. The verdict and detail are shown after every build. |

The event stream:

```json
{"event": "phase",    "phase": "ocr", "message": "OCR backend: surya"}
{"event": "progress", "phase": "ocr", "done": 40, "total": 312, "note": "12 from cache"}
{"event": "result",   "epub": "/path/out.epub"}
{"event": "error",    "message": "..."}
```

A stderr line counts as an event only if it parses as a JSON object with an
`event` key; anything else is log text. This is load-bearing: surya's own
stderr chatter (HF Hub warnings, llama-server lifecycle lines) shares the
stream in practice.

OCR caches live at `<calibre config>/plugins/codicology_ocr/cache/
<library>/<book_id>-<pdfhash8>.ocr.gz` — hashed so a replaced PDF format
cannot silently reuse OCR read from the file it replaced.

## Requirements, honestly

- **tesseract is required.** It is the pipeline's witness against
  fabrication, and `convert` itself refuses to start without it. Doctor
  says so before a job is ever queued.
- **On CPU and Apple Silicon, surya needs `llama-server`** (llama.cpp).
  The NVIDIA path needs Docker plus the NVIDIA Container Toolkit — more
  than a plugin should install — so an NVIDIA user gets CPU-speed OCR
  unless they configure vllm themselves, and the pre-flight dialog says
  so rather than letting the tool merely look slow.
- Surya is pinned (`==0.22.1`): its API has broken across minor versions,
  and v0.20.0 made inference delegation invisible to import checks.

## Licence

MIT, as codicology.
