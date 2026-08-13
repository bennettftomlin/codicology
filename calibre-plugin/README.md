# Codicology OCR — a Calibre plugin

Select a book with a PDF, press **OCR PDF**, and get back a well-made EPUB
attached to the same record: OCR'd text, the book's printed page numbers,
notes linked both directions, figures at their own resolution — and a
completeness check run automatically afterwards to say whether anything went
missing.

> **Pre-alpha.** Phase 0 only. The button currently runs a stub that proves
> the job plumbing; it does not convert anything yet. See
> [VERIFY.md](VERIFY.md).

## Why it needs a separate install

The plugin is a thin client, about a thousand lines of Qt and process
plumbing. The [codicology](../) pipeline it drives is not bundled and cannot
be: it needs torch and OpenCV, and Calibre embeds Python 3.14, which those
have no wheels for. So codicology lives in its own environment and the
plugin drives it as a subprocess.

From Phase 2 the plugin will build that environment for you. Until then,
point it at an existing one in the plugin's preferences.

## Layout

```
plugin/      contents become the ZIP root — Calibre wants __init__.py and
             plugin-import-name-*.txt at the top level
spike/       Phase 0 proofs; not shipped in the ZIP
build.py     writes codicology-ocr.zip, --install adds it to Calibre
make_icon.py regenerates images/icon.png without needing Pillow
```

`runner.py` deliberately imports nothing from calibre, so it can be tested
outside a running Calibre — which is what makes the spike possible.

## Developing

```bash
python3 calibre-plugin/build.py --install
```

Then restart Calibre; a plugin added while it is running is not picked up.
Neither `calibre-customize` nor `calibre-debug` is on `PATH` on macOS —
they live in `/Applications/calibre.app/Contents/MacOS/`.

## The contract with codicology

The plugin depends on three things from the pipeline, all of which are part
of its interface rather than incidental output:

| What | Why |
|---|---|
| `--progress-json` | Line-delimited JSON events on stderr, human log on stdout. Scraping log text would break silently whenever a print statement was reworded. |
| `codicology doctor` | Reports whether the environment actually works — which inference backend answered, and whether tesseract is present. An import check cannot tell: surya imports cleanly without `llama-server` and fails on the first page. |
| `codicology verify` | Already exists. Exit status and `VERDICT:` line drive what the plugin reports after every build. |

The event stream:

```json
{"event": "phase",    "phase": "ocr", "message": "OCR backend: surya"}
{"event": "progress", "phase": "ocr", "done": 40, "total": 312, "note": "12 from cache"}
{"event": "result",   "epub": "/path/out.epub"}
{"event": "error",    "message": "..."}
```

A stderr line counts as an event only if it parses as a JSON object with an
`event` key; anything else is treated as log text, so warnings and
tracebacks pass through harmlessly.

## Requirements

Tesseract is **required**, not optional. It is the pipeline's witness against
fabrication — it cannot invent, so its silence on a page the model read
paragraphs from is the evidence that catches an invention. Its absence is
silent in the pipeline today; the plugin refuses to proceed without it.

On CPU and Apple Silicon surya needs `llama-server` from llama.cpp. The
NVIDIA path needs Docker and the NVIDIA Container Toolkit, which is more
than a plugin should install — so an NVIDIA user gets CPU-speed OCR unless
they configure vllm themselves, and the plugin says so rather than letting
the tool merely look slow.

## Licence

MIT, as codicology.
