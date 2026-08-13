# Verifying the plugin

## Automated (run these first)

```bash
python3 calibre-plugin/build.py --install
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_runner.py
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_plugin.py
```

`check_runner.py` — 16 checks, no plugin needed: environment scrubbing, a
foreign interpreter running under Calibre's 3.14, progress streaming during
the run, cancellation killing the whole process tree, failure surfacing the
child's message.

`check_plugin.py` — 21 checks against the installed ZIP: registration,
imports, resources, the worker driven with Calibre's own Queue/Event/Log
against the protocol stub — and, when this machine has a codicology
environment, **a real conversion end to end through the plugin's worker**:
resolve → doctor gate → convert with cache → verify → verdict. It also
sets the plugin's codicology path if unset, so the hand test below needs
no configuration.

## By hand — Phase 1's last mile

The automated checks prove everything except Calibre drawing the dialogs
and the bar. Once:

1. Quit Calibre completely and restart it (a plugin replaced while it runs
   is not picked up). **OCR PDF** should be on the main toolbar.
2. Preferences → Plugins → Codicology OCR → Customize → **Check
   environment**. Expect "Environment ready" naming tesseract and
   llama-server.
3. Select a book with a PDF and press **OCR PDF**. The pre-flight dialog
   shows page count, first-read-vs-cache cost, and the options. Start it.
4. Watch the jobs list: the bar should advance page by page through
   "Reading pages n/m", and double-clicking the job shows the pipeline's
   own log filling live.
5. When it finishes: the EPUB appears on the book, and a dialog reports
   verify's verdict — "no holes found", or the list of pages that need a
   look.
6. Cancel test: start another conversion and kill the job. Status bar says
   cancelled; `pgrep -fl llama-server` finds nothing left behind.

A first read of a real book on CPU is genuinely long — hours for hundreds
of pages. The spike book (2 pages, cached) finishes in under a minute and
exercises every part of the chain.

## Findings worth remembering

**Calibre 9.13 sets no `PYTHONHOME`/`PYTHONPATH`.** The variables that
actually break a child are `SSL_CERT_DIR`, `OPENSSL_MODULES`,
`OPENSSL_ENGINES`, `FONTCONFIG_*` — all aimed inside `Calibre.app`. A
shell-launched calibre-debug also inherits a full shell `PATH`, which
*hides* the Finder-launch problem; the runner backfills PATH either way.

**The event/log split earns its keep in practice.** A real surya run
interleaves HF Hub warnings and llama-server lifecycle lines on stderr
between the JSON events; "a JSON object with an `event` key or it is log
text" sorted them correctly on the first live run.

**Killing the process group is necessary.** Surya spawns llama-server;
cancelling only the process we hold would leave inference running.
Measured cost of a cancel against a SIGTERM-ignoring child: up to five
seconds (the SIGKILL grace) — the UI should not promise instant.

**A plugin cannot execute files inside its ZIP**, and `get_icons` is a
name injected by Calibre's loader, not an import — the reason runner.py
imports nothing from calibre and stays testable alone.
