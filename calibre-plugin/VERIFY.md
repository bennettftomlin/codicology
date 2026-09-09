# Verifying the plugin

## Automated (run these first)

```bash
python3 calibre-plugin/build.py --install
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_runner.py
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_plugin.py
```

`check_runner.py` — 25 checks, no plugin needed: environment scrubbing, a
foreign interpreter running under Calibre's 3.14, progress streaming during
the run, cancellation killing the whole process tree, failure surfacing the
child's message, and a post-conversion stage stopping on demand instead of
running to its own timeout.

`check_plugin.py` — checks against the installed ZIP: registration,
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
   is not picked up). **OCR PDF** should be on the main toolbar — but a
   CLI (re)install does not manage toolbar layout, so if it is missing,
   add it once: Preferences → Toolbars & menus → The main toolbar. It
   survives later reinstalls of the same plugin name after that.
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
7. Stop-then-start test: kill a running conversion, then immediately start
   another one. Within a few seconds it must start. A dialog headed
   **Still stopping** is correct while the child tree is coming down; a
   dialog headed **A conversion is already running** naming the book you
   just stopped is the wedge returning, and it used to persist until
   Calibre was quit and reopened.

A first read of a real book on CPU is genuinely long — hours for hundreds
of pages. The spike book (2 pages, cached) finishes in under a minute and
exercises every part of the chain.

## Findings worth remembering

**Calibre 9.13 sets no `PYTHONHOME`/`PYTHONPATH`.** The variables that
actually break a child are `SSL_CERT_DIR`, `OPENSSL_MODULES`,
`OPENSSL_ENGINES`, `FONTCONFIG_*` — all aimed inside `Calibre.app`.

**The Finder PATH problem is real, and shell testing hides it.** The first
live "Check environment" from the GUI reported tesseract and llama-server
missing on a machine that has both: a Finder-launched Calibre inherits
launchd's PATH — four entries, no Homebrew — while every automated check
had run under calibre-debug from a shell, inheriting the shell's full
PATH. The runner now appends the well-known binary directories
(`/opt/homebrew/bin`, `/usr/local/bin`) to the child PATH — appended, so
anything the user put on PATH still wins — and both check suites
reproduce the launchd PATH explicitly. The GUI's own **Check
environment** button remains the only test that runs in the true Finder
context; trust it over a shell run when they disagree.

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

**Stopping a job does not call the completion callback.** `ThreadedJob`
says so in its own docstring — "it is not called if the user kills the
job" — and `kill()` compounds it: `_cleanup()` sets `self.log = None`, so
when the worker thread finally raises `Aborted`, `start_work`'s handler
runs `assert log is not None` and the thread dies there, before the
callback on any path. Anything the plugin releases only in `finished()` is
therefore held for the life of the session. The one-book-at-a-time guard
was released there, so one Stop made every later conversion impossible
until Calibre restarted. It is released in the worker thread's `finally`
now, which is also the honest moment: `runner.run` has by then taken the
whole child tree down, so the inference server really is free.

**`subprocess.run` cannot be stopped.** The post-conversion stages —
verify, adjudicate, review — used it, so a stage already under way when
Stop was pressed ran on to its own ceiling, an hour for the adjudicator,
holding the guard the whole time. `runner.run_captured` polls the abort
while it waits and kills the stage's process group, and a stopped stage
degrades to its own report line rather than failing the job.
