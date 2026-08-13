# Phase 0 — what was proved, and the one thing left to click

Two risks gated the whole plan: can a foreign Python survive being spawned
from Calibre's, and does the job plumbing carry progress. Both are proved
below except Calibre's own drawing of the progress bar.

## Automated

```bash
python3 calibre-plugin/build.py --install
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_runner.py
/Applications/calibre.app/Contents/MacOS/calibre-debug -e calibre-plugin/spike/check_plugin.py
```

`check_runner.py` — 16 checks, no plugin needed. Environment scrubbing, a
3.10 interpreter running under Calibre's 3.14, progress streaming during the
run rather than buffered to the end, cancellation killing the process tree,
and a non-zero exit surfacing the child's own message.

`check_plugin.py` — 19 checks against the installed ZIP. Registration,
imports from inside the ZIP, resources pulled back out, and the worker driven
with Calibre's real `Queue` / `Event` / `Log` so everything except the
rendering is exercised.

## By hand — the last mile

1. Quit Calibre completely, then start it. A plugin added while it is
   running is not picked up.
2. **OCR PDF** should appear on the main toolbar with a blue page icon. If
   it is missing: Preferences → Toolbars & menus → The main toolbar, and
   add it.
3. Select any book and press it. Expect, in order:
   - a status-bar line naming the selected book's PDF, or `none`;
   - a job appearing in the jobs list, bottom right;
   - **the progress bar advancing in steps while it runs** — this is the
     thing being tested. It takes about ten seconds. A bar that jumps
     straight from 0% to 100% at the end means output buffering, which is
     the failure this spike exists to rule out;
   - a dialog saying the plumbing works.
4. Double-click the running job to open its log. It should be filling with
   `OCR'd 4/40 pages…` lines *while the job runs*, not after.
5. Press it again and hit **Cancel** in the jobs list. The job should stop
   within a second or two and the status bar should say cancelled — no
   error dialog, no orphaned `python3.10` in Activity Monitor.

Nothing here touches your library: the stub writes `~/codicology-spike.epub`
and no format is added to any book.

## Findings worth carrying into Phase 1

**Calibre sets no `PYTHONHOME` or `PYTHONPATH`.** That was the headline
worry and it was overstated. What it *does* set is more insidious:
`SSL_CERT_DIR`, `OPENSSL_MODULES`, `OPENSSL_ENGINES` and `FONTCONFIG_*`, all
pointing inside `Calibre.app`. Inherited, those would send `uv`'s package
downloads and surya's model fetches looking for certificates in Calibre's
bundle — an HTTPS failure a long way from its cause. `runner.SCRUB` removes
them.

**Killing the process group works, and is necessary.** The test spawns a
child that ignores `SIGTERM` and starts a helper of its own — the shape
surya takes when it spawns `llama-server`. Signalling only the process we
hold leaves both alive; signalling the group takes down all three.

**Cancellation costs up to five seconds** against a child that ignores
`SIGTERM`, which is the grace period before `SIGKILL`. Real codicology should
honour the signal and stop promptly, but the UI should not promise instant.

**A plugin cannot execute its own files.** `__file__` points inside the ZIP,
so `load_resources()` plus a temporary file is the only route to disk. Phase
1 drops this — the real pipeline is an executable in its own environment.

**`get_icons` is a bare injected name**, not an import. It is undefined
outside a running Calibre, which is why `runner.py` imports nothing from
calibre and stays testable on its own.
