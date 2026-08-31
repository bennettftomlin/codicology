"""
The worker that a ThreadedJob runs.

It receives complete argv lists rather than building them, which keeps it
testable outside a GUI: the spike checks drive it with a stub that speaks
the same --progress-json protocol, and the real pipeline is just different
argv.

Progress mapping: the pipeline reports each phase's own done/total, so a
bar fed those fractions raw would sweep to 100% twice. Each phase gets a
span of the bar instead, sized to how long it actually takes — OCR is the
long pole by far.
"""
import subprocess

from calibre_plugins.codicology_ocr import runner

SPANS = {
    "ocr": (0.03, 0.86),
    "folios": (0.86, 0.96),
}
LABELS = {
    "ocr": "Reading pages",
    "folios": "Reading printed page numbers",
}


def _stage(out, key, argv, put, log, progress_at, label, timeout):
    """One post-conversion stage, contained: the conversion has already
    succeeded by the time these run, and a stage that times out or dies
    must degrade into its own report line — never into a failed job that
    throws away a finished EPUB."""
    put(progress_at, label)
    try:
        r = subprocess.run([str(a) for a in argv],
                           capture_output=True, text=True,
                           env=runner.child_env(), timeout=timeout)
        text = ((r.stdout or "") + (r.stderr or "")).strip()
        rc = r.returncode
    except subprocess.TimeoutExpired as exc:
        text = (f"did not finish within {timeout}s and was stopped; "
                f"the book itself is unaffected")
        rc = -1
    except Exception as exc:
        text = f"could not run: {exc}"
        rc = -1
    if log is not None:
        for line in text.splitlines():
            log(line)
    out[key] = {"rc": rc, "output": text}


def convert_worker(argv, verify_argv, adjudicate_argv, review_argv,
                   book_id, epub_path, title,
                   abort=None, log=None, notifications=None):
    """
    Run a conversion, then optionally verify and adjudicate, and return
    what finished() needs:
    {'book_id', 'epub', 'title', 'result', 'verify', 'adjudicate'}.

    Raises runner.Aborted on cancellation and runner.Failed when the
    pipeline exits non-zero — ThreadedJob turns either into job.failed.
    """
    state = {"phase": "ocr"}

    def put(frac, msg):
        if notifications is not None:
            notifications.put((max(0.0, min(frac, 1.0)), msg))

    def on_progress(event):
        kind = event.get("event")
        if kind == "phase":
            state["phase"] = event.get("phase") or state["phase"]
            if event.get("message") and log is not None:
                log(event["message"])
        elif kind == "progress":
            phase = event.get("phase") or state["phase"]
            lo, hi = SPANS.get(phase, (0.03, 0.96))
            total = event.get("total") or 1
            frac = lo + (hi - lo) * min(event.get("done", 0) / total, 1.0)
            label = LABELS.get(phase, phase)
            note = event.get("note")
            put(frac, f"{label} {event.get('done')}/{total}"
                + (f" ({note})" if note else ""))
        elif kind == "error" and log is not None:
            log("ERROR: " + event.get("message", ""))

    # The build's own flags. The pipeline marks anything worth a human
    # eye with a leading [!] — a sparse contents parse, chapters the
    # running heads describe that the nav lacked, folios out of order —
    # and those lines used to live only in the job log, which nobody
    # opens unless something already failed. Collected here so the
    # completion dialog can show them.
    warnings = []

    def on_log(stream, text):
        for line in text.splitlines():
            flagged = line.strip()
            if flagged.startswith("[!]"):
                warnings.append(flagged[3:].strip())
        if log is not None:
            log(text)

    put(0.01, "Starting codicology")
    result = runner.run(argv, on_progress=on_progress, on_log=on_log,
                        abort=abort)

    out = {"book_id": book_id, "epub": epub_path, "title": title,
           "result": result, "verify": None, "adjudicate": None,
           "review": None, "warnings": warnings}

    if verify_argv:
        # verify's exit status is a verdict, not a failure: 0 is clean,
        # 1 is "LOOK AT THIS", 2 is verify itself crashing. Anything else
        # means it could not run.
        _stage(out, "verify", verify_argv, put, log,
               0.97, "Checking the EPUB for holes", 600)

    if adjudicate_argv and not (abort is not None and abort.is_set()):
        # Slow on purpose: it re-reads the whole book with the witness
        # engines. The record is why — every word the readers disagreed on,
        # and which rule settled it — and it never changes the book.
        _stage(out, "adjudicate", adjudicate_argv, put, log,
               0.98, "Adjudicating: re-reading with the witness engines",
               3600)

    if (review_argv and out["adjudicate"] is not None
            and out["adjudicate"]["rc"] == 0
            and not (abort is not None and abort.is_set())):
        # The record, rendered for a human's eyes: ink crops beside every
        # disputed word. Geometry was recorded during adjudication, so
        # this is mostly page rendering — minutes, not a re-read.
        _stage(out, "review", review_argv, put, log,
               0.99, "Rendering the review sheet", 1800)

    put(1.0, "Done")
    return out
