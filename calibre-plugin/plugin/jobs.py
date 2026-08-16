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


def convert_worker(argv, verify_argv, adjudicate_argv, book_id, epub_path, title,
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

    def on_log(stream, text):
        if log is not None:
            log(text)

    put(0.01, "Starting codicology")
    result = runner.run(argv, on_progress=on_progress, on_log=on_log,
                        abort=abort)

    out = {"book_id": book_id, "epub": epub_path, "title": title,
           "result": result, "verify": None, "adjudicate": None}

    if verify_argv:
        put(0.97, "Checking the EPUB for holes")
        r = subprocess.run([str(a) for a in verify_argv],
                           capture_output=True, text=True,
                           env=runner.child_env(), timeout=600)
        text = ((r.stdout or "") + (r.stderr or "")).strip()
        if log is not None:
            for line in text.splitlines():
                log(line)
        # verify's exit status is a verdict, not a failure: 0 is clean,
        # 1 is "LOOK AT THIS". Anything else means it could not run.
        out["verify"] = {"rc": r.returncode, "output": text}

    if adjudicate_argv and not (abort is not None and abort.is_set()):
        # Slow on purpose: it re-reads the whole book with the witness
        # engines. The record is why — every word the readers disagreed on,
        # and which rule settled it — and it never changes the book.
        put(0.98, "Adjudicating: re-reading with the witness engines")
        r = subprocess.run([str(a) for a in adjudicate_argv],
                           capture_output=True, text=True,
                           env=runner.child_env(), timeout=3600)
        text = ((r.stdout or "") + (r.stderr or "")).strip()
        if log is not None:
            for line in text.splitlines():
                log(line)
        out["adjudicate"] = {"rc": r.returncode, "output": text}

    put(1.0, "Done")
    return out
