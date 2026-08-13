"""
A stand-in for `codicology convert --progress-json`, for Phase 0.

It emits the protocol the plugin expects without needing surya, torch, or
an hour of OCR, so the plumbing can be proved before any of that exists.
Human-readable lines go to stdout exactly as the real pipeline writes them;
progress events go to stderr as JSON.

Run it with a DIFFERENT interpreter than Calibre's — that is the point.

    --pages-from PDF   --epub OUT   (accepted and echoed, not used)
    --progress-json    emit the event stream
    --pages N          how many pages to pretend to read
    --delay SECONDS    per batch
    --fail-at N        exit 1 after page N, with an error event
    --hang             ignore SIGTERM, to prove the process group is killed
    --spawn-child      fork a sleeper, to prove the whole tree dies
"""
import argparse
import json
import os
import subprocess
import sys
import time


def emit(**event):
    """One progress event, on stderr, flushed."""
    sys.stderr.write(json.dumps(event) + "\n")
    sys.stderr.flush()


def log(text):
    """A human line, on stdout. Deliberately NOT flushed — the real pipeline
    does not flush either, so this also tests PYTHONUNBUFFERED."""
    sys.stdout.write(text + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", nargs="?", default="convert")
    p.add_argument("--pages-from")
    p.add_argument("--epub")
    p.add_argument("--ocr-cache")
    p.add_argument("--title")
    p.add_argument("--progress-json", action="store_true")
    p.add_argument("--pages", type=int, default=48)
    p.add_argument("--delay", type=float, default=0.05)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--fail-at", type=int)
    p.add_argument("--hang", action="store_true")
    p.add_argument("--spawn-child", action="store_true")
    args = p.parse_args()

    if args.hang:
        import signal
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    child = None
    if args.spawn_child:
        # A stand-in for the llama-server surya spawns underneath itself.
        child = subprocess.Popen([sys.executable, "-c",
                                  "import time; time.sleep(600)"])
        log(f"  spawned helper pid {child.pid}")

    # Prove which interpreter actually ran, and that it is not Calibre's.
    log(f"  interpreter: {sys.executable}")
    log(f"  python: {sys.version.split()[0]}")
    emit(event="hello", executable=sys.executable,
         python=sys.version.split()[0], pid=os.getpid(),
         child_pid=(child.pid if child else None))

    total = args.pages
    emit(event="phase", phase="ocr", message="OCR backend: stub (batch size 4)")
    log(f"  OCR backend: stub (batch size {args.batch})")

    done = 0
    while done < total:
        time.sleep(args.delay)
        done = min(done + args.batch, total)
        if args.fail_at and done >= args.fail_at:
            emit(event="error", message="stub failed on purpose")
            log("  [!] stub failed on purpose")
            return 1
        note = "8 from cache" if done > total // 2 else ""
        emit(event="progress", phase="ocr", done=done, total=total, note=note)
        log(f"    OCR'd {done}/{total} pages…{(' (' + note + ')') if note else ''}")

    emit(event="phase", phase="folios",
         message="Reading printed page numbers…")
    log("  Reading printed page numbers…")
    time.sleep(args.delay)
    emit(event="progress", phase="folios", done=total, total=total)

    if args.epub:
        with open(args.epub, "w") as fh:
            fh.write("stub epub\n")
    emit(event="result", epub=args.epub, pages=total)
    log(f"  wrote {args.epub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
