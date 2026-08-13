"""
Phase 0, risk 1: does a foreign Python survive being spawned from Calibre's?

Run this INSIDE Calibre's interpreter, where the answer actually matters:

    /Applications/calibre.app/Contents/MacOS/calibre-debug \
        -e calibre-plugin/spike/check_runner.py

It needs no GUI and no library. Risk 2 — ThreadedJob progress plumbing —
needs a click, and check_plugin.md says where to click.
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "plugin"))

import runner  # noqa: E402

STUB = os.path.join(HERE, "stub_codicology.py")
results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")


def foreign_python():
    """An interpreter that is not Calibre's, and not one torch cannot use."""
    seen = []
    for base in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        for minor in (12, 11, 13, 10):
            path = f"{base}/python3.{minor}"
            if not os.path.exists(path):
                continue
            # /usr/local on Apple Silicon is the Rosetta prefix; an x86_64
            # interpreter would run torch under emulation with no MPS.
            import subprocess
            arch = subprocess.run([path, "-c", "import platform;print(platform.machine())"],
                                  capture_output=True, text=True).stdout.strip()
            seen.append(f"{path} ({arch})")
            if arch == os.uname().machine:
                return path, seen
    return None, seen


def main():
    print(f"\nCalibre interpreter : {sys.version.split()[0]}")
    print(f"Calibre prefix      : {sys.prefix}\n")

    py, seen = foreign_python()
    if not py:
        print("No usable foreign interpreter found; saw: " + ", ".join(seen))
        return 1
    print(f"Foreign interpreter : {py}\n")

    # ── 1. the environment Calibre aims at itself is stripped ───────────
    print("1. Environment scrubbing")
    present = runner.scrubbed()
    env = runner.child_env()
    leaked = [k for k in runner.SCRUB if k in env]
    check("Calibre's own variables are removed from the child environment",
          not leaked, f"stripped {len(present)}: {', '.join(present) or 'none set'}")
    check("PYTHONUNBUFFERED is set", env.get("PYTHONUNBUFFERED") == "1")
    check("PATH survives and is usable", len(env["PATH"].split(os.pathsep)) >= 3)
    check("a prepended directory wins", runner.child_env(
        path_prepend=["/tmp/native/bin"])["PATH"].startswith("/tmp/native/bin"))

    # ── 2. a foreign interpreter actually runs ──────────────────────────
    print("\n2. Foreign interpreter under Calibre")
    events, logs, stamps = [], [], []
    started = time.time()
    out = runner.run(
        [py, STUB, "convert", "--progress-json", "--epub", "/tmp/spike.epub",
         "--pages", "24", "--delay", "0.06"],
        on_progress=lambda e: (events.append(e), stamps.append(time.time())),
        on_log=lambda s, t: logs.append(t))
    elapsed = time.time() - started

    hello = next((e for e in events if e["event"] == "hello"), {})
    # Compare resolved paths: /opt/homebrew/bin/python3.x is a symlink, and
    # the child reports where it actually lives.
    child_exe = os.path.realpath(hello.get("executable", ""))
    check("the child ran, and it was the interpreter we asked for",
          child_exe == os.path.realpath(py),
          f"child reported {child_exe} / {hello.get('python')}")
    check("the child is not running out of Calibre's bundle",
          "calibre.app" not in child_exe.lower(), child_exe)
    check("Calibre's 3.14 did not leak into the child",
          not hello.get("python", "").startswith("3.14"),
          f"child python {hello.get('python')}")
    check("a result event came back", out.get("epub") == "/tmp/spike.epub")

    # ── 3. progress arrives DURING the run, not in one lump at the end ──
    print("\n3. Progress streaming")
    prog = [e for e in events if e["event"] == "progress" and e["phase"] == "ocr"]
    check("every batch reported", [e["done"] for e in prog] == [4, 8, 12, 16, 20, 24],
          f"got {[e['done'] for e in prog]}")
    spread = (stamps[-1] - stamps[0]) / elapsed if len(stamps) > 1 else 0
    check("events were spread across the run, not buffered to the end",
          spread > 0.5, f"first-to-last spanned {spread:.0%} of the run")
    check("human log came through on stdout unbuffered",
          any("OCR'd 4/24" in line for line in logs), f"{len(logs)} log lines")

    # ── 4. cancelling kills the whole tree ─────────────────────────────
    print("\n4. Cancellation")
    abort = threading.Event()
    seen_child = {}

    def watch(e):
        if e["event"] == "hello":
            seen_child.update(e)
            threading.Timer(0.3, abort.set).start()

    t0 = time.time()
    try:
        runner.run([py, STUB, "convert", "--progress-json", "--pages", "400",
                    "--delay", "0.4", "--hang", "--spawn-child"],
                   on_progress=watch, abort=abort)
        check("cancelling raises Aborted", False, "it returned normally")
    except runner.Aborted:
        check("cancelling raises Aborted", True,
              f"took {time.time() - t0:.1f}s including the kill grace period")

    time.sleep(0.4)
    def alive(pid):
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    check("the child that ignored SIGTERM is dead", not alive(seen_child["pid"]),
          f"pid {seen_child['pid']}")
    check("its own spawned helper died with it",
          not alive(seen_child["child_pid"]),
          f"pid {seen_child['child_pid']} — this is the surya/llama-server case")

    # ── 5. failure is legible ──────────────────────────────────────────
    print("\n5. Failure reporting")
    try:
        runner.run([py, STUB, "convert", "--progress-json", "--pages", "20",
                    "--delay", "0.01", "--fail-at", "8"])
        check("a non-zero exit raises Failed", False, "it returned normally")
    except runner.Failed as exc:
        check("a non-zero exit raises Failed", True)
        check("the child's own message survives",
              "on purpose" in exc.message, repr(exc.message))

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    if bad:
        print("FAILED: " + "; ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
