"""
Phase 1 checks: the installed ZIP, its worker, and — when the machine has
a codicology environment — a real conversion driven end to end through the
plugin's own code, inside Calibre's interpreter, with no GUI.

    /Applications/calibre.app/Contents/MacOS/calibre-debug \
        -e calibre-plugin/spike/check_plugin.py

What remains after this is one click, described in VERIFY.md.
"""
import json
import os
import queue
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(HERE, "stub_codicology.py")
VENV_CODICOLOGY = os.path.expanduser(
    "~/claude_knowledge/.venv/bin/codicology")

results = []


def check(name, ok, detail=""):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f"        {detail}")


def foreign_python():
    """An interpreter that is not Calibre's, for driving the stub."""
    import platform
    want = platform.machine()
    for base in ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"):
        for minor in (12, 11, 13, 10):
            path = f"{base}/python3.{minor}"
            if not os.path.exists(path):
                continue
            try:
                got = subprocess.run(
                    [path, "-c", "import platform;print(platform.machine())"],
                    capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                continue
            if got == want:
                return path
    return None


def main():
    from calibre.customize.ui import find_plugin

    plugin = find_plugin("Codicology OCR")
    check("the plugin is registered with Calibre", plugin is not None)
    if plugin is None:
        return 1
    check("it is version 0.1.0 (Phase 1)", plugin.version == (0, 1, 0),
          str(plugin.version))
    check("it is loaded from the installed ZIP",
          plugin.plugin_path and plugin.plugin_path.endswith(".zip"),
          str(plugin.plugin_path))

    # ── modules import from inside the ZIP ──────────────────────────────
    import calibre_plugins.codicology_ocr.runner as runner
    check("runner strips what Calibre aims at itself",
          not [k for k in runner.SCRUB if k in runner.child_env()],
          f'would strip: {", ".join(runner.scrubbed()) or "none set"}')

    from calibre_plugins.codicology_ocr.config import prefs
    check("Phase 1 settings have defaults",
          prefs["lang"] == "en" and prefs["link_notes"] is True
          and prefs["verify_after_build"] is True)

    from calibre_plugins.codicology_ocr.ui import CodicologyOCRAction as A
    check("the toolbar button says OCR PDF", A.action_spec[0] == "OCR PDF")

    import calibre_plugins.codicology_ocr.dialogs as dialogs
    check("the pre-flight dialog is importable",
          hasattr(dialogs, "PreflightDialog"))

    data = plugin.load_resources(["images/icon.png"])
    check("the icon comes out of the ZIP",
          data.get("images/icon.png", b"")[:8] == b"\x89PNG\r\n\x1a\n")

    from calibre_plugins.codicology_ocr import env
    check("cache paths are computed, not searched for",
          env.cache_path("spike Lib", 7, STUB).endswith(".ocr.gz"),
          env.cache_path("spike Lib", 7, STUB))

    # ── the worker, against the stub: protocol, spans, abort ────────────
    print("\nWorker against the stub")
    from calibre_plugins.codicology_ocr.jobs import convert_worker

    py = foreign_python()
    check("a usable non-Calibre interpreter was found", bool(py), py or "")
    if not py:
        return 1

    try:
        from calibre.utils.logging import GUILog as Log
    except ImportError:
        from calibre.utils.logging import Log

    notes, log = queue.Queue(), Log()
    out = convert_worker(
        [py, STUB, "convert", "--progress-json", "--epub",
         "/tmp/check-plugin.epub", "--pages", "12", "--delay", "0.03"],
        None, 42, "/tmp/check-plugin.epub", "Stub Book",
        abort=threading.Event(), log=log, notifications=notes)
    sent = []
    while not notes.empty():
        sent.append(notes.get())
    check("the worker returns what finished() needs",
          out["book_id"] == 42 and out["epub"] == "/tmp/check-plugin.epub"
          and out["verify"] is None)
    fracs = [f for f, _ in sent]
    check("progress covers the bar once, monotonically",
          fracs == sorted(fracs) and fracs[-1] == 1.0 and len(fracs) >= 4,
          f"{len(fracs)} notifications, e.g. {sent[1]}")
    check("phase spans keep OCR inside its band",
          all(0.03 <= f <= 0.86 for f, m in sent if m.startswith("Reading pages")),
          str([round(f, 2) for f, m in sent if m.startswith("Reading pages")]))

    abort = threading.Event()
    threading.Timer(0.25, abort.set).start()
    try:
        convert_worker([py, STUB, "convert", "--progress-json",
                        "--pages", "400", "--delay", "0.4"],
                       None, 1, "/tmp/x.epub", "t",
                       abort=abort, log=Log(), notifications=queue.Queue())
        check("cancelling raises Aborted", False, "returned normally")
    except Exception as exc:
        check("cancelling raises Aborted",
              type(exc).__name__ == "Aborted", type(exc).__name__)

    # ── the real thing, when this machine has it ────────────────────────
    print("\nWorker against real codicology")
    if not os.path.exists(VENV_CODICOLOGY):
        check("codicology environment present", False,
              "not on this machine; skipping the live conversion")
    else:
        if not (prefs["codicology_path"] or "").strip():
            # Configure the plugin for the hand test while we are here.
            prefs["codicology_path"] = VENV_CODICOLOGY
            print(f"        configured codicology_path = {VENV_CODICOLOGY}")
        exe = env.resolve_codicology()
        check("env.resolve_codicology finds it", exe == os.path.expanduser(
            (prefs["codicology_path"] or VENV_CODICOLOGY)), str(exe))
        check("it answers --version through the scrubbed environment",
              bool(env.codicology_version(exe)),
              f"codicology {env.codicology_version(exe)}")
        doctor = env.quick_doctor(exe)
        check("doctor --json parses and is ready",
              bool(doctor) and doctor.get("ok") is True,
              "problems: " + "; ".join((doctor or {}).get("problems", ["none"])))

        # Reproduce the GUI exactly: launchd's PATH, then ask doctor.
        # This is the condition the first live Check environment failed.
        saved = os.environ.get("PATH")
        os.environ["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
        try:
            gui_doctor = env.quick_doctor(exe)
        finally:
            os.environ["PATH"] = saved
        check("doctor stays ready under a Finder-launched PATH (the GUI case)",
              bool(gui_doctor) and gui_doctor.get("ok") is True,
              ("problems: " + "; ".join(gui_doctor.get("problems", ["none"])))
              if gui_doctor else "no answer from doctor")

        scratch = os.environ.get("CODICOLOGY_SPIKE_DIR") or os.path.join(
            os.path.dirname(os.environ.get("CLAUDE_SCRATCH", "/tmp")), "")
        pdf = os.path.join(
            "/private/tmp/claude-502/-Users-Bennett1-claude-knowledge/"
            "2f3339b6-39a6-495e-83c6-174022c798f0/scratchpad",
            "spike-book.pdf")
        cache = os.path.join(os.path.dirname(pdf), "spike.ocr.gz")
        if not os.path.exists(pdf):
            check("spike book present", False, pdf + " missing; skipping")
        else:
            epub = "/tmp/check-plugin-real.epub"
            notes, log = queue.Queue(), Log()
            out = convert_worker(
                [exe, "convert", "--pages-from", pdf, "--epub", epub,
                 "--ocr-cache", cache, "--title", "Spike Book",
                 "--progress-json"],
                [exe, "verify", epub, pdf],
                7, epub, "Spike Book",
                abort=threading.Event(), log=log, notifications=notes)
            check("a real conversion ran through the plugin's worker",
                  os.path.exists(epub) and out["result"].get("epub") == epub,
                  f"{os.path.getsize(epub)} bytes")
            check("verify ran and found no holes",
                  out["verify"] is not None and out["verify"]["rc"] == 0,
                  (out["verify"] or {}).get("output", "").splitlines()[-1]
                  if out.get("verify") else "no verify")
            sent = []
            while not notes.empty():
                sent.append(notes.get())
            check("the real pipeline's events drove the bar",
                  any("Reading pages" in m for _, m in sent)
                  and any("holes" in m for _, m in sent),
                  f"{len(sent)} notifications")
            check("the human log reached the job viewer",
                  "OCR backend" in log.plain_text
                  and "VERDICT" in log.plain_text,
                  f"{len(log.plain_text)} chars")

    bad = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    if bad:
        print("FAILED: " + "; ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
