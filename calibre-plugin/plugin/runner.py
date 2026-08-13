"""
Run codicology as a subprocess from inside Calibre, and translate what it
says back into progress the job list understands.

This module deliberately imports nothing from calibre. The plugin uses it,
but so does spike/check_runner.py, which proves the hard part — that a
foreign Python survives being spawned from Calibre's — without needing a
GUI to click.

Two things make this more than a Popen call.

Environment. Calibre points several variables at its own app bundle, and a
process that is not Calibre must not inherit them. Measured on Calibre 9.13
(macOS), the offenders are SSL_CERT_DIR, OPENSSL_MODULES, OPENSSL_ENGINES
and FONTCONFIG_*. The SSL ones matter most: leave them in place and every
HTTPS fetch the child attempts — uv resolving packages, surya pulling model
weights — looks for certificates inside Calibre.app and fails somewhere far
from the cause. PYTHONHOME/PYTHONPATH are scrubbed too; that build does not
set them, but a future one might, and the cost of scrubbing an unset
variable is nothing.

PATH. When Calibre is launched from Finder it inherits launchd's minimal
PATH, not a shell's, so a binary the user can run in a terminal may be
invisible here. Nothing is resolved by bare name that we have not put on
PATH ourselves.
"""
import json
import os
import queue
import subprocess
import sys
import threading

# Variables that point into Calibre's own bundle. A foreign process that
# inherits these looks for libraries, certificates and fonts inside
# Calibre.app and fails in ways that do not name their cause.
SCRUB = (
    "SSL_CERT_DIR", "SSL_CERT_FILE", "OPENSSL_ENGINES", "OPENSSL_MODULES",
    "FONTCONFIG_FILE", "FONTCONFIG_PATH",
    "PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONEXECUTABLE",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "DYLD_FRAMEWORK_PATH",
    "QT_PLUGIN_PATH", "QML2_IMPORT_PATH",
    "MAGICK_HOME", "MAGICK_CONFIGURE_PATH", "GS_LIB",
)

# A PATH to fall back on when Calibre was launched from Finder and inherited
# launchd's, which contains almost nothing.
_BASE_PATH = {
    "win32": r"C:\Windows\system32;C:\Windows",
}.get(sys.platform, "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin")


class Aborted(Exception):
    """The user cancelled the job."""


class Failed(Exception):
    """The child exited non-zero."""

    def __init__(self, returncode, message=""):
        super().__init__(message or f"exited with status {returncode}")
        self.returncode = returncode
        self.message = message


def child_env(extra=None, path_prepend=()):
    """
    The environment a codicology subprocess should see.

    Everything Calibre aimed at itself is removed, PATH is guaranteed to
    hold something usable, and `path_prepend` puts our managed native
    binaries (tesseract, llama-server) ahead of anything the user has —
    codicology resolves tesseract by bare name, so PATH is the only lever.
    """
    env = {k: v for k, v in os.environ.items() if k not in SCRUB}
    env.pop("CALIBRE_WORKER", None)

    parts = [str(p) for p in path_prepend if p]
    existing = env.get("PATH") or _BASE_PATH
    # A Finder-launched Calibre has a PATH with nothing useful on it.
    if len(existing.split(os.pathsep)) < 3:
        existing = os.pathsep.join([existing, _BASE_PATH]) if existing else _BASE_PATH
    env["PATH"] = os.pathsep.join(parts + [existing])

    # Without this the child block-buffers behind a pipe and progress
    # arrives in one lump when the run is already over.
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update({k: str(v) for k, v in extra.items()})
    return env


def scrubbed(env=None):
    """Which variables child_env would strip. For diagnostics."""
    src = os.environ if env is None else env
    return sorted(k for k in SCRUB if k in src)


# ── the progress protocol ────────────────────────────────────────────────────
#
# codicology --progress-json writes one JSON object per line to stderr,
# leaving stdout as the human-readable log. stderr also carries warnings and
# tracebacks, so a line counts as progress only if it parses as a JSON
# object carrying an "event" key; everything else is log text.
#
#   {"event": "phase",    "phase": "ocr", "message": "OCR backend: surya"}
#   {"event": "progress", "phase": "ocr", "done": 40, "total": 312,
#                         "note": "12 from cache"}
#   {"event": "result",   "epub": "/path/out.epub"}
#   {"event": "error",    "message": "..."}

def parse_progress(line):
    """A progress event, or None if this line is just log text."""
    line = line.strip()
    if not line or line[0] != "{":
        return None
    try:
        obj = json.loads(line)
    except ValueError:
        return None
    return obj if isinstance(obj, dict) and "event" in obj else None


def _spawn_kwargs():
    """Put the child in its own process group so we can kill its children too."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate(proc, grace=5.0):
    """
    Kill the child and everything it spawned.

    Surya starts workers, and 0.2x spawns a llama-server of its own, so
    killing the process we hold leaves inference running. Signalling the
    group is the only way to take the whole tree down.
    """
    if proc.poll() is not None:
        return
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=grace)
        else:
            import signal
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except Exception:
        proc.terminate()
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            if sys.platform != "win32":
                import signal
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            else:
                proc.kill()
        except Exception:
            proc.kill()


def _pump(stream, name, q):
    try:
        for line in iter(stream.readline, ""):
            q.put((name, line))
    finally:
        q.put((name, None))
        try:
            stream.close()
        except Exception:
            pass


def run(argv, on_progress=None, on_log=None, abort=None,
        env_extra=None, path_prepend=(), cwd=None, poll=0.2):
    """
    Run codicology to completion.

    on_progress(event)  — one parsed JSON event
    on_log(stream, text) — a line of human output; stream is 'out' or 'err'
    abort               — anything with .is_set(); checked while waiting

    Returns the collected {"result": ...} event payload. Raises Aborted if
    cancelled, Failed if the child exits non-zero.
    """
    env = child_env(env_extra, path_prepend)
    proc = subprocess.Popen(
        [str(a) for a in argv],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=env, cwd=cwd, text=True, bufsize=1,
        errors="replace", **_spawn_kwargs())

    q = queue.Queue()
    for stream, name in ((proc.stdout, "out"), (proc.stderr, "err")):
        threading.Thread(target=_pump, args=(stream, name, q), daemon=True).start()

    result, error, open_streams = {}, None, 2
    try:
        while open_streams:
            if abort is not None and abort.is_set():
                _terminate(proc)
                raise Aborted()
            try:
                name, line = q.get(timeout=poll)
            except queue.Empty:
                continue
            if line is None:
                open_streams -= 1
                continue

            event = parse_progress(line) if name == "err" else None
            if event is None:
                if on_log:
                    on_log(name, line.rstrip("\n"))
                continue
            if event.get("event") == "result":
                result = event
            elif event.get("event") == "error":
                error = event.get("message") or "codicology reported an error"
            if on_progress:
                on_progress(event)
    finally:
        if proc.poll() is None:
            _terminate(proc)

    rc = proc.wait()
    if rc != 0:
        raise Failed(rc, error or "")
    return result
