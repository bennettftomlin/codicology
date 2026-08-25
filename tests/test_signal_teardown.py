"""A stopped job has to take the inference server down with it.

The server is spawned into its own session on purpose, so that it outlives
one page's work and can be attached to again. That also puts it beyond any
process-group kill aimed at us: the only thing that stops it is the
`atexit` handler the spawner registers, and Python runs `atexit` only for a
signal it actually handled.

Two servers were found orphaned on the development machine — a 2 GB
llama-server whose parent had gone, and a detection server orphaned the
same way eight days earlier — so this is tested by running a real process
and signalling it, not by inspecting which handler is installed.
"""
import subprocess
import sys
import textwrap

import pytest

PROG = textwrap.dedent('''
    import atexit, os, signal, sys
    sys.path.insert(0, {src!r})
    from codicology.cli import _exit_cleanly_on_signal
    {install}
    atexit.register(lambda: print("CLEANUP-RAN", flush=True))
    print("READY", flush=True)
    os.kill(os.getpid(), signal.{signame})
    import time; time.sleep(10)
    print("SURVIVED", flush=True)
''')


def _run(src_dir, install, signame):
    prog = PROG.format(src=str(src_dir), install=install, signame=signame)
    return subprocess.run([sys.executable, "-c", prog],
                          capture_output=True, text=True, timeout=60)


@pytest.fixture
def src_dir():
    import codicology
    return str(__import__("pathlib").Path(codicology.__file__).parent.parent)


@pytest.mark.parametrize("signame", ["SIGTERM", "SIGHUP"])
def test_a_signalled_run_still_cleans_up(src_dir, signame):
    r = _run(src_dir, "_exit_cleanly_on_signal()", signame)
    assert "CLEANUP-RAN" in r.stdout, r.stdout + r.stderr
    assert "SURVIVED" not in r.stdout, "the signal must still end the run"


def test_without_the_handler_the_cleanup_is_skipped(src_dir):
    """The control. Without this the test above passes for the wrong reason
    — an earlier version of this check accidentally installed the handler in
    both arms and reported the bug as already fixed."""
    r = _run(src_dir, "pass", "SIGTERM")
    assert "CLEANUP-RAN" not in r.stdout
    assert r.returncode < 0, "a default SIGTERM kills without unwinding"


def test_the_exit_status_still_names_the_signal(src_dir):
    """128+n is the shell's convention, so a caller reading the status can
    still tell what happened."""
    import signal as _s
    r = _run(src_dir, "_exit_cleanly_on_signal()", "SIGTERM")
    assert r.returncode == 128 + _s.SIGTERM


def test_installing_the_handler_off_the_main_thread_does_not_raise(vtb):
    """Python refuses signal handlers outside the main thread. A build run
    from a worker thread must still run, it just cleans up less well."""
    import threading
    from codicology.cli import _exit_cleanly_on_signal
    box = []
    t = threading.Thread(target=lambda: box.append(
        _exit_cleanly_on_signal() or "returned"))
    t.start()
    t.join()
    assert box == ["returned"]
