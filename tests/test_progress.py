"""The progress stream is a contract, not a log.

--progress-json is consumed by a GUI on the far side of a pipe (the Calibre
plugin), which fails silently when a field is renamed or a line stops being
one JSON object. These tests hold the shape still: one object per line on
stderr, an "event" key on every one, a result event when a build finishes,
and an error event carrying the refusal message when it does not.
"""
import json

import pytest

from codicology import progress


@pytest.fixture(autouse=True)
def _reset_progress():
    """The flag is process-global; a test that enables it must not leak."""
    yield
    progress.disable()


def _events(capsys):
    err = capsys.readouterr().err
    return [json.loads(line) for line in err.splitlines()
            if line.strip().startswith("{")]


def test_silent_until_enabled(capsys):
    progress.emit(event="progress", done=1, total=2)
    assert capsys.readouterr().err == ""


def test_one_json_object_per_line_with_an_event_key(capsys):
    progress.enable()
    progress.emit(event="progress", phase="ocr", done=4, total=10, note="")
    progress.emit(event="phase", phase="folios",
                  message="Reading printed page numbers…")
    events = _events(capsys)
    assert [e["event"] for e in events] == ["progress", "phase"]
    assert events[0]["done"] == 4 and events[0]["total"] == 10
    # Non-ASCII survives: the message above carries an ellipsis.
    assert events[1]["message"].endswith("…")


def test_convert_emits_a_result_event(tmp_path, monkeypatch, capsys, vtb):
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    out = tmp_path / "book.epub"

    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    called = {}
    monkeypatch.setattr(vtb, "process_video", lambda **kw: called.update(kw))

    vtb.main(["--pages-from", str(src), "--epub", str(out), "--progress-json"])

    events = _events(capsys)
    assert events and events[-1]["event"] == "result"
    assert events[-1]["epub"] == str(out)
    assert called["pages_from"] == str(src)


def test_a_refusal_reaches_the_stream_before_the_exit(tmp_path, monkeypatch,
                                                      capsys, vtb):
    """sys.exit(message) is the pipeline's idiom for refusing; the driver
    must receive the reason as an event, not only as a status code."""
    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    with pytest.raises(SystemExit):
        vtb.main(["--pages-from", str(tmp_path / "missing.pdf"),
                  "--epub", str(tmp_path / "out.epub"), "--progress-json"])
    events = _events(capsys)
    assert events and events[-1]["event"] == "error"
    assert "not found" in events[-1]["message"]


def test_a_crash_reaches_the_stream_before_the_traceback(tmp_path, monkeypatch,
                                                         capsys, vtb):
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-1.4\n")

    monkeypatch.setattr(vtb, "witness_available", lambda: True)

    def boom(**kw):
        raise RuntimeError("torch fell over")

    monkeypatch.setattr(vtb, "process_video", boom)
    with pytest.raises(RuntimeError):
        vtb.main(["--pages-from", str(src), "--epub",
                  str(tmp_path / "out.epub"), "--progress-json"])
    events = _events(capsys)
    assert events[-1]["event"] == "error"
    assert "torch fell over" in events[-1]["message"]


def test_without_the_flag_nothing_machine_readable_appears(tmp_path,
                                                           monkeypatch,
                                                           capsys, vtb):
    src = tmp_path / "book.pdf"
    src.write_bytes(b"%PDF-1.4\n")
    monkeypatch.setattr(vtb, "witness_available", lambda: True)
    monkeypatch.setattr(vtb, "process_video", lambda **kw: None)
    vtb.main(["--pages-from", str(src), "--epub", str(tmp_path / "o.epub")])
    assert _events(capsys) == []
