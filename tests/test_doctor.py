"""doctor tells the truth about an environment, cheaply by default.

The quick report must never import torch or surya — it is the gate a GUI
runs before every conversion. The --full smoke test exists because the
expensive failure is invisible to imports: a surya 0.2x without llama-server
imports cleanly and fails on the first real page.
"""
import json

import pytest

from codicology import cli, doctor


def test_quick_is_not_ready_without_tesseract(monkeypatch):
    monkeypatch.setattr(doctor.shutil, "which", lambda name: None)
    rep = doctor.report()
    assert rep["ok"] is False
    assert any("tesseract" in p for p in rep["problems"])


def test_missing_llama_server_is_a_problem_on_new_surya(monkeypatch):
    """The 0.2x trap: package present, server absent, imports clean."""
    monkeypatch.setattr(doctor.shutil, "which",
                        lambda name: "/usr/bin/tesseract"
                        if name == "tesseract" else None)
    monkeypatch.delenv("LLAMA_CPP_BINARY", raising=False)
    monkeypatch.setattr(doctor, "_pkg_version",
                        lambda name: "0.22.1" if name == "surya-ocr" else "1.0")
    rep = doctor.report()
    assert rep["ok"] is False
    assert any("llama" in p for p in rep["problems"])

    # The same environment on old surya has no such requirement.
    monkeypatch.setattr(doctor, "_pkg_version",
                        lambda name: "0.6.13" if name == "surya-ocr" else "1.0")
    assert doctor.report()["ok"] is True


@pytest.mark.parametrize("version,needs", [
    (None, False),          # not installed: nothing to serve
    ("0.6.13", False),
    ("0.17.1", False),
    ("0.20.0", True),
    ("0.22.1", True),
    ("weird", True),        # unparseable is newer than anything we know
])
def test_which_surya_generations_need_a_server(version, needs):
    assert doctor.surya_needs_server(version) is needs


def test_smoke_reports_what_actually_answered(monkeypatch, vtb):
    class Fake:
        name = "fake"
        _shape = "predictor"

        def run(self, images):
            return ["CODICOLOGY DOCTOR the quick brown fox reads page 42"]

    monkeypatch.setattr(vtb, "load_backend", lambda ocr, langs: Fake())
    s = doctor.smoke("fake", ["en"])
    assert s["ok"] is True
    assert s["backend"] == "fake" and s["shape"] == "predictor"
    assert "fox" in s["matched"]


def test_smoke_fails_on_a_backend_that_reads_nothing(monkeypatch, vtb):
    class Mute:
        name = "mute"

        def run(self, images):
            return [""]

    monkeypatch.setattr(vtb, "load_backend", lambda ocr, langs: Mute())
    assert doctor.smoke("mute", ["en"])["ok"] is False


def test_smoke_survives_a_backend_that_refuses_to_load(monkeypatch, vtb):
    def refuse(ocr, langs):
        import sys
        sys.exit("OCR backend 'surya' is not installed")

    monkeypatch.setattr(vtb, "load_backend", refuse)
    s = doctor.smoke("surya", ["en"])
    assert s["ok"] is False and "not installed" in s["error"]


def test_json_mode_prints_one_parseable_object(capsys):
    rc = cli.main(["doctor", "--json"])
    rep = json.loads(capsys.readouterr().out)
    assert isinstance(rep["ok"], bool)
    assert rc == (0 if rep["ok"] else 1)
    assert "tesseract" in rep and "packages" in rep


def test_human_mode_ends_with_a_verdict(capsys):
    cli.main(["doctor"])
    out = capsys.readouterr().out
    assert out.splitlines()[-1].startswith("VERDICT:")
