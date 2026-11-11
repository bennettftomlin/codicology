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
    monkeypatch.setattr(doctor, "layout_smoke",
                        lambda: {"ok": True, "boxes": 2})
    s = doctor.smoke("fake", ["en"])
    assert s["ok"] is True
    assert s["backend"] == "fake" and s["shape"] == "predictor"
    assert "fox" in s["matched"]
    assert s["layout"]["boxes"] == 2


def test_smoke_fails_when_the_fallback_layout_is_refused(monkeypatch, vtb):
    # the page reads, but the layout step behind it returns nothing — the
    # state the shelf ran in for two months, invisible to a read alone
    class Reads:
        name = "surya"
        _shape = "predictor"

        def run(self, images):
            return ["CODICOLOGY DOCTOR the quick brown fox reads page 42"]

    monkeypatch.setattr(vtb, "load_backend", lambda ocr, langs: Reads())
    monkeypatch.setattr(doctor, "layout_smoke",
                        lambda: {"ok": False, "boxes": 0})
    s = doctor.smoke("surya", ["en"])
    assert s["ok"] is False and "layout" in s["error"]


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


# ── which OpenCV imports ─────────────────────────────────────────────────────

def _with_opencv(monkeypatch, tmp_path, wheels, version, pin):
    """An environment whose cv2 directory was last written by `version`."""
    monkeypatch.setattr(doctor.shutil, "which",
                        lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(doctor, "_pkg_version",
                        lambda name: wheels.get(name)
                        or (None if name.startswith("opencv") else "1.0"))
    monkeypatch.setattr(doctor, "_surya_opencv_pin", lambda: pin)
    if version is None:
        monkeypatch.setattr(doctor.importlib.util, "find_spec",
                            lambda name: None)
        return
    cv2 = tmp_path / "cv2"
    cv2.mkdir()
    (cv2 / "version.py").write_text(f'opencv_version = "{version}"\n'
                                    "headless = False\n")

    class Spec:
        origin = str(cv2 / "__init__.py")
    monkeypatch.setattr(doctor.importlib.util, "find_spec", lambda name: Spec)


def test_two_opencv_wheels_are_named_with_the_one_that_won(monkeypatch,
                                                           tmp_path):
    """The state this project's own dependency line produced: pip lists
    Surya's pinned wheel as installed, and cv2 is the other one."""
    _with_opencv(monkeypatch, tmp_path,
                 {"opencv-python-headless": "4.11.0.86",
                  "opencv-python": "5.0.0.93"},
                 version="5.0.0.93", pin="4.11.0.86")
    rep = doctor.report()
    assert rep["ok"] is True                 # it reads; it is not what's pinned
    assert rep["opencv"]["version"] == "5.0.0.93"
    (said,) = [w for w in rep["warnings"] if "OpenCV" in w]
    assert "cv2 is 5.0.0.93 where surya-ocr pins 4.11.0.86" in said
    assert "pip uninstall opencv-python, then" in said
    assert "opencv-python-headless==4.11.0.86" in said


def test_the_pinned_headless_wheel_alone_says_nothing(monkeypatch, tmp_path):
    _with_opencv(monkeypatch, tmp_path,
                 {"opencv-python-headless": "4.11.0.86"},
                 version="4.11.0.86", pin="4.11.0.86")
    rep = doctor.report()
    assert rep["ok"] is True
    assert not [w for w in rep["warnings"] if "OpenCV" in w]


def test_one_wheel_that_is_not_the_pinned_release_is_said(monkeypatch,
                                                          tmp_path):
    _with_opencv(monkeypatch, tmp_path, {"opencv-python-headless": "4.12.0.88"},
                 version="4.12.0.88", pin="4.11.0.86")
    (said,) = [w for w in doctor.report()["warnings"] if "OpenCV" in w]
    assert "cv2 is 4.12.0.88 where surya-ocr pins 4.11.0.86" in said
    assert "uninstall" not in said


def test_without_surya_any_single_opencv_is_fine(monkeypatch, tmp_path):
    _with_opencv(monkeypatch, tmp_path, {"opencv-python": "4.9.0.80"},
                 version="4.9.0.80", pin=None)
    assert not [w for w in doctor.report(ocr="tesseract")["warnings"]
                if "OpenCV" in w]


def test_no_cv2_is_a_problem(monkeypatch, tmp_path):
    _with_opencv(monkeypatch, tmp_path, {}, version=None, pin="4.11.0.86")
    rep = doctor.report()
    assert rep["ok"] is False
    assert any("OpenCV is not installed" in p for p in rep["problems"])


def test_suryas_pin_is_read_from_its_own_metadata(monkeypatch):
    monkeypatch.setattr(doctor.importlib.metadata, "requires",
                        lambda name: ["pillow<11,>=10.2.0",
                                      "opencv-python-headless==4.11.0.86",
                                      "torch>=2.7"])
    assert doctor._surya_opencv_pin() == "4.11.0.86"
    monkeypatch.setattr(doctor.importlib.metadata, "requires",
                        lambda name: ["pillow<11,>=10.2.0"])
    assert doctor._surya_opencv_pin() is None


# ── the rectifier ────────────────────────────────────────────────────────────

def test_the_rectifier_is_reported_and_its_absence_is_not_a_problem(
        monkeypatch, capsys):
    """This line used to import a module that no longer exists, inside a
    try that hid the failure, so the doctor said nothing about geometry at
    all: someone photographing a book was told "ready" without it."""
    monkeypatch.setattr(doctor.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(doctor, "_pkg_version",
                        lambda name: None if name in ("transformers",
                                                      "scikit-image")
                        else "1.0")
    monkeypatch.setattr(doctor, "_opencv",
                        lambda: {"present": True, "wheels": {}, "version": "4.11",
                                 "surya_pins": None})
    rep = doctor.report()
    assert rep["rectifier"] == {"present": False,
                                "missing": ["transformers", "scikit-image"]}
    assert rep["ok"] is True
    doctor._human(rep)
    out = capsys.readouterr().out
    assert "rectifier: not installed (transformers, scikit-image missing)" in out
    assert "codicology[geometry]" in out


def test_the_rectifier_is_reported_when_it_is_there(monkeypatch, capsys):
    monkeypatch.setattr(doctor, "_pkg_version", lambda name: "1.0")
    assert doctor._rectifier() == {"present": True, "missing": []}


# ── the second reader ────────────────────────────────────────────────────────

def test_no_second_reader_is_neither_a_problem_nor_a_warning(monkeypatch, tmp_path,
                                                            capsys):
    """It is optional: an environment without the model is as ready as it
    was, and says where the model would go."""
    from codicology import second
    monkeypatch.setattr(second, "DEFAULT_DIR", str(tmp_path / "none"))
    monkeypatch.delenv(second.MODEL_ENV, raising=False)
    monkeypatch.delenv(second.MMPROJ_ENV, raising=False)
    rep = doctor.report()
    assert rep["second_reader"]["present"] is False
    assert not any("second" in s for s in rep["problems"] + rep["warnings"])
    doctor._human(rep)
    assert "second reader: none" in capsys.readouterr().out


def test_a_second_reader_is_reported_with_its_model(monkeypatch, tmp_path, capsys):
    from codicology import second
    for name in ("reader-Q8_0.gguf", "mmproj-reader-Q8_0.gguf"):
        (tmp_path / name).write_bytes(b"gguf")
    monkeypatch.setattr(second, "DEFAULT_DIR", str(tmp_path))
    monkeypatch.delenv(second.MODEL_ENV, raising=False)
    monkeypatch.delenv(second.MMPROJ_ENV, raising=False)
    monkeypatch.setattr(second, "server_binary", lambda: "/usr/bin/llama-server")
    rep = doctor.report()
    assert rep["second_reader"] == {
        "present": True, "name": second.NAME,
        "model": str(tmp_path / "reader-Q8_0.gguf"),
        "looked_in": str(tmp_path)}
    doctor._human(rep)
    assert "asked with --second-reader" in capsys.readouterr().out

