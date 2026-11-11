"""constraints.txt is the closure of what codicology asks for, as installed.

Surya leaves `transformers` with a floor and no ceiling, so "pip install
codicology[surya]" a month from now is not the environment the shelf was
built in. The file records that environment; these tests hold the tool
that writes it to what a constraints file must be.
"""
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import lock_env  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")


class _Dist:
    def __init__(self, name, version, requires=()):
        self.metadata, self.version, self.requires = {"Name": name}, version, list(requires)


def _installed(monkeypatch, dists):
    table = {lock_env._norm(d.metadata["Name"]): d for d in dists}

    def distribution(name):
        try:
            return table[lock_env._norm(name)]
        except KeyError:
            raise lock_env.md.PackageNotFoundError(name)
    monkeypatch.setattr(lock_env.md, "distribution", distribution)


def test_an_extra_brings_in_what_it_names_and_nothing_else(monkeypatch, capsys):
    _installed(monkeypatch, [
        _Dist("codicology", "0.1.0", ["numpy>=1.24", 'surya-ocr==0.22.1; extra == "surya"',
                                      'easyocr; extra == "easyocr"']),
        _Dist("numpy", "2.2.6"),
        _Dist("surya-ocr", "0.22.1", ["transformers>=5.12.1", "Pillow<11,>=10.2",
                                      'colorama; sys_platform == "win32"',
                                      "torch[opt]>=2.7"]),
        _Dist("transformers", "5.14.1", ["numpy>=1.17", 'tokenizers; extra == "torch"']),
        _Dist("Pillow", "10.4.0"),
        _Dist("torch", "2.13.0", ['opt-einsum; extra == "opt"']),
        _Dist("opt_einsum", "3.4.0"),
        _Dist("easyocr", "1.7.2"),
        _Dist("tokenizers", "0.21.0"),
    ])
    assert lock_env.closure("codicology", {"surya"}) == {
        "codicology": "0.1.0", "numpy": "2.2.6", "surya-ocr": "0.22.1",
        "transformers": "5.14.1", "pillow": "10.4.0", "torch": "2.13.0",
        "opt-einsum": "3.4.0"}
    assert lock_env.closure("codicology", set()) == {
        "codicology": "0.1.0", "numpy": "2.2.6"}


def test_a_requirement_that_is_not_installed_is_said_and_left_out(monkeypatch,
                                                                 capsys):
    _installed(monkeypatch, [_Dist("codicology", "0.1.0", ["numpy", "lxml"]),
                             _Dist("numpy", "2.2.6")])
    assert lock_env.closure("codicology", set()) == {
        "codicology": "0.1.0", "numpy": "2.2.6"}
    assert "lxml is required but not installed" in capsys.readouterr().err


def test_the_committed_file_pins_the_recogniser_and_what_it_leaves_open():
    lines = open(os.path.join(ROOT, "constraints.txt")).read().splitlines()
    pins = dict(l.split("==") for l in lines if l and not l.startswith("#"))
    assert all(re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) for name in pins)
    assert pins["surya-ocr"] == "0.22.1", "the pin pyproject.toml names"
    for open_ended in ("transformers", "torch", "pillow", "pypdfium2",
                       "opencv-python-headless", "numpy"):
        assert open_ended in pins, open_ended
    assert "codicology" not in pins
    # the wheel that fought the pinned one for the cv2 directory
    assert "opencv-python" not in pins
