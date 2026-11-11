"""The one piece of the Calibre plugin that can be run outside Calibre.

The plugin imports calibre's own modules, so its code cannot be loaded by
this suite as it stands. env.py needs only two of them, and both are easy
to stand in for; what it does with the codicology executable is real.
"""
import importlib.util
import json
import os
import pathlib
import sys
import types

import pytest

from test_decision_sites import _fix

PLUGIN = pathlib.Path(__file__).resolve().parents[1] / "calibre-plugin" / "plugin"
BODY = "<p>to be or not to be, he said; let it be so</p>"


@pytest.fixture
def env(monkeypatch):
    pkg = types.ModuleType("calibre_plugins")
    ours = types.ModuleType("calibre_plugins.codicology_ocr")
    runner = types.ModuleType("calibre_plugins.codicology_ocr.runner")
    runner.child_env = lambda *a, **k: dict(os.environ)
    config = types.ModuleType("calibre_plugins.codicology_ocr.config")
    config.prefs = {"codicology_path": ""}
    ours.runner, ours.config = runner, config
    for name, mod in (("calibre_plugins", pkg),
                      ("calibre_plugins.codicology_ocr", ours),
                      ("calibre_plugins.codicology_ocr.runner", runner),
                      ("calibre_plugins.codicology_ocr.config", config)):
        monkeypatch.setitem(sys.modules, name, mod)
    spec = importlib.util.spec_from_file_location("plugin_env", PLUGIN / "env.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def exe(tmp_path):
    """A codicology on disk that is this tree's own."""
    path = tmp_path / "codicology"
    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    path.write_text(f"#!{sys.executable}\nimport sys\n"
                    f"sys.path.insert(0, {str(src)!r})\n"
                    "from codicology.cli import main\nsys.exit(main())\n")
    path.chmod(0o755)
    return str(path)


def _exported(tmp_path, name, rows, sheet):
    path = tmp_path / name
    path.write_text(json.dumps({"name": "book-disputes", "page_files": 1,
                                "sheet": sheet, "decisions": rows}))
    return str(path)


def test_importing_a_second_round_keeps_the_first(env, exe, tmp_path):
    target = str(tmp_path / "cache" / "7-abcd1234-disputes.decisions.json")
    os.makedirs(os.path.dirname(target))
    one = env.file_decisions(exe, _exported(tmp_path, "a.json",
                                            [_fix(BODY, 1, "he")], "aaa"), target)
    assert one == {"rows": 1, "round": 1, "replaced": False,
                   "dropped_retired": 0, "retired": 0}
    two = env.file_decisions(exe, _exported(tmp_path, "b.json",
                                            [_fix(BODY, 9, "is")], "bbb"), target)
    assert two["round"] == 2 and two["replaced"] is False
    kept = target[:-len(".json")] + ".r1.json"
    assert json.load(open(kept))["decisions"][0]["new"] == "he"
    assert json.load(open(target))["decisions"][0]["new"] == "is"


def test_a_codicology_without_the_verb_is_none_and_the_plugin_copies(
        env, tmp_path):
    old = tmp_path / "codicology"
    old.write_text(f"#!{sys.executable}\nimport sys\n"
                   "print('usage: codicology …', file=sys.stderr)\nsys.exit(2)\n")
    old.chmod(0o755)
    target = str(tmp_path / "x.decisions.json")
    assert env.file_decisions(str(old), "exported.json", target) is None
    assert env.file_decisions(str(tmp_path / "absent"), "e.json", target) is None
    assert not os.path.exists(target)


def test_a_file_codicology_refuses_is_none(env, exe, tmp_path):
    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps({"disputes": []}))
    assert env.file_decisions(exe, str(bad), str(tmp_path / "t.json")) is None
