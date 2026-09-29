r"""Surya's guided schemas must be ones llama.cpp can compile.

Surya 0.22.1 ships its layout box pattern as `^\d{1,4} \d{1,4} \d{1,4}
\d{1,4}$`, and llama.cpp's schema-to-grammar converter has no `\d`: every
guided layout request came back "failed to parse grammar" with no boxes.
Layout is surya's fallback for a page whose full-page read fails, so for
two months that fallback read nothing. The patterns are judged here by a model of
llama.cpp's own escape rule — not by looking for `\d` — so a later surya
pattern using `\w` or `\s` fails too.
"""
import re
import sys
import types

import pytest

from codicology import pipeline as vtb

prompts = pytest.importorskip("surya.inference.prompts")

# llama.cpp common/json-schema-to-grammar.cpp (master, 2026-09-29):
# gbnf_escape_length and ESCAPED_IN_REGEXPS_BUT_NOT_IN_LITERALS
_GBNF_ESC = set("trn\\\"[]-")
_HEX = {"x": 2, "u": 4, "U": 8}
_REGEX_ONLY = set("^$.[]()|{}*+?")


def llamacpp_rejects(pattern):
    """The first escape llama.cpp's pattern parser would refuse, or None."""
    i, in_class = 0, False
    while i < len(pattern):
        c = pattern[i]
        if c == "\\":
            nxt = pattern[i + 1] if i + 1 < len(pattern) else ""
            if not in_class and nxt in _REGEX_ONLY and nxt:
                i += 2
                continue
            if nxt and nxt in _GBNF_ESC:
                i += 2
                continue
            n = _HEX.get(nxt)
            digits = pattern[i + 2:i + 2 + n] if n else ""
            if n and len(digits) == n and all(h in "0123456789abcdefABCDEF"
                                              for h in digits):
                i += 2 + n
                continue
            return pattern[i:i + 2]
        if c == "[" and not in_class:
            in_class = True
        elif c == "]" and in_class:
            in_class = False
        i += 1
    return None


def _props():
    out = []
    for name in ("LAYOUT_JSON_SCHEMA", "TABLE_REC_JSON_SCHEMA"):
        schema = getattr(prompts, name, None) or {}
        for key, prop in schema.get("items", {}).get("properties", {}).items():
            if "pattern" in prop:
                out.append((f"{name}.{key}", prop))
    return out


@pytest.fixture
def stock():
    """Surya's patterns as shipped, put back after each test — the repair
    is in place and process-wide, so a test that ran it would otherwise
    hide the defect from the next."""
    saved = {n: p["pattern"] for n, p in _props()}
    shipped = {n: p.replace("[0-9]", r"\d") for n, p in saved.items()}
    for n, p in _props():
        p["pattern"] = shipped[n]
    yield shipped
    for n, p in _props():
        p["pattern"] = saved[n]


def test_the_rule_model_sees_the_defect():
    assert llamacpp_rejects(r"^\d{1,4}$") == r"\d"
    assert llamacpp_rejects(r"\w") == r"\w"
    assert llamacpp_rejects(r"a\s") == r"\s"
    assert llamacpp_rejects(r"^[0-9]{1,4} \.\-$") is None


def test_every_pattern_compiles_after_the_repair(stock):
    assert stock, "surya ships no guided patterns — this file checks nothing"
    assert any(llamacpp_rejects(p) for p in stock.values()), \
        "surya's stock patterns are safe now; the repair can go"
    assert vtb.llamacpp_safe_schemas() >= 1
    for name, prop in _props():
        assert llamacpp_rejects(prop["pattern"]) is None, name


def test_the_repair_accepts_exactly_what_surya_meant(stock):
    vtb.llamacpp_safe_schemas()
    samples = ["0 0 0 0", "12 34 567 8901", "9999 1 22 333", "1 2 3",
               "12345 1 1 1", "a b c d", "1 2 3 4 ", " 1 2 3 4", "1  2 3 4",
               "١ 2 3 4"]      # an Arabic-Indic digit: \d would take it, JSON boxes never carry one
    for name, prop in _props():
        meant = re.compile(stock[name], re.ASCII)
        now = re.compile(prop["pattern"])
        assert [s for s in samples if bool(meant.match(s)) != bool(now.match(s))] == [], name


def test_the_repair_is_idempotent(stock):
    vtb.llamacpp_safe_schemas()
    assert vtb.llamacpp_safe_schemas() == 0


def _stub_predictor(module, cls_name, seen, result):
    class Stub:
        def __init__(self, *a, **k):
            seen.append([p["pattern"] for _, p in _props()])

        def __call__(self, images, *a, **k):
            return [result for _ in images]
    return types.SimpleNamespace(**{cls_name: Stub})


def test_the_backend_repairs_before_it_builds(stock, monkeypatch):
    seen = []
    monkeypatch.setitem(sys.modules, "surya.recognition",
                        _stub_predictor("surya.recognition",
                                        "RecognitionPredictor", seen, None))
    vtb.SuryaBackend(["en"])
    assert seen, "the backend built no predictor"
    assert all(llamacpp_rejects(p) is None for p in seen[0])


def test_relabel_repairs_and_stays_guided(stock, monkeypatch, tmp_path):
    from PIL import Image
    page = tmp_path / "page_0000.png"
    Image.new("RGB", (40, 40), "white").save(page)
    seen = []
    boxed = types.SimpleNamespace(bboxes=[types.SimpleNamespace(
        label="Text", bbox=[0, 0, 10, 10])])
    monkeypatch.setitem(sys.modules, "surya.layout",
                        _stub_predictor("surya.layout", "LayoutPredictor",
                                        seen, boxed))
    monkeypatch.setattr(vtb, "load_pages_from_pdf", lambda pdf, d: [str(page)])

    class Cache:
        entries = {"elsewhere": []}
        dirty = False

        def __init__(self, *a, **k):
            pass

        def _key(self, p):
            return "this page"
    monkeypatch.setattr(vtb, "OCRCache", Cache)
    from surya.settings import settings
    monkeypatch.setattr(settings, "SURYA_GUIDED_LAYOUT", True)

    st = vtb.relabel_cache("unused.ocr.gz", "unused.pdf")
    assert seen and all(llamacpp_rejects(p) is None for p in seen[0])
    assert settings.SURYA_GUIDED_LAYOUT is True
    assert st.get("page_miss") == 1
