"""The second reader: found where it is installed, cached beside the first
reader's cache, and read for its equations, tables and words."""
import io
import json

import pytest
from PIL import Image

from codicology import second

PAGE = r"""## PRODUCTION WITH A SURPLUS

Accordingly we add the rate of profits as an unknown which we call $r$ and
the system becomes

$$
(A_a p_a + B_a p_b + \ldots + K_a p_k)(1 + r) = A p_a
$$

$$
(A_b p_a + B_b p_b + \ldots + K_b p_k)(1 + r) = B p_b
$$

<table><tr><td>adults</td><td>2 tablets</td></tr>
<tr><td>children 9 to 14 years</td><td>1 ½ tablets</td></tr></table>

where **every** industry has the *same* rate, \(r\).
"""


def test_the_equations_set_on_their_own_lines_come_out_in_order():
    eqs = second.display_equations(PAGE)
    assert eqs == [r"(A_a p_a + B_a p_b + \ldots + K_a p_k)(1 + r) = A p_a",
                   r"(A_b p_a + B_b p_b + \ldots + K_b p_k)(1 + r) = B p_b"]
    assert second.display_equations(r"so \[ x = 1 \] and no more") == ["x = 1"]
    assert second.display_equations("") == [] and second.display_equations(None) == []


def test_tables_and_plain_words():
    (table,) = second.tables(PAGE)
    assert "1 ½ tablets" in table
    said = second.plain_words(PAGE)
    assert "PRODUCTION WITH A SURPLUS" in said and "#" not in said
    assert "children 9 to 14 years 1 ½ tablets" in said
    assert "every industry has the same rate" in said


def test_no_model_installed_is_no_second_reader(tmp_path, monkeypatch):
    monkeypatch.delenv(second.MODEL_ENV, raising=False)
    monkeypatch.delenv(second.MMPROJ_ENV, raising=False)
    monkeypatch.setattr(second, "DEFAULT_DIR", str(tmp_path))
    assert second.model_files() is None and not second.available()
    with pytest.raises(RuntimeError, match="no second reader is installed"):
        second.SecondReader()


def test_the_two_files_are_found_in_the_default_place(tmp_path, monkeypatch):
    monkeypatch.delenv(second.MODEL_ENV, raising=False)
    monkeypatch.delenv(second.MMPROJ_ENV, raising=False)
    monkeypatch.setattr(second, "DEFAULT_DIR", str(tmp_path))
    (tmp_path / "LightOnOCR-2-1B-Q8_0.gguf").write_bytes(b"x")
    assert second.model_files() is None, "a model with no projector cannot read"
    (tmp_path / "mmproj-LightOnOCR-2-1B-Q8_0.gguf").write_bytes(b"x")
    model, mmproj = second.model_files()
    assert model.endswith("LightOnOCR-2-1B-Q8_0.gguf") and "mmproj" in mmproj


def test_the_cache_keeps_what_was_read_and_what_was_refused(tmp_path):
    path = second.SecondCache.beside(str(tmp_path / "936-abc.ocr.gz"))
    assert path.endswith("936-abc.second.gz")
    cache = second.SecondCache(path)
    with pytest.raises(KeyError):
        cache.get("never asked")
    cache.put("k1", PAGE)
    cache.put("k2", None)               # asked, and it ran to the limit
    cache.save()
    again = second.SecondCache(path)
    assert again.get("k1") == PAGE and again.get("k2") is None
    assert second.SecondCache(path, tag="another-model").pages == {}


class _Answer:
    def __init__(self, payload):
        self._payload = json.dumps(payload).encode()
    def __enter__(self):
        return io.BytesIO(self._payload)
    def __exit__(self, *exc):
        return False


def _reader(monkeypatch, finish="stop"):
    reader = second.SecondReader(model="m.gguf", mmproj="p.gguf", server="llama-server")
    monkeypatch.setattr(reader, "start", lambda: None)
    reader._url = "http://127.0.0.1:1"
    seen = {}
    def urlopen(req, timeout=None):
        seen["body"] = json.loads(req.data)
        return _Answer({"choices": [{"finish_reason": finish,
                                     "message": {"content": PAGE}}]})
    monkeypatch.setattr(second.urllib.request, "urlopen", urlopen)
    return reader, seen


def test_a_page_is_sent_at_the_size_the_model_reads(monkeypatch):
    reader, seen = _reader(monkeypatch)
    assert reader.read(Image.new("RGB", (2550, 3300), "white")) == PAGE
    url = seen["body"]["messages"][0]["content"][0]["image_url"]["url"]
    import base64
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(sent.size) == second.PAGE_LONG_SIDE and reader.pages_read == 1


def test_a_reading_that_ran_to_the_limit_is_not_a_reading(monkeypatch):
    """The handwritten letter it was tried on produced 4,096 tokens."""
    reader, _ = _reader(monkeypatch, finish="length")
    assert reader.read(Image.new("RGB", (600, 800), "white")) is None


# ── a reader that could not be asked ─────────────────────────────────────────

def _silent_reader(monkeypatch, fails):
    """A reader whose server answers nothing for the first `fails` pages."""
    reader = second.SecondReader(model="m.gguf", mmproj="p.gguf", server="llama-server")
    monkeypatch.setattr(reader, "start", lambda: None)
    reader._url = "http://127.0.0.1:1"
    calls = []

    def urlopen(req, timeout=None):
        calls.append(1)
        if len(calls) <= fails:
            raise TimeoutError("timed out")
        return _Answer({"choices": [{"finish_reason": "stop",
                                     "message": {"content": PAGE}}]})
    monkeypatch.setattr(second.urllib.request, "urlopen", urlopen)
    return reader, calls


def test_no_answer_is_not_a_reading_and_not_the_lack_of_one(monkeypatch):
    """It used to come back as None, which is what a page it declined to
    read comes back as, and was filed as that for good."""
    reader, _ = _silent_reader(monkeypatch, fails=1)
    page = Image.new("RGB", (600, 800), "white")
    with pytest.raises(second.ReaderUnavailable):
        reader.read(page)
    assert reader.down is None, "one page is a page"
    assert reader.read(page) == PAGE, "and the next is asked as usual"


def test_two_pages_running_without_an_answer_and_it_is_not_asked_again(monkeypatch):
    reader, calls = _silent_reader(monkeypatch, fails=5)
    page = Image.new("RGB", (600, 800), "white")
    for _ in range(2):
        with pytest.raises(second.ReaderUnavailable):
            reader.read(page)
    assert reader.down and "2 pages running" in reader.down
    with pytest.raises(second.ReaderUnavailable):
        reader.read(page)
    assert len(calls) == 2, "the third page did not wait on a server that is down"


def test_a_server_that_will_not_start_is_tried_once(monkeypatch):
    reader = second.SecondReader(model="m.gguf", mmproj="p.gguf", server="llama-server")
    starts = []

    def start():
        starts.append(1)
        raise RuntimeError("the second reader's server did not start (exit 1)")
    monkeypatch.setattr(reader, "start", start)
    page = Image.new("RGB", (600, 800), "white")
    for _ in range(3):
        with pytest.raises(second.ReaderUnavailable, match="did not start"):
            reader.read(page)
    assert len(starts) == 1


def test_an_answer_that_is_not_one_counts_as_no_answer(monkeypatch):
    reader = second.SecondReader(model="m.gguf", mmproj="p.gguf", server="llama-server")
    monkeypatch.setattr(reader, "start", lambda: None)
    reader._url = "http://127.0.0.1:1"
    monkeypatch.setattr(second.urllib.request, "urlopen",
                        lambda req, timeout=None: _Answer({"error": "busy"}))
    with pytest.raises(second.ReaderUnavailable):
        reader.read(Image.new("RGB", (600, 800), "white"))

