"""A rebuild that reads nothing, and a record of what built a book.

Every fix means rebuilding books from their caches. Such a build used to
start the recognition server whether or not a page needed reading, and
would read a page its cache lacked without saying so, differently each
time: one page of one manual came out three ways in three builds. With
--cache-only a build reads nothing, and stops before it starts if the
cache cannot give it every page.

What built a book was nowhere but in the book: which passes it had were
read off its markup. The command now keeps its log and a record beside the
cache, and the book carries a line in its package.
"""
import json
import sys
import types
import zipfile

import pytest
from PIL import Image, ImageDraw

from codicology import binding, pipeline as vtb


def _page(path, rows=12):
    im = Image.new("RGB", (600, 800), "white")
    d = ImageDraw.Draw(im)
    for row in range(80, 80 + 40 * rows, 40):
        d.rectangle((60, row, 540, row + 14), fill=(30, 30, 30))
    im.save(path)
    return str(path)


class _Reader:
    """Reads what it is given, and says how often it was asked."""
    batch_size = 4
    name = "fake"
    langs = ["en"]

    def __init__(self):
        self.asked = 0

    def run_items(self, images):
        self.asked += len(images)
        return [[vtb.PageItem(html="<p>The firm has stood for some years, and its "
                                   "books were kept in a fair hand.</p>", label="Text")]
                for _ in images]


def _build(tmp_path, reader, cache, n=2):
    paths = [_page(tmp_path / f"page_{k:04d}.png", rows=10 + k) for k in range(n)]
    out = str(tmp_path / "book.epub")
    vtb.build_epub(paths, out, reader, "A Book", False, dedupe=False,
                   drop_blank=False, ocr_cache=str(cache))
    return paths, out


@pytest.fixture
def cache_only(monkeypatch):
    monkeypatch.setattr(vtb, "CACHE_ONLY", True)


# ── --cache-only ─────────────────────────────────────────────────────────────

def test_a_build_from_a_full_cache_reads_nothing(tmp_path, monkeypatch, capsys):
    cache = tmp_path / "book.ocr.gz"
    first = _Reader()
    _build(tmp_path, first, cache)
    assert first.asked == 2
    monkeypatch.setattr(vtb, "CACHE_ONLY", True)
    again = _Reader()
    _, out = _build(tmp_path, again, cache)
    assert again.asked == 0
    assert "every page has a reading in the cache" in capsys.readouterr().out
    assert len(binding.page_bodies(out)) == 2


def test_a_page_the_cache_lacks_stops_the_build_before_it_starts(tmp_path, cache_only):
    cache = tmp_path / "book.ocr.gz"
    reader = _Reader()
    with pytest.raises(SystemExit) as stopped:
        _build(tmp_path, reader, cache)
    assert "2 of this book's 2 page(s) have no reading in the cache (page_0000, page_0001)" in str(stopped.value)
    assert reader.asked == 0 and not (tmp_path / "book.epub").exists()


def test_the_recogniser_is_not_started_until_a_page_must_be_read(monkeypatch):
    built = []

    class Predictor:
        def __init__(self):
            built.append(self)

        def __call__(self, images):
            return [types.SimpleNamespace(blocks=[]) for _ in images]
    monkeypatch.setitem(sys.modules, "surya.recognition",
                        types.SimpleNamespace(RecognitionPredictor=Predictor))
    backend = vtb.SuryaBackend(["en"])
    assert not built                       # a server would have started here
    monkeypatch.setitem(vtb.READ_THIS_RUN, "pages", 0)
    backend.run_items([Image.new("RGB", (40, 40), "white")] * 3)
    backend.run_items([Image.new("RGB", (40, 40), "white")])
    assert len(built) == 1 and vtb.READ_THIS_RUN["pages"] == 4


def test_nothing_starts_the_recogniser_on_a_run_that_reads_nothing(monkeypatch, cache_only):
    class Predictor:
        def __init__(self):
            raise AssertionError("the recogniser was started")
    monkeypatch.setitem(sys.modules, "surya.recognition",
                        types.SimpleNamespace(RecognitionPredictor=Predictor))
    backend = vtb.SuryaBackend(["en"])
    with pytest.raises(SystemExit, match="a page was sent to be read"):
        backend.run_items([Image.new("RGB", (40, 40), "white")])


def test_surya_missing_is_still_said_at_once(monkeypatch):
    monkeypatch.delitem(sys.modules, "surya.recognition", raising=False)
    import importlib.util
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(RuntimeError, match="Could not find Surya's RecognitionPredictor"):
        vtb.SuryaBackend(["en"])


# ── the record ───────────────────────────────────────────────────────────────

def _convert(tmp_path, monkeypatch, *flags, fail=None):
    """The command, with the build itself stood in for."""
    (tmp_path / "scans").mkdir(exist_ok=True)
    cache = tmp_path / "book-1a2b.ocr.gz"

    def stands_in_for_the_build(**kw):
        print("    read 2 pages")
        if fail:
            sys.exit(fail)
        paths = [_page(tmp_path / f"page_{k:04d}.png", rows=10 + k) for k in range(2)]
        vtb.build_epub(paths, kw["output_epub"], _Reader(), "A Book", False, dedupe=False,
                       drop_blank=False, ocr_cache=kw.get("ocr_cache"))
    monkeypatch.setattr(vtb, "process_video", stands_in_for_the_build)
    out = str(tmp_path / "book.epub")
    vtb.main(["--from-images", str(tmp_path / "scans"), "--epub", out,
              "--ocr-cache", str(cache), *flags])
    return out, tmp_path / "book-1a2b.build.log", tmp_path / "book-1a2b.build.json"


def test_a_build_leaves_its_log_and_a_record_beside_the_cache(tmp_path, monkeypatch, capsys):
    out, log, record = _convert(tmp_path, monkeypatch, "--link-notes", "--check-folios")
    said = capsys.readouterr().out
    assert "read 2 pages" in said                    # still said where it was said
    kept = log.read_text()
    assert kept.startswith("codicology ") and "read 2 pages" in kept and "Bound 2 page(s)" in kept
    assert "--link-notes" in kept.splitlines()[2]
    r = json.loads(record.read_text())
    assert r["options"] == ["--check-folios", "--link-notes"]      # what is done, not where things are
    assert r["codicology"].startswith(vtb.__dict__.get("__version__", "") or "0")
    assert r["epub"]["pages"] == 2 and r["epub"]["bound"] is True
    assert r["pages_sent_to_be_read"] == 0 and r["cache_only"] is False
    assert r["cache"].endswith("book-1a2b.ocr.gz") and r["log"].endswith("book-1a2b.build.log")
    assert sys.stdout is not None and not isinstance(sys.stdout, vtb._Tee)


def test_the_book_says_what_built_it(tmp_path, monkeypatch):
    out, _, record = _convert(tmp_path, monkeypatch, "--typography")
    with zipfile.ZipFile(out) as z:
        opf = z.read("EPUB/content.opf").decode()
    r = json.loads(record.read_text())
    assert f'<meta name="codicology:build" content="{r["codicology"]}; --typography"/>' in opf
    assert binding.is_bound(zipfile.ZipFile(out).namelist())       # and binding kept the line


def test_a_build_that_stops_keeps_its_log_and_writes_no_record(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit):
        _convert(tmp_path, monkeypatch, fail="the witness is not installed")
    assert "[stopped] the witness is not installed" in (tmp_path / "book-1a2b.build.log").read_text()
    assert not (tmp_path / "book-1a2b.build.json").exists()
    assert not isinstance(sys.stdout, vtb._Tee) and vtb.BUILD_NOTE is None and vtb.CACHE_ONLY is False


def test_cache_only_is_a_switch_of_the_command_and_wants_a_cache(tmp_path, monkeypatch, capsys):
    _convert(tmp_path, monkeypatch)                  # which reads the pages into the cache
    out, log, record = _convert(tmp_path, monkeypatch, "--cache-only")
    assert "every page has a reading in the cache" in log.read_text()
    assert json.loads(record.read_text())["cache_only"] is True
    assert "--cache-only" in json.loads(record.read_text())["options"]
    assert vtb.CACHE_ONLY is False                   # it does not outlive the run
    with pytest.raises(SystemExit):
        vtb.main(["--from-images", str(tmp_path / "scans"), "--epub", out, "--cache-only"])
    assert "--cache-only needs --ocr-cache" in capsys.readouterr().err
