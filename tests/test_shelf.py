"""The shelf, built again and held against the last time.

A change to the pipeline is judged on the books it has already built. Two
measurements in this project's history said "no change" because they could
say nothing else, so what is tested here is mostly that a difference is
seen, and seen as the kind it is: a word, markup alone, a picture, the
binding. And that what belongs to the occasion (the date, the identifier,
the commit, which number a picture got, which file a page was bound into)
is not taken for one.
"""
import gzip
import importlib.util
import io
import json
import os
import subprocess
import sys
import types
import zipfile

import pytest
from PIL import Image

from codicology import binding as B, cli, shelf

FOLIO = ('<span epub:type="pagebreak" role="doc-pagebreak" id="pgb-%04d" '
         'aria-label="%d"></span>')
BOOK = [
    (0, "<h1>A Book</h1>"),
    (1, "<h1>One</h1><p>It was a bright cold day in April, and the clocks were</p>"),
    (2, (FOLIO % (2, 4)) + "<p>striking thirteen. See <a href=\"page_0004.xhtml#bib-1\">Orwell</a> "
        'and <a href="page_0001.xhtml">the start</a>.</p>'),
    (3, (FOLIO % (3, 5)) + "<h1>Two</h1><p>Winston Smith slipped in.</p>"),
    (4, (FOLIO % (4, 6)) + '<p id="bib-1">Orwell, George. 1949.</p>'),
]


def _picture(shade):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (shade, shade, shade)).save(buf, "PNG")
    return buf.getvalue()


def _built(path, pages=BOOK, identifier="one", pictures=(), bound=True, most=B.MOST):
    """A book written the way build_epub writes one, then bound."""
    from ebooklib import epub
    from codicology import pipeline as P
    P._restrict_page_list_to_pagebreaks()
    book = epub.EpubBook()
    book.set_identifier(identifier)
    book.set_title("A Book")
    book.set_language("en")
    book.add_metadata("OPF", "meta", "", {"name": "codicology:build",
                                          "content": f"0.1.0 {identifier}; --typography"})
    for name, raw in pictures:
        book.add_item(epub.EpubItem(uid=name, file_name=f"images/{name}",
                                    media_type="image/png", content=raw))
    files = []
    for i, body in pages:
        ch = epub.EpubHtml(title=f"Page {i + 1}", file_name=f"page_{i:04d}.xhtml")
        ch.content = f"<html><body>{body}</body></html>"
        book.add_item(ch)
        files.append(ch)
    book.toc = (epub.Link("page_0001.xhtml", "One", "c1"),
                epub.Link("page_0003.xhtml", "Two", "c3"))
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav"] + files
    epub.write_epub(str(path), book)
    if bound:
        B.bind_epub(str(path), most=most)
    return str(path)


def _with(pages, n, body):
    return [(i, body if i == n else b) for i, b in pages]


# ── the shelf file ───────────────────────────────────────────────────────────

def _shelf(tmp_path, books):
    path = tmp_path / "shelf.json"
    path.write_text(json.dumps({"books": books}))
    return str(path)


def test_a_shelf_file_is_read_with_every_field_filled_in(tmp_path):
    books = shelf.load(_shelf(tmp_path, [
        {"id": 12, "pdf": "a.pdf", "cache": "a.ocr.gz"},
        {"id": "13", "title": "B", "pdf": "b.pdf", "cache": "b.ocr.gz", "lang": "de",
         "options": ["--link-notes", "--cache-only"], "decisions": "b.json",
         "beside": ["b.second.gz"], "installed": "ignored.epub", "note": "anything else is ignored"}]))
    assert books[0] == {"id": "12", "title": "12", "pdf": "a.pdf", "cache": "a.ocr.gz", "lang": "en",
                        "options": [], "decisions": None, "beside": [], "installed": None}
    assert books[1]["installed"] == "ignored.epub"
    assert books[1]["options"] == ["--link-notes"]         # --cache-only is not a choice
    assert books[1]["lang"] == "de" and books[1]["beside"] == ["b.second.gz"]


@pytest.mark.parametrize("books, why", [
    ([{"id": "1", "pdf": "a.pdf"}], "no cache"),
    ([{"id": "1", "pdf": "a", "cache": "c"}, {"id": 1, "pdf": "a", "cache": "c"}], "two books"),
    ([{"id": "a/b", "pdf": "a", "cache": "c"}], "cannot be a book's id"),
    ([{"id": "1", "pdf": "a", "cache": "c", "options": ["--epub=x.epub"]}], "as an option"),
    ({"not": "a list"}, "no list of books"),
])
def test_a_shelf_that_cannot_be_built_from_is_refused(tmp_path, books, why):
    path = tmp_path / "shelf.json"
    path.write_text(json.dumps(books if isinstance(books, dict) else {"books": books}))
    with pytest.raises(shelf.ShelfError, match=why):
        shelf.load(str(path))


def test_only_the_books_asked_for_and_none_that_are_not_there(tmp_path):
    books = shelf.load(_shelf(tmp_path, [{"id": str(n), "pdf": "p", "cache": "c"} for n in (1, 2, 3)]))
    assert [b["id"] for b in shelf.chosen(books, ["3", 1])] == ["1", "3"]
    with pytest.raises(shelf.ShelfError, match="not on the shelf: 9"):
        shelf.chosen(books, ["9"])


# ── a print ──────────────────────────────────────────────────────────────────

def test_two_builds_of_one_book_have_one_print(tmp_path):
    """The identifier, the date and the commit are of the occasion."""
    a = shelf.print_of(_built(tmp_path / "a.epub", identifier="first"))
    b = shelf.print_of(_built(tmp_path / "b.epub", identifier="second"))
    assert a["built_by"] == "0.1.0 first; --typography"
    assert shelf.compare_prints(a, b) == {}
    # nor is where a reader stopped in an installed copy any part of the book
    with zipfile.ZipFile(str(tmp_path / "b.epub"), "a") as z:
        z.writestr(B.BOOKMARKS, "encoding=json+base64:\nW10=")
    assert shelf.compare_prints(a, shelf.print_of(str(tmp_path / "b.epub"))) == {}
    assert a["facts"]["pages"] == 5 and a["facts"]["page numbers"] == 3
    assert a["facts"]["contents entries"] == 2 and a["facts"]["page list"] == 3
    assert a["facts"]["chapter files"] == 3 and a["facts"]["links"] == 2


def test_a_word_that_changed_is_told_from_markup_that_did(tmp_path):
    a = shelf.print_of(_built(tmp_path / "a.epub"))
    words = shelf.print_of(_built(tmp_path / "b.epub", _with(
        BOOK, 3, (FOLIO % (3, 5)) + "<h1>Two</h1><p>Winston Smith crept in.</p>")))
    markup = shelf.print_of(_built(tmp_path / "c.epub", _with(
        BOOK, 3, (FOLIO % (3, 5)) + "<h1>Two</h1><p>Winston <i>Smith</i> slipped in.</p>")))
    diff = shelf.compare_prints(a, words)
    assert [p["page"] for p in diff["words"]] == [3] and "markup" not in diff
    assert diff["words"][0]["changes"] == [
        {"was": "slipped", "is": "crept", "before": "Two Winston Smith", "after": "in."}]
    assert shelf.compare_prints(a, markup) == {"markup": [3]}


def test_a_word_gone_is_counted_and_a_page_gone_is_said(tmp_path):
    a = shelf.print_of(_built(tmp_path / "a.epub"))
    b = shelf.print_of(_built(tmp_path / "b.epub", BOOK[:4]))
    diff = shelf.compare_prints(a, b)
    assert diff["pages"] == {"gone": [4], "new": []}
    assert diff["facts"]["pages"] == [5, 4] and diff["facts"]["words"][0] > diff["facts"]["words"][1]


def test_a_picture_is_named_by_its_bytes_and_not_by_its_number(tmp_path):
    """Drop the first picture of a book and every later one is renumbered.
    That is one page that differs, not every page with a picture on it."""
    two = [(i, b + (f'<img src="images/fig_{i - 1:04d}.png"/>' if i in (1, 2) else "")) for i, b in BOOK]
    one = [(i, b + ('<img src="images/fig_0000.png"/>' if i == 2 else "")) for i, b in BOOK]
    a = shelf.print_of(_built(tmp_path / "a.epub", two,
                              pictures=[("fig_0000.png", _picture(10)), ("fig_0001.png", _picture(200))]))
    b = shelf.print_of(_built(tmp_path / "b.epub", one, pictures=[("fig_0000.png", _picture(200))]))
    diff = shelf.compare_prints(a, b)
    assert diff["markup"] == [1]                          # page 2 shows the same picture as before
    assert diff["pictures"] == {"gone": 1, "new": 0}
    assert "picture:" in a["pages"]["2"] and a["pages"]["2"] == b["pages"]["2"]


def test_a_link_is_named_by_its_page_and_not_by_the_file_it_was_bound_into(tmp_path):
    """The same pages in fewer, longer files are the same pages."""
    a = shelf.print_of(_built(tmp_path / "a.epub", most=1))
    b = shelf.print_of(_built(tmp_path / "b.epub", most=60))
    flat = shelf.print_of(_built(tmp_path / "c.epub", bound=False))
    assert a["facts"]["chapter files"] > b["facts"]["chapter files"]
    assert a["pages"] == b["pages"]
    assert 'href="page_0004.xhtml#bib-1"' in a["pages"]["2"]
    diff = shelf.compare_prints(a, b)
    assert diff["binding"] is True and "words" not in diff and "markup" not in diff
    assert "chapter files" in diff["facts"]
    # and a book never bound at all has the same pages as one bound and taken apart
    never = shelf.compare_prints(flat, b)
    assert never["binding"] is True and "words" not in never and "markup" not in never


# ── a run ────────────────────────────────────────────────────────────────────

class _Builder:
    """Stands in for the command: writes the book it is told to, and does
    to the cache what a build may."""

    def __init__(self, tmp_path, fail=(), pages=BOOK, unread=()):
        self.tmp, self.fail, self.pages, self.lines = tmp_path, set(fail), pages, []
        self.unread = set(unread)       # books with a page the cache has no reading of

    def __call__(self, cmd, stdout=None, stderr=None, env=None, **kw):
        self.lines.append(cmd)
        if "verify" in cmd:
            return types.SimpleNamespace(returncode=0, stdout="  all there\nOK\n")
        epub = cmd[cmd.index("--epub") + 1]
        bid = os.path.basename(epub)[:-len(".epub")]
        cache = cmd[cmd.index("--ocr-cache") + 1]
        with open(cache, "ab") as fh:              # a build may write the cache it was given
            fh.write(b"refiled")
        if bid in self.unread:
            if "--cache-only" in cmd:
                stdout.write("[stopped] --cache-only: 1 of this book's 5 page(s) have no reading in "
                             "the cache (page_0003). Nothing was read and nothing was built.\n")
                return types.SimpleNamespace(returncode=1)
            with open(cache[:-len(".ocr.gz")] + ".build.json", "w") as fh:
                json.dump({"pages_sent_to_be_read": 1}, fh)
        else:
            assert "--cache-only" in cmd
        if bid in self.fail:
            stdout.write("reading\n--cache-only: 3 of this book's pages have no reading\n")
            return types.SimpleNamespace(returncode=1)
        stdout.write("building\n    stale decision: p3 'a' -> 'b' kept as read\n")
        _built(epub, self.pages if isinstance(self.pages, list) else self.pages[bid], identifier=bid)
        return types.SimpleNamespace(returncode=0)


def _three(tmp_path):
    books = []
    for bid in ("7", "8", "9"):
        (tmp_path / f"{bid}.pdf").write_bytes(b"%PDF")
        (tmp_path / f"{bid}-ab.ocr.gz").write_bytes(b"cache of " + bid.encode())
        books.append({"id": bid, "title": f"Book {bid}", "pdf": str(tmp_path / f"{bid}.pdf"),
                      "cache": str(tmp_path / f"{bid}-ab.ocr.gz"), "options": ["--link-notes"]})
    return books


def test_a_run_builds_from_borrowed_caches_and_keeps_a_print_of_each(tmp_path):
    books = _three(tmp_path)
    (tmp_path / "7-ab.second.gz").write_bytes(b"second")
    (tmp_path / "7.decisions.json").write_text("{}")
    books[0].update(beside=[str(tmp_path / "7-ab.second.gz")], decisions=str(tmp_path / "7.decisions.json"))
    run, said = _Builder(tmp_path), []
    out = str(tmp_path / "run")
    record = shelf.build(_shelf(tmp_path, books), out, jobs=2, run=run, say=said.append, python="py")
    assert record["built"] == 3 and record["failed"] == [] and record["books"] == ["7", "8", "9"]
    # the originals are as they were; what the build wrote, it wrote on its own copy
    assert (tmp_path / "7-ab.ocr.gz").read_bytes() == b"cache of 7"
    assert (tmp_path / "run" / "cache" / "7-ab.ocr.gz").read_bytes() == b"cache of 7refiled"
    assert (tmp_path / "run" / "cache" / "7-ab.second.gz").read_bytes() == b"second"
    line = next(c for c in run.lines if c[c.index("--epub") + 1].endswith("7.epub"))
    assert line[:4] == ["py", "-m", "codicology.cli", "convert"]
    assert line[line.index("--pages-from") + 1] == books[0]["pdf"]
    assert line[line.index("--ocr-cache") + 1] == os.path.join(out, "cache", "7-ab.ocr.gz")
    assert line[line.index("--title") + 1] == "Book 7" and "--link-notes" in line
    assert line[-2:] == ["--apply-decisions", books[0]["decisions"]]
    kept = shelf.kept(out, "7")
    assert kept["row"]["rc"] == 0 and kept["row"]["stale decisions"] == 1
    assert kept["row"]["inputs"]["cache"]["bytes"] == len(b"cache of 7")
    assert kept["print"]["facts"]["pages"] == 5 and os.path.exists(os.path.join(out, "books", "7.epub"))
    assert "building" in (tmp_path / "run" / "logs" / "7.log").read_text()
    assert any(s.startswith("3 of 3 built") for s in said)
    assert json.load(open(os.path.join(out, "run.json")))["built"] == 3


def test_a_book_that_does_not_build_is_said_and_the_others_are_built(tmp_path):
    books = _three(tmp_path)
    os.remove(books[2]["pdf"])
    said = []
    out = str(tmp_path / "run")
    record = shelf.build(_shelf(tmp_path, books), out, run=_Builder(tmp_path, fail={"8"}), say=said.append)
    assert record["built"] == 1 and record["failed"] == ["8", "9"]
    assert "no reading" in shelf.kept(out, "8")["row"]["said"][-1] and shelf.kept(out, "8")["print"] is None
    assert "there is no" in shelf.kept(out, "9")["row"]["said"][0]
    assert any("NOT BUILT" in s and s.split("] ")[1].startswith("8 ") for s in said)


def test_a_run_can_be_taken_up_again_and_can_keep_prints_alone(tmp_path):
    books = _three(tmp_path)
    out = str(tmp_path / "run")
    shelf.build(_shelf(tmp_path, books), out, run=_Builder(tmp_path, fail={"8"}), say=lambda s: None,
                prints_only=True)
    assert not os.path.exists(os.path.join(out, "books", "7.epub"))        # the print is what is kept
    assert not os.path.exists(os.path.join(out, "cache", "7-ab.ocr.gz"))
    assert shelf.kept(out, "7")["print"]["facts"]["pages"] == 5
    again = _Builder(tmp_path)
    record = shelf.build(_shelf(tmp_path, books), out, run=again, say=lambda s: None, resume=True)
    assert [c[c.index("--epub") + 1][-6:] for c in again.lines] == ["8.epub"]
    assert record["built"] == 3


def test_a_book_with_a_page_never_cached_is_built_the_ordinary_way_when_asked(tmp_path):
    """A page that reads empty is never cached, so a build from the cache
    alone has nothing for it. Asked to, the run builds such a book again
    when the others are done, reading what the cache lacks."""
    books = _three(tmp_path)
    out, said, waited = str(tmp_path / "run"), [], []
    run = _Builder(tmp_path, unread={"8"}, fail={"9"})
    record = shelf.build(_shelf(tmp_path, books), out, run=run, say=said.append)
    assert record["failed"] == ["8", "9"] and record["read_live"] == []
    assert shelf.lacks_a_reading(shelf.kept(out, "8")["row"])
    assert not shelf.lacks_a_reading(shelf.kept(out, "9")["row"])      # refused for something else
    assert "1 have a page with no reading in the cache: --read-missing" in said[-1]

    run = _Builder(tmp_path, unread={"8"}, fail={"9"})
    record = shelf.build(_shelf(tmp_path, books), out, run=run, say=said.append, read_missing=True,
                         settle=lambda: waited.append(1))
    assert record["failed"] == ["9"] and record["read_live"] == ["8"] and record["built"] == 2
    lines = [(os.path.basename(c[c.index("--epub") + 1]), "--cache-only" in c) for c in run.lines]
    assert lines[-1] == ("8.epub", False) and lines.count(("8.epub", True)) == 1    # last, and once each way
    assert ("9.epub", False) not in lines                              # what failed otherwise is not tried again
    assert waited == [1]                                               # the reader before it had gone
    row = shelf.kept(out, "8")["row"]
    assert row["rc"] == 0 and row["read live"] == 1
    assert any(s.startswith("[read 1/1] 8 ") and "(1 sent to the reader)" in s for s in said)
    assert "1 of them with pages read (8)" in said[-1]
    # and a comparison says so, because such a page may read differently the next time
    base = str(tmp_path / "base")
    shelf.build(_shelf(tmp_path, books), base, run=_Builder(tmp_path, unread={"8"}), say=lambda s: None,
                read_missing=True, settle=lambda: None)
    told = []
    result = shelf.compare(base, out)
    assert result["books"]["8"]["read live"] == {"base": 1, "new": 1}
    result["books"]["8"]["state"] = "differs"
    shelf.tell(result, say=told.append)
    assert any("pages sent to be read, the cache having none of them: base 1, new 1" in t for t in told)


def test_verify_is_run_when_asked_and_what_it_said_is_kept(tmp_path):
    books = _three(tmp_path)[:1]
    out = str(tmp_path / "run")
    shelf.build(_shelf(tmp_path, books), out, run=_Builder(tmp_path), say=lambda s: None, verify=True)
    assert shelf.kept(out, "7")["row"]["verify"] == {"rc": 0, "said": ["all there", "OK"]}


def test_the_code_a_run_builds_with_can_be_another_checkouts(tmp_path, monkeypatch):
    seen = {}

    def run(cmd, stdout=None, stderr=None, env=None, **kw):
        seen["path"] = env["PYTHONPATH"]
        seen["workers"] = env["CODICOLOGY_PDF_WORKERS"]
        return _Builder(tmp_path)(cmd, stdout=stdout)
    monkeypatch.setenv("PYTHONPATH", "/already/there")
    monkeypatch.delenv("CODICOLOGY_PDF_WORKERS", raising=False)
    shelf.build(_shelf(tmp_path, _three(tmp_path)[:1]), str(tmp_path / "run"), run=run,
                say=lambda s: None, code=str(tmp_path / "other" / "src"), jobs=4)
    assert seen["path"] == os.pathsep.join([str(tmp_path / "other" / "src"), "/already/there"])
    # and each build is told how many processes it may render with, the cores shared between them
    assert seen["workers"] == str(max(1, (os.cpu_count() or 1) // 4))


# ── two runs ─────────────────────────────────────────────────────────────────

def _two_runs(tmp_path):
    books = _three(tmp_path)
    base, new = str(tmp_path / "base"), str(tmp_path / "new")
    shelf.build(_shelf(tmp_path, books), base, run=_Builder(tmp_path), say=lambda s: None)
    changed = {"7": BOOK, "9": BOOK,
               "8": _with(BOOK, 3, (FOLIO % (3, 5)) + "<h1>Two</h1><p>Winston Smith crept in.</p>")}
    shelf.build(_shelf(tmp_path, books + [{"id": "10", "pdf": books[0]["pdf"], "cache": books[0]["cache"]}]),
                new, run=_Builder(tmp_path, fail={"9"}, pages=dict(changed, **{"10": BOOK})),
                say=lambda s: None)
    return base, new


def test_two_runs_are_compared_book_by_book(tmp_path):
    base, new = _two_runs(tmp_path)
    result = shelf.compare(base, new)
    assert {k: v["state"] for k, v in result["books"].items()} == {
        "7": "the same", "8": "differs", "9": "not built", "10": "only in the new run"}
    assert result["books"]["8"]["words"][0]["page"] == 3
    assert result["built_by"]["base"] == ["0.1.0 7", "0.1.0 8", "0.1.0 9"]
    said = []
    assert shelf.tell(result, say=said.append) == 3
    text = "\n".join(said)
    assert "4 book(s): 1 the same, 1 differs, 1 not built, 1 only in the new run" in text
    assert "words differ on 1 page(s), 1 change(s)" in text
    assert "p3: …Two Winston Smith [slipped] => [crept] in.…" in text
    assert "no reading" in text
    assert list(shelf.compare(base, new, only=["7"])["books"]) == ["7"]


def test_a_book_built_from_another_cache_is_said_to_be(tmp_path):
    books = _three(tmp_path)[:1]
    base, new = str(tmp_path / "base"), str(tmp_path / "new")
    shelf.build(_shelf(tmp_path, books), base, run=_Builder(tmp_path), say=lambda s: None)
    with open(books[0]["cache"], "ab") as fh:
        fh.write(b" and one page more")
    shelf.build(_shelf(tmp_path, books), new, run=_Builder(tmp_path), say=lambda s: None)
    result = shelf.compare(base, new)
    assert result["books"]["7"]["state"] == "differs"
    said = []
    shelf.tell(result, say=said.append)
    assert "    built from a different cache" in said


def test_a_run_is_held_against_the_shelf_as_it_stands(tmp_path):
    """Which books a rebuild would change: the copies installed, printed
    without building anything, are a run like any other."""
    books = _three(tmp_path)
    for b in books[:2]:
        b["installed"] = _built(tmp_path / f"installed-{b['id']}.epub", identifier="then")
    said = []
    stands, new = str(tmp_path / "stands"), str(tmp_path / "new")
    record = shelf.installed(_shelf(tmp_path, books), stands, say=said.append)
    assert record["built"] == 2 and record["failed"] == ["9"]
    assert said == ["2 of 3 installed copies printed; none for: 9"]
    changed = {"7": BOOK, "9": BOOK,
               "8": _with(BOOK, 3, (FOLIO % (3, 5)) + "<h1>Two</h1><p>Winston Smith crept in.</p>")}
    shelf.build(_shelf(tmp_path, books), new, run=_Builder(tmp_path, pages=changed), say=lambda s: None)
    result = shelf.compare(stands, new)
    assert {k: v["state"] for k, v in result["books"].items()} == {
        "7": "the same", "8": "differs", "9": "not built"}
    # an installed copy has no cache or options to have been built differently from
    assert "built differently" not in result["books"]["8"]


def test_the_command_says_by_its_status_whether_anything_differs(tmp_path, capsys):
    base, new = _two_runs(tmp_path)
    report = tmp_path / "report.json"
    assert cli.main(["shelf", "compare", base, new, "--json", str(report)]) == 1
    assert json.loads(report.read_text())["books"]["8"]["state"] == "differs"
    assert cli.main(["shelf", "compare", base, new, "--only", "7"]) == 0
    assert cli.main(["shelf", "compare", base, str(tmp_path / "nowhere")]) == 2
    assert "is not a run" in capsys.readouterr().out
    assert cli.main(["shelf", "list", _shelf(tmp_path, _three(tmp_path))]) == 0
    assert "3 book(s)" in capsys.readouterr().out


# ── the whole of it, for real ────────────────────────────────────────────────

pymupdf = pytest.importorskip("pymupdf")


@pytest.mark.skipif(importlib.util.find_spec("surya") is None, reason="the command wants surya to be there")
def test_a_real_book_built_twice_from_its_cache_is_the_same_book(tmp_path, monkeypatch):
    """The command itself, in a process of its own, from a cache alone."""
    from codicology import pipeline as vtb
    doc = pymupdf.open()
    for n in range(3):
        page = doc.new_page(width=420, height=595)
        page.insert_textbox(pymupdf.Rect(50, 60, 370, 540), (
            f"Chapter {n + 1}\n\nThe firm has stood for some years, and its books were kept in a "
            "fair hand by a clerk who had been with it from the first. ") * 3, fontsize=11)
    pdf = str(tmp_path / "book.pdf")
    doc.save(pdf)
    cache = str(tmp_path / "book-1a2b.ocr.gz")

    class Reader(vtb.SuryaBackend):
        def run_items(self, images):
            return [[vtb.PageItem(html="<p>The firm has stood for some years, and its books were kept "
                                       "in a fair hand by a clerk who had been with it from the first.</p>",
                                  label="Text", box=(0.1, 0.1, 0.9, 0.5))] for _ in images]
    monkeypatch.setitem(vtb.OCR_BACKENDS, "surya", Reader)
    monkeypatch.setattr(vtb, "warm_backend", lambda backend: None)      # it is asked to prove it reads
    vtb.main(["--pages-from", pdf, "--epub", str(tmp_path / "first.epub"), "--ocr-cache", cache,
              "--without-witness"])
    monkeypatch.undo()

    path = _shelf(tmp_path, [{"id": "1", "title": "A Book", "pdf": pdf, "cache": cache,
                              "options": ["--without-witness", "--typography"]}])
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    for out in (a, b):
        record = shelf.build(path, out, say=lambda s: None)
        assert record["failed"] == [], shelf.kept(out, "1")["row"]["said"]
    one = shelf.kept(a, "1")
    assert one["print"]["facts"]["pages"] == 3 and "--typography" in one["print"]["built_by"]
    assert json.load(open(os.path.join(a, "cache", "book-1a2b.build.json")))["pages_sent_to_be_read"] == 0
    assert shelf.tell(shelf.compare(a, b), say=lambda s: None) == 0
