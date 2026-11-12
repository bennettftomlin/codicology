"""The shelf, built again and held against the last time it was built.

Every change to this pipeline is judged on the books it has already made:
whether the ones it was written for changed as meant, and whether any other
changed at all. Until 12 November 2026 that was done with scripts written
for the occasion and kept beside each batch of rebuilds. This is those
scripts as one command.

    codicology shelf build SHELF.json OUT [--against BASE]
    codicology shelf compare BASE NEW
    codicology shelf installed SHELF.json OUT

A shelf is a file that names books: for each, the PDF it came from, the OCR
cache it was read into, and the options it is built with. `build` builds
every one from its cache with --cache-only, so a run is the caches and the
code and nothing else. No page is read and no model is loaded, which is
what lets several books build at once.

What a run keeps of a book is its *print*: the markup of every page, and a
digest of every other file in it. Three things are left out of a print
because they belong to the occasion and not to the book: the date, the
identifier, and the line that says which commit built it. Two more are
written in a form that does not depend on how the book happens to be filed:
a picture is named by its bytes, not by its number in the book, and a link
by the page it lands on, not by the chapter file that page was bound into.
One picture dropped on page 40 is then one page that differs, and not every
page after it.

`compare` holds the prints of two runs against each other and says, book by
book, what differs and of what kind: words, markup alone, the binding,
pictures, contents.

One kind of book cannot be built that way. A page that reads empty is
never cached (a failing recogniser reads empty too, and one cached run of
those blanked a book for every build after it), so a book with such a page
has no reading of it to build from, and --cache-only refuses the book. With
`--read-missing` those books are built again when the rest are done, the
ordinary way, one at a time: every cached page from the cache, and the one
or two without a reading read as any build would read them. The run says
which books those were and how many pages each had read, because such a
page may read differently the next time, and a difference on it between
two runs is then the reading's and not the code's.

`installed` takes the same prints of the copies a shelf file says are
installed, without building anything, so that a run can be held against
the shelf as it stands: which books a rebuild would change, and how.

Nothing a run reads is written. Each cache is borrowed into OUT first
(cloned where the filing system can do that for nothing, copied where it
cannot), because a build may refile an old cache under its pages' present
names and always leaves its record beside the cache it used.
"""
from __future__ import annotations

import concurrent.futures
import difflib
import gzip
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

from . import binding

JOBS = 4                # books built at once
SHOW = 3                # changes shown for a book, of each kind

# What a shelf may not say for itself: where things are is the harness's
# to decide, and --cache-only is not a choice.
_NOT_AN_OPTION = ("--epub", "--ocr-cache", "--pages-from", "--from-images",
                  "--apply-decisions", "--title", "--lang", "--progress-json",
                  "--review-sheet", "--patch", "--page-map")

# The stamps of the occasion, in the package and the old-style contents.
_STAMPS = [
    re.compile(rb'\s*<meta property="dcterms:modified">[^<]*</meta>'),
    re.compile(rb'\s*<meta\b(?=[^>]*\bname="codicology:build")[^>]*/>'),
    re.compile(rb'\s*<dc:identifier\b[^>]*>[^<]*</dc:identifier>'),
    re.compile(rb'\s*<meta\b(?=[^>]*\bname="dtb:uid")[^>]*/>'),
]
_BUILT_BY = re.compile(r'<meta\b(?=[^>]*\bname="codicology:build")[^>]*'
                       r'\bcontent="([^"]*)"')
_PICTURE = re.compile(r"\.(?:jpe?g|png|gif|svg|webp)$", re.I)
_SRC = re.compile(r'\b(src|xlink:href|href)="([^"#:?]+)"')


class ShelfError(Exception):
    """A shelf file, or a run's folder, that cannot be used as it is."""


def _say(text: str = "") -> None:
    # a run is watched from a file as often as from a terminal, and a
    # line kept back until the buffer fills is a run that looks stopped
    print(text, flush=True)


# ── the shelf ────────────────────────────────────────────────────────────────

def load(path: str) -> "list[dict]":
    """The books a shelf file names, each with every field filled in."""
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise ShelfError(f"{path}: {exc}") from None
    rows = data.get("books") if isinstance(data, dict) else data
    if not isinstance(rows, list):
        raise ShelfError(f"{path}: no list of books in it")
    books, seen = [], set()
    for n, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ShelfError(f"{path}: book {n} is not an object")
        lacking = [k for k in ("id", "pdf", "cache") if not row.get(k)]
        if lacking:
            raise ShelfError(f"{path}: book {row.get('id', n)!r} has no "
                             f"{' and no '.join(lacking)}")
        bid = str(row["id"])
        if not re.fullmatch(r"[\w.-]+", bid):
            raise ShelfError(f"{path}: {bid!r} cannot be a book's id: it "
                             f"names the files kept of it")
        if bid in seen:
            raise ShelfError(f"{path}: two books are called {bid}")
        seen.add(bid)
        options = [str(o) for o in row.get("options") or []
                   if o != "--cache-only"]
        for o in options:
            if o.split("=", 1)[0] in _NOT_AN_OPTION:
                raise ShelfError(
                    f"{path}: book {bid} gives {o} as an option. Where a "
                    f"book's files are is said by its pdf, cache, decisions "
                    f"and title")
        books.append({"id": bid, "title": row.get("title") or bid,
                      "pdf": row["pdf"], "cache": row["cache"],
                      "lang": row.get("lang") or "en",
                      "options": options,
                      "decisions": row.get("decisions") or None,
                      "beside": list(row.get("beside") or []),
                      "installed": row.get("installed") or None})
    return books


def chosen(books: "list[dict]", only) -> "list[dict]":
    if not only:
        return books
    want = [str(o) for o in only]
    have = {b["id"] for b in books}
    unknown = [o for o in want if o not in have]
    if unknown:
        raise ShelfError(f"not on the shelf: {', '.join(unknown)}")
    return [b for b in books if b["id"] in set(want)]


# ── a print ──────────────────────────────────────────────────────────────────

def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()[:24]


def _words(body: str) -> "list[str]":
    return html.unescape(re.sub(r"<[^>]+>", " ", body)).split()


def _page_files(path: str) -> "tuple[str, bool]":
    """A copy of the book with a file to a page, where it is bound into
    chapters and comes apart; the book itself where it is already so, or
    will not. The second value says a copy was made."""
    with zipfile.ZipFile(path) as z:
        if not binding.is_bound(z.namelist()):
            return path, False
    fd, tmp = tempfile.mkstemp(suffix=".epub", prefix="shelf-")
    os.close(fd)
    try:
        binding.unbind_epub(path, out=tmp)
        return tmp, True
    except Exception:
        # a print is still wanted of a book that will not come apart: its
        # pages are read out of the chapters, links as they stand
        if os.path.exists(tmp):
            os.remove(tmp)
        return path, False


def print_of(path: str) -> dict:
    """What `compare` needs of a built book, and nothing of the occasion."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        bound = {n: _digest(z.read(n)) for n in names
                 if binding.BOUND_FILE.match(n)}
        package = next((n for n in names if n.endswith(".opf")), None)
        built_by = None
        if package:
            said = _BUILT_BY.search(z.read(package).decode("utf-8", "replace"))
            built_by = html.unescape(said.group(1)) if said else None
            if bound:
                raw = z.read(package)
                for stamp in _STAMPS:
                    raw = stamp.sub(b"", raw)
                bound[package] = _digest(raw)
    flat, copied = _page_files(path)
    try:
        with zipfile.ZipFile(flat) as z:
            names = z.namelist()
            files, pictures = {}, {}
            for n in names:
                if binding.PAGE_FILE.match(n) or binding.BOUND_FILE.match(n):
                    continue
                if n == binding.BOOKMARKS:
                    # where a reader stopped is the reader's, not the book's
                    continue
                raw = z.read(n)
                if n.endswith((".opf", ".ncx")):
                    for stamp in _STAMPS:
                        raw = stamp.sub(b"", raw)
                if _PICTURE.search(n):
                    pictures[n] = _digest(raw)
                else:
                    files[n] = _digest(raw)
            where = os.path.dirname(next(
                (n for n in names if binding.PAGE_FILE.match(n)
                 or binding.BOUND_FILE.match(n)), ""))

            def by_its_bytes(m):
                name = os.path.normpath(os.path.join(where, m.group(2)))
                name = name.replace(os.sep, "/")
                if name in pictures:
                    return f'{m.group(1)}="picture:{pictures[name]}"'
                return m.group(0)

            pages = {str(k): _SRC.sub(by_its_bytes, body)
                     for k, body in sorted(binding.page_bodies(z).items())}
            nav = next((z.read(n).decode("utf-8", "replace")
                        for n in names if n.endswith("nav.xhtml")), "")
    finally:
        if copied:
            os.remove(flat)
    text = "".join(pages.values())
    at = nav.find('epub:type="page-list"')
    facts = {
        "pages": len(pages),
        "words": sum(len(_words(b)) for b in pages.values()),
        "chapter files": sum(1 for n in bound if binding.BOUND_FILE.match(n)),
        "contents entries": len(re.findall(r"<li\b", nav[:at] if at >= 0 else nav)),
        "page list": len(re.findall(r"<li\b", nav[at:])) if at >= 0 else 0,
        "page numbers": len(re.findall(r'epub:type="pagebreak"', text)),
        "note links": len(re.findall(r'epub:type="noteref"', text)),
        "links": len(re.findall(r"<a\b[^>]*\bhref=", text)),
        "pictures": len(pictures),
    }
    return {"pages": pages, "files": files,
            "pictures": sorted(pictures.values()), "bound": bound,
            "facts": facts, "built_by": built_by}


# ── a run ────────────────────────────────────────────────────────────────────

def _borrow(src: str, dst: str) -> None:
    """A copy of `src` at `dst` that a build may do as it likes with."""
    if os.path.exists(dst):
        a, b = os.stat(src), os.stat(dst)
        if a.st_size == b.st_size and int(a.st_mtime) == int(b.st_mtime):
            return
        os.remove(dst)
    if sys.platform == "darwin":
        # APFS makes a clone that shares the original's blocks until one of
        # the two is written: 1.7 GB of caches borrowed for nothing
        done = subprocess.run(["cp", "-c", "-p", src, dst],
                              capture_output=True)
        if done.returncode == 0 and os.path.exists(dst):
            return
    shutil.copy2(src, dst)


def _stamp(path: "str | None") -> "dict | None":
    """What tells one state of an input file from another, cheaply."""
    if not path or not os.path.isfile(path):
        return None
    st = os.stat(path)
    out = {"bytes": st.st_size, "modified": int(st.st_mtime)}
    if st.st_size < 4_000_000:
        with open(path, "rb") as fh:
            out["sha"] = _digest(fh.read())
    return out


def _print_path(out: str, bid: str) -> str:
    return os.path.join(out, "prints", bid + ".json.gz")


def _keep(out: str, row: dict, print_: "dict | None") -> None:
    path = _print_path(out, row["id"])
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
        json.dump({"row": row, "print": print_}, fh, ensure_ascii=False)
    os.replace(tmp, path)


def kept(out: str, bid: str) -> "dict | None":
    try:
        with gzip.open(_print_path(out, bid), "rt", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError, EOFError):
        return None


def command(book: dict, epub: str, cache: str, python: str,
            read: bool = False) -> "list[str]":
    """The line that builds this book: from its cache alone, or (`read`)
    the ordinary way, reading what the cache lacks."""
    cmd = [python, "-m", "codicology.cli", "convert",
           "--pages-from", book["pdf"], "--epub", epub,
           "--ocr-cache", cache, *(() if read else ("--cache-only",)),
           "--title", book["title"], "--lang", book["lang"],
           *book["options"]]
    if book["decisions"]:
        cmd += ["--apply-decisions", book["decisions"]]
    return cmd


def build_one(book: dict, out: str, code: "str | None" = None,
              python: "str | None" = None, verify: bool = False,
              prints_only: bool = False, pdf_workers: "int | None" = None,
              read: bool = False, run=subprocess.run) -> dict:
    """Build one book into `out` and keep its print. Returns its row.
    With `prints_only` the book and the borrowed cache are not kept once
    the print is taken: a base run of a hundred and fifty books is then
    forty megabytes and not several thousand."""
    bid = book["id"]
    cache = os.path.join(out, "cache", os.path.basename(book["cache"]))
    epub = os.path.join(out, "books", bid + ".epub")
    log = os.path.join(out, "logs", bid + ".log")
    row = {"id": bid, "title": book["title"], "options": book["options"],
           "inputs": {"pdf": _stamp(book["pdf"]),
                      "cache": _stamp(book["cache"]),
                      "decisions": _stamp(book["decisions"])},
           "rc": None, "seconds": 0, "said": []}
    lacking = [p for p in (book["pdf"], book["cache"], book["decisions"],
                           *book["beside"]) if p and not os.path.isfile(p)]
    if lacking:
        row["said"] = [f"not built: there is no {p}" for p in lacking]
        _keep(out, row, None)
        return row
    _borrow(book["cache"], cache)
    for extra in book["beside"]:
        _borrow(extra, os.path.join(out, "cache", os.path.basename(extra)))
    if os.path.exists(epub):
        os.remove(epub)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if pdf_workers:
        # each build renders its pages in processes of its own; with
        # several builds going, each is told how many it may start
        env.setdefault("CODICOLOGY_PDF_WORKERS", str(pdf_workers))
    if code:
        env["PYTHONPATH"] = os.pathsep.join(
            [os.path.abspath(code)]
            + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    python = python or sys.executable
    began = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        row["rc"] = run(command(book, epub, cache, python, read=read),
                        stdout=fh, stderr=subprocess.STDOUT,
                        env=env).returncode
    row["seconds"] = round(time.time() - began)
    if read:
        # how many pages this build had read, as its own record says
        row["read live"] = None
        try:
            with open(cache[:-len(".ocr.gz")] + ".build.json"
                      if cache.endswith(".ocr.gz") else cache + ".build.json",
                      encoding="utf-8") as fh:
                row["read live"] = json.load(fh).get("pages_sent_to_be_read")
        except (OSError, ValueError):
            pass
    with open(log, encoding="utf-8", errors="replace") as fh:
        lines = [ln.strip() for ln in fh.read().splitlines() if ln.strip()]
    row["stale decisions"] = sum(
        1 for ln in lines if ln.startswith(("stale decision", "refused:")))
    print_ = None
    if row["rc"] == 0 and os.path.exists(epub):
        try:
            print_ = print_of(epub)
        except Exception as exc:        # a book that cannot be opened
            row["rc"] = -1
            row["said"] = [f"built, and could not be read back: "
                           f"{type(exc).__name__}: {exc}"]
        if print_ is not None and verify:
            v = run([python, "-m", "codicology.cli", "verify", epub,
                     book["pdf"]], capture_output=True, text=True, env=env)
            row["verify"] = {
                "rc": v.returncode,
                "said": [ln.strip() for ln in (v.stdout or "").splitlines()
                         if ln.strip()][-2:]}
    else:
        if row["rc"] == 0:
            row["rc"] = -1
            lines.append("the build said it was done and left no book")
        row["said"] = [ln[:300] for ln in lines[-3:]]
    _keep(out, row, print_)
    if prints_only and print_ is not None:
        for used in (epub, cache, *(os.path.join(
                out, "cache", os.path.basename(x)) for x in book["beside"])):
            if os.path.exists(used):
                os.remove(used)
    return row


def lacks_a_reading(row: dict) -> bool:
    """Whether a book was refused for a page its cache has no reading of,
    and for nothing else."""
    return row.get("rc") not in (0, None) and any(
        "--cache-only" in ln and "no reading in the cache" in ln
        for ln in row.get("said") or [])


def _wait_for_the_reader_to_go(tries: int = 60, pause: float = 2.0) -> None:
    """A build that read pages stops its inference server as it exits, and
    one that starts while that server is still going down attaches to it
    and finds it gone. So the next waits until none is left."""
    for _ in range(tries):
        try:
            ps = subprocess.run(["ps", "-axo", "args="], capture_output=True,
                                text=True).stdout
        except OSError:
            return
        if not any(os.path.basename(ln.split(" ", 1)[0]) == "llama-server"
                   for ln in ps.splitlines() if ln.strip()):
            return
        time.sleep(pause)


def build(shelf: str, out: str, jobs: int = JOBS, only=None,
          code: "str | None" = None, verify: bool = False,
          resume: bool = False, prints_only: bool = False,
          read_missing: bool = False, python: "str | None" = None,
          run=subprocess.run, say=_say, settle=_wait_for_the_reader_to_go
          ) -> dict:
    """Build the shelf (or `only` some of it) into `out`."""
    books = chosen(load(shelf), only)
    for sub in ("cache", "books", "logs", "prints"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    rows: dict = {}
    todo = []
    for book in books:
        was = kept(out, book["id"]) if resume else None
        if was and was["row"].get("rc") == 0 and was.get("print"):
            rows[book["id"]] = was["row"]
        else:
            todo.append(book)
    if resume and len(todo) < len(books):
        say(f"{len(books) - len(todo)} book(s) are already built in {out} "
            f"and are left as they are")

    # the long ones first, so the last job to finish is a short one
    def weight(book):
        try:
            return os.path.getsize(book["cache"])
        except OSError:
            return 0
    todo.sort(key=weight, reverse=True)
    began = time.time()
    each = max(1, (os.cpu_count() or 1) // max(1, jobs))
    with concurrent.futures.ThreadPoolExecutor(max(1, jobs)) as pool:
        waiting = [pool.submit(build_one, book, out, code=code,
                               python=python, verify=verify,
                               prints_only=prints_only, pdf_workers=each,
                               run=run)
                   for book in todo]
        for n, done in enumerate(concurrent.futures.as_completed(waiting), 1):
            row = done.result()
            rows[row["id"]] = row
            how = (f"built in {row['seconds']} s" if row["rc"] == 0
                   else "NOT BUILT: " + " | ".join(row["said"])[:200])
            say(f"[{n}/{len(todo)}] {row['id']} {row['title'][:44]}: {how}")
    read = []
    if read_missing:
        # one at a time: each starts the recogniser, and two of those at
        # once are two servers on one port
        again = [b for b in todo if lacks_a_reading(rows[b["id"]])]
        for n, book in enumerate(again, 1):
            settle()
            row = build_one(book, out, code=code, python=python,
                            verify=verify, prints_only=prints_only,
                            read=True, run=run)
            rows[row["id"]] = row
            if row["rc"] == 0:
                read.append(row["id"])
            sent = row.get("read live")
            # a page is sent again when it comes back empty, so this is
            # how often the reader was asked, not how many pages lacked
            # a reading
            how = (f"built in {row['seconds']} s, what the cache lacked "
                   f"read as an ordinary build reads it"
                   + ("" if sent is None
                      else f" ({sent} sent to the reader)")
                   if row["rc"] == 0
                   else "NOT BUILT: " + " | ".join(row["said"])[:200])
            say(f"[read {n}/{len(again)}] {row['id']} "
                f"{row['title'][:40]}: {how}")
    record = {"shelf": os.path.abspath(shelf),
              "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "seconds": round(time.time() - began), "jobs": jobs,
              "code": os.path.abspath(code) if code else None,
              "books": [b["id"] for b in books],
              "built": sum(1 for r in rows.values() if r["rc"] == 0),
              "read_live": sorted(read),
              "failed": sorted(r["id"] for r in rows.values() if r["rc"] != 0)}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    lacking = [r["id"] for r in rows.values() if lacks_a_reading(r)]
    say(f"{record['built']} of {len(books)} built in "
        f"{record['seconds'] // 60} min {record['seconds'] % 60} s"
        + (f", {len(read)} of them with pages read ({', '.join(sorted(read))})"
           if read else "")
        + (f"; not built: {', '.join(record['failed'])}"
           if record["failed"] else "")
        + (f". {len(lacking)} have a page with no reading in the cache: "
           f"--read-missing builds those the ordinary way"
           if lacking and not read_missing else ""))
    return record


def installed(shelf: str, out: str, only=None, say=_say) -> dict:
    """Prints of the copies the shelf file says are installed, kept the
    way a run's are, so that a run can be compared with them."""
    books = chosen(load(shelf), only)
    os.makedirs(os.path.join(out, "prints"), exist_ok=True)
    taken, lacking = 0, []
    for book in books:
        path = book["installed"]
        row = {"id": book["id"], "title": book["title"],
               "options": book["options"], "installed": path,
               "rc": None, "seconds": 0, "said": []}
        print_ = None
        if not path or not os.path.isfile(path):
            row["said"] = ["no installed copy" if not path
                           else f"there is no {path}"]
        else:
            try:
                print_ = print_of(path)
                row["rc"] = 0
                row["copy"] = _stamp(path)
            except Exception as exc:
                row["said"] = [f"could not be read: "
                               f"{type(exc).__name__}: {exc}"]
        _keep(out, row, print_)
        if print_ is None:
            lacking.append(book["id"])
        else:
            taken += 1
    record = {"shelf": os.path.abspath(shelf), "installed": True,
              "when": time.strftime("%Y-%m-%dT%H:%M:%S"),
              "books": [b["id"] for b in books], "built": taken,
              "failed": lacking}
    with open(os.path.join(out, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    say(f"{taken} of {len(books)} installed copies printed"
        + (f"; none for: {', '.join(lacking)}" if lacking else ""))
    return record


# ── two runs, held against each other ────────────────────────────────────────

def _changes(a: "list[str]", b: "list[str]") -> "list[dict]":
    out = []
    matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag != "equal":
            out.append({"was": " ".join(a[i1:i2]), "is": " ".join(b[j1:j2]),
                        "before": " ".join(a[max(0, i1 - 3):i1]),
                        "after": " ".join(a[i2:i2 + 3])})
    return out


def _lost(a: "list[str]", b: "list[str]") -> int:
    """How many of `a` are not in `b`, counting one for one."""
    left = list(b)
    lost = 0
    for x in a:
        if x in left:
            left.remove(x)
        else:
            lost += 1
    return lost


def compare_prints(a: dict, b: dict) -> dict:
    """What differs between two prints of one book. Empty where nothing
    does."""
    out: dict = {}
    pa, pb = a["pages"], b["pages"]
    words, markup = [], []
    for k in sorted(set(pa) & set(pb), key=int):
        # the same page, as the binding counts pages the same: white space
        # between blocks says nothing, and a book left a file to a page
        # has more of it than one bound and taken apart again
        if pa[k] == pb[k] or binding._norm(pa[k]) == binding._norm(pb[k]):
            continue
        wa, wb = _words(pa[k]), _words(pb[k])
        if wa == wb:
            markup.append(int(k))
        else:
            words.append({"page": int(k), "changes": _changes(wa, wb)})
    if words:
        out["words"] = words
    if markup:
        out["markup"] = markup
    gone = sorted((int(k) for k in set(pa) - set(pb)))
    new = sorted((int(k) for k in set(pb) - set(pa)))
    if gone or new:
        out["pages"] = {"gone": gone, "new": new}
    lost = _lost(a["pictures"], b["pictures"])
    gained = _lost(b["pictures"], a["pictures"])
    if lost or gained:
        out["pictures"] = {"gone": lost, "new": gained}
    fa, fb = a["files"], b["files"]
    other = sorted(n for n in set(fa) | set(fb) if fa.get(n) != fb.get(n))
    if other:
        out["files"] = other
    if a["bound"] != b["bound"] and not (words or markup or gone or new):
        out["binding"] = True
    facts = {k: [a["facts"].get(k), b["facts"].get(k)]
             for k in sorted(set(a["facts"]) | set(b["facts"]))
             if a["facts"].get(k) != b["facts"].get(k)}
    if facts:
        out["facts"] = facts
    return out


def compare(base: str, new: str, only=None) -> dict:
    """Every book of two runs: the same, different (and how), not built,
    or in one run alone."""
    def ids(folder):
        there = os.path.join(folder, "prints")
        if not os.path.isdir(there):
            raise ShelfError(f"{folder} is not a run: it has no prints")
        return {n[:-len(".json.gz")] for n in os.listdir(there)
                if n.endswith(".json.gz")}
    in_a, in_b = ids(base), ids(new)
    if only:
        want = {str(o) for o in only}
        in_a, in_b = in_a & want, in_b & want
    books = {}
    built_by = {"base": set(), "new": set()}
    for bid in sorted(in_a | in_b, key=lambda s: (len(s), s)):
        a = kept(base, bid) if bid in in_a else None
        b = kept(new, bid) if bid in in_b else None
        for side, k in (("base", a), ("new", b)):
            note = ((k or {}).get("print") or {}).get("built_by")
            if note:
                built_by[side].add(note.split(";")[0].strip())
        title = ((b or a or {}).get("row") or {}).get("title", bid)
        entry = {"title": title}
        if a is None or b is None:
            entry["state"] = "only in the base" if b is None \
                else "only in the new run"
        elif not a.get("print") or not b.get("print"):
            entry["state"] = "not built"
            entry["said"] = {
                "base": None if a.get("print") else a["row"].get("said"),
                "new": None if b.get("print") else b["row"].get("said")}
        else:
            diff = compare_prints(a["print"], b["print"])
            ra, rb = a["row"], b["row"]
            # how each was built is only comparable between two builds
            built = not (ra.get("installed") or rb.get("installed"))
            for what in (("options", "inputs", "stale decisions", "verify")
                         if built else ()):
                if ra.get(what) != rb.get(what):
                    diff.setdefault("built differently", {})[what] = \
                        [ra.get(what), rb.get(what)]
            entry["state"] = "differs" if diff else "the same"
            entry.update(diff)
            if ra.get("read live") or rb.get("read live") \
                    or "read live" in ra or "read live" in rb:
                entry["read live"] = {"base": ra.get("read live"),
                                      "new": rb.get("read live")}
        books[bid] = entry
    return {"base": os.path.abspath(base), "new": os.path.abspath(new),
            "built_by": {k: sorted(v) for k, v in built_by.items()},
            "books": books}


def tell(result: dict, show: int = SHOW, say=_say) -> int:
    """Print what `compare` found. Returns how many books are not the
    same."""
    books = result["books"]
    count = {}
    for entry in books.values():
        count[entry["state"]] = count.get(entry["state"], 0) + 1
    order = ["the same", "differs", "not built", "only in the base",
             "only in the new run"]
    say(f"{len(books)} book(s): "
        + ", ".join(f"{count[s]} {s}" for s in order if count.get(s)))
    for side in ("base", "new"):
        say(f"  {side}: {result[side]}"
            + (f"  (built by {', '.join(result['built_by'][side])})"
               if result["built_by"].get(side) else ""))
    for bid, e in books.items():
        if e["state"] == "the same":
            continue
        say(f"\n{bid}  {e['title'][:70]}")
        if e.get("read live"):
            said = ", ".join(f"{side} {n if n is not None else 'some'}"
                             for side, n in e["read live"].items()
                             if n or (n is None and side in e["read live"]))
            say(f"    pages sent to be read, the cache having none of "
                f"them: {said} (a difference on such a page may be the "
                f"reading's)")
        if e["state"] != "differs":
            say(f"    {e['state']}")
            for side, said in (e.get("said") or {}).items():
                for ln in said or []:
                    say(f"      {side}: {ln[:200]}")
            continue
        if e.get("words"):
            n = sum(len(p["changes"]) for p in e["words"])
            say(f"    words differ on {len(e['words'])} page(s), "
                f"{n} change(s)")
            shown = 0
            for p in e["words"]:
                for c in p["changes"]:
                    if shown == show:
                        break
                    say(f"      p{p['page']}: …{c['before'][-40:]} "
                        f"[{c['was'][:60]}] => [{c['is'][:60]}] "
                        f"{c['after'][:40]}…")
                    shown += 1
            if n > shown:
                say(f"      and {n - shown} more")
        if e.get("markup"):
            pages = e["markup"]
            say(f"    markup alone differs on {len(pages)} page(s): "
                + ", ".join(f"p{p}" for p in pages[:12])
                + (" …" if len(pages) > 12 else ""))
        if e.get("pages"):
            say(f"    pages: {len(e['pages']['gone'])} gone, "
                f"{len(e['pages']['new'])} new")
        if e.get("pictures"):
            say(f"    pictures: {e['pictures']['gone']} gone, "
                f"{e['pictures']['new']} new")
        if e.get("binding"):
            say("    the pages are the same and are bound differently")
        if e.get("files"):
            say("    other files differ: " + ", ".join(e["files"][:8])
                + (" …" if len(e["files"]) > 8 else ""))
        if e.get("facts"):
            say("    " + "; ".join(f"{k} {v[0]} -> {v[1]}"
                                   for k, v in e["facts"].items()))
        for what, (was, now) in (e.get("built differently") or {}).items():
            if what == "inputs":
                changed = [k for k in set(was or {}) | set(now or {})
                           if (was or {}).get(k) != (now or {}).get(k)]
                say(f"    built from a different {' and '.join(sorted(changed))}")
            elif what == "verify":
                say(f"    verify: {(was or {}).get('said')} -> "
                    f"{(now or {}).get('said')}")
            else:
                say(f"    {what}: {was} -> {now}")
    return len(books) - count.get("the same", 0)


def listing(shelf: str, say=_say) -> None:
    books = load(shelf)
    for b in books:
        say(f"{b['id']:>6}  {b['title'][:50]:50}  "
            f"{' '.join(o[2:] for o in b['options'])}"
            + ("  +decisions" if b["decisions"] else ""))
    say(f"{len(books)} book(s)")
