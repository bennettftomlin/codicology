"""The OCR cache: what the recogniser made of each page, filed under the page
picture's own name (its pixels, not its file's bytes), so a rebuild reads
nothing it has read before.
"""
from __future__ import annotations

import base64
import gzip
import hashlib
import io
import json
import os
import re
import zlib
from PIL import Image
from .backends import OCR_BACKENDS
from . import switches
from .model import PageItem


# Pages read before the OCR cache is written out again. Small enough that an
# interrupted run loses minutes rather than an hour, large enough that
# re-compressing the file is not what the run spends its time on.
CACHE_CHECKPOINT_PAGES = 25


# What a cache entry must have been written by to be served. Bumped when a
# read gains something the whole pipeline downstream depends on — v3 kept
# running heads, v4 records Surya's layout labels. A page below the mark is
# re-read once and stamped current, whatever its ink turns out to hold.
CACHE_VERSION = 4


# ── what a page's reading is filed under ─────────────────────────────────────
#
# A reading was filed under the SHA-256 of the page's file. For a page
# rendered from a PDF that file is a PNG this pipeline encoded itself, so
# the name was a fact about Pillow's zlib as much as about the page. A warm
# rebuild of Kill or Get Killed spent 107 of its 202 seconds encoding 440
# PNGs it would otherwise barely open, to learn names it had worked out the
# day before. And the day the encoder changes — Pillow 11.1 moved its wheels
# to zlib-ng, and only Surya's pillow<11 holds that off — every rendered
# page of every book would have gone unrecognised and been read again.
#
# A PNG is now filed under what it shows: its mode, its size, its pixels.
# Anything else — a JPEG lifted whole out of a PDF, a photograph — is filed
# under its bytes as before: those bytes were given to us, not made, and a
# JPEG does not decode to the same pixels on every machine.
#
# A cache written before this still holds its readings under the old names.
# Each is refiled the first time its page is asked for: the page's old name
# is worked out once more (the loader does it in its workers when the cache
# needs it), the reading is moved, and nothing is read again.
PAGE_KEYING = 2


_PAGE_KEY = re.compile(r"[0-9a-f]{64}")


def _pixel_key(im: Image.Image) -> str:
    """A name for an image that depends on nothing but what it shows."""
    if im.mode not in ("1", "L", "RGB"):
        im = im.convert("RGB")
    digest = hashlib.sha256(
        f"codicology page {im.mode} {im.width}x{im.height}\n".encode("ascii"))
    digest.update(im.tobytes())
    return digest.hexdigest()


def _key_beside(page_path: str, suffix: str) -> "str | None":
    """A key the page loader left in a sidecar, if it left one."""
    try:
        with open(page_path + suffix, encoding="ascii") as fh:
            said = fh.read().strip()
    except (OSError, ValueError):
        return None
    return said if _PAGE_KEY.fullmatch(said) else None


def _bytes_key(page_path: str) -> str:
    with open(page_path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def page_key(page_path: str) -> str:
    """The name a page's reading is filed under."""
    said = _key_beside(page_path, ".key")
    if said:
        return said
    if page_path.lower().endswith(".png"):
        try:
            with Image.open(page_path) as im:
                return _pixel_key(im)
        except Exception:
            pass            # not a picture after all: it has only its bytes
    return _bytes_key(page_path)


def _as_formerly_written(im: Image.Image) -> bytes:
    """The PNG the loader used to write for a rendered page, whose bytes
    were the page's name."""
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "PNG", dpi=(300, 300))
    return buf.getvalue()


def former_page_key(page_path: str) -> str:
    """The name a page's reading had when readings were filed under bytes."""
    said = _key_beside(page_path, ".key1")
    if said:
        return said
    try:
        with open(page_path + ".source.json") as fh:
            rendered = not json.load(fh).get("facsimile", True)
    except (OSError, ValueError):
        rendered = False
    if rendered:        # the file on disk is no longer the one that was hashed
        with Image.open(page_path) as im:
            return hashlib.sha256(_as_formerly_written(im)).hexdigest()
    return _bytes_key(page_path)


class OCRCache:
    """
    Remembers what the OCR engine made of each page, so a second opinion about
    duplicates does not cost a second reading of the book.

    Recognising the page is by far the slowest part of this pipeline and by far
    the most settled: the words on a page do not change because the rule for
    discarding repeats did. Everything downstream — which pages are duplicates,
    which the reader struck out, how the book is assembled — is cheap to redo
    and is exactly what gets iterated on.

    Keyed on the page image (see page_key) together with the engine, so a
    re-crop or a different backend misses rather than serving a stale read.
    The languages join the engine in the tag only where they steer the
    reading. They used to be there always, and a tag that does not match
    loads nothing and is overwritten by the next save: typing "en,de" for
    one book would have discarded that book's whole reading, an hour's work,
    for a setting Surya never sees. Its only other effect is dc:language.
    All 155 caches on the shelf say "surya|en", and still load.
    """

    def __init__(self, path: str, backend_name: str, langs: list[str],
                 serve_stale: bool = False) -> None:
        self.path = path
        blind = getattr(OCR_BACKENDS.get(backend_name), "language_blind", False)
        self.tag = (backend_name if blind
                    else f"{backend_name}|{','.join(langs)}")
        self.entries: dict[str, list[dict]] = {}
        # the entries still filed under a file's bytes: all of them, in a
        # cache written before pages were named for their pixels
        self.by_bytes: set[str] = set()
        self.refiled = 0
        self._key_memo: dict = {}
        self.hits = 0
        self.stale = 0
        self.layer_served: list[str] = []   # pages remembered as unreadable
        self.serve_stale = serve_stale
        self.dirty = False
        self.unsaved = 0
        if os.path.exists(path):
            try:
                with gzip.open(path, "rt", encoding="utf-8") as fh:
                    stored = json.load(fh)
                was = stored.get("tag") or ""
                if was == self.tag or (
                        blind and was.split("|", 1)[0] == backend_name):
                    # The file keeps the tag it came with. A build made
                    # before the languages left the tag reads nothing from
                    # a file tagged any other way, and then writes over it.
                    self.tag = was
                    self.entries = stored.get("pages", {})
                    self.by_bytes = (
                        set(stored.get("by_bytes", ())) & set(self.entries)
                        if stored.get("keying") == PAGE_KEYING
                        else set(self.entries))
            except (OSError, ValueError, KeyError, EOFError, zlib.error):
                # an unreadable cache is a cold cache, never an error — and
                # that includes one cut short (a copy interrupted, a disk
                # that filled), which raised EOFError straight through here
                self.entries, self.by_bytes = {}, set()

    def _key(self, page_path: str) -> str:
        # a warm build asks for the same page's key several times (get,
        # knows_furniture, put) and each answer used to re-read and re-hash
        # the whole file — a thousand-plus redundant hashes per rebuild
        # (E1). The digest is remembered per (mtime, size); a touched file
        # re-hashes, an unchanged one costs a stat.
        st = os.stat(page_path)
        memo = self._key_memo.get(page_path)
        if memo and memo[0] == (st.st_mtime_ns, st.st_size):
            return memo[1]
        digest = page_key(page_path)
        self._key_memo[page_path] = ((st.st_mtime_ns, st.st_size), digest)
        return digest

    @staticmethod
    def filed_by_bytes(path: str) -> bool:
        """Whether the cache at `path` predates pixel names altogether, so
        that every page's old name will be wanted. Read from the first few
        bytes, not by loading it."""
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                head = fh.read(200)
        except (OSError, ValueError, EOFError):
            return False
        return bool(head) and '"keying"' not in head

    def _entry(self, page_path: str):
        """The page's entry, refiled under its present name if it was still
        under its old one."""
        key = self._key(page_path)
        entry = self.entries.get(key)
        if entry is not None:
            if key in self.by_bytes:    # its bytes are its name still
                self.by_bytes.discard(key)
                self.dirty = True
            return entry
        if not self.by_bytes:
            return None
        was = former_page_key(page_path)
        if was not in self.by_bytes:
            return None
        entry = self.entries[key] = self.entries.pop(was)
        self.by_bytes.discard(was)
        self.refiled += 1
        self.dirty = True
        return entry

    def get(self, page_path: str) -> list[PageItem] | None:
        entry = self._entry(page_path)
        if entry is None:
            # a page named --read-as-empty: what reading it gives, every time
            stem = os.path.splitext(os.path.basename(page_path))[0]
            if stem in switches.READ_AS_EMPTY:
                if stem not in switches.TAKEN_AS_EMPTY:
                    switches.TAKEN_AS_EMPTY.append(stem)
                return []
            return None
        # A page read before the pipeline recorded layout labels is not a
        # hit: it would build silently without the labels every consumer
        # downstream now expects — the footnote rules, the reattached
        # markers, the heading hierarchy. Re-read it. What makes this
        # converge rather than re-read forever is the version stamp and
        # not the labels themselves: a page whose ink genuinely yields no
        # labelled block is stamped current all the same, and is a hit
        # ever after.
        version = entry.get("v", 0) if isinstance(entry, dict) else 0
        if version < CACHE_VERSION and not self.serve_stale:
            self.stale += 1
            return None
        raw = entry["items"] if isinstance(entry, dict) else entry
        self.hits += 1
        items = []
        for it in raw:
            fig = None
            if it.get("fig"):
                fig = Image.open(io.BytesIO(base64.b64decode(it["fig"])))
                fig.load()
            items.append(PageItem(html=it.get("html", ""), figure=fig,
                                  is_caption=it.get("cap", False),
                                  is_furniture=it.get("furn", False),
                                  box=tuple(it["box"]) if it.get("box") else None,
                                  conf=it.get("conf"),
                                  label=it.get("lab")))
        if isinstance(entry, dict) and entry.get("outcome") == "layer":
            # A page the reader could not give, remembered as such (see
            # put): what was kept of its reading, and the page's own text
            # standing in for the rest, exactly as the build that read it.
            from .guards import _native_layer_body
            served = _native_layer_body(page_path) or []
            items += served
            self.layer_served.append(os.path.basename(page_path))
            print(f"    [~] {os.path.basename(page_path)}: remembered as a page "
                  f"the reader cannot give — "
                  + ("its own text layer stands in" if served else
                     "left without its text, as when it was read"))
        return items

    def knows_furniture(self, page_path: str) -> bool:
        """Whether this entry was written by a version that keeps running heads.
        A page taken as empty (--read-as-empty) has none, as reading it gives."""
        entry = self._entry(page_path)
        if entry is None:
            return os.path.splitext(os.path.basename(page_path))[0] in switches.READ_AS_EMPTY
        return isinstance(entry, dict)

    def put(self, page_path: str, items: list[PageItem],
            outcome: "str | None" = None) -> None:
        """Keep a page's reading. `outcome` "layer" keeps instead the fact
        that the reader could not give this page, with what was worth
        keeping of its reading (its pictures), so that a later build serves
        the page's own text layer in its place without asking the reader
        again; see _read_pages_resiliently."""
        raw = []
        for it in items:
            enc = None
            if it.figure is not None:
                buf = io.BytesIO()
                # Higher quality than the copy that reaches the book: this one is
                # encoded again on the way out, and stacking two lossy passes on
                # a halftone photograph is visible where one is not.
                it.figure.convert("RGB").save(buf, "JPEG", quality=97)
                enc = base64.b64encode(buf.getvalue()).decode("ascii")
            raw.append({"html": it.html, "fig": enc, "cap": it.is_caption,
                        "furn": it.is_furniture,
                        "box": list(it.box) if it.box else None,
                        "conf": it.conf, "lab": it.label})
        # The version marker is what tells a page with no running head apart
        # from a page read before heads were kept at all — without it, an old
        # cache would silently audit as a book with no folios anywhere. It
        # now also carries the same distinction for layout labels, which is
        # what lets a stale page be re-read exactly once.
        entry = {"v": CACHE_VERSION, "items": raw}
        if outcome:
            entry["outcome"] = outcome
        self.entries[self._key(page_path)] = entry
        self.dirty = True
        self.unsaved += 1
        # Checkpoint as we go. Reading a book takes the better part of an hour,
        # and a cache written only at the end is a cache that a stopped run
        # leaves with nothing in it — losing exactly the work it exists to save.
        if self.unsaved >= CACHE_CHECKPOINT_PAGES:
            self.save()

    def save(self) -> None:
        if not self.dirty:
            return
        self.unsaved = 0
        tmp = self.path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8") as fh:
            json.dump({"tag": self.tag, "keying": PAGE_KEYING,
                       "by_bytes": sorted(self.by_bytes),
                       "pages": self.entries}, fh)
        os.replace(tmp, self.path)  # never leave a half-written cache behind
        # clean now: without this, every later checkpoint rewrote the whole
        # gzip even when nothing had changed since the previous save (A3)
        self.dirty = False
