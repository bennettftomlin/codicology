"""A second reader, for what the first one reads worst.

The first reader gives a page its layout: blocks in reading order, each with
a label and a box, which everything downstream stands on. It is also a
generative model, and on three kinds of page it writes what is likely where
it should write what is there: an equation's small figures and subscripts, a
table of figures, and a line it cannot make out. Nothing classical can check
it there. Tesseract does not read mathematics, and a scan has no text layer.

A second generative reader of a different family can. Read side by side over
twelve pages the first had got wrong, LightOnOCR-2 had the printed fraction
where the first had dropped a digit from it in three equations, a system of
equations whose subscripts the first had scrambled, the leading 1 of a dose,
a ledger whose money columns the first had slid one cell over, a table that
had lost three of its six columns, and Arabic the first had left out. It
shared one of the first reader's mistakes (a power set in very small type).
On handwriting both invent, and invent differently.

So it does not replace anything. It has no layout to give, and it is a
generative reader too: where the two differ and nothing else can say which
is right, its word is no better than the first's. It reads the page once
more, whole, the way it was trained to, and what it says is kept beside the
first reading for the parts of the pipeline that can use a second opinion:
the tables and equations of a build (pipeline.mend_tables, mend_equations),
and the words two readers dispute (adjudicate.heard_by_the_readers). This
module is the reader and its cache and nothing else: what is done with a
second reading is decided there.

The model is not part of this package. It is looked for at
$CODICOLOGY_SECOND_MODEL / $CODICOLOGY_SECOND_MMPROJ, else as the two .gguf
files in ~/.cache/codicology/second/, and is served by the same llama-server
the first reader uses. Without it, nothing here is called.
"""
from __future__ import annotations

import atexit
import base64
import glob
import gzip
import io
import json
import os
import re
import shutil
import socket
import subprocess
import time
import urllib.request

NAME = "lightonocr-2"
MODEL_ENV = "CODICOLOGY_SECOND_MODEL"
MMPROJ_ENV = "CODICOLOGY_SECOND_MMPROJ"
DEFAULT_DIR = os.path.expanduser("~/.cache/codicology/second")

# The long side the model's own card renders a page to. Larger costs time
# and image tokens and reads no better; smaller loses the small type that is
# the reason for asking.
PAGE_LONG_SIDE = 1540
# A page of text is a few hundred tokens and a dense table two thousand. A
# reading that reaches this has stopped reading and started repeating: the
# handwritten letter it was tried on ran to the limit.
MAX_TOKENS = 4096
START_TIMEOUT = 180
READ_TIMEOUT = 600
# A page that fails is a page; two running is the reader. After that it is
# not asked again in this run, so a server that has stopped does not cost
# ten minutes for every page still to come.
FAILURES_BEFORE_DOWN = 2


class ReaderUnavailable(RuntimeError):
    """The second reader could not be asked: its server did not start, or
    did not answer. That is not a reading and not the absence of one, and
    nothing is to be remembered about the page on the strength of it."""


def model_files() -> "tuple[str, str] | None":
    """The model and its projector, or None where there is no second reader."""
    model, mmproj = os.environ.get(MODEL_ENV), os.environ.get(MMPROJ_ENV)
    if not (model and mmproj):
        ggufs = sorted(glob.glob(os.path.join(DEFAULT_DIR, "*.gguf")))
        proj = [f for f in ggufs if os.path.basename(f).startswith("mmproj")]
        body = [f for f in ggufs if f not in proj]
        if len(proj) != 1 or len(body) != 1:
            return None
        model, mmproj = body[0], proj[0]
    if os.path.isfile(model) and os.path.isfile(mmproj):
        return model, mmproj
    return None


def server_binary() -> "str | None":
    return shutil.which("llama-server") or next(
        (p for p in ("/opt/homebrew/bin/llama-server",
                     "/usr/local/bin/llama-server") if os.path.exists(p)), None)


def available() -> bool:
    return model_files() is not None and server_binary() is not None


class SecondReader:
    """LightOnOCR behind a llama-server of its own, started on first use."""

    name = NAME

    def __init__(self, model: "str | None" = None, mmproj: "str | None" = None,
                 server: "str | None" = None):
        if model is None or mmproj is None:
            found = model_files()
            if found is None:
                raise RuntimeError(
                    "no second reader is installed: put the model's two .gguf "
                    f"files in {DEFAULT_DIR}, or name them in ${MODEL_ENV} "
                    f"and ${MMPROJ_ENV}")
            model, mmproj = found
        self.model, self.mmproj = model, mmproj
        self.server = server or server_binary()
        if self.server is None:
            raise RuntimeError("llama-server is not installed")
        self._proc = None
        self._url = None
        self._failures = 0          # pages running that got no answer
        self.down: "str | None" = None      # why it is no longer asked
        self.pages_read = 0
        self.seconds = 0.0

    # ── the server ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        self._proc = subprocess.Popen(
            [self.server, "-m", self.model, "-mm", self.mmproj,
             "--host", "127.0.0.1", "--port", str(port),
             "-ngl", "99", "-c", "16384", "-np", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        atexit.register(self.close)
        self._url = f"http://127.0.0.1:{port}"
        deadline = time.time() + START_TIMEOUT
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise RuntimeError("the second reader's server did not start "
                                   f"(exit {self._proc.returncode})")
            try:
                with urllib.request.urlopen(self._url + "/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok":
                        return
            except Exception:
                pass
            time.sleep(0.5)             # still loading the model
        self.close()
        raise RuntimeError("the second reader's server did not come up in "
                           f"{START_TIMEOUT} s")

    def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ── a page ──────────────────────────────────────────────────────────────

    def read(self, image) -> "str | None":
        """The page as the second reader writes it (markdown, with TeX for
        mathematics and HTML for tables), or None where it read on to the
        limit without stopping, which it will do again and so is an answer
        of a kind. ReaderUnavailable where it could not be asked at all."""
        if self.down:
            raise ReaderUnavailable(self.down)
        try:
            self.start()
        except RuntimeError as exc:
            self.down = str(exc)        # a server that will not start will
            raise ReaderUnavailable(self.down) from exc     # not start again
        im = image.convert("RGB")
        if max(im.size) > PAGE_LONG_SIDE:
            k = PAGE_LONG_SIDE / max(im.size)
            im = im.resize((round(im.width * k), round(im.height * k)))
        buf = io.BytesIO()
        im.save(buf, "PNG")
        body = {"messages": [{"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64,"
                               + base64.b64encode(buf.getvalue()).decode()}}]}],
                "temperature": 0.0, "max_tokens": MAX_TOKENS}
        req = urllib.request.Request(
            self._url + "/v1/chat/completions", data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        began = time.time()
        try:
            with urllib.request.urlopen(req, timeout=READ_TIMEOUT) as r:
                answer = json.load(r)
            choice = answer["choices"][0]
            text = choice["message"]["content"]
        except Exception as exc:
            self._failures += 1
            if self._failures >= FAILURES_BEFORE_DOWN:
                self.down = (f"it gave no answer for {self._failures} pages "
                             f"running ({type(exc).__name__}: {exc})")
            raise ReaderUnavailable(f"{type(exc).__name__}: {exc}") from exc
        finally:
            self.seconds += time.time() - began
        self._failures = 0
        self.pages_read += 1
        if choice.get("finish_reason") == "length":
            return None                 # ran to the limit: not a reading
        return text


class SecondCache:
    """Second readings, kept beside the first reader's cache and filed
    under the same page names."""

    def __init__(self, path: str, tag: str = NAME):
        self.path, self.tag = path, tag
        self.pages: dict = {}
        self._dirty = False
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                data = json.load(fh)
            if data.get("tag") == tag:
                self.pages = data.get("pages", {})
        except (OSError, ValueError):
            pass

    @staticmethod
    def beside(first_cache_path: str) -> str:
        stem = first_cache_path[:-len(".ocr.gz")] \
            if first_cache_path.endswith(".ocr.gz") else first_cache_path
        return stem + ".second.gz"

    def get(self, key: str):
        """The reading filed under this page: a string, or None where the
        reader was asked and gave none; KeyError where it was never asked."""
        return self.pages[key]

    def put(self, key: str, text: "str | None") -> None:
        self.pages[key] = text
        self._dirty = True

    def save(self) -> None:
        if not self._dirty:
            return
        tmp = self.path + ".tmp"
        with gzip.open(tmp, "wt", encoding="utf-8", compresslevel=6) as fh:
            json.dump({"tag": self.tag, "pages": self.pages}, fh)
        os.replace(tmp, self.path)
        self._dirty = False


# ── what a second reading holds ──────────────────────────────────────────────

_DISPLAY = re.compile(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", re.S)
_TABLE = re.compile(r"<table\b.*?</table>", re.S | re.I)


def display_equations(reading: str) -> "list[str]":
    """The equations set on their own lines, as TeX, in reading order."""
    return [(a or b).strip() for a, b in _DISPLAY.findall(reading or "")
            if (a or b).strip()]


def tables(reading: str) -> "list[str]":
    """The tables, as the HTML the reader wrote them in, in reading order."""
    return _TABLE.findall(reading or "")


def plain_words(reading: str) -> str:
    """The reading with its markup taken off: what it says, for comparing
    word against word with another reading."""
    t = _TABLE.sub(lambda m: " " + re.sub(r"<[^>]+>", " ", m.group(0)) + " ",
                   reading or "")
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"\$\$?|\\[\[\]()]", " ", t)
    t = re.sub(r"\\(?:text|mathrm|mathit|mathbf|textbf|textit)\{([^{}]*)\}",
               r"\1", t)
    t = re.sub(r"^#{1,6}\s*", "", t, flags=re.M)
    t = re.sub(r"[*_`]{1,3}", "", t)
    return re.sub(r"\s+", " ", t).strip()
