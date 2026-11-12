"""shelf_from_calibre.py — a shelf file for `codicology shelf`, written from a
Calibre library and the plugin's cache folder.

    python tools/shelf_from_calibre.py --library "~/Calibre Library" \
        --cache ~/Library/Preferences/calibre/plugins/codicology_ocr/cache/<library id> \
        --out shelf.json

A book is on the shelf if the plugin kept a cache for it (<calibre id>-<hash>.ocr.gz)
and its folder in the library still holds a PDF and an EPUB. Reads both places and
writes one file; nothing else.

The options each book is built with are taken from the book. A copy built since
12 November 2026 says them itself (<meta name="codicology:build">). An older one is
read for its signs, as the rebuild scripts did: a page list means --check-folios, a
bibliography entry with an id means --link-citations, and --link-notes, --typography
and --link-index are taken as given, because every installed copy was built with them.
A book built with the second reader does not show it: name those with --second-for,
and say with --second where their second readings are kept.

The file is meant to be read and corrected by hand afterwards. It is the record of
how the shelf is built, and the harness builds what it says.
"""
import argparse
import glob
import gzip
import html
import json
import os
import re
import sys
import zipfile

ALWAYS = ["--link-index", "--link-notes", "--typography"]
NOT_REPLAYED = {"--cache-only", "--second-reader"}


def options_of(epub: str) -> "tuple[list, str]":
    """(the options the installed copy was built with, how that is known)."""
    with zipfile.ZipFile(epub) as z:
        names = z.namelist()
        package = next((n for n in names if n.endswith(".opf")), None)
        if package:
            said = re.search(r'<meta\b(?=[^>]*\bname="codicology:build")[^>]*\bcontent="([^"]*)"',
                             z.read(package).decode("utf-8", "replace"))
            if said and ";" in said.group(1):
                options = html.unescape(said.group(1)).split(";", 1)[1].split()
                return sorted(o for o in options if o not in NOT_REPLAYED), "the book says"
        content = [n for n in names if re.search(r"(?:page|ch)_\d{4}\.xhtml$", n)]
        nav = next((z.read(n).decode("utf-8", "replace") for n in names if n.endswith("nav.xhtml")), "")
        options = list(ALWAYS)
        text = "".join(z.read(n).decode("utf-8", "replace") for n in content)
        if 'id="bib-' in text:
            options.append("--link-citations")
        if "page-list" in nav or 'id="pgb-' in text:
            options.append("--check-folios")
        return sorted(options), "read from its signs"


def language_of(cache: str) -> str:
    try:
        with gzip.open(cache, "rt", encoding="utf-8") as fh:
            tag = re.search(r'"tag":\s*"([^"]*)"', fh.read(300))
    except (OSError, EOFError):
        tag = None
    if tag and "|" in tag.group(1):
        return tag.group(1).split("|", 1)[1] or "en"
    return "en"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--library", required=True, help="the Calibre library's folder")
    ap.add_argument("--cache", required=True, help="the plugin's cache folder for that library")
    ap.add_argument("--second", metavar="DIR", help="where second (and third) readings are kept: <cache stem>.second.gz")
    ap.add_argument("--second-for", nargs="*", default=[], metavar="ID",
                    help="the books that are built with --second-reader")
    ap.add_argument("--out", required=True, help="the shelf file to write")
    args = ap.parse_args(argv)
    library, cache_dir = os.path.expanduser(args.library), os.path.expanduser(args.cache)

    by_id: dict = {}
    for path in sorted(glob.glob(os.path.join(glob.escape(cache_dir), "*.ocr.gz"))):
        bid = os.path.basename(path).split("-", 1)[0]
        by_id.setdefault(bid, []).append(path)
    books, left_out = [], []
    for bid in sorted(by_id, key=lambda s: (len(s), s)):
        caches = sorted(by_id[bid], key=os.path.getmtime)
        cache = caches[-1]
        folders = glob.glob(os.path.join(glob.escape(library), "*", f"*({bid})"))
        pdfs = sorted(glob.glob(os.path.join(glob.escape(folders[0]), "*.pdf"))) if folders else []
        epubs = sorted(glob.glob(os.path.join(glob.escape(folders[0]), "*.epub"))) if folders else []
        if not (pdfs and epubs):
            left_out.append(f"{bid}: " + ("no folder in the library" if not folders
                                          else "no PDF" if not pdfs else "no EPUB"))
            continue
        try:
            with open(os.path.join(folders[0], "metadata.opf"), encoding="utf-8") as fh:
                title = html.unescape(re.search(r"<dc:title>([^<]*)", fh.read()).group(1))
        except (OSError, AttributeError):
            title = os.path.splitext(os.path.basename(pdfs[0]))[0]
        options, how = options_of(epubs[0])
        stem = cache[:-len(".ocr.gz")]
        beside = []
        if bid in args.second_for:
            if not args.second:
                ap.error("--second-for needs --second: where the second readings are kept")
            beside = sorted(p for p in glob.glob(os.path.join(
                glob.escape(os.path.expanduser(args.second)), os.path.basename(stem) + ".*.gz"))
                if not p.endswith(".ocr.gz"))
            if not any(p.endswith(".second.gz") for p in beside):
                left_out.append(f"{bid}: named in --second-for, and {args.second} has no second reading of it")
                continue
            options = sorted(options + ["--second-reader"])
        decisions = stem + "-disputes.decisions.json"
        book = {"id": bid, "title": title, "pdf": pdfs[0], "cache": cache, "lang": language_of(cache),
                "options": options, "options_known": how,
                "decisions": decisions if os.path.isfile(decisions) else None,
                "installed": epubs[0]}
        if beside:
            book["beside"] = beside
        notes = []
        if len(caches) > 1:
            notes.append(f"{len(caches)} caches; the newest is used")
        if len(pdfs) > 1:
            notes.append(f"{len(pdfs)} PDFs; the first is used")
        if notes:
            book["note"] = "; ".join(notes)
        books.append(book)
    with open(os.path.expanduser(args.out), "w", encoding="utf-8") as fh:
        json.dump({"library": library, "cache": cache_dir, "books": books}, fh, ensure_ascii=False, indent=1)
    print(f"{len(books)} book(s) written to {args.out}")
    for line in left_out:
        print(f"  left out — {line}")
    for b in books:
        if b.get("note"):
            print(f"  {b['id']}: {b['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
