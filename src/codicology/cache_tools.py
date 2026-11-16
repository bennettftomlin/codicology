"""Repairs to a book's OCR cache made without reading it again: Surya's layout
labels added to readings cached before labels were kept, and readings filed
under their page images' old names refiled under the new.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import time
from PIL import Image
from .backends import llamacpp_safe_schemas
from .cache import CACHE_VERSION, OCRCache
from .pdf_frame import _box_iou
from .pdf_pages import TURNS_SUFFIX, load_pages_from_pdf


def _match_layout_labels(items: list, blocks: list, width: int,
                         height: int) -> int:
    """Assign layout labels to cached items by greedy best-IoU, one block
    per item, nothing under 0.5. An item that matches no block cleanly
    stays unlabeled — a wrong label misleads every consumer downstream,
    an absent one merely declines to help."""
    cands = []
    for bi, (lab, bb) in enumerate(blocks):
        nb = (bb[0] / width, bb[1] / height, bb[2] / width, bb[3] / height)
        for ii, it in enumerate(items):
            if not it.get("box") or it.get("lab"):
                continue
            iou = _box_iou(nb, tuple(it["box"]))
            if iou >= 0.5:
                cands.append((iou, ii, bi, lab))
    cands.sort(key=lambda c: -c[0])
    used_i, used_b, n = set(), set(), 0
    for iou, ii, bi, lab in cands:
        if ii in used_i or bi in used_b:
            continue
        items[ii]["lab"] = lab
        used_i.add(ii)
        used_b.add(bi)
        n += 1
    return n


def relabel_cache(cache_path: str, pdf_path: str, backend_name: str = "surya",
                  langs: "list[str] | None" = None) -> dict:
    """Backfill layout labels into an existing cache without re-reading a
    word.

    Recognition is the slow half of a build and the settled half: the words
    in the cache do not improve by being read again by the same pinned
    model. Labels are the cheap half — one layout call per page instead of
    layout plus recognition; the layout head rides the same served
    foundation model, so the inference server must be free, and it is
    auto-started the way a build starts it. Pages are
    regenerated through the same extraction path the original build used,
    so their cache keys reproduce exactly; a page
    whose key misses is reported and skipped, never guessed at. The text is
    left byte-identical, which keeps every dispute record and decisions
    file valid."""
    from collections import Counter as _Counter
    langs = langs or ["en"]
    tmp = tempfile.mkdtemp(prefix="relabel_")
    try:
        page_paths = load_pages_from_pdf(
            pdf_path, tmp, former_keys=OCRCache.filed_by_bytes(cache_path),
            turns_path=cache_path + TURNS_SUFFIX)
        cache = OCRCache(cache_path, backend_name, langs)
        if not cache.entries:
            print(f"  [!] cache is cold or tagged differently: {cache_path}")
            return {"pages": 0}
        # Guided layout, surya's default and what a build's fallback uses,
        # with the schema repaired so llama.cpp can compile it. This ran
        # unguided from August to dodge the grammar error. It cost nothing
        # that can be seen: on Eagle Forgotten's last 22 pages the two
        # return identical boxes — including none at all on the 17 dense
        # back-matter pages, which neither mode can box.
        llamacpp_safe_schemas()
        from surya.layout import LayoutPredictor
        lp = LayoutPredictor()
        st = _Counter()
        # The served model cold-starts, and a call against a waking server
        # returns nothing — indistinguishable from a blank page. Warm up
        # against a mid-book page (guaranteed ink) with patience, so the
        # dry-streak guard below judges a live server only.
        probe_path = page_paths[len(page_paths) // 2]
        for _ in range(12):
            if lp([Image.open(probe_path)])[0].bboxes:
                break
            time.sleep(8)
        else:
            print("  [!] layout server never came up after 96s; aborting "
                  "with the cache untouched")
            return dict(st)
        dry_streak = 0
        for pp in page_paths:
            entry = cache._entry(pp)
            if entry is None:
                st["page_miss"] += 1
                continue
            raw = entry["items"] if isinstance(entry, dict) else entry
            boxed = [it for it in raw if it.get("box")]
            if boxed and all(it.get("lab") for it in boxed):
                st["already_labeled"] += 1
                continue
            img = Image.open(pp)
            result = lp([img])[0]
            blocks = [(b.label, list(b.bbox)) for b in result.bboxes]
            # a page with cached blocks but zero layout blocks is the shape
            # of a dead inference server retrying forever, not of a book;
            # three in a row and this run stops rather than spins
            if boxed and not blocks:
                dry_streak += 1
                if dry_streak >= 3:
                    print("  [!] layout returned nothing three pages "
                          "running — inference server down? aborting; "
                          "cache untouched beyond what was already saved")
                    break
            else:
                dry_streak = 0
            st["items_labeled"] += _match_layout_labels(
                raw, blocks, img.width, img.height)
            st["items_unmatched"] += sum(1 for it in raw
                                         if it.get("box") and not it.get("lab"))
            st["pages"] += 1
            # stamped current: these pages have been through the layout
            # model, so a later build must not read them again
            if isinstance(entry, dict):
                entry["v"] = CACHE_VERSION
            cache.dirty = True
        if cache.dirty:
            backup = cache_path + ".prelabel"
            if not os.path.exists(backup):
                shutil.copy2(cache_path, backup)
            cache.save()
        print(f"  relabel: {st['items_labeled']} item(s) labeled across "
              f"{st['pages']} page(s), {st['items_unmatched']} unmatched "
              f"(left unlabeled), {st['page_miss']} page key(s) not in cache, "
              f"{st['already_labeled']} already labeled")
        return dict(st)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def rekey_cache(cache_path: str, pdf_path: str, backend_name: str = "surya",
                langs: "list[str] | None" = None) -> dict:
    """Refile every reading in a cache under the name its page has now.

    A build does this for itself, page by page, the first time it meets a
    cache written before pages were named for their pixels (see page_key).
    This does a whole book at once without building it, which is the way to
    bring a shelf of caches over while the encoder that wrote their old
    names is still the one installed: once it changes, a reading that was
    never refiled can only be read again.

    Nothing is read and no reading is altered. The entries after are
    checked to be the entries before, one for one, and the file is left
    untouched if they are not.
    """
    from collections import Counter as _Counter

    def fingerprints(entries: dict) -> "_Counter":
        return _Counter(hashlib.sha256(
            json.dumps(e, sort_keys=True).encode("utf-8")).hexdigest()
            for e in entries.values())

    cache = OCRCache(cache_path, backend_name, langs or ["en"])
    st = {"entries": len(cache.entries), "pages": 0, "found": 0,
          "refiled": 0, "missing": 0, "left": len(cache.by_bytes)}
    if not cache.entries:
        print(f"  [!] cache is cold or tagged differently: {cache_path}")
        return st
    if not cache.by_bytes:
        print(f"  rekey: all {st['entries']} reading(s) in {cache_path} are "
              f"already filed under their pages' present names")
        return st
    before = fingerprints(cache.entries)
    tmp = tempfile.mkdtemp(prefix="rekey_")
    try:
        page_paths = load_pages_from_pdf(pdf_path, tmp, former_keys=True,
                                         turns_path=cache_path + TURNS_SUFFIX)
        st["pages"] = len(page_paths)
        for pp in page_paths:
            if cache._entry(pp) is None:
                st["missing"] += 1
            else:
                st["found"] += 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    st["refiled"], st["left"] = cache.refiled, len(cache.by_bytes)
    if fingerprints(cache.entries) != before:
        print(f"  [!] rekey: the readings after are not the readings before; "
              f"{cache_path} is left as it was")
        st["refiled"] = 0
        return st
    cache.save()
    print(f"  rekey: {st['found']} of {st['pages']} page(s) have a reading; "
          f"{st['refiled']} refiled under the page's pixels"
          + (f"; {st['missing']} page(s) of this PDF are not in the cache"
             if st["missing"] else "")
          + (f"; {st['left']} reading(s) match no page of this PDF and stay "
             f"under their old names" if st["left"] else ""))
    return st
