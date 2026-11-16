"""The pages to look at before a book ships: a page read twice (a turn
caught mid-way, a duplicate of its neighbour), a page that looks like a
hand or holds too little, and the review sheet that shows each beside
what was read from it.
"""
from __future__ import annotations

import base64
import cv2
import html
import io
import re
import statistics
from PIL import Image
from typing import NamedTuple
from .capture import _skin_mask


# Duplicate detection, on OCR'd text. Five-word runs are long enough that two
# different pages of prose share almost none, and Jaccard stays high across the
# word or two an OCR pass gets wrong between two shots of the same page.
SHINGLE_WORDS = 5


# Checked against the printed page numbers across a whole book: every drop was a
# true repeat, and the repeats scored 0.80 to 0.996 — far enough above this that
# the margin is real rather than lucky. Beware of tuning this against page
# numbers alone: OCR misreads a running head often enough that a kept page can
# look like it is missing when only its header was garbled.
DUPE_SIMILARITY = 0.55


DUPE_WINDOW = 10


MIN_SHINGLES = 12


# How far back to look for a repeat. A spread re-shot three or four times puts
# its copies further apart than the two pages a single re-take would, so a window
# of 6 let a genuine duplicate through at a distance of 8.
# Scores between this and DUPE_SIMILARITY are close enough to be worth a look.
# Measured over a book, almost nothing lands here - which is the point: a
# borderline verdict is rare, so reviewing them costs a glance.
DUPE_BORDERLINE = 0.35


REVIEW_THUMB_WIDTH = 150


# ── Duplicate pages ───────────────────────────────────────────────────────────

def _shingles(text: str, k: int = SHINGLE_WORDS) -> set[str]:
    words = re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split()
    if len(words) < k:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


class PageVerdict(NamedTuple):
    """What the duplicate pass concluded about one page, and on what evidence."""
    index: int
    status: str          # kept | duplicate | unjudged | borderline
    match: int | None    # the page it was measured against
    score: float


def review_pages(texts: list[str], window: int = DUPE_WINDOW) -> list[PageVerdict]:
    """
    Judge every page, recording the reasoning rather than only the conclusion.

    Two of these verdicts deserve a second look and the numbers say which. Across
    a whole book the scores pile up at the ends — near zero for a page unlike
    anything before it, above 0.8 for a page shot twice — and the middle is all
    but empty, so a *borderline* score is rare and genuinely close. Far more
    common is *unjudged*: a page holding too few words to compare at all, which
    is what a hand across the paper leaves behind. Those are invisible to this
    pass and obvious to a reader, so they are worth surfacing, not silently
    keeping.
    """
    fingerprints = [_shingles(t) for t in texts]
    verdicts: dict[int, PageVerdict] = {}
    duplicates: set[int] = set()

    for i in range(len(texts)):
        if len(fingerprints[i]) < MIN_SHINGLES:
            verdicts[i] = PageVerdict(i, "unjudged", None, 0.0)
            continue

        best_score, best_j = 0.0, None
        for j in range(max(0, i - window), i):
            if j in duplicates or len(fingerprints[j]) < MIN_SHINGLES:
                continue
            smaller = min(len(fingerprints[i]), len(fingerprints[j]))
            if not smaller:
                continue
            score = len(fingerprints[i] & fingerprints[j]) / smaller
            if score > best_score:
                best_score, best_j = score, j

        if best_j is not None and best_score >= DUPE_SIMILARITY:
            loser = i if len(texts[i]) <= len(texts[best_j]) else best_j
            keeper = best_j if loser == i else i
            duplicates.add(loser)
            verdicts[loser] = PageVerdict(loser, "duplicate", keeper, best_score)
            if loser != i:
                verdicts[i] = PageVerdict(i, "kept", best_j, best_score)
        elif best_j is not None and best_score >= DUPE_BORDERLINE:
            verdicts[i] = PageVerdict(i, "borderline", best_j, best_score)
        else:
            verdicts[i] = PageVerdict(i, "kept", best_j, best_score)

    return [verdicts[i] for i in range(len(texts))]


def _thumb_data_uri(path: str, width: int = REVIEW_THUMB_WIDTH) -> str:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((width, width * 2))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=70)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class PageFlag(NamedTuple):
    """A page worth a human's attention, and what specifically is wrong."""
    index: int
    kind: str            # hand | sparse | uncropped | blank
    detail: str
    action: str


# A hand over the page reads this much skin across the text body. Measured on
# this footage: a clear page sits under 0.03 and a page with fingers across the
# column reaches 0.10-0.23. The threshold sits above the clear pages and below
# anything that actually obscures words.
FLAG_HAND = 0.06


# A page holding this fraction of its neighbours' word count has almost
# certainly lost text to an occlusion or a bad crop, rather than simply being a
# short page — chapter ends run short but not this short.
FLAG_SPARSE = 0.35


# How far a page's shape may sit from the book's own median before it reads as
# a frame that was never cropped to the page at all.
FLAG_ASPECT = 0.35


# How much shorter than its neighbours a page must read before it is even
# considered a fragment rather than a page.
TURN_SHORTER = 0.5


# and how much of the little it does say must already appear on a page nearby.
# High deliberately: this is the test that makes discarding safe, because a
# fragment whose every phrase is already in the book costs nothing to lose.
TURN_CONTAINED = 0.7


# How far away to look for the page it repeats. A leaf caught in flight shows
# the page immediately beside it and no other, so this stays tight. Widening it
# starts eating real pages: at four, the title page was condemned because a
# preface several leaves later happens to name both authors, and matching on
# characters means short common words turn up in any long text.
TURN_WINDOW = 2


def find_page_turns(page_paths: list[str], texts: list[str] | None = None
                    ) -> list[tuple[int, str]]:
    """
    Captures that caught a leaf in flight rather than a page at rest.

    Identified by what they say, not by how they look. Three image measures were
    tried first and all three failed on a known fragment from this book — a page
    eighty percent hidden behind the next leaf, whose few visible characters were
    cut off mid-word. It reads as perfectly ordinary: the covering leaf's edge and
    the desk behind it put its dark-pixel count *above* the book's median, its
    proportions are those of any page, and that same leaf edge spreads its
    apparent text across the full width. Meanwhile the genuinely low-ink pages
    were the title page and the chapter openings — the ones it would be worst to
    lose. Geometry cannot tell a covered page from an uncovered one.

    What is decisive is that a fragment repeats a page that is already in the
    book. So a capture is called a turn only when it says much less than the
    pages around it *and* nearly everything it does say is already on a page
    close by. That pairing is what makes discarding it safe: not a judgement that
    it looks bad, but a demonstration that nothing is lost with it.
    """
    if texts is None or not page_paths:
        return []

    # Matched on characters rather than on words, because a covered page's text
    # is cut off mid-word: "Chicag", "orgies a", "disrepai". Those are prefixes
    # of what the full page says, so as words they match nothing at all — the
    # first attempt at this scored a fragment's overlap with the very page it
    # was a fragment of at zero. As character runs they match exactly.
    def squash(t: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", t.lower())

    def stubs(t: str) -> list[str]:
        return [s for s in (re.sub(r"[^a-z0-9]+", "", w.lower()) for w in t.split())
                if len(s) >= 4]

    words = [len(t.split()) for t in texts]
    squashed = [squash(t) for t in texts]
    turns: list[tuple[int, str]] = []

    for i in range(len(page_paths)):
        near = [words[j] for j in range(max(0, i - 2), min(len(words), i + 3)) if j != i]
        typical = statistics.median(near) if near else 0
        # Too short to be a page here — but a page has to be substantial for
        # "shorter than its neighbours" to mean anything at all.
        if typical < 60 or words[i] >= typical * TURN_SHORTER:
            continue
        mine = stubs(texts[i])
        if len(mine) < 5:
            continue
        best, best_j = 0.0, None
        for j in range(max(0, i - TURN_WINDOW), min(len(page_paths), i + TURN_WINDOW + 1)):
            if j == i or len(squashed[j]) < 200:
                continue
            share = sum(1 for s in mine if s in squashed[j]) / len(mine)
            if share > best:
                best, best_j = share, j
        if best >= TURN_CONTAINED and best_j is not None:
            turns.append((i, f"{words[i]} words against about {typical:.0f} nearby, and "
                             f"{best * 100:.0f}% of them already on page {best_j}"))
    return turns


def flag_pages(page_paths: list[str], texts: list[str],
               page_ids: list[str] | None = None) -> list[PageFlag]:
    """
    Find pages a person should look at, from measurements that can be trusted.

    Deliberately narrow. The obvious extra signal — reading the printed folio
    from the running head and checking it follows the last one — is not here,
    because at this capture resolution that type is around eleven pixels tall
    and every reader tried on it, human and machine, misreads it often enough
    that the flag would cry wolf. What is measurable is measurable: skin across
    the text, a page far shorter than its neighbours, a frame that was never
    cropped, a page with no ink on it.
    """
    flags: list[PageFlag] = []
    words = [len(t.split()) for t in texts]
    aspects: list[float] = []
    for p in page_paths:
        try:
            with Image.open(p) as im:
                aspects.append(im.width / max(1, im.height))
        except OSError:
            aspects.append(0.0)
    median_aspect = statistics.median([a for a in aspects if a > 0] or [0.75])

    def name(i: int) -> str:
        return page_ids[i] if page_ids and i < len(page_ids) else str(i)

    # The skin mask is chroma bounds, and the warm paper of an older book sits
    # inside them wholesale — a 1938 printing measured 99.9% "skin" on every
    # page, which flagged the entire book as handled. What means something is
    # not the absolute reading but a page standing out from this book's own
    # paper: a hand on white stock is far above a near-zero baseline, and on tan
    # stock nothing can stand out at all — reporting nothing there is honest,
    # because the measurement genuinely cannot see a hand against that paper.
    skins: list[float] = []
    for path in page_paths:
        img = cv2.imread(path)
        if img is None:
            skins.append(0.0)
            continue
        h, w = img.shape[:2]
        body = img[int(h * 0.10):int(h * 0.92), int(w * 0.06):int(w * 0.94)]
        skins.append(float((_skin_mask(body) > 0).mean()) if body.size else 0.0)
    hand_bar = max(FLAG_HAND, statistics.median(skins) + 0.05) if skins else FLAG_HAND

    for i, path in enumerate(page_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        skin = skins[i]
        if skin > hand_bar:
            flags.append(PageFlag(
                i, "hand", f"{skin * 100:.0f}% of the page body reads as skin",
                f"re-shoot this page: --patch {name(i)}=<photo>"))

        near = [words[j] for j in range(max(0, i - 2), min(len(words), i + 3)) if j != i]
        typical = statistics.median(near) if near else 0
        if typical >= 80 and words[i] < typical * FLAG_SPARSE:
            flags.append(PageFlag(
                i, "sparse", f"{words[i]} words against about {typical:.0f} nearby",
                f"check for a hand or a bad crop: --patch {name(i)}=<photo>"))

        if words[i] == 0:
            flags.append(PageFlag(
                i, "blank", "no text recognised",
                f"drop it if it is a blank leaf: --drop-pages {name(i)}"))

        if aspects[i] and abs(aspects[i] - median_aspect) / median_aspect > FLAG_ASPECT:
            flags.append(PageFlag(
                i, "uncropped", f"shape {aspects[i]:.2f} against a book median of "
                                f"{median_aspect:.2f} — the page outline was probably missed",
                f"re-shoot or drop: --patch {name(i)}=<photo>"))

    for i, why in find_page_turns(page_paths, texts):
        flags.append(PageFlag(
            i, "turn", f"looks like a page caught mid-turn — {why}",
            f"drop it: --drop-pages {name(i)}   (or --drop-turns for all of these)"))
    return flags


def write_review_sheet(
    output_html: str,
    page_paths: list[str],
    verdicts: list[PageVerdict],
    title: str,
    flags: list[PageFlag] | None = None,
    page_ids: list[str] | None = None,
) -> None:
    """
    Write a self-contained page showing every call the duplicate pass was unsure
    of, so it can be overruled.

    Only the doubtful pages go in. A sheet of all several hundred is a sheet
    nobody reads, and the confident verdicts are confident for a reason: the
    scores behind them sit at the extremes.

    Nothing here edits the book. The sheet's output is a list of page numbers to
    feed back through --drop-pages, which re-runs from the PDF in about a second
    rather than putting OCR through the whole book again.
    """
    dupes = [v for v in verdicts if v.status == "duplicate"]
    unjudged = [v for v in verdicts if v.status == "unjudged"]
    borderline = [v for v in verdicts if v.status == "borderline"]
    flags = flags or []
    if not (dupes or unjudged or borderline or flags):
        print("  nothing doubtful to review")
        return

    # A page that beats one earlier duplicate can still lose to a later one:
    # four shots of the same page chain as 60 loses to 62, then 62 itself
    # loses to 64. Shown pairwise, a card reading "62 kept" is only true
    # against 60 — three cards later, 62 has its own "dropped" card, and
    # nothing on the first card said so. Following each chain to wherever it
    # actually ends is what the reader needs on that first card, not the
    # immediate match, which is often just a stepping stone.
    by_index = {v.index: v for v in verdicts}

    def final_survivor(idx: int) -> int:
        seen = set()
        while idx not in seen:
            seen.add(idx)
            v = by_index.get(idx)
            if v is None or v.status != "duplicate" or v.match is None:
                return idx
            idx = v.match
        return idx  # a cycle would mean a scoring bug elsewhere; break rather than loop

    def card(idx: int, caption: str, checked: bool) -> str:
        uri = _thumb_data_uri(page_paths[idx])
        mark = " checked" if checked else ""
        return (f'<label class="c"><input type="checkbox" data-i="{idx}"{mark}>'
                f'<img src="{uri}" alt="page {idx}"><span>{caption}</span></label>')

    parts = [f"<h1>{html.escape(title)} — review</h1>",
             "<p class='lede'>Tick a page to leave it out of the book. Pages already "
             "dropped as duplicates start ticked, so the list below begins as what the "
             "duplicate pass decided; untick one to keep it, tick anything else to drop "
             "it. Run the whole command as given — it turns the duplicate pass off, so "
             "that what you tick here is what happens, rather than being decided again "
             "and overruling you.</p>"]

    if flags:
        # These are not duplicate calls and no checkbox fixes them — a page with
        # a thumb across the text needs a new photograph, which is a thing only
        # the person holding the book can do. Grouped by page so one trip to the
        # shelf settles everything about it.
        by_page: dict[int, list[PageFlag]] = {}
        for f in flags:
            by_page.setdefault(f.index, []).append(f)
        parts.append(
            f"<h2>Worth your eyes ({len(by_page)})</h2>"
            "<p>Pages the measurements say are damaged rather than repeated. Ticking "
            "does nothing useful here: most want a fresh photograph, which you can feed "
            "back with <code>--patch</code> without re-running the video. The suggested "
            "command is under each page. Note that a page can be perfectly sharp and "
            "still be wrong in a way no measurement catches — a misprinted folio, a leaf "
            "photographed out of turn — so this is a floor, not a guarantee.</p><div class='g'>")
        for idx in sorted(by_page):
            fs = by_page[idx]
            label = (page_ids[idx] if page_ids and idx < len(page_ids) else f"#{idx}")
            why = "; ".join(f.detail for f in fs)
            act = fs[0].action
            uri = _thumb_data_uri(page_paths[idx])
            parts.append(
                f'<div class="c"><img src="{uri}" alt="page {idx}">'
                f'<span><b>{html.escape(label)}</b> — {html.escape(why)}<br>'
                f'<code>{html.escape(act)}</code></span></div>')
        parts.append("</div>")

    if dupes:
        parts.append(f"<h2>Dropped as duplicates ({len(dupes)})</h2>"
                     "<p>Each pair is the page that was dropped beside the one kept "
                     "in its place — the page actually still in the book, even where "
                     "that took outlasting more than one other repeat.</p><div class='g'>")
        for v in dupes:
            survivor = final_survivor(v.match)
            label = (f"#{survivor} kept" if survivor == v.match
                     else f"#{survivor} kept · also outlasted #{v.match}")
            parts.append("<div class='pair'>"
                         + card(v.index, f"#{v.index} dropped · {v.score:.2f}", True)
                         + card(survivor, label, False)
                         + "</div>")
        parts.append("</div>")

    if unjudged:
        parts.append(f"<h2>Too little text to judge ({len(unjudged)})</h2>"
                     "<p>Blank pages, and pages a hand covered enough that OCR read "
                     "almost nothing. All were kept. Repeats sit next to each other.</p>"
                     "<div class='g'>")
        parts += [card(v.index, f"#{v.index}", False) for v in unjudged]
        parts.append("</div>")

    if borderline:
        parts.append(f"<h2>Borderline ({len(borderline)})</h2><div class='g'>")
        for v in borderline:
            parts.append("<div class='pair'>"
                         + card(v.index, f"#{v.index} · {v.score:.2f}", False)
                         + card(v.match, f"#{v.match}", False) + "</div>")
        parts.append("</div>")

    body = "\n".join(parts)
    doc = f"""<!doctype html><meta charset="utf-8">
<title>{html.escape(title)} — review</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:1.5rem 2rem 8rem;color:#111;background:#fafafa}}
 h1{{margin:0 0 .25rem}} h2{{margin:2rem 0 .25rem;font-size:1.05rem}}
 .lede,h2+p{{color:#555;margin:.15rem 0 .9rem;font-size:.9rem}}
 .g{{display:flex;flex-wrap:wrap;gap:.75rem}}
 .pair{{display:flex;gap:.35rem;padding:.4rem;background:#fff;border:1px solid #e3e3e3;border-radius:8px}}
 .c{{display:block;width:{REVIEW_THUMB_WIDTH}px;cursor:pointer;position:relative}}
 .c img{{width:100%;display:block;border-radius:5px;border:1px solid #ddd;background:#fff}}
 .c span{{display:block;font-size:.75rem;color:#666;padding-top:.2rem}}
 .c input{{position:absolute;top:.3rem;left:.3rem;width:1.1rem;height:1.1rem;z-index:1}}
 .c input:checked ~ img{{outline:3px solid #c0392b;opacity:.45}}
 footer{{position:fixed;left:0;right:0;bottom:0;background:#fff;border-top:1px solid #ddd;
   padding:.7rem 2rem;display:flex;gap:.75rem;align-items:center}}
 code{{background:#f2f2f2;padding:.35rem .5rem;border-radius:5px;flex:1;overflow:auto;white-space:nowrap}}
 button{{font:inherit;padding:.4rem .8rem;border-radius:6px;border:1px solid #bbb;background:#f6f6f6;cursor:pointer}}
 @media (prefers-color-scheme:dark){{
   body{{background:#151515;color:#eee}} .pair{{background:#1e1e1e;border-color:#333}}
   .c img{{border-color:#333;background:#222}} .lede,h2+p,.c span{{color:#aaa}}
   footer{{background:#1e1e1e;border-color:#333}} code{{background:#262626}}
   button{{background:#2a2a2a;border-color:#444;color:#eee}}}}
</style>
{body}
<footer><code id="out">--no-dedupe --drop-pages </code>
<button onclick="navigator.clipboard.writeText(document.getElementById('out').textContent.trim())">Copy</button></footer>
<script>
 const out=document.getElementById('out');
 function sync(){{
   const v=[...document.querySelectorAll('input[type=checkbox]')]
     .filter(c=>c.checked).map(c=>c.dataset.i).sort((a,b)=>a-b);
   out.textContent='--no-dedupe --drop-pages '+(v.join(',')||'(none)');
 }}
 document.addEventListener('change',sync); sync();
</script>"""
    with open(output_html, "w", encoding="utf-8") as fh:
        fh.write(doc)
    print(f"  review sheet: {output_html}  "
          f"({len(dupes)} dropped, {len(unjudged)} unjudged, {len(borderline)} borderline)")
