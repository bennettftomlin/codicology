"""codicology review — the dispute record, rendered for a human's eyes.

The adjudicator reports and never rewrites; this sheet is where rewriting
becomes a human act. Each dispute row shows the ink itself (a crop located
through tesseract's word geometry), what shipped, what the ladder proposed,
and a free-text field where the person who can actually read the page
overrules everyone. Engines that re-read ink are recorded, never crowned;
the human's reading — even one no engine produced — is crowned on sight.

Rows are ranked by how likely the shipped word is to be wrong:

  1. the ladder's catches — verdicts against the shipped reading whose
     surya/tesseract pair occurs once in the book. Repeats are convention,
     singletons are damage.
  2. abstains where the shipped word is not a word — nobody was confident,
     but the shipped text is probably broken.
  3. open abstains — two plausible readings; only a human settles these.
  4. systematic repeats, collapsed — the same pair dozens of times is the
     scholarly apparatus (ff., op. cit.) or a font quirk, not damage; one
     representative stands for the group.
  5. confirmed rows, collapsed — the ladder agreed with what shipped. Still
     editable: confirmation is a prior, the reader's eye is the authority.

The exported decisions file records {page, occurrence, old, new, source,
rung, before, after} per correction — source "human" for typed readings,
"ladder" for accepted verdicts. A correction is addressed to one word of
its page: `occurrence` is the word's ordinal among ALL the page's
instances of the same token, and `before`/`after` are the words either
side. The apply step finds the site by those neighbours and falls back on
the ordinal only to choose between places that read alike.

Sheets made before 2026-11-10 numbered `occurrence` among a page's
DISPUTES and recorded no neighbours, while the apply step counted every
instance: where an undisputed instance stood ahead of the disputed one,
the correction landed on the wrong word. A row without neighbours is
therefore applied only where the two counts must agree — every instance
of the word on the page accounted for by the file — and reported stale
otherwise.

Review happens in rounds: a book is reviewed, rebuilt with the corrections,
adjudicated again, and reviewed again against the corrected text. Each
round's file is addressed to the text the round before it left, so the
rounds are kept apart and applied in order. Beside `book.decisions.json`,
the round in hand, sit `book.decisions.r1.json`, `book.decisions.r2.json`
and so on, the earlier ones, oldest first (decision_rounds). A file from
the same sheet exported again takes the place of the round in hand; a
file from another sheet becomes a new round (file_decisions). The file
carries `sheet`, the fingerprint of the record its sheet was made from,
and that is how the two are told apart.

`retired` holds rows that must not be applied: a correction accepted in
error, with the sheet that could un-stage it gone. A retired row is never
applied, is carried from round to round, and is dropped from any later
file that offers it again.
"""
import base64
import collections
import hashlib
import html
import io
import json
import os
import re
import subprocess
import tempfile
import unicodedata

from .adjudicate import tokens_as
from .adjudicate import (_TYPO, STRUCK, WORD, _is_word, body_span,
                         epub_page_texts, fold_text, fold_word, hard_view,
                         page_words, tokens)

# battery v2, 2,591 truth-scored disputes — shown on the rung chips so the
# reviewer knows how much each verdict is worth
# battery v3, 2026-08-22: 2,545 truth-scored disputes over six born-digital
# books, clean + degraded renders, post apparatus/roman guards + seam gates
RUNG_ACCURACY = {"lexicon": "97.8%", "vision": "94.8%", "dictionary": "92.3%"}
# "readers" is Vision and the second reader agreeing, "second" the second
# reader alone with a view. Neither has a figure yet: they were right on
# all 31 disputes they settled between them, which is too few to print as
# a percentage beside ones measured on thousands.
RUNG_ORDER = {"lexicon": 0, "readers": 0, "vision": 1, "second": 1,
              "dictionary": 2}

SURYA_ONLY_TITLE = ("Words no other witness saw — surya read them, "
                    "tesseract found no trace")

TIER_TITLES = {
    "catches": "The ladder's catches — it believes what shipped is wrong",
    "broken": "Abstains where the shipped word is not a word",
    "open": "Open abstains — two plausible readings",
    "systematic": "Systematic repeats — convention, not damage",
    "confirmed": "Confirmed — the ladder agrees with what shipped",
}


def pair_counts(disputes) -> collections.Counter:
    return collections.Counter((d["surya"], d["tesseract"]) for d in disputes)


def assign_occurrences(disputes) -> list:
    """Each dispute's ordinal, as its row will export it.

    The adjudicator addresses every dispute itself (adjudicate.site_address:
    the ordinal among ALL the page's instances of the token, with its
    neighbours), and that address is passed through untouched. A report
    written before it did so has only this to go on: the ordinal among
    the page's DISPUTES sharing a shipped token. That is not the page
    ordinal whenever an undisputed instance stands earlier on the page,
    which is why such rows carry no neighbours and the appliers treat
    them as unaddressed."""
    seen = collections.Counter()
    out = []
    for d in disputes:
        # keyed by the LITERAL shipped token — a folded key merged 'The'
        # and 'the' into one stream and mis-addressed both (I1)
        key = (d["page"], d.get("shipped") or d["surya"])
        own = d.get("occurrence")
        out.append(own if isinstance(own, int) and "before" in d
                   else seen[key])
        seen[key] += 1
    return out


def tier_of(d, counts) -> str:
    shipped = d.get("shipped") or d["surya"]
    winner = d.get("winner")
    if winner is None:
        # identical abstains repeated across a book are convention too —
        # the apparatus guard turns ff/f into abstains by the dozen, and
        # they must not flood the tier reserved for genuine coin-flips.
        # Damage does not repeat identically; convention does.
        if counts[(d["surya"], d["tesseract"])] >= 4:
            return "systematic"
        folded = fold_word(shipped)
        return "broken" if folded and not _is_word(folded) else "open"
    if winner == shipped:
        return "confirmed"
    if counts[(d["surya"], d["tesseract"])] >= 2:
        return "systematic"
    return "catches"


def build_tiers(disputes) -> dict:
    counts = pair_counts(disputes)
    occs = assign_occurrences(disputes)
    tiers = {k: [] for k in TIER_TITLES}
    for idx, d in enumerate(disputes):
        row = dict(d, idx=idx, occurrence=occs[idx],
                   pair_count=counts[(d["surya"], d["tesseract"])])
        tiers[tier_of(d, counts)].append(row)
    tiers["catches"].sort(
        key=lambda r: (RUNG_ORDER.get(r.get("rung"), 9), r["page"]))
    return tiers


def _fingerprint(report) -> str:
    """Keys the browser-side staging state. It must change whenever the
    ROWS change: a count-and-paths key once let round one's staged choices
    reattach themselves by row index to a regenerated round-two sheet of
    the same book. Reopening the same sheet still resumes — the content
    is identical, so the key is."""
    rows = [(d.get("page"), d.get("surya"), d.get("tesseract"))
            for d in report.get("disputes", [])]
    runs = [(p.get("page"), r.get("text"))
            for p in report.get("surya_only", []) for r in p.get("runs", [])]
    raw = json.dumps([rows, runs, report.get("pdf", ""),
                      report.get("epub", "")], sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _outline_token(crop, rect):
    """Mark the disputed token's own ink inside its crop. The crop carries
    context — a neighboring line's tail, leader dots, the wrap — and eagle
    proved a reviewer can read the context as the subject: a contents
    entry's XXXII was overridden with the previous line's ALTGELD because
    the leader dots dominated the strip. The box says: judge THIS ink."""
    from PIL import ImageDraw
    d = ImageDraw.Draw(crop)
    x0, y0, x1, y1 = rect
    d.rectangle((max(0, x0 - 2), max(0, y0 - 2),
                 min(crop.width - 1, x1 + 2), min(crop.height - 1, y1 + 2)),
                outline=(224, 122, 0), width=2)
    return crop


def crop_data_uris(report, dpi=200, ctx_px=240, base_dir=None) -> dict:
    """Crop each dispute's ink from the geometry recorded at adjudication
    time — exact by construction. Rows without recorded geometry render
    text-only: the old relocate-by-fresh-OCR fallback conflated two
    counting bases and re-parsed TSV without the hyphen join (I6/R4,
    2026-08-18 ledger), so an honest "no crop" replaced it — a missing
    crop must never block the sheet, and neither may a missing PDF: the
    recorded path is tried as given and relative to the report's own
    directory, and failure degrades to a crop-less sheet (T3)."""
    import pypdfium2 as pdfium

    by_page = collections.defaultdict(list)
    occs = assign_occurrences(report["disputes"])
    for idx, d in enumerate(report["disputes"]):
        by_page[d.get("pdf_page", d["page"])].append((idx, d, occs[idx]))
    pdf = _beside(report.get("pdf"), base_dir)
    if not pdf:
        print(f"  [!] source PDF not found ({report.get('pdf')!r}) — "
              f"sheet renders without ink crops")
        return {}
    doc = pdfium.PdfDocument(pdf)
    uris = {}
    for pno, rows in sorted(by_page.items()):
        if pno >= len(doc):
            continue
        img = doc[pno].render(scale=dpi / 72, draw_annots=False).to_pil()
        for idx, d, occ in rows:
            # cbox is the crop's box when recorded: hyphen-joined fragments
            # unioned and line-edge context pulled across the wrap. tbox is
            # the token's own ink, outlined inside the crop.
            rec = d.get("cbox") or d.get("box")
            tbox = d.get("box")
            if rec:
                x0, y0 = int(rec[0] * img.width), int(rec[1] * img.height)
                x1, y1 = int(rec[2] * img.width), int(rec[3] * img.height)
            else:
                continue
            box = (max(0, x0 - ctx_px), max(0, y0 - 6),
                   min(img.width, x1 + ctx_px), min(img.height, y1 + 6))
            crop = img.crop(box).convert("RGB")
            if tbox:
                tpx = (int(tbox[0] * img.width), int(tbox[1] * img.height),
                       int(tbox[2] * img.width), int(tbox[3] * img.height))
            _outline_token(crop, (tpx[0] - box[0], tpx[1] - box[1],
                                  tpx[2] - box[0], tpx[3] - box[1]))
            buf = io.BytesIO()
            # JPEG, not PNG: the crops are slices of a grayscale scan and
            # the reviewer judges word identity, not pixels — a Citadel
            # sheet weighed 79MB in lossless crops and ~a tenth of that in
            # JPEG, with the amber outline baked in before encoding (E2)
            crop.save(buf, format="JPEG", quality=85)
            uris[idx] = ("data:image/jpeg;base64,"
                         + base64.b64encode(buf.getvalue()).decode())
    return uris


def _beside(path, base_dir=None) -> "str | None":
    """A book file the report names, as recorded or beside the report.

    Reports record the book's path relative to where the build ran; the
    sheet renders from wherever the report now sits. Files only: an empty
    name joined to the base directory IS the directory, and that once
    reached pypdfium2 as a document."""
    if not path:
        return None
    for cand in ([path] + ([os.path.join(base_dir, path)] if base_dir else [])):
        if os.path.isfile(cand):
            return cand
    return None


def _run_crops(report, dpi=200, ctx_px=120, base_dir=None) -> dict:
    """Ink crops for the surya-only runs, keyed (page, run-index)."""
    import pypdfium2 as pdfium
    rows = report.get("surya_only") or []
    if not rows:
        return {}
    pdf = _beside(report.get("pdf"), base_dir)
    if not pdf:
        return {}
    doc = pdfium.PdfDocument(pdf)
    uris = {}
    for prow in rows:
        pno = prow["page"]
        # The page of the PDF, which is not the page of the book wherever a
        # blank or a duplicate was dropped ahead of it. The word crops have
        # always used it; these used the book's number, and showed the ink
        # of the wrong page from the first dropped page on.
        at = prow.get("pdf_page", pno)
        if at >= len(doc):
            continue
        img = None
        for ri, r in enumerate(prow["runs"]):
            box = r.get("box")
            if img is None:
                img = doc[at].render(scale=dpi / 72,
                                     draw_annots=False).to_pil()
            if box:
                pad = 60 if r.get("approx") else 10
                cx = 200 if r.get("approx") else ctx_px
                x0, y0 = int(box[0] * img.width), int(box[1] * img.height)
                x1, y1 = int(box[2] * img.width), int(box[3] * img.height)
                crop = img.crop((max(0, x0 - cx), max(0, y0 - pad),
                                 min(img.width, x1 + cx),
                                 min(img.height, y1 + pad))).convert("RGB")
            else:
                # No location at all: the page itself, small — a reader
                # hunting handwriting needs the scene, not a blank row.
                w = 560
                crop = img.resize((w, int(img.height * w / img.width)))                           .convert("RGB")
            buf = io.BytesIO()
            crop.save(buf, format="JPEG", quality=85)
            uris[(pno, ri)] = ("data:image/jpeg;base64,"
                               + base64.b64encode(buf.getvalue()).decode())
    return uris


def _surya_only_html(report, run_crops, contexts=None) -> list:
    """The advisory tier: runs ranked no-ink first, then implausibly dense,
    then long. Not stageable — there is no old→new here; a fake run means
    the PAGE needs a human's ruling, not a word swap."""
    rows = report.get("surya_only") or []
    contexts = contexts or {}
    allr = [(p["page"], ri, r) for p in rows
            for ri, r in enumerate(p["runs"])]
    if not allr:
        return []
    confirmed = [t for t in allr
                 if t[2].get("verdict") in ("confirmed", "located")]
    resolved = []
    flat = [t for t in allr if t[2].get("verdict")
            in (None, "advisory", "silent")]

    def rank(t):
        _, _, r = t
        inkless = r.get("ink") is not None and r["ink"] < 0.005
        dense = (r.get("density") or 0) > 2.0
        return (not inkless, not dense, -r["n"])

    flat.sort(key=rank)
    if not flat and not confirmed:
        return []
    parts = [f"<h2>{SURYA_ONLY_TITLE} ({len(flat)})</h2>",
             (f"<p class='alt'>{len(confirmed)} run(s) the second look "
              f"found on the page are collapsed below.</p>"
              if confirmed else ""),
             "<p class='alt'>Advisory rows — tesseract cannot invent, so "
             "its silence under these words is testimony; but two honest "
             "causes dominate: tiny print it cannot resolve, and regions "
             "its page-level layout pass discards without reading — words "
             "it recognises instantly when handed the crop. No ink under "
             "a run is the suspicious case. Crops appear only where a "
             "re-read of the crop confirmed the words are in it. These "
             "rows stage deletions, not corrections.</p>"]
    for pno, ri, r in flat:
        ink = r.get("ink")
        chips = []
        if ink is not None and ink < 0.005:
            chips.append("<b style='color:#b00'>no ink under them</b>")
        elif ink is not None:
            chips.append("ink present")
        if r.get("density") and r["density"] > 2.0:
            chips.append(f"{r['density']:.1f}× the page's own type density")
        crop = run_crops.get((pno, ri))
        why = {"unlocated": "scattered blocks — not one span in the layer "
                            "(handwriting and inset documents live here)",
               "unrelated": "the located crop showed other ink",
               "inked-unreadable": "ink is there but no reader resolves it "
                                   "(rotated, display, or degraded type)",
               }.get(r.get("why"))
        if why:
            chips.append(why)
        if r.get("approx"):
            chips.append("crop ≈ anchored on the run's rarest word")
        shown = r["text"] if len(r["text"]) <= 240 else r["text"][:240] + "…"
        old_attr = html.escape(r["text"], quote=True)
        parts.append(
            f"<div class='row srow' data-sidx='so-{pno}-{ri}' "
            f"data-page='{pno}' data-old='{old_attr}'>"
            "<div>"
            f"p{pno} · {r['n']} words · {' · '.join(chips) or 'no geometry'}"
            f" · <button data-sact='delete'>delete run</button>"
            "<span class='state'></span>"
            f"<br>{_context_html(contexts.get((pno, ri)), shown)}</div>"
            + (f"<img class='ink' src='{crop}'>" if crop else "")
            + "</div>")
    if confirmed:
        parts.append(f"<details><summary>Seen by the second look "
                     f"({len(confirmed)})</summary>")
        for pno, ri, r in confirmed:
            crop = run_crops.get((pno, ri))
            parts.append("<div class='row'><div>"
                         f"p{pno} · {r['n']} words"
                         f"<br>{_context_html(contexts.get((pno, ri)), r['text'][:240])}</div>"
                         + (f"<img class='ink' src='{crop}'>" if crop else "")
                         + "</div>")
        parts.append("</details>")
    return parts


def _row_html(r, crops) -> str:
    shipped = r.get("shipped") or r["surya"]
    winner = r.get("winner") or ""
    rung = r.get("rung") or "abstain"
    acc = RUNG_ACCURACY.get(rung)
    chip = f"{rung} {acc}" if acc else rung
    if r.get("overruled"):
        # left open because the second reader reads the word the rule
        # ruled against: the reviewer should know a rule had spoken
        chip = f"{rung} · {r['overruled']} withdrawn"
    elif r.get("note") == "readers split":
        chip = f"{rung} · readers split"
    crop = crops.get(r["idx"])
    ink = (f'<img class="ink" alt="ink" src="{crop}">' if crop
           else '<span class="noink">no crop — ink not located</span>')
    proposal = (f'<button class="apply" data-act="apply">apply '
                f'<b>{html.escape(winner)}</b></button>' if winner
                and winner != shipped else "")
    # the word's neighbours, when the report addressed it: they go out
    # with the decision so the apply step can find the same word again
    site = ""
    if "before" in r:
        site = "".join(
            f'data-{side}="{html.escape(" ".join(r.get(side) or []), quote=True)}" '
            for side in ("before", "after"))
    return (
        f'<div class="row" tabindex="0" data-idx="{r["idx"]}" '
        f'data-page="{r["page"]}" data-occ="{r["occurrence"]}" {site}'
        f'data-old="{html.escape(shipped, quote=True)}" '
        f'data-new="{html.escape(winner, quote=True)}" '
        f'data-rung="{html.escape(rung, quote=True)}">'
        f'{ink}<div class="body">'
        f'<span class="loc">p.{r["page"]}</span>'
        f'<span class="chip">{html.escape(chip)}</span> '
        f'shipped <b class="w">{html.escape(shipped)}</b> '
        f'&nbsp;<span class="alt">surya {html.escape(r["surya"])} · '
        f'tesseract {html.escape(r["tesseract"])}</span><br>'
        f'<button class="keep" data-act="keep">keep</button> {proposal} '
        f'<input class="own" type="text" spellcheck="false" '
        f'value="{html.escape(shipped, quote=True)}" '
        f'title="your reading of the ink — overrides every rung">'
        f'<span class="state"></span>'
        f'</div></div>')


_JS = """
const KEY = 'codicology-review-' + document.body.dataset.fp;
const state = JSON.parse(localStorage.getItem(KEY) || '{}');
function paint(row) {
  const s = state[row.dataset.idx] || {};
  row.querySelector('.state').textContent =
    s.choice === 'apply' ? '\\u2192 ' + row.dataset.new :
    s.choice === 'human' ? '\\u2192 ' + s.text + ' (yours)' : '';
  row.classList.toggle('staged', !!s.choice && s.choice !== 'keep');
  const n = Object.values(state).filter(
    s => s.choice && s.choice !== 'keep').length;
  document.getElementById('count').textContent = n + ' corrections staged';
}
document.querySelectorAll('.row:not(.srow)').forEach(row => {
  row.addEventListener('click', e => {
    const act = e.target.closest('[data-act]');
    if (act) {
      state[row.dataset.idx] = {choice: act.dataset.act === 'apply'
        ? 'apply' : 'keep'};
      save(); paint(row);
    }
  });
  const own = row.querySelector('.own');
  if (!own) return;
  own.addEventListener('input', () => {
    state[row.dataset.idx] = own.value === row.dataset.old
      ? {choice: 'keep'} : {choice: 'human', text: own.value};
    save(); paint(row);
  });
  paint(row);
});
function paintS(row) {
  const s = state[row.dataset.sidx] || {};
  row.querySelector('.state').textContent =
    s.choice === 'delete' ? ' \u2192 run deleted' : '';
  row.classList.toggle('staged', s.choice === 'delete');
  const n = Object.values(state).filter(
    s => s.choice && s.choice !== 'keep').length;
  document.getElementById('count').textContent = n + ' corrections staged';
}
document.querySelectorAll('.srow').forEach(row => {
  row.querySelector('[data-sact]').addEventListener('click', () => {
    const cur = state[row.dataset.sidx] || {};
    state[row.dataset.sidx] = cur.choice === 'delete'
      ? {choice: 'keep'} : {choice: 'delete'};
    save(); paintS(row);
  });
  paintS(row);
});
function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
document.getElementById('export').addEventListener('click', () => {
  const decisions = [];
  document.querySelectorAll('.row').forEach(row => {
    const s = state[row.dataset.idx];
    if (!s || s.choice === 'keep') return;
    const d = {
      page: +row.dataset.page, occurrence: +row.dataset.occ,
      old: row.dataset.old,
      new: s.choice === 'human' ? s.text : row.dataset.new,
      source: s.choice === 'human' ? 'human' : 'ladder',
      rung: s.choice === 'human' ? 'human' : row.dataset.rung};
    if ('before' in row.dataset) {
      d.before = row.dataset.before.split(' ').filter(Boolean);
      d.after = row.dataset.after.split(' ').filter(Boolean);
    }
    decisions.push(d);
  });
  document.querySelectorAll('.srow').forEach(row => {
    const s = state[row.dataset.sidx];
    if (!s || s.choice !== 'delete') return;
    decisions.push({
      page: +row.dataset.page, occurrence: 0,
      old: row.dataset.old, new: '',
      source: 'human', rung: 'surya-only', kind: 'delete_run'});
  });
  const meta = JSON.parse(document.getElementById('meta').textContent);
  const blob = new Blob([JSON.stringify(
    Object.assign(meta, {decisions: decisions}), null, 1)],
    {type: 'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = meta.name + '.decisions.json';
  a.click();
});
document.addEventListener('keydown', e => {
  if (e.target.classList.contains('own')) return;
  const rows = [...document.querySelectorAll('.row')];
  const cur = rows.indexOf(document.activeElement.closest('.row'));
  if (e.key === 'j' || e.key === 'k') {
    const nxt = rows[cur + (e.key === 'j' ? 1 : -1)] || rows[0];
    nxt.focus(); nxt.scrollIntoView({block: 'center'});
  }
  const row = rows[cur];
  if (!row) return;
  if (e.key === 'a' && row.dataset.new) {
    state[row.dataset.idx] = {choice: 'apply'}; save(); paint(row);
  }
  if (e.key === 's') {
    state[row.dataset.idx] = {choice: 'keep'}; save(); paint(row);
  }
  if (e.key === 'e') { e.preventDefault(); row.querySelector('.own').focus(); }
});
"""

_CSS = """
body { font: 14px/1.45 system-ui, sans-serif; margin: 1.2rem auto;
       max-width: 62rem; padding: 0 1rem; background: #fafaf7; color: #222; }
h1 { font-size: 1.25rem; } h2 { font-size: 1rem; margin: 1.4rem 0 .4rem; }
.row { display: flex; gap: .8rem; align-items: center; padding: .45rem .5rem;
       border-bottom: 1px solid #e4e2da; }
.row:focus { outline: 2px solid #7a6a3a; }
.row.staged { background: #f2ecd9; }
.ink { max-height: 96px; max-width: 30rem; border: 1px solid #ddd; }
.noink { color: #999; font-style: italic; min-width: 12rem; }
.w { font-family: Georgia, serif; font-size: 1.1em; }
.alt, .loc { color: #777; font-size: .85em; }
.loc { margin-right: .5em; }
.chip { font-family: ui-monospace, monospace; font-size: .78em;
        background: #e8e4d6; border-radius: 3px; padding: 0 .35em; }
button { font: inherit; font-size: .85em; cursor: pointer; }
input.own { font-family: Georgia, serif; width: 11rem; margin-left: .5em; }
.state { color: #7a6a3a; margin-left: .5em; font-size: .85em; }
.ctx { font-family: Georgia, serif; color: #555; }
.ctx mark { background: #ffe9a8; color: #222; padding: 0 .1em; }
details { margin: .3rem 0; } summary { cursor: pointer; color: #555; }
#bar { position: sticky; top: 0; background: #fafaf7; padding: .5rem 0;
       border-bottom: 2px solid #d8d4c6; display: flex; gap: 1rem;
       align-items: center; }
@media (prefers-color-scheme: dark) {
  body, #bar { background: #191919; color: #ddd; }
  .row { border-color: #333; } .row.staged { background: #2c2717; }
  .chip { background: #333; }
  .ctx { color: #aaa; } .ctx mark { background: #5a4a14; color: #eee; }
  /* The crops stay UNINVERTED in dark mode, deliberately: this sheet's
     whole job is judging ink — faint marks, bleed-through, paper tone —
     and invert(1) hue-rotate(180deg), the usual dark-UI image trick,
     destroys exactly that evidence. A white scan on a dark page is a
     photograph, not a glare problem. */
  .ink { border-color: #444; } }
"""


def render_sheet(report, crops=None) -> str:
    crops = crops or {}
    tiers = build_tiers(report["disputes"])
    name = re.sub(r"\.json$", "", os.path.basename(str(report.get("name")
                  or report.get("epub") or "book")))
    meta = {"name": name, "pdf": report.get("pdf", ""),
            "epub": report.get("epub", ""),
            "page_files": report.get("page_files"),
            "sheet": _fingerprint(report)}
    if report.get("tokens") is not None:
        # the generation of word pattern the rows are addressed in; the
        # apply step reads the page the same way (adjudicate.tokens_as)
        meta["tokens"] = report["tokens"]
    meta = json.dumps(meta)
    parts = [
        "<!doctype html><meta charset='utf-8'>",
        f"<title>{html.escape(name)} — dispute review</title>",
        f"<style>{_CSS}</style>",
        f"<body data-fp='{_fingerprint(report)}'>",
        f"<script type='application/json' id='meta'>{meta}</script>",
        "<div id='bar'><h1 style='margin:0'>Dispute review — "
        f"{html.escape(name)}</h1>",
        "<span id='count'>0 corrections staged</span>",
        "<button id='export'>export decisions</button>",
        "<span class='alt'>keys: j/k move · a apply · s keep · "
        "e your reading</span></div>",
        "<p class='alt'>Nothing here has changed the book. The EPUB ships "
        "what “shipped” says; export your decisions and feed them "
        "to <code>codicology apply</code> or the next rebuild. Your typed "
        "reading outranks every rung. The amber box marks where the "
        "<em>witness</em> engine read its word — almost always the disputed "
        "ink itself, but when the witness misread page furniture (leader "
        "dots, rules) its box can sit apart from the shipped word's true "
        "position. When the box seems to mark nothing, read the whole "
        "crop before ruling.</p>",
    ]
    parts += _surya_only_html(
        report, _run_crops(report, base_dir=report.get("_base_dir")),
        _run_contexts(report, base_dir=report.get("_base_dir")))
    for key in ("catches", "broken", "open"):
        rows = tiers[key]
        if not rows:
            continue
        parts.append(f"<h2>{TIER_TITLES[key]} ({len(rows)})</h2>")
        parts += [_row_html(r, crops) for r in rows]
    if tiers["systematic"]:
        groups = collections.defaultdict(list)
        for r in tiers["systematic"]:
            groups[(r["surya"], r["tesseract"])].append(r)
        parts.append(f"<h2>{TIER_TITLES['systematic']} "
                     f"({len(tiers['systematic'])})</h2>")
        for (a, b), rows in sorted(groups.items(), key=lambda kv:
                                   -len(kv[1])):
            parts.append(
                f"<details><summary>{html.escape(a)} → "
                f"{html.escape(b)} ×{len(rows)}</summary>")
            parts += [_row_html(r, crops) for r in rows]
            parts.append("</details>")
    if tiers["confirmed"]:
        parts.append(
            f"<details><summary><h2 style='display:inline'>"
            f"{TIER_TITLES['confirmed']} ({len(tiers['confirmed'])})"
            f"</h2></summary>")
        parts += [_row_html(r, crops) for r in tiers["confirmed"]]
        parts.append("</details>")
    parts.append(f"<script>{_JS}</script>")
    return "\n".join(parts)


def _regex_sites(body, old) -> list:
    """Every word-boundary match of `old` in the body's text nodes, as
    spans of the body. Tags are never searched. This is how a hand-written
    `old` that is not one word of the page is found — the printed "&c.",
    a phrase."""
    pat = re.compile(r"(?<![\w])" + re.escape(old) + r"(?![\w])")
    out, at = [], 0
    for k, seg in enumerate(re.split(r"(<[^>]+>)", body)):
        if k % 2 == 0:
            out += [(at + m.start(), at + m.end()) for m in pat.finditer(seg)]
        at += len(seg)
    return out


def _run_pattern(words: list) -> "re.Pattern":
    """The words of a run, in order, as they may stand in the body.

    A run was recorded as TOKENS — punctuation stripped — while the
    body keeps its colons, commas and the periods in "U.S.": a legend
    reading "AA': twin hulls G: main windlass" must match the run
    "AA' twin hulls G main windlass". Anything non-word may stand
    between consecutive words; six of nine deletions once went stale on
    exactly this.
    …and whole NUMBERS: the tokenizer drops them, so "PATENT No SEPTEMBER"
    must match "PATENT No. 913, SEPTEMBER 12, 1838" — digits are word
    characters a punctuation-only gap cannot cross."""
    gap = r"(?:[^\w]|\b\d+\b)*?"
    return re.compile(r"(?<![\w])"
                      + gap.join(re.escape(w) for w in words)
                      + r"(?![\w])")


CONTEXT_WORDS = 20      # words of the shipped text framed on each side


def run_context(text: str, run: str, words: int = CONTEXT_WORDS) -> "tuple | None":
    """(before, run, after): the run as it stands in its own sentence.

    The advisory tier shows words tesseract never saw, and shown bare
    they all look alike — "never developed a comprehensive" reads as
    plausible invention, and a reviewer deleted it. Framed in the
    sentence the book prints it in, it is unmistakably prose. The frame
    is up to `words` words each side, cut back to the nearest sentence
    boundary where one falls inside the window and marked with an
    ellipsis where none does. Located on the tokenizer's own fold of
    the text, so the recorded words match as recorded; whitespace is
    flattened for the eye. None when the run is not in the text — the
    row then shows the bare words, as before.
    """
    ws = run.split()
    if not ws:
        return None
    folded = fold_text(text)
    m = _run_pattern(ws).search(folded)
    if not m:
        return None
    end_of = re.compile(r"[.!?][\"')\]]*\s")
    pre = folded[:m.start()].split()
    before = " ".join(pre[-words:])
    cuts = list(end_of.finditer(before + " "))
    if cuts:
        before = before[cuts[-1].end():]
    elif before and len(pre) > words:
        before = "…" + before
    post = folded[m.end():].split()
    after = " ".join(post[:words])
    cut = end_of.search(after + " ")
    if cut:
        after = after[:cut.end()].rstrip()
    elif after and len(post) > words:
        after += "…"
    hit = " ".join(m.group(0).split())
    lead = " " if before and folded[m.start() - 1].isspace() else ""
    trail = " " if after and folded[m.end():m.end() + 1].isspace() else ""
    return (before + lead if before else "", hit,
            trail + after if after else "")


def _run_contexts(report, base_dir=None) -> dict:
    """The sentence around each surya-only run, keyed (page, run-index),
    read from the shipped EPUB the way crops are read from the PDF."""
    rows = report.get("surya_only") or []
    epub = _beside(report.get("epub"), base_dir)
    if not rows or not epub:
        return {}
    try:
        pages = epub_page_texts(epub)
    except Exception:
        return {}
    out = {}
    for prow in rows:
        text = pages.get(prow["page"])
        if not text:
            continue
        for ri, r in enumerate(prow["runs"]):
            ctx = run_context(text, r.get("text") or "")
            if ctx:
                out[(prow["page"], ri)] = ctx
    return out


def _context_html(ctx, fallback: str) -> str:
    """The run marked inside its sentence, or the bare run words."""
    if not ctx:
        return f"<i>{html.escape(fallback)}</i>"
    before, hit, after = ctx
    return (f"<span class='ctx'>{html.escape(before)}<mark>"
            f"{html.escape(hit)}</mark>{html.escape(after)}</span>")


def _delete_from_xhtml(xhtml, old) -> "str | None":
    """Remove one run of words from a page's text nodes, tags untouched.

    A run is contiguous in the READING — but the page's markup may split
    it across several blocks: a diagram legend is one <li> per line, a
    flowchart's labels one <p> each, and six of a reviewer's nine
    deletions once came back stale for exactly that. The words are
    matched in order across consecutive text segments, with the markup
    between them standing where it stands; only text is removed, and a
    block the deletion empties entirely is dropped so no blank paragraph
    marks the spot. A run whose words are not found in order reports
    stale — partial deletion is worse than none."""
    words = old.split()
    if not words:
        return None
    segs = re.split(r"(<[^>]+>)", xhtml)
    # the text segments, concatenated with single spaces at the joins, and
    # a map from each concatenated character back to (segment, offset)
    text_parts, owner = [], []
    for k, seg in enumerate(segs):
        if k % 2 == 1:
            continue
        if text_parts:
            text_parts.append(" ")
            owner.append((None, None))
        text_parts.append(seg)
        owner.extend((k, off) for off in range(len(seg)))
    flat = "".join(text_parts)
    m = _run_pattern(words).search(flat)
    if not m:
        return None
    cuts: dict = {}
    for pos in range(m.start(), m.end()):
        k, off = owner[pos]
        if k is None:
            continue
        lo, hi = cuts.get(k, (off, off))
        cuts[k] = (min(lo, off), max(hi, off + 1))
    for k, (lo, hi) in cuts.items():
        seg = segs[k]
        left, right = seg[:lo], seg[hi:]
        # a cut that reaches a segment's edge leaves the neighbour's
        # separating space stranded at the block boundary
        if not left.strip():
            right = right.lstrip()
        if not right.strip():
            left = left.rstrip()
        segs[k] = re.sub(r"[ \t]{2,}", " ", left + right)
    out = "".join(segs)
    out = re.sub(r"<(p|li)\b[^>]*>\s*</\1>", "", out)
    return out


_CURL = str.maketrans({"'": "\u2019"})


def _shapes(old) -> list:
    """`old` in the shapes the page may store it: the reviewer's word was
    recorded through the adjudicator's fold (straight apostrophes) and
    lives in text space, while the book may carry curly ones and stores
    the printed &c. as &amp;c."""
    return list(dict.fromkeys((old, old.translate(_CURL),
                               old.translate(_TYPO),
                               html.escape(old, quote=False))))


def _context_score(toks, lo, hi, before, after) -> int:
    """How many of a site's recorded neighbours stand where they stood,
    were the site toks[lo:hi]."""
    n = 0
    for k, w in enumerate(reversed(before), 1):
        if lo - k >= 0 and fold_word(toks[lo - k]) == fold_word(w):
            n += 1
    for k, w in enumerate(after):
        if hi + k < len(toks) and fold_word(toks[hi + k]) == fold_word(w):
            n += 1
    return n


def _reads_already(folded, at, old, new) -> bool:
    """Whether the page already reads `new` around the `old` at
    folded[at:]. A correction that adds what the tokeniser drops — "gs"
    to "μgs", "D" to "[α]D" — leaves its own word standing, and applying
    it a second time must not double the addition."""
    if new == old:
        return False
    q = new.find(old)
    while q >= 0:
        if at >= q and folded[at - q:at - q + len(new)] == new:
            return True
        q = new.find(old, q + 1)
    return False


def _inside_a_longer_word(folded, at, n) -> bool:
    """Whether the token at folded[at:at + n] is a piece of a longer word.

    The tokeniser's alphabet is Latin-1 and it stops at any letter outside
    it: a page's "diabolē" is read as "diabol", and a correction to that
    token would be written inside the word — "diaboléē". Only a letter or
    an accent continues a word. A digit does not: a note's number printed
    hard against "Lloyd's" leaves "Lloyd's" a whole word, and correctable."""
    def continues(ch):
        return unicodedata.category(ch)[0] in "LM"
    return bool((at > 0 and continues(folded[at - 1]))
                or (at + n < len(folded) and continues(folded[at + n])))


def _hard_against(hard, at, n) -> tuple:
    """What the page prints with no space between it and the word at
    hard[at:at + n]: (the run before it, the run after it, where the
    first begins, where the second ends). `hard` is the page's hard_view:
    a change of type — a subscript, an italic — is no space."""
    a, b = at, at + n
    while a > 0 and not hard[a - 1].isspace():
        a -= 1
    while b < len(hard) and not hard[b].isspace():
        b += 1
    return (hard[a:at].replace(STRUCK, ""),
            hard[at + n:b].replace(STRUCK, ""), a, b)


def _fitted(new, toks, k, folded, at, hard=None) -> tuple:
    """What to write over the word toks[k], standing at folded[at:], for
    the page to read `new` there: (text, None), or (None, why not).

    A reviewer types the printed word, and the printed word can be more
    than the tokeniser's. "8IMM" is the token "IMM" with an 8 against it:
    the correction "81MM" is for both, and written over the token alone
    it prints "881MM". So what `new` shares with the characters standing
    hard against the word is not written a second time. Where `new`
    repeats its surroundings in a way that cannot be trimmed, the page is
    left as it is and the decision reported: a formula typed out whole
    for one of its symbols ("C15H23ClN2O" for "ClN") would be printed
    inside the formula, and three installed books had exactly that."""
    new_t = new.translate(_TYPO)
    left, right, a, b = _hard_against(hard or folded, at, len(toks[k]))
    ms = list(WORD.finditer(new_t))
    head = new_t[:ms[0].start()] if ms else new_t
    tail = new_t[ms[-1].end():] if ms else new_t
    lo = next((j for j in range(min(len(head), len(left)), 0, -1)
               if left.endswith(new_t[:j])), 0)
    hi = next((j for j in range(min(len(tail), len(right)), 0, -1)
               if right.startswith(new_t[len(new_t) - j:])), 0)
    if (lo or hi) and (lo + hi >= len(new_t)
                       or new_t[lo:len(new_t) - hi] == toks[k]):
        return None, f"the page already reads {new!r} there"
    if not lo and head.strip() and head.strip() in left:
        return None, (f"the page already prints {head.strip()!r} against "
                      f"it — correct the page by hand")
    if not hi and tail.strip() and tail.strip() in right:
        return None, (f"the page already prints {tail.strip()!r} against "
                      f"it — correct the page by hand")
    nt = [m.group() for m in ms]
    if (any(toks[k - j:k] == nt[:j] for j in range(1, min(len(nt), k + 1)))
            or any(toks[k + 1:k + 1 + j] == nt[len(nt) - j:]
                   for j in range(1, len(nt)))):
        return None, ("the correction repeats the words beside it — "
                      "correct the page by hand")
    if not ms and new_t.strip():
        # nothing in it the tokeniser can see: look at the characters
        want = new_t.strip()
        if (folded[b:b + len(want) + 8].lstrip().startswith(want)
                or folded[max(0, a - len(want) - 8):a].rstrip()
                .endswith(want)):
            return None, (f"the page already prints {want!r} beside it — "
                          f"correct the page by hand")
    if lo or hi:
        if len(new_t) != len(new):
            return None, ("the correction overlaps what the page prints "
                          "against the word — correct the page by hand")
        return new[lo:len(new) - hi], None
    return new, None


def _occurrence(d) -> int:
    try:
        return max(0, int(d.get("occurrence", 0)))
    except (TypeError, ValueError):
        return 0


def _locate(d, body, folded, words, kmax, trust_ordinal,
            hard=None) -> tuple:
    """The span of `body` one swap decision names and what to write
    there: ((start, end, text), None), or (None, why it cannot be placed).

    A decision that cannot be placed with certainty is stale. Stale
    decisions are surfaced and never guessed at — a correction on the
    wrong word is worse than none, and nobody is told."""
    old, new = d.get("old"), d.get("new")
    if not isinstance(old, str) or not old or not isinstance(new, str):
        return None, "it names no word"
    old_t, new_t = old.translate(_TYPO), new.translate(_TYPO)
    occ = _occurrence(d)
    toks = [w[0] for w in words]
    cands = [k for k, t in enumerate(toks) if t == old_t]
    already = f"the page already reads {new!r} there"
    k = None

    if "before" not in d and "after" not in d:
        # an unaddressed row: its ordinal counted disputes, not instances.
        # The two counts are certain to agree only when the file accounts
        # for every instance of the word the page prints.
        if cands:
            sites = [(words[k][1], words[k][2]) for k in cands]
        else:
            sites = next((got for got in (_regex_sites(body, shape)
                                          for shape in _shapes(old))
                          if got), [])
            if not sites:
                return None, "the word is no longer on the page"
        if len(sites) != kmax + 1:
            return None, (f"its sheet numbered only the disputed words and "
                          f"the page prints {old!r} {len(sites)} times — "
                          f"adjudicate again and review it on the new sheet")
        if cands:
            k = cands[occ]
            if _reads_already(folded, words[k][3], old_t, new_t):
                return None, already
        pick = sites[occ]
    else:
        if not cands:
            return None, "the word is no longer on the page"
        live = [k for k in cands
                if not _reads_already(folded, words[k][3], old_t, new_t)]
        if not live:
            return None, already
        before, after = list(d.get("before") or []), list(d.get("after") or [])
        n = len(before) + len(after)
        score = {k: _context_score(toks, k, k + 1, before, after)
                 for k in live}
        best = max(score.values())
        if 2 * best < n:
            return None, "the words around it have changed"
        # the same neighbours around `new`: the correction is already in
        new_toks = tokens(new)
        if n and new_toks and new_toks != [old_t]:
            m = len(new_toks)
            if any(toks[p:p + m] == new_toks
                   and _context_score(toks, p, p + m, before, after) >= best
                   for p in range(len(toks) - m + 1)):
                return None, already
        winners = [k for k in live if score[k] == best]
        if len(winners) == 1:
            k = winners[0]
        elif trust_ordinal and occ < len(cands) and cands[occ] in winners:
            k = cands[occ]
        else:
            return None, (f"{len(winners)} places on the page fit it "
                          f"equally")
        pick = (words[k][1], words[k][2])
    text = new
    if k is not None:
        if _inside_a_longer_word(folded, words[k][3], len(toks[k])):
            return None, "the page prints it inside a longer word"
        text, why = _fitted(new, toks, k, folded, words[k][3], hard)
        if text is None:
            return None, why
    if pick[0] is None:
        return None, "the word is split across the page's markup"
    return (pick[0], pick[1], text), None


def apply_page_decisions(xhtml, rows) -> tuple:
    """One page's decisions against that page: (xhtml, outcomes), where
    outcomes[k] is None for a decision that landed and otherwise the
    reason it did not. Both consumers of a decisions file MUST pass
    through here — a correction that lands via the rebuild but goes stale
    via `codicology apply` is the divergence the 2026-08-18 review
    confirmed (I2).

    Only the body is read or written: the <title> is not the reviewer's
    text. Deletions run first — both kinds were recorded against the same
    shipped text, and a word swap mutates the very characters a deletion
    must match; the swap inside a deleted run then reports stale, which
    is true: its site is gone, and gone was the point. Every swap is then
    located on the same text before any is written, so one correction
    cannot move another's neighbours or renumber its word."""
    lo, hi = body_span(xhtml)
    body = xhtml[lo:hi]
    outcomes = [None] * len(rows)
    deleted = False
    for k, d in enumerate(rows):
        if d.get("kind") != "delete_run":
            continue
        got = next((g for g in (_delete_from_xhtml(body, shape)
                                for shape in _shapes(str(d.get("old") or "")))
                    if g is not None), None)
        if got is None:
            outcomes[k] = "the run is not on the page"
        else:
            body, deleted = got, True
    swaps = [k for k, d in enumerate(rows) if d.get("kind") != "delete_run"]
    if swaps:
        folded, words = page_words(body)
        hard = hard_view(body)
        kmax = collections.Counter()
        for k in swaps:
            d = rows[k]
            if "before" not in d and "after" not in d:
                kmax[d.get("old")] = max(kmax[d.get("old")], _occurrence(d))
        edits = []
        for k in swaps:
            d = rows[k]
            # after a deletion the ordinal counts a page that is gone:
            # the neighbours alone must place the word
            span, why = _locate(d, body, folded, words, kmax[d.get("old")],
                                trust_ordinal=not deleted, hard=hard)
            if span is None:
                outcomes[k] = why
            elif any(a < span[1] and span[0] < b for a, b, _ in edits):
                outcomes[k] = "another decision already corrected this word"
            else:
                # the reviewer's word is text; the page is xhtml
                edits.append((span[0], span[1],
                              html.escape(span[2], quote=False)))
        for a, b, new in sorted(edits, reverse=True):
            body = body[:a] + new + body[b:]
    return xhtml[:lo] + body + xhtml[hi:], outcomes


def apply_one_decision(xhtml, d) -> "str | None":
    """One decision against one page's xhtml: the corrected page, or None
    when the decision is stale."""
    got, (why,) = apply_page_decisions(xhtml, [d])
    return got if why is None else None


def coerce_page(d) -> "int | None":
    """A decision's page as an index, or None with the caller printing why.
    Hand-edited files carry numeric strings; a rebuild must not crash on
    them, and must not silently discard them either (I3)."""
    try:
        return int(d.get("page"))
    except (TypeError, ValueError):
        return None


def pages_guard(recorded_page_files, actual_page_files) -> "str | None":
    """The refusal that keeps decisions off renumbered builds (T4): a
    decisions file recorded against N pages must not index a build with a
    different count — --drop-pages shifts every later index, and a
    coincidental word match would land a correction on the wrong page.
    Returns the refusal message, or None when the shapes agree (or the
    file predates the guard and carries no count)."""
    if recorded_page_files is None or recorded_page_files == actual_page_files:
        return None
    return (f"decisions were recorded against {recorded_page_files} page "
            f"file(s) but this build has {actual_page_files} — the page "
            f"indices cannot "
            f"be trusted; nothing applied. Re-adjudicate and re-review "
            f"under the current build shape.")


def apply_decisions(epub_path, decisions_path, out=None) -> dict:
    """Feed the reviewer's exported decisions back into the EPUB.

    The adjudicator never rewrites; this is the human act it reports
    toward. Corrections land in the page's text at the word each decision
    is addressed to, the original is kept beside the book as .preapply,
    and every decision is accounted for: applied or stale, never silent."""
    import shutil

    from . import binding

    with open(decisions_path) as fh:
        dec = json.load(fh)
    # Pages, not files: a book whose pages are bound into chapters has as
    # many pages as it had, and a decision is addressed to a word of one.
    n_pages = len(binding.page_bodies(epub_path))
    refusal = pages_guard(dec.get("page_files"), n_pages)
    if refusal:
        print(f"  refused: {refusal}")
        return {"applied": [], "stale": list(dec.get("decisions", [])),
                "refused": True}
    applied, stale = [], []          # stale: (decision, why)
    by_page = collections.defaultdict(list)
    for d in dec.get("decisions", []):
        i = coerce_page(d)
        if i is None:
            stale.append((d, f"its page {d.get('page')!r} is not a number"))
            continue
        by_page[i].append(d)
    out = out or epub_path
    if out == epub_path and not os.path.exists(epub_path + ".preapply"):
        # The first original only. A second round applied in place used to
        # overwrite this with the first round's result, and the book as it
        # was built could no longer be had.
        shutil.copy2(epub_path, epub_path + ".preapply")
    def corrected(i, body):
        """A page with its decisions applied, for binding.rewrite_pages."""
        if i not in by_page:
            return None
        rows = by_page.pop(i)
        with tokens_as(dec.get("tokens")):
            body, outcomes = apply_page_decisions(body, rows)
        for d, why in zip(rows, outcomes):
            if why is None:
                applied.append(d)
            else:
                stale.append((d, why))
        return body

    # Each entry goes back under its own record (its date, its mode, its
    # compression), so that what the corrections changed can be read off
    # the files; and a page bound into a chapter is taken out, corrected
    # and bound in again.
    binding.rewrite_pages(epub_path, out, corrected)
    for i in sorted(by_page):
        stale += [(d, f"page {i} is not a page of this book")
                  for d in by_page[i]]
    for d, why in stale:
        print(f"  stale: p{d.get('page')} {d.get('old')!r} -> "
              f"{d.get('new')!r} ({d.get('source', '?')}) — {why}; "
              f"kept as shipped")
    retired = retired_rows(dec)
    print(f"applied {len(applied)} correction(s)"
          + (f", {len(stale)} stale" if stale else "")
          + (f", {len(retired)} retired and left alone" if retired else "")
          + f": {out}")
    return {"applied": applied, "stale": [d for d, _ in stale]}


# ── rounds, and rows taken out of use ────────────────────────────────────────

def retired_rows(dec) -> list:
    """The rows a decisions file holds but must not apply."""
    rows = dec.get("retired") if isinstance(dec, dict) else None
    return [d for d in rows if isinstance(d, dict)] \
        if isinstance(rows, list) else []


def _site(d) -> tuple:
    """Which word of which page a row is addressed to."""
    where = (tuple(d["before"]), tuple(d["after"])) \
        if "before" in d and "after" in d else _occurrence(d)
    return (coerce_page(d), d.get("kind", "swap"), d.get("old"), where)


def _same_correction(a, b) -> bool:
    return _site(a) == _site(b) and a.get("new") == b.get("new")


def _earlier_rounds(path: str) -> list:
    """(number, file) for each earlier round kept beside `path`."""
    stem = path[:-len(".json")] if path.endswith(".json") else path
    folder = os.path.dirname(path) or "."
    named = re.compile(re.escape(os.path.basename(stem)) + r"\.r(\d+)\.json")
    found = []
    for name in os.listdir(folder):
        m = named.fullmatch(name)
        if m:
            found.append((int(m.group(1)), os.path.join(folder, name)))
    return sorted(found)


def decision_rounds(path: str) -> list:
    """Every round of decisions kept for a book, oldest first, ending with
    `path`, the round in hand. Empty when `path` is not there."""
    if not path or not os.path.exists(path):
        return []
    return [f for _, f in _earlier_rounds(path)] + [path]


def _load_decisions(path: str) -> dict:
    with open(path) as fh:
        dec = json.load(fh)
    if not isinstance(dec, dict) or not isinstance(dec.get("decisions"), list):
        raise ValueError(f"{path} is not a review sheet's exported decisions")
    return dec


def _same_sheet(a: dict, b: dict) -> bool:
    """Whether two decisions files were exported from one sheet. The sheet
    says which record it was made from; a file from before it did is
    judged by its rows, since a second round is about other words: the
    first round's are no longer in dispute."""
    if a.get("sheet") and b.get("sheet"):
        return a["sheet"] == b["sheet"]
    mine = {_site(d) for d in a["decisions"]}
    return any(_site(d) in mine for d in b["decisions"])


def file_decisions(src: str, target: str) -> dict:
    """Put an exported decisions file where a book's rebuilds will find it.

    Importing a second round's file used to replace the first's, and the
    next rebuild came out without the first round's corrections: the cache
    holds the reading as it was made, so every round has to be applied
    again each time. A file from another sheet now becomes a new round and
    the one in hand is kept as an earlier one. The same sheet exported
    again replaces the round in hand, as it should: that is the reviewer
    changing their mind.

    Rows that any round has retired are taken out of the incoming file.
    """
    incoming = _load_decisions(src)
    rounds = decision_rounds(target)
    retired = []
    for path in rounds:
        try:
            for d in retired_rows(_load_decisions(path)):
                if not any(_same_correction(d, r) for r in retired):
                    retired.append(d)
        except (OSError, ValueError):
            pass
    replaced = False
    if rounds:
        try:
            replaced = _same_sheet(_load_decisions(target), incoming)
        except (OSError, ValueError):
            replaced = True         # nothing there worth keeping as a round
        if not replaced:
            stem = target[:-len(".json")] if target.endswith(".json") \
                else target
            last = max((n for n, _ in _earlier_rounds(target)), default=0)
            os.replace(target, f"{stem}.r{last + 1}.json")
    dropped = [d for d in incoming["decisions"]
               if any(_same_correction(d, r) for r in retired)]
    if dropped:
        incoming["decisions"] = [d for d in incoming["decisions"]
                                 if not any(d is x for x in dropped)]
    carried = retired_rows(incoming)
    carried += [d for d in retired
                if not any(_same_correction(d, r) for r in carried)]
    if carried:
        incoming["retired"] = carried
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    with open(target + ".tmp", "w") as fh:
        json.dump(incoming, fh, indent=1)
    os.replace(target + ".tmp", target)
    return {"rows": len(incoming["decisions"]),
            "round": len(decision_rounds(target)),
            "replaced": replaced, "dropped_retired": len(dropped),
            "retired": len(carried)}


def retire_decisions(path: str, numbers: list) -> list:
    """Take rows out of use by their number in list_decisions' listing.
    They stay in the file, under `retired`, so that the next round cannot
    bring them back unnoticed. Returns the rows retired."""
    dec = _load_decisions(path)
    rows = dec["decisions"]
    wanted = sorted(set(numbers))
    if any(not 1 <= n <= len(rows) for n in wanted):
        raise ValueError(f"{path} has rows 1 to {len(rows)}; asked for "
                         f"{', '.join(str(n) for n in wanted)}")
    gone = [rows[n - 1] for n in wanted]
    dec["decisions"] = [d for k, d in enumerate(rows, 1) if k not in wanted]
    dec["retired"] = retired_rows(dec) + gone
    with open(path + ".tmp", "w") as fh:
        json.dump(dec, fh, indent=1)
    os.replace(path + ".tmp", path)
    return gone


def list_decisions(path: str) -> None:
    """Print a decisions file a row to a line, numbered for retire."""
    def line(d):
        what = (f"delete {len(str(d.get('old', '')).split())} word(s) "
                f"beginning {str(d.get('old', ''))[:40]!r}"
                if d.get("kind") == "delete_run"
                else f"{d.get('old')!r} -> {d.get('new')!r}")
        return (f"p{d.get('page')} {what}  ({d.get('source', '?')}"
                + (", addressed" if "before" in d else "") + ")")
    dec = _load_decisions(path)
    rounds = len(decision_rounds(path))
    print(f"{path}: {len(dec['decisions'])} decision(s)"
          + (f", the last of {rounds} rounds" if rounds > 1 else "")
          + (f", recorded against {dec['page_files']} page file(s)"
             if dec.get("page_files") is not None else ""))
    for k, d in enumerate(dec["decisions"], 1):
        print(f"  {k:>3}  {line(d)}")
    for d in retired_rows(dec):
        print(f"  ret  {line(d)}")


def main(report_path, out=None, dpi=200, crops=True) -> int:
    report = json.load(open(report_path))
    report.setdefault("name", str(report_path))
    report["_base_dir"] = os.path.dirname(os.path.abspath(report_path))
    uris = {}
    if crops and report.get("pdf"):
        uris = crop_data_uris(
            report, dpi=dpi,
            base_dir=os.path.dirname(os.path.abspath(report_path)))
        located = len(uris)
        print(f"ink located for {located} of {len(report['disputes'])} "
              f"disputes")
    out = out or re.sub(r"(\.json)?$", "", str(report_path), count=1) + ".html"
    with tokens_as(report.get("tokens")):
        html_text = render_sheet(report, uris)
    with open(out, "w") as f:
        f.write(html_text)
    tiers = build_tiers(report["disputes"])
    print(f"review sheet: {out}")
    print("  " + " · ".join(f"{k} {len(v)}" for k, v in tiers.items() if v))
    return 0
