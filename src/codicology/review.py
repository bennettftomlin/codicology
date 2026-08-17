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
rung} per correction — source "human" for typed readings, "ladder" for
accepted verdicts. Occurrence ordinals are counted among a page's disputes
sharing the same shipped token, in reading order; the apply step targets
the nth word-boundary match accordingly.
"""
import base64
import collections
import hashlib
import html
import io
import json
import re
import subprocess
import tempfile

from .adjudicate import _is_word, fold_word

# battery v2, 2,591 truth-scored disputes — shown on the rung chips so the
# reviewer knows how much each verdict is worth
RUNG_ACCURACY = {"lexicon": "97.8%", "vision": "95.0%", "dictionary": "88.4%"}
RUNG_ORDER = {"lexicon": 0, "vision": 1, "dictionary": 2}

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
    """Ordinal of each dispute among its page's disputes sharing a shipped
    token, in reading order — the apply step's word-boundary target index."""
    seen = collections.Counter()
    out = []
    for d in disputes:
        key = (d["page"], fold_word(d.get("shipped") or d["surya"]))
        out.append(seen[key])
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
    raw = json.dumps([len(report.get("disputes", [])),
                      report.get("pdf", ""), report.get("epub", "")],
                     sort_keys=True)
    return hashlib.sha1(raw.encode()).hexdigest()[:12]


def _locate(words, targets) -> list:
    """Find a dispute's word among a page's tesseract reads. Exact folded
    match first; then a target inside a larger token (ff inside 242ff);
    then a lone close variant — dispute sites are where tesseract reads
    unstably, so a fresh render often yields a third form. Every fallback
    demands uniqueness: a wrong crop misleads the reviewer in a way the
    honest "no crop" never does."""
    import difflib

    targets = [t for t in targets if t]
    hits = [w for w in words if w[4] in targets]
    if hits:
        return hits
    for t in targets:
        if len(t) >= 2:
            inside = [w for w in words if t in w[4]]
            if len(inside) == 1:
                return inside
    for t in targets:
        close = [w for w in words
                 if difflib.SequenceMatcher(None, t, w[4]).ratio() >= 0.8]
        if len(close) == 1:
            return close
    return []


def crop_data_uris(report, dpi=200, ctx_px=240) -> dict:
    """Crop each dispute's ink. Reports written since geometry landed carry
    a normalized bbox recorded at adjudication time — exact by construction.
    Older reports fall back to relocating the word through a fresh
    tesseract read, which fails precisely on the unstable sites disputes
    live on. Rows without ink render text-only — a missing crop must never
    block the sheet."""
    import pypdfium2 as pdfium

    by_page = collections.defaultdict(list)
    occs = assign_occurrences(report["disputes"])
    for idx, d in enumerate(report["disputes"]):
        by_page[d["page"]].append((idx, d, occs[idx]))
    doc = pdfium.PdfDocument(report["pdf"])
    uris = {}
    for pno, rows in sorted(by_page.items()):
        if pno >= len(doc):
            continue
        img = doc[pno].render(scale=dpi / 72).to_pil()
        words = None
        for idx, d, occ in rows:
            rec = d.get("box")
            if rec:
                x0, y0 = int(rec[0] * img.width), int(rec[1] * img.height)
                x1, y1 = int(rec[2] * img.width), int(rec[3] * img.height)
            else:
                if words is None:
                    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
                        img.save(tf.name)
                        tsv = subprocess.run(
                            ["tesseract", tf.name, "stdout", "tsv"],
                            capture_output=True, text=True).stdout
                    words = [(int(f[6]), int(f[7]), int(f[8]), int(f[9]),
                              fold_word(f[11]))
                             for f in (l.split("\t")
                                       for l in tsv.splitlines()[1:])
                             if len(f) == 12 and f[11].strip()]
                hits = _locate(words, [fold_word(d["tesseract"]),
                                       fold_word(d["surya"])])
                if not hits:
                    continue
                x, y, w, h, _ = hits[min(occ, len(hits) - 1)]
                x0, y0, x1, y1 = x, y, x + w, y + h
            box = (max(0, x0 - ctx_px), max(0, y0 - 6),
                   min(img.width, x1 + ctx_px), min(img.height, y1 + 6))
            buf = io.BytesIO()
            img.crop(box).save(buf, format="PNG")
            uris[idx] = ("data:image/png;base64,"
                         + base64.b64encode(buf.getvalue()).decode())
    return uris


def _row_html(r, crops) -> str:
    shipped = r.get("shipped") or r["surya"]
    winner = r.get("winner") or ""
    rung = r.get("rung") or "abstain"
    acc = RUNG_ACCURACY.get(rung)
    chip = f"{rung} {acc}" if acc else rung
    crop = crops.get(r["idx"])
    ink = (f'<img class="ink" alt="ink" src="{crop}">' if crop
           else '<span class="noink">no crop — ink not located</span>')
    proposal = (f'<button class="apply" data-act="apply">apply '
                f'<b>{html.escape(winner)}</b></button>' if winner
                and winner != shipped else "")
    return (
        f'<div class="row" tabindex="0" data-idx="{r["idx"]}" '
        f'data-page="{r["page"]}" data-occ="{r["occurrence"]}" '
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
document.querySelectorAll('.row').forEach(row => {
  row.addEventListener('click', e => {
    const act = e.target.closest('[data-act]');
    if (act) {
      state[row.dataset.idx] = {choice: act.dataset.act === 'apply'
        ? 'apply' : 'keep'};
      save(); paint(row);
    }
  });
  const own = row.querySelector('.own');
  own.addEventListener('input', () => {
    state[row.dataset.idx] = own.value === row.dataset.old
      ? {choice: 'keep'} : {choice: 'human', text: own.value};
    save(); paint(row);
  });
  paint(row);
});
function save() { localStorage.setItem(KEY, JSON.stringify(state)); }
document.getElementById('export').addEventListener('click', () => {
  const decisions = [];
  document.querySelectorAll('.row').forEach(row => {
    const s = state[row.dataset.idx];
    if (!s || s.choice === 'keep') return;
    decisions.push({
      page: +row.dataset.page, occurrence: +row.dataset.occ,
      old: row.dataset.old,
      new: s.choice === 'human' ? s.text : row.dataset.new,
      source: s.choice === 'human' ? 'human' : 'ladder',
      rung: s.choice === 'human' ? 'human' : row.dataset.rung});
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
.ink { max-height: 46px; max-width: 30rem; border: 1px solid #ddd; }
.noink { color: #999; font-style: italic; min-width: 12rem; }
.w { font-family: Georgia, serif; font-size: 1.1em; }
.alt, .loc { color: #777; font-size: .85em; }
.loc { margin-right: .5em; }
.chip { font-family: ui-monospace, monospace; font-size: .78em;
        background: #e8e4d6; border-radius: 3px; padding: 0 .35em; }
button { font: inherit; font-size: .85em; cursor: pointer; }
input.own { font-family: Georgia, serif; width: 11rem; margin-left: .5em; }
.state { color: #7a6a3a; margin-left: .5em; font-size: .85em; }
details { margin: .3rem 0; } summary { cursor: pointer; color: #555; }
#bar { position: sticky; top: 0; background: #fafaf7; padding: .5rem 0;
       border-bottom: 2px solid #d8d4c6; display: flex; gap: 1rem;
       align-items: center; }
@media (prefers-color-scheme: dark) {
  body, #bar { background: #191919; color: #ddd; }
  .row { border-color: #333; } .row.staged { background: #2c2717; }
  .chip { background: #333; } .ink { border-color: #444; filter: invert(1)
  hue-rotate(180deg); } }
"""


def render_sheet(report, crops=None) -> str:
    crops = crops or {}
    tiers = build_tiers(report["disputes"])
    name = re.sub(r"\.json$", "", str(report.get("name")
                  or report.get("epub") or "book").rsplit("/", 1)[-1])
    meta = json.dumps({"name": name, "pdf": report.get("pdf", ""),
                       "epub": report.get("epub", "")})
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
        "reading outranks every rung.</p>",
    ]
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


def _apply_to_xhtml(xhtml, old, new, occurrence) -> "str | None":
    """Replace the nth word-boundary occurrence of old in text nodes only.
    Tags are never touched — the segments are split apart and only text
    between them is searched. Returns None when the site is not there:
    a decision that no longer matches its book is stale, and stale
    decisions are surfaced, never guessed at."""
    segs = re.split(r"(<[^>]+>)", xhtml)
    pat = re.compile(r"(?<![\w])" + re.escape(old) + r"(?![\w])")
    seen = 0
    for i, seg in enumerate(segs):
        if seg.startswith("<"):
            continue
        for m in pat.finditer(seg):
            if seen == occurrence:
                segs[i] = seg[:m.start()] + new + seg[m.end():]
                return "".join(segs)
            seen += 1
    return None


def apply_decisions(epub_path, decisions_path, out=None) -> dict:
    """Feed the reviewer's exported decisions back into the EPUB.

    The adjudicator never rewrites; this is the human act it reports
    toward. Corrections land in text nodes at their recorded occurrence,
    the original is kept beside the book as .preapply, and every decision
    is accounted for: applied or stale, never silent."""
    import shutil
    import zipfile

    dec = json.load(open(decisions_path))
    by_page = collections.defaultdict(list)
    for d in dec.get("decisions", []):
        by_page[int(d["page"])].append(d)
    out = out or epub_path
    if out == epub_path:
        shutil.copy2(epub_path, epub_path + ".preapply")
    zin = zipfile.ZipFile(epub_path)
    applied, stale = [], []
    changed = {}
    for name in zin.namelist():
        m = re.match(r".*page_(\d{4})\.xhtml$", name)
        if not m or int(m.group(1)) not in by_page:
            continue
        xhtml = zin.read(name).decode("utf-8")
        for d in sorted(by_page[int(m.group(1))],
                        key=lambda d: -d.get("occurrence", 0)):
            got = _apply_to_xhtml(xhtml, d["old"], d["new"],
                                  d.get("occurrence", 0))
            if got is None:
                stale.append(d)
            else:
                xhtml = got
                applied.append(d)
        changed[name] = xhtml.encode("utf-8")
    names = zin.namelist()
    with zipfile.ZipFile(out + ".tmp", "w") as zout:
        for name in names:
            data = changed.get(name, zin.read(name))
            # the EPUB contract: mimetype first and stored, not deflated
            zout.writestr(name, data,
                          zipfile.ZIP_STORED if name == "mimetype"
                          else zipfile.ZIP_DEFLATED)
    zin.close()
    import os
    os.replace(out + ".tmp", out)
    for d in stale:
        print(f"  stale: p{d['page']} {d['old']!r} -> {d['new']!r} "
              f"({d.get('source', '?')}) — site not found, kept as shipped")
    print(f"applied {len(applied)} correction(s)"
          + (f", {len(stale)} stale" if stale else "")
          + f": {out}")
    return {"applied": applied, "stale": stale}


def main(report_path, out=None, dpi=200, crops=True) -> int:
    report = json.load(open(report_path))
    report.setdefault("name", str(report_path))
    uris = {}
    if crops and report.get("pdf"):
        uris = crop_data_uris(report, dpi=dpi)
        located = len(uris)
        print(f"ink located for {located} of {len(report['disputes'])} "
              f"disputes")
    out = out or re.sub(r"(\.json)?$", "", str(report_path), count=1) + ".html"
    html_text = render_sheet(report, uris)
    with open(out, "w") as f:
        f.write(html_text)
    tiers = build_tiers(report["disputes"])
    print(f"review sheet: {out}")
    print("  " + " · ".join(f"{k} {len(v)}" for k, v in tiers.items() if v))
    return 0
