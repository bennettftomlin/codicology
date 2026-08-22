"""Where the readers disagree, and what settles it.

This module never changes a book. It reads a built EPUB beside its source,
reads the pages again with the witnesses, and emits the dispute record: every
word the engines could not agree on, what each read, and which rung of the
ladder settled it — or that none could, which is also an answer. At the
measured rate (~0.6% of words before the ladder, ~0.2% after) a whole book's
record fits on a screen, which is the point: it is a research artifact for a
human, not an automation queue.

The ladder, in the order the design settled it:

    fold        hyphenation and diacritics are POLICY, not disagreement —
                the majority literally outvotes correct joins otherwise
    lexicon     the book's own vocabulary: a word every engine agreed on
                elsewhere, often enough, is a word of this book — which is
                how coinages, Wade-Giles, chemical names and dialect get
                adjudicated without an external dictionary voting for
                standardisation
    dictionary  a non-word loses TO A WORD; two non-words abstain. The
                scholarly apparatus (ff., op. cit., ibid.) is lexical and
                single letters other than a/I/O are not words — eagle
                measured 92 of 105 verdicts against the shipped text to be
                fluency standardizing the apparatus away
    vision      Apple Vision rereads the page, language correction off —
                an engine we operate and can calibrate, unlike the source's
                embedded layer, which is a different unknown engine per book
                and holds no vote (its witness role elsewhere is unchanged)
    abstain     recorded, never guessed

Each dispute row also records where the ink is: the witness token's bbox
from the same TSV read that produced the reading, plus a crop box that
follows a hyphen-joined word or a line-edge phrase across the wrap. The
review sheet crops from these instead of relocating words at a render the
engine may read differently — relocation was measured failing exactly on
the unstable sites disputes live on.

Calibration comes free from the born-digital shelf: books whose text was
reconciled against the publisher's own layer are ground truth for scoring
the engines we run. Downstream, the record feeds `codicology review` and,
through the reviewer's exported decisions, `codicology apply` and
`convert --apply-decisions`. The reporting stays here; the rewriting is a
human act.
"""
import json
import re
import subprocess
import tempfile
import unicodedata
from collections import Counter

# ── folding: policy differences are not disputes ─────────────────────────────

_TYPO = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"',
                       "–": "-", "—": "-", "ﬁ": "fi", "ﬂ": "fl"})


def fold_text(text: str) -> str:
    """Join line-break hyphenation; nothing else touches the words."""
    t = (text or "").translate(_TYPO)
    return re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)


def fold_word(w: str) -> str:
    """One word's comparison form: lowercase, hyphens out, diacritics off.

    Diacritics fold for COMPARISON only — Surya's reëntry against Vision's
    reentry is fidelity versus flatness, not a dispute, and the reading that
    ships is still whatever the pipeline shipped."""
    w = w.translate(_TYPO).lower().replace("-", "").replace("'", "")
    return "".join(c for c in unicodedata.normalize("NFD", w)
                   if unicodedata.category(c) != "Mn")


WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ][A-Za-zÀ-ÖØ-öø-ÿŒœ'’\-]*")


def tokens(text: str) -> list:
    return WORD.findall(fold_text(text))


# ── the readers ──────────────────────────────────────────────────────────────

def read_tesseract(png: str) -> "str | None":
    try:
        r = subprocess.run(["tesseract", png, "stdout", "--psm", "3"],
                           capture_output=True, text=True, timeout=180)
        return r.stdout
    except Exception:
        return None


def read_tesseract_tsv(png: str) -> "str | None":
    try:
        r = subprocess.run(["tesseract", png, "stdout", "--psm", "3", "tsv"],
                           capture_output=True, text=True, timeout=180)
        return r.stdout
    except Exception:
        return None


def _union(a: tuple, b: tuple) -> tuple:
    return (min(a[0], b[0]), min(a[1], b[1]),
            max(a[0] + a[2], b[0] + b[2]) - min(a[0], b[0]),
            max(a[1] + a[3], b[1] + b[3]) - min(a[1], b[1]))


def _tsv_tokens(tsv: str, width: int, height: int) -> tuple:
    """tesseract's TSV as a token stream with two parallel normalized boxes
    per token — recorded here, where the words are read, instead of
    relocated afterward at a render the engine may read differently.

    box is the token's own ink. cbox is the box a CROP should show: a
    hyphen-joined token unions all its fragments so both lines land in the
    crop, and a token at a line edge takes the neighboring line's tail or
    head along — its real context flows across the wrap, and a one-line
    strip ending at "crys-" asks the reviewer to guess."""
    lines, cur_key, cur = [], None, []
    for f in (l.split("\t") for l in tsv.splitlines()[1:]):
        if len(f) == 12 and f[11].strip():
            key = (f[1], f[2], f[3], f[4])
            if key != cur_key:
                if cur:
                    lines.append(cur)
                cur_key, cur = key, []
            cur.append((int(f[6]), int(f[7]), int(f[8]), int(f[9]),
                        f[11].translate(_TYPO)))
    if cur:
        lines.append(cur)
    flat = []
    for li, words in enumerate(lines):
        for wi, (x, y, w, h, raw) in enumerate(words):
            joins = (wi == len(words) - 1 and li + 1 < len(lines)
                     and len(raw) >= 2 and raw.endswith("-")
                     and raw[-2].isalnum()
                     and lines[li + 1][0][4][:1].isalnum())
            flat.append([raw, (x, y, w, h), joins,
                         li, wi == 0, wi == len(words) - 1])
    toks, boxes, cboxes = [], [], []
    i = 0
    while i < len(flat):
        raw, box, joins, li, first, last = flat[i]
        joined = False
        while joins and i + 1 < len(flat):
            nxt = flat[i + 1]
            raw = raw[:-1] + nxt[0]
            box = _union(box, nxt[1])
            joins = nxt[2]
            joined = True
            i += 1
        cbox = box
        if not joined:
            # context flows across the wrap: a line-initial token carries
            # the previous line's tail, a line-final one the next line's
            # head, so the crop shows the phrase and not a cliff edge.
            # Only a vertically NEAR neighbor qualifies — the next entry in
            # TSV order can be a distant block (a footnote after a
            # paragraph), and a crop spanning the page helps nobody.
            near = 3 * max(box[3], 1)
            if first and li > 0:
                prev = lines[li - 1][-1][:4]
                if abs(prev[1] - box[1]) <= near:
                    cbox = _union(cbox, prev)
            if last and li + 1 < len(lines):
                nxt_w = lines[li + 1][0][:4]
                if abs(nxt_w[1] - box[1]) <= near:
                    cbox = _union(cbox, nxt_w)
        for t in WORD.findall(raw):
            toks.append(t)
            boxes.append([box[0] / width, box[1] / height,
                          (box[0] + box[2]) / width,
                          (box[1] + box[3]) / height])
            cboxes.append([cbox[0] / width, cbox[1] / height,
                           (cbox[0] + cbox[2]) / width,
                           (cbox[1] + cbox[3]) / height])
        i += 1
    return toks, boxes, cboxes


def read_vision(png: str) -> "str | None":
    """Apple Vision, language correction OFF: the dictionary is exactly what
    must not vote here — it mangles the proper nouns disputes live on."""
    try:
        import Quartz
        import Vision
    except ImportError:
        return None
    url = Quartz.CFURLCreateWithFileSystemPath(
        None, png, Quartz.kCFURLPOSIXPathStyle, False)
    src = Quartz.CGImageSourceCreateWithURL(url, None)
    if src is None:
        return None
    img = Quartz.CGImageSourceCreateImageAtIndex(src, 0, None)
    handler = Vision.VNImageRequestHandler.alloc().initWithCGImage_options_(
        img, None)
    req = Vision.VNRecognizeTextRequest.alloc().init()
    req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    req.setUsesLanguageCorrection_(False)
    ok, _ = handler.performRequests_error_([req], None)
    if not ok:
        return None
    return "\n".join(str(o.topCandidates_(1)[0].string())
                     for o in (req.results() or []) if o.topCandidates_(1))


# ── the ladder ───────────────────────────────────────────────────────────────

try:
    with open("/usr/share/dict/words") as _fh:
        _DICT = {w.strip().lower() for w in _fh}
except OSError:
    _DICT = set()


# The scholarly apparatus is lexical. Eagle measured why: 92 of its 105
# verdicts against the shipped reading were the dictionary crowning 'f'
# over 'ff' (242ff.) and 'of' over 'op' (op. cit.) — fluency standardizing
# away the apparatus of a scholarly book. These tokens are words HERE.
APPARATUS = {
    "f", "ff", "op", "cit", "ibid", "cf", "pp", "seq", "vol", "vols",
    "ed", "eds", "ch", "chs", "et", "al", "fig", "figs", "fol", "fols",
    "no", "nos", "viz", "cp", "passim", "supra", "infra", "sic", "esp",
    "repr", "rev", "trans",
}


def _is_word(w: str) -> bool:
    """Lexical, allowing inflection: the system word list is lemma-only —
    "prisoner" is in it and "prisoners" is not — and without morphology the
    dictionary rung would abstain on nearly every real dispute, since real
    prose is mostly inflected.

    The system list also contains all 26 single letters, which let a lone
    'f' outvote the apparatus abbreviation 'ff'; only a, I, and O are words
    of running English."""
    if w in APPARATUS:
        return True
    if len(w) == 1:
        return w in {"a", "i", "o"}
    if w in _DICT:
        return True
    for strip, add in (("s", ""), ("es", ""), ("ed", ""), ("ed", "e"),
                       ("ing", ""), ("ing", "e"), ("ly", ""),
                       ("ies", "y"), ("est", ""), ("er", "")):
        if w.endswith(strip) and len(w) > len(strip) + 2 \
                and (w[:-len(strip)] + add) in _DICT:
            return True
    return False


def build_lexicon(agreed_pages: list, min_count: int = 3) -> Counter:
    """The book's own vocabulary: folded words from text every engine agreed
    on, kept once they recur. Hapax coinages never enter — a single
    occurrence protects nothing and a dispute on one is undecidable."""
    lex = Counter()
    for words in agreed_pages:
        lex.update(fold_word(w) for w in words)
    return Counter({w: n for w, n in lex.items() if n >= min_count})


_ROMAN = re.compile(
    r"^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")


def _is_roman(orig: str) -> bool:
    """A numeral as the book prints one: two or more letters, one case
    throughout (XXXII chapter heads, xxxii front-matter folios), valid
    Roman syntax. Judged on the original, not the fold — case is part of
    the signal — and strict syntax keeps real words out: CIVIL and MIDI
    fail the grammar even though every letter is a Roman digit."""
    t = orig.strip(".,;:")
    if len(t) < 2 or not (t.isupper() or t.islower()):
        return False
    return bool(_ROMAN.match(t.upper()))


def adjudicate_pair(a: str, b: str, lexicon: Counter,
                    attested_floor: int = 0) -> dict:
    """One dispute between two attested readings, through the ladder.

    attested_floor > 0 arms the dictionary gate the calibration battery
    demanded: a dictionary winner must also occur at least that often in the
    book's own agreed text. The battery measured 1-in-10 dictionary verdicts
    wrong, and the wrong ones were misreads that accidentally landed on an
    English word — accidents are words of the language, not words of the
    book."""
    fa, fb = fold_word(a), fold_word(b)
    if fa == fb:
        return {"rung": "fold", "winner": a}
    if _is_roman(a) or _is_roman(b):
        # a chapter numeral is apparatus: the book's common words and the
        # dictionary both outnumber every numeral, and eagle's lexicon
        # crowned 'a' over a contents-page XXXII on exactly that arithmetic.
        # Only ink evidence or a human settles a dispute touching one.
        return {"rung": "abstain", "winner": None, "note": "roman numeral"}
    la, lb = lexicon.get(fa, 0), lexicon.get(fb, 0)
    if la >= 3 and lb == 0:
        return {"rung": "lexicon", "winner": a, "count": la}
    if lb >= 3 and la == 0:
        return {"rung": "lexicon", "winner": b, "count": lb}
    da, db = _is_word(fa), _is_word(fb)
    if da and not db:
        if attested_floor and lexicon.get(fa, 0) < attested_floor:
            return {"rung": "abstain", "winner": None, "note": "ungated word"}
        return {"rung": "dictionary", "winner": a}
    if db and not da:
        if attested_floor and lexicon.get(fb, 0) < attested_floor:
            return {"rung": "abstain", "winner": None, "note": "ungated word"}
        return {"rung": "dictionary", "winner": b}
    return {"rung": "abstain", "winner": None}


def _stitch_page_turns(pages: dict, shipped_last: dict) -> int:
    """Join the witness's page-turn hyphen fragments the way the built book
    joined its own — and only then.

    The builder rejoins words split across page turns, so the shipped page
    ends with the whole word while the witness, whose world ends at the
    page edge, legitimately reads only the fragment. Left alone, every
    such join becomes a false dispute — and the ladder crowns the
    fragment, since 'in-' folds to a word the book attests everywhere.
    The stitch mirrors the builder exactly: the fragment joins the next
    page's first token only when the result folds to what the shipped
    page actually ends with. Where the builder refused (self- control),
    the condition fails symmetrically and both pages stay silent.

    Empty witness pages are skipped, not stitched across blindly — the
    builder joins across dropped blanks, and the fold-mirror condition is
    the gate that keeps a long gap honest.

    pages: page index -> [tokens, boxes, cboxes], mutated in place."""
    order = [i for i in sorted(pages) if pages[i][0]]
    stitched = 0
    for a, b in zip(order, order[1:]):
        ta, tb = pages[a][0], pages[b][0]
        frag = ta[-1]
        if not (len(frag) >= 2 and frag.endswith("-")
                and frag[-2].isalnum()):
            continue
        # the witness reads furniture too: the next page opens with its
        # running head before the continuation, so the completing token is
        # sought among the first few. No coincidence can slip in — the
        # join must fold to the shipped page's exact last word.
        for k, cand in enumerate(tb[:8]):
            joined = frag[:-1] + cand
            if fold_word(joined) == shipped_last.get(a):
                ta[-1] = joined
                del tb[k]
                del pages[b][1][k]
                del pages[b][2][k]
                stitched += 1
                break
    return stitched


# A replace block this long is a REGION the two readers disagree about, not
# a list of independent word disputes. Measured: a book both engines take in
# the same order never exceeds two (Chinese Communism, 21 blocks), while a
# book where tesseract's page segmentation walks the blocks in a different
# order produces runs of three to six — and those runs pair words that were
# never looking at the same ink ("nothing" against "a", "to" against
# "geere").
SEAM_RUN = 3
# Inside such a region, a pair can still be a real disagreement about real
# ink — a torn line garbles several words in a row. What separates the two
# is whether the readings could be the same ink at all: two readings of one
# word rhyme AND run to roughly the same length, because the letters are
# still there to be miscounted. A seam pairs whatever happened to line up,
# and pairs a whole word against a fragment of another one.
SEAM_KEEP_RATIO = 0.5
SEAM_KEEP_LEN = 2
# The other half of the same failure, in blocks too short for the run rule
# to see. A seam can pair one word against another the page really does
# contain — Citadel offered "well-cultured" against "although", "concluding"
# against "its" — and the giveaway is that surya read that very word
# somewhere else on this page. It was displaced, not misread. Only pairs
# that could not be one word read twice are judged this way, so a genuine
# truncation ("be" against "been") survives on a page that says "been"
# elsewhere.


def align_disputes(ours: list, theirs: list, stats: "Counter | None" = None
                   ) -> list:
    """Word-level disagreements between two readings of one page, by
    sequence alignment on folded forms — replacements only. Insertions and
    deletions are layout noise (furniture one side read and the other
    stripped), not character disputes."""
    import difflib
    fo = [fold_word(w) for w in ours]
    ft = [fold_word(w) for w in theirs]
    ours_bag = Counter(fo)
    sm = difflib.SequenceMatcher(None, fo, ft, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag != "replace" or (i2 - i1) != (j2 - j1):
            continue
        run = i2 - i1
        for k in range(run):
            a, b = ours[i1 + k], theirs[j1 + k]
            fa, fb = fo[i1 + k], ft[j1 + k]
            could_be_one_word = (
                abs(len(fa) - len(fb)) <= SEAM_KEEP_LEN
                and difflib.SequenceMatcher(None, fa, fb).ratio()
                >= SEAM_KEEP_RATIO)
            if not could_be_one_word and (run >= SEAM_RUN
                                          or ours_bag[fb]):
                if stats is not None:
                    stats["seam"] += 1
                continue
            out.append((a, b, j1 + k))
    return out


# ── the verb ─────────────────────────────────────────────────────────────────

def epub_page_texts(epub_path: str) -> dict:
    import zipfile
    z = zipfile.ZipFile(epub_path)
    out = {}
    for n in z.namelist():
        m = re.match(r".*page_(\d{4})\.xhtml$", n)
        if not m:
            continue
        h = z.read(n).decode("utf-8", "replace")
        b = re.sub(r"</body>.*$", "",
                   re.sub(r"^.*?<body[^>]*>", "", h, flags=re.S), flags=re.S)
        out[int(m.group(1))] = re.sub(r"<[^>]+>", " ", b)
    return out


def main(epub: str, pdf: str, report: "str | None" = None,
         scale: float = 200 / 72, limit: "int | None" = None) -> int:
    import pypdfium2 as pdfium
    pages = epub_page_texts(epub)
    doc = pdfium.PdfDocument(pdf)
    tmp = tempfile.mkdtemp(prefix="adjudicate_")
    have_vision = read_vision.__doc__ is not None and _vision_present()

    surya_tokens = {i: tokens(t) for i, t in pages.items()}
    align_stats = Counter()
    agreed, per_page = [], {}
    disputes = []
    n_done = 0
    witness = {}
    for i in sorted(pages):
        if i >= len(doc):
            continue
        if limit and n_done >= limit:
            break
        n_done += 1
        png = f"{tmp}/p{i}.png"
        img = doc[i].render(scale=scale).to_pil()
        img.save(png)
        tsv = read_tesseract_tsv(png)
        if tsv is None:
            continue
        witness[i] = list(_tsv_tokens(tsv, img.width, img.height)) + [png]
    shipped_last = {i: fold_word(surya_tokens[i][-1])
                    for i in witness if surya_tokens.get(i)}
    n_stitched = _stitch_page_turns(
        {i: w[:3] for i, w in witness.items()}, shipped_last)
    for i in sorted(witness):
        t_tok, t_box, t_cbox, png = witness[i]
        s_tok = surya_tokens[i]
        pairs = align_disputes(s_tok, t_tok, align_stats)
        pair_set = {fold_word(a) for a, _, _ in pairs} | \
                   {fold_word(b) for _, b, _ in pairs}
        agreed.append([fold_word(w) for w in s_tok
                       if fold_word(w) not in pair_set])
        per_page[i] = {"pairs": pairs, "png": png, "boxes": t_box,
                       "cboxes": t_cbox}

    lexicon = build_lexicon(agreed)
    rungs = Counter()
    for i, info in per_page.items():
        vis_tok = None
        for a, b, j in info["pairs"]:
            verdict = adjudicate_pair(a, b, lexicon)
            if verdict["rung"] == "fold":
                continue                    # policy, not a dispute
            if verdict["rung"] == "abstain" and have_vision:
                if vis_tok is None:
                    v = read_vision(info["png"])
                    vis_tok = {fold_word(w) for w in tokens(v or "")}
                fa, fb = fold_word(a), fold_word(b)
                if (fa in vis_tok) != (fb in vis_tok):
                    verdict = {"rung": "vision",
                               "winner": a if fa in vis_tok else b}
            rungs[verdict["rung"]] += 1
            boxes, cboxes = info["boxes"], info["cboxes"]
            row = {"page": i, "surya": a, "tesseract": b,
                   "rung": verdict["rung"],
                   "winner": verdict.get("winner"),
                   "shipped": a,
                   "box": boxes[j] if j < len(boxes) else None}
            if j < len(cboxes) and cboxes[j] != row["box"]:
                row["cbox"] = cboxes[j]
            disputes.append(row)

    print(f"pages examined: {n_done}")
    if align_stats["seam"]:
        print(f"reading order: {align_stats['seam']} word pair(s) discarded "
              f"where the readers walked the page in different orders — the "
              f"words were never looking at the same ink")
    if n_stitched:
        print(f"page-turn hyphens: {n_stitched} witness fragment(s) "
              f"stitched to match the built book's own joins")
    print(f"book lexicon: {len(lexicon)} recurring agreed words")
    print(f"disputes: {len(disputes)}"
          + (f"  ({', '.join(f'{k}:{v}' for k, v in rungs.most_common())})"
             if disputes else ""))
    for dd in disputes[:40]:
        w = dd["winner"]
        note = ("agrees with shipped" if w == dd["surya"] else
                "AGAINST shipped" if w else "unresolved")
        print(f"   p{dd['page']:>4} surya={dd['surya']!r:<20} "
              f"tess={dd['tesseract']!r:<20} {dd['rung']:<10} {note}")
    if len(disputes) > 40:
        print(f"   … and {len(disputes) - 40} more")
    if report:
        json.dump({"epub": epub, "pdf": pdf, "pages": n_done,
                   "page_files": len(pages),
                   "lexicon_size": len(lexicon),
                   "rungs": dict(rungs), "disputes": disputes},
                  open(report, "w"), indent=1)
        print(f"report written: {report}")
    return 0


def _vision_present() -> bool:
    try:
        import Vision  # noqa: F401
        return True
    except ImportError:
        return False


def calibrate(pdf: str, epub: "str | None" = None,
              limit: "int | None" = None, scale: float = 200 / 72,
              degrade: bool = False) -> dict:
    """Score the engines — and the ladder itself — against a born-digital
    book's own text.

    Engine recall/precision is the shallow layer. The number that decides
    whether the adjudicator may leave the house is RUNG ACCURACY: put real
    disputes (the shipped EPUB text against a witness reading) through the
    ladder where the truth is known, and count each rung's verdicts right,
    wrong, and the abstains truth could have settled. A rung that rules
    confidently wrong is a design fault, not a statistic.

    --degrade re-renders each page through synthetic damage (blur, noise,
    downscale) before reading it, because the adjudicator's real workload is
    degraded type and clean-render calibration cannot speak to it. The truth
    stays known — that is the entire point of synthesising the damage.
    """
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    tmp = tempfile.mkdtemp(prefix="calibrate_")
    engines = {"tesseract": Counter(), "vision": Counter()}
    cal_stats = Counter()
    shipped = epub_page_texts(epub) if epub else {}
    agreed, dispute_rows = [], []
    n = 0
    for i in range(len(doc)):
        if limit and n >= limit:
            break
        truth_txt = doc[i].get_textpage().get_text_bounded()
        truth = tokens(truth_txt)
        if len(truth) < 100:
            continue
        n += 1
        png = f"{tmp}/p{i}.png"
        img = doc[i].render(scale=scale).to_pil()
        if degrade:
            from PIL import ImageFilter
            img = img.convert("L").filter(ImageFilter.GaussianBlur(1.1))
            img = img.resize((img.width * 2 // 3, img.height * 2 // 3))
            img = img.resize((img.width * 3 // 2, img.height * 3 // 2))
        img.save(png)
        truth_f = Counter(fold_word(w) for w in truth)
        reads = {}
        vis_sets = {}
        for name, reader in (("tesseract", read_tesseract),
                             ("vision", read_vision)):
            got = reader(png)
            if got is None:
                continue
            reads[name] = got
            got_f = Counter(fold_word(w) for w in tokens(got))
            if name == "vision":
                vis_sets = set(got_f)
            inter = sum((truth_f & got_f).values())
            engines[name]["truth"] += sum(truth_f.values())
            engines[name]["got"] += sum(got_f.values())
            engines[name]["hit"] += inter
        # the ladder, on real disputes with the answer key in hand
        ours = tokens(shipped.get(i, "")) if shipped else truth
        wit = tokens(reads.get("tesseract", ""))
        if not wit:
            continue
        pairs = align_disputes(ours, wit, cal_stats)
        pair_set = {fold_word(a) for a, _, _ in pairs} | \
                   {fold_word(b) for _, b, _ in pairs}
        agreed.append([w for w in ours if fold_word(w) not in pair_set])
        for a, b, _ in pairs:
            fa, fb = fold_word(a), fold_word(b)
            dispute_rows.append({
                "page": i, "a": a, "b": b,
                "ta": truth_f.get(fa, 0) > 0, "tb": truth_f.get(fb, 0) > 0,
                "va": fa in vis_sets, "vb": fb in vis_sets})

    lexicon = build_lexicon(agreed)

    def score(rows, floor):
        table = {}
        for r in rows:
            v = adjudicate_pair(r["a"], r["b"], lexicon,
                                attested_floor=floor)
            rung = v["rung"]
            winner = v["winner"]
            if winner is None and rung == "abstain" and r["va"] != r["vb"]:
                rung = "vision"
                winner = r["a"] if r["va"] else r["b"]
            rs = table.setdefault(rung, Counter())
            if winner is None:
                rs["settleable" if (r["ta"] != r["tb"]) else "truly_open"] += 1
            else:
                right = r["ta"] if fold_word(winner) == fold_word(r["a"])                     else r["tb"]
                wrong = r["tb"] if fold_word(winner) == fold_word(r["a"])                     else r["ta"]
                if right and not wrong:
                    rs["right"] += 1
                elif wrong and not right:
                    rs["WRONG"] += 1
                else:
                    rs["unjudgeable"] += 1
        return table

    variants = {"ungated": score(dispute_rows, 0),
                "gated(1)": score(dispute_rows, 1),
                "gated(2)": score(dispute_rows, 2)}
    rung_score = variants["ungated"]

    print(f"pages scored: {n}" + ("  (degraded renders)" if degrade else ""))
    for name, c in engines.items():
        if not c.get("got"):
            print(f"  {name:<10} (unavailable)")
            continue
        print(f"  {name:<10} recall {c['hit']/max(1,c['truth']):.4f}  "
              f"precision {c['hit']/max(1,c['got']):.4f}")
    if cal_stats["seam"]:
        # the battery's blind spot, closed: seam-discarded pairs were
        # silently missing from the truth-scored rows, so a book whose
        # reading order confuses the gate looked better-scored than it
        # was (I5). The exclusion is still right — seam pairs are not
        # disputes — but it must be visible in the record it shapes.
        print(f"  seam-discarded: {cal_stats['seam']} pair(s) excluded "
              f"from scoring (readers walked the page in different orders)")
    if dispute_rows:
        print(f"  ladder, on {len(dispute_rows)} real disputes with truth known:")
        for name, table in variants.items():
            print(f"   [{name}]")
            for rung, rs in sorted(table.items()):
                print(f"    {rung:<12} " + "  ".join(f"{k}={v}" for k, v in
                                                     sorted(rs.items())))
        import os as _os
        rows_path = _os.environ.get("CODICOLOGY_ROWS")
        if rows_path:
            with open(rows_path, "a") as fh:
                for r in dispute_rows:
                    fh.write(json.dumps(r) + "\n")
            print(f"  rows appended: {rows_path}")
    return {"pages": n, "engines": {k: dict(v) for k, v in engines.items()},
            "rungs": {k: dict(v) for k, v in rung_score.items()}}
