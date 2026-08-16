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
    dictionary  a non-word loses TO A WORD; two non-words abstain
    vision      Apple Vision rereads the page, language correction off —
                an engine we operate and can calibrate, unlike the source's
                embedded layer, which is a different unknown engine per book
                and holds no vote (its witness role elsewhere is unchanged)
    abstain     recorded, never guessed

Calibration comes free from the born-digital shelf: books whose text was
reconciled against the publisher's own layer are ground truth for scoring
the engines we run.
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


def _is_word(w: str) -> bool:
    """Lexical, allowing inflection: the system word list is lemma-only —
    "prisoner" is in it and "prisoners" is not — and without morphology the
    dictionary rung would abstain on nearly every real dispute, since real
    prose is mostly inflected."""
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


def adjudicate_pair(a: str, b: str, lexicon: Counter) -> dict:
    """One dispute between two attested readings, through the ladder."""
    fa, fb = fold_word(a), fold_word(b)
    if fa == fb:
        return {"rung": "fold", "winner": a}
    la, lb = lexicon.get(fa, 0), lexicon.get(fb, 0)
    if la >= 3 and lb == 0:
        return {"rung": "lexicon", "winner": a, "count": la}
    if lb >= 3 and la == 0:
        return {"rung": "lexicon", "winner": b, "count": lb}
    da, db = _is_word(fa), _is_word(fb)
    if da and not db:
        return {"rung": "dictionary", "winner": a}
    if db and not da:
        return {"rung": "dictionary", "winner": b}
    return {"rung": "abstain", "winner": None}


def align_disputes(ours: list, theirs: list) -> list:
    """Word-level disagreements between two readings of one page, by
    sequence alignment on folded forms — replacements only. Insertions and
    deletions are layout noise (furniture one side read and the other
    stripped), not character disputes."""
    import difflib
    fo = [fold_word(w) for w in ours]
    ft = [fold_word(w) for w in theirs]
    sm = difflib.SequenceMatcher(None, fo, ft, autojunk=False)
    out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "replace" and (i2 - i1) == (j2 - j1):
            for k in range(i2 - i1):
                out.append((ours[i1 + k], theirs[j1 + k]))
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
    agreed, per_page = [], {}
    disputes = []
    n_done = 0
    for i in sorted(pages):
        if i >= len(doc):
            continue
        if limit and n_done >= limit:
            break
        n_done += 1
        png = f"{tmp}/p{i}.png"
        doc[i].render(scale=scale).to_pil().save(png)
        tess = read_tesseract(png)
        if tess is None:
            continue
        t_tok = tokens(tess)
        s_tok = surya_tokens[i]
        pairs = align_disputes(s_tok, t_tok)
        pair_set = {fold_word(a) for a, _ in pairs} | \
                   {fold_word(b) for _, b in pairs}
        agreed.append([fold_word(w) for w in s_tok
                       if fold_word(w) not in pair_set])
        per_page[i] = {"pairs": pairs, "png": png}

    lexicon = build_lexicon(agreed)
    rungs = Counter()
    for i, info in per_page.items():
        vis_tok = None
        for a, b in info["pairs"]:
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
            disputes.append({"page": i, "surya": a, "tesseract": b,
                             "rung": verdict["rung"],
                             "winner": verdict.get("winner"),
                             "shipped": a})

    print(f"pages examined: {n_done}")
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


def calibrate(pdf: str, limit: "int | None" = None,
              scale: float = 200 / 72) -> int:
    """Score the engines we run against a born-digital book's own text."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(pdf)
    tmp = tempfile.mkdtemp(prefix="calibrate_")
    stats = {"tesseract": Counter(), "vision": Counter()}
    n = 0
    for i in range(len(doc)):
        if limit and n >= limit:
            break
        truth = tokens(doc[i].get_textpage().get_text_bounded())
        if len(truth) < 100:
            continue
        n += 1
        png = f"{tmp}/p{i}.png"
        doc[i].render(scale=scale).to_pil().save(png)
        truth_f = Counter(fold_word(w) for w in truth)
        for name, reader in (("tesseract", read_tesseract),
                             ("vision", read_vision)):
            got = reader(png)
            if got is None:
                continue
            got_f = Counter(fold_word(w) for w in tokens(got))
            inter = sum((truth_f & got_f).values())
            stats[name]["truth"] += sum(truth_f.values())
            stats[name]["got"] += sum(got_f.values())
            stats[name]["hit"] += inter
    print(f"pages scored: {n}")
    for name, c in stats.items():
        if not c.get("got"):
            print(f"  {name:<10} (unavailable)")
            continue
        print(f"  {name:<10} recall {c['hit']/max(1,c['truth']):.4f}  "
              f"precision {c['hit']/max(1,c['got']):.4f}")
    return 0
