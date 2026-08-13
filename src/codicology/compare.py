"""Compare a built EPUB's OCR text against the source PDF's embedded text
layer, page by page.

The embedded layer is whatever the file carries: a scanner's OCR on a library
scan, or the publisher's own typesetting on a born-digital PDF. The second is
authoritative and the comparison becomes an accuracy audit of our side; the
first is just another reading, and disagreements have to be adjudicated.

Usage: codicology compare book.epub source.pdf

The output separates what always differs harmlessly (the embedded layer
leaves line-break hyphens split; photos contribute noise tokens) from what
deserves a look: head-to-head word disagreements paired by edit distance,
and pages where our side is empty while the layer holds real text — the
latter being where pipeline improvements hide."""
import re
import sys
import unicodedata
import zipfile
import html as H
from collections import Counter

import pypdfium2 as pdfium

WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿŒœ][A-Za-zÀ-ÖØ-öø-ÿŒœ'-]*")
# Two systems can render the same printed mark differently — a publisher's
# layer sets a typographic apostrophe, the OCR reports the ASCII one — and
# comparing them raw turns every possessive in the book into a fake
# disagreement. Fold the variants together before either side is tokenized.
PUNCT_FOLD = str.maketrans({
    "\u2019": "'", "\u2018": "'", "\u02bc": "'", "\u00b4": "'",
    "\u2013": "-", "\u2014": "-", "\u2010": "-", "\u2011": "-",
    "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
})


def words(s):
    s = unicodedata.normalize("NFKC", s).translate(PUNCT_FOLD)
    return [w.lower() for w in WORD.findall(s)]


def epub_pages(path):
    """page index -> (body words, caption words) with markup stripped."""
    out = {}
    with zipfile.ZipFile(path) as z:
        for n in z.namelist():
            m = re.match(r".*page_(\d{4})\.xhtml$", n)
            if not m:
                continue
            t = z.read(n).decode("utf-8")
            t = re.sub(r"^.*?<body[^>]*>", "", t, flags=re.S)  # drop <title>
            caps = " ".join(re.findall(r"<figcaption[^>]*>(.*?)</figcaption>",
                                       t, flags=re.S))
            body = re.sub(r"<figure.*?</figure>", " ", t, flags=re.S)
            strip = lambda x: H.unescape(re.sub(r"<[^>]+>", " ", x))
            out[int(m.group(1))] = (words(strip(body)), words(strip(caps)))
    return out


def edit1or2(a, b):
    """Edit distance <= 2, cheap bail-outs first."""
    if abs(len(a) - len(b)) > 2:
        return False
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1,
                           prev[j - 1] + (ca != cb)))
        if min(cur) > 2:
            return False
        prev = cur
    return prev[-1] <= 2


def main(pdf_path, epub_path):
    pdf = pdfium.PdfDocument(pdf_path)
    ours = epub_pages(epub_path)

    tot_o = tot_g = 0
    joins = 0                  # layer-only fragment pairs we set as one word
    duels = []                 # (page, our word, layer word) edit-distance<=2
    ours_extra, layer_extra = Counter(), Counter()
    empty_ours = []            # pages where we have nothing but the layer does

    for i in range(len(pdf)):
        gw = words(pdf[i].get_textpage().get_text_bounded())
        ow_body, ow_caps = ours.get(i, ([], []))
        ow = ow_body + ow_caps
        tot_o += len(ow)
        tot_g += len(gw)
        co, cg = Counter(ow), Counter(gw)
        only_o, only_g = co - cg, cg - co
        if not ow and len(gw) >= 5:
            empty_ours.append((i, len(gw), " ".join(gw[:10])))

        # hyphenation repair: an ours-only word assembled from two
        # layer-only fragments ("ante"+"dating") is the layer splitting,
        # not a disagreement
        for w in list(only_o):
            hits = 0
            for cut in range(2, len(w) - 1):
                a, b = w[:cut], w[cut:]
                if only_g[a] and only_g[b] and (a, b) != (w, ""):
                    only_g[a] -= 1
                    only_g[b] -= 1
                    only_o[w] -= 1
                    joins += 1
                    hits = 1
                    break
            if hits and not only_o[w]:
                del only_o[w]
        only_g = +only_g
        only_o = +only_o

        # head-to-head: leftover tokens two edits apart are the same printed
        # word read differently by the two systems — worth adjudicating
        gl = sorted(only_g.elements())
        for w in sorted(only_o.elements()):
            hit = next((g for g in gl if len(g) > 3 and edit1or2(w, g)), None)
            if hit is not None:
                gl.remove(hit)
                duels.append((i, w, hit))
                only_o[w] -= 1
                only_g[hit] -= 1
        ours_extra.update(+only_o)
        layer_extra.update(Counter(gl) & (+only_g))

    print(f"pages: {len(pdf)}   words ours={tot_o} layer={tot_g}")
    print(f"hyphen/line splits repaired by ours: {joins}")
    print(f"\nhead-to-head disagreements (same printed word, two readings): "
          f"{len(duels)}")
    for i, o, g in duels[:40]:
        print(f"  p{i:03d}  ours={o!r:24} layer={g!r}")
    if len(duels) > 40:
        print(f"  … and {len(duels) - 40} more")
    print(f"\nours-only surplus (top): {ours_extra.most_common(15)}")
    print(f"layer-only surplus (top): {layer_extra.most_common(15)}")
    print(f"\npages where ours is empty but the layer has text: "
          f"{len(empty_ours)}")
    for i, n, head in empty_ours:
        print(f"  p{i:03d} ({n} layer words) {head!r}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
