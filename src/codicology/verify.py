"""Check a built EPUB against the source PDF's own text, and say plainly
whether anything went missing.

Two content losses in this project were invisible in every run log and
visible here: a page cached empty and shipped as a hole, and a page deleted
by a duplicate detector. Both showed up as "we have nothing where the source
has words". That check is cheap and belongs after every build.

What the comparison MEANS depends on the source, and the script says which:

  * born-digital — the layer is the publisher's typesetting, so any
    disagreement is our error and the word counts should nearly match.
  * a scan — the layer is another OCR, usually worse than ours; disagreements
    need a human eye and are reported but not judged.

Usage: codicology verify book.epub source.pdf
"""
import re
import sys
import zipfile

import pypdfium2 as pdfium

WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")

try:
    with open("/usr/share/dict/words") as _fh:
        _WORDS = {w.strip().lower() for w in _fh}
except OSError:                      # no dictionary: judge every layer real
    _WORDS = None


class _Everything:
    def __contains__(self, _):
        return True


if _WORDS is None:
    _WORDS = _Everything()


def main(epub_path, pdf_path):
    pdf = pdfium.PdfDocument(pdf_path)
    z = zipfile.ZipFile(epub_path)

    # is the source a scan? a scan keeps a picture of the whole page behind
    # its text, which is also what decides whether we may trust that text
    scanned = 0
    for i in range(min(len(pdf), 12)):
        pg = pdf[i]
        pw, ph = pg.get_size()
        for o in pg.get_objects():
            if isinstance(o, pdfium.PdfImage):
                try:
                    b = o.get_bounds()
                except Exception:
                    continue
                if ((b[2] - b[0]) * (b[3] - b[1])) / max(1e-6, pw * ph) >= 0.9:
                    scanned += 1
                    break
    kind = "scan (its layer is another OCR)" if scanned > 6 else "born-digital"

    ours = {}
    for n in z.namelist():
        m = re.match(r".*page_(\d{4})\.xhtml$", n)
        if not m:
            continue
        t = z.read(n).decode("utf-8", "replace")
        body = re.sub(r"</body>.*$", "",
                      re.sub(r"^.*?<body[^>]*>", "", t, flags=re.S), flags=re.S)
        ours[int(m.group(1))] = WORD.findall(re.sub(r"<[^>]+>", " ", body))

    tot_o = tot_l = 0
    holes = []
    for i in range(len(pdf)):
        lw = WORD.findall(pdf[i].get_textpage().get_text_bounded())
        ow = ours.get(i, [])
        tot_o += len(ow)
        tot_l += len(lw)
        # The check that caught three real losses: the source has words here
        # and we have NONE. Two refinements, both learned from false alarms:
        # a page carrying four real words ("BOOK SIX THE STRIKE") is not a
        # hole, so the bar is emptiness rather than sparseness; and a scanner
        # hallucinating over a blank leaf produces text-shaped noise
        # ("e b ae i ogi ae t net yr"), so the layer must be saying something
        # a dictionary recognises before its absence counts against us.
        if ow or len(lw) < 25:
            continue
        real = sum(1 for w in lw if len(w) > 2 and w.lower() in _WORDS)
        if real >= max(8, len(lw) // 4):
            holes.append((i, len(lw), " ".join(lw[:9])))

    print(f"source looks {kind}")
    print(f"words: ours {tot_o}  source layer {tot_l}  "
          f"ratio {tot_o / max(1, tot_l):.2f}")
    print(f"pages where the source has text and we have none: {len(holes)}")
    for i, n, head in holes[:12]:
        print(f"   p{i}: {n} layer words :: {head!r}")
    if len(holes) > 12:
        print(f"   … and {len(holes) - 12} more")
    missing = [i for i in range(len(pdf)) if i not in ours]
    print(f"pages absent from the EPUB entirely: {len(missing)}"
          + (f" -> {missing[:12]}" if missing else ""))
    bad = len(holes) > 0
    print("VERDICT:", "LOOK AT THIS" if bad else "no holes found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
