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


# How much of an absent page must already be somewhere in the book before its
# absence stops counting against us. High deliberately: this is the one
# exemption the hole check makes, and the whole value of the check is that it
# does not make excuses. A leaf the scanner shot twice comes back at 0.93 and
# above; two pages that merely discuss the same subject come nowhere near.
RESCAN_COVERAGE = 0.75


def shingles(words, k: int = 6) -> set:
    """A page as its overlapping word-runs, for recognising it again."""
    w = [x.lower() for x in words]
    return {tuple(w[j:j + k]) for j in range(max(0, len(w) - k + 1))}


def covered_elsewhere(page_words, kept: list) -> float:
    """How much of this page is already present on some page we kept."""
    mine = shingles(page_words)
    if not mine:
        return 0.0
    return max((len(mine & k) / len(mine) for k in kept), default=0.0)


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

    # What we hold, as sets of overlapping word-runs, so a page the book
    # shows twice can be recognised wherever we kept it.
    kept = [shingles(v) for v in ours.values() if len(v) >= 25]

    tot_o = tot_l = 0
    holes, rescanned = [], []
    for i in range(len(pdf)):
        lw = WORD.findall(pdf[i].get_textpage().get_text_bounded())
        ow = ours.get(i, [])
        tot_o += len(ow)
        tot_l += len(lw)
        # The check that caught three real losses: the source has words here
        # and we have NONE. Three refinements, all learned from false alarms:
        # a page carrying four real words ("BOOK SIX THE STRIKE") is not a
        # hole, so the bar is emptiness rather than sparseness; a scanner
        # hallucinating over a blank leaf produces text-shaped noise
        # ("e b ae i ogi ae t net yr"), so the layer must be saying something
        # a dictionary recognises before its absence counts against us; and a
        # leaf the SOURCE captured twice is not missing when we kept one of
        # them. TiHKAL's scanner shot the xxii/xxiii spread over again, and
        # dropping the second pair was right — but by index those pages are
        # simply gone, and this read them as two holes and halted a batch of
        # eighteen books over content that was never lost.
        if ow or len(lw) < 25:
            continue
        real = sum(1 for w in lw if len(w) > 2 and w.lower() in _WORDS)
        if real < max(8, len(lw) // 4):
            continue
        elsewhere = covered_elsewhere(lw, kept)
        if elsewhere >= RESCAN_COVERAGE:
            rescanned.append((i, elsewhere))
        else:
            holes.append((i, len(lw), " ".join(lw[:9])))

    print(f"source looks {kind}")
    print(f"words: ours {tot_o}  source layer {tot_l}  "
          f"ratio {tot_o / max(1, tot_l):.2f}")
    print(f"pages where the source has text and we have none: {len(holes)}")
    for i, n, head in holes[:12]:
        print(f"   p{i}: {n} layer words :: {head!r}")
    if len(holes) > 12:
        print(f"   … and {len(holes) - 12} more")
    # Reported, never hidden: this is the one exemption the hole check makes,
    # so it says out loud which pages took it and how sure it was.
    if rescanned:
        print(f"pages the source captured twice, kept once: {len(rescanned)}"
              + " -> " + ", ".join(f"p{i} ({s:.2f} of it is elsewhere)"
                                   for i, s in rescanned[:8]))
    missing = [i for i in range(len(pdf)) if i not in ours]
    print(f"pages absent from the EPUB entirely: {len(missing)}"
          + (f" -> {missing[:12]}" if missing else ""))
    bad = len(holes) > 0
    print("VERDICT:", "LOOK AT THIS" if bad else "no holes found")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
