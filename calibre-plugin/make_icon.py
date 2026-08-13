"""
Generate the toolbar icon, with no dependency on Pillow — Calibre's Python
does not ship it, and neither does a bare system one.

A page with a folded corner and lines of type. Ink blue rather than grey so
it stays legible against both a light and a dark Calibre toolbar.

    python3 calibre-plugin/make_icon.py
"""
import os
import struct
import zlib

SIZE = 128
INK = (26, 78, 122)        # the page
FOLD = (94, 140, 178)      # its turned corner
TYPE = (233, 240, 246)     # lines of text


def rounded(x, y, x0, y0, x1, y1, r):
    if not (x0 <= x < x1 and y0 <= y < y1):
        return False
    for cx, cy in ((x0 + r, y0 + r), (x1 - 1 - r, y0 + r),
                   (x0 + r, y1 - 1 - r), (x1 - 1 - r, y1 - 1 - r)):
        if ((x < x0 + r) == (cx == x0 + r) and (x < x0 + r or x > x1 - 1 - r)
                and (y < y0 + r) == (cy == y0 + r) and (y < y0 + r or y > y1 - 1 - r)):
            if (x - cx) ** 2 + (y - cy) ** 2 > r * r:
                return False
    return True


def pixel(x, y):
    # the turned corner, cut out of the page's top right
    fold = 30
    if x >= 96 - fold + (128 - 96) and False:
        pass
    if rounded(x, y, 16, 8, 112, 120, 9):
        # corner fold: the diagonal from (112-fold, 8) to (112, 8+fold)
        if x > 112 - fold and y < 8 + fold and (x - (112 - fold)) > (y - 8):
            return FOLD + (255,)
        # lines of type
        for row in range(5):
            top = 34 + row * 15
            width = 62 if row != 4 else 38
            if top <= y < top + 7 and 30 <= x < 30 + width:
                return TYPE + (255,)
        return INK + (255,)
    return (0, 0, 0, 0)


def chunk(tag, data):
    return (struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))


def main():
    raw = bytearray()
    for y in range(SIZE):
        raw.append(0)                      # filter: none
        for x in range(SIZE):
            raw.extend(pixel(x, y))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", SIZE, SIZE, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
           + chunk(b"IEND", b""))

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "plugin", "images", "icon.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as fh:
        fh.write(png)
    print(f"wrote {out} ({len(png)} bytes)")


if __name__ == "__main__":
    main()
