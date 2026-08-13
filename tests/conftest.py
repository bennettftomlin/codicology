"""Shared fixtures for the codicology pipeline tests.

Everything here is synthetic and deterministic. Nothing reads a real book
capture: a test that needs a forty-minute extraction is a test nobody runs.

The fixture keeps its historical name — vtb, for video_to_book, the
pipeline's name when it was a single script — because it appears in every
test in this suite and renaming it would smear real history across six
hundred assertion lines for nothing.
"""
import numpy as np
import cv2
import pytest


@pytest.fixture(scope="session")
def vtb():
    from codicology import pipeline
    return pipeline


def _text_block(img, x0, y0, w, lines=16, gap=58, height=26, shade=35):
    for k in range(lines):
        y = y0 + k * gap
        cv2.rectangle(img, (x0, y), (x0 + w, y + height), (shade, shade, shade), -1)
    return img


@pytest.fixture
def make_spread():
    """
    An open book on a dark desk: two text pages either side of a soft gutter.

    The gutter is a shadow rather than a drawn line, because a hard-edged one is
    read as the outer edge of a single page — detect_page then crops to the left
    half and the facing page is lost without a word of complaint. That was a real
    surprise while building this, and it is the reason the fixture is shaped the
    way it is rather than the simplest way.
    """
    def build(number=0, shift=0, blur=0, width=2000, height=1400):
        img = np.full((height, width, 3), 40, np.uint8)
        cv2.rectangle(img, (120 + shift, 90), (width - 120 + shift, height - 90),
                      (238, 236, 232), -1)
        shadow = np.zeros((height, width), np.float32)
        cx = width // 2 + shift
        for x in range(max(0, cx - 90), min(width, cx + 90)):
            shadow[90:height - 90, x] = 95 * np.exp(-((x - cx) / 38.0) ** 2)
        img = np.clip(img.astype(np.float32) - shadow[..., None], 0, 255).astype(np.uint8)
        for side in (0, 1):
            x0 = 190 + side * 830 + shift
            cv2.putText(img, str(number * 2 + side), (x0, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 3)
            _text_block(img, x0, 260, 620)
        return cv2.GaussianBlur(img, (blur | 1, blur | 1), 0) if blur else img
    return build


@pytest.fixture
def make_page():
    """A single upright page, optionally with a hand across it or little text."""
    def build(lines=16, hand=False, width=900, height=1300):
        img = np.full((height, width, 3), 236, np.uint8)
        _text_block(img, 90, 150, width - 220, lines=lines)
        if hand:
            cv2.ellipse(img, (width // 2, int(height * 0.7)), (260, 150), 20, 0, 360,
                        (150, 190, 225), -1)     # skin in BGR
        return img
    return build


@pytest.fixture
def write_images(tmp_path):
    """Write frames to disk as IMG_0000.jpg… and hand back the folder."""
    def build(frames, name="shots"):
        d = tmp_path / name
        d.mkdir(exist_ok=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(d / f"IMG_{i:04d}.jpg"), f)
        return str(d)
    return build


@pytest.fixture
def fake_img2pdf():
    """Stands in for img2pdf so _finish can be exercised without writing a PDF."""
    class Fake:
        def __init__(self):
            self.pages = []

        def convert(self, paths):
            self.pages = list(paths)
            return b"%PDF-1.4 fake"
    return Fake()


@pytest.fixture
def page_files(tmp_path):
    """N page images on disk, each carrying its own index as its only content."""
    def build(n, texts=None):
        paths = []
        for i in range(n):
            img = np.full((1300, 900, 3), 236, np.uint8)
            _text_block(img, 90, 150, 680, lines=16)
            p = tmp_path / f"page_{i:04d}.jpg"
            cv2.imwrite(str(p), img)
            paths.append(str(p))
        return paths
    return build
