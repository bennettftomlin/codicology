"""The page rectifier and what surrounds it.

The model itself is a dependency, tested once as a smoke test when it is
on hand. Everything this module adds around it — sizing the sheet from the
grid, sampling along it, trimming the surround, refusing a sheet that lost
its words — is deterministic and tested without the model, by handing the
sampler grids whose right answer is known.
"""
import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
from codicology import rectify as R

needs_model = pytest.mark.skipif(not R.available(), reason="no rectifier on hand")


def _page(w=900, h=1200, seed=3):
    """Paper with rows of dark strokes, the way every geometry test here draws text."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), 236, np.uint8)
    for y in range(120, h - 120, 40):
        x = 100
        while x < w - 100:
            run = int(rng.integers(20, 70))
            img[y:y + 14, x:x + run] = 30
            x += run + int(rng.integers(12, 30))
    return img


def _identity_grid(w, h, rows=31, cols=45):
    xs = np.linspace(0, w - 1, cols, dtype=np.float32)
    ys = np.linspace(0, h - 1, rows, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx, gy], axis=-1)


def test_sheet_size_is_the_grids_own_extent():
    g = _identity_grid(900, 1200)
    assert R.sheet_size(g) == (900, 1200)
    # a grid over columns 300..599 of a photograph is a sheet 300 wide
    g2 = g.copy(); g2[..., 0] = np.linspace(300, 599, g.shape[1], dtype=np.float32)[None, :]
    assert R.sheet_size(g2) == (300, 1200)


def test_sampling_along_an_identity_grid_returns_the_photograph():
    img = _page()
    out = R.sample(img, _identity_grid(900, 1200))
    assert out.shape == img.shape
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 2.0


def test_sampling_undoes_a_known_perspective():
    """A page photographed off-axis is a page seen through a homography; a
    grid that is the page's rectangle pushed through the same homography
    must bring the page back, at the size the camera recorded it — which
    for a roughly overhead shot is the page's own."""
    page = _page()
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[150, 100], [1060, 120], [1040, 1300], [130, 1290]])
    H = cv2.getPerspectiveTransform(src, dst)
    photo = cv2.warpPerspective(page, H, (1300, 1400), borderValue=(40, 40, 40))
    g = _identity_grid(w, h)
    pts = cv2.perspectiveTransform(g.reshape(-1, 1, 2), H).reshape(g.shape)
    out = R.sample(photo, pts.astype(np.float32))
    ow, oh = out.shape[1], out.shape[0]
    assert abs(ow - w) < 0.08 * w and abs(oh - h) < 0.08 * h, (ow, oh)
    back = cv2.resize(out, (w, h))
    a = cv2.cvtColor(back, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    b = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
    assert np.corrcoef(a, b)[0, 1] > 0.9


def test_paper_crop_trims_dark_surround_but_never_paper_or_a_plate():
    page = _page()
    h, w = page.shape[:2]
    framed = page.copy()
    framed[:, :int(w * 0.05)] = 25          # a strip of desk on the left
    framed[:int(h * 0.03)] = 40             # and the cover's edge along the top
    out = R.paper_crop(framed)
    assert out.shape[1] <= w - int(w * 0.05) + 3 and out.shape[1] >= w - int(w * 0.05) - 3
    assert out.shape[0] <= h - int(h * 0.03) + 3 and out.shape[0] >= h - int(h * 0.03) - 3
    assert R.paper_crop(page).shape == page.shape, "clean paper is left alone"
    plate = np.full((h, w, 3), 45, np.uint8)  # a dark full-page plate is content
    assert R.paper_crop(plate).shape == plate.shape
    deep = page.copy(); deep[:, :int(w * 0.3)] = 25   # more than the cap wants: refuse
    assert R.paper_crop(deep).shape[1] >= w - int(w * R.PAPER_MAX_TRIM) - 3


def test_paper_crop_trims_through_a_sliver_to_the_spine_shadow():
    """The split leaves a bright sliver of the facing page outside the
    spine's shadow; a trim that stopped at the sliver kept the shadow's
    dark line on a fifth of one book's pages."""
    page = _page()
    h, w = page.shape[:2]
    cut = page.copy()
    cut[:, :int(w * 0.02)] = 225                    # the facing page's sliver
    cut[:, int(w * 0.02):int(w * 0.04)] = 35        # the spine shadow
    out = R.paper_crop(cut)
    assert abs(out.shape[1] - (w - int(w * 0.04))) <= 3, out.shape
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    assert np.median(g[:, :4]) > 150, "no dark line left at the edge"
    # A rule under a running head sits behind a real margin, and rows never
    # chain: the head above it stays.
    ruled = page.copy()
    ruled[int(h * 0.07):int(h * 0.07) + 4, :] = 30
    assert R.paper_crop(ruled).shape == page.shape
    # A wider gap that is DIM — the facing page's penumbra — is surround too.
    penumbra = page.copy()
    penumbra[:, :int(w * 0.05)] = 160                # 0.68 of this paper
    penumbra[:, int(w * 0.05):int(w * 0.07)] = 35    # then the shadow
    out = R.paper_crop(penumbra)
    assert abs(out.shape[1] - (w - int(w * 0.07))) <= 3, out.shape
    # ...but a vertical rule behind a margin at paper brightness is content.
    margined = page.copy()
    margined[:, int(w * 0.06):int(w * 0.06) + 3] = 30
    assert R.paper_crop(margined).shape == page.shape


def test_paper_crop_never_walks_through_text_to_reach_a_plate():
    """A plate covering the top 55% of the page, flush with the text
    column's edge, once read as surround by the column median, and the
    trim chained through the text beside it: a text column cut at the
    page's inner edge. Surround is dark along the whole column; a plate
    is not."""
    page = _page(w=900, h=1200)
    h, w = page.shape[:2]
    plated = page.copy()
    plated[:int(h * 0.55), int(w * 0.5):int(w * 0.95)] = 40    # the plate, nearly to the edge
    plated[:, int(w * 0.97):] = 60                              # a thin shadow at the edge itself
    out = R.paper_crop(plated)
    assert out.shape[1] >= int(w * 0.97) - 3, out.shape           # only the shadow goes


def _justified_page(w=1400, h=1900, n_lines=28):
    """Paper with full-measure lines of dark strokes: a justified block.

    Strokes are 7px — under the 9px black-hat kernel the edge reader
    uses, which cannot see a feature thicker than itself (real type at
    the probe scale is thinner still)."""
    img = np.full((h, w, 3), 236, np.uint8)
    rng = np.random.default_rng(11)
    for k in range(n_lines):
        y = 220 + k * 50
        x = 180
        while x < w - 180:
            run = int(rng.integers(30, 90))
            img[y:y + 7, x:min(x + run, w - 180)] = 30
            x += run + int(rng.integers(10, 24))
        img[y:y + 7, w - 200:w - 180] = 30       # every line reaches the measure
    return img


def _keystoned(page, top_inset=45):
    """The page seen with its top narrower than its bottom."""
    h, w = page.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[top_inset, 0], [w - 1 - top_inset, 0], [w - 1, h - 1], [0, h - 1]])
    return cv2.warpPerspective(page, cv2.getPerspectiveTransform(src, dst), (w, h),
                               borderValue=(236, 236, 236))


def _convergence(page):
    found = R.text_edges(page)
    assert found is not None
    (aL, _), (aR, _), _n = found
    return float(np.degrees(np.arctan(aR) - np.arctan(aL)))


def test_text_edges_read_a_justified_block():
    page = _justified_page()
    (aL, bL), (aR, bR), n = R.text_edges(page)
    assert n >= R.SQUARE_MIN_LINES
    assert abs(aL) < 0.005 and abs(aR) < 0.005          # vertical edges
    assert abs(bL - 180) < 12 and abs(bR - (1400 - 180)) < 12


def test_square_removes_a_trapezoid_and_leaves_a_square_page_alone():
    page = _justified_page()
    same, conv = R.square(page)
    assert same is page and abs(conv) < R.SQUARE_MIN_DEG
    skewed = _keystoned(page)
    before = _convergence(skewed)
    assert before > 2.0, before
    out, conv = R.square(skewed)
    assert abs(conv - before) < 0.2
    assert abs(_convergence(out)) < 0.3, _convergence(out)
    # the canvas grows to hold the warped page, never shrinks
    assert out.shape[0] >= skewed.shape[0] and out.shape[1] >= skewed.shape[1]
    assert out.shape[1] < skewed.shape[1] * 1.1


def test_square_levels_a_leaning_block_with_parallel_edges():
    """A block whose edges are parallel but tilted has no trapezoid to
    remove and still needs levelling: the deskew that used to do it
    turned pages toward their insets. The edges are the measure."""
    page = _justified_page()
    h, w = page.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.2, 1.0)
    leaning = cv2.warpAffine(page, M, (w, h), borderValue=(236, 236, 236))
    (aL, _), (aR, _), _n = R.text_edges(leaning)
    lean = np.degrees((np.arctan(aR) + np.arctan(aL)) / 2)
    assert abs(lean) > 0.8, lean
    out, conv = R.square(leaning)
    assert out is not leaning
    (aL2, _), (aR2, _), _n2 = R.text_edges(out)
    assert abs(np.degrees((np.arctan(aR2) + np.arctan(aL2)) / 2)) < 0.3


def test_square_keeps_every_pixel_of_the_page():
    """Widening the block's narrow end pushes that end's margin outward;
    a canvas of the page's own size lost whatever sat there. The canvas
    now holds the whole warped page: ink that touched the page's edge
    is still there, on paper fill, after squaring."""
    page = _justified_page()
    h, w = page.shape[:2]
    page[40:52, 20:w - 20] = 30                        # a rule along the very top edge
    page[h - 52:h - 40, 20:w - 20] = 30                # and along the bottom
    skewed = _keystoned(page)
    out, conv = R.square(skewed)
    assert abs(conv) > R.SQUARE_MIN_DEG
    assert out.shape[0] >= skewed.shape[0] and out.shape[1] >= skewed.shape[1]
    g = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    ink_rows = np.where((g < 100).mean(axis=1) > 0.5)[0]
    assert len(ink_rows) >= 2 and ink_rows.min() < 0.1 * out.shape[0] and ink_rows.max() > 0.9 * out.shape[0]


def test_square_refuses_what_it_cannot_measure():
    sparse = np.full((1900, 1400, 3), 236, np.uint8)
    sparse[300:316, 200:1200] = 30                    # one line is not a block
    out, conv = R.square(sparse)
    assert out is sparse and conv == 0.0
    absurd = _keystoned(_justified_page(), top_inset=260)   # ~8° is a misread, not a page
    out, conv = R.square(absurd)
    assert out is absurd and abs(conv) > R.SQUARE_MAX_DEG


def _flat_spread(w=2400, h=1300):
    """What a rectified spread looks like: two text blocks, a dark gutter,
    a sliver of desk left along one edge."""
    img = np.full((h, w, 3), 232, np.uint8)
    for x0 in (160, 1360):
        for y in range(150, h - 150, 36):
            img[y:y + 12, x0:x0 + 880] = 30
    img[:, 1188:1212] = 60            # the spine's shadow
    img[:, :30] = 28                  # desk the boundary let through
    return img


def test_capture_pages_splits_a_flat_spread_and_trims_each_page(vtb, monkeypatch):
    spread = _flat_spread()
    monkeypatch.setattr(vtb._rectify, "rectify", lambda img: (spread, True))
    photo = np.full((1300, 2400, 3), 90, np.uint8)
    parts, info = vtb.capture_pages(photo, enhance=False, deskew=False, workdir=None)
    assert info["rectified"] and info["split"] and info["kept_photo"] is None
    assert len(parts) == 2
    left, right = parts
    # the desk strip is gone from the left page, the spine shadow from both
    assert left.shape[1] < 1200 - 20 and right.shape[1] < 1200 - 10
    for p in parts:
        g = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        assert np.median(g[:, :5]) > 150 and np.median(g[:, -5:]) > 150


def test_a_missing_rectifier_keeps_the_photograph_and_says_so(vtb, monkeypatch, make_spread):
    monkeypatch.setattr(vtb._rectify, "rectify", lambda img: (img, False))
    photo = make_spread()
    parts, info = vtb.capture_pages(photo, enhance=False, deskew=False, workdir=None)
    assert info["kept_photo"] == "no rectifier" and not info["rectified"]
    # a page can be seen in this photograph, so the spread is still split
    assert len(parts) == 2 and info["split"]


def test_a_kept_photograph_of_a_closed_book_is_not_halved(vtb, monkeypatch, make_page):
    """The rectifier refused a cover shot (its sheet read no words) and
    the photograph was kept — a landscape frame holding a portrait
    cover. Halving the frame made two pages of half a cover."""
    page = make_page()
    ph, pw = page.shape[:2]
    frame = np.full((ph + 200, int((ph + 200) * 1.4), 3), 30, np.uint8)
    x0 = (frame.shape[1] - pw) // 2
    frame[100:100 + ph, x0:x0 + pw] = page                 # portrait cover, ~40% of the frame
    monkeypatch.setattr(vtb._rectify, "rectify", lambda img: (img, False))
    parts, info = vtb.capture_pages(frame, enhance=False, deskew=False, workdir=None)
    assert not info["rectified"] and len(parts) == 1 and not info["split"]


def test_a_sheet_that_lost_its_words_is_refused(vtb, monkeypatch, tmp_path):
    photo = _page(w=1400, h=1000)
    blank = np.full((1000, 1400, 3), 230, np.uint8)
    monkeypatch.setattr(vtb._rectify, "rectify", lambda img: (blank, True))
    monkeypatch.setattr(vtb._rectify, "witness",
                        lambda img, wd, scale=0.6: (120, 90.0, 1.0) if img is photo else (7, 40.0, 0.2))
    monkeypatch.setattr(vtb.shutil, "which", lambda name: "/usr/bin/tesseract")
    parts, info = vtb.capture_pages(photo, enhance=False, deskew=False,
                                    split_spreads=False, workdir=str(tmp_path))
    assert info["kept_photo"] == (120, 7) and not info["rectified"]
    assert parts[0] is photo or np.array_equal(parts[0], photo)


def test_a_witness_that_gave_no_testimony_cannot_refuse_a_sheet(vtb, monkeypatch, tmp_path):
    """A timed-out tesseract once counted as "0 words" and a sheet the
    photograph had read 462 words from was refused. No reading, no
    verdict."""
    import subprocess
    photo = _page(w=1400, h=1000)
    sheet = np.full((900, 1300, 3), 230, np.uint8)
    monkeypatch.setattr(vtb._rectify, "rectify", lambda img: (sheet, True))
    def slow(*a, **k):
        raise subprocess.TimeoutExpired(cmd="tesseract", timeout=1)
    monkeypatch.setattr(vtb._rectify.subprocess, "run", slow)
    assert R.witness(photo, str(tmp_path)) is None
    monkeypatch.setattr(vtb.shutil, "which", lambda name: "/usr/bin/tesseract")
    parts, info = vtb.capture_pages(photo, enhance=False, deskew=False,
                                    split_spreads=False, workdir=str(tmp_path))
    assert info["rectified"] and info["kept_photo"] is None


def test_the_witness_is_not_asked_about_a_page_with_nothing_to_read(vtb, monkeypatch, tmp_path):
    photo = np.full((1000, 1400, 3), 200, np.uint8)
    sheet = np.full((900, 1300, 3), 230, np.uint8)
    monkeypatch.setattr(vtb._rectify, "rectify", lambda img: (sheet, True))
    monkeypatch.setattr(vtb._rectify, "witness", lambda img, wd, scale=0.6: (3, 50.0, 0.1) if img is photo else (0, 0.0, 0.0))
    monkeypatch.setattr(vtb.shutil, "which", lambda name: "/usr/bin/tesseract")
    parts, info = vtb.capture_pages(photo, enhance=False, deskew=False,
                                    split_spreads=False, workdir=str(tmp_path))
    assert info["rectified"] and info["kept_photo"] is None


@needs_model
def test_the_model_flattens_a_photographed_spread(vtb, make_spread):
    """Smoke test on the synthetic spread every geometry test uses: the
    model must run, return a sheet of plausible size, and leave less of
    the desk in it than the photograph carries."""
    photo = make_spread()
    sheet, ok = R.rectify(photo)
    assert ok
    ph, pw = photo.shape[:2]
    assert sheet.shape[1] > 0.4 * pw and sheet.shape[0] > 0.4 * ph
    dark = lambda im: float((cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 70).mean())
    assert dark(sheet) < dark(photo)
