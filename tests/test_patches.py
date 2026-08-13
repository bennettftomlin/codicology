"""Replacing a page with a re-shot source.

Some pages the capture can never resolve, however cleverly the frames are
scored: the information is simply not in the video. Re-photographing that one
page is the only fix, so the patch path has to be trustworthy — it must land on
the page named and no other, treat the re-shot source exactly as it treats a
captured one, and pick the good frame out of a hand-held clip.

Everything here is synthetic: a few small images, a few three-to-five frame
clips written with cv2.VideoWriter into tmp_path. No OCR, no real video.
"""
import json
import os

import cv2
import numpy as np
import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

PATCH_DEFAULTS = dict(min_area_ratio=0.15, rotate=0, no_warp=True,
                      enhance=False, deskew=False)


def apply(vtb, page_paths, page_ids, patches, pages_dir, **overrides):
    """apply_patches with the crop/enhance/deskew stage off unless asked for."""
    opts = dict(PATCH_DEFAULTS, **overrides)
    return vtb.apply_patches(page_paths, page_ids, patches, pages_dir, **opts)


def write_clip(path, frames, fps=10):
    """A short hand-held clip of one page, as a real mp4v file."""
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    assert writer.isOpened(), "no mp4v encoder available for this test"
    for frame in frames:
        writer.write(frame)
    writer.release()
    return str(path)


def on_desk(page, pad=180):
    """The re-shot page as a phone actually sees it: paper on a dark desk."""
    height, width = page.shape[:2]
    desk = np.full((height + 2 * pad, width + 2 * pad, 3), 38, np.uint8)
    desk[pad:pad + height, pad:pad + width] = page
    return desk


def dark_fraction(img):
    """How much of the image is desk rather than paper."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float((gray < 80).mean())


def skin_fraction(vtb, img):
    """Skin over the middle of the frame — a hand lying across the text."""
    height, width = img.shape[:2]
    middle = vtb._skin_mask(img)[int(height * 0.18):int(height * 0.82),
                                 int(width * 0.20):int(width * 0.80)]
    return float((middle > 0).mean())


@pytest.fixture
def pages_dir(tmp_path):
    d = tmp_path / "pages"
    d.mkdir()
    return str(d)


@pytest.fixture
def still(make_page, write_images):
    """A re-shot still, visibly different from any page in page_files."""
    def build(name="reshot", **kw):
        folder = write_images([make_page(**kw)], name)
        return os.path.join(folder, "IMG_0000.jpg")
    return build


def ids_for(vtb, n):
    """Run ids as an extraction would have written them: two pages per spread."""
    return [vtb.page_id(i // 2, i % 2) for i in range(n)]


# ── which page a patch lands on ───────────────────────────────────────────────

def test_patch_by_run_id_replaces_only_the_named_page(vtb, page_files, still,
                                                      pages_dir):
    paths = page_files(4)
    before = list(paths)
    ids = ids_for(vtb, 4)
    source = still(lines=4)

    applied = apply(vtb, paths, ids, {ids[2]: source}, pages_dir)

    assert applied == 1
    assert paths[2] != before[2]
    assert [paths[i] for i in (0, 1, 3)] == [before[i] for i in (0, 1, 3)]
    assert os.path.dirname(paths[2]) == pages_dir
    # The page on disk is the re-shot one, not the page it displaced.
    patched = cv2.imread(paths[2]).mean()
    assert abs(patched - cv2.imread(source).mean()) < 3
    assert abs(patched - cv2.imread(before[2]).mean()) > 20


def test_patch_by_position_works_without_a_page_map(vtb, page_files, still,
                                                    pages_dir):
    paths = page_files(4)
    before = list(paths)

    applied = apply(vtb, paths, None, {"1": still(lines=4)}, pages_dir)

    assert applied == 1
    assert paths[1] != before[1] and os.path.exists(paths[1])
    assert [paths[i] for i in (0, 2, 3)] == [before[i] for i in (0, 2, 3)]


def test_run_id_without_a_page_map_stops_the_run(vtb, page_files, still,
                                                 pages_dir):
    """
    Naming a page by run with no page map to resolve it against is fatal.

    It would be gentler to warn and carry on, but patches run BEFORE anything is
    dropped, so an id that resolves to nothing here is always a typo or a missing
    map — never a page legitimately absent from this build. Carrying on hands
    back a book the reader believes was mended and was not, and a warning printed
    an hour into a run is a warning nobody sees.
    """
    paths = page_files(2)
    before = list(paths)

    with pytest.raises(SystemExit, match="r000p1"):
        apply(vtb, paths, None, {"r000p1": still(lines=4)}, pages_dir)
    assert paths == before


def test_unknown_id_stops_the_run(vtb, page_files, still, pages_dir):
    """A patch naming a page that does not exist is a typo, and typos stop."""
    paths = page_files(4)
    before = list(paths)
    ids = ids_for(vtb, 4)

    with pytest.raises(SystemExit, match="r099p0"):
        apply(vtb, paths, ids, {"r099p0": still(lines=4)}, pages_dir)
    assert paths == before


def test_position_outside_the_book_stops_the_run(vtb, page_files, still, pages_dir):
    """A position past the end of the book is likewise a mistake, not a no-op."""
    paths = page_files(3)
    before = list(paths)

    with pytest.raises(SystemExit, match="99"):
        apply(vtb, paths, ids_for(vtb, 3), {"99": still(lines=4)}, pages_dir)
    assert paths == before


def test_unreadable_source_is_skipped_rather_than_written(vtb, page_files,
                                                          tmp_path, pages_dir,
                                                          capsys):
    paths = page_files(2)
    before = list(paths)
    junk = tmp_path / "not_really.jpg"
    junk.write_bytes(b"this is not an image")

    applied = apply(vtb, paths, ids_for(vtb, 2), {"0": str(junk)}, pages_dir)

    assert applied == 0
    assert paths == before
    assert "not_really.jpg" in capsys.readouterr().out


def test_count_matches_the_pages_actually_replaced(vtb, page_files, still,
                                                   pages_dir):
    paths = page_files(5)
    before = list(paths)
    ids = ids_for(vtb, 5)
    source = still(lines=4)

    applied = apply(vtb, paths, ids, {
        ids[0]: source,
        "3": source,
    }, pages_dir)

    changed = [i for i, (a, b) in enumerate(zip(before, paths)) if a != b]
    assert changed == [0, 3]
    assert applied == len(changed) == 2


# ── a patch is held to the same standard as a capture ─────────────────────────

def test_patch_goes_through_the_same_crop_and_clean_path(vtb, make_page,
                                                         page_files, tmp_path,
                                                         pages_dir):
    """
    The promise in apply_patches is that a patched page is not distinguishable
    from one the capture produced: same detect/warp/enhance/deskew. A source
    dropped in raw would arrive with the desk still around it.
    """
    source = str(tmp_path / "on_desk.jpg")
    cv2.imwrite(source, on_desk(make_page()))
    raw = cv2.imread(source)
    paths = page_files(2)

    applied = apply(vtb, paths, ids_for(vtb, 2), {"1": source}, pages_dir,
                    no_warp=False, enhance=True, deskew=True)
    written = cv2.imread(paths[1])

    assert applied == 1
    assert written.shape[0] < raw.shape[0] and written.shape[1] < raw.shape[1]
    assert dark_fraction(raw) > 0.5          # the source is mostly desk
    assert dark_fraction(written) < 0.35     # the page that lands is not
    # Identical to running the capture's own preparation over the same source.
    expected = vtb._prepare_page_image(raw, 0.15, 0, False, True, True)
    assert written.shape == expected.shape
    assert np.abs(written.astype(float) - expected.astype(float)).mean() < 2.0


def test_patch_left_alone_when_no_stage_is_asked_for(vtb, make_page, page_files,
                                                     tmp_path, pages_dir):
    source = str(tmp_path / "flat.jpg")
    cv2.imwrite(source, make_page())
    raw = cv2.imread(source)
    paths = page_files(1)

    apply(vtb, paths, None, {"0": source}, pages_dir)
    written = cv2.imread(paths[0])

    assert written.shape == raw.shape
    assert np.abs(written.astype(float) - raw.astype(float)).mean() < 2.0


# ── a patch source may be a clip ──────────────────────────────────────────────

def test_still_gives_one_frame_and_a_clip_gives_all_of_them(vtb, make_page,
                                                            still, tmp_path):
    assert len(vtb._patch_frames(still())) == 1

    frames = [make_page(), make_page(lines=8), make_page(lines=4)]
    clip = write_clip(tmp_path / "three.mp4", frames)
    read_back = vtb._patch_frames(clip)

    assert len(read_back) == len(frames)
    assert all(f.shape == frames[0].shape for f in read_back)


def test_undecodable_source_yields_no_frames(vtb, tmp_path):
    junk = tmp_path / "broken.png"
    junk.write_bytes(b"\x00\x01\x02 not an image")
    assert vtb._patch_frames(junk.as_posix()) == []


def test_clip_patch_keeps_the_sharp_frame(vtb, make_page, page_files, tmp_path,
                                          pages_dir):
    sharp = make_page()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    clip = write_clip(tmp_path / "wobble.mp4",
                      [blurred, blurred, sharp, blurred, blurred])
    paths = page_files(3)

    applied = apply(vtb, paths, ids_for(vtb, 3), {"1": clip}, pages_dir)
    written = cv2.imread(paths[1])

    assert applied == 1
    decoded = vtb._patch_frames(clip)
    worst = max(vtb.blur_score(decoded[i]) for i in (0, 1, 3, 4))
    assert vtb.blur_score(written) > 10 * worst
    assert vtb.blur_score(written) > 0.5 * vtb.blur_score(decoded[2])


def test_clip_patch_rejects_the_sharpest_frame_when_a_hand_covers_the_page(
        vtb, make_page, page_files, tmp_path, pages_dir):
    """
    The frame with fingers across the text is the *sharpest* frame in the clip —
    a hand pressed flat photographs harder-edged than the paper under it — so
    ranking on sharpness alone hands the book a page with a hand on it. The
    patch path must score clips the way the capture does, hand penalty included,
    and keep the marginally softer frame that shows the whole page.
    """
    clear = cv2.GaussianBlur(make_page(), (5, 5), 0)
    handed = make_page(hand=True)
    assert vtb.blur_score(handed) > 5 * vtb.blur_score(clear), \
        "fixture no longer poses the trap this test exists for"

    clip = write_clip(tmp_path / "hand.mp4", [clear, handed, clear])
    paths = page_files(2)

    applied = apply(vtb, paths, ids_for(vtb, 2), {"0": clip}, pages_dir)
    written = cv2.imread(paths[0])

    assert applied == 1
    decoded = vtb._patch_frames(clip)
    assert skin_fraction(vtb, decoded[1]) > 0.1   # the trap really is in the clip
    assert skin_fraction(vtb, written) < 0.02     # and it is not what got kept


# ── _prepare_page_image ───────────────────────────────────────────────────────

def test_prepare_page_image_returns_the_frame_untouched_when_nothing_is_asked(
        vtb, make_page):
    img = make_page()
    out = vtb._prepare_page_image(img, 0.15, 0, True, False, False)
    assert np.array_equal(out, img)


def test_prepare_page_image_rotates_before_anything_else(vtb, make_page):
    img = make_page()
    out = vtb._prepare_page_image(img, 0.15, 90, True, False, False)
    assert out.shape[:2] == img.shape[:2][::-1]
    assert np.array_equal(out, vtb.rotate_image(img, 90))


def test_prepare_page_image_crops_the_desk_away(vtb, make_page):
    desk = on_desk(make_page())
    out = vtb._prepare_page_image(desk, 0.15, 0, False, False, False)
    assert out.shape[0] < desk.shape[0] and out.shape[1] < desk.shape[1]
    assert dark_fraction(out) < dark_fraction(desk) / 2


# ── load_patches ──────────────────────────────────────────────────────────────

PAGE_ID_BUG = "load_patches references an undefined PAGE_ID, so every id check raises NameError"


def test_missing_patch_file_is_fatal(vtb, tmp_path):
    """A patch naming a file that is not there must stop the run, not run on
    and fail an hour later when the page is written."""
    with pytest.raises(SystemExit):
        vtb.load_patches(f"r060p1={tmp_path / 'never_taken.jpg'}")


def test_inline_pairs_and_a_json_file_describe_the_same_patches(vtb, still,
                                                                tmp_path):
    one, two = still(name="one"), still(name="two")
    spec = tmp_path / "patches.json"
    spec.write_text(json.dumps({"r060p1": one, "12": two}))

    from_file = vtb.load_patches(str(spec))
    inline = vtb.load_patches(f"r060p1={one},12={two}")

    assert from_file == inline == {"r060p1": one, "12": two}


def test_id_shaped_like_nothing_is_caught_when_it_is_resolved(vtb, page_files,
                                                              still, pages_dir):
    """
    An id is no longer checked for shape when the patch list is read, because a
    page may be named for the photograph it came from and a filename can look
    like anything. It is caught instead where it can actually be checked —
    against the book's own page ids.
    """
    src = still()
    assert vtb.load_patches(f"folio-sixty={src}") == {"folio-sixty": src}

    paths = page_files(2)
    with pytest.raises(SystemExit, match="folio-sixty"):
        apply(vtb, paths, ids_for(vtb, 2), {"folio-sixty": still()}, pages_dir)


def test_token_without_a_file_is_fatal(vtb):
    with pytest.raises(SystemExit):
        vtb.load_patches("r060p1")


def test_json_patch_file_must_hold_an_object(vtb, tmp_path):
    spec = tmp_path / "patches.json"
    spec.write_text(json.dumps(["r060p1", "photo.jpg"]))
    with pytest.raises(SystemExit):
        vtb.load_patches(str(spec))
