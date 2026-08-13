"""Building a book from photographs: collect_image_paths, group_burst, pages_from_images.

--from-images exists because a still carries far more detail than a video frame, and
because a photograph is by definition a moment someone chose — it sidesteps the whole
business of deciding when the book was held still. These tests pin the three things
that path is responsible for: which files are read and in what order, how consecutive
shots of one resting spread collapse to a single page, and what comes back out.

Everything here is synthetic. deskew is off throughout (it costs a Hough transform per
page and none of these behaviours depend on it) and split_spreads is on, because a
photograph of an open book is a spread and splitting it is the interesting case.
"""
import os

import cv2
import numpy as np
import pytest


# Two shots of one resting spread differ by a fraction of a grey level on a 64px
# thumbnail; two different spreads differ by ten or more. Any threshold in between
# separates them, and a fixed one keeps the grouping tests about grouping rather
# than about the automatic threshold, which has tests of its own below.
RESTING = 5.0


def build_pages(vtb, image_paths, tmp_path, name="pages", **kw):
    """pages_from_images with the settings these tests share."""
    pages_dir = tmp_path / name
    pages_dir.mkdir(exist_ok=True)
    opts = dict(min_area_ratio=0.15, rotate=0, no_warp=False, enhance=False,
                deskew=False, split_spreads=True, burst_threshold=RESTING)
    opts.update(kw)
    return vtb.pages_from_images(image_paths, str(pages_dir), **opts)


def stems(page_ids):
    """The source-filename half of each page id: IMG_0004p1 → IMG_0004."""
    return [pid.rsplit("p", 1)[0] for pid in page_ids]


# ── collect_image_paths ───────────────────────────────────────────────────────

def test_photographs_are_ordered_by_filename_not_by_timestamp(vtb, make_page, tmp_path):
    """
    Filename order is shutter order. The docstring rejects EXIF/mtime explicitly —
    phones and cameras disagree about timestamps often enough not to trust them —
    so a folder whose modification times run backwards must still come back in
    name order, or the book is bound back to front.
    """
    folder = tmp_path / "outoforder"
    folder.mkdir()
    names = ["IMG_0009.jpg", "IMG_0001.jpg", "IMG_0010.jpg", "IMG_0002.jpg"]
    for age, name in enumerate(names):          # written, and stamped, in the wrong order
        p = folder / name
        cv2.imwrite(str(p), make_page())
        os.utime(p, (1_600_000_000 - age * 3600,) * 2)

    got = [os.path.basename(p) for p in vtb.collect_image_paths(str(folder))]
    assert got == ["IMG_0001.jpg", "IMG_0002.jpg", "IMG_0009.jpg", "IMG_0010.jpg"]


def test_non_images_and_lookalike_directories_are_ignored(vtb, make_page, tmp_path):
    """Sidecars a camera or an OS leaves behind are not photographs of pages."""
    folder = tmp_path / "mixed"
    folder.mkdir()
    cv2.imwrite(str(folder / "IMG_0001.jpg"), make_page())
    cv2.imwrite(str(folder / "IMG_0002.png"), make_page())
    (folder / "notes.txt").write_text("page 41 is smudged")
    (folder / ".DS_Store").write_bytes(b"\x00\x01")
    (folder / "IMG_9999.mov").write_bytes(b"\x00\x01")
    (folder / "contact_sheet.jpg").mkdir()      # a directory that ends .jpg

    got = [os.path.basename(p) for p in vtb.collect_image_paths(str(folder))]
    assert got == ["IMG_0001.jpg", "IMG_0002.png"]


def test_extension_match_ignores_case(vtb, make_page, tmp_path):
    """Cameras write .JPG as readily as .jpg; both are photographs."""
    folder = tmp_path / "shouty"
    folder.mkdir()
    cv2.imwrite(str(folder / "IMG_0001.JPG"), make_page())
    cv2.imwrite(str(folder / "IMG_0002.PNG"), make_page())

    got = [os.path.basename(p) for p in vtb.collect_image_paths(str(folder))]
    assert got == ["IMG_0001.JPG", "IMG_0002.PNG"]


def test_a_glob_selects_a_subset_of_a_folder(vtb, make_page, tmp_path):
    """A spec that is not a folder is a pattern, so one shoot can be sliced up."""
    folder = tmp_path / "shoot"
    folder.mkdir()
    for name in ("IMG_0001.jpg", "IMG_0002.jpg", "SCAN_0001.jpg"):
        cv2.imwrite(str(folder / name), make_page())

    got = [os.path.basename(p) for p in vtb.collect_image_paths(str(folder / "IMG_*.jpg"))]
    assert got == ["IMG_0001.jpg", "IMG_0002.jpg"]


def test_empty_folder_exits_rather_than_building_an_empty_book(vtb, tmp_path):
    """A typo'd path must stop the run, not quietly produce a nought-page PDF."""
    empty = tmp_path / "nothing_here"
    empty.mkdir()

    with pytest.raises(SystemExit) as exc:
        vtb.collect_image_paths(str(empty))
    assert str(empty) in str(exc.value)


# ── group_burst ───────────────────────────────────────────────────────────────

def test_three_shots_of_four_spreads_make_four_groups(vtb, make_spread, write_images):
    """
    Interval or burst shooting: the shutter fires while the book rests, so several
    frames show the same spread. Those runs are one page each, exactly as a video's
    still-runs are — twelve photographs of four spreads is four pages, not twelve.
    """
    frames = []
    for k, shift in enumerate([0, 45, 95, 140]):
        for jitter in (0, 2, 1):                # a hand-held camera never sits still
            frames.append(make_spread(k, shift=shift + jitter))
    paths = vtb.collect_image_paths(write_images(frames, "bursts"))

    assert vtb.group_burst(paths) == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def test_mid_turn_shot_forms_its_own_group(vtb, make_spread, write_images):
    """
    A frame caught with the leaf in the air resembles neither the spread before nor
    the one after, so it becomes a group of its own. That matters: folded into
    either neighbour it would compete to be that page's kept shot, and a smeared
    half-turned leaf can win on nothing more than the sharpness of its own blur.
    """
    frames = [make_spread(0, shift=0), make_spread(0, shift=0, blur=1),
              make_spread(0, shift=110, blur=61),          # the leaf in the air
              make_spread(1, shift=45), make_spread(1, shift=46)]
    paths = vtb.collect_image_paths(write_images(frames, "midturn"))

    assert vtb.group_burst(paths) == [[0, 1], [2], [3, 4]]


def test_identical_spread_seen_again_later_is_not_folded_into_the_earlier_group(
        vtb, make_spread, write_images):
    """
    Groups are runs of CONSECUTIVE shots, not clusters of similar ones. Photograph a
    spread, turn to the next, then come back — clustering by appearance would merge
    the two visits into one group and the book would lose a page. Contiguity is what
    stops that, and it is why the same rule copes with a book of near-blank pages.
    """
    frames = [make_spread(0, shift=0), make_spread(1, shift=95), make_spread(0, shift=0)]
    paths = vtb.collect_image_paths(write_images(frames, "revisit"))

    assert vtb.group_burst(paths, threshold=RESTING) == [[0], [1], [2]]


def test_every_photograph_lands_in_exactly_one_group_in_order(
        vtb, make_spread, write_images):
    """Grouping partitions the shoot: nothing is dropped, duplicated or reordered."""
    frames = [make_spread(k // 2, shift=(k // 2) * 50 + k % 2) for k in range(9)]
    paths = vtb.collect_image_paths(write_images(frames, "partition"))

    groups = vtb.group_burst(paths, threshold=RESTING)
    assert [i for g in groups for i in g] == list(range(len(paths)))
    assert all(g == list(range(g[0], g[-1] + 1)) for g in groups)


def test_single_photograph_is_a_single_group(vtb, make_spread, write_images):
    """One photograph has no neighbour to compare against, and is still a page."""
    paths = vtb.collect_image_paths(write_images([make_spread(0)], "lonely"))

    assert vtb.group_burst(paths) == [[0]]
    assert vtb.group_burst([]) == []


def test_explicit_threshold_keeps_deliberate_shots_apart(vtb, make_spread, write_images):
    """
    With the threshold supplied, six photographs of six spreads are six pages. This
    is the same data as the xfail below; the difference is only where the threshold
    came from, which is what localises that failure to the automatic one.
    """
    frames = [make_spread(k, shift=s) for k, s in enumerate([0, 45, 95, 140, 190, 240])]
    paths = vtb.collect_image_paths(write_images(frames, "deliberate_fixed"))

    assert vtb.group_burst(paths, threshold=RESTING) == [[0], [1], [2], [3], [4], [5]]


def test_one_deliberate_shot_per_page_stays_one_group_each(
        vtb, make_spread, write_images):
    """
    The docstring's own claim: "One deliberate photograph per page falls out of this
    untouched, since every consecutive pair then differs and every group holds a
    single image." Six photographs of six different spreads must be six pages.

    They are not. The run collapses to three groups, so three spreads never reach
    the PDF — and nothing in the output says so, because a merged group looks
    exactly like a burst that was correctly collapsed.
    """
    frames = [make_spread(k, shift=s) for k, s in enumerate([0, 45, 95, 140, 190, 240])]
    paths = vtb.collect_image_paths(write_images(frames, "deliberate"))

    assert vtb.group_burst(paths) == [[0], [1], [2], [3], [4], [5]]


# ── pages_from_images ─────────────────────────────────────────────────────────

def test_page_ids_are_taken_from_the_source_filename(vtb, make_spread, write_images,
                                                     tmp_path):
    """
    A page's name is where it came from: the photograph's own filename and which
    half of the spread it was. Position in the book is not a durable name — insert
    a re-shot spread and every later index moves, silently re-pointing a
    hand-curated drop list at the wrong pages. A filename survives all of that.
    """
    paths = vtb.collect_image_paths(write_images([make_spread(0)], "named"))

    page_paths, page_ids = build_pages(vtb, paths, tmp_path)
    assert page_ids == ["IMG_0000p0", "IMG_0000p1"]
    assert len(page_paths) == len(page_ids)
    assert all(os.path.exists(p) for p in page_paths)


def test_spread_is_split_into_two_pages_at_the_gutter(vtb, make_spread, write_images,
                                                      tmp_path):
    """One photograph of an open book is two pages of the book, split near the middle."""
    paths = vtb.collect_image_paths(write_images([make_spread(0)], "spread"))

    page_paths, page_ids = build_pages(vtb, paths, tmp_path)
    assert len(page_paths) == 2
    halves = [cv2.imread(p) for p in page_paths]
    total = sum(h.shape[1] for h in halves)
    for h in halves:
        assert h.shape[0] > h.shape[1]                    # a book page is portrait
        assert 0.4 < h.shape[1] / total < 0.6             # cut near the gutter, not at an edge
    assert page_ids == ["IMG_0000p0", "IMG_0000p1"]


def test_split_is_opt_in_so_a_spread_can_be_kept_whole(vtb, make_spread, write_images,
                                                       tmp_path):
    """With split_spreads off the spread stays one landscape page carrying one id."""
    paths = vtb.collect_image_paths(write_images([make_spread(0)], "whole"))

    page_paths, page_ids = build_pages(vtb, paths, tmp_path, split_spreads=False)
    assert page_ids == ["IMG_0000p0"]
    img = cv2.imread(page_paths[0])
    assert img.shape[1] > img.shape[0]


def test_burst_of_a_resting_spread_yields_one_pair_of_pages(vtb, make_spread,
                                                            write_images, tmp_path):
    """
    Six photographs of two spreads are four pages, not twelve. Both halves of a
    spread carry the id of the one shot that was kept, so a burst leaves no trace
    downstream beyond having been the better exposure.
    """
    frames = []
    for k, shift in enumerate([0, 95]):
        for jitter in (0, 2, 1):
            frames.append(make_spread(k, shift=shift + jitter))
    paths = vtb.collect_image_paths(write_images(frames, "burstpages"))

    page_paths, page_ids = build_pages(vtb, paths, tmp_path)
    assert len(page_paths) == 4
    assert [pid[-2:] for pid in page_ids] == ["p0", "p1", "p0", "p1"]
    left, right = stems(page_ids)[:2], stems(page_ids)[2:]
    assert left[0] == left[1] and right[0] == right[1]    # one source per spread
    assert left[0] != right[0]                            # and a different one each


def test_sharpest_shot_of_a_burst_is_the_one_kept(vtb, make_spread, write_images,
                                                  tmp_path):
    """
    The run is scored and the best frame wins — taking the first of the group would
    be free and would hand the book whichever moment the shutter happened to open
    on. Here the sharp exposure sits second of three, so its id can only appear if
    the group was actually scored.
    """
    frames = [make_spread(0, blur=9), make_spread(0), make_spread(0, blur=5)]
    paths = vtb.collect_image_paths(write_images(frames, "sharpest"))

    page_paths, page_ids = build_pages(vtb, paths, tmp_path, burst_threshold=3.0)
    assert len(page_ids) == 2                             # the three shots were one page
    assert set(stems(page_ids)) == {"IMG_0001"}           # not IMG_0000, the first


def test_unreadable_photograph_is_skipped_without_shifting_the_rest(
        vtb, make_spread, write_images, tmp_path):
    """
    A truncated or corrupt file in the middle of a shoot is dropped, and the pages
    either side keep their own names and a contiguous run of files behind them.
    """
    folder = write_images([make_spread(0), make_spread(1, shift=95)], "corrupt")
    open(os.path.join(folder, "IMG_0000b.jpg"), "wb").write(b"not an image at all")
    paths = vtb.collect_image_paths(folder)
    assert len(paths) == 3                                # the junk file was collected

    page_paths, page_ids = build_pages(vtb, paths, tmp_path)
    assert page_ids == ["IMG_0000p0", "IMG_0000p1", "IMG_0001p0", "IMG_0001p1"]
    assert [os.path.basename(p) for p in page_paths] == [
        f"page_{i:04d}.jpg" for i in range(4)]
    assert all(cv2.imread(p) is not None for p in page_paths)


def test_portrait_photograph_of_a_single_page_is_not_split(vtb, make_page,
                                                           write_images, tmp_path):
    """
    Photographing one page at a time is a legitimate way to shoot a book, and there
    is no gutter in the frame to cut at. Only a landscape crop is a spread.
    """
    paths = vtb.collect_image_paths(write_images([make_page()], "singles"))

    page_paths, page_ids = build_pages(vtb, paths, tmp_path)
    assert page_ids == ["IMG_0000p0"]
    img = cv2.imread(page_paths[0])
    assert img.shape[0] > img.shape[1]


def test_written_pages_are_real_images_at_print_resolution(vtb, make_spread,
                                                           write_images, tmp_path):
    """The pages handed downstream are files on disk that a PDF can actually embed."""
    from PIL import Image

    paths = vtb.collect_image_paths(write_images([make_spread(0)], "resolution"))
    page_paths, _ = build_pages(vtb, paths, tmp_path)

    for p in page_paths:
        with Image.open(p) as im:
            assert im.format == "JPEG"
            assert im.info.get("dpi") == (300, 300)
            assert min(im.size) > 100
            assert np.asarray(im).std() > 1.0             # not a blank rectangle
