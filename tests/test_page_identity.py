"""Stable page ids and selectors.

A page's position in the book is not a durable name for it: re-extracting with a
different frame-selection rule can split a still-run into a different number of
parts, and every index after that point shifts. A hand-curated drop list written
against the old positions then quietly removes the wrong pages. The whole point
of `rNNNpP` / `IMG_xxxxpP` ids is that they survive that, so the tests that
matter most here are the ones that re-extract the book and check the same
selector still names the same page.

Everything is synthetic: `_finish` is called directly with a fake img2pdf, no
EPUB and no OCR backend, so nothing here needs a model or a video.
"""
import json

import pytest


# ── helpers ───────────────────────────────────────────────────────────────────

def finish(vtb, fake_img2pdf, paths, tmp_path, name="book.pdf", **kw):
    """Run _finish for PDF only and hand back the pages it laid into the PDF."""
    out = tmp_path / name
    vtb._finish(paths, str(out), None, None, "Book", False, fake_img2pdf, **kw)
    return list(fake_img2pdf.pages)


def named(result_paths, paths, ids):
    """Say which ids the surviving pages carry, in the order they ended up."""
    lookup = dict(zip(paths, ids))
    return [lookup[p] for p in result_paths]


# ── page_id ───────────────────────────────────────────────────────────────────

def test_page_id_from_video_names_the_run_and_the_half(vtb):
    assert vtb.page_id(60, 1) == "r060p1"
    assert vtb.page_id(60, 0) == "r060p0"
    assert vtb.page_id(0, 0) == "r000p0"
    assert vtb.page_id(7, 0) == "r007p0"


def test_page_id_pads_the_run_ordinal_but_never_truncates_it(vtb):
    # Zero padding is only so ids sort readably; a book with more than a
    # thousand runs must still get a distinct id per run.
    assert vtb.page_id(9, 0) == "r009p0"
    assert vtb.page_id(1234, 0) == "r1234p0"
    assert vtb.page_id(1234, 0) != vtb.page_id(234, 0)


def test_page_id_from_photograph_keeps_the_filename_stem(vtb):
    # A filename survives even a change of capture, where a run ordinal only
    # survives a change of selection.
    assert vtb.page_id("IMG_4471", 0) == "IMG_4471p0"
    assert vtb.page_id("IMG_4471", 1) == "IMG_4471p1"


def test_page_id_is_never_a_bare_number_so_it_cannot_be_read_as_a_position(vtb):
    # The part suffix is what keeps the two namespaces from colliding, even for
    # a photograph whose filename is nothing but digits.
    for made in (vtb.page_id(60, 1), vtb.page_id(4471, 0),
                 vtb.page_id("IMG_4471", 0), vtb.page_id("4471", 0)):
        assert not vtb.POSITIONAL.match(made), made


# ── POSITIONAL ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("tok", ["0", "7", "42", "123"])
def test_positional_matches_a_bare_integer(vtb, tok):
    assert vtb.POSITIONAL.match(tok)


@pytest.mark.parametrize("tok", ["r060p1", "IMG_4471p0", "4471p0", "", "3a",
                                 "-1", "1.0", "p1"])
def test_positional_rejects_anything_that_is_not_a_bare_integer(vtb, tok):
    assert not vtb.POSITIONAL.match(tok)


# ── resolve_selector ──────────────────────────────────────────────────────────

def test_bare_integer_selects_by_position_without_any_page_map(vtb):
    assert vtb.resolve_selector("3", None) == 3
    assert vtb.resolve_selector("0", None) == 0


def test_bare_integer_still_means_position_even_when_a_page_map_exists(vtb):
    ids = ["r000p0", "r001p0", "r002p0"]
    assert vtb.resolve_selector("1", ids) == 1


def test_id_selects_by_source_not_position(vtb):
    ids = ["r000p0", "r001p0", "r001p1", "IMG_4471p0"]
    assert vtb.resolve_selector("r001p1", ids) == 2
    assert vtb.resolve_selector("IMG_4471p0", ids) == 3


def test_id_survives_reextraction_where_a_position_silently_shifts(vtb):
    """The failure the whole scheme exists to prevent.

    Selection decides which *frame* represents a run, never which runs exist,
    so a re-extraction that splits run 1 into two parts pushes every later page
    down by one. The id follows the page; the number does not.
    """
    first = ["r000p0", "r001p0", "r002p0", "r003p0"]
    again = ["r000p0", "r001p0", "r001p1", "r002p0", "r003p0"]

    assert vtb.resolve_selector("r003p0", first) == 3
    assert vtb.resolve_selector("r003p0", again) == 4        # id followed it

    assert vtb.resolve_selector("3", first) == 3
    assert vtb.resolve_selector("3", again) == 3             # number did not
    assert again[vtb.resolve_selector("3", again)] == "r002p0"


def test_id_without_a_page_map_raises_pointing_at_the_way_to_get_one(vtb):
    with pytest.raises(ValueError, match="no page map is available") as exc:
        vtb.resolve_selector("r060p1", None)
    assert "--page-map" in str(exc.value)


def test_empty_page_map_counts_as_no_page_map(vtb):
    with pytest.raises(ValueError, match="no page map is available"):
        vtb.resolve_selector("r060p1", [])


def test_unknown_id_raises_naming_the_page_that_was_asked_for(vtb):
    with pytest.raises(ValueError, match="no page r099p0 in this book"):
        vtb.resolve_selector("r099p0", ["r000p0", "r001p0"])


def test_selector_ignores_whitespace_around_the_token(vtb):
    ids = ["r000p0", "r001p0"]
    assert vtb.resolve_selector("  r001p0 ", ids) == 1
    assert vtb.resolve_selector(" 1 ", None) == 1


# ── load_page_map ─────────────────────────────────────────────────────────────

def test_page_map_found_beside_the_pdf_as_book_pdf_pages_json(vtb, tmp_path):
    pdf = tmp_path / "book.pdf"
    (tmp_path / "book.pdf.pages.json").write_text(
        json.dumps({"page_ids": ["r000p0", "r001p0"], "pages": 2}))
    assert vtb.load_page_map(str(pdf)) == ["r000p0", "r001p0"]


def test_page_map_found_beside_the_pdf_as_book_pages_json(vtb, tmp_path):
    pdf = tmp_path / "book.pdf"
    (tmp_path / "book.pages.json").write_text(
        json.dumps({"page_ids": ["r000p0", "r001p0"], "pages": 2}))
    assert vtb.load_page_map(str(pdf)) == ["r000p0", "r001p0"]


def test_page_map_is_none_when_no_sidecar_exists(vtb, tmp_path):
    assert vtb.load_page_map(str(tmp_path / "book.pdf")) is None


def test_page_map_is_none_rather_than_an_error_when_the_sidecar_is_unusable(
        vtb, tmp_path):
    # A broken sidecar must degrade to "select by position", not crash the run.
    (tmp_path / "torn.pdf.pages.json").write_text("{not json at all")
    assert vtb.load_page_map(str(tmp_path / "torn.pdf")) is None

    (tmp_path / "wrong.pdf.pages.json").write_text(json.dumps({"pages": 4}))
    assert vtb.load_page_map(str(tmp_path / "wrong.pdf")) is None

    (tmp_path / "empty.pdf.pages.json").write_text(json.dumps({"page_ids": []}))
    assert vtb.load_page_map(str(tmp_path / "empty.pdf")) is None


def test_page_map_ids_come_back_as_strings(vtb, tmp_path):
    # Selectors arrive from the command line as text, so ids that happen to
    # have been written as numbers still have to be matchable.
    (tmp_path / "book.pdf.pages.json").write_text(
        json.dumps({"page_ids": [1, 2], "pages": 2}))
    ids = vtb.load_page_map(str(tmp_path / "book.pdf"))
    assert ids == ["1", "2"]


# ── _finish: dropping ─────────────────────────────────────────────────────────

def test_dropping_by_id_and_by_position_give_the_same_book(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(5)
    ids = [vtb.page_id(i, 0) for i in range(5)]

    by_position = finish(vtb, fake_img2pdf, paths, tmp_path,
                         drop_pages=["1", "3"], page_ids=ids)
    by_id = finish(vtb, fake_img2pdf, paths, tmp_path,
                   drop_pages=["r001p0", "r003p0"], page_ids=ids)

    assert by_position == by_id == [paths[0], paths[2], paths[4]]


def test_pdf_is_written_from_the_pages_that_survived(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(4)
    kept = finish(vtb, fake_img2pdf, paths, tmp_path, drop_pages=["0"])
    assert kept == paths[1:]
    assert (tmp_path / "book.pdf").read_bytes() == b"%PDF-1.4 fake"


def test_ids_and_positions_mix_freely_in_one_drop_list(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(5)
    ids = ["IMG_4471p0", "IMG_4471p1", "IMG_4472p0", "IMG_4473p0", "IMG_4474p0"]

    kept = finish(vtb, fake_img2pdf, paths, tmp_path,
                  drop_pages=["0", "IMG_4472p0", "4"], page_ids=ids)

    assert named(kept, paths, ids) == ["IMG_4471p1", "IMG_4473p0"]


def test_curated_drop_list_still_removes_the_right_pages_after_reextraction(
        vtb, fake_img2pdf, page_files, tmp_path):
    """The one that bit twice for real.

    The same hand-written drop list is applied to two extractions of one book;
    the second split a run into two parts, so every later position moved. Named
    by id the list removes the pages it meant; named by position it does not.
    """
    curated = ["r002p0", "r003p0"]

    first_paths = page_files(4)
    first_ids = ["r000p0", "r001p0", "r002p0", "r003p0"]
    again_paths = page_files(5)
    again_ids = ["r000p0", "r001p0", "r001p1", "r002p0", "r003p0"]

    kept_first = finish(vtb, fake_img2pdf, first_paths, tmp_path,
                        drop_pages=curated, page_ids=first_ids)
    kept_again = finish(vtb, fake_img2pdf, again_paths, tmp_path,
                        drop_pages=curated, page_ids=again_ids)

    assert named(kept_first, first_paths, first_ids) == ["r000p0", "r001p0"]
    assert named(kept_again, again_paths, again_ids) == \
        ["r000p0", "r001p0", "r001p1"]

    # And the positional spelling of the same list is exactly the damage the
    # ids are there to avoid: it takes two pages that were meant to be kept.
    by_number = finish(vtb, fake_img2pdf, again_paths, tmp_path,
                       drop_pages=["2", "3"], page_ids=again_ids)
    assert named(by_number, again_paths, again_ids) == \
        ["r000p0", "r001p0", "r003p0"]


# ── _finish: swapping ─────────────────────────────────────────────────────────

def test_swapping_by_id_and_by_position_give_the_same_book(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(4)
    ids = [vtb.page_id(i, 0) for i in range(4)]

    by_position = finish(vtb, fake_img2pdf, paths, tmp_path,
                         swap_pairs=[("1", "2")], page_ids=ids)
    by_id = finish(vtb, fake_img2pdf, paths, tmp_path,
                   swap_pairs=[("r001p0", "r002p0")], page_ids=ids)

    assert by_position == by_id == [paths[0], paths[2], paths[1], paths[3]]


def test_swap_names_pages_as_they_went_in_so_a_drop_cannot_renumber_it(
        vtb, fake_img2pdf, page_files, tmp_path):
    """Drop and swap compose the same whichever is imagined to happen first.

    Both are numbered against the pages as they went in — what the review sheet
    shows — so dropping page 0 must not make "3" and "4" mean the pages that
    have slid into those slots.
    """
    paths = page_files(6)

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 drop_pages=["0"], swap_pairs=[("3", "4")])

    # Swap first, then drop, in the original numbering: the same book.
    swapped_then_dropped = [paths[1], paths[2], paths[4], paths[3], paths[5]]
    assert got == swapped_then_dropped
    # Had the drop renumbered the pages, "3"/"4" would have moved paths[4]/[5].
    assert got != [paths[1], paths[2], paths[3], paths[5], paths[4]]


def test_drop_and_swap_compose_the_same_in_either_order(
        vtb, fake_img2pdf, page_files, tmp_path):
    # The hard case for composition: the dropped page sits *between* the two
    # pages being swapped, so applying the drop first shifts one partner and
    # not the other unless both are tracked by their original index.
    paths = page_files(6)

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 drop_pages=["2"], swap_pairs=[("1", "4")])

    swap_first = [paths[0], paths[4], paths[2], paths[3], paths[1], paths[5]]
    then_drop = [p for p in swap_first if p != paths[2]]
    assert got == then_drop == [paths[0], paths[4], paths[3], paths[1], paths[5]]


def test_swap_skipped_when_partner_dropped(
        vtb, fake_img2pdf, page_files, tmp_path, capsys):
    paths = page_files(5)
    ids = [vtb.page_id(i, 0) for i in range(5)]

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 drop_pages=["r002p0"], swap_pairs=[("r002p0", "r003p0")],
                 page_ids=ids)
    out = capsys.readouterr().out

    # The surviving pages keep their order — the swap is not half-applied onto
    # whatever page happens to sit where the dropped one used to be.
    assert got == [paths[0], paths[1], paths[3], paths[4]]
    assert "skipped" in out and "already dropped" in out
    assert "r002p0" in out
    assert "Swapped 0 page pair(s)" in out


def test_swap_skipped_when_the_second_partner_was_dropped(
        vtb, fake_img2pdf, page_files, tmp_path, capsys):
    paths = page_files(5)

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 drop_pages=["3"], swap_pairs=[("2", "3")])
    out = capsys.readouterr().out

    assert got == [paths[0], paths[1], paths[2], paths[4]]
    assert "skipped" in out and "already dropped" in out


def test_a_skipped_swap_does_not_abandon_the_rest_of_the_list(
        vtb, fake_img2pdf, page_files, tmp_path, capsys):
    # One stale pair in a hand-curated list must not cost the other pairs.
    paths = page_files(5)

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 drop_pages=["2"], swap_pairs=[("2", "3"), ("0", "1")])
    out = capsys.readouterr().out

    assert got == [paths[1], paths[0], paths[3], paths[4]]
    assert "Swapped 1 page pair(s)" in out


def test_a_second_swap_sees_where_the_first_one_moved_the_page(
        vtb, fake_img2pdf, page_files, tmp_path):
    # Swaps chain: the second pair must act on where page 0 is *now*, not on
    # the slot it started in, or the two swaps undo each other.
    paths = page_files(3)

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 swap_pairs=[("0", "1"), ("0", "2")])

    assert got == [paths[1], paths[2], paths[0]]


def test_swaps_and_drops_together_keep_every_other_page_in_place(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(6)
    ids = ["r000p0", "r001p0", "r001p1", "r002p0", "r003p0", "IMG_9p0"]

    got = finish(vtb, fake_img2pdf, paths, tmp_path,
                 drop_pages=["r001p1"], swap_pairs=[("r002p0", "r003p0")],
                 page_ids=ids)

    assert named(got, paths, ids) == \
        ["r000p0", "r001p0", "r003p0", "r002p0", "IMG_9p0"]


# ── _finish: selectors that cannot mean anything ──────────────────────────────

def test_unknown_id_in_a_drop_list_stops_the_run(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(3)
    ids = [vtb.page_id(i, 0) for i in range(3)]

    with pytest.raises(SystemExit) as exc:
        finish(vtb, fake_img2pdf, paths, tmp_path,
               drop_pages=["r099p0"], page_ids=ids)

    assert "--drop-pages" in str(exc.value)
    assert "no page r099p0" in str(exc.value)


def test_unknown_id_in_a_swap_stops_the_run(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(3)
    ids = [vtb.page_id(i, 0) for i in range(3)]

    with pytest.raises(SystemExit) as exc:
        finish(vtb, fake_img2pdf, paths, tmp_path,
               swap_pairs=[("r000p0", "r099p0")], page_ids=ids)

    assert "--swap" in str(exc.value)
    assert "no page r099p0" in str(exc.value)


def test_an_id_with_no_page_map_stops_the_run_rather_than_guessing(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(3)

    with pytest.raises(SystemExit) as exc:
        finish(vtb, fake_img2pdf, paths, tmp_path, drop_pages=["r001p0"])

    assert "no page map is available" in str(exc.value)
    assert "--page-map" in str(exc.value)


def test_a_position_past_the_end_of_the_book_stops_the_run(
        vtb, fake_img2pdf, page_files, tmp_path):
    paths = page_files(5)

    with pytest.raises(SystemExit) as exc:
        finish(vtb, fake_img2pdf, paths, tmp_path, drop_pages=["99"])

    assert "outside this book (0-4)" in str(exc.value)


def test_single_small_illustration_does_not_become_the_page(vtb, tmp_path):
    """
    The failure a field manual exposed: a born-digital page holding ONE small
    illustration also holds exactly one image object, and the single-image
    shortcut kept the 616x330 illustration as "the page" — discarding every
    word of vector text on it. Forty-seven pages of procedure prose vanished
    silently. The shortcut must check the image actually covers the page.
    """
    import os
    src = "/Users/Bennett1/claude_knowledge/assets/books/FM4-25x11.pdf"
    if not os.path.exists(src):
        import pytest
        pytest.skip("fixture book not present")
    from PIL import Image
    paths = vtb.load_pages_from_pdf(src, str(tmp_path))
    with Image.open(paths[25]) as im:
        w, h = im.size
    # rendered whole page (~1600px wide at 300dpi), not the 616px illustration
    assert w > 1200, (w, h)
