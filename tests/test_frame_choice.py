"""Which frame of a still run is kept.

The pipeline ranks candidate frames by STILLNESS, not sharpness. A Laplacian
variance cannot tell ghosting from detail: motion prints every edge of the page
twice, and the variance counts the extra edges as extra detail, so a smeared
frame routinely outscores a clean one. These tests pin that trap (it is
reproduced here, not assumed), the hand penalty that keeps fingers off the
kept page, and the detectability probes that stop either rule from costing a
page outright.

Everything is synthetic: pages come from the make_page fixture, motion is a
number handed to the picker, and no OCR backend is ever touched.
"""
import cv2
import numpy as np
import pytest


# ── frame builders ────────────────────────────────────────────────────────────

def ghosted(frame, shift=29):
    """The same page printed twice, half a line of text apart.

    A real hand-held smear lays the page's ink down in two places at once. Taking
    the darker of the two positions keeps both copies at full contrast, which is
    exactly the doubled-edge frame the sharpness score was shown to love.
    """
    M = np.float32([[1, 0, 0], [0, 1, shift]])
    moved = cv2.warpAffine(frame, M, (frame.shape[1], frame.shape[0]),
                           borderMode=cv2.BORDER_REPLICATE)
    return np.minimum(frame, moved)


def far_away(frame):
    """The page pushed into one corner of a dark frame, too small to be a page.

    Sharp, unobscured, and still nothing detect_page will accept at the ratio the
    main pass uses — the "no page can be found in it" case.
    """
    h, w = frame.shape[:2]
    out = np.full((h, w, 3), 18, np.uint8)
    small = cv2.resize(frame, (w // 6, h // 6))
    out[40:40 + small.shape[0], 40:40 + small.shape[1]] = small
    return out


def softened(frame, k=9):
    """Slightly out of focus, but held still and unobscured."""
    return cv2.GaussianBlur(frame, (k, k), 0)


def entry(vtb, unsteadiness, frame):
    """A _pick_quietest candidate, scored the way select_page_frames scores one."""
    score, sharp = vtb._frame_score(frame)
    return (unsteadiness, score, sharp, frame)


def scored(vtb, frame):
    """A _pick_detectable candidate: (score, sharp, frame)."""
    score, sharp = vtb._frame_score(frame)
    return (score, sharp, frame)


# ── blur_score ────────────────────────────────────────────────────────────────

def test_blur_score_drops_when_the_page_goes_out_of_focus(vtb, make_page):
    page = make_page()
    assert vtb.blur_score(softened(page)) < vtb.blur_score(page)


def test_motion_doubled_edges_outscore_the_clean_frame(vtb, make_page):
    """The failure that replaced sharpness ranking with stillness ranking.

    A frame smeared by movement reads as *more* detailed than the clean frame it
    came from, because every edge now appears twice. Any picker that ranks a
    still run on blur_score therefore ships the ghosted frame.
    """
    page = make_page()
    smeared = ghosted(page)
    assert vtb.blur_score(smeared) > vtb.blur_score(page)


def test_blur_score_reads_colour_and_greyscale_alike(vtb, make_page):
    page = make_page()
    grey = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    assert vtb.blur_score(grey) == pytest.approx(vtb.blur_score(page))


# ── _frame_score ──────────────────────────────────────────────────────────────

def test_hand_across_the_text_scores_below_the_same_page_clean(vtb, make_page):
    clean_score, _ = vtb._frame_score(make_page(hand=False))
    hand_score, _ = vtb._frame_score(make_page(hand=True))
    assert clean_score > 0
    assert hand_score < clean_score


def test_softer_unobscured_frame_outscores_the_sharper_hand_held_one(vtb, make_page):
    """The reason the score exists: a hand pressed flat photographs sharper.

    Sharpness alone hands the book the fingered frame. The penalty has to let a
    marginally softer but unobscured frame win instead.
    """
    hand = make_page(hand=True)
    soft_clean = softened(make_page())

    hand_score, hand_sharp = vtb._frame_score(hand)
    soft_score, soft_sharp = vtb._frame_score(soft_clean)

    assert hand_sharp > soft_sharp          # sharpness alone would pick the hand
    assert soft_score > hand_score          # the score does not


def test_skin_only_at_the_edges_leaves_the_score_unpenalised(vtb, make_page):
    """Measured over the whole frame the hands holding the book never leave.

    That reading is a constant, and a constant ranks nothing; only skin over the
    middle of the page — fingers across the text — may cost a frame anything.
    """
    page = make_page()
    h, w = page.shape[:2]
    cv2.circle(page, (60, 80), 45, (150, 190, 225), -1)          # thumb at a corner
    cv2.circle(page, (w - 60, h - 80), 45, (150, 190, 225), -1)  # and the far one

    small = cv2.resize(page, (480, int(h * 480 / w)))
    assert (vtb._skin_mask(small) > 0).mean() > 0.01   # skin really is in frame

    score, sharp = vtb._frame_score(page)
    assert score == pytest.approx(sharp)               # yet nothing was deducted


def test_reported_sharpness_is_never_hand_penalised(vtb, make_page):
    """The second return value is focus, so the run's report is not distorted."""
    hand = make_page(hand=True)
    score, sharp = vtb._frame_score(hand)
    assert score == 0.0                                # penalised out of contention
    assert sharp == pytest.approx(vtb._sharpness(hand))


# ── _pick_quietest ────────────────────────────────────────────────────────────

def test_stillest_frame_wins_over_a_sharper_shakier_one(vtb, make_page):
    """The whole point of ranking a shaky hold on stillness.

    The ghosted frame scores far higher on sharpness than the clean one — that is
    the documented trap, asserted here so the test fails if the trap ever stops
    being live — and the picker must still take the still frame.
    """
    page = make_page()
    smeared = ghosted(page)

    clean = entry(vtb, 0.4, page)
    shaky = entry(vtb, 3.6, smeared)
    assert shaky[2] > clean[2]      # the smeared frame is the "sharper" one

    frame, sharp = vtb._pick_quietest([shaky, clean], 0.15)
    assert frame is page
    assert sharp == pytest.approx(clean[2])


def test_frame_with_a_hand_across_the_text_is_stepped_over(vtb, make_page):
    """A hand resting on the page is the stillest thing in the run."""
    hand = make_page(hand=True)
    page = make_page()

    stillest_but_fingered = entry(vtb, 0.1, hand)
    assert stillest_but_fingered[1] == 0.0
    shakier_but_clean = entry(vtb, 1.9, page)

    frame, sharp = vtb._pick_quietest([stillest_but_fingered, shakier_but_clean], 0.15)
    assert frame is page
    assert sharp == pytest.approx(shakier_but_clean[2])


def test_undetectable_frame_is_stepped_over_even_when_stillest(vtb, make_page):
    """Stillness only decides among frames a page can be cropped out of."""
    page = make_page()
    distant = far_away(page)
    assert vtb.detect_page(distant, 0.15) is None

    stillest_but_useless = entry(vtb, 0.2, distant)
    assert stillest_but_useless[1] > 0        # not rejected for a hand
    usable = entry(vtb, 2.1, page)

    frame, _ = vtb._pick_quietest([stillest_but_useless, usable], 0.15)
    assert frame is page


def test_returns_the_stillest_detectable_frame_when_every_score_is_zero(vtb, make_page):
    """Every frame of the hold has a hand on it — a page is never dropped for that."""
    stillest = make_page(hand=True)
    middling = make_page(hand=True, lines=12)
    worst = make_page(hand=True, lines=8)
    candidates = [entry(vtb, 2.4, worst), entry(vtb, 0.3, stillest), entry(vtb, 1.1, middling)]
    assert all(c[1] == 0.0 for c in candidates)

    frame, sharp = vtb._pick_quietest(candidates, 0.15)
    assert frame is stillest
    assert sharp == pytest.approx(vtb._sharpness(stillest))


def test_returns_a_frame_even_when_no_page_can_be_found_in_any(vtb, make_page):
    """Rather than reaching back to a moving frame, or handing back nothing."""
    stillest = far_away(make_page())
    shakier = far_away(softened(make_page()))
    assert vtb.detect_page(stillest, 0.15) is None
    assert vtb.detect_page(shakier, 0.15) is None

    quiet = [entry(vtb, 2.8, shakier), entry(vtb, 0.5, stillest)]
    frame, sharp = vtb._pick_quietest(quiet, 0.15)
    assert frame is stillest
    assert sharp == pytest.approx(vtb._sharpness(stillest))


def test_reports_the_sharpness_of_the_frame_it_chose(vtb, make_page):
    """The run's focus report must describe the frame that was actually kept."""
    hand = make_page(hand=True)
    page = make_page()
    quiet = [entry(vtb, 0.2, hand), entry(vtb, 1.4, page)]

    frame, sharp = vtb._pick_quietest(quiet, 0.15)
    assert frame is page
    assert sharp == pytest.approx(vtb._sharpness(page))
    assert sharp != pytest.approx(vtb._sharpness(hand))


# ── _pick_detectable ──────────────────────────────────────────────────────────

def test_detectable_frame_beats_a_higher_scoring_undetectable_one(vtb, make_page):
    """Nothing in the scoring stops the top-scoring frame from being uncroppable.

    Committing to whichever frame the score alone preferred is what used to cost
    a page that the run could perfectly well have kept.
    """
    page = make_page()
    distant = far_away(page)
    soft = softened(page)

    top_but_useless = scored(vtb, distant)
    lower_but_usable = scored(vtb, soft)
    assert top_but_useless[0] > lower_but_usable[0]
    assert vtb.detect_page(distant, 0.15) is None
    assert vtb.detect_page(soft, 0.15) is not None

    frame, sharp = vtb._pick_detectable([lower_but_usable, top_but_useless], 0.15)
    assert frame is soft
    assert sharp == pytest.approx(lower_but_usable[1])


def test_detection_is_probed_at_the_caller_s_min_area_ratio(vtb, make_page):
    """A frame that "detects" under a looser threshold can still fail the real one."""
    page = make_page()
    distant = far_away(page)
    soft = softened(page)
    pool = [scored(vtb, soft), scored(vtb, distant)]

    strict, _ = vtb._pick_detectable(pool, 0.15)
    loose, _ = vtb._pick_detectable(pool, 0.02)
    assert strict is soft            # too small to be a page at the real ratio
    assert loose is distant          # accepted, and it outscores the soft frame


def test_falls_back_to_the_highest_scoring_frame_when_none_detects(vtb, make_page):
    page = make_page()
    weakest = far_away(softened(page, 15))
    best = far_away(page)
    middling = far_away(softened(page))
    pool = [scored(vtb, weakest), scored(vtb, best), scored(vtb, middling)]
    assert all(vtb.detect_page(c[2], 0.15) is None for c in pool)

    frame, sharp = vtb._pick_detectable(pool, 0.15)
    assert frame is best
    assert sharp == pytest.approx(vtb._sharpness(best))
