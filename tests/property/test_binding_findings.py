"""What the property tests of binding found, shrunk to named cases.

Each is marked xfail (strict) until binding is mended: when one starts to
pass, the mark comes off and it stays as an ordinary test.
"""
import re

import pytest

from codicology import binding as B


@pytest.mark.xfail(strict=True, reason=(
    "_BROKEN_WORD and the close-set test read the end of the paragraph's "
    "markup, so a hyphen inside </i> is not seen: a space is put after it "
    "and the printer's hyphen is kept"))
@pytest.mark.parametrize("a, b", [
    ("<i>manu-</i>", "<i>facturing</i>"),
    ("<i>manu-</i>", "facturing"),
])
def test_a_word_in_italic_broken_at_the_foot_is_made_whole(a, b):
    pages = [(1, f"<p>It was a matter for the whole of the {a}</p>"),
             (2, f"<p>{b} interest of the town.</p>")]
    flow = B.bind(pages)
    assert B.unbind(flow) == pages
    assert "manufacturing" in re.sub(r"<[^>]*>", "", flow)


# fixed 2026-11-19
@pytest.mark.parametrize("a, b, whole", [
    ("Égyp-", "tian", "Égyptian"),
    ("naï-", "veté", "naïveté"),
    ("Ré-", "sumé", "Résumé"),
])
def test_a_word_with_an_accent_broken_at_the_foot_is_made_whole(a, b, whole):
    pages = [(1, f"<p>It was a matter for the whole of the ancient {a}</p>"),
             (2, f"<p>{b} kings of the river.</p>")]
    flow = B.bind(pages)
    assert B.unbind(flow) == pages
    assert whole in re.sub(r"<[^>]*>", "", flow)
