"""An embedded image faces the way the page draws it, not the way it is stored.

FM 20-3's electromagnetic-spectrum chart is placed with a negative vertical
scale. Taken as its own bytes it shipped upside down, and the confirmation
check waved it through because a broad banded diagram correlates with its
own reflection at thumbnail size. The placement matrix is the authority.
"""
from PIL import Image

from codicology import pipeline as vtb


class _M:
    def __init__(self, a=1.0, b=0.0, c=0.0, d=1.0, e=0.0, f=0.0):
        self.a, self.b, self.c, self.d, self.e, self.f = a, b, c, d, e, f


class _Obj:
    def __init__(self, m):
        self._m = m

    def get_matrix(self):
        return self._m


def test_upright_placement_needs_no_turning(vtb_=None):
    assert vtb.native_orientation(_Obj(_M(a=423.1, d=159.1))) == []


def test_a_negative_vertical_scale_is_upside_down(vtb_=None):
    """FM 20-3 page 13: a=423.1, d=-159.1."""
    assert vtb.native_orientation(_Obj(_M(a=423.1, d=-159.1))) == \
        [Image.FLIP_TOP_BOTTOM]


def test_a_negative_horizontal_scale_is_mirrored(vtb_=None):
    assert vtb.native_orientation(_Obj(_M(a=-423.1, d=159.1))) == \
        [Image.FLIP_LEFT_RIGHT]


def test_both_negative_is_a_half_turn(vtb_=None):
    turns = vtb.native_orientation(_Obj(_M(a=-1, d=-1)))
    assert turns == [Image.FLIP_TOP_BOTTOM, Image.FLIP_LEFT_RIGHT]


def test_rotation_declines_the_shortcut(vtb_=None):
    """A rotated or skewed placement is not a flip, and guessing at one
    would ship a picture at the wrong angle. The caller renders instead."""
    assert vtb.native_orientation(_Obj(_M(a=0, b=1, c=-1, d=0))) is None


def test_an_unreadable_matrix_declines_too(vtb_=None):
    class Broken:
        def get_matrix(self):
            raise RuntimeError("no matrix")
    assert vtb.native_orientation(Broken()) is None
