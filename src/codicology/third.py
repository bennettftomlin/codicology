"""A third reader, for the equations two readers cannot agree on.

On a scan nothing printed says which of two readings of an equation is the
page: there is no text layer, and neither Tesseract nor Apple Vision reads a
built-up fraction. The second reader (second.py) is right more often than
the first where they differ, and is wrong often enough that its word alone
is taken only for a digit the first reader dropped. Everything else was
listed for a look.

A third generative reader breaks most of those ties. GLM-OCR is a reader of
regions, not of pages: it is given the picture of one block and told what
kind of thing it is. The first reader's layout already says where each
equation is, so the third reader is shown that block and asked for the
formula, which takes under a second. Put to the fourteen equations of
Sraffa's Production of Commodities on which the first two readers differ,
and judged against the page: the reading that has another reader on its
side at every point of difference is the printed equation in twelve. Nine
of those are the second reader's; two are the third's own, where the first
and the second each had one symbol wrong (a Λ read as A by one, a ρ read
as p by the other); one is the first's. The other two stay as the first
has them and are wrong, because two readers misread the same mark.

It is asked about nothing else, and only where the first two differ. Like
the second reader it is not part of this package: its two .gguf files are
looked for at $CODICOLOGY_THIRD_MODEL / $CODICOLOGY_THIRD_MMPROJ, else in
~/.cache/codicology/third/, and it runs on the same llama-server.
"""
from __future__ import annotations

import os
import re

from . import second as _second

NAME = "glm-ocr"
MODEL_ENV = "CODICOLOGY_THIRD_MODEL"
MMPROJ_ENV = "CODICOLOGY_THIRD_MMPROJ"
DEFAULT_DIR = os.path.expanduser("~/.cache/codicology/third")

# What the model is told the picture is. It knows three such tasks (text,
# formula, table) and this is the one it is here for.
FORMULA = "Formula Recognition:"
# Round the block's box, as fractions of the page. A system of equations is
# a block a line, and the lines stand close: with more above and below, the
# crop took in the line before and the reading was of both.
PAD_X, PAD_Y = 0.012, 0.004
# An equation is a few dozen tokens and a page of them a few hundred.
MAX_TOKENS = 1536


def model_files() -> "tuple[str, str] | None":
    """The model and its projector, or None where there is no third reader."""
    return _second.model_files(DEFAULT_DIR, MODEL_ENV, MMPROJ_ENV)


def available() -> bool:
    return model_files() is not None and _second.server_binary() is not None


class ThirdReader(_second.SecondReader):
    """GLM-OCR behind a llama-server of its own, asked for a formula."""

    name = NAME
    prompt = FORMULA
    long_side = None            # a block is small: it is sent as it is cut
    max_tokens = MAX_TOKENS

    def __init__(self, model: "str | None" = None, mmproj: "str | None" = None,
                 server: "str | None" = None):
        if model is None or mmproj is None:
            found = model_files()
            if found is None:
                raise RuntimeError(
                    "no third reader is installed: put the model's two .gguf "
                    f"files in {DEFAULT_DIR}, or name them in ${MODEL_ENV} "
                    f"and ${MMPROJ_ENV}")
            model, mmproj = found
        super().__init__(model, mmproj, server)


def crop_box(size: "tuple[int, int]", box) -> "tuple[int, int, int, int]":
    """The pixels of a block's box on a page of this size, with the little
    round it that the reader is given."""
    w, h = size
    x0, y0, x1, y1 = box
    return (int(max(0.0, x0 - PAD_X) * w), int(max(0.0, y0 - PAD_Y) * h),
            int(min(1.0, x1 + PAD_X) * w), int(min(1.0, y1 + PAD_Y) * h))


def formula(answer: "str | None") -> "str | None":
    """The TeX of the reader's answer, without the marks it sets it off by."""
    if not answer:
        return None
    t = re.sub(r"\$\$|\\\[|\\\]", " ", answer).strip()
    return t or None
