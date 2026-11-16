"""The switches a run sets once, from its command line, and every part of a
build reads: whether it may read pages at all, which further readers it
has, what built it, and the counts kept for its closing report. One place,
so that a module holding the code that reads them reads them here
(switches.CACHE_ONLY), and the command sets them here.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import SecondOpinion, ThirdOpinion

# Set by the command line (--second-reader), like WITNESS_WAIVED: a build
# either has a second reader for the whole of its run or has none. A third
# is there where one is installed beside it.
SECOND_OPINION: "SecondOpinion | None" = None
THIRD_OPINION: "ThirdOpinion | None" = None

# --cache-only: this run reads no page. What the cache holds is the book.
CACHE_ONLY = False

# Pages sent to the recogniser on this run, counting a page sent twice twice.
READ_THIS_RUN = {"pages": 0}

# What built the book, for the book to carry: set by the command, written
# into the package as <meta name="codicology:build">.
BUILD_NOTE: "str | None" = None

# Set when the operator has said, in as many words, to run without a witness.
# Nothing consults it to decide anything; it exists so the count of checks
# that went unwitnessed can be reported at the end of the run instead of the
# checks quietly not happening.
WITNESS_WAIVED = False

UNWITNESSED = {"fabrication": 0, "blank": 0}
