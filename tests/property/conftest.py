"""Property tests: rules that must hold for any page, checked on generated
ones (Hypothesis).

They run with the rest of the suite at the "quick" setting. For a long
search set CODICOLOGY_PROPERTY=deep, e.g.

    CODICOLOGY_PROPERTY=deep python -m pytest tests/property -q -p no:cacheprovider

A failure Hypothesis finds is shrunk to a small case and printed; turn it into
an ordinary named test beside the fix, so it stays checked whatever the
generators later become.
"""
import os

from hypothesis import HealthCheck, settings

settings.register_profile(
    "quick", max_examples=150, deadline=None,
    suppress_health_check=[HealthCheck.too_slow])
settings.register_profile(
    "deep", max_examples=5000, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much])
settings.load_profile(os.environ.get("CODICOLOGY_PROPERTY", "quick"))
