"""How many pages are read at once is one number, not two.

The batch the pipeline sends and the slots the inference server holds must
move together: a slot with no page in it is only memory, and a page with
no slot waits. The measured knee on an M1 Pro is 4 — nine percent per page
up to it, nothing past it — but that is a fact about one machine, which is
why the number is reachable from the command line.
"""
import os

import pytest

from codicology import pipeline as vtb


@pytest.fixture(autouse=True)
def restore():
    was_env = os.environ.get("SURYA_INFERENCE_PARALLEL")
    was_batch = vtb.SuryaBackend.batch_size
    yield
    vtb.SuryaBackend.batch_size = was_batch
    if was_env is None:
        os.environ.pop("SURYA_INFERENCE_PARALLEL", None)
    else:
        os.environ["SURYA_INFERENCE_PARALLEL"] = was_env


def test_both_halves_move_together(vtb_=None):
    vtb.set_pages_in_flight(12)
    assert vtb.SuryaBackend.batch_size == 12
    assert os.environ["SURYA_INFERENCE_PARALLEL"] == "12"


def test_the_default_is_the_measured_knee(vtb_=None):
    assert vtb.SuryaBackend.batch_size == 4
    assert os.environ.get("SURYA_INFERENCE_PARALLEL") == "4"


def test_the_flag_reaches_both(vtb_=None):
    parser_args = ["--pages-from", "x.pdf", "--batch-size", "2"]
    with pytest.raises(SystemExit):
        # no such PDF: the run refuses, but only after the flag is applied
        vtb.main(parser_args)
    assert vtb.SuryaBackend.batch_size == 2
    assert os.environ["SURYA_INFERENCE_PARALLEL"] == "2"


def test_zero_pages_in_flight_is_refused(vtb_=None):
    with pytest.raises(SystemExit):
        vtb.main(["--pages-from", "x.pdf", "--batch-size", "0"])
    assert vtb.SuryaBackend.batch_size == 4, "the refusal changes nothing"
