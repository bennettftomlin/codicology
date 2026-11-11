"""A second round of review does not cost the first.

The OCR cache holds a book's reading as it was made, so a rebuild has to
apply every correction again. Importing the second round's decisions used
to replace the first round's file, and the next rebuild came out without
them. Rounds are now kept apart on disk and applied in order; a row can be
taken out of use without being forgotten; and the book as it was built
survives a second in-place apply.
"""
import base64
import io
import json
import os
import re
import zipfile

import pytest
from PIL import Image

from codicology import adjudicate as adj
from codicology import cli, review
from test_decision_sites import _epub, _fix, _tsv

BODY = "<p>to be or not to be, he said; let it be so</p>"


def _exported(tmp_path, name, rows, sheet=None, **more):
    path = tmp_path / name
    data = {"name": "book-disputes", "page_files": 1, "decisions": rows, **more}
    if sheet:
        data["sheet"] = sheet
    path.write_text(json.dumps(data))
    return str(path)


def _read(path):
    with open(path) as fh:
        return json.load(fh)


@pytest.fixture
def target(tmp_path):
    return str(tmp_path / "cache" / "book-disputes.decisions.json")


# ── rounds ───────────────────────────────────────────────────────────────────

def test_the_first_file_is_the_first_round(tmp_path, target):
    src = _exported(tmp_path, "one.json", [_fix(BODY, 1, "he")], sheet="aaa")
    done = review.file_decisions(src, target)
    assert done == {"rows": 1, "round": 1, "replaced": False,
                    "dropped_retired": 0, "retired": 0}
    assert review.decision_rounds(target) == [target]
    assert _read(target)["decisions"] == _read(src)["decisions"]


def test_the_same_sheet_exported_again_replaces_its_round(tmp_path, target):
    """The reviewer changing their mind, not a new round."""
    review.file_decisions(_exported(tmp_path, "a.json", [_fix(BODY, 1, "he")],
                                    sheet="aaa"), target)
    again = _exported(tmp_path, "b.json",
                      [_fix(BODY, 1, "he"), _fix(BODY, 2, "nor")], sheet="aaa")
    done = review.file_decisions(again, target)
    assert done["replaced"] is True and done["round"] == 1
    assert review.decision_rounds(target) == [target]
    assert len(_read(target)["decisions"]) == 2


def test_another_sheets_file_is_a_new_round_and_the_old_one_is_kept(tmp_path,
                                                                    target):
    one = [_fix(BODY, 1, "he")]
    review.file_decisions(_exported(tmp_path, "a.json", one, sheet="aaa"),
                          target)
    done = review.file_decisions(
        _exported(tmp_path, "b.json", [_fix(BODY, 9, "is")], sheet="bbb"),
        target)
    assert done["replaced"] is False and done["round"] == 2
    first = target[:-len(".json")] + ".r1.json"
    assert review.decision_rounds(target) == [first, target]
    assert _read(first)["decisions"] == one
    assert _read(target)["decisions"][0]["new"] == "is"

    review.file_decisions(
        _exported(tmp_path, "c.json", [_fix(BODY, 3, "never")], sheet="ccc"),
        target)
    second = target[:-len(".json")] + ".r2.json"
    assert review.decision_rounds(target) == [first, second, target]
    assert _read(second)["decisions"][0]["new"] == "is"


def test_files_from_before_sheets_named_themselves_are_judged_by_their_rows(
        tmp_path, target):
    """A second round is about other words: the first round's are no
    longer in dispute."""
    review.file_decisions(_exported(tmp_path, "a.json", [_fix(BODY, 1, "he")]),
                          target)
    same = _exported(tmp_path, "b.json",
                     [_fix(BODY, 1, "he"), _fix(BODY, 2, "nor")])
    assert review.file_decisions(same, target)["replaced"] is True
    other = _exported(tmp_path, "c.json", [_fix(BODY, 9, "is")], sheet="new")
    assert review.file_decisions(other, target)["replaced"] is False
    assert len(review.decision_rounds(target)) == 2


def test_a_gap_in_the_numbering_overwrites_nothing(tmp_path, target):
    review.file_decisions(_exported(tmp_path, "a.json", [_fix(BODY, 1, "he")],
                                    sheet="aaa"), target)
    stem = target[:-len(".json")]
    with open(stem + ".r3.json", "w") as fh:
        json.dump({"decisions": [_fix(BODY, 4, "kept")]}, fh)
    review.file_decisions(_exported(tmp_path, "b.json", [_fix(BODY, 9, "is")],
                                    sheet="bbb"), target)
    assert _read(stem + ".r3.json")["decisions"][0]["new"] == "kept"
    assert [os.path.basename(p) for p in review.decision_rounds(target)] == [
        "book-disputes.decisions.r3.json", "book-disputes.decisions.r4.json",
        "book-disputes.decisions.json"]


def test_something_that_is_not_a_decisions_file_is_refused(tmp_path, target):
    bad = tmp_path / "notes.json"
    bad.write_text(json.dumps({"disputes": []}))
    with pytest.raises(ValueError):
        review.file_decisions(str(bad), target)
    assert not os.path.exists(target)


# ── a build applies every round, in order ────────────────────────────────────

def test_a_rebuild_applies_the_earlier_rounds_it_was_not_told_about(vtb,
                                                                    tmp_path,
                                                                    target,
                                                                    capsys):
    """The plugin names one file. Round two was reviewed against round
    one's text, so it only lands if round one has."""
    review.file_decisions(
        _exported(tmp_path, "a.json", [_fix(BODY, 1, "he")], sheet="aaa"),
        target)
    after_one = BODY.replace("to be or", "to he or")
    review.file_decisions(
        _exported(tmp_path, "b.json", [_fix(after_one, 1, "she")], sheet="bbb"),
        target)
    rounds = vtb._decision_rounds(target)
    assert [os.path.basename(p) for p in rounds] == [
        "book-disputes.decisions.r1.json", "book-disputes.decisions.json"]
    # named twice, applied once
    assert vtb._decision_rounds([rounds[0], target]) == rounds

    bodies = [BODY]
    for path in rounds:
        vtb.apply_reviewer_decisions(bodies, path)
    assert bodies == ["<p>to she or not to be, he said; let it be so</p>"]

    alone = [BODY]                  # what the build did before: round two only
    vtb.apply_reviewer_decisions(alone, target)
    assert alone == [BODY]
    assert "stale decision" in capsys.readouterr().out


# ── a row taken out of use ───────────────────────────────────────────────────

def test_a_retired_row_is_not_applied_and_is_said(vtb, tmp_path, target,
                                                  capsys):
    review.file_decisions(
        _exported(tmp_path, "a.json",
                  [_fix(BODY, 1, "he"), _fix(BODY, 2, "nor")], sheet="aaa"),
        target)
    (gone,) = review.retire_decisions(target, [1])
    assert gone["new"] == "he"
    assert [d["new"] for d in _read(target)["decisions"]] == ["nor"]
    assert [d["new"] for d in _read(target)["retired"]] == ["he"]

    bodies = [BODY]
    vtb.apply_reviewer_decisions(bodies, target)
    assert bodies == ["<p>to be nor not to be, he said; let it be so</p>"]
    assert "1 decision(s) in book-disputes.decisions.json are retired" in \
        capsys.readouterr().out


def test_a_later_round_cannot_bring_a_retired_row_back(tmp_path, target):
    wrong = _fix(BODY, 1, "he")
    review.file_decisions(_exported(tmp_path, "a.json", [wrong], sheet="aaa"),
                          target)
    review.retire_decisions(target, [1])
    done = review.file_decisions(
        _exported(tmp_path, "b.json", [dict(wrong), _fix(BODY, 9, "is")],
                  sheet="bbb"), target)
    assert done["dropped_retired"] == 1 and done["rows"] == 1
    now = _read(target)
    assert [d["new"] for d in now["decisions"]] == ["is"]
    assert [d["new"] for d in now["retired"]] == ["he"], "and still on record"


def test_retiring_a_row_that_is_not_there_changes_nothing(tmp_path, target):
    review.file_decisions(_exported(tmp_path, "a.json", [_fix(BODY, 1, "he")]),
                          target)
    before = open(target).read()
    with pytest.raises(ValueError, match="rows 1 to 1"):
        review.retire_decisions(target, [2])
    assert open(target).read() == before


def test_apply_leaves_a_retired_row_alone(tmp_path, capsys):
    epub = _epub(tmp_path, {0: BODY})
    dec = _exported(tmp_path, "d.json", [_fix(BODY, 2, "nor")],
                    retired=[_fix(BODY, 1, "he")])
    review.apply_decisions(epub, dec)
    assert "1 retired and left alone" in capsys.readouterr().out
    page = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert "to be nor not" in page


# ── the book as it was built ─────────────────────────────────────────────────

def test_the_original_survives_a_second_apply_in_place(tmp_path):
    epub = _epub(tmp_path, {0: BODY})
    original = open(epub, "rb").read()
    review.apply_decisions(epub, _exported(tmp_path, "a.json",
                                           [_fix(BODY, 1, "he")]))
    after_one = BODY.replace("to be or", "to he or")
    review.apply_decisions(epub, _exported(tmp_path, "b.json",
                                           [_fix(after_one, 2, "nor")]))
    assert open(epub + ".preapply", "rb").read() == original
    page = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert "to he nor not" in page


def test_entries_no_correction_touched_are_written_back_as_they_were(tmp_path):
    epub = str(tmp_path / "book.epub")
    with zipfile.ZipFile(epub, "w") as z:
        z.writestr("mimetype", "application/epub+zip", zipfile.ZIP_STORED)
        for n, body in ((0, BODY), (1, "<p>untouched prose, page two</p>")):
            info = zipfile.ZipInfo(f"EPUB/page_{n:04d}.xhtml",
                                   date_time=(2026, 9, 12, 8, 30, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, f"<html><body>{body}</body></html>")

    def record(path):
        with zipfile.ZipFile(path) as z:
            return {i.filename: (i.date_time, i.compress_type, i.external_attr,
                                 i.CRC, i.compress_size) for i in z.infolist()}
    before = record(epub)
    review.apply_decisions(epub, _exported(tmp_path, "a.json",
                                           [_fix(BODY, 1, "he")], page_files=2))
    after = record(epub)
    assert list(after) == list(before), "same entries, same order"
    for name in ("mimetype", "EPUB/page_0001.xhtml"):
        assert after[name] == before[name], name
    assert after["EPUB/page_0000.xhtml"][:3] == before["EPUB/page_0000.xhtml"][:3]
    assert after["EPUB/page_0000.xhtml"][3] != before["EPUB/page_0000.xhtml"][3]


# ── the sheet says which record it came from ─────────────────────────────────

def test_an_exported_file_carries_its_sheets_fingerprint():
    report = {"epub": "book.epub", "pdf": "book.pdf", "page_files": 1,
              "disputes": [{"page": 0, "surya": "be", "tesseract": "he",
                            "rung": "abstain", "winner": None, "shipped": "be",
                            "box": None, "occurrence": 0, "before": [],
                            "after": []}],
              "surya_only": []}
    sheet = review.render_sheet(report)
    meta = json.loads(re.search(
        r"<script type='application/json' id='meta'>(.*?)</script>",
        sheet, re.S).group(1))
    assert meta["sheet"] == review._fingerprint(report)
    other = dict(report, disputes=[dict(report["disputes"][0], surya="so")])
    assert review._fingerprint(other) != meta["sheet"]


# ── run crops come from the page the run was read on ─────────────────────────

def test_a_runs_ink_is_cut_from_the_pdf_page_not_the_book_page(tmp_path):
    """Wherever a blank or a duplicate was dropped ahead of it, a book page
    sits later in the PDF. The word crops knew; these did not."""
    pdf = str(tmp_path / "book.pdf")
    Image.new("RGB", (300, 400), (255, 0, 0)).save(
        pdf, "PDF", save_all=True,
        append_images=[Image.new("RGB", (300, 400), (0, 0, 255))])
    report = {"pdf": pdf, "surya_only": [
        {"page": 0, "pdf_page": 1,
         "runs": [{"text": "some words", "n": 2, "box": None}]}]}
    uri = review._run_crops(report)[(0, 0)]
    ink = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    r, g, b = ink.convert("RGB").getpixel((ink.width // 2, ink.height // 2))
    assert b > 200 and r < 60, "the second PDF page is the blue one"


# ── a long run can be deleted ────────────────────────────────────────────────

def test_a_long_runs_text_is_recorded_whole(tmp_path, monkeypatch):
    """Cut at 2,000 characters, the text a "delete run" decision carries no
    longer matched the page, and the deletion could never be applied."""
    words = [chr(97 + k // 26) + chr(97 + k % 26) + "moxen"
             for k in range(260)]              # 260 words nobody printed
    body = f"<p>Real prose here. {' '.join(words)} More real prose.</p>"
    epub = _epub(tmp_path, {0: body})
    pdf = str(tmp_path / "book.pdf")
    Image.new("RGB", (600, 200), "white").save(pdf, "PDF")
    monkeypatch.setattr(adj, "_vision_present", lambda: False)
    monkeypatch.setattr(adj, "_tesseract_present", lambda: True)
    monkeypatch.setattr(adj, "read_tesseract_tsv", lambda png: _tsv(
        "Real prose here. More real prose.".split()))
    report = str(tmp_path / "book-disputes.json")
    assert adj.main(epub, pdf, report=report) == 0
    (run,) = _read(report)["surya_only"][0]["runs"]
    assert run["text"] == " ".join(words) and len(run["text"]) > 2000

    out = review.apply_decisions(epub, _exported(tmp_path, "d.json", [{
        "page": 0, "occurrence": 0, "old": run["text"], "new": "",
        "source": "human", "rung": "surya-only", "kind": "delete_run"}]))
    assert len(out["applied"]) == 1
    page = zipfile.ZipFile(epub).read("EPUB/page_0000.xhtml").decode()
    assert "moxen" not in page and "Real prose here." in page


# ── the verb ─────────────────────────────────────────────────────────────────

def test_the_decisions_verb_files_lists_and_retires(tmp_path, target, capsys):
    one = _exported(tmp_path, "a.json", [_fix(BODY, 1, "he")], sheet="aaa")
    two = _exported(tmp_path, "b.json",
                    [_fix(BODY, 9, "is"), _fix(BODY, 2, "nor")], sheet="bbb")
    assert cli.main(["decisions", "file", one, target]) == 0
    assert "1 decision(s) filed as round 1" in capsys.readouterr().out
    assert cli.main(["decisions", "file", two, target, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["round"] == 2

    assert cli.main(["decisions", "list", target]) == 0
    listed = capsys.readouterr().out
    assert "2 decision(s), the last of 2 rounds" in listed
    assert "    1  p0 'it' -> 'is'" in listed

    assert cli.main(["decisions", "retire", target, "1"]) == 0
    assert "retired: p0 'it' -> 'is'" in capsys.readouterr().out
    cli.main(["decisions", "list", target])
    assert "  ret  p0 'it' -> 'is'" in capsys.readouterr().out

    assert cli.main(["decisions", "retire", target, "7"]) == 2
    assert cli.main(["decisions", "file", str(tmp_path / "absent.json"),
                     target]) == 2
