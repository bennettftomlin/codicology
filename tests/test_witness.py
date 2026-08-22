"""The classical witness is required, and its absence is never silent.

Two checks rest entirely on tesseract. One asks whether a page with no text
layer to consult was invented — on photographs, video and image-only scans it
is the only witness there is. The other asks whether a page about to be
deleted as blank was merely one the pipeline failed to read, which nothing
else can tell apart and which applies to every source.

When it was optional, both simply evaluated to False and the page went
through unexamined with nothing said. That is the failure these tests exist
to prevent: not the absence of the witness, which an operator may knowingly
accept, but the absence going unmentioned.
"""
import pytest


def test_the_witness_is_detected_by_the_binary_not_the_package(vtb, monkeypatch):
    """pytesseract is a different thing — it drives the tesseract OCR
    backend. The witness is the executable, which no pip install provides."""
    monkeypatch.setattr(vtb.shutil, "which", lambda name: "/usr/bin/" + name)
    assert vtb.witness_available() is True
    monkeypatch.setattr(vtb.shutil, "which", lambda name: None)
    assert vtb.witness_available() is False


def test_a_missing_witness_yields_no_count_not_a_zero(vtb, monkeypatch, tmp_path):
    """None and 0 mean opposite things here: "nobody looked" versus "looked
    and found nothing". Conflating them is what let pages through."""
    monkeypatch.setattr(vtb.shutil, "which", lambda name: None)
    page = str(tmp_path / "p.png")
    open(page, "wb").close()
    assert vtb._classical_word_count(page) is None


def test_unwitnessed_pages_are_counted(vtb):
    """The counters are what make the silence impossible; they are read at
    the end of a run and printed."""
    before = dict(vtb.UNWITNESSED)
    vtb.UNWITNESSED["fabrication"] += 1
    vtb.UNWITNESSED["blank"] += 2
    try:
        assert vtb.UNWITNESSED["fabrication"] == before["fabrication"] + 1
        assert vtb.UNWITNESSED["blank"] == before["blank"] + 2
    finally:
        vtb.UNWITNESSED.update(before)


def test_the_run_refuses_to_start_without_a_witness(vtb, monkeypatch):
    monkeypatch.setattr(vtb.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as e:
        vtb.main(["--pages-from", "/nonexistent.pdf", "--epub", "/tmp/x.epub"])
    msg = str(e.value)
    assert "tesseract" in msg
    assert "brew install tesseract" in msg, "the fix is not in the message"
    assert "--without-witness" in msg, "the escape hatch is not offered"


def test_the_escape_hatch_gets_past_the_gate(vtb, monkeypatch, capsys):
    """It must not be quiet about it: an operator who waives the witness
    should see so before the run, not only in the summary afterwards."""
    monkeypatch.setattr(vtb.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as e:
        vtb.main(["--pages-from", "/nonexistent.pdf", "--epub", "/tmp/x.epub",
                  "--without-witness"])
    # it got past the witness gate and died on the missing source instead
    assert "PDF not found" in str(e.value)
    out = capsys.readouterr().out
    assert "WITHOUT a witness" in out


def test_an_untested_backend_says_so_before_it_is_used(vtb, capsys):
    """Surya is the only backend this pipeline was built for. The others
    return flat text, so a book built on one loses figures, headings,
    contents and note links with nothing downstream noticing."""
    with pytest.raises(SystemExit):
        vtb.load_backend("easyocr", ["en"])
    out = capsys.readouterr().out
    assert "never been run" in out
    assert "untested" in out


def test_surya_is_not_warned_about(vtb, capsys, monkeypatch):
    class Fake:
        def __init__(self, langs):
            pass
    monkeypatch.setitem(vtb.OCR_BACKENDS, "surya", Fake)
    vtb.load_backend("surya", ["en"])
    assert capsys.readouterr().out == ""


def test_compare_survives_a_malformed_byte(vtb, tmp_path):
    """One bad byte in one page file degrades that page's words to a
    replacement character; it must not crash the whole audit (R3) — the
    other two extraction implementations already behaved this way."""
    import zipfile
    from codicology import compare
    p = tmp_path / "b.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("EPUB/page_0000.xhtml",
                   b"<html><body><p>fine text</p></body></html>")
        z.writestr("EPUB/page_0001.xhtml",
                   b"<html><body><p>bro\xffken byte</p></body></html>")
    pages = compare.epub_pages(str(p))
    assert "fine" in pages[0][0]
    assert any("ken" in w or "bro" in w for w in pages[1][0])


def test_entities_read_the_same_through_compare_and_adjudicate(vtb, tmp_path):
    """The printed &c. is stored as &amp;c.; compare unescaped it and
    adjudicate did not, so the two audits tokenized different books —
    adjudicate's side grew a phantom 'amp' (R3, second half). Both
    extraction paths must now hand the readers the printed text."""
    import zipfile
    from codicology import compare
    from codicology.adjudicate import epub_page_texts
    p = tmp_path / "b.epub"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("EPUB/page_0000.xhtml",
                   "<html><body><p>letters, &amp;c. by A &amp; B</p>"
                   "</body></html>")
    text = epub_page_texts(str(p))[0]
    assert "&c." in text and "amp" not in text
    body_words, _ = compare.epub_pages(str(p))[0]
    assert "amp" not in body_words and "letters" in body_words
