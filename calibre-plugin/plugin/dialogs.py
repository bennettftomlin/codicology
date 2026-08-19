"""
The pre-flight dialog: everything worth knowing before committing hours.

It shows the cost side (page count, whether an OCR cache makes this a
cheap rebuild) and the choice side (the options that change what gets
built). Doctor's warnings surface here too — this is the moment someone
decides whether to start a long job, so this is where "an NVIDIA card is
present but idle" belongs, per the plan's GPU notice.
"""
import os

from qt.core import (QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
                     QLabel, QLineEdit, QVBoxLayout)

from calibre_plugins.codicology_ocr.config import prefs


class PreflightDialog(QDialog):

    def __init__(self, parent, title, pdf_path, pages, cache_exists,
                 has_epub, doctor, adjudicate=None):
        QDialog.__init__(self, parent)
        self.setWindowTitle("OCR PDF — codicology")
        layout = QVBoxLayout(self)

        head = QLabel(f"<b>{title}</b>")
        head.setWordWrap(True)
        layout.addWidget(head)

        mb = os.path.getsize(pdf_path) / (1 << 20)
        pages_part = f"{pages} pages, " if pages else ""
        layout.addWidget(QLabel(
            f"{os.path.basename(pdf_path)} — {pages_part}{mb:.1f} MB"))

        if cache_exists:
            cost = QLabel("An OCR cache exists for this exact PDF — the "
                          "rebuild reuses it and costs minutes, not hours.")
        else:
            cost = QLabel("First read: every page goes through OCR. On CPU "
                          "this is hours for a long book, and the job list "
                          "shows progress page by page.")
        cost.setWordWrap(True)
        layout.addWidget(cost)

        for warning in (doctor or {}).get("warnings", []):
            w = QLabel(f"⚠ {warning}")
            w.setWordWrap(True)
            layout.addWidget(w)

        form = QFormLayout()
        self.lang = QLineEdit(prefs["lang"])
        self.lang.setToolTip("Comma-separated OCR language codes")
        form.addRow("Languages:", self.lang)

        self.link_notes = QCheckBox("Link footnotes and endnotes both ways, "
                                    "where the binding is certain")
        self.link_notes.setChecked(bool(prefs["link_notes"]))
        form.addRow(self.link_notes)

        self.link_citations = QCheckBox("Link in-text citations to the "
                                        "bibliography (author-date and "
                                        "Chicago notes)")
        self.link_citations.setToolTip(
            "Driven from the bibliography itself: only names and years it "
            "actually holds are sought, an ambiguous match binds nothing, "
            "and a page whose words would change is left untouched")
        self.link_citations.setChecked(bool(prefs["link_citations"]))
        form.addRow(self.link_citations)

        self.typography = QCheckBox("Restore directional quotes and collapse "
                                    "double spaces (letters never touched)")
        self.typography.setToolTip(
            "The printed page had curly quotes; OCR flattens them. Spaced "
            "ellipses are left exactly as the book set them")
        self.typography.setChecked(bool(prefs["typography"]))
        form.addRow(self.typography)

        self.link_index = QCheckBox("Link the index's page numbers to their "
                                    "pages")
        self.link_index.setToolTip(
            "Joins the printed index to the page list. An index that "
            "declares paragraph references is believed and left alone")
        self.link_index.setChecked(bool(prefs["link_index"]))
        form.addRow(self.link_index)

        self.check_folios = QCheckBox("Read printed page numbers and build "
                                      "the EPUB's page list")
        self.check_folios.setChecked(bool(prefs["check_folios"]))
        form.addRow(self.check_folios)

        self.embed_images = QCheckBox("Embed the page scans alongside the text")
        self.embed_images.setChecked(bool(prefs["embed_images"]))
        form.addRow(self.embed_images)

        self.keep_blank = QCheckBox("Keep blank pages")
        self.keep_blank.setChecked(bool(prefs["keep_blank_pages"]))
        form.addRow(self.keep_blank)

        # Per-run, not a preference: the standing answer lives in the
        # plugin's settings, and this is for the build where it differs.
        # The rebuild that applies a reviewed decisions file is exactly
        # that build — the record was made before the review, and making
        # it again costs the same minutes to say the same thing.
        self.adjudicate = QCheckBox("Re-read with the witnesses and make a "
                                    "new dispute record")
        self.adjudicate.setChecked(bool(prefs["adjudicate_after_build"])
                                   if adjudicate is None else adjudicate)
        self.adjudicate.setToolTip(
            "Several minutes on a long book. Turn it off when the record "
            "you already reviewed still describes this text — applying "
            "decisions does not change what the readers saw.")
        form.addRow(self.adjudicate)

        self.extra = QLineEdit(prefs["extra_flags"])
        self.extra.setPlaceholderText("--drop-pages r060p1  --cover none  …")
        self.extra.setToolTip("Any codicology convert flag not surfaced "
                              "above; see codicology convert -h")
        form.addRow("Extra flags:", self.extra)
        layout.addLayout(form)

        if has_epub:
            warn = QLabel("⚠ This book already has an EPUB. Finishing "
                          "replaces it.")
            warn.setWordWrap(True)
            layout.addWidget(warn)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok
                                   | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Start conversion")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def options(self):
        """Chosen options, persisted as the defaults for next time."""
        opts = {
            "lang": self.lang.text().strip() or "en",
            "link_notes": self.link_notes.isChecked(),
            "link_citations": self.link_citations.isChecked(),
            "typography": self.typography.isChecked(),
            "link_index": self.link_index.isChecked(),
            "check_folios": self.check_folios.isChecked(),
            "embed_images": self.embed_images.isChecked(),
            "keep_blank_pages": self.keep_blank.isChecked(),
            "extra_flags": self.extra.text().strip(),
        }
        for key, value in opts.items():
            prefs[key] = value
        # deliberately not persisted: a build that skipped the record
        # should not quietly make every later build skip it too
        opts["adjudicate"] = self.adjudicate.isChecked()
        return opts
