"""
Settings, and the preferences widget.

One setting matters more than the rest: where codicology lives. It is the
escape hatch that always wins (D3) — anyone with their own environment
points here and never meets the managed one Phase 2 adds.
"""
from calibre.utils.config import JSONConfig
from qt.core import (QCheckBox, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                     QPushButton, QVBoxLayout, QWidget)

prefs = JSONConfig('plugins/codicology_ocr')

prefs.defaults['codicology_path'] = ''      # empty: probe well-known dirs
prefs.defaults['verify_after_build'] = True
prefs.defaults['adjudicate_after_build'] = True
prefs.defaults['review_sheet_after_build'] = True
prefs.defaults['open_review_sheet'] = True
prefs.defaults['lang'] = 'en'
# The research conveniences default ON: linkers refuse rather than guess,
# so on a book without the apparatus they cost nothing, and on a book
# with it they are the point.
prefs.defaults['link_notes'] = True
prefs.defaults['link_citations'] = True
prefs.defaults['typography'] = True
prefs.defaults['link_index'] = True
prefs.defaults['check_folios'] = True
prefs.defaults['embed_images'] = False
prefs.defaults['keep_blank_pages'] = False
prefs.defaults['extra_flags'] = ''


class ConfigWidget(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        layout = QVBoxLayout(self)

        blurb = QLabel(
            'Codicology runs in its own Python environment — it needs torch '
            'and OpenCV, which Calibre\'s interpreter cannot load. Point '
            'this at the codicology executable inside that environment '
            '(…/venv/bin/codicology), or leave it blank to search the '
            'usual places.')
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        row = QHBoxLayout()
        self.path_edit = QLineEdit(prefs['codicology_path'])
        self.path_edit.setPlaceholderText('search ~/.local/bin, /opt/homebrew/bin, …')
        row.addWidget(self.path_edit)
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        form.addRow('codicology:', row)
        layout.addLayout(form)

        check = QPushButton('Check environment')
        check.setToolTip('Runs codicology doctor: binaries, packages, '
                         'tesseract and its languages')
        check.clicked.connect(self._check)
        layout.addWidget(check)

        self.verify_box = QCheckBox(
            'After every build, check the EPUB against the source for '
            'missing pages (codicology verify)')
        self.verify_box.setChecked(bool(prefs['verify_after_build']))
        layout.addWidget(self.verify_box)

        self.adjudicate_box = QCheckBox(
            'After every build, re-read the book with the witness engines '
            'and save the dispute record (codicology adjudicate)')
        self.adjudicate_box.setToolTip(
            'Every word the OCR engines read differently, and which rule '
            'settled it. Adds several minutes per book; never changes the '
            'EPUB. The record is saved beside the OCR cache.')
        self.adjudicate_box.setChecked(bool(prefs['adjudicate_after_build']))
        layout.addWidget(self.adjudicate_box)

        self.review_sheet_box = QCheckBox(
            'Also render the dispute record as a review sheet '
            '(codicology review)')
        self.review_sheet_box.setToolTip(
            'A single HTML file beside the cache: the ink itself next to '
            'every disputed word, ranked by how likely the shipped reading '
            'is wrong, with a field for your own reading — which outranks '
            'every rule. Export your decisions from the sheet and drop the '
            'file beside the cache; the next rebuild applies them. Needs '
            'the dispute record above.')
        self.review_sheet_box.setChecked(
            bool(prefs['review_sheet_after_build']))
        layout.addWidget(self.review_sheet_box)

        self.open_sheet_box = QCheckBox(
            'Open the review sheet in the browser when a build finishes')
        self.open_sheet_box.setToolTip(
            'Only when the readers actually disagreed about something. '
            'The sheet opens as a local file so your staged decisions '
            'survive closing the tab.')
        self.open_sheet_box.setChecked(bool(prefs['open_review_sheet']))
        layout.addWidget(self.open_sheet_box)
        layout.addStretch(1)

    def _browse(self):
        from qt.core import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, 'Locate the codicology executable')
        if path:
            self.path_edit.setText(path)

    def _check(self):
        from calibre.gui2 import error_dialog, info_dialog
        # Resolution must see the path as typed, not as last saved.
        prefs['codicology_path'] = self.path_edit.text().strip()
        from calibre_plugins.codicology_ocr.env import (quick_doctor,
                                                        resolve_codicology)
        exe = resolve_codicology()
        if not exe:
            return error_dialog(
                self, 'codicology not found',
                'Nothing executable at the configured path, and nothing '
                'in the usual places. Install codicology in its own '
                'environment and point the path at its executable.',
                show=True)
        report = quick_doctor(exe)
        if report is None:
            return error_dialog(
                self, 'No answer from codicology',
                f'{exe} did not answer `doctor --json` — it may be an '
                'older codicology. Update it; the plugin depends on '
                'doctor to know the environment is ready.', show=True)
        lines = [f"codicology {report.get('codicology')} on "
                 f"Python {report.get('python')}", ""]
        tess = report.get('tesseract') or {}
        lines.append('tesseract: ' + (tess.get('version', 'present')
                                      if tess.get('present') else 'MISSING'))
        llama = report.get('llama_server') or {}
        lines.append('llama-server: '
                     + (llama.get('path') or 'not found'))
        for w in report.get('warnings', []):
            lines.append(f'⚠ {w}')
        for p in report.get('problems', []):
            lines.append(f'✗ {p}')
        text = '\n'.join(lines)
        if report.get('ok'):
            info_dialog(self, 'Environment ready',
                        text + '\n\nReady to convert.', show=True)
        else:
            error_dialog(self, 'Environment not ready', text, show=True)

    def save_settings(self):
        prefs['codicology_path'] = self.path_edit.text().strip()
        prefs['verify_after_build'] = self.verify_box.isChecked()
        prefs['adjudicate_after_build'] = self.adjudicate_box.isChecked()
        prefs['review_sheet_after_build'] = self.review_sheet_box.isChecked()
        prefs['open_review_sheet'] = self.open_sheet_box.isChecked()
