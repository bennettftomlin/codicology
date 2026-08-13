"""
Settings. Phase 0 carries only what the spike needs — where codicology
lives, and which interpreter to drive the stub with.

The full option set (OCR backend, language, link notes, check folios, and
the rest) arrives with the real pre-flight dialog in Phase 1.
"""
from calibre.utils.config import JSONConfig
from qt.core import (QCheckBox, QFormLayout, QLabel, QLineEdit, QVBoxLayout,
                     QWidget)

prefs = JSONConfig('plugins/codicology_ocr')

# Empty means "probe for it". The escape hatch that always wins, per D3.
prefs.defaults['codicology_path'] = ''
prefs.defaults['spike_python'] = ''
prefs.defaults['verify_after_build'] = True


class ConfigWidget(QWidget):

    def __init__(self):
        QWidget.__init__(self)
        layout = QVBoxLayout(self)

        blurb = QLabel(
            'Codicology runs in its own Python environment — it needs torch '
            'and OpenCV, which Calibre\'s interpreter cannot load. Leave the '
            'path blank to search for it.')
        blurb.setWordWrap(True)
        layout.addWidget(blurb)

        form = QFormLayout()
        self.path_edit = QLineEdit(prefs['codicology_path'])
        self.path_edit.setPlaceholderText('search for it')
        form.addRow('codicology:', self.path_edit)

        self.python_edit = QLineEdit(prefs['spike_python'])
        self.python_edit.setPlaceholderText('search for a usable Python')
        form.addRow('Spike interpreter:', self.python_edit)
        layout.addLayout(form)

        self.verify_box = QCheckBox('Check the finished EPUB for missing pages')
        self.verify_box.setChecked(bool(prefs['verify_after_build']))
        layout.addWidget(self.verify_box)
        layout.addStretch(1)

    def save_settings(self):
        prefs['codicology_path'] = self.path_edit.text().strip()
        prefs['spike_python'] = self.python_edit.text().strip()
        prefs['verify_after_build'] = self.verify_box.isChecked()
