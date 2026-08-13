"""
Codicology OCR — a Calibre plugin.

Hands a book's PDF to the codicology pipeline and files the resulting EPUB
back into the library. The pipeline itself is NOT bundled: it needs torch
and OpenCV, which have no wheels for the Python 3.14 Calibre embeds, so it
lives in its own environment and is driven as a subprocess.

Phase 1: real conversions, one book at a time. The environment is supplied
by hand — the configured path, or a probe of the usual places; the setup
wizard that builds one arrives in Phase 2.
"""
from calibre.customize import InterfaceActionBase

__version__ = (0, 1, 0)


class CodicologyOCRPlugin(InterfaceActionBase):
    name = 'Codicology OCR'
    description = ('Convert a PDF to a well-made EPUB with the codicology '
                   'pipeline: OCR, printed page numbers, linked notes, and '
                   'a completeness check after every build')
    supported_platforms = ['osx']      # linux, windows once D6 escalates
    author = 'Bennett Tomlin'
    version = __version__
    minimum_calibre_version = (6, 0, 0)
    icon = 'images/icon.png'

    actual_plugin = 'calibre_plugins.codicology_ocr.ui:CodicologyOCRAction'

    def is_customizable(self):
        return True

    def config_widget(self):
        from calibre_plugins.codicology_ocr.config import ConfigWidget
        return ConfigWidget()

    def save_settings(self, config_widget):
        config_widget.save_settings()
