"""
The toolbar action: OCR PDF.

Select a book with a PDF, press the button, answer the pre-flight dialog,
and the conversion runs as a Calibre job — live progress, viewable log,
working cancel. On success the EPUB is added to the book and verify's
verdict is reported, loudly when it is bad.

Phase 1 scope: one book at a time, environment supplied by hand (the
configured path or a probe of the usual places). The setup wizard and
batch queueing are Phase 2.
"""
import os
import shlex

from calibre.gui2 import error_dialog, info_dialog, warning_dialog
from calibre.gui2.actions import InterfaceAction
from calibre.gui2.threaded_jobs import ThreadedJob
from calibre.ptempfile import PersistentTemporaryFile

from calibre_plugins.codicology_ocr.config import prefs
from calibre_plugins.codicology_ocr.jobs import convert_worker


class CodicologyOCRAction(InterfaceAction):

    name = 'Codicology OCR'
    # (text, icon, tooltip, keyboard shortcut)
    action_spec = ('OCR PDF', 'images/icon.png',
                   'Convert this book\'s PDF to an EPUB with codicology',
                   None)
    action_type = 'current'

    def genesis(self):
        # get_icons is injected into the plugin namespace by Calibre's
        # loader — a bare name, undefined outside a running Calibre.
        icon = get_icons('images/icon.png')          # noqa: F821
        if icon:
            self.qaction.setIcon(icon)
        self.qaction.triggered.connect(self.start)

    # ── entry point ─────────────────────────────────────────────────────
    def start(self):
        from calibre_plugins.codicology_ocr import env

        db = self.gui.current_db.new_api
        book_id = self._selected_book()
        if book_id is None:
            return error_dialog(self.gui, 'No book selected',
                                'Select the book whose PDF should become '
                                'an EPUB.', show=True)
        pdf = db.format_abspath(book_id, 'PDF')
        if not pdf:
            return error_dialog(self.gui, 'No PDF in this book',
                                'This book has no PDF format for '
                                'codicology to read.', show=True)

        exe = env.resolve_codicology()
        if not exe:
            return error_dialog(
                self.gui, 'codicology not found',
                'No codicology executable at the configured path or in '
                'the usual places.\n\nInstall it in its own Python '
                'environment, then point the plugin at it: Preferences → '
                'Plugins → Codicology OCR → Customize.', show=True)

        # The cheap gate before an expensive job. Doctor knows what the
        # pipeline needs; the plugin only relays its reasons.
        doctor = env.quick_doctor(exe)
        if doctor is not None and not doctor.get('ok', False):
            problems = '\n'.join('• ' + p for p in doctor.get('problems', []))
            return error_dialog(
                self.gui, 'The codicology environment is not ready',
                'codicology doctor reports:\n\n' + problems +
                '\n\nFix these, or run "Check environment" in the '
                'plugin\'s preferences after changing the path.',
                show=True)

        title = db.field_for('title', book_id) or 'Scanned Book'
        cache = env.cache_path(getattr(db, 'library_id', 'default'),
                               book_id, pdf)

        from calibre_plugins.codicology_ocr.dialogs import PreflightDialog
        dlg = PreflightDialog(
            self.gui, title, pdf, env.pdf_page_count(pdf),
            cache_exists=os.path.exists(cache),
            has_epub=db.has_format(book_id, 'EPUB'),
            doctor=doctor)
        if not dlg.exec():
            return
        opts = dlg.options()

        out = PersistentTemporaryFile('_codicology.epub')
        out.close()

        argv = [exe, 'convert', '--pages-from', pdf, '--epub', out.name,
                '--ocr-cache', cache, '--title', title, '--progress-json']
        if opts['link_notes']:
            argv.append('--link-notes')
        if opts.get('link_citations'):
            argv.append('--link-citations')
        if opts.get('typography'):
            argv.append('--typography')
        if opts.get('link_index'):
            argv.append('--link-index')
        if opts['check_folios']:
            argv.append('--check-folios')
        if opts['embed_images']:
            argv.append('--embed-images')
        if opts['keep_blank_pages']:
            argv.append('--keep-blank-pages')
        if opts['lang'] and opts['lang'] != 'en':
            argv += ['--lang', opts['lang']]
        if opts['extra_flags']:
            argv += shlex.split(opts['extra_flags'])

        verify_argv = ([exe, 'verify', out.name, pdf]
                       if prefs['verify_after_build'] else None)

        job = ThreadedJob(
            'codicology_ocr',
            f'OCR PDF — {title}',
            convert_worker,
            [argv, verify_argv, book_id, out.name, title], {},
            self.Dispatcher(self.finished))
        self.gui.job_manager.run_threaded_job(job)
        self.gui.status_bar.show_message(
            f'Codicology: reading "{title}". Progress is in the jobs '
            'list, bottom right.', 5000)

    def _selected_book(self):
        rows = self.gui.library_view.selectionModel().selectedRows()
        if not rows:
            return None
        return self.gui.library_view.model().id(rows[0])

    # ── completion ──────────────────────────────────────────────────────
    def finished(self, job):
        if job.failed:
            exc = getattr(job, 'exception', None)
            if type(exc).__name__ == 'Aborted':
                self.gui.status_bar.show_message('Codicology: cancelled.',
                                                 5000)
                return
            return self.gui.job_manager.launch_job_error_dialog(job,
                                                                self.gui)

        result = job.result or {}
        book_id, epub = result.get('book_id'), result.get('epub')
        title = result.get('title', '')

        db = self.gui.current_db.new_api
        try:
            db.add_format(book_id, 'EPUB', epub)
        finally:
            try:
                os.remove(epub)
            except OSError:
                pass
        self.gui.library_view.model().refresh_ids((book_id,))

        verify = result.get('verify')
        if verify is None:
            return info_dialog(
                self.gui, 'EPUB added',
                f'"{title}" now has a codicology EPUB.\n\n'
                'The after-build check was disabled, so nothing has '
                'looked for missing pages.', show=True)
        if verify['rc'] == 0:
            return info_dialog(
                self.gui, 'EPUB added — no holes found',
                f'"{title}" now has a codicology EPUB, and the check '
                'against the source found nothing missing.',
                det_msg=verify['output'], show=True)
        if verify['rc'] == 1:
            return warning_dialog(
                self.gui, 'EPUB added — LOOK AT THIS',
                f'The EPUB for "{title}" was added, but verify found '
                'pages where the source has text and the EPUB has '
                'none. The detail below names them.',
                det_msg=verify['output'], show=True)
        return warning_dialog(
            self.gui, 'EPUB added — verify could not run',
            f'The EPUB for "{title}" was added, but the after-build '
            f'check itself failed (exit {verify["rc"]}).',
            det_msg=verify['output'], show=True)
