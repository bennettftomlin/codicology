"""
The toolbar action.

Phase 0 wires the button to a stub rather than to codicology, because the
thing being proved here is the plumbing: does a job started from this button
report progress while it runs, cancel cleanly, and come back with a result?
Swapping the stub for the real command is a change of argv, not of shape.
"""
import os

from calibre.gui2 import error_dialog, info_dialog
from calibre.gui2.actions import InterfaceAction
from calibre.gui2.threaded_jobs import ThreadedJob
from calibre.utils.ipc.job import BaseJob

from calibre_plugins.codicology_ocr.config import prefs


def _spike_worker(python, stub, epub_out, pages,
                  abort=None, log=None, notifications=None):
    """
    Runs in a worker thread. Everything it touches is either passed in or
    imported here — nothing from the GUI thread crosses into it.
    """
    from calibre_plugins.codicology_ocr import runner

    seen = {'pages': 0, 'total': pages}

    def on_progress(event):
        kind = event.get('event')
        if kind == 'progress':
            done, total = event.get('done', 0), event.get('total') or 1
            seen['pages'] = done
            note = event.get('note')
            phase = {'ocr': 'Reading', 'folios': 'Reading page numbers'}.get(
                event.get('phase'), event.get('phase', ''))
            msg = f'{phase} {done}/{total}' + (f' ({note})' if note else '')
            # Calibre wants a fraction and a label; this is what moves the
            # bar in the job list and the status line.
            notifications.put((done / total, msg))
        elif kind == 'phase':
            log(event.get('message', ''))
        elif kind == 'error':
            log('ERROR: ' + event.get('message', ''))

    def on_log(stream, text):
        log(text)

    result = runner.run(
        [python, stub, 'convert', '--progress-json',
         '--epub', epub_out, '--pages', str(pages), '--delay', '0.25'],
        on_progress=on_progress, on_log=on_log, abort=abort)
    return result


class CodicologyOCRAction(InterfaceAction):

    name = 'Codicology OCR'
    # (text, icon, tooltip, keyboard shortcut)
    action_spec = ('OCR PDF', 'images/icon.png',
                   'Convert this book\'s PDF to an EPUB with codicology', None)
    action_type = 'current'

    def genesis(self):
        # get_icons and get_resources are injected into a plugin's namespace
        # by Calibre's loader — bare names, no import, and undefined outside
        # a running Calibre.
        icon = get_icons('images/icon.png')          # noqa: F821
        if icon:
            self.qaction.setIcon(icon)
        self.qaction.triggered.connect(self.start)

    # ── Phase 0 entry point ─────────────────────────────────────────────
    def start(self):
        python = prefs['spike_python'] or find_python()
        if not python:
            return error_dialog(
                self.gui, 'No usable Python found',
                'The spike needs an interpreter that is not Calibre\'s own. '
                'Set one in the plugin\'s preferences.', show=True)

        # The plugin is imported from inside the ZIP, so __file__ is not a
        # real path and nothing can be executed in place. Resources come out
        # as bytes and go to disk before a subprocess can reach them.
        try:
            stub = self.stub_on_disk()
        except Exception as exc:
            return error_dialog(self.gui, 'Could not unpack the spike stub',
                                str(exc), show=True)

        title = 'the spike'
        rows = self.gui.library_view.selectionModel().selectedRows()
        if rows:
            book_id = self.gui.library_view.model().id(rows[0])
            db = self.gui.current_db.new_api
            title = db.field_for('title', book_id) or title
            pdf = db.format_abspath(book_id, 'PDF')
            self.gui.status_bar.show_message(
                f'PDF for "{title}": {pdf or "none"}', 4000)

        epub_out = os.path.join(
            os.path.expanduser('~'), 'codicology-spike.epub')

        job = ThreadedJob(
            'codicology_ocr_spike',
            f'OCR PDF (spike) — {title}',
            _spike_worker,
            [python, stub, epub_out, 40], {},
            self.Dispatcher(self.finished))
        self.gui.job_manager.run_threaded_job(job)
        self.gui.status_bar.show_message(
            'Codicology: started. Watch the jobs list.', 4000)

    def stub_on_disk(self):
        """
        Write the bundled stub out of the ZIP so a subprocess can run it.

        Phase 1 deletes this: the real pipeline is an executable in its own
        environment, and nothing needs unpacking.
        """
        from calibre.ptempfile import PersistentTemporaryFile
        name = 'spike_stub.py'
        data = self.load_resources([name])[name]
        pt = PersistentTemporaryFile('_codicology_stub.py')
        pt.write(data)
        pt.close()
        return pt.name

    def finished(self, job):
        if job.failed:
            # A cancelled job also lands here; Calibre marks it killed.
            if getattr(job, 'killed', False) or job.duration is None:
                return self.gui.status_bar.show_message(
                    'Codicology: cancelled.', 4000)
            return self.gui.job_manager.launch_job_error_dialog(job, self.gui)
        result = job.result or {}
        info_dialog(
            self.gui, 'Spike finished',
            'The job plumbing works.\n\n'
            f'Pages reported: {result.get("pages")}\n'
            f'Wrote: {result.get("epub")}\n\n'
            'Progress moved while it ran, the log filled up, and the result '
            'came back — so the same shape will carry the real pipeline.',
            show=True)


def find_python():
    """
    An interpreter that is not Calibre's, and not one torch cannot use.

    Phase 1 replaces this with the real resolution order — configured path,
    managed environment, PATH probe, then offer setup.
    """
    import platform
    import subprocess
    want = platform.machine()
    for base in ('/opt/homebrew/bin', '/usr/local/bin', '/usr/bin'):
        for minor in (12, 11, 13, 10):
            path = f'{base}/python3.{minor}'
            if not os.path.exists(path):
                continue
            try:
                got = subprocess.run(
                    [path, '-c', 'import platform;print(platform.machine())'],
                    capture_output=True, text=True, timeout=10).stdout.strip()
            except Exception:
                continue
            if got == want:
                return path
    return ''
