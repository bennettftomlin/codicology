"""
Phase 0, risk 2 (the half that can be checked without a GUI): does the
installed ZIP load, and can it get its own files back out?

    /Applications/calibre.app/Contents/MacOS/calibre-debug \
        -e calibre-plugin/spike/check_plugin.py

What remains after this is one click, described in VERIFY.md.
"""
import sys

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    if detail:
        print(f'        {detail}')


def main():
    from calibre.customize.ui import initialized_plugins, find_plugin

    plugin = find_plugin('Codicology OCR')
    check('the plugin is registered with Calibre', plugin is not None,
          f'{len(list(initialized_plugins()))} plugins loaded')
    if plugin is None:
        return 1

    check('version and metadata are readable',
          plugin.version == (0, 0, 1) and plugin.author == 'Bennett Tomlin',
          f'{plugin.name} {plugin.version} — {plugin.author}')
    check('it declares an InterfaceAction',
          plugin.actual_plugin.endswith('ui:CodicologyOCRAction'),
          plugin.actual_plugin)
    check('it is loaded from the installed ZIP, not a checkout',
          plugin.plugin_path and plugin.plugin_path.endswith('.zip'),
          str(plugin.plugin_path))

    # ── the modules inside the ZIP import ───────────────────────────────
    import calibre_plugins.codicology_ocr.runner as runner
    check('runner imports from inside the ZIP', hasattr(runner, 'run'),
          runner.__file__)
    check('it strips the variables Calibre aims at itself',
          not [k for k in runner.SCRUB if k in runner.child_env()],
          f'would strip: {", ".join(runner.scrubbed()) or "none set"}')

    from calibre_plugins.codicology_ocr.config import prefs
    check('settings load and have defaults',
          prefs['verify_after_build'] is True,
          f'codicology_path={prefs["codicology_path"]!r}')

    # ── resources come back out of the ZIP ──────────────────────────────
    from calibre.customize.ui import plugin_for_catalog_format  # noqa: F401
    data = plugin.load_resources(['spike_stub.py', 'images/icon.png'])
    check('the stub can be pulled out of the ZIP',
          data.get('spike_stub.py', b'').startswith(b'"""'),
          f'{len(data.get("spike_stub.py", b""))} bytes')
    check('the icon can be pulled out of the ZIP',
          data.get('images/icon.png', b'')[:8] == b'\x89PNG\r\n\x1a\n',
          f'{len(data.get("images/icon.png", b""))} bytes')

    # ── the action class is importable and shaped right ─────────────────
    from calibre_plugins.codicology_ocr.ui import CodicologyOCRAction as A
    check('the toolbar button says OCR PDF', A.action_spec[0] == 'OCR PDF',
          f'{A.action_spec[0]!r}, icon {A.action_spec[1]!r}')
    check('it has a tooltip', bool(A.action_spec[2]), A.action_spec[2])
    check('the worker is importable outside the GUI thread',
          callable(sys.modules['calibre_plugins.codicology_ocr.ui']._spike_worker))

    # ── the worker, driven with Calibre's own job primitives ────────────
    # ThreadedJob hands the worker a Queue, an Event and a Log. Supplying
    # those directly exercises everything except Calibre drawing the bar.
    print()
    import queue
    import threading
    from calibre.ptempfile import PersistentTemporaryFile
    from calibre_plugins.codicology_ocr.ui import _spike_worker, find_python

    python = find_python()
    check('a usable non-Calibre interpreter was found', bool(python), python)
    if not python:
        return 1

    pt = PersistentTemporaryFile('_stub.py')
    pt.write(data['spike_stub.py'])
    pt.close()

    try:
        from calibre.utils.logging import GUILog as Log
    except ImportError:
        from calibre.utils.logging import Log
    log, notes, abort = Log(), queue.Queue(), threading.Event()
    out = _spike_worker(python, pt.name, '/tmp/spike-plugin.epub', 12,
                        abort=abort, log=log, notifications=notes)

    sent = []
    while not notes.empty():
        sent.append(notes.get())
    check('the worker returned a result', out.get('pages') == 12, str(out))
    check('it pushed progress notifications', len(sent) >= 3,
          f'{len(sent)} sent, e.g. {sent[0] if sent else None}')
    check('each is the (fraction, message) pair Calibre expects',
          all(isinstance(f, float) and 0 <= f <= 1 and isinstance(m, str)
              for f, m in sent),
          f'last: {sent[-1] if sent else None}')
    check('the fraction advances monotonically to 1.0',
          [f for f, _ in sent] == sorted(f for f, _ in sent)
          and sent[-1][0] == 1.0)
    check('the human log was captured for the job viewer',
          'OCR backend' in log.plain_text,
          f'{len(log.plain_text)} chars')

    # cancellation, through the same path the Cancel button uses
    abort.set()
    try:
        _spike_worker(python, pt.name, '/tmp/spike-plugin.epub', 400,
                      abort=abort, log=Log(), notifications=queue.Queue())
        check('an aborted worker raises rather than returning', False)
    except Exception as exc:
        check('an aborted worker raises rather than returning',
              type(exc).__name__ == 'Aborted', type(exc).__name__)

    bad = [n for n, ok, _ in results if not ok]
    print(f'\n{len(results) - len(bad)}/{len(results)} passed')
    if bad:
        print('FAILED: ' + '; '.join(bad))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
