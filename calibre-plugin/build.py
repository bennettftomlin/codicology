"""
Build the plugin ZIP, and optionally install it.

    python3 calibre-plugin/build.py            # writes codicology-ocr.zip
    python3 calibre-plugin/build.py --install  # and adds it to Calibre

Contents of plugin/ become the ZIP root — Calibre expects __init__.py and
plugin-import-name-*.txt at the top level, not inside a directory.
"""
import argparse
import os
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'plugin')
OUT = os.path.join(HERE, 'codicology-ocr.zip')

CALIBRE_CUSTOMIZE = (
    '/Applications/calibre.app/Contents/MacOS/calibre-customize',
    'calibre-customize',
)
SKIP = {'__pycache__', '.DS_Store'}


def build():
    count = 0
    with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk(SRC):
            dirs[:] = [d for d in dirs if d not in SKIP]
            for name in sorted(files):
                if name in SKIP or name.endswith('.pyc'):
                    continue
                path = os.path.join(root, name)
                z.write(path, os.path.relpath(path, SRC))
                count += 1
    print(f'wrote {OUT} ({count} files, {os.path.getsize(OUT)} bytes)')
    return OUT


def install(path):
    for exe in CALIBRE_CUSTOMIZE:
        if os.path.exists(exe) or not os.path.isabs(exe):
            try:
                r = subprocess.run([exe, '-a', path], capture_output=True,
                                   text=True)
            except FileNotFoundError:
                continue
            print(r.stdout.strip() or r.stderr.strip())
            return r.returncode
    print('calibre-customize not found', file=sys.stderr)
    return 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--install', action='store_true')
    args = p.parse_args()
    zip_path = build()
    sys.exit(install(zip_path) if args.install else 0)
