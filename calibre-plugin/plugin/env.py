"""
Finding codicology, and asking it about itself.

Resolution order, per the plan's D3: the configured path always wins as the
escape hatch; then well-known directories are probed, because a
Finder-launched Calibre inherits launchd's PATH, which contains almost
nothing a user installed. The managed environment slots in between those
two when Phase 2 builds it.

Everything here talks to the executable through runner.child_env(), so the
probe sees the same world a conversion will.
"""
import json
import os
import subprocess

from calibre_plugins.codicology_ocr import runner
from calibre_plugins.codicology_ocr.config import prefs

# Where a person plausibly has codicology: pipx and pip --user, both
# Homebrew prefixes, pyenv. A venv of their own is what the configured
# path is for.
CANDIDATE_DIRS = (
    "~/.local/bin",
    "/opt/homebrew/bin",
    "/usr/local/bin",
    "~/.pyenv/shims",
)


def resolve_codicology():
    """Full path to a codicology executable, or None."""
    configured = (prefs["codicology_path"] or "").strip()
    if configured:
        path = os.path.expanduser(configured)
        return path if os.access(path, os.X_OK) else None
    for d in CANDIDATE_DIRS:
        path = os.path.join(os.path.expanduser(d), "codicology")
        if os.access(path, os.X_OK):
            return path
    return None


def _run(argv, timeout):
    return subprocess.run(
        [str(a) for a in argv], capture_output=True, text=True,
        timeout=timeout, env=runner.child_env())


def codicology_version(exe, timeout=20):
    """'codicology 0.1.0' → '0.1.0', or None if the executable is not ours."""
    try:
        r = _run([exe, "--version"], timeout)
    except Exception:
        return None
    out = (r.stdout or "").strip()
    return out.split()[-1] if r.returncode == 0 and out.startswith("codicology") else None


def quick_doctor(exe, timeout=30):
    """
    The fast readiness report — binaries, packages, languages. Returns the
    parsed dict, or None when this codicology predates the doctor verb (the
    conversion itself still gates on the witness, so proceeding is safe,
    just less legible).
    """
    try:
        r = _run([exe, "doctor", "--json"], timeout)
        return json.loads(r.stdout)
    except Exception:
        return None


def cache_dir(library_id):
    """OCR caches live under Calibre's config dir, keyed by library."""
    from calibre.constants import config_dir
    safe = "".join(c if c.isalnum() or c in "-_" else "_"
                   for c in str(library_id or "default"))
    path = os.path.join(config_dir, "plugins", "codicology_ocr",
                        "cache", safe)
    os.makedirs(path, exist_ok=True)
    return path


def cache_path(library_id, book_id, pdf_path):
    """
    <config>/plugins/codicology_ocr/cache/<library>/<book>-<pdfhash8>.ocr.gz

    The hash is of the PDF's bytes, so replacing the format cannot silently
    reuse OCR read from the file it replaced.
    """
    import hashlib
    h = hashlib.sha256()
    with open(pdf_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return os.path.join(cache_dir(library_id),
                        f"{book_id}-{h.hexdigest()[:8]}.ocr.gz")


def pdf_page_count(pdf_path):
    """Via Calibre's bundled podofo; None when it cannot say."""
    try:
        from calibre.utils.podofo import get_podofo
        doc = get_podofo().PDFDoc()
        doc.open(pdf_path)
        return doc.page_count()
    except Exception:
        return None
