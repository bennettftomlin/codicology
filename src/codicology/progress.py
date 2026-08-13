"""Structured progress, for a program driving this pipeline through a pipe.

`codicology convert --progress-json` emits one JSON object per line on
stderr — stdout stays the human-readable log. A line is an event only if
it parses as a JSON object carrying an "event" key, so warnings and
tracebacks can share the stream without ambiguity.

The vocabulary is deliberately small, because the consumer is a GUI on the
far side of a pipe and fails silently when a name changes:

    {"event": "phase",    "phase": "ocr", "message": "OCR backend: surya"}
    {"event": "progress", "phase": "ocr", "done": 40, "total": 312,
     "note": "12 from cache"}
    {"event": "result",   "epub": "out.epub", "pdf": "out.pdf"}
    {"event": "error",    "message": "..."}

Adding a field or an event is safe; renaming one is a breaking change to
whoever is listening (the Calibre plugin is).
"""
import json
import sys

_enabled = False


def enable() -> None:
    global _enabled
    _enabled = True


def disable() -> None:
    """For tests; the flag is process-global."""
    global _enabled
    _enabled = False


def enabled() -> bool:
    return _enabled


def emit(**event) -> None:
    """One event on stderr, flushed immediately. A no-op unless enabled."""
    if not _enabled:
        return
    sys.stderr.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    sys.stderr.flush()
