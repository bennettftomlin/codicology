#!/usr/bin/env python3
"""Write constraints.txt: the versions this pipeline is known to work with.

    python tools/lock_env.py                 # surya, geometry and dev extras
    python tools/lock_env.py surya           # only what the surya extra needs
    pip install -c constraints.txt -e ".[surya]"

Surya pins a handful of its dependencies and leaves the rest open:
`transformers` has a floor and no ceiling, and it is the one most likely to
break the recogniser. pyproject.toml deliberately stays loose, since it
describes what the code needs; this file records what was actually
installed the last time the shelf was built and the tests passed.

It is the closure of codicology's own requirements, with the extras named,
taken from the metadata of the environment this is run in. Nothing is
resolved or downloaded. Run it in the environment that works, after
`codicology doctor` is clean, and commit the result.
"""
import importlib.metadata as md
import platform
import re
import sys

from packaging.markers import default_environment
from packaging.requirements import Requirement

HERE = __file__.rsplit("/tools/", 1)[0]


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def closure(root: str, extras: set) -> dict:
    """{normalised name: version} for `root`[extras] and all it requires."""
    found: dict = {}
    todo = [(root, frozenset(extras))]
    seen = set()
    while todo:
        name, wanted = todo.pop()
        if (_norm(name), wanted) in seen:
            continue
        seen.add((_norm(name), wanted))
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            print(f"  [~] {name} is required but not installed; left out",
                  file=sys.stderr)
            continue
        found[_norm(dist.metadata["Name"])] = dist.version
        for line in dist.requires or []:
            req = Requirement(line)
            env = default_environment()
            if req.marker and not any(
                    req.marker.evaluate({**env, "extra": e})
                    for e in (wanted or {""})):
                continue
            todo.append((req.name, frozenset(req.extras)))
    return found


def main(argv) -> int:
    extras = set(argv) or {"surya", "geometry", "dev"}
    pins = closure("codicology", extras)
    pins.pop("codicology", None)
    out = f"{HERE}/constraints.txt"
    with open(out, "w") as fh:
        fh.write(
            "# The versions codicology was last built and tested with: the\n"
            f"# closure of codicology[{','.join(sorted(extras))}] as "
            "installed, on\n"
            f"# Python {platform.python_version()}, {platform.system()} "
            f"{platform.machine()}.\n"
            "# Written by tools/lock_env.py; do not edit by hand.\n"
            "#\n"
            "#     pip install -c constraints.txt -e \".[surya]\"\n")
        for name in sorted(pins):
            fh.write(f"{name}=={pins[name]}\n")
    print(f"{out}: {len(pins)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
