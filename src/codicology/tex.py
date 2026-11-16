"""TeX as the recogniser writes it: its Greek letters and signs as the
characters they stand for, a command's braced argument, a fraction made
flat, and the pattern of an equation set on its own line. Shared by the
equation mending of the second and third readers and the MathML writer.
"""
from __future__ import annotations

import re


_TEX_DOTS = {"ldots": "...", "dots": "...", "cdots": "...", "sum": "Σ"}


def _tex_name(word: str) -> str:
    """The character a TeX name stands for, or the name where it has none."""
    return _TEX_DOTS.get(word) or GREEK.get(word) or MATH_SIGNS.get(word) \
        or word


def _brace_group(t: str, i: int) -> "tuple[str, int]":
    """The group that opens with the brace at t[i]: its inside, and the
    index after its closing brace."""
    def written_out(j: int) -> bool:
        """Whether the brace at j is \\{ or \\}: one backslash before
        it, or any odd number. Two are a row's end and the brace is a brace."""
        k = j
        while k > 0 and t[k - 1] == "\\":
            k -= 1
        return (j - k) % 2 == 1

    depth = 0
    for j in range(i, len(t)):
        if t[j] == "{" and not written_out(j):
            depth += 1
        elif t[j] == "}" and not written_out(j):
            depth -= 1
            if depth == 0:
                return t[i + 1:j], j + 1
    return t[i + 1:], len(t)


def _unfrac(t: str) -> str:
    """\\frac{A}{B} as A/B, however deep its arguments nest: a power in
    the numerator is a brace inside the brace, which a pattern cannot count."""
    for _ in range(64):
        m = re.search(r"\\[dt]?frac\s*(?=\{)", t)
        if not m:
            break
        a, j = _brace_group(t, m.end())
        k = j
        while k < len(t) and t[k].isspace():
            k += 1
        if k >= len(t) or t[k] != "{":
            t = t[:m.start()] + a + t[j:]
            continue
        b, e = _brace_group(t, k)
        t = t[:m.start()] + a + "/" + b + t[e:]
    return t


GREEK = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ",
         "mu": "μ", "pi": "π", "psi": "ψ", "epsilon": "ε", "omega": "ω",
         "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
         "lambda": "λ", "nu": "ν", "xi": "ξ", "rho": "ρ", "sigma": "σ",
         "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ", "Gamma": "Γ",
         "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ",
         "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω"}


# The signs the recogniser spells out, measured over every inline <math> in
# 151 books' readings (5,746 of them): \times 108, \circ 91, \dots 39,
# \rightarrow 21, and a handful each of the rest. Stripping the backslash,
# as a name not listed here still has done to it, left "4 times 7" and
# "14^{circ}F" in the prose.
MATH_SIGNS = {"times": "×", "div": "÷", "pm": "±", "cdot": "·",
              "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥", "neq": "≠",
              "ne": "≠", "approx": "≈", "rightarrow": "→", "to": "→",
              "leftarrow": "←", "dots": "…", "ldots": "…", "cdots": "…",
              "infty": "∞"}


_DISPLAY_MATH = re.compile(r'<math\b(?=[^>]*\bdisplay="block")(?![^>]*\bxmlns=)'
                           r'[^>]*>(.*?)</math>', re.S)
