"""
grounding.py — check that every number in a model's answer comes from a
source (tool results, the question, a document).

A number in the answer is *verified* when some source number rounds to it
at the precision it is written with (2.4 matches 2.43; 43 % matches 0.43
as a share). Years (1990-2035) and small counts (0-10) are not checked:
they are labels far more often than estimates.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from typing import Any

_NUM = re.compile(r"(?<![\w.])([-−+]?)(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(?![\w])")


def numbers_in(text: str) -> list[tuple[str, float, int]]:
    """(as written, value, decimals) for every number in *text*."""
    out = []
    for m in _NUM.finditer(text):
        sign, whole, frac = m.groups()
        s = whole.replace(",", "") + ("." + frac if frac else "")
        v = float(s) * (-1 if sign in "-−" and sign else 1)
        out.append((m.group(0), v, len(frac) if frac else 0))
    return out


def _skip(v: float, dec: int) -> bool:
    return dec == 0 and (0 <= v <= 10 or 1990 <= v <= 2035)


def source_numbers(sources: Iterable[Any]) -> list[float]:
    vals: list[float] = []

    def walk(o):
        if isinstance(o, bool) or o is None:
            return
        if isinstance(o, (int, float)):
            vals.append(float(o))
        elif isinstance(o, str):
            vals.extend(v for _, v, _ in numbers_in(o))
        elif isinstance(o, dict):
            for x in o.values():
                walk(x)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        else:
            walk(json.loads(json.dumps(o, default=str)))
    for s in sources:
        walk(s)
    return vals


def _matches(v: float, dec: int, src: list[float]) -> bool:
    tol = 0.5 * 10 ** (-dec) + 1e-9
    for s in src:
        for cand in (s, s * 100):          # shares written as percentages
            if abs(abs(cand) - abs(v)) <= tol:
                return True
    return False


def check(answer: str, sources: Iterable[Any]) -> dict:
    """{'verified': [...], 'unverified': [...]} (numbers as written)."""
    src = source_numbers(sources)
    verified, unverified = [], []
    for raw, v, dec in numbers_in(answer):
        if _skip(v, dec):
            continue
        (verified if _matches(v, dec, src) else unverified).append(raw)
    return {"verified": verified, "unverified": unverified}
