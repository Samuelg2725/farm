"""Plate normalisation and tolerant whitelist matching.

OCR commonly confuses similar characters (O/0, I/1, B/8, ...). Plates are compared in a
canonical form in which these characters are folded together, with a small permitted edit
distance, so a single misread character does not cause a whitelisted vehicle to be flagged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_LOOKALIKES = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
                             "Z": "2", "S": "5", "B": "8", "G": "6"})


def normalise(plate: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (plate or "").upper())


def canonical(plate: str) -> str:
    return normalise(plate).translate(_LOOKALIKES)


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


@dataclass(frozen=True)
class Vehicle:
    plate: str
    owner: str = ""


def find_match(plate: str, vehicles: list[Vehicle], max_distance: int = 1) -> Vehicle | None:
    """Return the closest whitelisted vehicle within `max_distance`, or None."""
    read = canonical(plate)
    if not read:
        return None
    best, best_d = None, max_distance + 1
    for v in vehicles:
        d = levenshtein(read, canonical(v.plate))
        if d < best_d:
            best, best_d = v, d
    return best if best_d <= max_distance else None
