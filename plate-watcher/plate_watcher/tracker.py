"""Collects plate reads per Frigate tracked object and decides once per vehicle.

A car produces many reads as it moves; early ones are often partial. We keep
the best-scoring read and decide when the object leaves (event end) or after
`delay` seconds, whichever comes first.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pending:
    event_id: str
    camera: str
    first_seen: float
    plate: str = ""
    score: float = 0.0
    ended: bool = False


class Tracker:
    def __init__(self, delay: float = 20.0, min_score: float = 0.0):
        self.delay = delay
        self.min_score = min_score
        self.pending: dict[str, Pending] = {}
        self.decided: set[str] = set()

    def add_read(self, event_id: str, camera: str, plate: str, score: float, now: float) -> None:
        if not plate or event_id in self.decided or score < self.min_score:
            return
        p = self.pending.setdefault(event_id, Pending(event_id, camera, now))
        if score >= p.score:
            p.plate, p.score = plate, score

    def mark_ended(self, event_id: str) -> None:
        if event_id in self.pending:
            self.pending[event_id].ended = True

    def due(self, now: float) -> list[Pending]:
        out = [p for p in self.pending.values() if p.ended or now - p.first_seen >= self.delay]
        for p in out:
            del self.pending[p.event_id]
            self.decided.add(p.event_id)
        if len(self.decided) > 5000:  # keep memory bounded
            self.decided = set(list(self.decided)[-1000:])
        return out
