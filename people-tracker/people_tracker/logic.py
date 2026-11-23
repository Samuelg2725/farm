"""Pure logic: daily person numbering and when to send alerts (no I/O, easy to test)."""
from __future__ import annotations

from datetime import datetime, time


class DailyNumbers:
    """Gives each tracked person a friendly number (Person 1, 2, 3...) that restarts each day."""

    def __init__(self):
        self.day = None
        self.map: dict[int, int] = {}
        self.next = 1

    def number_for(self, track_id: int, now: datetime) -> tuple[int, bool]:
        """Return (number, is_new)."""
        if now.date() != self.day:
            self.day, self.map, self.next = now.date(), {}, 1
        if track_id in self.map:
            return self.map[track_id], False
        self.map[track_id] = self.next
        self.next += 1
        return self.map[track_id], True


def parse_hours(spec: str):
    """'always' | 'never' | 'HH:MM-HH:MM' (may wrap past midnight, e.g. 18:00-07:00)."""
    spec = (spec or "always").strip().lower()
    if spec in ("always", "never"):
        return spec
    a, b = spec.split("-")
    return (time.fromisoformat(a.strip()), time.fromisoformat(b.strip()))


def in_hours(hours, t: time) -> bool:
    if hours == "always":
        return True
    if hours == "never":
        return False
    start, end = hours
    return start <= t < end if start <= end else (t >= start or t < end)


class AlertPolicy:
    def __init__(self, hours: str = "always", cooldown: float = 300):
        self.hours = parse_hours(hours)
        self.cooldown = cooldown
        self.last = -1e18

    def should_alert(self, now: datetime) -> bool:
        if not in_hours(self.hours, now.time()):
            return False
        ts = now.timestamp()
        if ts - self.last < self.cooldown:
            return False
        self.last = ts
        return True
