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


class LagMeter:
    """Estimates how far behind live we are from stream timestamps.

    For a live stream, (wall clock - stream timestamp) stays constant when we keep up.
    If decoding is too slow, frames queue up and that difference grows: that growth is our lag.
    """

    def __init__(self):
        self.base: float | None = None

    def reset(self) -> None:
        self.base = None

    def update(self, pts: float, wall: float) -> float:
        d = wall - pts
        if self.base is None or d < self.base:
            self.base = d
        return d - self.base


class PersonAlerter:
    """Decides when to send "person seen" alerts while person alerts are switched on.

    A new person is announced `delay` seconds after they appear (so we can say what they're doing),
    people arriving together share one alert, and alerts are at least `gap` seconds apart (later
    arrivals wait and are included in the next one). Each behaviour of each person (running,
    crouching…) alerts at most once per `behaviour_cooldown`.
    """

    def __init__(self, delay: float = 2.0, gap: float = 20.0, behaviour_cooldown: float = 60.0):
        self.delay, self.gap, self.behaviour_cooldown = delay, gap, behaviour_cooldown
        self.pending: dict[int, tuple[int, float]] = {}   # track id -> (person number, time seen)
        self.last_sent = -1e18
        self.behaviour_sent: dict[tuple[int, str], float] = {}

    def arrived(self, tid: int, number: int, t: float) -> None:
        self.pending.setdefault(tid, (number, t))

    def is_pending(self, tid: int) -> bool:
        return tid in self.pending

    def due(self, t: float, present: set[int]) -> list[tuple[int, int]]:
        """People to announce now: [(track id, number)]. People who already left are dropped."""
        for tid in [k for k in self.pending if k not in present]:
            del self.pending[tid]
        ready = [(tid, n) for tid, (n, since) in self.pending.items() if t - since >= self.delay]
        if not ready or t - self.last_sent < self.gap:
            return []
        out = [(tid, n) for tid, (n, _) in self.pending.items()]  # include anyone else waiting
        self.pending.clear()
        self.last_sent = t
        return sorted(out, key=lambda x: x[1])

    def behaviour_ok(self, number: int, behaviour: str, t: float) -> bool:
        key = (number, behaviour)
        if t - self.behaviour_sent.get(key, -1e18) < self.behaviour_cooldown:
            return False
        self.behaviour_sent[key] = t
        if len(self.behaviour_sent) > 2000:
            self.behaviour_sent.clear()
        return True
