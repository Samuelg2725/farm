"""Persistence for zones defined in the dashboard (settings/zones.yaml)."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import yaml

from .behaviour import Zone
from .logic import parse_hours

log = logging.getLogger("vision")


class ZoneStore:
    def __init__(self, path: str):
        self.path, self._mtime, self.zones = Path(path), None, []
        self.lock = threading.Lock()

    def get(self) -> list[Zone]:
        with self.lock:
            try:
                m = self.path.stat().st_mtime
            except OSError:
                return self.zones
            if m != self._mtime:
                try:
                    data = yaml.safe_load(self.path.read_text()) or {}
                    self.zones = [Zone(str(z["name"]), [[float(x), float(y)] for x, y in z["points"]],
                                       str(z.get("type", "restricted")), str(z.get("hours", "always")))
                                  for z in data.get("zones") or [] if len(z.get("points") or []) >= 3]
                    log.info("Zones loaded: %s", ", ".join(z.name for z in self.zones) or "none")
                except Exception:
                    log.exception("Could not read zones file %s", self.path)
                self._mtime = m
            return self.zones

    def as_dicts(self) -> list[dict]:
        return [{"name": z.name, "type": z.kind, "hours": z.hours, "points": z.points} for z in self.get()]

    def save(self, zones: list[dict]) -> list[dict]:
        clean = []
        for z in zones:
            pts = [[round(min(1, max(0, float(x))), 4), round(min(1, max(0, float(y))), 4)] for x, y in z["points"]]
            if len(pts) < 3 or not str(z.get("name", "")).strip():
                continue
            kind = z.get("type", "restricted") if z.get("type") in ("restricted", "watch") else "restricted"
            hours = str(z.get("hours") or "always").strip().lower()
            try:
                parse_hours(hours)
            except ValueError:
                raise ValueError(f"invalid hours for zone '{z['name']}': use 'always' or e.g. 18:00-07:00")
            clean.append({"name": str(z["name"]).strip()[:40], "type": kind, "hours": hours, "points": pts})
        header = ("# Zones defined in the dashboard. Points are normalised image coordinates (0-1).\n"
                  "# type: restricted (notify on entry) or watch (log only)\n"
                  "# hours: always, or a time range such as 18:00-07:00\n")
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(header + yaml.safe_dump({"zones": clean}, sort_keys=False))
            self._mtime = None
        return clean
