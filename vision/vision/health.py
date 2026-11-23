"""System health monitoring: camera availability, processing latency and disk space."""
from __future__ import annotations

import shutil
from collections import deque
from dataclasses import dataclass


@dataclass
class HealthAlert:
    title: str
    body: str
    priority: str = "high"
    tags: str = "warning"


def duration_text(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60} min"


def disk_space(path: str) -> tuple[float, float]:
    """(free GB, total GB) for the filesystem holding `path`."""
    u = shutil.disk_usage(path)
    return u.free / 1e9, u.total / 1e9


class HealthMonitor:
    """Evaluates health inputs periodically and returns notifications on state changes.

    camera:  offline when no frame has been received for `stale_after` seconds; a notification is
             sent once it has been offline for `offline_alert_after` seconds, and again on recovery.
    latency: notifies when the stream has been resynchronised `lag_skips` times within `lag_window`.
    disk:    notifies when free space falls below `min_free_gb` (at most once per `disk_cooldown`).
    startup: a notification is sent on the first check.
    """

    def __init__(self, stale_after: float = 15, offline_alert_after: float = 60, lag_skips: int = 3,
                 lag_window: float = 600, lag_cooldown: float = 3600, min_free_gb: float = 10,
                 disk_cooldown: float = 6 * 3600):
        self.stale_after, self.offline_alert_after = stale_after, offline_alert_after
        self.lag_skips, self.lag_window, self.lag_cooldown = lag_skips, lag_window, lag_cooldown
        self.min_free_gb, self.disk_cooldown = min_free_gb, disk_cooldown
        self.started: float | None = None
        self.offline_since: float | None = None
        self.offline_alerted = False
        self.skip_times: deque = deque()
        self.last_skips = 0
        self.last_lag_alert = -1e18
        self.last_disk_alert = -1e18
        self.status: dict = {}

    def check(self, now: float, last_frame: float | None, skips: int, free_gb: float | None,
              total_gb: float | None = None, camera: str = "camera") -> list[HealthAlert]:
        alerts: list[HealthAlert] = []
        if self.started is None:
            self.started = now
            alerts.append(HealthAlert("System started", f"Monitoring {camera}.", "default", "white_check_mark"))

        # camera availability
        age = None if last_frame is None else now - last_frame
        online = age is not None and age <= self.stale_after
        if not online:
            if self.offline_since is None:
                self.offline_since = last_frame if last_frame is not None else self.started
            down_for = now - self.offline_since
            if not self.offline_alerted and down_for >= self.offline_alert_after:
                self.offline_alerted = True
                alerts.append(HealthAlert("Camera offline", f"No video from {camera} for {duration_text(down_for)}. "
                                          "Check power, network cable and camera address.", "urgent"))
        else:
            if self.offline_alerted:
                alerts.append(HealthAlert("Camera back online",
                                          f"{camera} recovered after {duration_text(now - self.offline_since)}.",
                                          "default", "white_check_mark"))
            self.offline_since, self.offline_alerted = None, False

        # processing latency (stream resynchronisations)
        for _ in range(max(0, skips - self.last_skips)):
            self.skip_times.append(now)
        self.last_skips = skips
        while self.skip_times and now - self.skip_times[0] > self.lag_window:
            self.skip_times.popleft()
        lagging = len(self.skip_times) >= self.lag_skips
        if lagging and now - self.last_lag_alert >= self.lag_cooldown:
            self.last_lag_alert = now
            alerts.append(HealthAlert("Processing falling behind",
                                      f"The live stream was resynchronised {len(self.skip_times)} times in "
                                      f"{duration_text(self.lag_window)}. Reduce the camera frame rate or FPS."))

        # disk space
        low_disk = free_gb is not None and free_gb < self.min_free_gb
        if low_disk and now - self.last_disk_alert >= self.disk_cooldown:
            self.last_disk_alert = now
            alerts.append(HealthAlert("Disk space low", f"{free_gb:.1f} GB free. Oldest recordings are being removed."))

        self.status = {
            "camera_online": online,
            "camera_offline_for": None if online or self.offline_since is None else round(now - self.offline_since),
            "lagging": lagging,
            "disk_free_gb": None if free_gb is None else round(free_gb, 1),
            "disk_total_gb": None if total_gb is None else round(total_gb, 1),
            "low_disk": low_disk,
            "uptime": round(now - self.started),
            "ok": online and not lagging and not low_disk,
        }
        return alerts
