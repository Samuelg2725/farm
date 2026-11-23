"""Records the camera's original 4K video while a person or vehicle is in view.

Uses ffmpeg in "copy" mode: the camera's video is saved as-is (no re-encoding),
so it's full quality and uses almost no CPU. Clips go in /recordings/YYYY-MM-DD/.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("vision")


def ffmpeg_cmd(url: str, out: Path, codec: str) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    if url.startswith("rtsp"):
        cmd += ["-rtsp_transport", "tcp"]
    else:  # a test video file: play it at real speed
        cmd += ["-re"]
    cmd += ["-i", url, "-map", "0:v", "-map", "0:a?", "-c", "copy"]
    if codec == "hevc":
        cmd += ["-tag:v", "hvc1"]  # lets Safari / QuickTime play H.265
    cmd += ["-movflags", "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", str(out)]
    return cmd


def clip_name(start: datetime, labels: set[str]) -> str:
    what = "+".join(sorted(labels)) or "motion"
    return f"{start:%H-%M-%S}_{what}"


class Recorder:
    def __init__(self, url: str, out_dir: str, post_roll: float = 10, max_length: float = 600,
                 keep_days: int = 30, enabled: bool = True):
        self.url, self.out_dir = url, Path(out_dir)
        self.post_roll, self.max_length, self.keep_days, self.enabled = post_roll, max_length, keep_days, enabled
        self.proc: subprocess.Popen | None = None
        self.path: Path | None = None
        self.started: datetime | None = None
        self.last_active: datetime | None = None
        self.labels: set[str] = set()
        self.codec = "h264"
        if enabled and not shutil.which("ffmpeg"):
            log.error("ffmpeg not installed – recording disabled")
            self.enabled = False

    @property
    def recording(self) -> bool:
        return self.proc is not None

    def update(self, now: datetime, labels: set[str], snapshot_jpeg: bytes = b"") -> tuple[str, Path] | None:
        """Call every frame with what's in view. Returns ("started"|"stopped", path) when that happens."""
        if not self.enabled:
            return None
        if self.proc is not None and self.proc.poll() is not None:  # ffmpeg died (stream dropped?)
            log.warning("Recording process ended unexpectedly")
            return self._stop(now)
        if labels:
            self.last_active = now
            self.labels |= labels
            if self.proc is None:
                return self._start(now, labels, snapshot_jpeg)
            if (now - self.started).total_seconds() > self.max_length:
                return self._stop(now)  # a new clip starts next frame
        elif self.proc is not None and (now - self.last_active).total_seconds() > self.post_roll:
            return self._stop(now)
        return None

    def _start(self, now: datetime, labels: set[str], snapshot_jpeg: bytes):
        self.cleanup(now)
        day = self.out_dir / f"{now:%Y-%m-%d}"
        day.mkdir(parents=True, exist_ok=True)
        self.path = day / (clip_name(now, labels) + ".mp4")
        if snapshot_jpeg:
            self.path.with_suffix(".jpg").write_bytes(snapshot_jpeg)
        self.proc = subprocess.Popen(ffmpeg_cmd(self.url, self.path, self.codec),
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.started, self.labels = now, set(labels)
        log.info("Recording started: %s", self.path)
        return "started", self.path

    def _stop(self, now: datetime):
        proc, path = self.proc, self.path
        self.proc = None
        if proc.poll() is None:
            proc.terminate()  # ffmpeg finishes the file cleanly on SIGTERM
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        err = proc.stderr.read().decode(errors="replace").strip() if proc.stderr else ""
        if err:
            log.warning("ffmpeg: %s", err[-300:])
        # rename so the file name lists everything seen during the clip
        final = path.with_name(clip_name(self.started, self.labels) + ".mp4")
        if final != path and path.exists():
            path.rename(final)
            if path.with_suffix(".jpg").exists():
                path.with_suffix(".jpg").rename(final.with_suffix(".jpg"))
        log.info("Recording stopped: %s (%.0f s)", final, (now - self.started).total_seconds())
        return "stopped", final

    def cleanup(self, now: datetime) -> None:
        cutoff = (now - timedelta(days=self.keep_days)).strftime("%Y-%m-%d")
        if not self.out_dir.exists():
            return
        for d in self.out_dir.iterdir():
            if d.is_dir() and d.name < cutoff:
                shutil.rmtree(d, ignore_errors=True)
                log.info("Deleted old recordings %s", d.name)

    def list_clips(self, limit: int = 200) -> list[dict]:
        if not self.out_dir.exists():
            return []
        clips = sorted(self.out_dir.glob("*/*.mp4"), reverse=True)[:limit]
        out = []
        for c in clips:
            rel = c.relative_to(self.out_dir).as_posix()
            thumb = c.with_suffix(".jpg")
            out.append({"file": rel, "day": c.parent.name, "name": c.stem, "size_mb": round(c.stat().st_size / 1e6, 1),
                        "thumb": thumb.relative_to(self.out_dir).as_posix() if thumb.exists() else "",
                        "live": self.path is not None and c == self.path and self.recording})
        return out
