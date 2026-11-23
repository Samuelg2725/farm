"""Event recording while people or vehicles are in view.

Each clip produces, under <out_dir>/YYYY-MM-DD/:
  HH-MM-SS.mp4     annotated video (1080p H.264)
  HH-MM-SS_4k.mp4  original camera stream, stream-copied without re-encoding
  HH-MM-SS.jpg     thumbnail
Directories older than `keep_days` are removed.
"""
from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("vision")

_FRAG = ["-movflags", "+frag_keyframe+empty_moov+default_base_moof"]  # fragmented MP4: playable while still being written


def raw_cmd(url: str, out: Path, codec: str) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
    cmd += ["-rtsp_transport", "tcp"] if url.startswith("rtsp") else ["-re"]  # -re: read file inputs at native rate
    cmd += ["-i", url, "-map", "0:v", "-map", "0:a?", "-c", "copy"]
    if codec == "hevc":
        cmd += ["-tag:v", "hvc1"]  # required by Safari/QuickTime for HEVC
    return cmd + _FRAG + ["-f", "mp4", str(out)]


def annotated_cmd(out: Path) -> list[str]:
    return ["ffmpeg", "-hide_banner", "-loglevel", "error",
            # wall-clock timestamps at millisecond resolution (variable frame rate)
            "-f", "image2pipe", "-framerate", "1000", "-use_wallclock_as_timestamps", "1", "-c:v", "mjpeg", "-i", "-",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "25", "-pix_fmt", "yuv420p", "-bf", "0",
            "-g", "24", "-fps_mode", "passthrough", "-video_track_timescale", "1000", *_FRAG, "-f", "mp4", str(out)]


class _AnnotatedWriter:
    """Encodes JPEG frames via ffmpeg on a background thread without blocking the caller."""

    def __init__(self, out: Path):
        self.proc = subprocess.Popen(annotated_cmd(out), stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.q: queue.Queue = queue.Queue(maxsize=30)
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def _run(self):
        while True:
            frame = self.q.get()
            if frame is None:
                break
            try:
                self.proc.stdin.write(frame)
            except (BrokenPipeError, ValueError):
                break

    def feed(self, jpeg: bytes) -> None:
        try:
            self.q.put_nowait(jpeg)
        except queue.Full:
            pass  # drop frames rather than block

    def close(self) -> str:
        self.q.put(None)
        self.t.join(timeout=10)
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        return self.proc.stderr.read().decode(errors="replace").strip() if self.proc.stderr else ""


class Recorder:
    def __init__(self, url: str, out_dir: str, post_roll: float = 10, max_length: float = 600,
                 keep_days: int = 30, enabled: bool = True, keep_raw: bool = True):
        self.url, self.out_dir = url, Path(out_dir)
        self.post_roll, self.max_length, self.keep_days = post_roll, max_length, keep_days
        self.enabled, self.keep_raw = enabled, keep_raw
        self.raw: subprocess.Popen | None = None
        self.annotated: _AnnotatedWriter | None = None
        self.clip: str = ""             # current clip path relative to out_dir
        self.started: datetime | None = None
        self.last_active: datetime | None = None
        self.codec = "h264"
        self.lock = threading.Lock()
        if enabled and not shutil.which("ffmpeg"):
            log.error("ffmpeg not installed – recording disabled")
            self.enabled = False

    @property
    def recording(self) -> bool:
        return bool(self.clip)

    def feed(self, jpeg: bytes) -> None:
        """Submit an annotated frame to the current clip."""
        with self.lock:
            if self.annotated is not None:
                self.annotated.feed(jpeg)

    def update(self, now: datetime, active: bool, thumb_jpeg: bytes = b"") -> tuple[str, dict] | None:
        """Advance the recording state; returns ("started" | "stopped", info) on a transition, else None."""
        if not self.enabled:
            return None
        if active:
            self.last_active = now
            if not self.recording:
                return self._start(now, thumb_jpeg)
            if (now - self.started).total_seconds() > self.max_length:
                return self._stop(now)  # split long clips; a new clip starts on the next cycle
        elif self.recording and (now - self.last_active).total_seconds() > self.post_roll:
            return self._stop(now)
        return None

    def _start(self, now: datetime, thumb_jpeg: bytes):
        self.cleanup(now)
        day = self.out_dir / f"{now:%Y-%m-%d}"
        day.mkdir(parents=True, exist_ok=True)
        base = day / f"{now:%H-%M-%S}"
        info = {"file": f"{base.relative_to(self.out_dir)}.mp4", "raw_file": "", "thumb": ""}
        if thumb_jpeg:
            base.with_suffix(".jpg").write_bytes(thumb_jpeg)
            info["thumb"] = f"{base.relative_to(self.out_dir)}.jpg"
        with self.lock:
            self.annotated = _AnnotatedWriter(base.with_suffix(".mp4"))
        if self.keep_raw:
            raw_path = base.parent / f"{base.name}_4k.mp4"
            self.raw = subprocess.Popen(raw_cmd(self.url, raw_path, self.codec), stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            info["raw_file"] = str(raw_path.relative_to(self.out_dir))
        self.clip, self.started = info["file"], now
        log.info("Recording started: %s", info["file"])
        return "started", info

    def _stop(self, now: datetime):
        with self.lock:
            writer, self.annotated = self.annotated, None
        errors = [writer.close()] if writer else []
        if self.raw is not None:
            if self.raw.poll() is None:
                self.raw.terminate()  # ffmpeg finalises the file on SIGTERM
                try:
                    self.raw.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.raw.kill()
            errors.append(self.raw.stderr.read().decode(errors="replace").strip() if self.raw.stderr else "")
            self.raw = None
        for e in filter(None, errors):
            log.warning("ffmpeg: %s", e[-300:])
        info = {"file": self.clip, "seconds": round((now - self.started).total_seconds())}
        log.info("Recording stopped: %s (%d s)", self.clip, info["seconds"])
        self.clip = ""
        return "stopped", info

    def cleanup(self, now: datetime) -> list[str]:
        cutoff = (now - timedelta(days=self.keep_days)).strftime("%Y-%m-%d")
        removed = []
        if self.out_dir.exists():
            for d in self.out_dir.iterdir():
                if d.is_dir() and d.name < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    removed.append(d.name)
                    log.info("Deleted old recordings %s", d.name)
        return removed
