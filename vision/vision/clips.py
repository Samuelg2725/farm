"""Pictures and videos made for the phone."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger("vision")


def plate_photo(vehicle_bgr: np.ndarray, plate_box, plate_text: str = "", width: int = 1280) -> bytes:
    """The vehicle, with an enlarged close-up of its number plate inset at the top-left."""
    h, w = vehicle_bgr.shape[:2]
    f = width / w
    img = cv2.resize(vehicle_bgr, (width, int(h * f)), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    if plate_box is not None:
        x1, y1, x2, y2 = plate_box
        px, py = int((x2 - x1) * 0.15), int((y2 - y1) * 0.3)
        plate = vehicle_bgr[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
        cv2.rectangle(img, (int(x1 * f), int(y1 * f)), (int(x2 * f), int(y2 * f)), (0, 215, 255), 3)
        if plate.size:
            pw = int(min(img.shape[1] * 0.55, max(360, plate.shape[1] * 3)))
            ph = int(plate.shape[0] * pw / plate.shape[1])
            ph = min(ph, img.shape[0] // 2)
            big = cv2.resize(plate, (pw, ph), interpolation=cv2.INTER_CUBIC)
            cv2.rectangle(img, (8, 8), (16 + pw, 16 + ph + (34 if plate_text else 0)), (0, 215, 255), -1)
            img[12:12 + ph, 12:12 + pw] = big
            if plate_text:
                cv2.putText(img, plate_text, (16, 12 + ph + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes() if ok else b""


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    try:
        return float(out)
    except ValueError:
        return 0.0


def phone_clip(src: Path, dst: Path, max_seconds: float = 90, max_mb: float = 14) -> Path | None:
    """Copy of a recording small enough to send to a phone (ntfy.sh allows 15 MB per file).
    Uses the clip as-is if it already fits, otherwise re-encodes at 720p (and shortens if very long)."""
    if not src.is_file():
        return None
    length = duration(src)
    if src.stat().st_size <= max_mb * 1e6 and length <= max_seconds:
        return src
    secs = min(length or max_seconds, max_seconds)
    # bitrate that fits the size limit (leave 10% headroom), capped at a sensible quality
    kbps = int(min(2500, max(250, max_mb * 8000 * 0.9 / max(secs, 1))))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-t", f"{secs:.1f}",
           "-vf", "scale=-2:720", "-c:v", "libx264", "-preset", "veryfast", "-b:v", f"{kbps}k",
           "-maxrate", f"{kbps}k", "-bufsize", f"{kbps * 2}k", "-pix_fmt", "yuv420p", "-an",
           "-movflags", "+faststart", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not dst.is_file():
        log.warning("Could not make phone clip: %s", r.stderr[-300:])
        return None
    return dst
