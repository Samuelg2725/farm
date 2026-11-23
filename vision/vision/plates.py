"""Number plate reading (FastALPR, MIT) + per-vehicle decisions against the whitelist."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from .matching import Vehicle, find_match, normalise

log = logging.getLogger("vision")

DETECTOR_MODEL = os.getenv("ALPR_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end")
OCR_MODEL = os.getenv("ALPR_OCR_MODEL", "cct-s-v2-global-model")


def crop_scaled(frame: np.ndarray, box, src_size: tuple[int, int], pad: float = 0.15) -> np.ndarray:
    """Crop `box` (x1,y1,x2,y2 measured on a src_size=(w,h) image) out of a bigger frame of the same view."""
    h, w = frame.shape[:2]
    sx, sy = w / src_size[0], h / src_size[1]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, x2 = max(0, int((x1 - px) * sx)), min(w, int((x2 + px) * sx))
    y1, y2 = max(0, int((y1 - py) * sy)), min(h, int((y2 + py) * sy))
    return frame[y1:y2, x1:x2]


def grab_frame(url: str, tries: int = 30) -> np.ndarray | None:
    """Open an RTSP stream, return one decoded full-resolution frame."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        for _ in range(tries):
            ok, frame = cap.read()
            if ok and frame is not None:
                return frame
        return None
    finally:
        cap.release()


class PlateReader:
    def __init__(self):
        from fast_alpr import ALPR  # heavy import
        self.alpr = ALPR(detector_model=DETECTOR_MODEL, ocr_model=OCR_MODEL, detector_conf_thresh=0.3)
        log.info("Plate reader ready (%s / %s)", DETECTOR_MODEL, OCR_MODEL)

    def read(self, image_bgr: np.ndarray) -> list[tuple[str, float]]:
        out = []
        for r in self.alpr.predict(image_bgr):
            if not r.ocr or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            conf = float(np.mean(conf)) if isinstance(conf, (list, tuple, np.ndarray)) else float(conf)
            out.append((normalise(r.ocr.text), conf * float(r.detection.confidence) ** 0.5))
        return out


class Whitelist:
    """Reloads the YAML file automatically when it changes."""

    def __init__(self, path: str):
        self.path, self._mtime, self.vehicles = Path(path), 0.0, []

    def get(self) -> list[Vehicle]:
        try:
            m = self.path.stat().st_mtime
            if m != self._mtime:
                data = yaml.safe_load(self.path.read_text()) or {}
                self.vehicles = [Vehicle(str(v["plate"]), str(v.get("owner", "")))
                                 for v in data.get("vehicles") or []]
                self._mtime = m
                log.info("Whitelist loaded: %d vehicles", len(self.vehicles))
        except Exception:
            log.exception("Could not read whitelist %s", self.path)
        return self.vehicles


@dataclass
class PlateInfo:
    first_seen: float
    plate: str = ""
    score: float = 0.0
    reads: int = 0
    grabs: int = 0
    last_grab: float = -1e18
    crop_jpeg: bytes = b""
    decided: bool = False
    known: bool = False
    owner: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def status(self) -> str:
        if self.decided:
            return "known" if self.known else "unknown"
        return "checking" if self.plate else "no-plate"


class PlateBook:
    """Tracks plate reads for each vehicle on screen and decides known / unknown once.

    Decision when we have a confident read AND (2+ reads agree-ish, or the vehicle
    has left, or `delay` seconds passed since first seen).
    """

    def __init__(self, min_score=0.7, delay=8.0, grab_interval=2.0, max_grabs=4, max_distance=1):
        self.min_score, self.delay = min_score, delay
        self.grab_interval, self.max_grabs, self.max_distance = grab_interval, max_grabs, max_distance
        self.items: dict[int, PlateInfo] = {}

    def get(self, tid: int, now: float) -> PlateInfo:
        return self.items.setdefault(tid, PlateInfo(first_seen=now))

    def want_grab(self, tid: int, now: float) -> bool:
        p = self.get(tid, now)
        if p.decided or p.grabs >= self.max_grabs or now - p.last_grab < self.grab_interval:
            return False
        p.grabs += 1
        p.last_grab = now
        return True

    def add_read(self, tid: int, plate: str, score: float, now: float, crop_jpeg: bytes = b"") -> None:
        p = self.get(tid, now)
        if p.decided or not plate:
            return
        p.reads += 1
        if score >= p.score:
            p.plate, p.score = plate, score
            if crop_jpeg:
                p.crop_jpeg = crop_jpeg

    def decide(self, tid: int, now: float, whitelist: list[Vehicle], lost: bool = False) -> PlateInfo | None:
        """Returns the PlateInfo if it was decided just now, else None."""
        p = self.items.get(tid)
        if p is None or p.decided or p.score < self.min_score:
            return None
        if not (p.reads >= 2 or lost or now - p.first_seen >= self.delay):
            return None
        m = find_match(p.plate, whitelist, self.max_distance)
        p.decided, p.known, p.owner = True, m is not None, (m.owner if m else "")
        return p

    def forget(self, tid: int) -> None:
        self.items.pop(tid, None)
