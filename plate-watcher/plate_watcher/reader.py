"""Free, open-source plate reading (FastALPR – MIT licence).

Grabs a full-resolution 4K frame from the camera, crops round the car Frigate
found, then finds and reads the plate.
"""
from __future__ import annotations

import logging
import os

import cv2
import numpy as np

log = logging.getLogger("plate-watcher")

DETECTOR_MODEL = os.getenv("ALPR_DETECTOR_MODEL", "yolo-v9-t-384-license-plate-end2end")
OCR_MODEL = os.getenv("ALPR_OCR_MODEL", "cct-s-v2-global-model")


def crop_box(frame: np.ndarray, box, detect_size: tuple[int, int], pad: float = 0.15) -> np.ndarray:
    """Crop `box` (x1,y1,x2,y2 in Frigate detect-stream pixels) out of a full-res frame."""
    h, w = frame.shape[:2]
    sx, sy = w / detect_size[0], h / detect_size[1]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, x2 = max(0, int((x1 - px) * sx)), min(w, int((x2 + px) * sx))
    y1, y2 = max(0, int((y1 - py) * sy)), min(h, int((y2 + py) * sy))
    return frame[y1:y2, x1:x2]


def grab_frame(url: str, tries: int = 30) -> np.ndarray | None:
    """Open the RTSP stream, return one decoded full-res frame."""
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
        from fast_alpr import ALPR  # imported lazily: heavy
        self.alpr = ALPR(detector_model=DETECTOR_MODEL, ocr_model=OCR_MODEL,
                         detector_conf_thresh=0.3)
        log.info("Plate reader ready (%s / %s)", DETECTOR_MODEL, OCR_MODEL)

    def read(self, image_bgr: np.ndarray) -> list[tuple[str, float]]:
        out = []
        for r in self.alpr.predict(image_bgr):
            if not r.ocr or not r.ocr.text:
                continue
            conf = r.ocr.confidence
            conf = float(np.mean(conf)) if isinstance(conf, (list, tuple, np.ndarray)) else float(conf)
            out.append((r.ocr.text, conf * float(r.detection.confidence) ** 0.5))
        return out
