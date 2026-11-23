"""Live people tracker: numbered boxes round every person + phone alerts.

Reads the camera's low-res stream (via Frigate's restream), detects people with
YOLOX, follows each one with Norfair (same tracker Frigate uses), labels them
"Person 1, 2, 3..." and serves a live view at http://<host>:8090
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import requests
from norfair import Detection, Tracker

from .detector import PersonDetector
from .logic import AlertPolicy, DailyNumbers

log = logging.getLogger("people-tracker")

CAMERA = os.getenv("CAMERA", "yard_ptz")
STREAM_URL = os.getenv("STREAM_URL", "rtsp://frigate:8554/yard_ptz_sub")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/yolox_s.onnx")
FPS = float(os.getenv("FPS", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))
ALERT_HOURS = os.getenv("PERSON_ALERT_HOURS", "always")  # always | never | 18:00-07:00
ALERT_COOLDOWN = float(os.getenv("PERSON_ALERT_COOLDOWN", "300"))
NTFY_URL = os.getenv("NTFY_URL", "").rstrip("/")
DB_PATH = os.getenv("DB_PATH", "/data/people.db")
PORT = int(os.getenv("PORT", "8090"))

COLOURS = [(0, 200, 255), (255, 120, 0), (0, 220, 0), (200, 0, 255), (0, 0, 255), (255, 0, 160)]


class FrameGrabber(threading.Thread):
    """Keeps only the newest frame so processing never lags behind live."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url, self.frame, self.lock = url, None, threading.Lock()

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("Cannot open %s – retrying in 5 s", self.url)
                time.sleep(5)
                continue
            log.info("Connected to stream %s", self.url)
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Stream dropped – reconnecting")
                    break
                with self.lock:
                    self.frame = frame
            cap.release()
            time.sleep(2)

    def latest(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def draw(frame: np.ndarray, people: list[dict]) -> np.ndarray:
    f = 720 / frame.shape[0] if frame.shape[0] < 720 else 1.0  # upscale low-res stream so labels are readable
    if f != 1.0:
        frame = cv2.resize(frame, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR)
    scale = frame.shape[0] / 720
    for p in people:
        x1, y1, x2, y2 = (int(v * f) for v in p["box"])
        c = COLOURS[p["number"] % len(COLOURS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, max(2, int(2 * scale)))
        label = f"Person {p['number']}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
        ty = max(th + 6, y1)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 6, ty), c, -1)
        cv2.putText(frame, label, (x1 + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), 2)
    banner = f"{datetime.now():%d %b %H:%M:%S}   People in view: {len(people)}"
    cv2.putText(frame, banner, (8, int(22 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), 4)
    cv2.putText(frame, banner, (8, int(22 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (255, 255, 255), 2)
    return frame


def send_ntfy(title: str, body: str, jpeg: bytes) -> None:
    if not NTFY_URL:
        return
    try:
        requests.put(NTFY_URL, data=jpeg, timeout=20, headers={
            "Title": title.encode(), "Message": body.encode(), "Filename": "person.jpg",
            "Priority": "high", "Tags": "bust_in_silhouette"}).raise_for_status()
    except Exception:
        log.exception("ntfy alert failed")


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes = b""
        self.people: list[dict] = []
        self.updated = 0.0


STATE = State()

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Farm Live People</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
header{padding:10px 16px;font-size:18px}img{width:100%;max-width:1280px;display:block;margin:auto}
#list{padding:10px 16px;font-size:15px}</style></head><body>
<header>Live people – <span id="count">…</span></header>
<img src="/stream.mjpg" alt="live view"><div id="list"></div>
<script>
async function tick(){try{const r=await fetch('/status.json');const s=await r.json();
document.getElementById('count').textContent=s.people.length+' in view';
document.getElementById('list').textContent=s.people.map(p=>'Person '+p.number+' ('+Math.round(p.score*100)+'%)').join(' · ');}catch(e){}}
setInterval(tick,1000);tick();
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/":
            self._send(200, "text/html; charset=utf-8", PAGE.encode())
        elif self.path == "/status.json":
            with STATE.lock:
                body = json.dumps({"camera": CAMERA, "people": STATE.people, "updated": STATE.updated})
            self._send(200, "application/json", body.encode())
        elif self.path == "/snapshot.jpg":
            with STATE.lock:
                jpeg = STATE.jpeg
            self._send(200 if jpeg else 503, "image/jpeg", jpeg)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            last = b""
            try:
                while True:
                    with STATE.lock:
                        jpeg = STATE.jpeg
                    if jpeg and jpeg is not last:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                        last = jpeg
                    time.sleep(0.05)
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self._send(404, "text/plain", b"not found")

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = PersonDetector(MODEL_PATH, conf=MIN_SCORE)
    log.info("Person detector ready (%s)", MODEL_PATH)
    tracker = Tracker(distance_function="iou", distance_threshold=0.75,
                      initialization_delay=2, hit_counter_max=int(FPS * 3))
    numbers = DailyNumbers()
    policy = AlertPolicy(ALERT_HOURS, ALERT_COOLDOWN)
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS people (ts TEXT, camera TEXT, number INTEGER, score REAL)")

    grabber = FrameGrabber(STREAM_URL)
    grabber.start()
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever, daemon=True).start()
    log.info("Live view on port %d", PORT)

    period = 1.0 / FPS
    while True:
        t0 = time.time()
        frame = grabber.latest()
        if frame is None:
            time.sleep(0.5)
            continue
        dets = [Detection(points=np.array([[x1, y1], [x2, y2]]), scores=np.array([s, s]))
                for x1, y1, x2, y2, s in detector.detect(frame)]
        now = datetime.now()
        people, new = [], []
        for t in tracker.update(detections=dets):
            if t.last_detection is None or t.id is None:
                continue
            (x1, y1), (x2, y2) = t.estimate.astype(int)
            num, is_new = numbers.number_for(t.id, now)
            score = float(t.last_detection.scores[0])
            people.append({"number": num, "box": [int(x1), int(y1), int(x2), int(y2)], "score": score})
            if is_new:
                new.append((num, score))
        annotated = draw(frame, people)
        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        with STATE.lock:
            STATE.jpeg, STATE.people, STATE.updated = (buf.tobytes() if ok else STATE.jpeg), people, time.time()

        for num, score in new:
            log.info("New person: Person %d (%.0f%%)", num, score * 100)
            db.execute("INSERT INTO people VALUES (?,?,?,?)", (now.isoformat(timespec="seconds"), CAMERA, num, score))
        if new:
            db.commit()
            if ok and policy.should_alert(now):
                nums = ", ".join(f"Person {n}" for n, _ in new)
                send_ntfy(f"👤 {nums} seen on {CAMERA}",
                          f"{len(people)} people in view · {now:%d %b %H:%M:%S}", buf.tobytes())
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
