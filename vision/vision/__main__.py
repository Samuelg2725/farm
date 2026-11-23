"""Farm vision: people tracking + vehicle number plate checks, at the same time, on one live view.

4K camera stream → YOLOX finds people AND vehicles in one pass → Norfair follows each one
→ people get boxes "Person 1, 2, 3…"; vehicles get boxes "Vehicle 1" + plate + KNOWN/UNKNOWN
(plates are read by FastALPR from the same 4K frame, cropped round the vehicle).
The live view is the full-quality picture scaled to 1080p, refreshed faster than detection runs.
Live view: http://<host>:8090   Alerts: ntfy.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import requests
from norfair import Detection, Tracker

from .detector import PERSON, VEHICLES, Detector
from .logic import AlertPolicy, DailyNumbers, LagMeter
from .plates import PlateBook, PlateReader, Whitelist, crop_scaled
from .recorder import Recorder

log = logging.getLogger("vision")

CAMERA = os.getenv("CAMERA", "yard_ptz")
STREAM_URL = os.getenv("STREAM_URL", "rtsp://frigate:8554/yard_ptz")  # 4K main stream
DISPLAY_WIDTH = int(os.getenv("DISPLAY_WIDTH", "1920"))     # live view size (1920 = 1080p)
DISPLAY_FPS = float(os.getenv("DISPLAY_FPS", "12"))         # live view refresh rate
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
MAX_LAG = float(os.getenv("MAX_LAG", "2"))  # seconds behind live before we skip ahead (reconnect)
MODEL_PATH = os.getenv("MODEL_PATH", "/models/yolox_s.onnx")
FPS = float(os.getenv("FPS", "5"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.45"))
PERSON_ALERT_HOURS = os.getenv("PERSON_ALERT_HOURS", "always")
PERSON_ALERT_COOLDOWN = float(os.getenv("PERSON_ALERT_COOLDOWN", "300"))
PLATE_ALERT_COOLDOWN = float(os.getenv("PLATE_ALERT_COOLDOWN", "3600"))  # same unknown plate
PLATE_MIN_SCORE = float(os.getenv("PLATE_MIN_SCORE", "0.7"))
OWN_ALPR = os.getenv("PLATES", "1") == "1"
NTFY_URL = os.getenv("NTFY_URL", "").rstrip("/")
WHITELIST_PATH = os.getenv("WHITELIST_PATH", "/settings/whitelist.yaml")
DB_PATH = os.getenv("DB_PATH", "/data/vision.db")
PORT = int(os.getenv("PORT", "8090"))
RECORD = os.getenv("RECORD", "1") == "1"
RECORDINGS_DIR = os.getenv("RECORDINGS_DIR", "/recordings")
RECORD_POST_ROLL = float(os.getenv("RECORD_POST_ROLL", "10"))   # keep recording this long after they leave
RECORD_KEEP_DAYS = int(os.getenv("RECORD_KEEP_DAYS", "30"))

PERSON_COLOURS = [(0, 200, 255), (255, 120, 0), (200, 0, 255), (255, 255, 0), (255, 0, 160), (120, 255, 120)]
VEHICLE_COLOURS = {"known": (0, 200, 0), "unknown": (0, 0, 255), "checking": (0, 215, 255), "no-plate": (180, 180, 180)}


# ---------------------------------------------------------------- video in
class FrameGrabber(threading.Thread):
    """Keeps only the newest frame so processing never lags behind live."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url, self.frame, self.lock = url, None, threading.Lock()
        self.frames, self.fps, self.seq, self.lag, self.skips = 0, 0.0, 0, 0.0, 0
        self.codec = ""

    def run(self):
        while True:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log.warning("Cannot open %s – retrying in 5 s", self.url)
                time.sleep(5)
                continue
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
            self.codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ").lower()
            log.info("Connected to stream %s (%s %dx%d)", self.url, self.codec,
                     int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            t_count, n, meter = time.time(), 0, LagMeter()
            while True:
                ok, frame = cap.read()
                if not ok:
                    log.warning("Stream dropped – reconnecting")
                    break
                now = time.time()
                with self.lock:
                    self.frame = frame
                    self.seq += 1
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if pts > 0:
                    self.lag = meter.update(pts, now)
                    if self.lag > MAX_LAG:
                        self.skips += 1
                        log.warning("Fell %.1f s behind live (decoding %.1f fps) – skipping ahead. "
                                    "If this repeats, lower the camera's frame rate.", self.lag, self.fps)
                        self.lag = 0.0
                        break
                n += 1
                if now - t_count >= 10:
                    self.fps, t_count, n = n / (now - t_count), now, 0
            cap.release()
            time.sleep(2)

    def latest(self):
        """Newest frame (read-only – never drawn on) and its sequence number."""
        with self.lock:
            return self.frame, self.seq


# ---------------------------------------------------------------- drawing
def _label(frame, text, x, y, colour, scale):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
    ty = max(th + 6, y)
    cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 6, ty), colour, -1)
    cv2.putText(frame, text, (x + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), 2)


def draw(frame: np.ndarray, people: list[dict], vehicles: list[dict], width: int = DISPLAY_WIDTH) -> np.ndarray:
    """Returns a NEW image `width` wide with boxes drawn (boxes are in `frame` pixels)."""
    f = width / frame.shape[1]
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR  # INTER_AREA keeps downscaled 4K sharp
    frame = cv2.resize(frame, (width, int(round(frame.shape[0] * f))), interpolation=interp)
    scale = frame.shape[0] / 720
    thick = max(2, int(2 * scale))
    for v in vehicles:
        x1, y1, x2, y2 = (int(c * f) for c in v["box"])
        c = VEHICLE_COLOURS[v["status"]]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick + 1)
        text = f"Vehicle {v['number']}"
        if v["plate"]:
            text += f"  {v['plate']}"
        text += {"known": f"  OK {v['owner']}".rstrip(), "unknown": "  UNKNOWN!",
                 "checking": "  checking...", "no-plate": ""}[v["status"]]
        _label(frame, text, x1, y1, c, scale)
    for p in people:
        x1, y1, x2, y2 = (int(c * f) for c in p["box"])
        c = PERSON_COLOURS[p["number"] % len(PERSON_COLOURS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, thick)
        _label(frame, f"Person {p['number']}", x1, y1, c, scale)
    banner = f"{datetime.now():%d %b %H:%M:%S}   People: {len(people)}   Vehicles: {len(vehicles)}"
    for colour, w in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(frame, banner, (8, int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, colour, w)
    return frame


# ---------------------------------------------------------------- alerts
def send_ntfy(title: str, body: str, jpeg: bytes, tags: str) -> None:
    log.warning("ALERT: %s – %s", title, body)
    if not NTFY_URL:
        return
    try:
        requests.put(NTFY_URL, data=jpeg, timeout=20, headers={
            "Title": title.encode(), "Message": body.encode(), "Filename": "alert.jpg",
            "Priority": "high", "Tags": tags}).raise_for_status()
    except Exception:
        log.exception("ntfy alert failed")


# ---------------------------------------------------------------- web
class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.jpeg: bytes = b""
        self.people: list[dict] = []
        self.vehicles: list[dict] = []
        self.stream_fps = 0.0
        self.lag = 0.0


def render_loop(grabber: "FrameGrabber") -> None:
    """Draws the latest boxes on the newest camera frame for a smooth, sharp live view."""
    period, last_seq = 1.0 / DISPLAY_FPS, -1
    while True:
        t0 = time.time()
        frame, seq = grabber.latest()
        if frame is not None and seq != last_seq:
            last_seq = seq
            with STATE.lock:
                people, vehicles = STATE.people, STATE.vehicles
            ok, buf = cv2.imencode(".jpg", draw(frame, people, vehicles), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                with STATE.lock:
                    STATE.jpeg, STATE.stream_fps, STATE.lag = buf.tobytes(), grabber.fps, grabber.lag
        time.sleep(max(0.0, period - (time.time() - t0)))


STATE = State()
RECORDER: Recorder | None = None


def recordings_page() -> str:
    import html
    clips = RECORDER.list_clips() if RECORDER else []
    rows, day = [], None
    for c in clips:
        if c["day"] != day:
            day = c["day"]
            rows.append(f"<h3>{html.escape(day)}</h3>")
        name = html.escape(c["name"].replace("_", " · ").replace("-", ":", 2))
        thumb = f'<img src="/rec/{c["thumb"]}" loading="lazy">' if c["thumb"] else ""
        live = ' <b style="color:#f44">● recording now</b>' if c["live"] else ""
        rows.append(f'<a class="clip" href="/rec/{c["file"]}" target="_blank">{thumb}<div>{name}{live}<br>'
                    f'<small>{c["size_mb"]} MB</small></div></a>')
    body = "".join(rows) or "<p>No recordings yet – they appear here when a person or vehicle is seen.</p>"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Farm Recordings</title>
<style>body{{margin:0;padding:10px 16px;background:#111;color:#eee;font-family:system-ui,sans-serif}}
a{{color:#8cf}}.clip{{display:inline-block;width:300px;margin:6px;vertical-align:top;color:#eee;text-decoration:none}}
.clip img{{width:300px;border-radius:4px;display:block}}</style></head><body>
<p><a href="/">← Live view</a></p><h2>Recordings</h2>
<p><small>Full-quality camera video, saved whenever a person or vehicle is in view. Click to play.
Kept {RECORD_KEEP_DAYS} days.</small></p>{body}</body></html>"""

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Farm Live View</title>
<style>body{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}
header{padding:10px 16px;font-size:18px}img{width:100%;max-width:1920px;display:block;margin:auto}
.row{padding:6px 16px;font-size:15px}.unknown{color:#ff5a5a;font-weight:bold}.known{color:#5ad15a}</style></head><body>
<header>Farm live view – <span id="count">…</span> <small id="health" style="color:#999"></small>
<span id="rec" style="color:#ff4040;font-weight:bold"></span>
<a href="/recordings" style="float:right;color:#8cf">Recordings</a></header>
<img src="/stream.mjpg" alt="live view"><div class="row" id="people"></div><div class="row" id="vehicles"></div>
<script>
async function tick(){try{const s=await (await fetch('/status.json')).json();
document.getElementById('count').textContent=s.people.length+' people, '+s.vehicles.length+' vehicles';
document.getElementById('health').textContent='· camera '+s.stream_fps+' fps · delay '+s.lag+' s';
document.getElementById('rec').textContent=s.recording?' ● REC':'';
document.getElementById('people').textContent='People: '+(s.people.map(p=>'Person '+p.number).join(' - ')||'none');
document.getElementById('vehicles').innerHTML='Vehicles: '+(s.vehicles.map(v=>'<span class="'+v.status+'">Vehicle '+v.number+
 (v.plate?' '+v.plate:'')+' ('+v.status+(v.owner?' – '+v.owner:'')+')</span>').join(' - ')||'none');}catch(e){}}
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
                body = json.dumps({"camera": CAMERA, "people": STATE.people, "vehicles": STATE.vehicles,
                                   "stream_fps": round(STATE.stream_fps, 1), "lag": round(STATE.lag, 1),
                                   "recording": RECORDER is not None and RECORDER.recording})
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
        elif self.path == "/recordings":
            self._send(200, "text/html; charset=utf-8", recordings_page().encode())
        elif self.path.startswith("/rec/") and RECORDER is not None:
            self._send_file(self.path[len("/rec/"):])
        else:
            self._send(404, "text/plain", b"not found")

    def _send_file(self, rel: str):
        """Serves a recording, with Range support so browsers can play/seek video."""
        from urllib.parse import unquote
        base = Path(RECORDINGS_DIR).resolve()
        f = (base / unquote(rel)).resolve()
        if base not in f.parents or not f.is_file() or f.suffix not in (".mp4", ".jpg"):
            return self._send(404, "text/plain", b"not found")
        size = f.stat().st_size
        ctype = "video/mp4" if f.suffix == ".mp4" else "image/jpeg"
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes="):
            a, _, b = rng[6:].partition("-")
            start = int(a) if a else max(0, size - int(b))
            end = min(int(b), size - 1) if (a and b) else end
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        try:
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------- plate worker
def plate_worker(jobs: queue.Queue, book: PlateBook, lock: threading.Lock) -> None:
    reader = PlateReader()
    while True:
        tid, crop = jobs.get()
        try:
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            for plate, score in reader.read(crop):
                log.info("Vehicle track %d: read %s (%.2f)", tid, plate, score)
                with lock:
                    book.add_read(tid, plate, score, time.time(), jpg.tobytes() if ok else b"")
        except Exception:
            log.exception("Plate reading failed")


def to_detections(dets):
    return [Detection(points=np.array([[x1, y1], [x2, y2]]), scores=np.array([s, s]), label=c)
            for x1, y1, x2, y2, s, c in dets]


def tracked(tracks):
    for t in tracks:
        if t.last_detection is not None and t.id is not None:
            (x1, y1), (x2, y2) = t.estimate.astype(int)
            yield t, [int(x1), int(y1), int(x2), int(y2)], float(t.last_detection.scores[0])


# ---------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    detector = Detector(MODEL_PATH, conf=MIN_SCORE)
    log.info("Detector ready – people + vehicles (%s)", MODEL_PATH)

    hit_max = int(FPS * 3)
    people_tracker = Tracker(distance_function="iou", distance_threshold=0.75, initialization_delay=2, hit_counter_max=hit_max)
    vehicle_tracker = Tracker(distance_function="iou", distance_threshold=0.75, initialization_delay=2, hit_counter_max=hit_max)
    person_nums, vehicle_nums = DailyNumbers(), DailyNumbers()
    person_policy = AlertPolicy(PERSON_ALERT_HOURS, PERSON_ALERT_COOLDOWN)
    whitelist = Whitelist(WHITELIST_PATH)
    whitelist.get()
    book = PlateBook(min_score=PLATE_MIN_SCORE)
    book_lock = threading.Lock()
    last_plate_alert: dict[str, float] = {}

    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    db = sqlite3.connect(DB_PATH)
    db.execute("CREATE TABLE IF NOT EXISTS people (ts TEXT, camera TEXT, number INTEGER, score REAL)")
    db.execute("CREATE TABLE IF NOT EXISTS vehicles (ts TEXT, camera TEXT, number INTEGER, plate TEXT, "
               "score REAL, known INTEGER, owner TEXT)")

    jobs: queue.Queue = queue.Queue(maxsize=8)
    if OWN_ALPR:
        threading.Thread(target=plate_worker, args=(jobs, book, book_lock), daemon=True).start()
    global RECORDER
    if RECORD:
        RECORDER = Recorder(STREAM_URL, RECORDINGS_DIR, post_roll=RECORD_POST_ROLL, keep_days=RECORD_KEEP_DAYS)
    grabber = FrameGrabber(STREAM_URL)
    grabber.start()
    threading.Thread(target=render_loop, args=(grabber,), daemon=True).start()
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever, daemon=True).start()
    log.info("Live view on port %d", PORT)

    period, last_log = 1.0 / FPS, 0.0
    while True:
        t0 = time.time()
        frame, _ = grabber.latest()
        if frame is None:
            time.sleep(0.5)
            continue
        if t0 - last_log > 60:
            last_log = t0
            log.info("Stream %dx%d at %.1f fps, %.1f s behind live, %d skips", frame.shape[1], frame.shape[0],
                     grabber.fps, grabber.lag, grabber.skips)
        dets = detector.detect(frame)
        now, ts = datetime.now(), time.time()

        # ---- people
        people, new_people = [], []
        for t, box, score in tracked(people_tracker.update(to_detections([d for d in dets if d[5] == PERSON]))):
            num, is_new = person_nums.number_for(t.id, now)
            people.append({"number": num, "box": box, "score": score})
            if is_new:
                new_people.append(num)

        # ---- vehicles + plates
        vehicles, decided = [], []
        seen = set()
        vtracks = list(tracked(vehicle_tracker.update(to_detections([d for d in dets if d[5] in VEHICLES]))))
        with book_lock:
            wl = whitelist.get()
            for t, box, score in vtracks:
                seen.add(t.id)
                num, _ = vehicle_nums.number_for(t.id, now)
                if OWN_ALPR and not jobs.full() and book.want_grab(t.id, ts):
                    crop = crop_scaled(frame, box, (frame.shape[1], frame.shape[0])).copy()
                    if crop.size:
                        jobs.put_nowait((t.id, crop))
                info = book.get(t.id, ts)
                if book.decide(t.id, ts, wl):
                    decided.append((num, info))
                vehicles.append({"number": num, "box": box, "score": score, "kind": VEHICLES[t.last_detection.label],
                                 "plate": info.plate, "status": info.status, "owner": info.owner})
            for tid in [k for k in book.items if k not in seen]:  # vehicle left the view
                info = book.items[tid]
                if book.decide(tid, ts, wl, lost=True):
                    decided.append((vehicle_nums.map.get(tid, 0), info))
                book.forget(tid)

        with STATE.lock:
            STATE.people, STATE.vehicles = people, vehicles
        jpeg = b""
        if RECORDER is not None:
            labels = ({"person"} if people else set()) | ({"vehicle"} if vehicles else set())
            snap = b""
            if labels and not RECORDER.recording:
                RECORDER.codec = grabber.codec or "h264"
                ok_s, b_s = cv2.imencode(".jpg", draw(frame, people, vehicles, 640), [cv2.IMWRITE_JPEG_QUALITY, 80])
                snap = b_s.tobytes() if ok_s else b""
            RECORDER.update(now, labels, snap)

        if new_people or decided:  # picture for alerts
            ok, buf = cv2.imencode(".jpg", draw(frame, people, vehicles), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            jpeg = buf.tobytes() if ok else b""

        # ---- logging + alerts
        stamp = now.isoformat(timespec="seconds")
        for num in new_people:
            log.info("New person: Person %d", num)
            db.execute("INSERT INTO people VALUES (?,?,?,?)", (stamp, CAMERA, num, 0.0))
        if new_people and jpeg and person_policy.should_alert(now):
            send_ntfy(f"Person seen: {', '.join(f'Person {n}' for n in new_people)}",
                      f"{CAMERA} - {len(people)} people in view - {now:%d %b %H:%M:%S}", jpeg, "bust_in_silhouette")
        for num, info in decided:
            log.info("Vehicle %d plate %s → %s %s", num, info.plate, "KNOWN" if info.known else "UNKNOWN", info.owner)
            db.execute("INSERT INTO vehicles VALUES (?,?,?,?,?,?,?)",
                       (stamp, CAMERA, num, info.plate, info.score, int(info.known), info.owner))
            if not info.known and ts - last_plate_alert.get(info.plate, -1e18) >= PLATE_ALERT_COOLDOWN:
                last_plate_alert[info.plate] = ts
                send_ntfy(f"UNKNOWN VEHICLE: {info.plate}",
                          f"Vehicle {num} on {CAMERA} - {info.score:.0%} confidence - {now:%d %b %H:%M:%S}",
                          info.crop_jpeg or jpeg, "rotating_light")
        if new_people or decided:
            db.commit()
        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
