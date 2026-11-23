"""Farm vision: people tracking + vehicle number plates, at the same time, with a dashboard.

4K camera stream → YOLOX finds people AND vehicles in one pass → Norfair follows each one
→ people: "Person 1, 2, 3…"   vehicles: "Vehicle 1" + plate (FastALPR on a 4K crop) + KNOWN/UNKNOWN
→ everything logged (with confidence %), recorded (with boxes) and alerted to phones (ntfy).
Dashboard: http://<host>:8090
"""
from __future__ import annotations

import logging
import os
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from norfair import Detection, Tracker

from .alerts import Notifier
from .behaviour import LABELS, BehaviourAnalyzer, CameraMotion
from .db import DB
from .detector import PERSON, VEHICLES, Detector
from .logic import AlertPolicy, DailyNumbers, LagMeter
from .plateinfo import describe, estimate_colour
from .plates import PlateBook, PlateReader, Whitelist, crop_scaled
from .pose import SKELETON, PoseEstimator
from .recorder import Recorder
from .web import App, LiveState, serve
from .zones import ZoneStore

log = logging.getLogger("vision")

env = os.getenv
CAMERA = env("CAMERA", "yard_ptz")
STREAM_URL = env("STREAM_URL", "rtsp://frigate:8554/yard_ptz")
MODEL_PATH = env("MODEL_PATH", "/models/yolox_s.onnx")
FPS = float(env("FPS", "5"))
MIN_SCORE = float(env("MIN_SCORE", "0.45"))
DISPLAY_WIDTH = int(env("DISPLAY_WIDTH", "1920"))
DISPLAY_FPS = float(env("DISPLAY_FPS", "12"))
JPEG_QUALITY = int(env("JPEG_QUALITY", "85"))
MAX_LAG = float(env("MAX_LAG", "2"))
PERSON_ALERT_HOURS = env("PERSON_ALERT_HOURS", "always")
PERSON_ALERT_COOLDOWN = float(env("PERSON_ALERT_COOLDOWN", "300"))
PLATE_ALERT_COOLDOWN = float(env("PLATE_ALERT_COOLDOWN", "3600"))
PLATE_MIN_SCORE = float(env("PLATE_MIN_SCORE", "0.7"))
PLATES = env("PLATES", "1") == "1"
NTFY_SERVER = env("NTFY_SERVER", "https://ntfy.sh")
NTFY_TOPIC = env("NTFY_TOPIC", "")
NTFY_PUBLIC_URL = env("NTFY_PUBLIC_URL", "")
WHITELIST_PATH = env("WHITELIST_PATH", "/settings/whitelist.yaml")
DATA_DIR = env("DATA_DIR", "/data")
DB_PATH = env("DB_PATH", os.path.join(DATA_DIR, "vision.db"))
PORT = int(env("PORT", "8090"))
RECORD = env("RECORD", "1") == "1"
RECORD_KEEP_RAW = env("RECORD_KEEP_RAW", "1") == "1"
RECORDINGS_DIR = env("RECORDINGS_DIR", "/recordings")
RECORD_POST_ROLL = float(env("RECORD_POST_ROLL", "10"))
RECORD_KEEP_DAYS = int(env("RECORD_KEEP_DAYS", "30"))
POSE = env("POSE", "1") == "1"
POSE_MODEL = env("POSE_MODEL", "/models/rtmpose-m.onnx")
LOW_SCORE = float(env("LOW_SCORE", "0.25"))  # weaker detections accepted only for people already tracked
LOITER_SECONDS = float(env("LOITER_SECONDS", "120"))
BEHAVIOUR_ALERTS = {b.strip() for b in env("BEHAVIOUR_ALERTS", "lying down,climbing,restricted zone,loitering").split(",")}
BEHAVIOUR_ALERT_COOLDOWN = float(env("BEHAVIOUR_ALERT_COOLDOWN", "120"))
ZONES_PATH = env("ZONES_PATH", "/settings/zones.yaml")
GONE_AFTER = 5.0  # seconds out of view before a person/vehicle's log entry is finalised

PERSON_COLOURS = [(0, 200, 255), (255, 120, 0), (200, 0, 255), (255, 255, 0), (255, 0, 160), (120, 255, 120)]
VEHICLE_COLOURS = {"known": (0, 200, 0), "unknown": (0, 0, 255), "checking": (0, 215, 255), "no-plate": (180, 180, 180)}


# ---------------------------------------------------------------- video in
class FrameGrabber(threading.Thread):
    """Reads the camera continuously and keeps only the newest frame (so we never lag behind live)."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url, self.frame, self.lock = url, None, threading.Lock()
        self.fps, self.seq, self.lag, self.skips, self.codec = 0.0, 0, 0.0, 0, ""

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
            time.sleep(1)

    def latest(self):
        with self.lock:
            return self.frame, self.seq


# ---------------------------------------------------------------- drawing
def _label(img, text, x, y, colour, scale):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
    ty = max(th + 6, y)
    cv2.rectangle(img, (x, ty - th - 6), (x + tw + 6, ty), colour, -1)
    cv2.putText(img, text, (x + 3, ty - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, (0, 0, 0), 2)


def draw(frame: np.ndarray, people: list[dict], vehicles: list[dict], width: int = DISPLAY_WIDTH,
         zones: list[dict] | None = None) -> np.ndarray:
    """New image `width` wide with every box drawn (boxes are in `frame` pixels)."""
    f = width / frame.shape[1]
    img = cv2.resize(frame, (width, int(round(frame.shape[0] * f))),
                     interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
    scale = img.shape[0] / 720
    thick = max(2, int(2 * scale))
    if zones:
        overlay = img.copy()
        for z in zones:
            pts = (np.array(z["points"]) * [img.shape[1], img.shape[0]]).astype(np.int32)
            colour = (0, 0, 255) if z["type"] == "restricted" else (0, 200, 255)
            cv2.fillPoly(overlay, [pts], colour)
            cv2.polylines(img, [pts], True, colour, thick)
            cv2.putText(img, z["name"], tuple(pts[0] + [6, int(22 * scale)]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6 * scale, colour, 2)
        img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    for v in vehicles:
        x1, y1, x2, y2 = (int(c * f) for c in v["box"])
        c = VEHICLE_COLOURS[v["status"]]
        cv2.rectangle(img, (x1, y1), (x2, y2), c, thick + 1)
        text = f"Vehicle {v['number']} {int(v['score'] * 100)}%"
        if v["plate"]:
            text += f"  {v['plate']}"
        text += {"known": f"  OK {v['owner']}".rstrip(), "unknown": "  UNKNOWN!",
                 "checking": "  checking...", "no-plate": ""}[v["status"]]
        _label(img, text, x1, y1, c, scale)
    for p in people:
        x1, y1, x2, y2 = (int(c * f) for c in p["box"])
        c = PERSON_COLOURS[p["number"] % len(PERSON_COLOURS)]
        if p.get("kpts"):
            k = [(int(x * f), int(y * f), s) for x, y, s in p["kpts"]]
            for a, b in SKELETON:
                if k[a][2] >= 0.4 and k[b][2] >= 0.4:
                    cv2.line(img, k[a][:2], k[b][:2], c, max(1, thick - 1), cv2.LINE_AA)
            for x, y, s in k:
                if s >= 0.4:
                    cv2.circle(img, (x, y), max(2, thick), c, -1, cv2.LINE_AA)
        alarm = bool(p.get("alerts")) or bool(p.get("restricted"))
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255) if alarm else c, thick + (2 if alarm else 0))
        text = f"Person {p['number']} {int(p['score'] * 100)}%"
        if p.get("doing"):
            text += f"  {p['doing']}"
        _label(img, text, x1, y1, (0, 0, 255) if alarm else c, scale)
        if alarm:
            warn = ", ".join([LABELS.get(a, a) for a in p.get("alerts", [])] + [f"in {z}" for z in p.get("restricted", [])])
            _label(img, warn.upper() + "!", x1, y2 + int(26 * scale), (0, 0, 255), scale)
    banner = f"{datetime.now():%d %b %H:%M:%S}   People: {len(people)}   Vehicles: {len(vehicles)}"
    for colour, w in (((0, 0, 0), 4), ((255, 255, 255), 2)):
        cv2.putText(img, banner, (8, int(24 * scale)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, colour, w)
    return img


def render_loop(grabber: FrameGrabber, state: LiveState, recorder: Recorder | None) -> None:
    """Draws the latest boxes on the newest frame → live view + recording with boxes."""
    period, last_seq, last_raw = 1.0 / DISPLAY_FPS, -1, 0.0
    while True:
        t0 = time.time()
        frame, seq = grabber.latest()
        if frame is not None and seq != last_seq:
            last_seq = seq
            if t0 - last_raw > 2:  # plain picture (no boxes) for drawing zones on the dashboard
                last_raw = t0
                small = cv2.resize(frame, (DISPLAY_WIDTH, int(frame.shape[0] * DISPLAY_WIDTH / frame.shape[1])),
                                   interpolation=cv2.INTER_AREA)
                ok_r, raw = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok_r:
                    with state.lock:
                        state.raw_jpeg = raw.tobytes()
            with state.lock:
                people, vehicles, zones = state.people, state.vehicles, state.zones
            ok, buf = cv2.imencode(".jpg", draw(frame, people, vehicles, zones=zones),
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if ok:
                jpeg = buf.tobytes()
                with state.lock:
                    state.jpeg, state.stream_fps, state.lag = jpeg, grabber.fps, grabber.lag
                if recorder is not None and recorder.recording:
                    recorder.feed(jpeg)
        time.sleep(max(0.0, period - (time.time() - t0)))


# ---------------------------------------------------------------- plates
def plate_worker(jobs: queue.Queue, book: PlateBook, lock: threading.Lock) -> None:
    reader = PlateReader()
    while True:
        tid, crop = jobs.get()
        try:
            colour = estimate_colour(crop)
            ok, jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            reads = reader.read(crop)
            with lock:
                info = book.get(tid, time.time())
                if colour and not info.colour:
                    info.colour = colour
                for plate, score, country, country_conf in reads:
                    log.info("Vehicle track %d: read %s (%.0f%%)", tid, plate, score * 100)
                    book.add_read(tid, plate, score, time.time(), jpg.tobytes() if ok else b"", country, country_conf)
        except Exception:
            log.exception("Plate reading failed")


def plate_fields(info) -> dict:
    d = describe(info.plate, info.ocr_country, info.ocr_country_conf) if info.plate else {}
    return {"plate": info.plate, "plate_conf": round(info.score, 3) if info.plate else None,
            "country": d.get("country", ""), "plate_format": d.get("format", ""), "region": d.get("region", ""),
            "registered": d.get("registered", ""), "colour": info.colour, "status": info.status, "owner": info.owner}


# ---------------------------------------------------------------- helpers
def to_detections(dets):
    return [Detection(points=np.array([[x1, y1], [x2, y2]]), scores=np.array([s, s]), label=c)
            for x1, y1, x2, y2, s, c in dets]


def tracked(tracks, current=None, last_fresh=None, now=0.0, keep=1.0):
    """Tracked objects to show. A track not matched to a detection this frame (person just left or
    hidden) is kept for `keep` seconds so boxes don't flicker, then hidden. Yields (track, box, score, fresh)."""
    ids = {id(d) for d in current} if current is not None else None
    for t in tracks:
        if t.last_detection is None or t.id is None:
            continue
        fresh = ids is None or id(t.last_detection) in ids
        if last_fresh is not None:
            if fresh:
                last_fresh[t.id] = now
            elif now - last_fresh.get(t.id, now) > keep:
                continue
        (x1, y1), (x2, y2) = t.estimate.astype(int)
        yield t, [int(x1), int(y1), int(x2), int(y2)], float(t.last_detection.scores[0]), fresh


def duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds} s" if seconds < 60 else f"{seconds // 60} min" if seconds < 3600 else f"{seconds // 3600} h {seconds % 3600 // 60} min"


def iou(a, b) -> float:
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def stamp(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


# ---------------------------------------------------------------- main
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(DATA_DIR, exist_ok=True)
    db = DB(DB_PATH)
    notifier = Notifier(NTFY_SERVER, NTFY_TOPIC, db, DATA_DIR, NTFY_PUBLIC_URL)
    log.info("Phone alerts go to %s (subscribe in the ntfy app: %s)", notifier.url, notifier.subscribe_url)
    detector = Detector(MODEL_PATH, conf=min(MIN_SCORE, LOW_SCORE))
    log.info("Detector ready – people + vehicles")
    pose = None
    if POSE and os.path.exists(POSE_MODEL):
        pose = PoseEstimator(POSE_MODEL)
        log.info("Pose model ready – behaviour detection on")
    elif POSE:
        log.warning("Pose model %s missing – behaviour detection uses movement only", POSE_MODEL)
    analyzer = BehaviourAnalyzer(loiter_seconds=LOITER_SECONDS)
    camera_motion = CameraMotion()
    zone_store = ZoneStore(ZONES_PATH)
    last_behaviour_alert: dict[str, float] = {}
    prev_person_boxes: list = []
    last_fresh: dict[int, float] = {}    # people: track id → last time the detector really saw them
    last_fresh_v: dict[int, float] = {}  # vehicles: same
    last_state: dict[int, dict] = {}     # people: last behaviour state, shown while briefly hidden

    hit_max = int(FPS * 3)
    people_tracker = Tracker(distance_function="iou", distance_threshold=0.75, initialization_delay=2, hit_counter_max=hit_max)
    vehicle_tracker = Tracker(distance_function="iou", distance_threshold=0.75, initialization_delay=2, hit_counter_max=hit_max)
    person_nums, vehicle_nums = DailyNumbers(), DailyNumbers()
    person_policy = AlertPolicy(PERSON_ALERT_HOURS, PERSON_ALERT_COOLDOWN)
    whitelist = Whitelist(WHITELIST_PATH)
    whitelist.get()
    book, book_lock = PlateBook(min_score=PLATE_MIN_SCORE), threading.Lock()
    last_plate_alert: dict[str, float] = {}

    recorder = Recorder(STREAM_URL, RECORDINGS_DIR, post_roll=RECORD_POST_ROLL, keep_days=RECORD_KEEP_DAYS,
                        keep_raw=RECORD_KEEP_RAW) if RECORD else None
    rec_row = None
    # open log rows for people/vehicles currently being tracked: track id -> {"id", "max", "seen"}
    people_rows: dict[int, dict] = {}
    vehicle_rows: dict[int, dict] = {}

    state = LiveState()
    serve(App(state, db, notifier, RECORDINGS_DIR, CAMERA, RECORD_KEEP_DAYS, zone_store), PORT)
    log.info("Dashboard on port %d", PORT)
    jobs: queue.Queue = queue.Queue(maxsize=8)
    if PLATES:
        threading.Thread(target=plate_worker, args=(jobs, book, book_lock), daemon=True).start()
    grabber = FrameGrabber(STREAM_URL)
    grabber.start()
    threading.Thread(target=render_loop, args=(grabber, state, recorder), daemon=True).start()

    period, last_log = 1.0 / FPS, 0.0
    while True:
        t0 = time.time()
        frame, _ = grabber.latest()
        if frame is None:
            time.sleep(0.5)
            continue
        if t0 - last_log > 60:
            last_log = t0
            log.info("Stream %dx%d at %.1f fps, %.1f s behind live, %d skips",
                     frame.shape[1], frame.shape[0], grabber.fps, grabber.lag, grabber.skips)
        dets = detector.detect(frame)
        now, ts = datetime.now(), time.time()
        # strong detections always count; weak ones only if they overlap someone already tracked
        # (keeps hold of a person who falls or crouches, when the detector is less sure)
        person_dets = [d for d in dets if d[5] == PERSON and
                       (d[4] >= MIN_SCORE or any(iou(d[:4], b) > 0.3 for b in prev_person_boxes))]
        dets = [d for d in dets if d[5] != PERSON and d[4] >= MIN_SCORE]

        # ---- track people + vehicles
        people, vehicles, decided, behaviour_events = [], [], [], []
        zones = zone_store.get()
        pdets = to_detections(person_dets)
        for t, box, score, fresh in tracked(people_tracker.update(pdets), pdets, last_fresh, ts):
            num, _ = person_nums.number_for(t.id, now)
            people.append({"number": num, "box": box, "score": score, "tid": t.id, "fresh": fresh})
        prev_person_boxes = [p["box"] for p in people if p["fresh"]]
        seen = [p for p in people if p["fresh"]]  # pose + behaviour only from what the detector saw this frame
        poses = pose.estimate(frame, [p["box"] for p in seen]) if pose is not None and seen else []
        cam_offset, jumped = camera_motion.update(frame)
        if jumped:
            analyzer.reset_motion()
        for i, p in enumerate(seen):
            pts, sc = poses[i] if i < len(poses) else (None, None)
            st, new = analyzer.update(p["tid"], ts, p["box"], pts, sc, (frame.shape[1], frame.shape[0]),
                                      zones, now.time(), cam_offset)
            p.update(doing=st["doing"], alerts=st["alerts"], zones=st["zones"], seconds=st["seconds"], speed=st["speed"],
                     restricted=[z.name for z in zones if z.name in st["zones"] and z.kind == "restricted"])
            if pts is not None:
                p["kpts"] = [[round(float(x), 1), round(float(y), 1), round(float(s), 2)] for (x, y), s in zip(pts, sc)]
            last_state[p["tid"]] = {k: p[k] for k in ("doing", "alerts", "zones", "seconds", "speed", "restricted", "kpts") if k in p}
            behaviour_events += [(p, b) for b in new]
        for p in people:
            if not p["fresh"]:  # briefly hidden: keep showing what they were last doing
                p.update(last_state.get(p["tid"], {}))
                p.pop("kpts", None)
        vdets = to_detections([d for d in dets if d[5] in VEHICLES])
        vtracks = list(tracked(vehicle_tracker.update(vdets), vdets, last_fresh_v, ts))
        with book_lock:
            wl = whitelist.get()
            for t, box, score, fresh in vtracks:
                num, _ = vehicle_nums.number_for(t.id, now)
                if PLATES and fresh and not jobs.full() and book.want_grab(t.id, ts):
                    crop = crop_scaled(frame, box, (frame.shape[1], frame.shape[0])).copy()
                    if crop.size:
                        jobs.put_nowait((t.id, crop))
                info = book.get(t.id, ts)
                if book.decide(t.id, ts, wl):
                    decided.append((t.id, num, info))
                vehicles.append({"number": num, "box": box, "score": score, "tid": t.id,
                                 "kind": VEHICLES[t.last_detection.label], **plate_fields(info)})

        with state.lock:
            state.people, state.vehicles = people, vehicles
            state.zones = [{"name": z.name, "type": z.kind, "points": z.points} for z in zones]

        # ---- recording (start before logging so new rows point at the clip)
        if recorder is not None:
            thumb = b""
            if (people or vehicles) and not recorder.recording:
                recorder.codec = grabber.codec or "h264"
                ok, b = cv2.imencode(".jpg", draw(frame, people, vehicles, 640), [cv2.IMWRITE_JPEG_QUALITY, 80])
                thumb = b.tobytes() if ok else b""
            change = recorder.update(now, bool(people or vehicles), thumb)
            if change and change[0] == "started":
                rec_row = db.insert("recordings", start=stamp(now), **change[1])
            elif change and change[0] == "stopped" and rec_row:
                counts = db.query("SELECT SUM(kind='person') p, SUM(kind='vehicle') v, "
                                  "GROUP_CONCAT(DISTINCT plate) plates FROM events WHERE clip = ?", change[1]["file"])[0]
                db.update("recordings", rec_row, end=stamp(now), people=counts["p"] or 0,
                          vehicles=counts["v"] or 0, plates=counts["plates"] or "")
                rec_row = None
            with state.lock:
                state.recording = recorder.clip
        clip = recorder.clip if recorder is not None else ""

        # ---- activity log: one row per person / vehicle, kept up to date
        new_people = []
        for p in people:
            row = people_rows.get(p["tid"])
            if row is None:
                rid = db.insert("events", ts=stamp(now), last_seen=stamp(now), camera=CAMERA, kind="person",
                                number=p["number"], label="person", confidence=round(p["score"], 3), clip=clip)
                people_rows[p["tid"]] = {"id": rid, "max": p["score"], "seen": ts}
                new_people.append(p["number"])
                log.info("New person: Person %d (%.0f%%)", p["number"], p["score"] * 100)
            else:
                row["seen"], row["max"] = ts, max(row["max"], p["score"])
        for v in vehicles:
            row = vehicle_rows.get(v["tid"])
            if row is None:
                rid = db.insert("events", ts=stamp(now), last_seen=stamp(now), camera=CAMERA, kind="vehicle",
                                number=v["number"], label=v["kind"], confidence=round(v["score"], 3), clip=clip,
                                status=v["status"])
                vehicle_rows[v["tid"]] = {"id": rid, "max": v["score"], "seen": ts}
                log.info("New vehicle: Vehicle %d %s (%.0f%%)", v["number"], v["kind"], v["score"] * 100)
            else:
                row["seen"], row["max"] = ts, max(row["max"], v["score"])

        # vehicles gone for a while: final plate decision (or record that no plate was read)
        with book_lock:
            current = {v["tid"] for v in vehicles}
            for tid in [k for k in book.items
                        if k not in current and ts - vehicle_rows.get(k, {}).get("seen", 0) > GONE_AFTER]:
                info = book.items[tid]
                if book.decide(tid, ts, wl, lost=True):
                    decided.append((tid, vehicle_nums.map.get(tid, 0), info))
                elif tid in vehicle_rows:
                    db.update("events", vehicle_rows[tid]["id"], **plate_fields(info))
                book.forget(tid)

        # plate decisions → log + alert
        for tid, num, info in decided:
            fields = plate_fields(info)
            if tid in vehicle_rows:
                db.update("events", vehicle_rows[tid]["id"], **fields)
            log.info("Vehicle %d plate %s → %s %s", num, info.plate, "KNOWN" if info.known else "UNKNOWN", info.owner)
            if not info.known and ts - last_plate_alert.get(info.plate, -1e18) >= PLATE_ALERT_COOLDOWN:
                last_plate_alert[info.plate] = ts
                extra = " - ".join(filter(None, [
                    fields["country"], fields["region"] and f"registered in {fields['region']}",
                    fields["registered"] and f"plate issued {fields['registered']}", fields["colour"]]))
                notifier.send(f"UNKNOWN VEHICLE: {info.plate}",
                              f"Vehicle {num} on {CAMERA} - plate {info.score:.0%} sure - {now:%d %b %H:%M:%S}"
                              + (f"\n{extra}" if extra else ""), info.crop_jpeg, "rotating_light")

        # people / vehicles out of view for a while → finalise their log rows
        for rows in (people_rows, vehicle_rows):
            for tid in [k for k, r in rows.items() if ts - r["seen"] > GONE_AFTER]:
                r = rows.pop(tid)
                if rows is people_rows:
                    analyzer.forget(tid)
                    last_state.pop(tid, None)
                    last_fresh.pop(tid, None)
                else:
                    last_fresh_v.pop(tid, None)
                db.update("events", r["id"], confidence=round(r["max"], 3),
                          last_seen=stamp(datetime.fromtimestamp(r["seen"])))

        # ---- behaviours: log each one, add to the person's row, alert on the serious ones
        for p, b in behaviour_events:
            zone = next((z for z in zones if b == f"entered {z.name}"), None)
            label = LABELS.get(b, b)
            row = people_rows.get(p["tid"])
            db.insert("events", ts=stamp(now), last_seen=stamp(now), camera=CAMERA, kind="behaviour",
                      number=p["number"], label=label, confidence=round(p["score"], 3), clip=clip,
                      details=(f"{zone.kind} zone" if zone else p.get("doing", "")),
                      status=("alert" if (zone and zone.kind == "restricted") or b in ("lying down", "climbing") else ""))
            if row is not None:
                row.setdefault("actions", [])
                if label not in row["actions"]:
                    row["actions"].append(label)
                    db.update("events", row["id"], actions=", ".join(row["actions"]))
            log.info("Person %d: %s", p["number"], label)
            kind = "restricted zone" if zone and zone.kind == "restricted" else b
            if kind in BEHAVIOUR_ALERTS:
                # safety alerts are per person; loitering / zone alerts once per cooldown (no spam for groups)
                key = f"{kind}:{p['number']}" if kind in ("lying down", "climbing") else f"{kind}:{b}"
                if ts - last_behaviour_alert.get(key, -1e18) >= BEHAVIOUR_ALERT_COOLDOWN:
                    last_behaviour_alert[key] = ts
                    title = {"lying down": f"Possible fall: Person {p['number']} lying down",
                             "climbing": f"Person {p['number']} climbing",
                             "loitering": f"Loitering: Person {p['number']} in view for {duration(p.get('seconds', 0))}",
                             }.get(kind, f"Person {p['number']} entered {zone.name if zone else ''}".strip())
                    ok, b2 = cv2.imencode(".jpg", draw(frame, people, vehicles, zones=state.zones),
                                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                    notifier.send(title, f"{CAMERA} - {p['score']:.0%} sure it's a person - {now:%d %b %H:%M:%S}",
                                  b2.tobytes() if ok else b"", "warning")

        # person alert
        if new_people and person_policy.should_alert(now):
            ok, b = cv2.imencode(".jpg", draw(frame, people, vehicles, zones=state.zones),
                                 [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            best = max(p["score"] for p in people if p["number"] in new_people)
            notifier.send(f"Person seen: {', '.join(f'Person {n}' for n in new_people)}",
                          f"{CAMERA} - {len(people)} people in view - {best:.0%} sure - {now:%d %b %H:%M:%S}",
                          b.tobytes() if ok else b"", "bust_in_silhouette")

        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
