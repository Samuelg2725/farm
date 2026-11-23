"""Vision service entry point.

Pipeline per frame of the camera's main stream:
  1. YOLOX detects people and vehicles in a single pass.
  2. Norfair tracks each object so it keeps a stable identifier.
  3. People: RTMPose keypoints and motion analysis classify behaviour.
     Vehicles: FastALPR reads the number plate from a full-resolution crop and
     the result is checked against the whitelist.
  4. Events are logged to SQLite, clips are recorded, and notifications are sent via ntfy.

The web dashboard is served on PORT (default 8090).
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

from .alerts import Notifier, Switches
from .behaviour import LABELS, BehaviourAnalyzer, CameraMotion
from .clips import phone_clip, plate_photo
from .db import DB
from .detector import PERSON, VEHICLES, Detector
from .health import HealthMonitor, disk_space
from .logic import DailyNumbers, LagMeter, PersonAlerter, in_hours, parse_hours
from .plateinfo import describe, estimate_colour
from .plates import PlateBook, PlateReader, Whitelist, crop_scaled
from .pose import SKELETON, PoseEstimator
from .recorder import Recorder, free_space
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
PERSON_ALERTS = env("PERSON_ALERTS", "1") == "1"            # initial state of the person-alert switch
PERSON_ALERT_HOURS = env("PERSON_ALERT_HOURS", "always")     # active hours for person alerts
PERSON_ALERT_GAP = float(env("PERSON_ALERT_GAP", "20"))      # minimum interval between arrival notifications (s)
PERSON_BEHAVIOUR_COOLDOWN = float(env("PERSON_BEHAVIOUR_COOLDOWN", "60"))  # per person and behaviour (s)
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
LOW_SCORE = float(env("LOW_SCORE", "0.25"))  # lower threshold, applied only to existing tracks
LOITER_SECONDS = float(env("LOITER_SECONDS", "120"))
BEHAVIOUR_ALERTS = {b.strip() for b in env("BEHAVIOUR_ALERTS", "lying down,climbing,restricted zone,loitering").split(",")}
BEHAVIOUR_ALERT_COOLDOWN = float(env("BEHAVIOUR_ALERT_COOLDOWN", "120"))
ZONES_PATH = env("ZONES_PATH", "/settings/zones.yaml")
SEND_CLIPS = env("SEND_CLIPS", "1") == "1"               # attach the recording after an alert
CLIP_MAX_SECONDS = float(env("CLIP_MAX_SECONDS", "90"))
CLIP_MAX_MB = float(env("CLIP_MAX_MB", "14"))            # ntfy.sh attachment limit is 15 MB
OFFLINE_ALERT_AFTER = float(env("OFFLINE_ALERT_AFTER", "60"))  # seconds without video before notifying
DISK_MIN_FREE_GB = float(env("DISK_MIN_FREE_GB", "10"))         # below this, notify and remove oldest recordings
STALE_AFTER = 5.0  # seconds without a new frame before the feed is treated as interrupted
GONE_AFTER = 5.0  # seconds unseen before a track's log entry is closed

PERSON_COLOURS = [(0, 200, 255), (255, 120, 0), (200, 0, 255), (255, 255, 0), (255, 0, 160), (120, 255, 120)]
VEHICLE_COLOURS = {"known": (0, 200, 0), "unknown": (0, 0, 255), "checking": (0, 215, 255), "no-plate": (180, 180, 180)}


# ---------------------------------------------------------------- video input
class FrameGrabber(threading.Thread):
    """Reads the stream continuously and retains only the latest frame, so processing never falls behind live."""

    def __init__(self, url: str):
        super().__init__(daemon=True)
        self.url, self.frame, self.lock = url, None, threading.Lock()
        self.fps, self.seq, self.lag, self.skips, self.codec = 0.0, 0, 0.0, 0, ""
        self.last_frame_time: float | None = None

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
                    self.last_frame_time = now
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

    def stale(self, now: float | None = None) -> bool:
        t = self.last_frame_time
        return t is None or (now or time.time()) - t > STALE_AFTER

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
    """Return a copy of `frame` resized to `width` with zones, boxes, labels and skeletons drawn.
    Box and keypoint coordinates are in `frame` pixels."""
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
    """Renders annotated frames for the live view and the annotated recording."""
    period, last_seq, last_raw, last_offline = 1.0 / DISPLAY_FPS, -1, 0.0, 0.0
    while True:
        t0 = time.time()
        frame, seq = grabber.latest()
        if grabber.stale(t0) and t0 - last_offline >= 1:
            last_offline = t0
            with state.lock:
                state.jpeg = offline_image(frame, grabber.last_frame_time)
        if frame is not None and seq != last_seq and not grabber.stale(t0):
            last_seq = seq
            if t0 - last_raw > 2:  # unannotated still used by the zone editor
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


def offline_image(last_frame: np.ndarray | None, last_time: float | None) -> bytes:
    """Live-view placeholder shown while the camera feed is interrupted."""
    h = int(DISPLAY_WIDTH * 9 / 16)
    if last_frame is not None:
        img = cv2.resize(last_frame, (DISPLAY_WIDTH, int(last_frame.shape[0] * DISPLAY_WIDTH / last_frame.shape[1])))
        img = (img * 0.35).astype(np.uint8)
    else:
        img = np.zeros((h, DISPLAY_WIDTH, 3), np.uint8)
    scale = img.shape[0] / 720
    lines = ["CAMERA OFFLINE - reconnecting"]
    if last_time is not None:
        lines.append(f"last frame {datetime.fromtimestamp(last_time):%d %b %H:%M:%S}")
    for i, text in enumerate(lines):
        size = (1.4 if i == 0 else 0.8) * scale
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, size, 3)
        org = ((img.shape[1] - tw) // 2, img.shape[0] // 2 + int(i * 60 * scale))
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, size, (60, 60, 255) if i == 0 else (220, 220, 220), 3)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return buf.tobytes() if ok else b""


# ---------------------------------------------------------------- health
def health_loop(grabber: FrameGrabber, notifier: Notifier, state: LiveState, interval: float = 5.0) -> None:
    """Periodic health checks; sends notifications on state changes and frees disk space when low."""
    monitor = HealthMonitor(offline_alert_after=OFFLINE_ALERT_AFTER, min_free_gb=DISK_MIN_FREE_GB)
    while True:
        try:
            try:
                free_gb, total_gb = disk_space(RECORDINGS_DIR)
            except OSError:
                free_gb = total_gb = None
            for a in monitor.check(time.time(), grabber.last_frame_time, grabber.skips, free_gb, total_gb, CAMERA):
                log.warning("Health: %s - %s", a.title, a.body)
                notifier.send(a.title, a.body, tags=a.tags, priority=a.priority)
            if monitor.status.get("low_disk"):
                free_space(RECORDINGS_DIR, DISK_MIN_FREE_GB)
            with state.lock:
                state.health = dict(monitor.status)
        except Exception:
            log.exception("Health check failed")
        time.sleep(interval)


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
                for plate, score, country, country_conf, box in reads:
                    log.info("Vehicle track %d: read %s (%.0f%%)", tid, plate, score * 100)
                    better = score >= info.score
                    book.add_read(tid, plate, score, time.time(), jpg.tobytes() if ok else b"", country, country_conf,
                                  plate_photo(crop, box, plate) if better else b"")
        except Exception:
            log.exception("Plate reading failed")


def plate_fields(info) -> dict:
    d = describe(info.plate, info.ocr_country, info.ocr_country_conf) if info.plate else {}
    return {"plate": info.plate, "plate_conf": round(info.score, 3) if info.plate else None,
            "country": d.get("country", ""), "plate_format": d.get("format", ""), "region": d.get("region", ""),
            "registered": d.get("registered", ""), "colour": info.colour, "status": info.status, "owner": info.owner}


# ---------------------------------------------------------------- clip notifications
def send_clip(notifier: Notifier, db: DB, clip: str, seconds: float) -> None:
    """Send the annotated recording of a clip that produced an alert, reduced to fit the attachment limit."""
    from pathlib import Path
    try:
        src = Path(RECORDINGS_DIR) / clip
        small = phone_clip(src, src.with_name(src.stem + "_phone.mp4"), CLIP_MAX_SECONDS, CLIP_MAX_MB)
        if small is None:
            return
        rows = db.query("SELECT kind, number, label, plate, status FROM events WHERE clip = ? ORDER BY id", clip)
        seen = []
        for r in rows:
            if r["kind"] == "person":
                seen.append(f"Person {r['number']}")
            elif r["kind"] == "vehicle":
                seen.append(f"Vehicle {r['number']}" + (f" {r['plate']}" + (" UNKNOWN" if r["status"] == "unknown" else "")
                                                         if r["plate"] else ""))
            elif r["kind"] == "behaviour":
                seen.append(f"Person {r['number']} {r['label']}")
        cut = " (first %d s)" % CLIP_MAX_SECONDS if seconds > CLIP_MAX_SECONDS else ""
        notifier.send(f"Video: {Path(clip).stem.replace('-', ':')} ({int(seconds)} s){cut}",
                      ", ".join(dict.fromkeys(seen))[:600] or "Recording", small.read_bytes(),
                      "movie_camera", priority="default", filename=Path(clip).stem + ".mp4")
    except Exception:
        log.exception("Could not send video clip")


# ---------------------------------------------------------------- helpers
def to_detections(dets):
    return [Detection(points=np.array([[x1, y1], [x2, y2]]), scores=np.array([s, s]), label=c)
            for x1, y1, x2, y2, s, c in dets]


def tracked(tracks, current=None, last_fresh=None, now=0.0, keep=1.0):
    """Yield (track, box, score, fresh) for active tracks.

    `fresh` is True when the track was matched to a detection in the current frame. Unmatched
    tracks are shown for up to `keep` seconds to avoid flicker, then suppressed."""
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
    last_fresh: dict[int, float] = {}    # person track id -> time of last matched detection
    last_fresh_v: dict[int, float] = {}  # vehicle track id -> time of last matched detection
    last_state: dict[int, dict] = {}     # person track id -> last behaviour state

    hit_max = int(FPS * 3)
    people_tracker = Tracker(distance_function="iou", distance_threshold=0.75, initialization_delay=2, hit_counter_max=hit_max)
    vehicle_tracker = Tracker(distance_function="iou", distance_threshold=0.75, initialization_delay=2, hit_counter_max=hit_max)
    person_nums, vehicle_nums = DailyNumbers(), DailyNumbers()
    switches = Switches(DATA_DIR, {"person_alerts": PERSON_ALERTS})
    alerter = PersonAlerter(delay=2.0, gap=PERSON_ALERT_GAP, behaviour_cooldown=PERSON_BEHAVIOUR_COOLDOWN)
    alert_hours = parse_hours(PERSON_ALERT_HOURS)
    whitelist = Whitelist(WHITELIST_PATH)
    whitelist.get()
    book, book_lock = PlateBook(min_score=PLATE_MIN_SCORE), threading.Lock()
    last_plate_alert: dict[str, float] = {}

    recorder = Recorder(STREAM_URL, RECORDINGS_DIR, post_roll=RECORD_POST_ROLL, keep_days=RECORD_KEEP_DAYS,
                        keep_raw=RECORD_KEEP_RAW) if RECORD else None
    rec_row = None
    clip_alerted = False   # whether the current clip produced an alert
    # track id -> {"id": event row, "max": best confidence, "seen": last seen}
    people_rows: dict[int, dict] = {}
    vehicle_rows: dict[int, dict] = {}

    state = LiveState()
    serve(App(state, db, notifier, RECORDINGS_DIR, CAMERA, RECORD_KEEP_DAYS, zone_store, switches), PORT)
    log.info("Dashboard on port %d", PORT)
    jobs: queue.Queue = queue.Queue(maxsize=8)
    if PLATES:
        threading.Thread(target=plate_worker, args=(jobs, book, book_lock), daemon=True).start()
    grabber = FrameGrabber(STREAM_URL)
    grabber.start()
    threading.Thread(target=render_loop, args=(grabber, state, recorder), daemon=True).start()
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    threading.Thread(target=health_loop, args=(grabber, notifier, state), daemon=True).start()

    period, last_log = 1.0 / FPS, 0.0
    while True:
        t0 = time.time()
        frame, _ = grabber.latest()
        if frame is None or grabber.stale(t0):
            # feed interrupted: clear the scene and let any recording close after its post-roll
            with state.lock:
                state.people, state.vehicles = [], []
            if recorder is not None and recorder.recording:
                change = recorder.update(datetime.now(), False)
                if change and change[0] == "stopped":
                    if clip_alerted and SEND_CLIPS:
                        threading.Thread(target=send_clip, args=(notifier, db, change[1]["file"], change[1]["seconds"]),
                                         daemon=True).start()
                    if rec_row:
                        db.update("recordings", rec_row, end=stamp(datetime.now()))
                        rec_row = None
                with state.lock:
                    state.recording = recorder.clip
            time.sleep(0.5)
            continue
        if t0 - last_log > 60:
            last_log = t0
            log.info("Stream %dx%d at %.1f fps, %.1f s behind live, %d skips",
                     frame.shape[1], frame.shape[0], grabber.fps, grabber.lag, grabber.skips)
        dets = detector.detect(frame)
        now, ts = datetime.now(), time.time()
        # Low-confidence person detections are kept only when they overlap an existing track.
        # This maintains tracking through postures the detector is less certain about (e.g. after a fall).
        person_dets = [d for d in dets if d[5] == PERSON and
                       (d[4] >= MIN_SCORE or any(iou(d[:4], b) > 0.3 for b in prev_person_boxes))]
        dets = [d for d in dets if d[5] != PERSON and d[4] >= MIN_SCORE]

        # ---- tracking
        people, vehicles, decided, behaviour_events = [], [], [], []
        zones = zone_store.get()
        pdets = to_detections(person_dets)
        for t, box, score, fresh in tracked(people_tracker.update(pdets), pdets, last_fresh, ts):
            num, _ = person_nums.number_for(t.id, now)
            people.append({"number": num, "box": box, "score": score, "tid": t.id, "fresh": fresh})
        prev_person_boxes = [p["box"] for p in people if p["fresh"]]
        seen = [p for p in people if p["fresh"]]  # behaviour is derived from matched detections only
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
            if not p["fresh"]:  # temporarily unmatched: show last known state
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

        # ---- recording (started before logging so new events reference the clip)
        if recorder is not None:
            thumb = b""
            if (people or vehicles) and not recorder.recording:
                recorder.codec = grabber.codec or "h264"
                ok, b = cv2.imencode(".jpg", draw(frame, people, vehicles, 640), [cv2.IMWRITE_JPEG_QUALITY, 80])
                thumb = b.tobytes() if ok else b""
            change = recorder.update(now, bool(people or vehicles), thumb)
            if change and change[0] == "started":
                rec_row = db.insert("recordings", start=stamp(now), **change[1])
                clip_alerted = False
            elif change and change[0] == "stopped" and clip_alerted and SEND_CLIPS:
                threading.Thread(target=send_clip, args=(notifier, db, change[1]["file"], change[1]["seconds"]),
                                 daemon=True).start()
            if change and change[0] == "stopped" and rec_row:
                counts = db.query("SELECT SUM(kind='person') p, SUM(kind='vehicle') v, "
                                  "GROUP_CONCAT(DISTINCT plate) plates FROM events WHERE clip = ?", change[1]["file"])[0]
                db.update("recordings", rec_row, end=stamp(now), people=counts["p"] or 0,
                          vehicles=counts["v"] or 0, plates=counts["plates"] or "")
                rec_row = None
            with state.lock:
                state.recording = recorder.clip
        clip = recorder.clip if recorder is not None else ""

        # ---- event log: one row per tracked person / vehicle
        new_people = []
        for p in people:
            row = people_rows.get(p["tid"])
            if row is None:
                rid = db.insert("events", ts=stamp(now), last_seen=stamp(now), camera=CAMERA, kind="person",
                                number=p["number"], label="person", confidence=round(p["score"], 3), clip=clip)
                people_rows[p["tid"]] = {"id": rid, "max": p["score"], "seen": ts}
                new_people.append(p["number"])
                alerter.arrived(p["tid"], p["number"], ts)
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

        # vehicles no longer in view: final plate decision
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

        # plate decisions: log and notify
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
                              + (f"\n{extra}" if extra else "") + ("\nVideo follows when the clip ends." if clip and SEND_CLIPS else ""),
                              info.photo_jpeg or info.crop_jpeg, "rotating_light")
                clip_alerted = True

        # close log entries for tracks no longer in view
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

        # ---- behaviour events
        alerts_on = switches.get("person_alerts") and in_hours(alert_hours, now.time())
        behaviour_alerts = []
        for p, b in behaviour_events:
            zone = next((z for z in zones if b == f"entered {z.name}"), None)
            label = LABELS.get(b, b)
            row = people_rows.get(p["tid"])
            serious = (zone is not None and zone.kind == "restricted") or b in ("lying down", "climbing")
            db.insert("events", ts=stamp(now), last_seen=stamp(now), camera=CAMERA, kind="behaviour",
                      number=p["number"], label=label, confidence=round(p["score"], 3), clip=clip,
                      details=(f"{zone.kind} zone" if zone else p.get("doing", "")), status="alert" if serious else "")
            if row is not None:
                row.setdefault("actions", [])
                if label not in row["actions"]:
                    row["actions"].append(label)
                    db.update("events", row["id"], actions=", ".join(row["actions"]))
            log.info("Person %d: %s", p["number"], label)
            kind = "restricted zone" if zone and zone.kind == "restricted" else b
            if zone is not None and zone.kind != "restricted":
                continue  # watch zones are logged without notification
            # notify, unless a pending arrival notification will already report it
            if alerts_on and (kind in BEHAVIOUR_ALERTS or not alerter.is_pending(p["tid"])) \
                    and alerter.behaviour_ok(p["number"], b, ts):
                behaviour_alerts.append((p, label, serious))

        def photo() -> bytes:
            ok, buf = cv2.imencode(".jpg", draw(frame, people, vehicles, zones=state.zones),
                                   [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            return buf.tobytes() if ok else b""

        def describe_person(p) -> str:
            extra = [p.get("doing") or ""] + [f"in {z}" for z in p.get("zones", [])]
            return f"Person {p['number']}: " + (", ".join(e for e in extra if e) or "seen") + f" ({p['score']:.0%} sure)"

        if behaviour_alerts:
            serious = any(s_ for _, _, s_ in behaviour_alerts)
            title = " · ".join(f"Person {p['number']} {lab}" for p, lab, _ in behaviour_alerts)
            if serious:
                title = "⚠ " + title
            clip_alerted = True
            notifier.send(title, "\n".join(describe_person(p) for p in people if p.get("fresh"))
                          + f"\n{CAMERA} - {now:%d %b %H:%M:%S}", photo(), "warning" if serious else "bust_in_silhouette",
                          priority="urgent" if serious else "high")

        # ---- arrival notifications (grouped, with current behaviour and annotated image)
        due = alerter.due(ts, {p["tid"] for p in people})
        if due and alerts_on:
            by_tid = {p["tid"]: p for p in people}
            who = [by_tid[tid] for tid, _ in due if tid in by_tid]
            if who:
                if len(who) == 1:
                    title = f"Person seen: Person {who[0]['number']}" + (f" - {who[0]['doing']}" if who[0].get("doing") else "")
                else:
                    title = f"{len(who)} people seen: " + ", ".join(f"Person {p['number']}" for p in who)
                body = "\n".join(describe_person(p) for p in who)
                clip_alerted = True
                notifier.send(title, body + f"\n{len(people)} in view on {CAMERA} - {now:%d %b %H:%M:%S}"
                              + ("\nVideo follows when the clip ends." if clip and SEND_CLIPS else ""),
                              photo(), "bust_in_silhouette")

        time.sleep(max(0.0, period - (time.time() - t0)))


if __name__ == "__main__":
    main()
