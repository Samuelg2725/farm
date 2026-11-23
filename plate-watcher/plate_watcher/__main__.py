"""Listens to Frigate over MQTT, checks plates against the whitelist, alerts on unknowns."""
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

import paho.mqtt.client as mqtt
import requests
import yaml

from .matching import Vehicle, find_match, normalise
from .tracker import Tracker

log = logging.getLogger("plate-watcher")

MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
FRIGATE_URL = os.getenv("FRIGATE_URL", "http://localhost:5000").rstrip("/")
NTFY_URL = os.getenv("NTFY_URL", "").rstrip("/")   # e.g. http://ntfy/farm-alerts
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
WHITELIST_PATH = Path(os.getenv("WHITELIST_PATH", "whitelist.yaml"))
DB_PATH = os.getenv("DB_PATH", "sightings.db")
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.7"))
DECISION_DELAY = float(os.getenv("DECISION_DELAY", "20"))
ALERT_COOLDOWN = float(os.getenv("ALERT_COOLDOWN", "600"))
MAX_DISTANCE = int(os.getenv("MATCH_DISTANCE", "1"))
# Our own plate reading on full-res frames (set to 0 to rely only on Frigate's built-in LPR)
OWN_ALPR = os.getenv("OWN_ALPR", "1") == "1"
STREAM_URL = os.getenv("STREAM_URL", "rtsp://frigate:8554/{camera}")  # 4K restream from Frigate
DETECT_SIZE = tuple(int(x) for x in os.getenv("DETECT_SIZE", "1280x720").split("x"))
GRAB_INTERVAL = float(os.getenv("GRAB_INTERVAL", "2"))
MAX_GRABS = int(os.getenv("MAX_GRABS", "5"))
VEHICLE_LABELS = {"car", "truck", "motorcycle", "bus"}


class Whitelist:
    def __init__(self, path: Path):
        self.path, self._mtime, self.vehicles = path, 0.0, []

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


def open_db(path: str) -> sqlite3.Connection:
    db = sqlite3.connect(path, check_same_thread=False)
    db.execute("""CREATE TABLE IF NOT EXISTS sightings (
        ts TEXT, camera TEXT, event_id TEXT, plate TEXT, score REAL,
        known INTEGER, owner TEXT)""")
    return db


def send_alert(event_id: str, camera: str, plate: str, score: float) -> None:
    text = (f"🚨 Unknown vehicle\nPlate: {plate}  ({score:.0%} confidence)\n"
            f"Camera: {camera}\nTime: {datetime.now():%d %b %Y %H:%M:%S}")
    log.warning(text.replace("\n", " | "))
    if not (NTFY_URL or (TG_TOKEN and TG_CHAT)):
        return
    photo = b""
    try:
        snap = requests.get(f"{FRIGATE_URL}/api/events/{event_id}/snapshot.jpg",
                            params={"bbox": 1}, timeout=10)
        photo = snap.content if snap.ok else b""
    except Exception:
        log.exception("Could not fetch snapshot")
    if NTFY_URL:
        send_ntfy(text, photo)
    if TG_TOKEN and TG_CHAT:
        send_telegram(text, photo)


def send_ntfy(text: str, photo: bytes) -> None:
    title, _, body = text.partition("\n")
    headers = {"Title": title.encode(), "Priority": "high", "Tags": "rotating_light"}
    try:
        if photo:
            headers.update({"Message": body.replace("\n", " | ").encode(), "Filename": "vehicle.jpg"})
            requests.put(NTFY_URL, data=photo, headers=headers, timeout=20).raise_for_status()
        else:
            requests.post(NTFY_URL, data=body.encode(), headers=headers, timeout=10).raise_for_status()
    except Exception:
        log.exception("ntfy alert failed")


def send_telegram(text: str, photo: bytes) -> None:
    api = f"https://api.telegram.org/bot{TG_TOKEN}"
    try:
        if photo:
            requests.post(f"{api}/sendPhoto", data={"chat_id": TG_CHAT, "caption": text},
                          files={"photo": ("snap.jpg", photo)}, timeout=20).raise_for_status()
            return
    except Exception:
        log.exception("Telegram photo failed, falling back to text")
    try:
        requests.post(f"{api}/sendMessage", data={"chat_id": TG_CHAT, "text": text},
                      timeout=10).raise_for_status()
    except Exception:
        log.exception("Telegram alert failed")


def parse_message(topic: str, payload: dict):
    """Yield ('read', id, camera, plate, score) and ('end', id) tuples."""
    if topic.endswith("tracked_object_update") and payload.get("type") == "lpr":
        yield ("read", payload.get("id"), payload.get("camera", ""),
               payload.get("plate") or "", float(payload.get("score") or 0))
    elif topic.endswith("/events"):
        after = payload.get("after") or {}
        plate = after.get("recognized_license_plate")
        score = after.get("recognized_license_plate_score") or 0
        if isinstance(plate, (list, tuple)):  # some versions send [plate, score]
            plate, score = (list(plate) + [None, 0])[:2]
        if plate:
            yield ("read", after.get("id"), after.get("camera", ""), str(plate), float(score or 0))
        if (after.get("label") in VEHICLE_LABELS and payload.get("type") != "end"
                and not after.get("stationary") and after.get("box")):
            yield ("vehicle", after.get("id"), after.get("camera", ""), after["box"])
        if payload.get("type") == "end":
            yield ("end", after.get("id"))


class GrabScheduler:
    """Decides when to grab another full-res frame for a moving vehicle."""

    def __init__(self, interval: float, max_grabs: int):
        self.interval, self.max_grabs = interval, max_grabs
        self.state: dict[str, tuple[float, int]] = {}

    def should_grab(self, event_id: str, now: float) -> bool:
        last, count = self.state.get(event_id, (-1e9, 0))
        if count >= self.max_grabs or now - last < self.interval:
            return False
        self.state[event_id] = (now, count + 1)
        if len(self.state) > 2000:
            self.state.clear()
        return True


def alpr_worker(jobs: "queue.Queue", tracker: Tracker, lock: threading.Lock) -> None:
    from .reader import PlateReader, crop_box, grab_frame
    reader = PlateReader()
    while True:
        event_id, camera, box = jobs.get()
        try:
            frame = grab_frame(STREAM_URL.format(camera=camera))
            if frame is None:
                log.warning("Could not grab frame from %s", camera)
                continue
            for plate, score in reader.read(crop_box(frame, box, DETECT_SIZE)):
                log.info("Read %s (%.2f) on %s", plate, score, event_id)
                with lock:
                    tracker.add_read(event_id, camera, plate, score, time.time())
        except Exception:
            log.exception("Plate reading failed")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    whitelist = Whitelist(WHITELIST_PATH)
    whitelist.get()
    db = open_db(DB_PATH)
    tracker = Tracker(delay=DECISION_DELAY, min_score=MIN_SCORE)
    lock = threading.Lock()
    last_alert: dict[str, float] = {}
    grabs = GrabScheduler(GRAB_INTERVAL, MAX_GRABS)
    jobs: queue.Queue = queue.Queue(maxsize=20)
    if OWN_ALPR:
        threading.Thread(target=alpr_worker, args=(jobs, tracker, lock), daemon=True).start()

    def on_connect(client, userdata, flags, reason_code, properties=None):
        log.info("Connected to MQTT (%s)", reason_code)
        client.subscribe([("frigate/events", 0), ("frigate/tracked_object_update", 0)])

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload)
        except ValueError:
            return
        with lock:
            for item in parse_message(msg.topic, payload):
                if item[0] == "read" and item[1]:
                    tracker.add_read(item[1], item[2], item[3], item[4], time.time())
                elif item[0] == "vehicle" and OWN_ALPR and grabs.should_grab(item[1], time.time()):
                    try:
                        jobs.put_nowait(item[1:])
                    except queue.Full:
                        pass
                elif item[0] == "end":
                    tracker.mark_ended(item[1])

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="plate-watcher")
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect_async(MQTT_HOST, 1883)
    client.loop_start()

    while True:
        time.sleep(1)
        with lock:
            due = tracker.due(time.time())
        for p in due:
            plate = normalise(p.plate)
            match = find_match(plate, whitelist.get(), MAX_DISTANCE)
            db.execute("INSERT INTO sightings VALUES (?,?,?,?,?,?,?)",
                       (datetime.now().isoformat(timespec="seconds"), p.camera, p.event_id,
                        plate, p.score, int(bool(match)), match.owner if match else ""))
            db.commit()
            if match:
                log.info("Known vehicle %s (%s)", plate, match.owner)
                continue
            if time.time() - last_alert.get(plate, 0) < ALERT_COOLDOWN:
                log.info("Unknown %s again – within cooldown, not re-alerting", plate)
                continue
            last_alert[plate] = time.time()
            send_alert(p.event_id, p.camera, plate, p.score)


if __name__ == "__main__":
    main()
