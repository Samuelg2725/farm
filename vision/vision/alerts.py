"""Notifications via ntfy (self-hosted or ntfy.sh) and persisted dashboard switches."""
from __future__ import annotations

import base64
import io
import logging
import secrets
from datetime import datetime
from pathlib import Path

import requests

log = logging.getLogger("vision")

GUESSABLE = {"", "farm-alerts", "alerts", "farm", "test"}


def resolve_topic(server: str, topic: str, data_dir: str) -> str:
    """Return the topic to publish to.

    On a public server a guessable topic would expose notifications, so a random topic is
    generated once and persisted to <data_dir>/ntfy_topic.txt."""
    public = not server.startswith("http://ntfy")  # the bundled ntfy container
    if not public or topic.strip().lower() not in GUESSABLE:
        return topic.strip() or "farm-alerts"
    f = Path(data_dir) / "ntfy_topic.txt"
    try:
        saved = f.read_text().strip()
        if saved:
            return saved
    except OSError:
        pass
    new = f"farm-{secrets.token_hex(6)}"
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(new)
    except OSError:
        log.exception("Could not save generated ntfy topic")
    log.info("Generated private alert topic: %s", new)
    return new


def header_value(text: str) -> str:
    """Encode a value for an HTTP header: newlines as the literal "\\n" (interpreted by ntfy)
    and non-ASCII text using RFC 2047."""
    text = text.replace("\r", "").replace("\n", "\\n")
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        return "=?UTF-8?B?" + base64.b64encode(text.encode()).decode() + "?="


class Notifier:
    def __init__(self, server: str, topic: str, db=None, data_dir: str = "/data", public_url: str = ""):
        self.server = server.rstrip("/")
        self.topic = resolve_topic(self.server, topic, data_dir)
        self.db = db
        # server address as reached from client devices
        self.phone_server = (public_url or self.server).rstrip("/")

    @property
    def url(self) -> str:
        return f"{self.server}/{self.topic}"

    @property
    def subscribe_url(self) -> str:
        return f"{self.phone_server}/{self.topic}"

    def send(self, title: str, body: str, jpeg: bytes = b"", tags: str = "", priority: str = "high",
             filename: str = "alert.jpg") -> bool:
        """Publish a notification. `jpeg` is an optional attachment; `filename` sets its name and type."""
        headers = {"Title": header_value(title), "Priority": priority}
        if tags:
            headers["Tags"] = tags
        ok, err = True, ""
        try:
            if jpeg:
                headers.update({"Message": header_value(body), "Filename": filename})
                r = requests.put(self.url, data=jpeg, headers=headers, timeout=20 + len(jpeg) / 100_000)
            else:
                r = requests.post(self.url, data=body.encode(), headers=headers, timeout=10)
            r.raise_for_status()
        except Exception as e:  # network or server error
            ok, err = False, str(e)[:300]
            log.warning("Alert not delivered: %s", err)
        log.info("ALERT %s: %s – %s", "sent" if ok else "FAILED", title, body)
        if self.db is not None:
            self.db.insert("alerts", ts=datetime.now().isoformat(timespec="seconds"),
                           title=title, body=body, ok=int(ok), error=err)
        return ok

    def qr_svg(self) -> str:
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(self.subscribe_url, image_factory=qrcode.image.svg.SvgPathImage, box_size=8, border=2)
        buf = io.BytesIO()
        img.save(buf)
        return buf.getvalue().decode()


class Switches:
    """Boolean settings changed from the dashboard and persisted to <data_dir>/switches.json."""

    def __init__(self, data_dir: str, defaults: dict):
        import json
        import threading
        self._json, self.lock = json, threading.Lock()
        self.path = Path(data_dir) / "switches.json"
        self.values = dict(defaults)
        try:
            self.values.update({k: bool(v) for k, v in json.loads(self.path.read_text()).items() if k in defaults})
        except (OSError, ValueError):
            pass

    def get(self, name: str) -> bool:
        with self.lock:
            return bool(self.values.get(name))

    def set(self, name: str, value: bool) -> None:
        with self.lock:
            self.values[name] = bool(value)
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self.path.write_text(self._json.dumps(self.values))
            except OSError:
                log.exception("Could not save switches")
        log.info("Switch %s → %s", name, "ON" if value else "OFF")
