"""SQLite storage for events, recordings and notifications."""
from __future__ import annotations

import sqlite3
import threading

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,            -- when first seen
    last_seen TEXT,
    camera TEXT,
    kind TEXT NOT NULL,          -- person | vehicle | behaviour
    number INTEGER,              -- Person 3 / Vehicle 2 (per day)
    label TEXT,                  -- person / car / truck / bus / motorcycle
    confidence REAL,             -- best detection confidence 0..1
    plate TEXT, plate_conf REAL,
    country TEXT, plate_format TEXT, region TEXT, registered TEXT, colour TEXT,
    status TEXT,                 -- vehicles: known | unknown | checking | no-plate
    owner TEXT,
    clip TEXT,                   -- recording file this happened in
    actions TEXT,                -- people: what they were seen doing (running, crouching…)
    details TEXT                 -- behaviours: extra info
);
CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
CREATE TABLE IF NOT EXISTS recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    start TEXT NOT NULL, end TEXT,
    file TEXT, raw_file TEXT, thumb TEXT,
    people INTEGER DEFAULT 0, vehicles INTEGER DEFAULT 0, plates TEXT DEFAULT ''
);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, title TEXT, body TEXT, ok INTEGER, error TEXT
);
"""


class DB:
    """Thread-safe wrapper around a single SQLite connection."""

    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        with self.lock:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.executescript(SCHEMA)
            have = {r[1] for r in self.conn.execute("PRAGMA table_info(events)")}
            for col in ("actions", "details"):  # migrate databases created by earlier versions
                if col not in have:
                    self.conn.execute(f"ALTER TABLE events ADD COLUMN {col} TEXT")
            self.conn.commit()

    def insert(self, table: str, **values) -> int:
        cols = ", ".join(values)
        marks = ", ".join("?" * len(values))
        with self.lock:
            cur = self.conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(values.values()))
            self.conn.commit()
            return cur.lastrowid

    def update(self, table: str, row_id: int, **values) -> None:
        if not values:
            return
        sets = ", ".join(f"{k} = ?" for k in values)
        with self.lock:
            self.conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (*values.values(), row_id))
            self.conn.commit()

    def query(self, sql: str, *args) -> list[dict]:
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]
