from vision.alerts import Notifier, resolve_topic
from vision.db import DB


def test_public_server_gets_private_random_topic(tmp_path):
    t1 = resolve_topic("https://ntfy.sh", "farm-alerts", str(tmp_path))
    assert t1.startswith("farm-") and t1 != "farm-alerts" and len(t1) > 12
    assert resolve_topic("https://ntfy.sh", "", str(tmp_path)) == t1          # remembered
    assert resolve_topic("https://ntfy.sh", "my-own-topic-x9", str(tmp_path)) == "my-own-topic-x9"
    assert resolve_topic("http://ntfy", "farm-alerts", str(tmp_path)) == "farm-alerts"  # local server: fine


def test_notifier_logs_failures(tmp_path):
    db = DB(str(tmp_path / "v.db"))
    n = Notifier("http://127.0.0.1:9", "t", db, str(tmp_path))   # nothing listening
    assert n.send("Hello", "body") is False
    [row] = db.query("SELECT * FROM alerts")
    assert row["title"] == "Hello" and row["ok"] == 0 and row["error"]
    assert "<svg" in n.qr_svg()


def test_db_insert_update(tmp_path):
    db = DB(str(tmp_path / "v.db"))
    i = db.insert("events", ts="2026-11-23T12:00:00", kind="vehicle", number=1, confidence=0.8)
    db.update("events", i, plate="KL70XYZ", status="unknown")
    assert db.query("SELECT plate, status FROM events")[0] == {"plate": "KL70XYZ", "status": "unknown"}


def test_header_value_handles_newlines_and_unicode():
    import base64
    from vision.alerts import header_value
    assert header_value("plain text") == "plain text"
    assert header_value("line1\nline2") == "line1\\nline2"
    enc = header_value("Sep 2020 – Feb 2021\nblue")
    assert enc.startswith("=?UTF-8?B?") and "\n" not in enc
    assert base64.b64decode(enc[10:-2]).decode() == "Sep 2020 – Feb 2021\\nblue"


def test_switches_remembered(tmp_path):
    from vision.alerts import Switches
    s = Switches(str(tmp_path), {"person_alerts": True})
    assert s.get("person_alerts")
    s.set("person_alerts", False)
    assert not Switches(str(tmp_path), {"person_alerts": True}).get("person_alerts")
