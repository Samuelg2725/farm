from vision.health import HealthMonitor


def titles(alerts):
    return [a.title for a in alerts]


def test_startup_and_camera_offline_then_recovery():
    h = HealthMonitor(stale_after=15, offline_alert_after=60)
    assert titles(h.check(0, last_frame=0, skips=0, free_gb=100)) == ["System started"]
    assert h.status["ok"]
    assert h.check(30, last_frame=10, skips=0, free_gb=100) == []          # offline 20 s: too early
    assert not h.status["camera_online"]
    assert titles(h.check(75, last_frame=10, skips=0, free_gb=100)) == ["Camera offline"]
    assert h.check(200, last_frame=10, skips=0, free_gb=100) == []         # only once
    a = h.check(400, last_frame=399, skips=0, free_gb=100)
    assert titles(a) == ["Camera back online"] and "6 min" in a[0].body
    assert h.status["camera_online"]


def test_short_blip_is_silent():
    h = HealthMonitor(stale_after=15, offline_alert_after=60)
    h.check(0, 0, 0, 100)
    h.check(30, 10, 0, 100)                                                # offline briefly
    assert h.check(40, 39, 0, 100) == []                                   # back before alerting: no messages


def test_never_connected_counts_as_offline():
    h = HealthMonitor(offline_alert_after=60)
    h.check(0, None, 0, 100)
    assert titles(h.check(61, None, 0, 100)) == ["Camera offline"]


def test_lag_and_disk():
    h = HealthMonitor(lag_skips=3, lag_window=600, min_free_gb=10, disk_cooldown=3600)
    h.check(0, 0, 0, 100)
    assert h.check(10, 10, 2, 100) == []
    assert titles(h.check(20, 20, 3, 100)) == ["Processing falling behind"]
    assert h.check(30, 30, 4, 100) == []                                   # cooldown
    assert titles(h.check(40, 40, 4, 5.0)) == ["Disk space low"]
    assert h.status["low_disk"] and not h.status["ok"]
    assert h.check(50, 50, 4, 5.0) == []
