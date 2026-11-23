from datetime import datetime, time

from vision.logic import AlertPolicy, DailyNumbers, in_hours, parse_hours


def test_numbers_sequential_and_reset_daily():
    n = DailyNumbers()
    d1 = datetime(2026, 11, 23, 10)
    assert n.number_for(17, d1) == (1, True)
    assert n.number_for(42, d1) == (2, True)
    assert n.number_for(17, d1) == (1, False)
    assert n.number_for(99, datetime(2026, 11, 24, 0, 1)) == (1, True)


def test_hours_overnight():
    h = parse_hours("18:00-07:00")
    assert in_hours(h, time(23, 0)) and in_hours(h, time(6, 59))
    assert not in_hours(h, time(12, 0))
    assert in_hours(parse_hours("always"), time(12)) and not in_hours(parse_hours("never"), time(12))
    assert in_hours(parse_hours("09:00-17:00"), time(12))


def test_alert_cooldown_and_hours():
    p = AlertPolicy("18:00-07:00", cooldown=300)
    assert not p.should_alert(datetime(2026, 11, 23, 12, 0))
    assert p.should_alert(datetime(2026, 11, 23, 22, 0, 0))
    assert not p.should_alert(datetime(2026, 11, 23, 22, 2, 0))
    assert p.should_alert(datetime(2026, 11, 23, 22, 6, 0))


def test_lag_meter():
    from vision.logic import LagMeter
    m = LagMeter()
    assert m.update(0.0, 100.0) == 0
    assert m.update(1.0, 101.0) == 0          # keeping up
    assert m.update(2.0, 103.5) == 1.5        # 1.5 s behind
    assert m.update(5.0, 104.8) == 0          # caught up (new baseline)


def test_person_alerter_waits_groups_and_spaces_out():
    from vision.logic import PersonAlerter
    a = PersonAlerter(delay=2, gap=20)
    a.arrived(10, 1, 0.0)
    a.arrived(11, 2, 0.5)
    assert a.due(1.0, {10, 11}) == []                 # wait to see what they're doing
    assert a.due(2.0, {10, 11}) == [(10, 1), (11, 2)]  # one alert for both
    a.arrived(12, 3, 5.0)
    assert a.due(8.0, {12}) == []                      # too soon after the last alert
    assert a.due(22.0, {12}) == [(12, 3)]              # sent once the gap has passed
    a.arrived(13, 4, 30.0)
    assert a.due(50.0, set()) == [] and not a.is_pending(13)   # left before we could alert: dropped


def test_behaviour_cooldown():
    from vision.logic import PersonAlerter
    a = PersonAlerter(behaviour_cooldown=60)
    assert a.behaviour_ok(1, "running", 0)
    assert not a.behaviour_ok(1, "running", 30)
    assert a.behaviour_ok(1, "crouching", 30)
    assert a.behaviour_ok(2, "running", 30)
    assert a.behaviour_ok(1, "running", 61)
