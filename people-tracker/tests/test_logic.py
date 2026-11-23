from datetime import datetime, time

from people_tracker.logic import AlertPolicy, DailyNumbers, in_hours, parse_hours


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
