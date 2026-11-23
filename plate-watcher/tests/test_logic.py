from plate_watcher.__main__ import parse_message
from plate_watcher.matching import Vehicle, find_match, normalise
from plate_watcher.tracker import Tracker

WL = [Vehicle("AB12 CDE", "Owner"), Vehicle("YX19ABC", "Haulier")]


def test_normalise():
    assert normalise(" ab12 cde ") == "AB12CDE"


def test_exact_and_spacing():
    assert find_match("ab12cde", WL).owner == "Owner"


def test_lookalike_ocr_errors():
    assert find_match("A812CDE", WL).owner == "Owner"   # B read as 8
    assert find_match("YX19A8C", WL).owner == "Haulier"


def test_one_wrong_char_ok_two_not():
    assert find_match("AB12CDF", WL) is not None
    assert find_match("AB12CXY", WL) is None


def test_unknown():
    assert find_match("KL70XYZ", WL) is None
    assert find_match("", WL) is None


def test_tracker_keeps_best_and_decides_on_end():
    t = Tracker(delay=20, min_score=0.5)
    t.add_read("e1", "cam", "AB12C", 0.6, 0)
    t.add_read("e1", "cam", "AB12CDE", 0.9, 1)
    t.add_read("e1", "cam", "AB1", 0.3, 2)  # below min score
    assert t.due(5) == []
    t.mark_ended("e1")
    [p] = t.due(6)
    assert p.plate == "AB12CDE"
    t.add_read("e1", "cam", "ZZZ", 0.99, 7)  # already decided: ignored
    assert t.due(100) == []


def test_tracker_timeout():
    t = Tracker(delay=20)
    t.add_read("e2", "cam", "KL70XYZ", 0.9, 0)
    assert t.due(10) == []
    assert len(t.due(21)) == 1


def test_parse_messages():
    r = list(parse_message("frigate/tracked_object_update",
                           {"type": "lpr", "id": "x", "camera": "c", "plate": "AB12CDE", "score": 0.9}))
    assert r == [("read", "x", "c", "AB12CDE", 0.9)]
    r = list(parse_message("frigate/events", {"type": "end", "after": {
        "id": "y", "camera": "c", "recognized_license_plate": ["KL70XYZ", 0.8]}}))
    assert r == [("read", "y", "c", "KL70XYZ", 0.8), ("end", "y")]


def test_parse_vehicle_event():
    from plate_watcher.__main__ import parse_message
    msg = {"type": "update", "after": {"id": "v1", "camera": "c", "label": "car",
                                       "box": [10, 20, 110, 120], "stationary": False}}
    assert list(parse_message("frigate/events", msg)) == [("vehicle", "v1", "c", [10, 20, 110, 120])]
    msg["after"]["stationary"] = True
    assert list(parse_message("frigate/events", msg)) == []
    msg["after"].update(stationary=False, label="person")
    assert list(parse_message("frigate/events", msg)) == []


def test_grab_scheduler():
    from plate_watcher.__main__ import GrabScheduler
    g = GrabScheduler(interval=2, max_grabs=2)
    assert g.should_grab("e", 0)
    assert not g.should_grab("e", 1)
    assert g.should_grab("e", 3)
    assert not g.should_grab("e", 10)  # hit max


def test_crop_box_scales_to_full_res():
    import numpy as np
    from plate_watcher.reader import crop_box
    frame = np.zeros((2160, 3840, 3), np.uint8)
    c = crop_box(frame, [640, 360, 960, 540], (1280, 720), pad=0)
    assert c.shape[:2] == (540, 960)
