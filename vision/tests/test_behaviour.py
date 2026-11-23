import numpy as np

from vision.behaviour import BehaviourAnalyzer, Zone, point_in_polygon, posture


def skeleton(kind="standing"):
    """Synthetic 17-point pose inside box (0,0)-(100,300)."""
    p = np.zeros((17, 2), np.float32)
    head, sh, hip, knee, ankle = 20, 60, 160, 230, 295
    p[0] = (50, head); p[1:5] = (50, head)
    p[5], p[6] = (35, sh), (65, sh)
    p[7], p[8] = (30, 110), (70, 110)
    p[9], p[10] = (30, 150), (70, 150)
    p[11], p[12] = (40, hip), (60, hip)
    p[13], p[14] = (40, knee), (60, knee)
    p[15], p[16] = (40, ankle), (60, ankle)
    if kind == "crouching":
        p[11:13, 1] = 250; p[13:15, 1] = 270; p[15:17, 1] = 295
    if kind == "arms":
        p[9, 1] = p[10, 1] = 5; p[7, 1] = p[8, 1] = 30
    if kind == "climbing":
        p[13, 1], p[15, 1] = 150, 180      # left leg raised high
        p[9, 1] = 40                       # left hand above shoulder
    if kind == "lying":
        p = np.stack([p[:, 1], p[:, 0] + 100], 1)  # rotate 90°: body horizontal
    return p, np.full(17, 0.9, np.float32)


def test_postures():
    assert posture(*skeleton("standing"), (0, 0, 100, 300)) == set()
    assert "crouching" in posture(*skeleton("crouching"), (0, 0, 100, 300))
    assert "arms raised" in posture(*skeleton("arms"), (0, 0, 100, 300))
    assert "climbing" in posture(*skeleton("climbing"), (0, 0, 100, 300))
    assert "lying down" in posture(*skeleton("lying"), (0, 100, 300, 200))
    assert "lying down" in posture(None, None, (0, 0, 300, 100))   # no pose: wide box


def test_lying_must_hold_4s_and_is_flagged_once():
    a = BehaviourAnalyzer()
    pts, sc = skeleton("lying")
    news = []
    for i in range(30):  # 6 s at 5 fps
        state, new = a.update(1, i * 0.2, (0, 100, 300, 200), pts, sc)
        news += [(i * 0.2, n) for n in new]
    assert [n for _, n in news] == ["lying down"] and news[0][0] >= 4.0
    assert "lying down" in state["alerts"] and "lying down" in state["doing"]


def test_walking_running_still_and_loitering():
    a = BehaviourAnalyzer(loiter_seconds=10)
    pts, sc = skeleton()
    for i in range(10):                       # stands still 2 s
        s, _ = a.update(1, i * 0.2, (0, 0, 100, 300), pts, sc)
    assert s["movement"] == "still" and s["doing"] == "standing still"
    for i in range(10):                       # walks: 0.5 body heights/s (threshold 0.25)
        x = i * 0.2 * 150
        s, _ = a.update(2, i * 0.2, (x, 0, x + 100, 300), pts, sc)
    assert s["movement"] == "walking"
    new_all = []
    for i in range(20):                       # runs: 2 body heights/s
        x = i * 0.2 * 600
        s, new = a.update(3, i * 0.2, (x, 0, x + 100, 300), pts, sc)
        new_all += new
    assert s["movement"] == "running" and "running" in new_all
    new_all = []
    for i in range(60):                       # 12 s in view → loitering after 10 s
        s, new = a.update(4, i * 0.2, (0, 0, 100, 300), pts, sc)
        new_all += new
    assert "loitering" in new_all


def test_zone_entry():
    assert point_in_polygon(0.5, 0.5, [[0, 0], [1, 0], [1, 1], [0, 1]])
    assert not point_in_polygon(1.5, 0.5, [[0, 0], [1, 0], [1, 1], [0, 1]])
    z = Zone("Lairage", [[0.5, 0.5], [1, 0.5], [1, 1], [0.5, 1]])
    a = BehaviourAnalyzer()
    _, new = a.update(1, 0, (100, 100, 200, 400), frame_size=(1000, 1000), zones=[z])   # feet at (150, 400): outside
    assert new == []
    _, new = a.update(1, 1, (600, 500, 700, 800), frame_size=(1000, 1000), zones=[z])   # feet at (650, 800): inside
    assert new == ["entered Lairage"]
    s, new = a.update(1, 2, (600, 500, 700, 800), frame_size=(1000, 1000), zones=[z])
    assert new == [] and s["zones"] == ["Lairage"]


def test_cut_off_person_is_not_called_crouching_or_lying():
    pts, sc = skeleton("crouching")
    assert posture(pts, sc, (0, 0, 100, 300), cut_off=True) == set()
    a = BehaviourAnalyzer()
    for i in range(30):  # box touching bottom of a 1000x300 frame, crouch-like pose, 6 s
        s, new = a.update(1, i * 0.2, (400, 0, 500, 300), pts, sc, frame_size=(1000, 300))
        assert "crouching" not in new
    assert s["doing"] == "standing still"


def test_doing_text():
    a = BehaviourAnalyzer()
    pts, sc = skeleton()
    for i in range(10):
        x = i * 0.2 * 150
        s, _ = a.update(1, i * 0.2, (x, 0, x + 100, 300), pts, sc)
    assert s["doing"] == "walking"


def test_camera_pan_is_not_walking():
    import cv2
    from vision.behaviour import CameraMotion
    rng = np.random.default_rng(0)
    world = cv2.GaussianBlur((rng.random((900, 2400)) * 255).astype(np.uint8), (0, 0), 3)
    world = cv2.cvtColor(world, cv2.COLOR_GRAY2BGR)
    cm, a = CameraMotion(), BehaviourAnalyzer()
    pts, sc = skeleton()
    for i in range(15):                     # camera pans right 60 px per frame; person stands still in the world
        pan = i * 60
        frame = world[100:820, pan:pan + 1280]
        offset, jumped = cm.update(frame)
        assert not jumped
        x = 600 - pan                        # so they slide left across the picture
        s, _ = a.update(1, i * 0.2, (x, 300, x + 100, 600), pts, sc, (1280, 720), camera_offset=offset)
    assert abs(offset[0] + 14 * 60) < 30      # picture content moved left by the pan
    assert s["movement"] == "still"
    other = cv2.GaussianBlur((np.random.default_rng(7).random((720, 1280)) * 255).astype(np.uint8), (0, 0), 3)
    _, jumped = cm.update(cv2.cvtColor(other, cv2.COLOR_GRAY2BGR))  # picture changes completely (preset move)
    assert jumped


def test_zone_store_validates_and_saves(tmp_path):
    import pytest
    from vision.zones import ZoneStore
    zs = ZoneStore(str(tmp_path / "zones.yaml"))
    sq = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5]]
    out = zs.save([{"name": " Gate ", "type": "watch", "hours": "18:00-07:00", "points": sq},
                   {"name": "", "points": sq}, {"name": "Tiny", "points": sq[:2]}])
    assert [z["name"] for z in out] == ["Gate"] and zs.get()[0].hours == "18:00-07:00"
    with pytest.raises(ValueError):
        zs.save([{"name": "Bad", "hours": "teatime", "points": sq}])
