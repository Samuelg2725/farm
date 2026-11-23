"""Rule-based behaviour classification from pose keypoints, motion and zones.

Posture (keypoints): standing, crouching, lying down, arms raised, climbing
Motion (tracking):   still, walking, running, loitering
Location (zones):    entered <zone>

Each state must persist for a minimum duration (HOLD) before it is reported.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .logic import in_hours, parse_hours
from .pose import (L_ANKLE, L_HIP, L_KNEE, L_SHOULDER, L_WRIST, NOSE, R_ANKLE, R_HIP, R_KNEE, R_SHOULDER,
                   R_WRIST)

KP_MIN = 0.4  # minimum keypoint confidence; occluded joints typically score ~0.1

# minimum duration (s) before a state is reported
HOLD = {"lying down": 4.0, "crouching": 3.0, "arms raised": 2.0, "climbing": 2.0, "running": 1.5}

LABELS = {  # display labels
    "lying down": "lying down (possible fall)", "climbing": "climbing", "running": "running",
    "crouching": "crouching", "arms raised": "arms raised", "loitering": "loitering",
}


def _mid(pts, sc, a, b):
    ok = [i for i in (a, b) if sc[i] >= KP_MIN]
    return pts[ok].mean(0) if ok else None


def posture(pts: np.ndarray | None, sc: np.ndarray | None, box, cut_off: bool = False) -> set[str]:
    """Posture flags for a single frame; falls back to box aspect ratio without usable keypoints.

    When `cut_off` is True (the box touches the frame edge), crouching and lying down are not
    reported, since a truncated body would otherwise be misclassified."""
    if cut_off:
        pts_ok = pts is not None and sc is not None
        flags = posture(pts, sc, box) if pts_ok else set()
        return flags - {"crouching", "lying down"}
    x1, y1, x2, y2 = box
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    flags: set[str] = set()
    if pts is None or sc is None:
        if w / h > 1.3:
            flags.add("lying down")
        return flags
    sh, hip = _mid(pts, sc, L_SHOULDER, R_SHOULDER), _mid(pts, sc, L_HIP, R_HIP)
    if sh is None or hip is None:
        if w / h > 1.3:
            flags.add("lying down")
        return flags
    torso = hip - sh
    torso_len = float(np.hypot(*torso)) or 1.0
    tilt = math.degrees(math.atan2(abs(torso[0]), abs(torso[1]) + 1e-6))  # torso angle from vertical (degrees)
    if tilt > 60 or (tilt > 45 and w / h > 1.1):
        flags.add("lying down")
        return flags
    # crouching: hip height above ankles/knees small relative to torso length
    ankle = _mid(pts, sc, L_ANKLE, R_ANKLE)
    knee = _mid(pts, sc, L_KNEE, R_KNEE)
    low = ankle if ankle is not None else knee
    if low is not None and tilt < 45:
        hip_height = float(low[1] - hip[1])
        limit = 0.75 if ankle is not None else 0.35
        if hip_height < limit * torso_len:
            flags.add("crouching")
    # arms raised: both wrists above the head (or estimated head position)
    top = pts[NOSE][1] if sc[NOSE] >= KP_MIN else sh[1] - 0.3 * torso_len
    wrists_up = [sc[i] >= KP_MIN and pts[i][1] < top for i in (L_WRIST, R_WRIST)]
    if all(wrists_up):
        flags.add("arms raised")
    # climbing: large vertical leg separation and a hand at or above shoulder height
    feet = [(L_ANKLE, R_ANKLE), (L_KNEE, R_KNEE)]
    for a, b in feet:
        if sc[a] >= KP_MIN and sc[b] >= KP_MIN and abs(pts[a][1] - pts[b][1]) > 0.6 * torso_len:
            if any(sc[i] >= KP_MIN and pts[i][1] < sh[1] for i in (L_WRIST, R_WRIST)):
                flags.add("climbing")
            break
    return flags


def point_in_polygon(x: float, y: float, poly: list) -> bool:
    inside, j = False, len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class Zone:
    name: str
    points: list            # polygon vertices, normalised to 0-1
    kind: str = "restricted"  # 'restricted' (notify) or 'watch' (log only)
    hours: str = "always"     # 'always' or a time range such as 18:00-07:00

    def active(self, t) -> bool:
        return in_hours(parse_hours(self.hours), t)


@dataclass
class PersonState:
    first_seen: float
    history: deque = field(default_factory=lambda: deque(maxlen=60))   # (t, foot_x, foot_y, box height)
    since: dict = field(default_factory=dict)       # flag -> start time
    active: set = field(default_factory=set)        # flags that have satisfied HOLD
    zones: set = field(default_factory=set)
    movement: str = "still"


class CameraMotion:
    """Estimates global image translation between frames (camera pan/tilt) using phase correlation."""

    def __init__(self, width: int = 480):
        self.width, self.prev, self.offset, self.window = width, None, [0.0, 0.0], None

    def update(self, frame_bgr) -> tuple[tuple[float, float], bool]:
        """Return (cumulative offset in full-frame pixels, jumped); `jumped` indicates a scene change."""
        import cv2
        h, w = frame_bgr.shape[:2]
        small = cv2.cvtColor(cv2.resize(frame_bgr, (self.width, int(h * self.width / w)), interpolation=cv2.INTER_AREA),
                             cv2.COLOR_BGR2GRAY).astype(np.float32)
        jumped = False
        if self.prev is not None and self.prev.shape == small.shape:
            if self.window is None:
                self.window = cv2.createHanningWindow(small.shape[::-1], cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(self.prev, small, self.window)
            if not response > 0.1:  # also handles NaN
                jumped = True
            elif abs(dx) > 0.3 or abs(dy) > 0.3:  # ignore sub-pixel jitter
                k = w / self.width
                self.offset[0] += dx * k
                self.offset[1] += dy * k
        self.prev = small
        return (self.offset[0], self.offset[1]), jumped


class BehaviourAnalyzer:
    def __init__(self, loiter_seconds: float = 120, run_speed: float = 1.3, walk_speed: float = 0.25):
        self.loiter_seconds, self.run_speed, self.walk_speed = loiter_seconds, run_speed, walk_speed
        self.people: dict[int, PersonState] = {}

    def forget(self, tid: int) -> None:
        self.people.pop(tid, None)

    def reset_motion(self) -> None:
        """Clear motion history after a scene change (preset move or stream interruption)."""
        for st in self.people.values():
            st.history.clear()

    def update(self, tid: int, t: float, box, pts=None, sc=None, frame_size=(1, 1),
               zones: list[Zone] | None = None, clock=None, camera_offset=(0.0, 0.0)) -> tuple[dict, list[str]]:
        """Update a track and return (display state, newly started behaviours).

        `camera_offset` is the cumulative image translation from CameraMotion; it is subtracted
        from positions so camera movement is not interpreted as subject movement."""
        st = self.people.setdefault(tid, PersonState(first_seen=t))
        x1, y1, x2, y2 = box
        foot = ((x1 + x2) / 2, y2)
        h = max(1.0, y2 - y1)
        world = (foot[0] - camera_offset[0], foot[1] - camera_offset[1])
        st.history.append((t, world[0], world[1], h))

        # speed in body heights per second over the last ~1.5 s
        old = [p for p in st.history if t - p[0] <= 1.5]
        speed = 0.0
        if len(old) >= 3 and t - old[0][0] > 0.4:
            dt = t - old[0][0]
            dist = math.hypot(world[0] - old[0][1], world[1] - old[0][2])
            speed = min(10.0, dist / dt / max(20.0, float(np.median([p[3] for p in old]))))
        st.movement = "running" if speed >= self.run_speed else "walking" if speed >= self.walk_speed else "still"

        fw, fh = frame_size
        cut = x1 <= fw * 0.01 or y1 <= fh * 0.01 or x2 >= fw * 0.99 or y2 >= fh * 0.99
        flags = posture(pts, sc, box, cut_off=cut and frame_size != (1, 1))
        if st.movement == "running":
            flags.add("running")
        for f in list(st.since):
            if f not in flags:
                del st.since[f]
                st.active.discard(f)
        new = []
        for f in flags:
            st.since.setdefault(f, t)
            if f not in st.active and t - st.since[f] >= HOLD.get(f, 2.0):
                st.active.add(f)
                new.append(f)
        if "loitering" not in st.active and t - st.first_seen >= self.loiter_seconds:
            st.active.add("loitering")
            new.append("loitering")

        # zone membership by foot position
        now_in = set()
        for z in zones or []:
            if clock is not None and not z.active(clock):
                continue
            fx, fy = min(foot[0], frame_size[0] - 1), min(foot[1], frame_size[1] - 1)  # clamp to frame for truncated boxes
            if point_in_polygon(fx / frame_size[0], fy / frame_size[1], z.points):
                now_in.add(z.name)
        for name in now_in - st.zones:
            new.append(f"entered {name}")
        st.zones = now_in

        posture_word = ("lying down" if "lying down" in flags else "crouching" if "crouching" in flags
                        else "standing" if pts is not None else "")
        if posture_word in ("", "standing"):
            doing = ["standing still" if st.movement == "still" and posture_word else st.movement]
        else:  # posture takes precedence over motion
            doing = [posture_word] + ([st.movement] if st.movement != "still" else [])
        doing += [b for b in ("climbing", "arms raised", "loitering") if b in st.active]
        state = {"doing": ", ".join(dict.fromkeys(doing)), "movement": st.movement, "speed": round(speed, 2),
                 "alerts": sorted(b for b in st.active if b in ("lying down", "climbing")),
                 "zones": sorted(st.zones), "seconds": round(t - st.first_seen)}
        return state, new
