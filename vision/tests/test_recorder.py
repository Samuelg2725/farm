import shutil
import subprocess
import time
from datetime import datetime, timedelta

import pytest

from vision.recorder import Recorder, clip_name

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def video(tmp_path):
    f = tmp_path / "cam.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "30",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "10", str(f)], check=True)
    return f


def test_clip_name():
    assert clip_name(datetime(2026, 1, 1, 9, 5, 7), {"vehicle", "person"}) == "09-05-07_person+vehicle"


def test_records_while_in_view_then_stops(video, tmp_path):
    out = tmp_path / "rec"
    r = Recorder(str(video), str(out), post_roll=2)
    t = datetime(2026, 11, 23, 12, 0, 0)
    assert r.update(t, {"person"}, b"\xff\xd8jpeg")[0] == "started"
    assert r.recording
    time.sleep(3)  # ffmpeg writes ~3 s of video
    assert r.update(t + timedelta(seconds=1), {"person", "vehicle"}) is None   # still in view
    assert r.update(t + timedelta(seconds=2), set()) is None                    # within post-roll
    what, path = r.update(t + timedelta(seconds=4), set())                      # post-roll over
    assert what == "stopped" and not r.recording
    assert path.name == "12-00-00_person+vehicle.mp4" and path.parent.name == "2026-11-23"
    assert path.with_suffix(".jpg").exists()
    dur = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    assert float(dur) >= 1.5
    clips = r.list_clips()
    assert clips[0]["file"] == "2026-11-23/12-00-00_person+vehicle.mp4" and clips[0]["thumb"]


def test_cleanup_old_days(tmp_path):
    out = tmp_path / "rec"
    (out / "2026-01-01").mkdir(parents=True)
    (out / "2026-11-20").mkdir(parents=True)
    Recorder("x", str(out), keep_days=30).cleanup(datetime(2026, 11, 23))
    assert not (out / "2026-01-01").exists() and (out / "2026-11-20").exists()
