import shutil
import subprocess
import time
from datetime import datetime, timedelta

import cv2
import numpy as np
import pytest

from vision.recorder import Recorder

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


@pytest.fixture
def video(tmp_path):
    f = tmp_path / "cam.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "30",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "10", str(f)], check=True)
    return f


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out or 0)


def test_records_annotated_and_raw(video, tmp_path):
    out = tmp_path / "rec"
    r = Recorder(str(video), str(out), post_roll=2)
    t = datetime(2026, 11, 23, 12, 0, 0)
    what, info = r.update(t, True, b"\xff\xd8thumb")
    assert what == "started" and info == {"file": "2026-11-23/12-00-00.mp4",
                                          "raw_file": "2026-11-23/12-00-00_4k.mp4", "thumb": "2026-11-23/12-00-00.jpg"}
    frame = np.zeros((360, 640, 3), np.uint8)
    for i in range(30):  # 3 s of "live view" frames with a moving box
        cv2.rectangle(frame, (i * 10, 50), (i * 10 + 80, 200), (0, 200, 255), 3)
        r.feed(cv2.imencode(".jpg", frame)[1].tobytes())
        time.sleep(0.1)
    assert r.update(t + timedelta(seconds=1), False) is None        # within post-roll
    what, info = r.update(t + timedelta(seconds=4), False)
    assert what == "stopped" and not r.recording
    assert duration(out / "2026-11-23/12-00-00.mp4") >= 2.0       # boxes version
    assert duration(out / "2026-11-23/12-00-00_4k.mp4") >= 1.5    # original
    assert (out / "2026-11-23/12-00-00.jpg").exists()


def test_cleanup_old_days(tmp_path):
    out = tmp_path / "rec"
    (out / "2026-01-01").mkdir(parents=True)
    (out / "2026-11-20").mkdir(parents=True)
    Recorder("x", str(out), keep_days=30).cleanup(datetime(2026, 11, 23))
    assert not (out / "2026-01-01").exists() and (out / "2026-11-20").exists()
