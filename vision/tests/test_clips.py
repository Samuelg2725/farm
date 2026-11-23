import shutil
import subprocess

import cv2
import numpy as np
import pytest

from vision.clips import duration, phone_clip, plate_photo

ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_plate_photo_has_enlarged_plate_inset():
    car = np.full((400, 680, 3), 90, np.uint8)
    car[270:366, 150:531] = (0, 210, 250)            # yellow plate
    jpg = plate_photo(car, (150, 270, 531, 366), "YX19ABC")
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    assert img.shape[1] == 1280
    assert img[60, 100].tolist() != [90, 90, 90]      # top-left now shows the plate close-up, not the car body


@ffmpeg
def test_phone_clip_keeps_small_and_shrinks_big(tmp_path):
    small = tmp_path / "small.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=10", "-t", "3",
                    "-pix_fmt", "yuv420p", str(small)], check=True)
    assert phone_clip(small, tmp_path / "s_phone.mp4") == small          # already fits → sent as-is
    big = tmp_path / "big.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=1920x1080:rate=25", "-t", "20",
                    "-pix_fmt", "yuv420p", "-c:v", "libx264", "-b:v", "8M", str(big)], check=True)
    out = phone_clip(big, tmp_path / "b_phone.mp4", max_seconds=12, max_mb=2)
    assert out is not None and out.stat().st_size <= 2.2e6
    assert 11 <= duration(out) <= 12.5
    h = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "stream=height",
                        "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout.strip()
    assert h == "720"
