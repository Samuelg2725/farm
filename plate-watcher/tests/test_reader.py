"""End-to-end check of the free plate reader. Downloads models on first run."""
from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("fast_alpr")


def test_reads_uk_plate_from_car_crop():
    from plate_watcher.reader import PlateReader, crop_box
    img = cv2.imread(str(Path(__file__).parent / "fixtures" / "plate_test.jpg"))
    car = crop_box(img, [300, 200, 980, 600], (1280, 720), pad=0)
    reads = PlateReader().read(car)
    assert reads and reads[0][0] == "YX19ABC" and reads[0][1] > 0.7
