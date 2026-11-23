"""Runs the real model if available: TEST_MODEL=path/to/yolox_s.onnx TEST_IMAGE=photo_with_people.jpg"""
import os

import numpy as np
import pytest

from people_tracker.detector import decode, make_grids


def test_decode_single_box():
    size = (640, 640)
    grids, strides = make_grids(size)
    raw = np.full((grids.shape[0], 85), -10.0, dtype=np.float32)
    raw[:, 4] = 0
    i = 100  # a cell in the stride-8 grid
    raw[i, :2] = 0.5
    raw[i, 2:4] = np.log(4.0)  # 4 * stride 8 = 32 px
    raw[i, 4], raw[i, 5] = 1.0, 0.9
    [(x1, y1, x2, y2, s)] = decode(raw, grids, strides, ratio=1.0, conf=0.5, nms=0.45)
    cx, cy = (grids[i] + 0.5) * 8
    assert (x2 - x1) == pytest.approx(32) and (x1 + x2) / 2 == pytest.approx(cx) and s == pytest.approx(0.9)


@pytest.mark.skipif(not (os.getenv("TEST_MODEL") and os.getenv("TEST_IMAGE")), reason="model/image not set")
def test_real_model_finds_people():
    import cv2
    from people_tracker.detector import PersonDetector
    img = cv2.imread(os.environ["TEST_IMAGE"])
    dets = PersonDetector(os.environ["TEST_MODEL"]).detect(img)
    print(dets)
    assert len(dets) >= 3
    for x1, y1, x2, y2, s in dets:
        assert 0 <= x1 < x2 <= img.shape[1] + 5 and 0 <= y1 < y2 <= img.shape[0] + 5
