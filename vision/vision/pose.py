"""Body pose (17 points: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles)
with RTMPose-m (Apache-2.0, OpenMMLab) via ONNX Runtime. Runs on each person box we already track."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

INPUT_W, INPUT_H = 192, 256
MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)

# COCO keypoint ids
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


def box_to_center_scale(box, padding: float = 1.25):
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    w, h = (x2 - x1) * padding, (y2 - y1) * padding
    aspect = INPUT_W / INPUT_H
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float32)


def affine(center, scale, inverse=False):
    src = np.array([center, center + [0, -scale[1] / 2], center + [-scale[0] / 2, 0]], dtype=np.float32)
    dst = np.array([[INPUT_W / 2, INPUT_H / 2], [INPUT_W / 2, 0], [0, INPUT_H / 2]], dtype=np.float32)
    return cv2.getAffineTransform(dst, src) if inverse else cv2.getAffineTransform(src, dst)


class PoseEstimator:
    def __init__(self, model_path: str, threads: int = 2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])

    def estimate(self, frame_bgr: np.ndarray, boxes: list) -> list[tuple[np.ndarray, np.ndarray]]:
        """For each box → (keypoints (17, 2) in frame pixels, scores (17,)). Scores above ~0.3 are usable."""
        if not boxes:
            return []
        crops, metas = [], []
        for box in boxes:
            c, s = box_to_center_scale(box)
            crop = cv2.warpAffine(frame_bgr, affine(c, s), (INPUT_W, INPUT_H), flags=cv2.INTER_LINEAR)
            crop = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) - MEAN) / STD
            crops.append(crop.transpose(2, 0, 1))
            metas.append((c, s))
        sx, sy = self.sess.run(None, {"input": np.stack(crops).astype(np.float32)})
        out = []
        for i, (c, s) in enumerate(metas):
            x, y = sx[i].argmax(1), sy[i].argmax(1)
            score = np.minimum(sx[i].max(1), sy[i].max(1))
            pts = np.stack([x / 2.0, y / 2.0], 1).astype(np.float32)  # simcc split ratio 2
            m = affine(c, s, inverse=True)
            pts = pts @ m[:, :2].T + m[:, 2]
            out.append((pts, score))
        return out
