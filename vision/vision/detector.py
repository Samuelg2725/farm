"""People + vehicle detection with YOLOX (Apache-2.0) via ONNX Runtime – one pass finds both."""
from __future__ import annotations

import cv2
import numpy as np
import onnxruntime as ort

# COCO class ids we care about
PERSON = 0
VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASSES = (PERSON, *VEHICLES)


def letterbox(img: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, float]:
    """Resize keeping aspect ratio, pad with grey (114) – YOLOX's expected input."""
    h, w = img.shape[:2]
    r = min(size[0] / h, size[1] / w)
    resized = cv2.resize(img, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
    padded = np.full((size[0], size[1], 3), 114, dtype=np.uint8)
    padded[: resized.shape[0], : resized.shape[1]] = resized
    return padded, r


def make_grids(size: tuple[int, int], strides=(8, 16, 32)) -> tuple[np.ndarray, np.ndarray]:
    grids, exp = [], []
    for s in strides:
        hs, ws = size[0] // s, size[1] // s
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        g = np.stack((xv, yv), 2).reshape(-1, 2)
        grids.append(g)
        exp.append(np.full((g.shape[0], 1), s))
    return np.concatenate(grids).astype(np.float32), np.concatenate(exp).astype(np.float32)


def decode(raw: np.ndarray, grids: np.ndarray, strides: np.ndarray, ratio: float,
           conf: float, nms: float, classes=CLASSES) -> list[tuple[float, float, float, float, float, int]]:
    """raw: (N, 85) YOLOX output -> [(x1, y1, x2, y2, score, class_id)] in original image pixels."""
    xy = (raw[:, :2] + grids) * strides
    wh = np.exp(raw[:, 2:4]) * strides
    cls_scores = raw[:, 4:5] * raw[:, [5 + c for c in classes]]
    best = cls_scores.argmax(1)
    scores = cls_scores[np.arange(len(best)), best]
    keep = scores >= conf
    if not keep.any():
        return []
    xy, wh, scores = xy[keep], wh[keep], scores[keep]
    cls_ids = np.array(classes)[best[keep]]
    x1y1 = (xy - wh / 2) / ratio
    wh = wh / ratio
    # NMS per group (people vs vehicles) so a person next to a car isn't suppressed
    group = np.where(cls_ids == PERSON, 0, 1)
    idx = cv2.dnn.NMSBoxesBatched(np.hstack([x1y1, wh]).tolist(), scores.tolist(), group.tolist(), conf, nms)
    out = []
    for i in np.array(idx).flatten():
        x1, y1 = x1y1[i]
        w, h = wh[i]
        out.append((float(x1), float(y1), float(x1 + w), float(y1 + h), float(scores[i]), int(cls_ids[i])))
    return out


class Detector:
    def __init__(self, model_path: str, input_size=(640, 640), conf=0.45, nms=0.45, threads=2):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.size, self.conf, self.nms = input_size, conf, nms
        self.grids, self.strides = make_grids(input_size)

    def detect(self, frame_bgr: np.ndarray):
        img, r = letterbox(frame_bgr, self.size)
        blob = np.ascontiguousarray(img.transpose(2, 0, 1)[None], dtype=np.float32)
        raw = self.sess.run(None, {self.input_name: blob})[0][0]
        return decode(raw, self.grids, self.strides, r, self.conf, self.nms)
