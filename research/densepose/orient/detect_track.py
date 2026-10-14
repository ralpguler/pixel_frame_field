"""Demo-quality person detection: RT-DETRv2-R101 (transformer COCO detector, not YOLO) + a light
IoU+EMA tracker for temporally stable boxes across video frames. Used by the video viz scripts.
(Eval uses detectron2 instead — see d2_detect.py.)
"""
import numpy as np
import torch

_MODEL = None
_PROC = None
_NAME = "PekingU/rtdetr_v2_r101vd"


def _load(dev="cuda"):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor
        _PROC = RTDetrImageProcessor.from_pretrained(_NAME)
        _MODEL = RTDetrV2ForObjectDetection.from_pretrained(_NAME).to(dev).eval()
        _MODEL._person = [i for i, n in _MODEL.config.id2label.items() if n == "person"][0]
    return _MODEL, _PROC


@torch.no_grad()
def detect_persons(rgb, thr=0.4, size=640, dev="cuda"):
    """rgb (H,W,3) uint8 -> list of [x0,y0,x1,y1,score] for the person class.
    LETTERBOX to size×size (aspect-preserving) before RT-DETR's 640 input — plain resize squashes
    16:9 frames and misses people. Boxes mapped back to original coords."""
    import cv2
    m, proc = _load(dev)
    H, W = rgb.shape[:2]
    s = size / max(H, W); nh, nw = int(round(H * s)), int(round(W * s))
    pad = np.zeros((size, size, 3), np.uint8)
    pad[:nh, :nw] = cv2.resize(rgb, (nw, nh))
    proc.size = {"height": size, "width": size}                       # input already size×size -> identity resize
    inp = proc(images=pad, return_tensors="pt").to(dev)
    out = m(**inp)
    res = proc.post_process_object_detection(out, target_sizes=[(size, size)], threshold=thr)[0]
    boxes = []
    for box, sc, lb in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(),
                           res["labels"].cpu().numpy()):
        if int(lb) == m._person:
            x0, y0, x1, y1 = (box / s).tolist()                       # letterbox -> original coords
            boxes.append([max(0, x0), max(0, y0), min(W, x1), min(H, y1), float(sc)])
    return boxes


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Greedy IoU association + EMA box smoothing -> stable boxes across frames (kills crop jitter)."""
    def __init__(self, iou_thr=0.4, ema=0.5, max_age=12, min_hits=2):
        self.iou_thr, self.ema, self.max_age, self.min_hits = iou_thr, ema, max_age, min_hits
        self.tracks = []          # dict(box[4], age, hits)
        self._n = 0

    def update(self, dets):
        dets = sorted(dets, key=lambda d: -d[4])         # high score first
        used = [False] * len(self.tracks)
        for d in dets:
            db = d[:4]; best, bi = self.iou_thr, -1
            for i, t in enumerate(self.tracks):
                if used[i]:
                    continue
                v = _iou(db, t["box"])
                if v > best:
                    best, bi = v, i
            if bi >= 0:
                t = self.tracks[bi]; used[bi] = True
                t["box"] = [self.ema * n + (1 - self.ema) * o for n, o in zip(db, t["box"])]
                t["age"] = 0; t["hits"] += 1
            else:
                self.tracks.append({"box": list(db), "age": 0, "hits": 1}); used.append(True)
        # age & drop unmatched
        for i, t in enumerate(self.tracks):
            if i < len(used) and not used[i]:
                t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return [t["box"] for t in self.tracks if t["hits"] >= self.min_hits or t["age"] == 0]
