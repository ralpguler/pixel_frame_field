"""Person boxes for the demo: RT-DETRv2-R101 (Hugging Face transformers, PekingU/rtdetr_v2_r101vd) and a light
IoU + EMA tracker that keeps boxes stable across video frames. Evaluation uses ground-truth or given boxes instead.

Boxes here are [x0, y0, x1, y1, score] in image pixels.
"""
import numpy as np
import torch

NAME = "PekingU/rtdetr_v2_r101vd"
_MODEL = None
_PROC = None


def _load(device):
    global _MODEL, _PROC
    if _MODEL is None:
        from transformers import RTDetrImageProcessor, RTDetrV2ForObjectDetection
        _PROC = RTDetrImageProcessor.from_pretrained(NAME)
        _MODEL = RTDetrV2ForObjectDetection.from_pretrained(NAME).to(device).eval()
        _MODEL._person = [i for i, n in _MODEL.config.id2label.items() if n == "person"][0]
    return _MODEL, _PROC


@torch.no_grad()
def detect_persons(rgb, thr=0.4, size=640, device="cuda"):
    """rgb (H,W,3) uint8 -> list of [x0,y0,x1,y1,score] for people. The frame is letterboxed to size x size
    (aspect preserved; a plain resize squashes wide frames and misses people) and boxes are mapped back."""
    import cv2
    m, proc = _load(device)
    H, W = rgb.shape[:2]
    s = size / max(H, W); nh, nw = int(round(H * s)), int(round(W * s))
    pad = np.zeros((size, size, 3), np.uint8)
    pad[:nh, :nw] = cv2.resize(rgb, (nw, nh))
    proc.size = {"height": size, "width": size}
    out = m(**proc(images=pad, return_tensors="pt").to(device))
    res = proc.post_process_object_detection(out, target_sizes=[(size, size)], threshold=thr)[0]
    boxes = []
    for box, sc, lb in zip(res["boxes"].cpu().numpy(), res["scores"].cpu().numpy(), res["labels"].cpu().numpy()):
        if int(lb) == m._person:
            x0, y0, x1, y1 = (box / s).tolist()
            boxes.append([max(0, x0), max(0, y0), min(W, x1), min(H, y1), float(sc)])
    return boxes


def _iou(a, b):
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class Tracker:
    """Greedy IoU association and EMA box smoothing: stable boxes across frames, so crops do not jitter."""

    def __init__(self, iou_thr=0.4, ema=0.5, max_age=12, min_hits=2):
        self.iou_thr, self.ema, self.max_age, self.min_hits = iou_thr, ema, max_age, min_hits
        self.tracks = []                                   # dict(box, age, hits, score)

    def update(self, dets):
        """dets: [x0,y0,x1,y1,score] -> confirmed tracks as [x0,y0,x1,y1,score]."""
        dets = sorted(dets, key=lambda d: -d[4])
        used = [False] * len(self.tracks)
        for d in dets:
            best, bi = self.iou_thr, -1
            for i, t in enumerate(self.tracks):
                if not used[i]:
                    v = _iou(d[:4], t["box"])
                    if v > best:
                        best, bi = v, i
            if bi >= 0:
                t = self.tracks[bi]; used[bi] = True
                t["box"] = [self.ema * n + (1 - self.ema) * o for n, o in zip(d[:4], t["box"])]
                t["age"] = 0; t["hits"] += 1; t["score"] = d[4]
            else:
                self.tracks.append({"box": list(d[:4]), "age": 0, "hits": 1, "score": d[4]}); used.append(True)
        for i, t in enumerate(self.tracks):
            if i < len(used) and not used[i]:
                t["age"] += 1
        self.tracks = [t for t in self.tracks if t["age"] <= self.max_age]
        return [t["box"] + [t["score"]] for t in self.tracks if t["hits"] >= self.min_hits or t["age"] == 0]
