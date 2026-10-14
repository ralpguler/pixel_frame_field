"""Scan pexels clips: sample a few frames each, count NMS'd person detections, rank by max persons.
-> prints clips sorted by crowd size so we can pick genuinely multi-person ones for the MAMMA showcase."""
import os, sys, glob, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from orient.detect_track import detect_persons


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, iou_thr, min_side):
    dets = [d for d in dets if max(d[2] - d[0], d[3] - d[1]) >= min_side]
    dets = sorted(dets, key=lambda d: -d[4]); keep = []
    for d in dets:
        if all(_iou(d, k) < iou_thr for k in keep):
            keep.append(d)
    return keep


clips = sorted(glob.glob(os.path.expanduser("~/data/datasets/pexels_hifps/clips/*_conformed16.mp4")))
rows = []
for p in clips:
    cap = cv2.VideoCapture(p); n = int(cap.get(7))
    if n <= 0:
        continue
    counts = []
    for fr_i in [int(n * 0.3), int(n * 0.6)]:                 # sample 2 frames
        cap.set(1, fr_i); ok, bgr = cap.read()
        if not ok:
            continue
        rgb = bgr[..., ::-1]
        d = nms(detect_persons(rgb, thr=0.5), 0.6, 0.06 * rgb.shape[0])   # min 6% frame-height persons
        counts.append(len(d))
    cap.release()
    mx = max(counts) if counts else 0
    rows.append((mx, os.path.basename(p).replace("_conformed16.mp4", "")))
    print(f"{mx:2d}  {os.path.basename(p)}", flush=True)

rows.sort(reverse=True)
print("\n=== TOP MULTI-PERSON CLIPS ===", flush=True)
for mx, cid in rows[:15]:
    print(f"{mx} persons  {cid}", flush=True)
