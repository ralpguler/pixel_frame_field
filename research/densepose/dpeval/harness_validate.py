"""Harness cross-validation WITHOUT a trained model: feed GROUND-TRUTH IUV (and degraded versions)
as the 'prediction' and check the AP^GPS our harness reports.

If the eval plumbing (prediction encoding, point sampling, vertex lookup, geodesic distance, GPS)
is faithful and NOT inflating:
  - perfect predictor (dt == GT at the GT points)  -> AP ~ 1.0
  - GT with small UV noise                          -> high but < 1
  - GT with large UV noise                          -> drops further
  - random surface coords                           -> collapses toward 0
A monotone curve anchored at ~1.0 (perfect) and ~0 (random) proves the metric scale is correct and
our 0.72/0.73 is a faithful reading between those bounds, not a leniency artifact.

Indexing mirrors failure_report.py's per_point_gps EXACTLY (that code reproduces the official GPS),
so the dt map we build is read back by the eval at precisely the GT point pixels.
"""
import os, sys, numpy as np, torch
sys.path.insert(0, ".."); sys.path.insert(0, ".")
from pycocotools.coco import COCO
from densepose_cocoeval import denseposeCOCOeval

GT = "data/densepose_coco_2014_minival.json"
coco = COCO(GT)
rng = np.random.RandomState(0)


def build_results(mode, sigma=0.0):
    """mode: 'perfect' | 'noise' | 'random'. Returns COCO-format dt list with dense uv per instance."""
    results = []
    for ii in coco.getImgIds():
        for ann in coco.loadAnns(coco.getAnnIds(imgIds=ii)):
            if "dp_x" not in ann:
                continue
            bx, by, bw, bh = ann["bbox"]
            Wt, Ht = int(bw) + 1, int(bh) + 1
            I = np.zeros((Ht, Wt), np.uint8)
            U = np.zeros((Ht, Wt), np.uint8)
            V = np.zeros((Ht, Wt), np.uint8)
            dpx = np.asarray(ann["dp_x"], np.float64) * bw / 255.0   # within-bbox px (col)
            dpy = np.asarray(ann["dp_y"], np.float64) * bh / 255.0   # within-bbox px (row)
            col = np.clip(dpx.astype(int), 0, Wt - 1)
            row = np.clip(dpy.astype(int), 0, Ht - 1)
            dI = np.asarray(ann["dp_I"], np.float64)
            dU = np.asarray(ann["dp_U"], np.float64)
            dV = np.asarray(ann["dp_V"], np.float64)
            if mode == "random":
                dU = rng.rand(len(dU)); dV = rng.rand(len(dV))
                dI = rng.randint(1, 25, size=len(dI)).astype(np.float64)
            elif mode == "noise":
                dU = np.clip(dU + rng.normal(0, sigma, len(dU)), 0, 1)
                dV = np.clip(dV + rng.normal(0, sigma, len(dV)), 0, 1)
            I[row, col] = dI.astype(np.uint8)
            U[row, col] = np.clip(dU * 255, 0, 255).astype(np.uint8)
            V[row, col] = np.clip(dV * 255, 0, 255).astype(np.uint8)
            results.append({"image_id": ii, "category_id": 1, "bbox": [bx, by, bw, bh],
                            "score": 1.0, "uv": np.stack([I, U, V], 0)})
    return results


def run(mode, sigma=0.0):
    res = build_results(mode, sigma)
    ev = denseposeCOCOeval(coco, coco.loadRes(res), iouType="uv")
    ev.params.imgIds = coco.getImgIds()
    ev.evaluate(); ev.accumulate()
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ev.summarize()
    ap = ev.stats[0]
    tag = f"{mode}" + (f"(sigma={sigma})" if mode == "noise" else "")
    print(f"  {tag:22s} AP^GPS = {ap:.4f}", flush=True)
    return ap


if __name__ == "__main__":
    print("HARNESS CROSS-VALIDATION (GT fed as prediction):", flush=True)
    run("perfect")
    run("noise", 0.02)
    run("noise", 0.05)
    run("noise", 0.10)
    run("noise", 0.20)
    run("random")
