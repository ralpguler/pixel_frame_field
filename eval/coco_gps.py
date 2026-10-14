"""DensePose-COCO minival, the official geodesic point similarity (GPS) evaluation, GT-box protocol by default.

Port of research/densepose/dpeval/run_eval.py with the released model's settings as defaults (input 512, and the
v2 -> v1 chart remap: the model predicts the fold-free v2 charts on hands and feet, the COCO annotations are v1).
Each DensePose-annotated ground-truth person box is a detection of score 1 unless --boxes gives detections. The
per-person prediction is pasted into the tight box as [I, U*255, V*255] and scored by the vendored official
evaluator (eval/densepose_cocoeval.py), which reads its geodesic tables from eval/data/ (eval/README.md).

  python eval/coco_gps.py --ckpt checkpoints/pff_dp_w_MF_3k.pt \
      --ann eval/data/densepose_coco_2014_minival.json --images /data/coco/val2014

Scoring holds the evaluator's geodesic table (1.3 GB on disk; scoring peaked at 3.1 GB), so on a machine with little memory
predict and score in two processes; the scoring one never loads the model or torch:
  python eval/coco_gps.py --ckpt ... --images ... --save-dets dets.pkl --predict-only
  python eval/coco_gps.py --load-dets dets.pkl
"""
import argparse
import json
import os
import sys

import pickle

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from pff.assets import asset                                     # noqa: E402


def remap_v2_to_v1(part, U, V, remap):
    """Hands and feet (parts 3-6): v2 chart (U,V) -> v1 chart (U,V) by a nearest lookup in the (25,K,K,2) table."""
    K = remap.shape[1]
    U, V = U.copy(), V.copy()
    for pp in (3, 4, 5, 6):
        mm = part == pp
        if mm.any():
            iu = np.clip((U[mm] * K).astype(int), 0, K - 1); iv = np.clip((V[mm] * K).astype(int), 0, K - 1)
            r = remap[pp][iv, iu]
            U[mm] = r[:, 0]; V[mm] = r[:, 1]
    return np.clip(U, 0, 1), np.clip(V, 0, 1)


def maps_for_eval(pred, crop, remap):
    """(part uint8, U, V) at 512, as research run_eval.infer_maps."""
    p = pred.predict_crop(crop)
    part, U, V = p["part"], p["U"], p["V"]
    if remap is not None:
        U, V = remap_v2_to_v1(part, U, V, remap)
    return part.astype(np.uint8), U, V


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--ann", default=os.path.join(HERE, "data", "densepose_coco_2014_minival.json"))
    ap.add_argument("--images", default="", help="folder with the COCO val2014 images of minival")
    ap.add_argument("--uv-remap", default="default",
                    help="v2 -> v1 chart table (default assets/uv_v2_to_v1.npy); 'none' for a v1-chart model")
    ap.add_argument("--boxes", default="", help="detections json {image_id: [[x,y,w,h,score], ...]}")
    ap.add_argument("--limit", type=int, default=0, help="first N images only (a smoke test)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--save-dets", default="", help="write the predictions (pickle) before scoring")
    ap.add_argument("--predict-only", action="store_true", help="stop after --save-dets")
    ap.add_argument("--load-dets", default="", help="score saved predictions; no model is loaded")
    a = ap.parse_args()
    assert a.load_dets or (a.ckpt and a.images), "need --ckpt and --images, or --load-dets"

    from pycocotools.coco import COCO

    if a.load_dets:
        d = pickle.load(open(a.load_dets, "rb"))
        run_gps(COCO(a.ann), d["results"], d["img_ids"])
        return
    from pff.infer import Predictor, crop_box
    remap = None
    if a.uv_remap != "none":
        remap = np.load(asset("uv_v2_to_v1.npy") if a.uv_remap == "default" else a.uv_remap).astype(np.float32)
    boxes = {int(k): v for k, v in json.load(open(a.boxes)).items()} if a.boxes else None
    pred = Predictor(a.ckpt, a.device)
    print(f"eval {a.ckpt} on {pred.device}, input 512, uv remap {'off' if remap is None else 'v2->v1'}", flush=True)
    coco = COCO(a.ann)
    img_ids = coco.getImgIds()
    if a.limit:
        img_ids = img_ids[:a.limit]

    results = []
    for ii in img_ids:
        info = coco.loadImgs(ii)[0]
        p = os.path.join(a.images, info["file_name"])
        if not os.path.exists(p):
            continue
        rgb = cv2.imread(p)[..., ::-1].copy()
        if boxes is not None:
            bxs = [(d[:4], float(d[4])) for d in boxes.get(ii, [])]
        else:
            bxs = [(an["bbox"], 1.0) for an in coco.loadAnns(coco.getAnnIds(imgIds=ii)) if "dp_x" in an]
        for bbox, score in bxs:
            bx, by, bw, bh = bbox
            crop, (ix0, iy0, S) = crop_box(rgb, bbox)
            part, U, V = maps_for_eval(pred, crop, remap)
            partS = cv2.resize(part, (S, S), interpolation=cv2.INTER_NEAREST)
            US, VS = cv2.resize(U, (S, S)), cv2.resize(V, (S, S))
            side = 1.3 * max(bw, bh)
            sx0, sy0 = bx + bw / 2 - side / 2, by + bh / 2 - side / 2
            ox, oy = int(round(bx - sx0)), int(round(by - sy0)); Wt, Ht = int(bw) + 1, int(bh) + 1
            ox, oy = min(max(ox, 0), S - Wt), min(max(oy, 0), S - Ht)
            I = partS[oy:oy + Ht, ox:ox + Wt]
            uv = np.stack([I.astype(np.uint8), (US[oy:oy + Ht, ox:ox + Wt] * 255).astype(np.uint8),
                           (VS[oy:oy + Ht, ox:ox + Wt] * 255).astype(np.uint8)], 0)
            results.append({"image_id": ii, "category_id": 1, "bbox": [bx, by, bw, bh], "score": score, "uv": uv})
    print(f"built {len(results)} predictions over {len(set(r['image_id'] for r in results))} images", flush=True)
    if a.save_dets:
        pickle.dump({"results": results, "img_ids": img_ids, "ckpt": a.ckpt}, open(a.save_dets, "wb"))
        print(f"wrote {a.save_dets}", flush=True)
    if a.predict_only:
        return
    del pred                                   # the evaluator's geodesic table is large: free the model first
    import gc; gc.collect()
    run_gps(coco, results, img_ids)


def run_gps(coco, results, img_ids):
    from densepose_cocoeval import denseposeCOCOeval
    ev = denseposeCOCOeval(coco, coco.loadRes(results), iouType="uv")
    ev.params.imgIds = img_ids
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return ev


if __name__ == "__main__":
    main()
