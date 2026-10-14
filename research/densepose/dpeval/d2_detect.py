"""Run a strong detectron2 detector -> person boxes json  {image_id: [[x,y,w,h,score],...]}.
Runs INSIDE the NGC pytorch apptainer container (where detectron2 is built). Output is consumed by
run_eval.py --boxes (no detectron2 needed in our venv) and by the video scripts.

Modes:
  --mode coco  : boxes for the DensePose minival images (keyed by COCO image_id)
  --mode dir   : boxes for every frame_*.png/jpg in --img-dir (keyed by zero-based frame index)

Default detector = the strongest standard model-zoo config (Cascade Mask R-CNN X-152, box AP ~50);
override with --config. COCO person class id = 0 in detectron2.
"""
import os, sys, json, glob, argparse, numpy as np, cv2, torch
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["coco", "dir"], default="coco")
ap.add_argument("--config", default="Misc/cascade_mask_rcnn_X_152_32x8d_FPN_IN5k_gn_dconv.yaml")
ap.add_argument("--ann", default="data/densepose_coco_2014_minival.json")
ap.add_argument("--img-dir", default=os.path.expanduser("~/data/coco_val2014_minival"))
ap.add_argument("--out", required=True)
ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(a.config))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(a.config)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = a.thr
cfg.MODEL.DEVICE = "cuda"
pred = DefaultPredictor(cfg)
print(f"detectron2 detector: {a.config} thr={a.thr}", flush=True)


def detect(bgr):
    inst = pred(bgr)["instances"]
    keep = inst.pred_classes == 0                       # person
    bx = inst.pred_boxes.tensor[keep].cpu().numpy()
    sc = inst.scores[keep].cpu().numpy()
    return [[float(x0), float(y0), float(x1 - x0), float(y1 - y0), float(s)]
            for (x0, y0, x1, y1), s in zip(bx, sc)]


out = {}
if a.mode == "coco":
    from pycocotools.coco import COCO
    coco = COCO(a.ann)
    ids = coco.getImgIds()
    for n, ii in enumerate(ids):
        info = coco.loadImgs(ii)[0]; p = os.path.join(a.img_dir, info["file_name"])
        if not os.path.exists(p):
            continue
        out[str(ii)] = detect(cv2.imread(p))
        if n % 200 == 0:
            print(f"  {n}/{len(ids)}", flush=True)
else:
    frames = sorted(glob.glob(os.path.join(a.img_dir, "*.png")) + glob.glob(os.path.join(a.img_dir, "*.jpg")))
    for n, p in enumerate(frames):
        out[str(n)] = detect(cv2.imread(p))
        if n % 100 == 0:
            print(f"  {n}/{len(frames)}", flush=True)

json.dump(out, open(a.out, "w"))
print(f"wrote {a.out}: {len(out)} images, {sum(len(v) for v in out.values())} person boxes", flush=True)
