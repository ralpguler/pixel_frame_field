"""Export OUR DensePose predictions in MCLD's conditioning format (arm B of the garment eval).

MCLD's preprocess (mcld/preprocess_data/densepose_interface.py) writes, per image, a full-image
uint8 canvas with channels (I, U*255, V*255) pasted into the TOP-scoring person box (background
zeros), saved with cv2.imwrite. We reproduce that exactly: torchvision person detector -> square
1.3x crop -> our model -> gather by predicted part -> v2->v1 chart remap (MCLD's texture atlas
and pose guider are v1-convention) -> paste into the box.

  python -m orient.export_mcld_densepose --images .../train --out .../train_densepose_ours \
      --ckpt ~/data/bedlam2/dp_runs/dp_csc_C/ckpt_025000.pt
"""
import argparse
import os

import cv2
import numpy as np
import torch

from model import DINOv3DPT, part_pred

ap = argparse.ArgumentParser()
ap.add_argument("--images", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_csc_C/ckpt_025000.pt"))
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--input", type=int, default=512)
ap.add_argument("--uv-remap", default=os.path.expanduser("~/data/bedlam2/dp_assets/uv_v2_to_v1.npy"))
ap.add_argument("--det-thr", type=float, default=0.5)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args()

dev = "cuda"
Z = a.input
os.makedirs(a.out, exist_ok=True)

REMAP = None
if a.uv_remap:
    REMAP = np.load(os.path.expanduser(a.uv_remap)).astype(np.float32)
    print(f"v2->v1 remap active {REMAP.shape}", flush=True)

_sd = torch.load(a.ckpt, map_location="cpu")["model"]
_dec = any(k.startswith("mask_head") for k in _sd)
m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
              pretrained=False, decoupled=_dec).to(dev).eval()
m.load_state_dict(_sd, strict=False)
print(f"ckpt={a.ckpt} decoupled={_dec}", flush=True)

from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
DET = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)


@torch.no_grad()
def infer_iuv(crop):
    img = torch.from_numpy(crop).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0]                                     # (Z,Z) 0..24
    uv = o["uv"].float()[0].view(24, 2, Z, Z)
    sel = (part - 1).clamp(0)
    U = uv[:, 0].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    V = uv[:, 1].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    P = part.cpu().numpy()
    if REMAP is not None:
        K = REMAP.shape[1]
        for pp in (3, 4, 5, 6):
            mm = P == pp
            if mm.any():
                iu = np.clip((U[mm] * K).astype(int), 0, K - 1)
                iv = np.clip((V[mm] * K).astype(int), 0, K - 1)
                r = REMAP[pp][iv, iu]
                U[mm] = r[:, 0]; V[mm] = r[:, 1]
    return P.astype(np.uint8), U, V


names = sorted(n for n in os.listdir(a.images) if n.lower().endswith((".jpg", ".png", ".jpeg")))
if a.limit:
    names = names[:a.limit]
n_ok = n_skip = 0
for i, name in enumerate(names):
    op = os.path.join(a.out, os.path.splitext(name)[0] + ".png")
    if os.path.exists(op):
        continue
    bgr = cv2.imread(os.path.join(a.images, name))
    if bgr is None:
        n_skip += 1
        continue
    rgb = bgr[..., ::-1].copy()
    H, W = rgb.shape[:2]
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(dev) / 255
    with torch.no_grad():
        d = DET([t])[0]
    keep = [(float(s), b.tolist()) for b, l, s in zip(d["boxes"], d["labels"], d["scores"])
            if int(l) == 1 and float(s) > a.det_thr]
    if not keep:
        print(f"{name} not detected !", flush=True)
        n_skip += 1
        continue
    x0, y0, x1, y1 = max(keep)[1]                              # top-scoring person, like MCLD
    bx, by, bw, bh = int(x0), int(y0), int(x1 - x0), int(y1 - y0)
    cx, cy = x0 + (x1 - x0) / 2, y0 + (y1 - y0) / 2
    side = 1.3 * max(x1 - x0, y1 - y0)
    sx0, sy0 = cx - side / 2, cy - side / 2
    S = int(round(side))
    canv = np.zeros((S, S, 3), np.uint8)
    ix0, iy0 = int(round(sx0)), int(round(sy0))
    ax0, ay0 = max(0, -ix0), max(0, -iy0)
    bx0, by0 = max(0, ix0), max(0, iy0)
    bx1, by1 = min(W, ix0 + S), min(H, iy0 + S)
    canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
    P, U, V = infer_iuv(cv2.resize(canv, (Z, Z)))
    PS = cv2.resize(P, (S, S), interpolation=cv2.INTER_NEAREST)
    US = cv2.resize(U, (S, S)); VS = cv2.resize(V, (S, S))
    ox, oy = int(round(bx - sx0)), int(round(by - sy0))
    Wt, Ht = min(bw + 1, S - max(ox, 0)), min(bh + 1, S - max(oy, 0))
    ox, oy = max(ox, 0), max(oy, 0)
    iuv = np.stack([PS[oy:oy + Ht, ox:ox + Wt],
                    (US[oy:oy + Ht, ox:ox + Wt] * 255),
                    (VS[oy:oy + Ht, ox:ox + Wt] * 255)], -1).astype(np.uint8)
    bg = np.zeros((H, W, 3), np.uint8)
    ex, ey = min(bx + Wt, W) - bx, min(by + Ht, H) - by       # clip paste to image bounds
    bg[by:by + ey, bx:bx + ex] = iuv[:ey, :ex]
    cv2.imwrite(op, bg)
    n_ok += 1
    if (i + 1) % 200 == 0:
        print(f"{i + 1}/{len(names)} ok={n_ok} skip={n_skip}", flush=True)
print(f"DONE {n_ok} written, {n_skip} skipped -> {a.out}", flush=True)
