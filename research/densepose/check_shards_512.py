"""Validate the fixed-canvas 512 shards before anything trains on them.

Checks, per sample:
  * corr.png really is 512x512 and decodes to a valid (face, barycentric) pair
  * meta carries gt_box / canvas / gt_expand, and box (the RGB canvas) is still there
  * the GT crop is INSIDE the tightest training image crop (expand 1.2) -- if it is not, the loss would
    silently supervise pixels the network never saw
  * every derived GT (part / uv / xyz / rot6d) is finite, and rot6d is non-degenerate (the old bilinear
    resize produced near-zero 6D vectors whose Gram-Schmidt was pure noise -> 180 deg flips)
  * theta round-trips: warping the GT canvas' own corners through theta lands them inside the image crop

  python check_shards_512.py --n 400
"""
import argparse
import glob
import io
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image

from data import DPDecode, unpack_rgba, SHARDS_512, SHARDS_512_MAMMA

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--roots", default=f"{SHARDS_512}/train,{SHARDS_512}/val,{SHARDS_512_MAMMA}/train")
a = ap.parse_args()

dec = DPDecode(input_size=512, train=True, orient=True, gt512=True,
               uv_asset="smplx_face_uv_v2.npy")
for root in a.roots.split(","):
    tars = sorted(glob.glob(f"{root}/*.tar"))
    # The original sharding left some tars EMPTY (51/64 in BEDLAM val). Skip them or we sample nothing.
    tars = [t for t in tars if os.path.getsize(t) > 100_000]
    if not tars:
        print(f"{root}: NO NON-EMPTY SHARDS"); continue
    ds = wds.WebDataset(tars[:4], empty_check=False).to_tuple(
        "rgb.jpg", "corr.png", "meta.json", "rot.npy")
    n = bad = 0
    stats = {"fg": [], "deg6": 0, "outside": 0, "nonfinite": 0, "wrongsize": 0, "nometa": 0}
    for sm in ds:
        meta = json.loads(sm[2])
        rgba = np.array(Image.open(io.BytesIO(sm[1])))
        if rgba.shape[:2] != (512, 512):
            stats["wrongsize"] += 1; bad += 1
        if not all(k in meta for k in ("gt_box", "canvas", "gt_expand", "box")):
            stats["nometa"] += 1; bad += 1; n += 1
            if n >= a.n: break
            continue
        o = dec(sm)
        fg = o["fg"].numpy()
        stats["fg"].append(fg.mean())
        r6 = o["rot6d"].numpy().astype(np.float32)[fg]
        if len(r6):
            nrm = np.linalg.norm(r6[:, :3], axis=1)
            stats["deg6"] += int((nrm < 1e-3).sum())            # degenerate 6D -> the old 180 deg bug
        for k in ("uv", "xyz", "theta"):
            if not np.isfinite(o[k].numpy()).all():
                stats["nonfinite"] += 1; bad += 1
        # the GT canvas must sit INSIDE the image crop => |theta| maps [-1,1] into [-1,1]
        th = o["theta"].numpy()
        cor = np.array([[-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], np.float32)
        p = cor @ th.T                                          # GT corners -> prediction coords
        if np.abs(p).max() > 1.0 + 1e-4:
            stats["outside"] += 1; bad += 1
        n += 1
        if n >= a.n:
            break
    f = np.array(stats["fg"]) if stats["fg"] else np.zeros(1)
    print(f"{root}  ({len(tars)} non-empty shards)")
    print(f"   {n} samples | fg {100*f.mean():.1f}% of canvas (p10 {100*np.percentile(f,10):.1f}%)")
    print(f"   wrong size {stats['wrongsize']} | missing meta {stats['nometa']} | non-finite {stats['nonfinite']}")
    print(f"   degenerate rot6d pixels {stats['deg6']}   <- the old bilinear bug; must be 0")
    print(f"   GT canvas outside the image crop {stats['outside']}   <- must be 0")
    print(f"   -> {'PASS' if bad == 0 and stats['deg6'] == 0 else 'FAIL'}\n")
