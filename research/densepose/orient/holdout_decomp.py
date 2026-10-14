"""Decompose the F0 holdout degradation: occlusion forgetting vs annotation-convention shift.

Two mechanisms, two fingerprints (Riza's hypothesis vs mine, 2026-07-17):
  A. occlusion forgetting  -> degradation concentrates in the fg-BOUNDARY band and in
     low-coverage (occluded) instances; error growth lives in the TAIL (chart jumps).
  B. COCO-convention shift -> degradation ~uniform across the DEEP INTERIOR, shows up as a
     SIGNED per-part UV bias (median shift), equally on isolated instances.

Runs two ckpts over the SAME held-out MAMMA batches (native decode, probe protocol) and prints:
  region split (interior / boundary band) x {part_acc, uv_L1, geo_e2e}
  coverage-tercile split of per-sample geo_e2e
  per-part signed UV bias (du, dv) -- the convention fingerprint
  uv error quantiles (median vs p95) -- median shift vs tail growth

  python -m orient.holdout_decomp --ckpts A.pt B.pt --shards "~/.../holdout_links/*.tar"
"""
import argparse
import glob
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from data import build_loader
from model import DINOv3DPT, sixd_to_R, gather_rot6d, N_PARTS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--shards", required=True)
ap.add_argument("--nshards", type=int, default=59)
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--band", type=int, default=12, help="boundary band width px (native res)")
a = ap.parse_args()
dev = "cuda"

allsh = sorted(glob.glob(os.path.expanduser(a.shards)))
sub = list(np.random.RandomState(0).permutation(allsh)[: a.nshards])


@torch.no_grad()
def run(ckpt):
    sd = torch.load(ckpt, map_location="cpu")["model"]
    rpp = sd["rot_head.2.weight"].shape[0] != 6
    m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23),
                  pretrained=False, decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
    m.load_state_dict(sd)
    ld = build_loader("val", batch=16, workers=2, input_size=512, train=False, orient=True,
                      shards=sub)
    reg = {r: {"pc": np.zeros(2), "uv": np.zeros(2), "geo": np.zeros(2)}
           for r in ("interior", "band")}
    samples = []                                            # (coverage, geo_e2e)
    bias = np.zeros((N_PARTS, 3))                           # sum du, sum dv, n
    uv_abs = []                                             # per-pixel |duv| subsample
    for i, b in enumerate(ld):
        if i >= a.batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(b["img"].to(dev))
        for bi, g in enumerate(b["gt"]):
            part, uvg, fg = g[0].to(dev), g[1].to(dev), g[2].numpy()
            H, W = part.shape
            if fg.sum() < 500:
                continue
            er = cv2.erode(fg.astype(np.uint8), np.ones((a.band, a.band), np.uint8)) > 0
            masks = {"interior": torch.from_numpy(er).to(dev),
                     "band": torch.from_numpy(fg & ~er).to(dev)}
            p = F.interpolate(out["part"][bi:bi + 1].float(), (H, W), mode="bilinear",
                              align_corners=False)[0]
            pred = p.argmax(0) + 1
            u = F.interpolate(out["uv"][bi:bi + 1].float(), (H, W), mode="bilinear",
                              align_corners=False)[0].view(N_PARTS, 2, H, W)
            idx = (part - 1).clamp(0)
            sel = u.gather(0, idx[None, None].expand(1, 2, H, W)).squeeze(0)   # GT-part gather
            duv = (sel - uvg).permute(1, 2, 0)                                  # (H,W,2) signed
            rg = g[3].to(dev).float(); rfg = g[4].numpy()
            Rh, Rw = rfg.shape
            r_pp = F.interpolate(out["rot6d"][bi:bi + 1].float(), (Rh, Rw), mode="bilinear",
                                 align_corners=False)
            pm = F.interpolate(pred[None, None].float(), (Rh, Rw), mode="nearest")[0, 0].long()
            err = cv2.erode(rfg.astype(np.uint8),
                            np.ones((max(2, a.band * Rh // H),) * 2, np.uint8)) > 0
            rmasks = {"interior": torch.from_numpy(err).to(dev),
                      "band": torch.from_numpy(rfg & ~err).to(dev)}
            geo_all = None
            for rname in ("interior", "band"):
                mk = masks[rname]
                if mk.any():
                    reg[rname]["pc"] += [(pred[mk] == part[mk]).sum().item(), mk.sum().item()]
                    reg[rname]["uv"] += [duv[mk].abs().sum().item(), mk.sum().item() * 2]
                rmk = rmasks[rname] & torch.from_numpy(rfg).to(dev)
                if rmk.any():
                    Rp = sixd_to_R(gather_rot6d(r_pp, pm[None])[0].permute(1, 2, 0)[rmk])
                    Rg = sixd_to_R(rg[rmk])
                    tr = torch.matmul(Rp.transpose(1, 2), Rg).diagonal(dim1=1, dim2=2).sum(1)
                    gsum = (torch.arccos(((tr - 1) / 2).clamp(-1, 1)).sum() * 180 / np.pi).item()
                    reg[rname]["geo"] += [gsum, rmk.sum().item()]
                    geo_all = gsum if geo_all is None else geo_all + gsum
            fgt = torch.from_numpy(fg).to(dev)
            samples.append((fg.mean(), ((pred[fgt] != part[fgt]).float().mean().item())))
            for pp in range(1, N_PARTS + 1):
                mk = (part == pp) & fgt
                if mk.sum() > 50:
                    d = duv[mk]
                    bias[pp - 1] += [d[:, 0].sum().item(), d[:, 1].sum().item(), mk.sum().item()]
            sub_px = duv[fgt].norm(dim=1)
            uv_abs.append(sub_px[:: max(1, len(sub_px) // 500)].cpu().numpy())
    return reg, np.array(samples), bias, np.concatenate(uv_abs)


names = [os.path.basename(os.path.dirname(c)) for c in a.ckpts]
res = {n: run(c) for n, c in zip(names, a.ckpts)}

print(f"\n=== region split (band={a.band}px) ===")
print(f"{'model':>12} {'region':>9} {'part_acc':>9} {'uv_L1':>8} {'geo_e2e':>8}")
for n in names:
    reg = res[n][0]
    for r in ("interior", "band"):
        d = reg[r]
        print(f"{n:>12} {r:>9} {d['pc'][0]/max(d['pc'][1],1):>9.4f} "
              f"{d['uv'][0]/max(d['uv'][1],1):>8.4f} {d['geo'][0]/max(d['geo'][1],1):>8.2f}")

print("\n=== per-sample part-error vs coverage (occluded = low coverage) ===")
for n in names:
    s = res[n][1]
    q = np.quantile(s[:, 0], [1 / 3, 2 / 3])
    lo = s[s[:, 0] <= q[0], 1].mean(); mid = s[(s[:, 0] > q[0]) & (s[:, 0] <= q[1]), 1].mean()
    hi = s[s[:, 0] > q[1], 1].mean()
    print(f"{n:>12}  part-err%: low-cov {100*lo:.2f}  mid {100*mid:.2f}  high-cov {100*hi:.2f}")

print("\n=== uv |error| quantiles (median shift vs tail growth) ===")
for n in names:
    e = res[n][3]
    print(f"{n:>12}  p50 {np.median(e):.4f}  p90 {np.quantile(e,.9):.4f}  p99 {np.quantile(e,.99):.4f}")

print("\n=== per-part signed UV bias (convention fingerprint; top 8 by |bias| of 2nd model) ===")
b0, b1 = res[names[0]][2], res[names[1]][2]
mag = np.hypot(b1[:, 0] / np.clip(b1[:, 2], 1, None), b1[:, 1] / np.clip(b1[:, 2], 1, None))
for pp in np.argsort(-mag)[:8]:
    d0 = b0[pp, :2] / max(b0[pp, 2], 1); d1 = b1[pp, :2] / max(b1[pp, 2], 1)
    print(f"  part {pp+1:2d}: {names[0]} ({d0[0]:+.4f},{d0[1]:+.4f})  "
          f"{names[1]} ({d1[0]:+.4f},{d1[1]:+.4f})")
