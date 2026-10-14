"""Evaluate an oriented-DensePose checkpoint on the CLEAN fixed-canvas 512 GT.

Reports the aggregate metrics AND the local ones, because we established the aggregate is blind to the
defect that actually shows up in the visualisations:

  geo_e2e (median geodesic rotation error) moved by 0.04 deg between gathering the rotation by the GT part
  and by argmax -- while argmax simultaneously produced a +13 deg SPURIOUS JUMP on ~2% of edges and MISSED
  a real 13 deg discontinuity on another ~2%. A mean over pixels cannot see a misplaced discontinuity.

So every run is scored on:
  * geo_e2e          median geodesic rotation error (deg)      -- the old headline
  * surf_mm          median |X_pred - X_gt| on the template (mm)
  * part_acc         argmax part accuracy
  * JUMP metrics     spurious / missed rotation discontinuities at part boundaries  <-- the local one
  * per-region       the same, restricted to hands+feet (parts 3,4,5,6) and to the head

  python -m orient.eval_512 --ckpt .../ckpt_25000.pt [--n 512]
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import build_loader
from model import DINOv3DPT, N_PARTS, gather_rot6d, sixd_to_R, backbone_from_state_dict, default_layers
from orient.csc import CSC

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--n", type=int, default=512, help="samples")
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--uv-asset", default="smplx_face_uv_v2.npy")
ap.add_argument("--csc-K", type=int, default=512)
a = ap.parse_args()
dev = "cuda"

sd = torch.load(a.ckpt, map_location="cpu")["model"]
m = DINOv3DPT(backbone=backbone_from_state_dict(sd), layers=default_layers(backbone_from_state_dict(sd)), pretrained=False,
              decoupled=True, orient=True,
              rot_per_part=sd["rot_head.2.weight"].shape[0] != 6).to(dev).eval()
m.load_state_dict(sd)
csc = CSC(a.csc_K).to(dev)

HANDS_FEET = {3, 4, 5, 6}
HEAD = {23, 24}


def geo(A, B):
    tr = torch.einsum("...ij,...ij->...", A, B)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


acc = {k: [] for k in ("geo", "surf", "pacc", "geo_hf", "surf_hf", "geo_hd")}
JUMP = {"spurious": [0, 0.0, 0.0], "missed": [0, 0.0, 0.0], "real": [0, 0.0, 0.0], "interior": [0, 0.0, 0.0]}
n = 0
ld = build_loader("val", batch=a.batch, workers=6, input_size=512, train=False, orient=True,
                  uv_asset=a.uv_asset, gt512=True)
for b in ld:
    img = b["img"].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out = m(img)
    part = b["part"].to(dev); fg = b["fg"].to(dev); theta = b["theta"].to(dev).float()
    rgt = b["rot6d"].to(dev).float(); Xg = b["xyz"].to(dev).float().permute(0, 3, 1, 2)
    B, G, _ = part.shape
    grid = F.affine_grid(theta, (B, 1, G, G), align_corners=False)
    W = lambda x: F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border",
                                align_corners=False)
    with torch.no_grad():
        pl = W(out["part"]); uvp = W(out["uv"]).view(B, N_PARTS, 2, G, G); rp = W(out["rot6d"])
        pid = pl.argmax(1) + 1                                        # TEST-TIME part
        Rg = sixd_to_R(rgt)
        Rax = sixd_to_R(gather_rot6d(rp, pid).permute(0, 2, 3, 1))    # test-time rotation field
        e = geo(Rax, Rg)
        sel = uvp.gather(1, (pid - 1)[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
        Xp = torch.stack([csc(sel[i], pid[i]) for i in range(B)])
        s = (Xp - Xg).norm(dim=1) * 1000
    hf = torch.isin(part, torch.tensor(sorted(HANDS_FEET), device=dev)) & fg
    hd = torch.isin(part, torch.tensor(sorted(HEAD), device=dev)) & fg
    if fg.any():
        acc["geo"].append(float(e[fg].median())); acc["surf"].append(float(s[fg].median()))
        acc["pacc"].append(float((pid[fg] == part[fg]).float().mean()))
    if hf.any():
        acc["geo_hf"].append(float(e[hf].median())); acc["surf_hf"].append(float(s[hf].median()))
    if hd.any():
        acc["geo_hd"].append(float(e[hd].median()))
    # ---- LOCAL: is the rotation discontinuity in the RIGHT PLACE? ----
    P = part; Q = pid
    for dy, dx in ((0, 1), (1, 0)):
        sh = lambda A: torch.roll(torch.roll(A, dy, 1), dx, 2)
        m2 = fg & sh(fg)
        gflip = P != sh(P); aflip = Q != sh(Q)
        stp = geo(Rax, sh(Rax)); stg = geo(Rg, sh(Rg))
        for nm, msk in (("spurious", m2 & aflip & ~gflip), ("missed", m2 & ~aflip & gflip),
                        ("real", m2 & aflip & gflip), ("interior", m2 & ~aflip & ~gflip)):
            if msk.any():
                JUMP[nm][0] += int(msk.sum())
                JUMP[nm][1] += float(stp[msk].sum()); JUMP[nm][2] += float(stg[msk].sum())
    n += B
    if n >= a.n:
        break

print(f"\n{os.path.basename(os.path.dirname(a.ckpt))}/{os.path.basename(a.ckpt)}   ({n} val samples)\n")
print("AGGREGATE (the old headline -- blind to misplaced discontinuities):")
print(f"   geo_e2e        {np.mean(acc['geo']):6.2f} deg      part_acc {100*np.mean(acc['pacc']):5.1f}%")
print(f"   surf_err       {np.mean(acc['surf']):6.2f} mm")
print(f"   hands+feet     geo {np.mean(acc['geo_hf']):6.2f} deg   surf {np.mean(acc['surf_hf']):6.2f} mm")
print(f"   head           geo {np.mean(acc['geo_hd']):6.2f} deg")
tot = sum(v[0] for v in JUMP.values())
print("\nLOCAL -- is the rotation discontinuity in the right place?")
print(f"   {'edge type':28s} {'share':>7s} {'pred step':>10s} {'GT step':>9s} {'excess':>9s}")
for nm, v in JUMP.items():
    if not v[0]:
        continue
    p_, g_ = v[1] / v[0], v[2] / v[0]
    print(f"   {nm:28s} {100*v[0]/tot:6.2f}% {p_:9.2f}d {g_:8.2f}d {p_-g_:+8.2f}d")
sp = JUMP["spurious"][0] / max(tot, 1); ms = JUMP["missed"][0] / max(tot, 1)
print(f"\n   MISPLACED-DISCONTINUITY RATE: {100*(sp+ms):.2f}% of edges "
      f"(spurious {100*sp:.2f}% + missed {100*ms:.2f}%)   <- lower is better")
