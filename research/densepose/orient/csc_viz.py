"""Visualise the Canonical Surface Coordinate map and the CSC losses on real training samples.

Per sample, a row of panels:
  1 image
  2 GT canonical coords      (template XYZ -> RGB, "NOCS" colouring)
  3 PREDICTED canonical coords, i.e. CSC(part, uv_pred) -- the thing the new loss supervises
  4 surface error |X_pred - X_gt| in mm
  5 gradient residual |grad X_pred - grad X_gt| -- what L_grad actually penalises
  6 validity: where the chart REPRODUCES the GT point (green) vs where it cannot (red = the fold:
    hand palm/back, foot, head eyeballs+interior). Red pixels are DROPPED by the loss.

Panel 6 is also the direct picture of the chart-fold bug: run with --uv-asset smplx_face_uv.npy to see
the broken charts, and with smplx_face_uv_v2.npy to see them fixed.

  python -m orient.csc_viz --ckpt .../ckpt_120000.pt --uv-asset smplx_face_uv.npy --n 6
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from data import build_loader
from model import DINOv3DPT, N_PARTS
from orient.csc import CSC

ap = argparse.ArgumentParser()
D = os.path.expanduser("~/data/bedlam2/dp_runs")
ap.add_argument("--ckpt", default=f"{D}/dp_pp_llrd_1e3_mamma/ckpt_120000.pt")
ap.add_argument("--uv-asset", default="smplx_face_uv.npy")
ap.add_argument("--csc-K", type=int, default=512)
ap.add_argument("--gather", choices=["gt", "pred"], default="pred")
ap.add_argument("--n", type=int, default=6)
ap.add_argument("--tau", type=float, default=0.005)
ap.add_argument("--clip", type=float, default=0.05)
ap.add_argument("--emax", type=float, default=40.0, help="colour ceiling for the error panel (mm)")
ap.add_argument("--gmax", type=float, default=25.0, help="colour ceiling for the gradient panel (mm/px)")
ap.add_argument("--out", default=os.path.expanduser("~/data/dp_viz_compare/csc/csc_diag.png"))
a = ap.parse_args()

dev = "cuda"
os.makedirs(os.path.dirname(a.out), exist_ok=True)

sd = torch.load(a.ckpt, map_location="cpu")["model"]
rpp = sd["rot_head.2.weight"].shape[0] != 6
m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23), pretrained=False,
              decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
m.load_state_dict(sd)
csc = CSC(a.csc_K).to(dev)

loader = build_loader("val", batch=a.n, workers=2, input_size=512, train=False, orient=True,
                      csc=True, uv_asset=a.uv_asset)
b = next(iter(loader))
img = b["img"].to(dev)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(img)

# template bbox -> a stable RGB code for canonical XYZ (same mapping for GT and pred, so they compare)
vt = np.load(os.path.expanduser("~/data/bedlam2/dp_assets/smplx_v_template.npy"))
LO = torch.tensor(vt.min(0), device=dev); SP = torch.tensor(vt.max(0) - vt.min(0), device=dev)


def nocs(X, m_):                                        # X (3,H,W) metres -> (H,W,3) uint8
    c = ((X.permute(1, 2, 0) - LO) / SP).clamp(0, 1)
    c = (c * 255).byte().cpu().numpy()
    c[~m_.cpu().numpy()] = 20
    return c


def heat(v, m_, vmax):                                  # scalar (H,W) -> turbo-ish RGB, black where !m_
    t = (v / vmax).clamp(0, 1).cpu().numpy()
    r = np.clip(1.5 - np.abs(4 * t - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * t - 2), 0, 1)
    bl = np.clip(1.5 - np.abs(4 * t - 1), 0, 1)
    c = (np.stack([r, g, bl], -1) * 255).astype(np.uint8)
    c[~m_.cpu().numpy()] = 20
    return c


rows = []
stats = []
for i in range(a.n):
    g = b["gt"][i]
    cfg = g[8].to(dev); R = cfg.shape[0]
    Xg = g[5].to(dev).float().permute(2, 0, 1)
    cpart = g[6].to(dev).long().clamp(min=1)
    cuv = g[7].to(dev).float()
    with torch.no_grad():
        ok = (csc(cuv, cpart) - Xg).norm(dim=0) < a.tau          # chart reproduces the GT point?
        u_r = F.interpolate(out["uv"][i:i + 1].float(), (R, R), mode="bilinear",
                            align_corners=False)[0].view(N_PARTS, 2, R, R).clamp(0, 1)
        if a.gather == "pred":
            p_r = F.interpolate(out["part"][i:i + 1].float(), (R, R), mode="bilinear",
                                align_corners=False)[0]
            pid = p_r.argmax(0) + 1
        else:
            pid = cpart
        sel = u_r.gather(0, (pid - 1).clamp(min=0)[None, None].expand(1, 2, R, R)).squeeze(0)
        Xp = csc(sel, pid)
        err = (Xp - Xg).norm(dim=0) * 1000                        # mm
        # gradient residual, exactly what L_grad sees (both sides clipped)
        gr = torch.zeros_like(err)
        for d in (-1, -2):
            gp = torch.diff(Xp, dim=d).clamp(-a.clip, a.clip)
            gg = torch.diff(Xg, dim=d).clamp(-a.clip, a.clip)
            r_ = (gp - gg).abs().sum(0) * 1000
            gr += F.pad(r_[None, None], (0, 1, 0, 0) if d == -1 else (0, 0, 0, 1))[0, 0]
    mv = cfg & ok
    im = (img[i].permute(1, 2, 0) * 255).byte().cpu().numpy()
    im = np.array(torch.nn.functional.interpolate(
        torch.from_numpy(im).permute(2, 0, 1)[None].float(), (R, R), mode="bilinear")[0]
        .permute(1, 2, 0).byte())
    val = np.zeros((R, R, 3), np.uint8)
    val[(cfg & ok).cpu().numpy()] = (60, 200, 90)                 # used by the loss
    val[(cfg & ~ok).cpu().numpy()] = (220, 60, 60)                # chart is degenerate here -> dropped
    row = np.concatenate([im, nocs(Xg, cfg), nocs(Xp, cfg),
                          heat(err, mv, a.emax), heat(gr, mv, a.gmax), val], 1)
    import cv2                                             # GT res is per-sample: normalise the row height
    S = 300
    rows.append(cv2.resize(row, (6 * S, S), interpolation=cv2.INTER_AREA))
    e = err[mv]
    stats.append((float(e.median()), float(e.mean()), float((cfg & ~ok).sum()) / max(float(cfg.sum()), 1)))

grid = np.concatenate(rows, 0)
import imageio.v2 as iio
iio.imwrite(a.out, grid)
med = np.mean([s[0] for s in stats]); mn = np.mean([s[1] for s in stats])
drop = np.mean([s[2] for s in stats])
print(f"[csc_viz] {a.ckpt.split('/')[-2]}/{a.ckpt.split('/')[-1]}  uv_asset={a.uv_asset}  gather={a.gather}")
print(f"[csc_viz] surface error over {a.n} samples: median {med:.1f} mm | mean {mn:.1f} mm")
print(f"[csc_viz] chart-degenerate pixels DROPPED by the loss: {100*drop:.1f}% of foreground")
print(f"[csc_viz] panels: image | GT coords | PRED coords | error(mm) | grad residual | validity")
print(f"[csc_viz] wrote {a.out}")
