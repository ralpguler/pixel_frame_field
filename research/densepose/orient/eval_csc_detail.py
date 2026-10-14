"""Detailed CSC-loss decomposition on the clean 512 val: WHERE does the surface gain live?

Per checkpoint:
  * surf error percentiles (p50/p75/p90/p99) -- tail vs body of the distribution
  * per-region surf + geo: torso / arms / legs / hands / feet / head
  * SEAM analysis (the loss's raison d'etre): GT part-boundary edges are physically adjacent points
    on the body, so the TRUE XYZ step across them is tiny; a chart-jump prediction makes it huge.
    Reported: median |Xp step| vs |Xg step| on seam edges, and the fraction of seam edges whose
    predicted step exceeds 30mm while the GT step is <10mm ("seam breaks").
  * interior gradient fidelity: | |grad Xp| - |grad Xg| | on same-part edges (the MiDaS-style term).
  * band analysis: surf error within 3px of a GT part boundary vs interior.

  python -m orient.eval_csc_detail --ckpt path1 [path2 ...] [--n 512]
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
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--uv-asset", default="smplx_face_uv_v2.npy")
a = ap.parse_args()
dev = "cuda"

REG = {"torso": (1, 2), "hands": (3, 4), "feet": (5, 6), "legs": tuple(range(7, 15)),
       "arms": tuple(range(15, 23)), "head": (23, 24)}

csc = CSC(512).to(dev)


def run(ckpt):
    sd = torch.load(ckpt, map_location="cpu")["model"]
    m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23), pretrained=False,
                  decoupled=True, orient=True,
                  rot_per_part=sd["rot_head.2.weight"].shape[0] != 6).to(dev).eval()
    m.load_state_dict(sd)
    ld = build_loader("val", batch=a.batch, workers=6, input_size=512, train=False, orient=True,
                      uv_asset=a.uv_asset, gt512=True)
    S = {k: [] for k in REG}
    G = {k: [] for k in REG}
    allsurf = []
    seam_p, seam_g, seam_break, seam_tot = [], [], 0, 0
    band_in, band_out = [], []
    grad_res = []
    n = 0
    for b in ld:
        img = b["img"].to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(img)
        part = b["part"].to(dev); fg = b["fg"].to(dev)
        theta = b["theta"].to(dev).float()
        Xg = b["xyz"].to(dev).float().permute(0, 3, 1, 2)
        B, Gs, _ = part.shape
        grid = F.affine_grid(theta, (B, 1, Gs, Gs), align_corners=False)
        W = lambda x: F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border",
                                    align_corners=False)
        with torch.no_grad():
            pl = W(out["part"]); uvp = W(out["uv"]).view(B, N_PARTS, 2, Gs, Gs)
            pid = pl.argmax(1) + 1
            sel = uvp.gather(1, (pid - 1)[:, None, None].expand(B, 1, 2, Gs, Gs)).squeeze(1)
            Xp = torch.stack([csc(sel[i], pid[i]) for i in range(B)])
            e = (Xp - Xg.permute(0, 2, 3, 1).permute(0, 3, 1, 2) if False else Xp - Xg).norm(dim=1) * 1000
        allsurf.append(e[fg].cpu())
        # geodesic rotation per region needs rot; skip (already in eval_512) -- focus on surface
        for k, ps in REG.items():
            msk = torch.isin(part, torch.tensor(ps, device=dev)) & fg
            if msk.any():
                S[k].append(e[msk].cpu())
        # edges
        for dy, dx in ((0, 1), (1, 0)):
            sh = lambda A_: torch.roll(torch.roll(A_, dy, -2), dx, -1)
            m2 = fg & sh(fg)
            gflip = (part != sh(part)) & m2
            same = (~(part != sh(part))) & m2
            stp = (Xp - sh(Xp)).norm(dim=1) * 1000
            stg = (Xg - sh(Xg)).norm(dim=1) * 1000
            true_seam = gflip & (stg < 10.0)                     # physically adjacent across the seam
            if true_seam.any():
                seam_p.append(stp[true_seam].cpu()); seam_g.append(stg[true_seam].cpu())
                seam_break += int((stp[true_seam] > 30.0).sum()); seam_tot += int(true_seam.sum())
            if same.any():
                grad_res.append((stp[same] - stg[same]).abs().cpu())
        # band: within 3px of a GT part boundary
        pb = torch.zeros_like(fg)
        for dy, dx in ((0, 1), (1, 0)):
            sh = lambda A_: torch.roll(torch.roll(A_, dy, -2), dx, -1)
            pb |= (part != sh(part)) & fg & sh(fg)
        pbf = F.max_pool2d(pb.float()[:, None], 7, 1, 3)[:, 0] > 0
        band_in.append(e[pbf & fg].cpu()); band_out.append(e[~pbf & fg].cpu())
        n += B
        if n >= a.n:
            break
    def pc(t, q): return float(np.percentile(torch.cat(t).numpy(), q))
    name = os.path.basename(os.path.dirname(ckpt))
    print(f"\n===== {name} =====")
    print(f"surf percentiles  p50={pc(allsurf,50):6.2f}  p75={pc(allsurf,75):6.2f}  "
          f"p90={pc(allsurf,90):7.2f}  p99={pc(allsurf,99):8.2f} mm")
    print("per-region surf p50/p90 (mm):")
    for k in REG:
        if S[k]:
            print(f"   {k:6s} {pc(S[k],50):7.2f} / {pc(S[k],90):8.2f}")
    sp = torch.cat(seam_p); sg = torch.cat(seam_g)
    print(f"TRUE-SEAM edges (GT step <10mm): pred step p50={float(sp.median()):.2f}mm "
          f"(GT {float(sg.median()):.2f}mm)  | seam-break rate (>30mm): {100*seam_break/max(seam_tot,1):.2f}%")
    gr = torch.cat(grad_res)
    print(f"interior gradient residual | |dXp|-|dXg| |  p50={float(gr.median()):.3f}  "
          f"p90={pc([gr],90):.3f} mm/px")
    bi, bo = torch.cat(band_in), torch.cat(band_out)
    print(f"surf near boundary (<=3px) p50={float(bi.median()):.2f}mm  vs interior p50={float(bo.median()):.2f}mm")


for ck in a.ckpt:
    run(ck)
print("DETAIL DONE")
