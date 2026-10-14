"""Rotation and surface accuracy on the synthetic validation set (fixed 512 x 512 ground-truth canvas).

Port of research/densepose/orient/eval_512.py. Every prediction is warped into the ground-truth canvas exactly as
in training, the rotation and the chart coordinates are read at the PREDICTED part (end to end), and it reports:
  geo_e2e     geodesic rotation error (deg)            surf_err  distance on the template (mm), via the CSC map
  part_acc    part accuracy                             hands+feet and head breakdowns
  jumps       spurious / missed rotation discontinuities at part boundaries (local, which a mean cannot see)
geo_e2e and surf_err are, as in the research script and the paper, the MEAN OVER BATCHES OF PER-BATCH MEDIANS;
the pooled mean and median over all pixels are printed too.

  python eval/bedlam_val.py --ckpt checkpoints/pff_dp_w_MF_3k.pt --shards '/data/dp_shards_512/val/shard-*.tar'
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pff.csc import CSC                                                      # noqa: E402
from pff.data.synthetic import build_loader                                   # noqa: E402
from pff.geometry import N_PARTS, gather_rot6d, sixd_to_R                     # noqa: E402
from pff.model import load_model                                              # noqa: E402

HANDS_FEET = (3, 4, 5, 6)
HEAD = (23, 24)


def geo(A, B):
    tr = torch.einsum("...ij,...ij->...", A, B)
    return torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shards", required=True, help="glob of fixed-canvas shards (the dp_shards_512 val split)")
    ap.add_argument("--n", type=int, default=512, help="samples")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--uv-asset", default="smplx_face_uv_v2.npy")
    ap.add_argument("--csc-K", type=int, default=512)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    dev = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    m = load_model(a.ckpt, dev)
    csc = CSC(a.csc_K).to(dev)
    shards = sorted(glob.glob(os.path.expanduser(a.shards)))
    assert shards, a.shards
    # hidden_mask=False: the research evaluation scored every rendered pixel, eyeballs and mouth interior included
    ld = build_loader(shards, batch=a.batch, workers=a.workers, input_size=512, train=False, uv_asset=a.uv_asset,
                      hidden_mask=False)

    acc = {k: [] for k in ("geo", "surf", "pacc", "geo_hf", "surf_hf", "geo_hd")}
    pooled = {"geo": [], "surf": []}
    jump = {k: [0, 0.0, 0.0] for k in ("spurious", "missed", "real", "interior")}
    n = 0
    for b in ld:
        img = b["img"].to(dev)
        with torch.no_grad():
            if dev.startswith("cuda"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = m(img)
            else:
                out = m(img)
        part, fg = b["part"].to(dev), b["fg"].to(dev)
        theta = b["theta"].to(dev).float()
        rgt = b["rot6d"].to(dev).float(); Xg = b["xyz"].to(dev).float().permute(0, 3, 1, 2)
        B, G, _ = part.shape
        grid = F.affine_grid(theta, (B, 1, G, G), align_corners=False)
        warp = lambda x: F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border",   # noqa: E731
                                       align_corners=False)
        with torch.no_grad():
            pl = warp(out["part"]); uvp = warp(out["uv"]).view(B, N_PARTS, 2, G, G); rp = warp(out["rot6d"])
            pid = pl.argmax(1) + 1
            Rg = sixd_to_R(rgt)
            Rp = sixd_to_R(gather_rot6d(rp, pid).permute(0, 2, 3, 1))
            e = geo(Rp, Rg)
            sel = uvp.gather(1, (pid - 1)[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
            Xp = torch.stack([csc(sel[i], pid[i]) for i in range(B)])
            s = (Xp - Xg).norm(dim=1) * 1000
        hf = torch.isin(part, torch.tensor(HANDS_FEET, device=dev)) & fg
        hd = torch.isin(part, torch.tensor(HEAD, device=dev)) & fg
        if fg.any():
            acc["geo"].append(float(e[fg].median())); acc["surf"].append(float(s[fg].median()))
            acc["pacc"].append(float((pid[fg] == part[fg]).float().mean()))
            pooled["geo"].append(e[fg].cpu().numpy()); pooled["surf"].append(s[fg].cpu().numpy())
        if hf.any():
            acc["geo_hf"].append(float(e[hf].median())); acc["surf_hf"].append(float(s[hf].median()))
        if hd.any():
            acc["geo_hd"].append(float(e[hd].median()))
        for dy, dx in ((0, 1), (1, 0)):                                    # is each discontinuity in the right place?
            sh = lambda A: torch.roll(torch.roll(A, dy, 1), dx, 2)           # noqa: E731
            m2 = fg & sh(fg)
            gflip, aflip = part != sh(part), pid != sh(pid)
            stp, stg = geo(Rp, sh(Rp)), geo(Rg, sh(Rg))
            for nm, msk in (("spurious", m2 & aflip & ~gflip), ("missed", m2 & ~aflip & gflip),
                            ("real", m2 & aflip & gflip), ("interior", m2 & ~aflip & ~gflip)):
                if msk.any():
                    jump[nm][0] += int(msk.sum()); jump[nm][1] += float(stp[msk].sum()); jump[nm][2] += float(stg[msk].sum())
        n += B
        if n >= a.n:
            break

    pg, ps = np.concatenate(pooled["geo"]), np.concatenate(pooled["surf"])
    print(f"\n{a.ckpt}   ({n} samples)\n")
    print("AGGREGATE (mean over batches of per-batch medians, as reported):")
    print(f"   geo_e2e        {np.mean(acc['geo']):6.2f} deg      part_acc {100 * np.mean(acc['pacc']):5.1f}%")
    print(f"   surf_err       {np.mean(acc['surf']):6.2f} mm")
    print(f"   hands+feet     geo {np.mean(acc['geo_hf']):6.2f} deg   surf {np.mean(acc['surf_hf']):6.2f} mm")
    print(f"   head           geo {np.mean(acc['geo_hd']):6.2f} deg")
    print(f"POOLED over all pixels: geo mean {pg.mean():.2f} / median {np.median(pg):.2f} deg, "
          f"surf mean {ps.mean():.2f} / median {np.median(ps):.2f} mm")
    tot = sum(v[0] for v in jump.values())
    print("\nLOCAL: is the rotation discontinuity in the right place?")
    print(f"   {'edge type':12s} {'share':>7s} {'pred step':>10s} {'GT step':>9s} {'excess':>9s}")
    for nm, v in jump.items():
        if v[0]:
            p_, g_ = v[1] / v[0], v[2] / v[0]
            print(f"   {nm:12s} {100 * v[0] / tot:6.2f}% {p_:9.2f}d {g_:8.2f}d {p_ - g_:+8.2f}d")
    sp, ms = jump["spurious"][0] / max(tot, 1), jump["missed"][0] / max(tot, 1)
    print(f"\n   misplaced-discontinuity rate: {100 * (sp + ms):.2f}% of edges "
          f"(spurious {100 * sp:.2f}% + missed {100 * ms:.2f}%)")


if __name__ == "__main__":
    main()
