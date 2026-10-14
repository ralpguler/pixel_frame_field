"""Probe undertraining: in-domain held-out BEDLAM dense metrics (part-acc, uv-L1, rotation geo) across
checkpoints. If they keep improving at the last ckpt -> longer training helps (the COCO sparse val is
sim2real-capped and can't show this)."""
import os, sys, glob, argparse, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import torch.nn.functional as F
from data import build_loader
from model import DINOv3DPT, sixd_to_R, gather_rot6d, N_PARTS, backbone_from_state_dict, default_layers

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", default=[24000, 48000, 72000])
ap.add_argument("--run", default="dp_scratch_bedlam_orient")
ap.add_argument("--batches", type=int, default=40)
ap.add_argument("--shards", default=None, help="glob for an explicit shard set (e.g. MAMMA), else BEDLAM val")
ap.add_argument("--nshards", type=int, default=40, help="random subset size when --shards is given")
a = ap.parse_args()
dev = "cuda"
RUN = os.path.expanduser(f"~/data/bedlam2/dp_runs/{a.run}")


@torch.no_grad()
def evalc(ckpt):
    sd = torch.load(ckpt, map_location="cpu")["model"]
    rpp = sd["rot_head.2.weight"].shape[0] != 6                        # auto-detect per-part vs single
    m = DINOv3DPT(backbone=backbone_from_state_dict(sd), layers=default_layers(backbone_from_state_dict(sd)),
                  pretrained=False, decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
    m.load_state_dict(sd)
    if a.shards:
        allsh = sorted(glob.glob(os.path.expanduser(a.shards)))
        sub = list(np.random.RandomState(0).permutation(allsh)[:a.nshards])
        ld = build_loader("val", batch=16, workers=2, input_size=512, train=False, orient=True, shards=sub)
    else:
        ld = build_loader("val", batch=16, workers=2, input_size=512, train=False, orient=True)
    c = np.zeros(2); uv = np.zeros(2); geo = np.zeros(2); geo_e = np.zeros(2)   # (sum, n); geo_e = end-to-end
    for i, b in enumerate(ld):
        if i >= a.batches:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(b["img"].to(dev))
        for bi, g in enumerate(b["gt"]):
            part, uvg, fg = g[0].to(dev), g[1].to(dev), g[2].to(dev)
            H, W = part.shape
            if not fg.any():
                continue
            p = F.interpolate(out["part"][bi:bi + 1].float(), (H, W), mode="bilinear", align_corners=False)[0]
            pred = p.argmax(0) + 1
            c[0] += (pred[fg] == part[fg]).sum().item(); c[1] += fg.sum().item()
            u = F.interpolate(out["uv"][bi:bi + 1].float(), (H, W), mode="bilinear", align_corners=False)[0].view(N_PARTS, 2, H, W)
            idx = (part - 1).clamp(0); sel = u.gather(0, idx[None, None].expand(1, 2, H, W)).squeeze(0)
            uv[0] += (sel.permute(1, 2, 0)[fg] - uvg.permute(1, 2, 0)[fg]).abs().sum().item(); uv[1] += fg.sum().item() * 2
            rg = g[3].to(dev).float(); rfg = g[4].to(dev)
            if rfg.any():
                Rh, Rw = rfg.shape
                r_pp = F.interpolate(out["rot6d"][bi:bi + 1].float(), (Rh, Rw), mode="bilinear", align_corners=False)
                Rg = sixd_to_R(rg[rfg])

                def geo_of(part_map):                          # gather rot by given part map, geo vs GT
                    pm = F.interpolate(part_map[None, None].float(), (Rh, Rw), mode="nearest")[0, 0].long()
                    Rp = sixd_to_R(gather_rot6d(r_pp, pm[None])[0].permute(1, 2, 0)[rfg])
                    tr = torch.matmul(Rp.transpose(1, 2), Rg).diagonal(dim1=1, dim2=2).sum(1)
                    return (torch.arccos(((tr - 1) / 2).clamp(-1, 1)).sum() * 180 / np.pi).item()

                geo[0] += geo_of(part); geo[1] += rfg.sum().item()             # ORACLE: gather by GT part
                pred_full = F.interpolate(out["part"][bi:bi + 1].float(), (H, W),
                                          mode="bilinear", align_corners=False)[0].argmax(0) + 1
                geo_e[0] += geo_of(pred_full); geo_e[1] += rfg.sum().item()    # END-TO-END: gather by pred part
    return c[0] / c[1], uv[0] / uv[1], geo[0] / geo[1], geo_e[0] / geo_e[1]


src = f"shards={a.shards} (n={a.nshards})" if a.shards else "in-domain BEDLAM val"
print(f"{src} ({a.batches} batches) — run {a.run}")
print(f"{'ckpt':>8} {'part_acc':>9} {'uv_L1':>8} {'geo_oracle':>11} {'geo_e2e':>9}")
for s in a.ckpts:
    ck = f"{RUN}/ckpt_{int(s):06d}.pt"
    if not os.path.exists(ck):
        print(f"{s:>8}  (missing)"); continue
    pa, uvl, gd, gde = evalc(ck)
    print(f"{int(s):>8} {pa:>9.4f} {uvl:>8.4f} {gd:>11.2f} {gde:>9.2f}", flush=True)
