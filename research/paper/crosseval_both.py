"""Score any set of generated images with BOTH metric harnesses on the SAME image subset:
  - VtonEval / Leffa axis : torchmetrics LPIPS(squeeze), SSIM, PSNR
  - MCLD axis             : lpips(alex), MCLD's SSIM_256 / PSNR

Used to put Leffa's own outputs and our two arms on a common footing over identical pairs.

  python crosseval_both.py --dirs leffa:$H/results_leffa origA:... origB:... --res 256
"""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

H = os.path.expanduser("~/data/mcld/")
ap = argparse.ArgumentParser()
ap.add_argument("--dirs", nargs="+", required=True,   # name:path (path holds 256/ and 512/)
                help="e.g. leffa:~/data/mcld/results_leffa origA:~/data/mcld/results_origA/mcld_origA")
ap.add_argument("--res", default="256")
ap.add_argument("--gt", default=H + "released_results/unpacked/gt/")
ap.add_argument("--mcld_repo", default=os.path.expanduser("~/codes/wan/mcld_orig"))
a = ap.parse_args()

entries = []
for spec in a.dirs:
    name, path = spec.split(":", 1)
    entries.append((name, os.path.expanduser(path).rstrip("/") + "/" + a.res))

gt_dir = a.gt + a.res
# common subset across ALL dirs -> a like-for-like comparison
common = set(os.listdir(gt_dir))
for _, p in entries:
    common &= set(os.listdir(p)) if os.path.isdir(p) else set()
names = sorted(common)
print(f"res {a.res}: {len(names)} images common to {[n for n, _ in entries]}", flush=True)
if not names:
    sys.exit("no common images")

to_t = transforms.ToTensor()


def load_pair(gt_p, pr_p):
    gt = Image.open(gt_p).convert("RGB")
    pr = Image.open(pr_p).convert("RGB")
    if pr.size != gt.size:
        gt = gt.resize(pr.size, Image.LANCZOS)
    return to_t(gt), to_t(pr)


# ---------------- axis 1: VtonEval (Leffa) ----------------
@torch.no_grad()
def vtoneval(pdir):
    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").cuda()
    L = S = P = 0.0
    for i in range(0, len(names), 16):
        b = names[i:i+16]
        g, p = zip(*[load_pair(gt_dir + "/" + n, pdir + "/" + n) for n in b])
        g, p = torch.stack(g).cuda(), torch.stack(p).cuda()
        L += lp((g*2)-1, (p*2)-1).item() * len(b)
        S += ssim(p, g).item() * len(b)
        P += psnr(p, g).item() * len(b)
    return L/len(names), S/len(names), P/len(names)


# ---------------- axis 2: MCLD ----------------
sys.path.insert(0, a.mcld_repo)
from validation.metrics import build_metric      # noqa: E402


@torch.no_grad()
def mcld(pdir, metric):
    L = S = P = 0.0
    for n in names:
        g, p = load_pair(gt_dir + "/" + n, pdir + "/" + n)
        _, l, ps, ss, s256 = metric(g.unsqueeze(0).cuda(), p.unsqueeze(0).cuda())
        L += l.item(); S += s256.item(); P += ps.item()
    return L/len(names), S/len(names), P/len(names)


metric = build_metric().cuda()
print(f"\n{'system':10} | {'VtonEval axis (squeeze)':>30} | {'MCLD axis (alex)':>30}")
print(f"{'':10} | {'LPIPS':>9} {'SSIM':>9} {'PSNR':>9} | {'LPIPS':>9} {'SSIM':>9} {'PSNR':>9}")
for name, pdir in entries:
    v = vtoneval(pdir)
    m = mcld(pdir, metric)
    print(f"{name:10} | {v[0]:9.4f} {v[1]:9.4f} {v[2]:9.3f} | {m[0]:9.4f} {m[1]:9.4f} {m[2]:9.3f}",
          flush=True)
print(f"\nn={len(names)} identical pairs for every row.")
print("Leffa published @256 LPIPS 0.114 SSIM 0.732 | @512 LPIPS 0.119 SSIM 0.755 (VtonEval axis)")
