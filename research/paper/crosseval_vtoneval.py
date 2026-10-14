"""Score OUR garment-transfer arms with LEFFA's metric definitions (VtonEval), so our numbers
land on the same axis as Leffa's published DeepFashion results.

Leffa's VtonEval and MCLD's evaluate.py differ:
  LPIPS backbone   squeeze (torchmetrics)      vs  alex (lpips pkg)
  SSIM / PSNR      torchmetrics                vs  MCLD's own implementation
so their 0.114 and our 0.138 are NOT comparable as published. This recomputes ours their way.
We reimplement the metric definitions verbatim from VtonEval/eval.py rather than calling its
CLI, because its filename->id matcher assumes VITON-style names and collides on DeepFashion pairs.

  python crosseval_vtoneval.py --res 256
"""
import argparse
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from tqdm import tqdm

H = os.path.expanduser("~/data/mcld/")
ap = argparse.ArgumentParser()
ap.add_argument("--res", default="256", choices=["256", "512"])
ap.add_argument("--arms", nargs="+", default=["origA", "origB"])
ap.add_argument("--gt", default=H + "released_results/unpacked/gt/")
ap.add_argument("--batch", type=int, default=16)
a = ap.parse_args()


class Paired(Dataset):
    """VtonEval's transform: PIL -> ToTensor, both already at the target resolution."""

    def __init__(self, gt_dir, pred_dir, names):
        self.gt_dir, self.pred_dir, self.names = gt_dir, pred_dir, names
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        n = self.names[i]
        gt = Image.open(os.path.join(self.gt_dir, n)).convert("RGB")
        pr = Image.open(os.path.join(self.pred_dir, n)).convert("RGB")
        if pr.size != gt.size:                      # VtonEval resizes GT to the pred's size
            gt = gt.resize(pr.size, Image.LANCZOS)
        return self.to_tensor(gt), self.to_tensor(pr)


@torch.no_grad()
def score(dl, n_total):
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to("cuda")
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to("cuda")
    p = s = l = 0.0
    for gt, pr in tqdm(dl, leave=False):
        b = gt.size(0)
        gt, pr = gt.cuda(), pr.cuda()
        p += psnr_m(pr, gt).item() * b
        s += ssim_m(pr, gt).item() * b
        l += lpips_m((gt * 2) - 1, (pr * 2) - 1).item() * b       # VtonEval passes (gt, pred)
    return l / n_total, s / n_total, p / n_total


gt_dir = a.gt + a.res
print(f"resolution {a.res}, GT {gt_dir}")
print(f"{'arm':10} {'LPIPS(squeeze)':>15} {'SSIM(tm)':>10} {'PSNR(tm)':>10} {'n':>7}")
for arm in a.arms:
    pred_dir = f"{H}results_{arm}/mcld_{arm}/{a.res}"
    if not os.path.isdir(pred_dir):
        print(f"{arm:10} MISSING {pred_dir}")
        continue
    names = sorted(set(os.listdir(pred_dir)) & set(os.listdir(gt_dir)))
    dl = DataLoader(Paired(gt_dir, pred_dir, names), batch_size=a.batch, num_workers=4)
    lp, ss, ps = score(dl, len(names))
    print(f"{arm:10} {lp:15.4f} {ss:10.4f} {ps:10.3f} {len(names):7d}", flush=True)
print("\nLeffa published @256: LPIPS 0.114  SSIM 0.732 | @512: LPIPS 0.119  SSIM 0.755")
