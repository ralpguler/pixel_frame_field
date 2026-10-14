"""Qualitative 3-way: Leffa vs our two arms, rows chosen BY per-image LPIPS (not by eye)."""
import os, cv2, numpy as np, torch
from PIL import Image
from torchvision import transforms
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

H = os.path.expanduser("~/data/mcld/")
GT = H + "released_results/unpacked/gt/256"
D = {"leffa": H+"results_leffa/256", "armA": H+"results_origA/mcld_origA/256",
     "armB": H+"results_origB/mcld_origB/256"}
names = sorted(set(os.listdir(GT)) & set.intersection(*[set(os.listdir(p)) for p in D.values()]))
print(len(names), "common")
tt = transforms.ToTensor()
lp = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").cuda()
sc = {k: [] for k in D}
with torch.no_grad():
    for i in range(0, len(names), 32):
        b = names[i:i+32]
        g = torch.stack([tt(Image.open(f"{GT}/{n}").convert("RGB")) for n in b]).cuda()*2-1
        for k, p in D.items():
            x = torch.stack([tt(Image.open(f"{p}/{n}").convert("RGB")) for n in b]).cuda()*2-1
            sc[k] += [lp(g[j:j+1], x[j:j+1]).item() for j in range(len(b))]
sc = {k: np.array(v) for k, v in sc.items()}
np.savez(H+"leffa_perimage.npz", names=names, **sc)
adv = sc["leffa"] - sc["armB"]                       # >0 : ours better than Leffa
order = np.argsort(-adv)
picks = [("largest margin over Leffa", order[0]), ("2nd", order[1]),
         ("median case", order[len(order)//2]), ("Leffa wins most", order[-1])]
O = H+"fashion_orig/test/"
def ld(p, sz=(176,256)):
    im = cv2.imread(p); return cv2.resize(im, sz) if im is not None else np.zeros((sz[1],sz[0],3),np.uint8)
rows = []
for lab, i in picks:
    n = names[i]; src, tgt = os.path.splitext(n)[0].split("_2_")
    rows.append(np.concatenate([ld(O+src+".jpg"), ld(D["leffa"]+"/"+n), ld(D["armA"]+"/"+n),
                                ld(D["armB"]+"/"+n), ld(O+tgt+".jpg")], 1))
    print(f"{lab:28} leffa {sc['leffa'][i]:.3f} armA {sc['armA'][i]:.3f} armB {sc['armB'][i]:.3f}")
sheet = np.concatenate([np.pad(r, ((0,4),(0,0),(0,0)), constant_values=255) for r in rows], 0)
cv2.imwrite(os.path.expanduser("~/codes/wan/paper/figs/fig_leffa.png"), sheet)
print("wrote figs/fig_leffa.png", sheet.shape)
print("summary: ours better on", f"{100*(adv>0).mean():.1f}% of images")
