"""Method overview figure: image -> backbone -> heads -> per-pixel (part, u, v, R).

Laid out on a real gridspec (no hand-computed coordinates) so nothing can collide.
Row 1: pipeline.  Row 2: what the frame buys, as three rendered outputs.
"""
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

B = os.path.expanduser("~/data/dp_viz_compare/beauty")
RA = f"{B}/report_assets"
OUT = os.path.expanduser("~/codes/wan/paper/figs")
CLIP = "15731089"
NPZ = f"{B}/{CLIP}_cscC/f_0075.npz"

plt.rcParams.update({"font.size": 7.2, "font.family": "DejaVu Sans",
                     "figure.dpi": 300, "savefig.dpi": 300})
INK, MUT = "#1b1b22", "#6c6c78"
OURS, WARM = "#6a3fc0", "#b8791f"


def rgb(p):
    im = cv2.imread(p)
    return None if im is None else im[..., ::-1]


def crop(im, m, pad=0.04):
    ys, xs = np.where(m)
    H, W = im.shape[:2]
    sy, sx = H / m.shape[0], W / m.shape[1]
    y0, y1, x0, x1 = int(ys.min()*sy), int(ys.max()*sy), int(xs.min()*sx), int(xs.max()*sx)
    py, px = int((y1-y0)*pad), int((x1-x0)*pad)
    return im[max(0, y0-py):min(H, y1+py), max(0, x0-px):min(W, x1+px)]


d = np.load(NPZ)
mask = d["mask"].astype(bool)
part, uv, n_cam = d["part"], d["uv"], d["n_cam"].astype(np.float32)

photo = crop(rgb(f"{RA}/{CLIP}_f_0075_photo.jpg"), mask)
hsv = np.zeros((*mask.shape, 3), np.uint8)
hsv[..., 0] = np.clip(part * 7.2, 0, 179).astype(np.uint8)
hsv[..., 1] = np.where(mask, 205, 0)
hsv[..., 2] = np.where(mask, 60 + 190 * uv[0], 0).astype(np.uint8)
iuv = crop(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), mask)
v = n_cam / np.clip(np.linalg.norm(n_cam, axis=-1, keepdims=True), 1e-6, None)
nrm = np.zeros((*mask.shape, 3), np.uint8)
nrm[mask] = np.clip((np.stack([v[..., 0], -v[..., 1], -v[..., 2]], -1)[mask]*.5+.5)*255, 0, 255)
nrm = crop(nrm, mask)
satin = crop(rgb(f"{RA}/{CLIP}_satin_0075.jpg"), mask)
objects = crop(rgb(f"{RA}/{CLIP}_objects_0075.jpg"), mask)

fig = plt.figure(figsize=(7.0, 2.15))
gs = fig.add_gridspec(1, 9, width_ratios=[1.05, .34, 1.25, .34, 1.15, .40, 1.05, 1.05, 1.05],
                      wspace=0.06, left=.005, right=.995, top=.86, bottom=.10)


def show(col, im, caption, capcolor=INK):
    ax = fig.add_subplot(gs[0, col]); ax.axis("off")
    if im is not None:
        ax.imshow(im)
    ax.set_title(caption, fontsize=6.9, color=capcolor, pad=3)
    return ax


def link(col, txt=None):
    ax = fig.add_subplot(gs[0, col]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.annotate("", xy=(.92, .5), xytext=(.08, .5),
                arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.0, mutation_scale=8))
    if txt:
        ax.text(.5, .60, txt, ha="center", fontsize=6.2, color=MUT)


show(0, photo, "input crop")
link(1)

# backbone + heads column
ax = fig.add_subplot(gs[0, 2]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.add_patch(FancyBboxPatch((.06, .30), .88, .42, boxstyle="round,pad=0.02,rounding_size=0.03",
                            fc="#efeaf9", ec=OURS, lw=1.0))
ax.text(.5, .51, "DINOv3 ViT-L/16\n+ DPT decoder", ha="center", va="center",
        fontsize=7.0, color=INK)
ax.set_title("backbone", fontsize=6.9, color=INK, pad=3)
link(3)

ax = fig.add_subplot(gs[0, 4]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
for i, name in enumerate(["mask $1$", "part $25$", "UV $24{\\times}2$", "rot $24{\\times}6$"]):
    y = .74 - i * .155
    ax.add_patch(FancyBboxPatch((.04, y), .92, .12,
                                boxstyle="round,pad=0.012,rounding_size=0.025",
                                fc="#ffffff", ec="#b9b3cc", lw=.75))
    ax.text(.5, y + .06, name, ha="center", va="center", fontsize=6.7, color=INK)
ax.set_title("prediction heads", fontsize=6.9, color=INK, pad=3)

ax = fig.add_subplot(gs[0, 5]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.annotate("", xy=(.92, .5), xytext=(.08, .5),
            arrowprops=dict(arrowstyle="-|>", color=MUT, lw=1.0, mutation_scale=8))
ax.text(.5, .60, "per\npixel", ha="center", fontsize=6.0, color=MUT, linespacing=1.0)

show(6, iuv, "$i,\\,u,\\,v$")
show(7, nrm, "$\\mathbf{R}\\,\\mathbf{n}_{\\mathrm{tpl}}$")
show(8, satin, "anisotropic shading")

fig.text(.505, .022,
         "per pixel: body part $i$, chart coordinates $(u,v)$, and a full frame "
         "$\\mathbf{R}\\!\\in\\!SO(3)$   —   supervised with the CSC loss "
         "$\\mathcal{T}(i,u,v)\\!\\rightarrow\\!\\mathbf{X}\\!\\in\\!\\mathbb{R}^{3}$",
         ha="center", fontsize=6.7, color=MUT)

fig.savefig(f"{OUT}/fig_method.pdf", bbox_inches="tight", pad_inches=0.015)
fig.savefig(f"{OUT}/fig_method.png", bbox_inches="tight", pad_inches=0.015)
print("wrote fig_method")
