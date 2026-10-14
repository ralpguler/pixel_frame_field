"""Compose paper Fig. 1 (pixel frame field as boxes) from orient/blender_boxes.py renders.
Row: (a) input | (b) pixel frame field on the image | (c) the same boxes on the canonical template | (d) zoom insets.
  python -m orient.fig1_compose DIR OUT.png [--inset x0 y0 x1 y1 ...]  (inset boxes in render_image pixel coords)
"""
import argparse, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("out")
ap.add_argument("--inset", type=float, nargs=4, action="append", default=None,
                help="x0 y0 x1 y1 as FRACTIONS of the image-render panel (repeatable)")
ap.add_argument("--height", type=int, default=1200)
ap.add_argument("--inset-img", action="append", default=None, help="pre-rendered zoom image per --inset (same order); else crop")
ap.add_argument("--tscale", type=float, default=0.92, help="template panel height relative to the image panels")
ap.add_argument("--link", type=float, nargs=2, action="append", default=None,
                help="x y (fractions of the image-render panel): number the nearest box here and in the template panel")
ap.add_argument("--glyph", default="box", help="box | cone_n | cone_t (selects render_*_<glyph>.png)")
a = ap.parse_args(); D = os.path.expanduser(a.dir); meta = json.load(open(f"{D}/meta.json"))
W, H = meta["W"], meta["H"]; x0, y0, x1, y1 = meta["person_bbox"]; mg = 0.04 * H
bx = (max(0, x0 - mg), max(0, y0 - mg), min(W, x1 + mg), min(H, y1 + mg))       # same border as the render
photo = np.asarray(Image.open(f"{D}/bg.png").convert("RGB").crop(tuple(int(v) for v in bx)))
suf = "" if a.glyph == "box" else f"_{a.glyph}"
rimg = np.asarray(Image.open(f"{D}/render_image{suf}.png").convert("RGB"))
t = Image.open(f"{D}/render_template{suf}.png").convert("RGBA"); al = np.asarray(t)[..., 3]
ys, xs = np.nonzero(al > 8); pad = 20
tx0, ty0 = max(0, xs.min() - pad), max(0, ys.min() - pad)
t = t.crop((tx0, ty0, xs.max() + pad, ys.max() + pad))
wbg = Image.new("RGBA", t.size, (255, 255, 255, 255)); wbg.alpha_composite(t); timg = np.asarray(wbg.convert("RGB"))
insets = a.inset or [[0.30, 0.27, 0.62, 0.45], [0.10, 0.36, 0.34, 0.54]]          # chest/neck, near forearm
Hh = a.height
rs = lambda im, h: np.asarray(Image.fromarray(im).resize((max(1, int(im.shape[1] * h / im.shape[0])), h), Image.LANCZOS))
pa, pb, pc = rs(photo, Hh), rs(rimg, Hh), rs(timg, int(Hh * a.tscale))
ins = []
for k, f in enumerate(insets):
    if a.inset_img and k < len(a.inset_img):                  # a finer, higher-resolution render of the same window
        ins.append(rs(np.asarray(Image.open(os.path.expanduser(a.inset_img[k])).convert("RGB")), Hh // len(insets) - 10)); continue
    X0, Y0, X1, Y1 = (int(f[0] * rimg.shape[1]), int(f[1] * rimg.shape[0]), int(f[2] * rimg.shape[1]), int(f[3] * rimg.shape[0]))
    ins.append(rs(rimg[Y0:Y1, X0:X1], Hh // len(insets) - 10))
wd = max(i.shape[1] for i in ins); gap = 24
Wt = pa.shape[1] + pb.shape[1] + pc.shape[1] + wd + 3 * gap; Ht = Hh + 90
fig = plt.figure(figsize=(Wt / 100, Ht / 100), dpi=100)
def put(im, x, y, title=None):
    ax = fig.add_axes([x / Wt, (Ht - y - im.shape[0]) / Ht, im.shape[1] / Wt, im.shape[0] / Ht]); ax.imshow(im); ax.axis("off")
    if title:                                                 # all captions on one baseline under the image row
        fig.text((x + im.shape[1] / 2) / Wt, (Ht - Hh - 18) / Ht, title, ha="center", va="top", fontsize=19)
    return ax
x = 0; put(pa, x, 0, "(a) input"); x += pa.shape[1] + gap
axb = put(pb, x, 0, "(b) pixel frame field")
for k, f in enumerate(insets):
    axb.add_patch(Rectangle((f[0] * pb.shape[1], f[1] * pb.shape[0]), (f[2] - f[0]) * pb.shape[1], (f[3] - f[1]) * pb.shape[0],
                            fill=False, ec="white", lw=2.2))
x += pb.shape[1] + gap
axc = put(pc, x, (Hh - pc.shape[0]) // 2, "(c) on the template"); x += pc.shape[1] + gap
yy = 0
for k, im in enumerate(ins):
    put(im, x, yy, "(d) zoom" if k == len(ins) - 1 else None); yy += im.shape[0] + 10
# numbered links: the same box (same index) in the image panel and in the template panel
si, stp = f"{D}/screen_image{suf}.json", f"{D}/screen_template{suf}.json"
if os.path.exists(si) and os.path.exists(stp):
    XI = np.array(json.load(open(si))["xy"]); XT = np.array(json.load(open(stp))["xy"])
    kb, kc = pb.shape[0] / rimg.shape[0], pc.shape[0] / timg.shape[0]
    links = a.link or [[0.12, 0.45], [0.47, 0.33], [0.80, 0.55]]
    for k, (fx, fy) in enumerate(links):
        i = int(np.argmin(((XI - [fx * rimg.shape[1], fy * rimg.shape[0]]) ** 2).sum(1)))
        for ax, (px, py) in ((axb, XI[i] * kb), (axc, (XT[i] - [tx0, ty0]) * kc)):
            ax.add_patch(plt.Circle((px, py), 0.022 * Hh, fill=False, ec="white", lw=3.2))
            ax.add_patch(plt.Circle((px, py), 0.022 * Hh, fill=False, ec="black", lw=1.4))
            ax.text(px + 0.03 * Hh, py - 0.03 * Hh, str(k + 1), color="white", fontsize=20, fontweight="bold", ha="left", va="bottom",
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["x"]).withStroke(linewidth=3, foreground="black")])
fig.savefig(a.out, dpi=100, facecolor="white"); print("saved", a.out)
