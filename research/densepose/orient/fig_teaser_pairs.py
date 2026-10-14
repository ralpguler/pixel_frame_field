"""Teaser: the SAME frames on the canonical template and on the image, with matched zoom pairs.
  python -m orient.fig_teaser_pairs FULL_DIR OUT --zoom DIR NAME [--zoom DIR NAME ...] [--glyph tripod] [--width 3400]
Row 1: full template frames | full predicted pixel frame field.  Row 2: the zoom pairs (template | image) side by side.
Both rows are justified to the same width. Colour-coded rectangles mark every zoom region on both full panels (template
rectangle = bbox of the FULL export's glyphs inside the zoom's ROI and parts, via screen_template_<glyph>.json).
FULL_DIR / zoom DIRs = orient/framefield_boxes.py exports rendered by orient/blender_boxes.py (zoom template: --frame glyphs)."""
import argparse, json, os, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

ap = argparse.ArgumentParser(); ap.add_argument("full"); ap.add_argument("out")
ap.add_argument("--zoom", nargs=2, action="append", metavar=("DIR", "NAME"), required=True)
ap.add_argument("--glyph", default="tripod"); ap.add_argument("--width", type=int, default=3400)
ap.add_argument("--bg", default="#ffffff")
a = ap.parse_args(); g = a.glyph; COLS = ["#f08c00", "#7c3aed", "#e8559a", "#6b7280"]   # avoid the axis red/green/blue
BG = tuple(int(a.bg[i:i + 2], 16) for i in (1, 3, 5)) + (255,)

def dark(path, pad=20):
    t = Image.open(path).convert("RGBA"); al = np.asarray(t)[..., 3]; ys, xs = np.nonzero(al > 8)
    box = (max(0, xs.min() - pad), max(0, ys.min() - pad), min(t.width, xs.max() + pad), min(t.height, ys.max() + pad))
    t = t.crop(box); bg = Image.new("RGBA", t.size, BG); bg.alpha_composite(t)
    return np.asarray(bg.convert("RGB")), box

def rs(im, h):
    return np.asarray(Image.fromarray(im).resize((max(1, int(round(im.shape[1] * h / im.shape[0]))), max(1, int(round(h)))), Image.LANCZOS))

F = os.path.expanduser(a.full); meta = json.load(open(f"{F}/meta.json"))
timg, tbox = dark(f"{F}/render_template_{g}.png")
iimg = np.asarray(Image.open(f"{F}/render_image_{g}.png").convert("RGB"))
XT = np.array(json.load(open(f"{F}/screen_template_{g}.json"))["xy"])
rows = np.loadtxt(f"{F}/img_boxes.csv", delimiter=",", skiprows=1)
W, H = meta["W"], meta["H"]; px = rows[:, 0] * W - 0.5; py = (1 - rows[:, 1]) * H - 0.5
partmap = np.load(f"{meta['dump']}/f_{meta['frame']:04d}.npz")["part"].astype(int)
spart = partmap[np.clip(np.round(py).astype(int), 0, H - 1), np.clip(np.round(px).astype(int), 0, W - 1)]
x0, y0, x1, y1 = meta["person_bbox"]; mg = 0.04 * H
v = meta.get("valid") or [0, 0, W, H]
bx0, by0, bx1 = max(0, x0 - mg, v[0]), max(0, y0 - mg, v[1]), min(W, x1 + mg, v[2])
ks = iimg.shape[1] / (bx1 - bx0)                                   # image-render px per crop px

zooms = []
for k, (D, name) in enumerate(a.zoom):
    D = os.path.expanduser(D); zm = json.load(open(f"{D}/meta.json")); r = zm["roi"]; parts = zm.get("parts")
    sel = (px >= r[0]) & (px < r[2]) & (py >= r[1]) & (py < r[3])
    if parts: sel &= np.isin(spart, parts)
    zooms.append(dict(name=name, col=COLS[k % len(COLS)], roi=r, sel=sel,
                      zt=dark(f"{D}/render_template_{g}.png")[0], zi=np.asarray(Image.open(f"{D}/render_image_{g}.png").convert("RGB"))))

G1, GP, GZ, LAB = 30, 10, 40, 64                                    # gaps: row-1 panels, within a pair, between pairs; label band
# row 1 at a common height, then scaled to the target width
h1 = 1000.0; T1, I1 = rs(timg, h1), rs(iimg, h1)
s1 = (a.width - G1) / (T1.shape[1] + I1.shape[1]); h1 *= s1; T1, I1 = rs(timg, h1), rs(iimg, h1)
# row 2: every zoom panel at a common height, scaled so the row fills the same width
h2 = 600.0; Z = [(rs(z["zt"], h2), rs(z["zi"], h2)) for z in zooms]
w2 = sum(t.shape[1] + GP + i.shape[1] for t, i in Z) + GZ * (len(Z) - 1)
s2 = (a.width - GP * len(Z) - GZ * (len(Z) - 1)) / (w2 - GP * len(Z) - GZ * (len(Z) - 1)); h2 = min(h2 * s2, 0.62 * T1.shape[0])                          # tall/narrow zooms: cap the row height, centre it
Z = [(rs(z["zt"], h2), rs(z["zi"], h2)) for z in zooms]
x2 = max(0, (a.width - sum(t.shape[1] + GP + i.shape[1] for t, i in Z) - GZ * (len(Z) - 1)) // 2)
Wt = a.width; Ht = int(T1.shape[0] + LAB + Z[0][0].shape[0] + LAB)
fig = plt.figure(figsize=(Wt / 100, Ht / 100), dpi=100, facecolor="white")

def put(im, x, y, edge=None):
    ax = fig.add_axes([x / Wt, (Ht - y - im.shape[0]) / Ht, im.shape[1] / Wt, im.shape[0] / Ht]); ax.imshow(im)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(edge is not None)
        if edge: sp.set_edgecolor(edge); sp.set_linewidth(4)
    return ax

def label(x, w, y, text, **kw):
    fig.text((x + w / 2) / Wt, (Ht - y - 10) / Ht, text, ha="center", va="top", fontsize=kw.pop("fs", 26), **kw)

axT = put(T1, 0, 0); axI = put(I1, T1.shape[1] + G1, 0)
label(0, T1.shape[1], T1.shape[0], "canonical template frames")
label(T1.shape[1] + G1, I1.shape[1], T1.shape[0], "predicted pixel frame field")
kt, ki = T1.shape[0] / timg.shape[0], I1.shape[0] / iimg.shape[0]
y2 = T1.shape[0] + LAB; x = x2
for z, (zt, zi) in zip(zooms, Z):
    if z["sel"].any():
        p = (XT[z["sel"]] - [tbox[0], tbox[1]]) * kt; pd = 14
        axT.add_patch(Rectangle((p[:, 0].min() - pd, p[:, 1].min() - pd), np.ptp(p[:, 0]) + 2 * pd, np.ptp(p[:, 1]) + 2 * pd, fill=False, ec=z["col"], lw=3.5))
    r = z["roi"]; s = ks * ki
    rx0, ry0 = np.clip([(r[0] - bx0) * s, (r[1] - by0) * s], 2, [I1.shape[1] - 2, I1.shape[0] - 2])     # clipped to the panel
    rx1, ry1 = np.clip([(r[2] - bx0) * s, (r[3] - by0) * s], 2, [I1.shape[1] - 2, I1.shape[0] - 2])
    axI.add_patch(Rectangle((rx0, ry0), rx1 - rx0, ry1 - ry0, fill=False, ec=z["col"], lw=3.5))
    put(zt, x, y2, z["col"]); put(zi, x + zt.shape[1] + GP, y2, z["col"])
    label(x, zt.shape[1] + GP + zi.shape[1], y2 + zt.shape[0], z["name"], color=z["col"], fontweight="bold", fs=24)
    x += zt.shape[1] + GP + zi.shape[1] + GZ
fig.savefig(a.out, dpi=100, facecolor="white"); print("saved", a.out, (Wt, Ht))
