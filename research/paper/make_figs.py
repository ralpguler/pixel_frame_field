"""Assemble paper-ready figures from the experiment assets.

Outputs to ~/codes/wan/paper/figs/ as PNG (LaTeX-ready via \\includegraphics).
"""
import os
import subprocess

import cv2
import numpy as np

B = os.path.expanduser("~/data/dp_viz_compare/beauty")
RA = f"{B}/report_assets"
OUT = os.path.expanduser("~/codes/wan/paper/figs")
os.makedirs(OUT, exist_ok=True)
CLIP = "15731089"          # hero clip (dancer, full body, good motion)


def load(p, w=None):
    im = cv2.imread(p)
    if im is None:
        print(f"  MISSING {p}")
        return None
    if w:
        im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])), interpolation=cv2.INTER_AREA)
    return im


def crop_body(im, npz=None, pad=0.06):
    """Crop to the person using the field-dump mask when available."""
    if npz is not None and os.path.exists(npz):
        d = np.load(npz)
        m = d["mask"] if "mask" in d else None
        if m is not None and m.any():
            ys, xs = np.where(m)
            H, W = im.shape[:2]
            sy, sx = H / m.shape[0], W / m.shape[1]
            y0, y1 = int(ys.min() * sy), int(ys.max() * sy)
            x0, x1 = int(xs.min() * sx), int(xs.max() * sx)
            py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
            return im[max(0, y0 - py):min(H, y1 + py), max(0, x0 - px):min(W, x1 + px)]
    return im


def hstack(ims, h=520, gap=6, bg=255):
    out = []
    for im in ims:
        if im is None:
            continue
        s = h / im.shape[0]
        out.append(cv2.resize(im, (int(im.shape[1] * s), h), interpolation=cv2.INTER_AREA))
    if not out:
        return None
    strips = []
    for i, im in enumerate(out):
        strips.append(im)
        if i < len(out) - 1:
            strips.append(np.full((h, gap, 3), bg, np.uint8))
    return np.concatenate(strips, 1)


def normal_rgb(npz):
    """Camera-space normal map from the field dump: n_cam = R @ n_template."""
    d = np.load(npz)
    n = d["n_cam"].astype(np.float32)                     # (H,W,3)
    m = d["mask"].astype(bool)
    rgb = np.zeros((*m.shape, 3), np.uint8)
    v = n / np.clip(np.linalg.norm(n, axis=-1, keepdims=True), 1e-6, None)
    # OpenCV cam -> conventional: negate Y,Z so facing = blue, up = green
    col = np.stack([v[..., 0], -v[..., 1], -v[..., 2]], -1)
    rgb[m] = np.clip((col[m] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    return rgb[..., ::-1]                                  # to BGR for cv2


def iuv_rgb(npz):
    """DensePose IUV visual: hue = part, value = u."""
    d = np.load(npz)
    p, uv, m = d["part"], d["uv"], d["mask"].astype(bool)
    hsv = np.zeros((*m.shape, 3), np.uint8)
    hsv[..., 0] = np.clip(p * 7.2, 0, 179).astype(np.uint8)
    hsv[..., 1] = np.where(m, 205, 0)
    hsv[..., 2] = np.where(m, 60 + 190 * uv[0], 0).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------- 1. teaser
print("[1] teaser")
f = f"{B}/{CLIP}_cscC/f_0075.npz"
if not os.path.exists(f):
    f = f"{B}/{CLIP}/f_0075.npz"
photo = load(f"{RA}/{CLIP}_f_0075_photo.jpg")
panels = [photo]
if os.path.exists(f):
    panels += [iuv_rgb(f), normal_rgb(f)]
panels += [load(f"{RA}/{CLIP}_satin_0075.jpg"), load(f"{RA}/{CLIP}_objects_0075.jpg")]
panels = [crop_body(p, f) if p is not None else None for p in panels]
t = hstack(panels, h=560)
if t is not None:
    cv2.imwrite(f"{OUT}/fig_teaser.png", t, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print("   ", t.shape)

# ------------------------------------------------------- 2. frame-field suite
print("[2] frame-field suite")
suite = [("satin", "satin_0075"), ("sequins", "sequin_f0075"), ("chrome", "chrome_0075"),
         ("diffraction", "diffract_f0075"), ("fur", "fur_0075"), ("objects", "objects_0075")]
ims = []
for _, tag in suite:
    p = f"{RA}/{CLIP}_{tag}.jpg"
    im = load(p) if os.path.exists(p) else load(f"{RA}/15153663_{tag}.jpg")
    ims.append(crop_body(im, f) if im is not None else None)
row1 = hstack(ims[:3], h=460)
row2 = hstack(ims[3:], h=460)
if row1 is not None and row2 is not None:
    w = max(row1.shape[1], row2.shape[1])
    pad = lambda r: np.concatenate([r, np.full((r.shape[0], w - r.shape[1], 3), 255, np.uint8)], 1)
    grid = np.concatenate([pad(row1), np.full((8, w, 3), 255, np.uint8), pad(row2)], 0)
    cv2.imwrite(f"{OUT}/fig_framefield.png", grid, [cv2.IMWRITE_PNG_COMPRESSION, 6])
    print("   ", grid.shape)

# ------------------------------------------------------------ 3-5. carry over
print("[3-5] carrying over existing panels")
for src, dst in [(f"{B}/hands_4way.png", "fig_csc_hands.png"),
                 (f"{B}/face_zoom_4way.png", "fig_csc_face.png"),
                 (os.path.expanduser("~/data/dp_viz_compare/charts_v2/fig_charts_v1v2.png"),
                  "fig_charts.png"),
                 (os.path.expanduser("~/data/mcld/report_figs/qualitative.jpg"),
                  "fig_garment.png"),
                 (os.path.expanduser("~/data/mcld/report_figs/detail.jpg"),
                  "fig_garment_detail.png")]:
    if os.path.exists(src):
        im = cv2.imread(src)
        if im.shape[1] > 2200:
            im = cv2.resize(im, (2200, int(im.shape[0] * 2200 / im.shape[1])),
                            interpolation=cv2.INTER_AREA)
        cv2.imwrite(f"{OUT}/{dst}", im, [cv2.IMWRITE_PNG_COMPRESSION, 6])
        print(f"    {dst} {im.shape}")
    else:
        print(f"    MISSING {src}")

print("\nfigures in", OUT)
for fn in sorted(os.listdir(OUT)):
    print(f"  {os.path.getsize(OUT+'/'+fn)/1024:8.0f} KB  {fn}")
