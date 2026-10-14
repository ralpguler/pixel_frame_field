"""FUR (stage 1) -- bake real 3D strand geometry from the predicted frame field.

Roots are anchored in CANONICAL space (a per-part UV grid, like the sequins): each visible cell
contributes the pixel whose predicted (part,uv) is nearest the cell centre, so roots track the body
across frames and appear/disappear with visibility -- no template->image search. Each strand grows
from its root along the UNPROJECTED 3D predicted tangent (the z component tilts it out of the image
plane -- invisible to the ortho camera geometrically, but real to the LIGHTS), advecting through the
field with per-cell deterministic droop/jitter so strand shape is temporally coherent. Colour = twist
(twilight), like the LIC.

Writes strands_XXXX.npz  { pos (N,S,3) plane coords, col (N,3), rad (S,) }  for blender_fur.py.

  python -m orient.viz_fur_bake --dump DIR --out DIR [--mm 3.2] [--len 20] [--maxframes 0]
"""
import argparse
import os

import cv2
import numpy as np

from orient.beauty_common import bilinear, erode_mask, frames_of, load, plasma

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--mm", type=float, default=6.0, help="root grid pitch (mm on the template)")
ap.add_argument("--len", type=float, default=44.0, help="strand length (px at crop res)")
ap.add_argument("--seg", type=int, default=6, help="segments per strand")
ap.add_argument("--droop", type=float, default=0.20, help="downward droop as fraction of length")
ap.add_argument("--maxframes", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

A = os.path.expanduser("~/data/bedlam2/dp_assets")
fuv = np.load(f"{A}/smplx_face_uv_v2.npy")
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
vt = np.load(f"{A}/smplx_v_template.npy")
FA = np.load(f"{A}/smplx_faces.npy").astype(int)
SCALE = np.ones(25, np.float32) * 1000
for p in range(1, 25):
    fm = fpart == p
    if not fm.any():
        continue
    e3d = np.linalg.norm(vt[FA[fm][:, 1]] - vt[FA[fm][:, 0]], axis=1) * 1000
    euv = np.linalg.norm(fuv[fm][:, 1] - fuv[fm][:, 0], axis=1)
    SCALE[p] = np.median(e3d / np.clip(euv, 1e-9, None))


def hashf(k, salt):
    h = np.sin(k * 0.0001301 + salt * 269.5) * 43758.5453
    return h - np.floor(h)


def bake(d, fi):
    mask = erode_mask(d["mask"], 1)
    H, W = mask.shape
    part = d["part"].astype(np.int64)
    U, Vv = d["uv"][0].astype(np.float32), d["uv"][1].astype(np.float32)
    ys, xs = np.nonzero(mask)
    p = part[ys, xs]
    pitch = (a.mm / SCALE)[p]
    ix = np.floor(U[ys, xs] / pitch).astype(np.int64); iy = np.floor(Vv[ys, xs] / pitch).astype(np.int64)
    key = (p * 8192 + np.clip(ix, 0, 8191)) * 8192 + np.clip(iy, 0, 8191)
    cu = (ix + 0.5) * pitch; cv = (iy + 0.5) * pitch
    dc = np.hypot(U[ys, xs] - cu, Vv[ys, xs] - cv)
    order = np.lexsort((dc, key))
    kk = key[order]
    first = np.r_[True, kk[1:] != kk[:-1]]
    sel = order[first]                                          # one root pixel per visible cell
    N = len(sel)
    root = np.stack([xs[sel].astype(np.float32), ys[sel].astype(np.float32)], 1)
    ck = key[sel].astype(np.float64)
    # deterministic per-cell strand character (same every frame -> temporally coherent fur)
    j_ang = hashf(ck, 1.0) * 2 * np.pi
    j_mag = 0.10 * hashf(ck, 2.0)
    l_var = 0.6 + 0.55 * hashf(ck, 3.0)
    droop = a.droop * (0.6 + 0.8 * hashf(ck, 4.0))
    S = a.seg + 1
    pos = np.zeros((N, S, 3), np.float32)                       # image px x,y + z in px units
    pos[:, 0, :2] = root
    P = root.copy(); Z = np.zeros(N, np.float32)
    h = (a.len * l_var / a.seg).astype(np.float32)
    jd = np.stack([np.cos(j_ang), np.sin(j_ang)], 1).astype(np.float32) * j_mag[:, None]
    for s in range(1, S):
        t3 = bilinear(d["t_cam"], P)                            # camera-space tangent at current tip
        d2 = np.stack([t3[:, 0], -t3[:, 1]], 1)                 # image-space direction (y down)
        d2 = d2 + jd
        nrm = np.linalg.norm(d2, axis=1, keepdims=True)
        d2 /= np.clip(nrm, 1e-6, None)
        frac = (s / a.seg)
        P = P + d2 * h[:, None] + np.stack(
            [np.zeros(N, np.float32), (droop * a.len * l_var * frac * 2 / a.seg).astype(np.float32)], 1) * 0.5
        Z = Z + (-t3[:, 2]) * h * 0.8                           # camera z<0 faces viewer -> +z out of plane
        pos[:, s, :2] = P; pos[:, s, 2] = Z
    tw = bilinear(d["twist"], root)
    tw = np.clip((tw - 0.5) * 2.2 + 0.5, 0, 1)                  # stretch: limbs vs torso -> vivid range
    col = plasma(tw).astype(np.float32) * 0.85
    col *= (0.6 + 0.55 * hashf(ck, 5.0))[:, None]
    # -> backdrop-plane coordinates (plane is 1x1 at z=0, image y down -> plane y up)
    out = np.empty_like(pos)
    out[..., 0] = pos[..., 0] / W - 0.5
    out[..., 1] = 0.5 - pos[..., 1] / H
    out[..., 2] = np.clip(pos[..., 2], 0, None) / W + 0.0015    # keep strands in front of the plane
    rad = (np.linspace(1.0, 0.12, S) * 1.25 / W).astype(np.float32)
    # raw binary (Blender's python has no numpy): int32 N,S | f32 pos N*S*3 | f32 col N*3 | f32 rad S
    with open(f"{a.out}/strands_{fi:04d}.bin", "wb") as f:
        np.array([N, S], np.int32).tofile(f)
        out.astype(np.float32).tofile(f)
        np.clip(col, 0, 1).astype(np.float32).tofile(f)
        rad.tofile(f)
    return N


frames = frames_of(a.dump)
if a.maxframes:
    frames = frames[:a.maxframes]
for k, fi in enumerate(frames):
    n = bake(load(a.dump, fi), fi)
    if k % 20 == 0:
        print(f"frame {fi}: {n} strands", flush=True)
print(f"FUR BAKE DONE -> {a.out} ({len(frames)} frames)", flush=True)
