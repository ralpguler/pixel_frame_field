"""OBJECT PLACEMENT (stage 1) -- track canonical anchors and their full 6-DoF frames per frame.

Anchors are canonical (part, uv) points chosen ONCE on the template (sternum, forearm, head, ...).
Per frame, fully forward: the pixels whose predicted (part,uv) fall within a fixed template-mm
radius of the anchor vote for its image position (inverse-distance weights), its frame (weighted
mean of per-pixel t/n, re-orthonormalised), and its physical scale -- the pixel count inside the
fixed-mm disc gives px/mm directly (count = pi*(r*ppmm)^2). Too few pixels -> anchor not visible.
Position/frame are Gaussian-smoothed over time; visibility gets majority-vote hysteresis.

Output: objects.json  { anchors:[{name,kind,size_mm}], frames:{fi:[{vis,x,y,R(3x3 cols t,b,n,
BLENDER basis),ppmm}] } }  consumed by blender_objects.py.

  python -m orient.viz_objects_bake --dump DIR --out DIR
"""
import argparse
import json
import os

import numpy as np

from orient.beauty_common import erode_mask, frames_of, load

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--radius-mm", type=float, default=9.0)
ap.add_argument("--min-px", type=int, default=25)
ap.add_argument("--smooth", type=int, default=9)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

A = os.path.expanduser("~/data/bedlam2/dp_assets")
vt = np.load(f"{A}/smplx_v_template.npy")
vert_part = np.load(f"{A}/smplx_vert_to_dp24.npy").astype(int)
vert_uv = np.load(f"{A}/smplx_vert_uv.npy")
fuv = np.load(f"{A}/smplx_face_uv_v2.npy")
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
FA = np.load(f"{A}/smplx_faces.npy").astype(int)
SCALE = np.ones(25, np.float32) * 1000
for p in range(1, 25):
    fm = fpart == p
    if fm.any():
        e3d = np.linalg.norm(vt[FA[fm][:, 1]] - vt[FA[fm][:, 0]], axis=1) * 1000
        euv = np.linalg.norm(fuv[fm][:, 1] - fuv[fm][:, 0], axis=1)
        SCALE[p] = np.median(e3d / np.clip(euv, 1e-9, None))


# ---- anchors picked on a REFERENCE frame (click-to-attach AR): the predicted (part,uv) at a
# chosen image point becomes the canonical anchor, guaranteed visible and correctly predicted there.
ap_ref = int(os.environ.get("OBJ_REF", "75"))
_frames = frames_of(a.dump)
ref_fi = _frames[min(range(len(_frames)), key=lambda i: abs(_frames[i] - ap_ref))]
dref = load(a.dump, ref_fi)
rmask = erode_mask(dref["mask"], 2)
rys, rxs = np.nonzero(rmask)
bx0, bx1, by0, by1 = rxs.min(), rxs.max(), rys.min(), rys.max()
bw, bh = bx1 - bx0, by1 - by0


def at(fx, fy, parts=None, name=""):
    """Anchor = predicted (part,uv) at the body pixel nearest the bbox-relative point (fx,fy)."""
    sel = rmask.copy()
    if parts is not None:
        sel &= np.isin(dref["part"], list(parts))
    ys, xs = np.nonzero(sel)
    if len(ys) == 0:
        ys, xs = rys, rxs
    tx, ty = bx0 + fx * bw, by0 + fy * bh
    j = np.argmin(np.hypot(xs - tx, ys - ty))
    p = int(dref["part"][ys[j], xs[j]])
    uv = [float(dref["uv"][0][ys[j], xs[j]]), float(dref["uv"][1][ys[j], xs[j]])]
    print(f"[anchor] {name:6s} ref px=({xs[j]},{ys[j]}) part={p:2d} uv=({uv[0]:.3f},{uv[1]:.3f})")
    return p, uv


ARMS = {15, 16, 17, 18, 19, 20, 21, 22}
ANCH = []
for name, kind, size, (p, uv) in [
    ("badge", "badge", 65, at(0.50, 0.22, {1, 2}, "badge")),        # sternum
    ("watch", "watch", 46, at(0.20, 0.52, ARMS, "watch")),          # her right wrist area
    ("crown", "crown", 150, at(0.50, 0.02, {23, 24}, "crown")),     # top of head
    ("fly1", "butterfly", 85, at(0.72, 0.18, {1, 2}, "fly1")),      # left shoulder
    ("fly2", "butterfly", 75, at(0.38, 0.55, None, "fly2")),        # right hip
    ("pin1", "pin", 45, at(0.52, 0.35, {1, 2}, "pin1")),            # belly
    ("pin2", "pin", 45, at(0.60, 0.68, None, "pin2")),              # left thigh
    ("pin3", "pin", 45, at(0.80, 0.42, ARMS, "pin3")),              # left forearm
]:
    ANCH.append({"name": name, "kind": kind, "size_mm": size, "part": p, "uv": uv})

frames = frames_of(a.dump)
raw = {an["name"]: [] for an in ANCH}
for fi in frames:
    d = load(a.dump, fi)
    mask = erode_mask(d["mask"], 1)
    part = d["part"]; U, Vv = d["uv"][0], d["uv"][1]
    n_cam = d["n_cam"]; t_cam = d["t_cam"]
    for an in ANCH:
        p = an["part"]
        sel = mask & (part == p)
        ys, xs = np.nonzero(sel)
        if len(ys) < a.min_px:
            raw[an["name"]].append(None); continue
        dmm = np.hypot(U[ys, xs] - an["uv"][0], Vv[ys, xs] - an["uv"][1]) * SCALE[p]
        K = min(48, len(dmm))
        nn = np.argpartition(dmm, K - 1)[:K]
        if np.median(dmm[nn]) > 20.0:                          # nearest chart patch too far -> occluded
            raw[an["name"]].append(None); continue
        w = 1.0 / (dmm[nn] + 2.0); w /= w.sum()
        px = (xs[nn] * w).sum(); py = (ys[nn] * w).sum()
        t = (t_cam[ys[nn], xs[nn]] * w[:, None]).sum(0)
        n = (n_cam[ys[nn], xs[nn]] * w[:, None]).sum(0)
        n /= max(np.linalg.norm(n), 1e-9)
        t -= (t @ n) * n; t /= max(np.linalg.norm(t), 1e-9)
        dpx = np.hypot(xs[nn] - px, ys[nn] - py)
        good = dmm[nn] > 2.0
        ppmm = float(np.median(dpx[good] / dmm[nn][good])) if good.sum() >= 8 else None
        if ppmm is None or not np.isfinite(ppmm) or ppmm <= 0:
            raw[an["name"]].append(None); continue
        raw[an["name"]].append({"px": px, "py": py, "t": t, "n": n, "ppmm": ppmm})

H = W = load(a.dump, frames[0])["mask"].shape[0]
r = a.smooth // 2
ker = np.exp(-0.5 * (np.arange(-r, r + 1) / (a.smooth / 4.0)) ** 2)
out_frames = {}
for k, fi in enumerate(frames):
    lst = []
    for an in ANCH:
        seq = raw[an["name"]]
        win = [seq[j] for j in range(max(0, k - r), min(len(seq), k + r + 1))]
        vis_frac = np.mean([v is not None for v in win])
        ok = seq[k] is not None and vis_frac >= 0.6
        if not ok:
            lst.append({"vis": 0}); continue
        idxs = [j for j in range(max(0, k - r), min(len(seq), k + r + 1)) if seq[j] is not None]
        wk = np.array([ker[j - (k - r)] for j in idxs]); wk /= wk.sum()
        px = sum(seq[j]["px"] * wj for j, wj in zip(idxs, wk))
        py = sum(seq[j]["py"] * wj for j, wj in zip(idxs, wk))
        ppmm = sum(seq[j]["ppmm"] * wj for j, wj in zip(idxs, wk))
        t = sum(seq[j]["t"] * wj for j, wj in zip(idxs, wk))
        n = sum(seq[j]["n"] * wj for j, wj in zip(idxs, wk))
        n /= max(np.linalg.norm(n), 1e-9)
        t -= (t @ n) * n; t /= max(np.linalg.norm(t), 1e-9)
        b = np.cross(n, t)
        FLIP = np.array([1., 1., -1.])
        tb, bb, nb = t * FLIP, b * FLIP, n * FLIP              # camera -> Blender basis
        lst.append({"vis": 1, "x": px / W - 0.5, "y": 0.5 - py / H,
                    "R": [list(map(float, tb)), list(map(float, bb)), list(map(float, nb))],
                    "ppmm": ppmm})
    out_frames[fi] = lst

json.dump({"anchors": ANCH, "W": W, "frames": out_frames},
          open(f"{a.out}/objects.json", "w"))
vis_stats = {an["name"]: np.mean([out_frames[fi][i]["vis"] for fi in frames])
             for i, an in enumerate(ANCH)}
print("visibility:", {k: f"{100*v:.0f}%" for k, v in vis_stats.items()})
print(f"OBJECTS BAKE DONE -> {a.out}/objects.json ({len(frames)} frames)", flush=True)
