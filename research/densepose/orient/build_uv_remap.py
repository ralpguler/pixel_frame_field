"""Bake the v2->v1 chart remap: (part, uv_v2) -> uv_v1, exact through face+barycentrics.

COCO DensePose annotations are official-v1 chart coordinates; models trained on face_uv_v2 predict
v2 coordinates on the sheet-split parts (hands 3,4 / feet 5,6). Both charts parameterise the SAME
mesh, so for every v2 texel: locate its triangle in the v2 chart -> barycentric -> read the same
face's v1 corner UVs. Identity for the 20 unchanged parts. Output: (25, K, K, 2) float16.

  python -m orient.build_uv_remap [--K 512]
"""
import argparse
import os

import numpy as np
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--K", type=int, default=512)
ap.add_argument("--out", default=os.path.expanduser("~/data/bedlam2/dp_assets/uv_v2_to_v1.npy"))
a = ap.parse_args()

A = os.path.expanduser("~/data/bedlam2/dp_assets")
UV1 = np.load(f"{A}/smplx_face_uv.npy").astype(np.float64)
UV2 = np.load(f"{A}/smplx_face_uv_v2.npy").astype(np.float64)
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
K = a.K
SHEET = (3, 4, 5, 6)

grid = (np.stack(np.meshgrid(np.arange(K), np.arange(K), indexing="xy"), -1) + 0.5) / K  # (K,K,2) u,v
out = np.tile(grid[None].astype(np.float32), (25, 1, 1, 1))     # identity default


for p in SHEET:
    fm = np.where(fpart == p)[0]
    t2 = UV2[fm]                                                # (T,3,2)
    t1 = UV1[fm]
    tree = cKDTree(t2.mean(1))
    q = grid.reshape(-1, 2)
    kk = min(24, len(fm))
    _, cand = tree.query(q, k=kk)
    a_, b_, c_ = t2[cand][:, :, 0], t2[cand][:, :, 1], t2[cand][:, :, 2]
    v0, v1v, v2v = b_ - a_, c_ - a_, q[:, None] - a_
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1v).sum(-1); d11 = (v1v * v1v).sum(-1)
    d20 = (v2v * v0).sum(-1); d21 = (v2v * v1v).sum(-1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-14, 1e-14, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.stack([1 - v - w, v, w], -1)
    mn = bc.min(-1); best = mn.argmax(1); ar = np.arange(len(q))
    tsel = cand[ar, best]
    bsel = np.clip(bc[ar, best], 0, None)
    bsel /= np.clip(bsel.sum(1, keepdims=True), 1e-9, None)
    uv1 = (bsel[:, :, None] * t1[tsel]).sum(1)                  # same face, v1 corner UVs
    inside = mn[ar, best] >= -0.02                              # off-chart (gutter): keep nearest proj
    out[p] = uv1.reshape(K, K, 2).astype(np.float32)
    print(f"part {p}: {100 * inside.mean():.1f}% of texels inside a v2 triangle")

np.save(a.out, out.astype(np.float16))
print(f"wrote {a.out}  {out.shape}")
# round-trip sanity: v2 corner -> remap -> compare with v1 corner, on sheet parts
err = []
for p in SHEET:
    fm = np.where(fpart == p)[0][::7]
    q2 = UV2[fm].reshape(-1, 2)
    ij = np.clip((q2 * K).astype(int), 0, K - 1)
    got = out[p][ij[:, 1], ij[:, 0]]
    err.append(np.linalg.norm(got - UV1[fm].reshape(-1, 2), axis=1))
e = np.concatenate(err)
print(f"corner round-trip |uv_v1 err|: median {np.median(e):.4f}  p95 {np.percentile(e,95):.4f} (chart units)")
