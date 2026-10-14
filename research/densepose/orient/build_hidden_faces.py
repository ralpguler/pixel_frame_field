"""Build the SMPL-X hidden-geometry face set: faces whose rasterised GT labels are unlearnable.

Two populations (measured on GT: 1-3% of head pixels carry these labels -> the eye/lip speckle):
  1. EYEBALLS: two closed sphere components separate from the body mesh. They win rasterisation in
     the eye opening, but their UV is a nearest-socket-vertex constant (v2 charts) and their rot6d
     comes from a rigid sphere, not skin deformation.
  2. INTERIOR faces (head parts 23/24): mouth cavity, inner lips, eye sockets behind the eyeballs.
     Detected by a short outward ray from each face centroid: interior faces hit the mesh again
     within RAY_MAX *and* the hit is itself a head/eyeball face (the second condition stops the
     under-chin -> chest false positive).

Output: dp_assets/smplx_hidden_faces.npy (F,) bool. Training masks these pixels out of the
part/UV/rot/CSC losses (data.py hidden_mask) while keeping them in the body-mask BCE.

  python -m orient.build_hidden_faces
"""
import os

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

A = os.path.expanduser("~/data/bedlam2/dp_assets")
RAY_MAX = 0.10          # m; cavity hits are <~7 cm, chin->chest is >10 cm
HEAD_PARTS = (23, 24)

v = np.load(f"{A}/smplx_v_template.npy").astype(np.float64)
f = np.load(f"{A}/smplx_faces.npy").astype(int)
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
F = len(f)

# ---- 1. eyeballs: face components over shared vertices ----
r = np.repeat(np.arange(F), 3)
c = f.ravel()
vf = coo_matrix((np.ones(len(r)), (r, c)), shape=(F, len(v)))
adj = vf @ vf.T
ncomp, lab = connected_components(adj, directed=False)
sizes = np.bincount(lab)
main = sizes.argmax()
eyeball = lab != main
print(f"components: {ncomp}, sizes {sorted(sizes, reverse=True)[:4]} -> eyeball faces {eyeball.sum()}")

# ---- 2. interior head faces: short outward ray re-hits head geometry ----
tri = v[f]                                                  # (F,3,3)
cent_all = tri.mean(1)
n_all = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
n_all /= np.clip(np.linalg.norm(n_all, axis=1, keepdims=True), 1e-12, None)

head = np.where(np.isin(fpart, HEAD_PARTS) & ~eyeball)[0]
# candidate hit set: head + eyeball triangles only (also encodes the "hit must be head geometry" rule)
cand = np.where(np.isin(fpart, HEAD_PARTS) | eyeball)[0]
e1 = tri[cand, 1] - tri[cand, 0]                            # Moller-Trumbore, vectorised over triangles
e2 = tri[cand, 2] - tri[cand, 0]
v0 = tri[cand, 0]

interior = np.zeros(F, bool)
for i0 in range(0, len(head), 256):
    hs = head[i0:i0 + 256]
    o = cent_all[hs][:, None] + n_all[hs][:, None] * 2e-4   # (R,1,3)
    d = n_all[hs][:, None]
    p = np.cross(d, e2[None])                               # (R,C,3)
    det = (e1[None] * p).sum(-1)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
    tv = o - v0[None]
    u = (tv * p).sum(-1) * inv
    q = np.cross(tv, e1[None])
    w = (d * q).sum(-1) * inv
    t = (e2[None] * q).sum(-1) * inv
    hit = ok & (u >= 0) & (w >= 0) & (u + w <= 1) & (t > 1e-4) & (t < RAY_MAX)
    hit &= cand[None] != hs[:, None]                        # not the face itself
    interior[hs[hit.any(1)]] = True
print(f"head faces {len(head)}, interior {interior.sum()} "
      f"(per half: {interior[fpart == 23].sum()}/{interior[fpart == 24].sum()})")

hidden = eyeball | interior
np.save(f"{A}/smplx_hidden_faces.npy", hidden)
print(f"wrote {A}/smplx_hidden_faces.npy  hidden {hidden.sum()}/{F} faces "
      f"({100 * hidden.mean():.2f}%)")
