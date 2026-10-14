"""Globally-optimal smooth unit tangent field on the SMPL-X template (Knoppel et al. 2013,
"Globally Optimal Direction Fields") -- the proper combing for the frame-field visualisations.

WHY: wind/LIC/hatch display t_cam = R_pred @ t_template, so every defect of t_template reads as a
model artifact. The old choice (gradient of geodesic distance from the crown) has a critical point
at EVERY extremity (source at crown, sink at each fingertip/toe, saddles at armpits/crotch) plus
C0 creases across the cut locus. Poincare-Hopf only forces total index 2 on a genus-0 surface; the
smoothest field realises it with the minimum number of singular points, placed where the energy
wants them, and is C-infinity elsewhere.

Method: per-vertex local tangent bases; complex connection Laplacian with cotan weights (edge
transport angle = difference of the edge's angle in the two endpoint bases); smallest eigenvector
of L - lambda*A where A softly aligns to a guide field (head->toe geodesic gradient) so the comb
keeps flowing down the body. Direction = arg of the eigenvector in each vertex basis.

  python -m orient.template_field [--align 0.05] [--rosy 1]
writes  ~/data/bedlam2/dp_assets/smplx_tangent_smooth.npy  (10475,3) unit, in tangent planes,
and prints the singularity census (faces with nonzero holonomy winding).
"""
import argparse
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ap = argparse.ArgumentParser()
ap.add_argument("--align", type=float, default=0.05, help="soft alignment weight to head->toe guide")
ap.add_argument("--rosy", type=int, default=1, help="1=vector field, 2=line field")
ap.add_argument("--out", default=os.path.expanduser("~/data/bedlam2/dp_assets/smplx_tangent_smooth.npy"))
a = ap.parse_args()

A = os.path.expanduser("~/data/bedlam2/dp_assets")
vt = np.load(f"{A}/smplx_v_template.npy").astype(np.float64)
F = np.load(f"{A}/smplx_faces.npy").astype(int)
V = len(vt)

# vertex normals (area-weighted)
N = np.zeros_like(vt)
fn = np.cross(vt[F[:, 1]] - vt[F[:, 0]], vt[F[:, 2]] - vt[F[:, 0]])
for k in range(3):
    np.add.at(N, F[:, k], fn)
N /= np.clip(np.linalg.norm(N, axis=1, keepdims=True), 1e-12, None)

# local tangent bases (e1, e2)
ref = np.tile([0.0, 1.0, 0.0], (V, 1))
deg = np.abs((ref * N).sum(1)) > 0.95
ref[deg] = [1.0, 0.0, 0.0]
E1 = ref - (ref * N).sum(1, keepdims=True) * N
E1 /= np.clip(np.linalg.norm(E1, axis=1, keepdims=True), 1e-12, None)
E2 = np.cross(N, E1)


def edge_angles(P):
    """angle of edge vectors i->j in i's tangent basis, for an array of (i,j) pairs."""
    d = vt[P[:, 1]] - vt[P[:, 0]]
    return np.arctan2((d * E2[P[:, 0]]).sum(1), (d * E1[P[:, 0]]).sum(1))


# unique edges + cotan weights
edges = {}
for t in range(len(F)):
    i, j, k = F[t]
    for (u, v, w) in ((i, j, k), (j, k, i), (k, i, j)):
        key = (min(u, v), max(u, v))
        e1 = vt[u] - vt[w]; e2 = vt[v] - vt[w]
        cot = (e1 * e2).sum() / max(np.linalg.norm(np.cross(e1, e2)), 1e-12)
        edges.setdefault(key, 0.0)
        edges[key] += 0.5 * cot
EIJ = np.array(list(edges.keys()), int)
W = np.clip(np.array(list(edges.values())), 1e-6, None)     # clamp: keep L positive-ish

# transport angle rho_ij: rotating i's basis into j's along the edge.
# angle of edge (i->j) in basis_i must equal angle of (i->j)=-(j->i) in basis_j after transport.
a_i = edge_angles(EIJ)
a_j = edge_angles(EIJ[:, ::-1]) + np.pi
rho = a_j - a_i                                              # z_j ~ z_i * e^{i n rho}
n = a.rosy

# energy sum_ij w|z_j - e^{i n rho} z_i|^2  ->  L[j,i] = -w e^{+i n rho}, L[i,j] = conj
rows = np.r_[EIJ[:, 1], EIJ[:, 0], EIJ[:, 0], EIJ[:, 1]]
cols = np.r_[EIJ[:, 0], EIJ[:, 1], EIJ[:, 0], EIJ[:, 1]]
vals = np.r_[-W * np.exp(1j * n * rho), -W * np.exp(-1j * n * rho), W, W]
L = sp.coo_matrix((vals, (rows, cols)), shape=(V, V)).tocsr()

# guide: head->toe geodesic gradient (the OLD field), as a soft alignment only
from scipy.sparse.csgraph import dijkstra
w = np.linalg.norm(vt[EIJ[:, 0]] - vt[EIJ[:, 1]], axis=1)
G = sp.coo_matrix((np.r_[w, w], (np.r_[EIJ[:, 0], EIJ[:, 1]], np.r_[EIJ[:, 1], EIJ[:, 0]])),
                  shape=(V, V))
crown = int(np.argmax(vt[:, 1]))
geo = dijkstra(G.tocsr(), indices=crown)
geo[~np.isfinite(geo)] = geo[np.isfinite(geo)].max()
gg = np.zeros_like(vt)
p0, p1, p2 = vt[F[:, 0]], vt[F[:, 1]], vt[F[:, 2]]
fnv = np.cross(p1 - p0, p2 - p0)
A2 = np.linalg.norm(fnv, axis=1, keepdims=True)
un = fnv / np.clip(A2, 1e-12, None)
g0, g1, g2 = geo[F[:, 0]], geo[F[:, 1]], geo[F[:, 2]]
fg = (g0[:, None] * np.cross(un, p2 - p1) + g1[:, None] * np.cross(un, p0 - p2)
      + g2[:, None] * np.cross(un, p1 - p0)) / np.clip(A2, 1e-12, None)
for k in range(3):
    np.add.at(gg, F[:, k], fg * (A2 / 2))
gg -= (gg * N).sum(1, keepdims=True) * N
gnorm = np.linalg.norm(gg, axis=1)
z_guide = ((gg * E1).sum(1) + 1j * (gg * E2).sum(1))
z_guide = np.where(gnorm > 1e-9, z_guide / np.clip(np.abs(z_guide), 1e-12, None), 0) ** n

# Knoppel aligned formulation, one linear solve: minimise  z*Lz + align*|z - z_guide|^2
#   ->  (L + align*I) z = align * z_guide.
# align -> 0 recovers the pure smoothest field (then: smallest eigenvector); align sets how hard
# the comb is pulled toward head->toe.
if a.align > 0:
    z = spla.spsolve((L + a.align * sp.eye(V)).tocsc(), a.align * z_guide)
else:
    _, vecs = spla.eigsh(L, k=1, sigma=-1e-6, which="LM")
    z = vecs[:, 0].astype(complex)
E_sm = np.real(np.vdot(z, L @ z)) / max(np.real(np.vdot(z, z)), 1e-12)
z /= np.clip(np.abs(z), 1e-12, None)

# singularity census: winding of transported phase around each face
ang = {}
for (i, j), r in zip(map(tuple, EIJ), rho):
    ang[(i, j)] = r; ang[(j, i)] = -r
phase = np.angle(z)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


sing = []
for t in range(len(F)):
    i, j, k = F[t]
    h = (wrap(phase[j] - phase[i] - n * ang[(i, j)]) + wrap(phase[k] - phase[j] - n * ang[(j, k)])
         + wrap(phase[i] - phase[k] - n * ang[(k, i)]))
    # face holonomy: total transport angle around the triangle (angle defect)
    defect = wrap(n * (ang[(i, j)] + ang[(j, k)] + ang[(k, i)]))
    idx = (h + defect) / (2 * np.pi)
    if abs(idx) > 0.25:
        sing.append((t, idx))

# separate the VISIBLE skin from SMPL-X's hidden components (eyeball spheres, mouth interior):
# vertex connected components; the outer body is the biggest one.
comp = sp.csgraph.connected_components(
    sp.coo_matrix((np.ones(len(EIJ)), (EIJ[:, 0], EIJ[:, 1])), shape=(V, V)))[1]
main = np.bincount(comp).argmax()
skin = [(t, ix) for t, ix in sing if comp[F[t, 0]] == main]

theta = phase / n
T = np.cos(theta)[:, None] * E1 + np.sin(theta)[:, None] * E2
np.save(a.out, T.astype(np.float32))
guide_dot = (T * gg).sum(1) / np.clip(gnorm, 1e-9, None)
print(f"[field] rosy={n} align={a.align}  smoothness E={E_sm:.4e}")
print(f"[field] singular faces: {len(sing)} total (index {sum(s[1] for s in sing):+.1f}); "
      f"ON SKIN: {len(skin)} (index {sum(s[1] for s in skin):+.1f}, Poincare-Hopf demands +2)")
for t, idx in sorted(skin, key=lambda s: -abs(s[1]))[:12]:
    c = vt[F[t]].mean(0)
    print(f"    skin face {t}: index {idx:+.2f} at (x,y,z)=({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})")
print(f"[field] alignment with head->toe guide: mean cos={np.nanmean(guide_dot):+.3f}")
print(f"wrote {a.out}")
