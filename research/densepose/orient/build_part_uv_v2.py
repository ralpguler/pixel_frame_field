"""Rebuild the per-part UV charts so they are INJECTIVE (fold-free).

The existing charts (build_part_uv.py) are a planar PCA projection of each part. That is fine for the
18 half-cylinder parts, but a hand is ONE part (palm AND back AND fingers) and a plane folds the palm
onto the back of the hand. Measured on the current asset:

    hand_L/R   55-56% of surface samples share their (part,uv) with another triangle, up to 149 mm apart
    foot_L/R   65%,  up to 104 mm
    head_L/R   ~50%, up to 38 mm
    other 18   0.0%  (perfectly injective)

So the GT itself is ambiguous there: two surface points 15 cm apart carry the SAME (part,u,v) label, and
the network is asked to predict one target for both. That is the label, not the model -- and it is the
prime suspect for the hand/foot artifacts in the predicted normals.

Fix: re-flatten ONLY the 6 broken parts with a Tutte embedding (boundary pinned to the unit square,
mean-value/Floater interior weights). Floater's theorem: a disk-topology patch with a CONVEX boundary and
POSITIVE weights has a guaranteed fold-free embedding. All 6 are disk-topology (the head halves after the
eyeball components are removed). The 18 good parts keep their exact current UV, so the trained UV head
stays valid on them and only 6 charts have to be relearned.

The loader derives GT UV from this asset on the fly (data.py:133), so replacing it needs NO shard
regeneration -- 3.7 TB of shards stay untouched.

  python -m orient.build_part_uv_v2 --write     # -> smplx_face_uv_v2.npy
"""
import argparse
import os
from collections import defaultdict

import numpy as np
from scipy.sparse import coo_matrix, identity
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")
SHEET = [3, 4, 5, 6]  # hands (palm-vs-back) and feet (top-vs-sole): two height-field sheets over the
TUTTE = []            # part's own PCA plane. Tutte IS exactly injective on feet, but it compresses the
# toes so hard that the chart loses resolution there (texture round-trip median 0.203 mm vs 0.010 mm for
# a sheet split, and the MODEL's uv precision degrades the same way). The sheet split keeps the original
# PCA scale -- uniform, and much closer to the chart the network already learned -- at the cost of a 2-3%
# residual overlap on the silhouette rim, which the per-texel ambiguity mask drops.
# head (23,24) is LEFT ALONE. Its "50% overlap" is mostly SMPL-X hidden geometry inside the part
# (1088 eyeball faces + 756 inward-facing interior faces per half); the outer skin alone overlaps only
# 14-18% (ear pinna + jaw). Neither Tutte (38%), sheet-split (32%) nor spherical (69%) beats that, so we
# keep the existing chart and let the per-texel AMBIGUITY MASK in orient/csc.py drop the degenerate pixels.
BROKEN = SHEET + TUTTE


def boundary_loop(fl, n):
    """Ordered boundary vertex loop of a disk patch. Raises if it is not a single simple loop."""
    E = defaultdict(int)
    for t in fl:
        for i in range(3):
            a, b = sorted((int(t[i]), int(t[(i + 1) % 3]))); E[(a, b)] += 1
    bnd = [e for e, c in E.items() if c == 1]
    adj = defaultdict(list)
    for a, b in bnd:
        adj[a].append(b); adj[b].append(a)
    deg = {v: len(a) for v, a in adj.items()}
    assert all(d == 2 for d in deg.values()), f"boundary is not a simple loop (degrees {set(deg.values())})"
    start = bnd[0][0]; loop = [start]; prev = -1; cur = start
    while True:
        nxt = [x for x in adj[cur] if x != prev]
        prev, cur = cur, nxt[0]
        if cur == start:
            break
        loop.append(cur)
    assert len(loop) == len(adj), f"loop walk covered {len(loop)}/{len(adj)} boundary verts"
    return np.array(loop)


def mean_value_weights(fl, P, n):
    """Floater mean-value weights, accumulated PER FACE so no one-ring ordering is needed.

    In triangle (i,j,k) the angle g at i sits between edges ij and ik, so that face contributes
    tan(g/2)/|ij| to w_ij and tan(g/2)/|ik| to w_ik. All weights are strictly positive (g in (0,pi)),
    which is exactly what Tutte/Floater needs.
    """
    r, c, w = [], [], []
    for t in fl:
        for a in range(3):
            i, j, k = int(t[a]), int(t[(a + 1) % 3]), int(t[(a + 2) % 3])
            eij = P[j] - P[i]; eik = P[k] - P[i]
            lij = np.linalg.norm(eij); lik = np.linalg.norm(eik)
            if lij < 1e-12 or lik < 1e-12:
                continue
            g = np.arccos(np.clip(eij @ eik / (lij * lik), -1.0, 1.0))
            h = np.tan(g / 2)
            r += [i, i]; c += [j, k]; w += [h / lij, h / lik]
    W = coo_matrix((w, (r, c)), shape=(n, n)).tocsr()
    s = np.asarray(W.sum(1)).ravel()
    return W, np.clip(s, 1e-12, None)


def tutte(part, faces, fp, vt):
    """Fold-free UV in [0,1]^2 for one part. Returns (verts, uv, faces_local, n_eyeball_verts)."""
    f = faces[fp == part]
    vs = np.unique(f)
    rm = {int(v): i for i, v in enumerate(vs)}
    fl = np.array([[rm[int(x)] for x in t] for t in f])
    n = len(vs)
    # The SMPL-X head halves carry an EYEBALL as a separate closed component (chi=3, 2 face-components).
    # A closed sphere cannot be flattened to a disk: drop it from the solve, then give each eyeball vertex
    # the UV of its nearest vertex on the head surface (they sit in the socket; sub-cm, and the loss can
    # afford it -- it is orders of magnitude smaller than the 150 mm fold we are removing).
    ef = defaultdict(list)
    for i, t in enumerate(fl):
        for j in range(3):
            a, b = sorted((int(t[j]), int(t[(j + 1) % 3]))); ef[(a, b)].append(i)
    fr, fc = [], []
    for e, fs in ef.items():
        if len(fs) == 2:
            fr += [fs[0], fs[1]]; fc += [fs[1], fs[0]]
    ncf, lab = connected_components(coo_matrix((np.ones(len(fr)), (fr, fc)), shape=(len(fl),) * 2),
                                    directed=False)
    if ncf > 1:                                              # keep the largest face-component (the skin)
        big = np.bincount(lab).argmax()
        keep_f = lab == big
    else:
        keep_f = np.ones(len(fl), bool)
    fl_main = fl[keep_f]
    main_v = np.unique(fl_main)
    sub = {int(v): i for i, v in enumerate(main_v)}
    fl_sub = np.array([[sub[int(x)] for x in t] for t in fl_main])
    P = vt[vs[main_v]]
    m = len(main_v)

    loop = boundary_loop(fl_sub, m)
    d = np.linalg.norm(P[np.roll(loop, -1)] - P[loop], axis=1)
    s = np.concatenate([[0], np.cumsum(d)]); s = s[:-1] / s[-1]
    t4 = s * 4
    uvb = np.zeros((len(loop), 2))                            # boundary -> unit square (CONVEX: required)
    for i, ti in enumerate(t4):
        if ti < 1:   uvb[i] = [ti, 0]
        elif ti < 2: uvb[i] = [1, ti - 1]
        elif ti < 3: uvb[i] = [3 - ti, 1]
        else:        uvb[i] = [0, 4 - ti]

    W, rs = mean_value_weights(fl_sub, P, m)
    M = (identity(m, format="csr") - coo_matrix((1.0 / rs, (np.arange(m), np.arange(m))),
                                                shape=(m, m)).tocsr() @ W).tolil()
    for i in loop:
        M[i] = 0; M[i, i] = 1
    b = np.zeros((m, 2)); b[loop] = uvb
    uv_main = spsolve(M.tocsr(), b)

    uv = np.zeros((n, 2))
    uv[main_v] = uv_main
    other = np.setdiff1d(np.arange(n), main_v)                # eyeball verts -> nearest skin vertex's UV
    if len(other):
        uv[other] = uv_main[cKDTree(P).query(vt[vs[other]])[1]]
    return vs, uv, fl, len(other)


def overlap(uv, fl, P, n=4000, seed=7):
    """Fraction of surface samples whose (u,v) is claimed by >1 triangle, and the worst 3D collision."""
    rng = np.random.RandomState(seed)
    tuv = uv[fl]; txyz = P[fl]
    tree = cKDTree(tuv.mean(1))
    ti = rng.randint(0, len(fl), n); bb = rng.dirichlet([1, 1, 1], n)
    Q = (bb[:, :, None] * tuv[ti]).sum(1)
    k = min(64, len(fl)); _, cand = tree.query(Q, k=k); cand = cand.reshape(n, k)

    def bary(Pp, T):
        v0 = T[:, 1] - T[:, 0]; v1 = T[:, 2] - T[:, 0]; v2 = Pp - T[:, 0]
        d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
        d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
        den = d00 * d11 - d01 * d01; den = np.where(np.abs(den) < 1e-20, 1e-20, den)
        v = (d11 * d20 - d01 * d21) / den; w = (d00 * d21 - d01 * d20) / den
        return np.stack([1 - v - w, v, w], 1)

    nh = np.zeros(n, int); sp = np.zeros(n)
    for i in range(n):
        b2 = bary(np.repeat(Q[i][None], k, 0), tuv[cand[i]])
        ins = (b2 >= -1e-9).all(1); nh[i] = ins.sum()
        if ins.sum() > 1:
            X = (b2[ins][:, :, None] * txyz[cand[i][ins]]).sum(1)
            sp[i] = np.linalg.norm(X[:, None] - X[None], axis=2).max()
    return 100 * (nh > 1).mean(), sp.max() * 1000


def sheet_split(part, faces, fp, vt, charts):
    """Two height-field sheets over the part's OWN chart plane, packed into halves of the unit square.

    A hand folds because palm and back are two sheets over one plane. Split by which side of the chart
    plane each face looks at: each sheet is then single-valued over the plane, so the SAME PCA projection
    that folded becomes injective on each sheet.
    """
    sel = fp == part
    f = faces[sel]
    mu = charts[f"p{part}_mu"]; ax = charts[f"p{part}_axes"]
    nrm = np.cross(ax[0], ax[1]); nrm /= np.linalg.norm(nrm)
    fn = np.cross(vt[f[:, 1]] - vt[f[:, 0]], vt[f[:, 2]] - vt[f[:, 0]])
    fn /= np.clip(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12, None)
    side = (fn @ nrm) >= 0
    Pc = (vt[f] - mu) @ ax.T                                          # (T,3,2) corners in the chart plane
    uv = np.zeros_like(Pc)
    for s_, lo_u in ((True, 0.0), (False, 0.5)):
        m = side == s_
        if not m.any():
            continue
        Q = Pc[m].reshape(-1, 2)
        lo = Q.min(0); span = np.clip(Q.max(0) - lo, 1e-9, None)
        Qn = (Q - lo) / span
        Qn[:, 0] = lo_u + 0.5 * Qn[:, 0] * 0.92 + 0.02                 # WIDE gutter: a bilinear stencil
        Qn[:, 1] = Qn[:, 1] * 0.96 + 0.02                              # must never span both sheets
        uv[m] = Qn.reshape(-1, 3, 2)
    return uv, int(side.sum()), int((~side).sum())


def main(write=False):
    fp = np.load(f"{ASSETS}/smplx_face_to_dp24.npy").astype(int)
    faces = np.load(f"{ASSETS}/smplx_faces.npy")
    vt = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float64)
    fuv = np.load(f"{ASSETS}/smplx_face_uv.npy").astype(np.float64)     # (20908,3,2) current
    charts = np.load(f"{ASSETS}/dp24_charts.npz")
    new = fuv.copy()
    print(f"{'part':16s} {'method':12s} {'overlap':>18s} {'worst collision':>16s}")
    for p in BROKEN:
        sel = fp == p
        T = int(sel.sum()); tri = np.arange(T * 3).reshape(T, 3)
        ov0, _ = overlap(fuv[sel].reshape(-1, 2), tri, vt[faces[sel]].reshape(-1, 3))
        if p in SHEET:
            uvf, na, nb = sheet_split(p, faces, fp, vt, charts)
            ov1, mx1 = overlap(uvf.reshape(-1, 2), tri, vt[faces[sel]].reshape(-1, 3))
            note = f"sheets {na}/{nb}"
            new[sel] = uvf
        else:
            vs, uv, fl, neye = tutte(p, faces, fp, vt)
            ov1, mx1 = overlap(uv, fl, vt[vs])
            note = "disk"
            new[sel] = uv[fl]
        meth = "sheet-split" if p in SHEET else "tutte"
        print(f"p{p:<3d} {'':11s} {meth:12s} {ov0:7.1f}% -> {ov1:5.1f}% {mx1:13.1f} mm   {note}")
    if write:
        out = f"{ASSETS}/smplx_face_uv_v2.npy"
        np.save(out, new.astype(np.float32))
        print(f"\n[v2] wrote {out}  (18 parts unchanged, {len(BROKEN)} re-flattened)")
        same = np.abs(new[~np.isin(fp, BROKEN)] - fuv[~np.isin(fp, BROKEN)]).max()
        print(f"[v2] untouched parts bit-identical: max diff {same:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    main(a.write)
