"""E1 + E6 of paper/REFRAME.md -- evaluation only.

E1: a pixel frame derived from the correspondence ALONE (no rotation head): template point map P(x) through the
    chart, its image Jacobian J = dP/dx, the 2x2 map A (template tangent coords -> image), and the closed-form
    orthonormal solution of A = s M R [T1 T2] (two-fold tilt/Necker ambiguity), under full perspective from the
    GT-canvas intrinsics (or weak perspective). Disambiguation: oracle (upper bound) and two GT-free rules.
E6: every method's error split into NORMAL (swing) and TWIST (about the GT template normal) + full geodesic.

Protocol (pre-declared): ~/data/dp_frames_eval/PROTOCOL.txt. Outputs: ~/data/dp_frames_eval/.

  python -m orient.eval_frames_e1e6 selftest
  python -m orient.eval_frames_e1e6 bake
  python -m orient.eval_frames_e1e6 s2list
  python -m orient.eval_frames_e1e6 run --split S1 --ckpt dp_w_M/ckpt_120000.pt
  python -m orient.eval_frames_e1e6 report
"""
import argparse
import glob
import io
import json
import math
import os
import pickle
import sys
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import webdataset as wds                                                               # noqa: E402
from PIL import Image                                                                  # noqa: E402

from data import DPDecode, unpack_rgba, ASSETS, SHARDS_512                             # noqa: E402

OUT = os.path.expanduser("~/data/dp_frames_eval")
RUNS = os.path.expanduser("~/data/bedlam2/dp_runs")
POSED3D = os.path.expanduser("~/data/bedlam2/correspondence/posed3d")
HOLD = os.path.expanduser("~/data/datasets/dp_shards_rot_mamma/holdout_links")
M512 = os.path.expanduser("~/data/datasets/dp_shards_512_mamma/train")
MSRC = os.path.expanduser("~/data/datasets/mammasyn")
NTEX = f"{OUT}/assets/dp24_uv2n_K512_v2.npz"
N_PARTS = 24
GROUPS = {"torso": (1, 2), "hands_feet": (3, 4, 5, 6), "legs": tuple(range(7, 15)),
          "arms": tuple(range(15, 23)), "head": (23, 24)}
DBINS = (("d<4", 0, 4), ("d4-12", 4, 12), ("d>=12", 12, 1e9))
NBIN = 9000                                       # 0.02 deg histogram bins over [0, 180]
METRICS = ("geo", "normal", "twist")
R_LIST = (1, 2, 3)
ANG30 = 1 + 2 * math.cos(math.radians(30))        # trace threshold for a 30 deg geodesic


# ================================================================================================ geometry
def vertex_normals(V, Fc):
    """== orient.rot6d.vertex_normals (area-weighted, unit): the n_rest the GT rotations were pinned to."""
    n = np.zeros_like(V, dtype=np.float64)
    tri = V[Fc]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for k in range(3):
        np.add.at(n, Fc[:, k], fn)
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


def tangent_basis(N):
    """Orthonormal right-handed (T1, T2) with T1 x T2 = N. T1 = Gram-Schmidt of the template axis least aligned
    with N. The derived rotation does not depend on this choice (a tangent-plane rotation of T rotates A and the
    solution columns identically)."""
    a = torch.zeros_like(N)
    a.scatter_(-1, N.abs().argmin(-1, keepdim=True), 1.0)
    T1 = F.normalize(a - (a * N).sum(-1, keepdim=True) * N, dim=-1)
    T2 = torch.cross(N, T1, dim=-1)
    return T1, T2


def inv2(A):
    """Closed-form inverse of (...,2,2); returns (inv, det)."""
    a, b, c, d = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
    det = a * d - b * c
    sdet = torch.where(det.abs() > 0, det, torch.ones_like(det))
    inv = torch.stack([torch.stack([d, -b], -1), torch.stack([-c, a], -1)], -2) / sdet[..., None, None]
    return inv, det


def solve_frames(J, N, xt, yt, fx, fy, weak=False):
    """Closed-form frames from a correspondence Jacobian.

    J  (n,3,2) dP/d(canvas px), columns = (d/dcol, d/drow);  N (n,3) unit template normal at P
    xt, yt (n,) normalised PyTorch3D ray coords of the pixel (x left, y up);  fx, fy (n,) canvas focal (px)
    Model: A_n = s M R [T1 T2], A_n = diag(-1/fx, -1/fy) (T^T J)^-1, M = [[1,0,-xt],[0,1,-yt]] (weak: [I 0]).
    Returns R_plus, R_minus (n,3,3) (the two solutions: sign of the along-ray components), ok (n,) bool,
            ndotd (n,) = n.d of the derived camera normal (d = unit ray; < 0 = faces the camera; same for both)."""
    T1, T2 = tangent_basis(N)
    T = torch.stack([T1, T2], -1)                                           # (n,3,2)
    B2 = T.transpose(-1, -2) @ J                                            # canvas px -> tangent coords
    Ac, detB = inv2(B2)                                                     # tangent coords -> canvas px
    An = torch.stack([-Ac[:, 0] / fx[:, None], -Ac[:, 1] / fy[:, None]], 1)  # -> normalised image coords
    if weak:
        xt = torch.zeros_like(xt); yt = torch.zeros_like(yt)
    one = torch.ones_like(xt)
    d = F.normalize(torch.stack([xt, yt, one], -1), dim=-1)                 # unit ray
    e1 = F.normalize(torch.stack([d[:, 2], torch.zeros_like(xt), -d[:, 0]], -1), dim=-1)
    e2 = torch.cross(d, e1, dim=-1)                                         # (e1,e2,d) right-handed
    E = torch.stack([e1, e2], -1)                                           # (n,3,2)
    Mm = torch.stack([torch.stack([one, 0 * one, -xt], -1), torch.stack([0 * one, one, -yt], -1)], 1)  # (n,2,3)
    G = Mm @ E                                                              # (n,2,2), M d = 0
    Gi, detG = inv2(G)
    Ap = Gi @ An                                                            # weak-perspective problem in ray frame
    U, S, Vh = torch.linalg.svd(Ap)
    s1, s2 = S[:, 0], S[:, 1]
    v2 = Vh[:, 1, :]                                                        # 2nd right singular vector
    beta = Ap / s1[:, None, None]
    tmag = torch.sqrt((1 - (s2 / s1) ** 2).clamp(min=0))
    t = tmag[:, None] * v2                                                  # (n,2)
    base = E @ beta                                                         # (n,3,2)
    Cp = base + d[:, :, None] * t[:, None, :]
    Cm = base - d[:, :, None] * t[:, None, :]
    Tf = torch.stack([T1, T2, N], -1)                                       # (n,3,3) columns

    def frame(C):
        c1, c2 = C[..., 0], C[..., 1]
        return torch.stack([c1, c2, torch.cross(c1, c2, dim=-1)], -1) @ Tf.transpose(-1, -2)

    Rp, Rm = frame(Cp), frame(Cm)
    ok = (detB.abs() > 0) & (detG.abs() > 0) & (s2 > 1e-6 * s1) & torch.isfinite(Rp).all((-1, -2)) \
        & torch.isfinite(Rm).all((-1, -2))
    ndotd = torch.cross(Cp[..., 0], Cp[..., 1], dim=-1).mul(d).sum(-1)
    return Rp, Rm, ok, ndotd


def err_deg(Rp, Rg, N):
    """(geodesic, normal, twist) in degrees, float64. Normal = angle(R_p N, R_g N) = swing angle of R_g^T R_p about
    N; twist = rotation about R_g N left after aligning R_p N onto R_g N by the minimal rotation (== the twist of the
    swing-twist split of R_g^T R_p about N)."""
    Rp = Rp.double(); Rg = Rg.double(); N = N.double()
    D = Rg.transpose(-1, -2) @ Rp
    tr = D.diagonal(dim1=-2, dim2=-1).sum(-1)
    ax = torch.stack([D[..., 2, 1] - D[..., 1, 2], D[..., 0, 2] - D[..., 2, 0], D[..., 1, 0] - D[..., 0, 1]], -1) / 2
    geo = torch.rad2deg(torch.atan2(ax.norm(dim=-1), (tr - 1) / 2))
    a = (Rp @ N[..., None])[..., 0]; b = (Rg @ N[..., None])[..., 0]
    w = torch.cross(a, b, dim=-1); c = (a * b).sum(-1)
    nrm = torch.rad2deg(torch.atan2(w.norm(dim=-1), c))
    T1, _ = tangent_basis(N)
    tp = (Rp @ T1[..., None])[..., 0]; tg = (Rg @ T1[..., None])[..., 0]
    # minimal rotation a->b applied to tp:  c v + w x v + (w.v) w / (1+c)
    tpa = c[..., None] * tp + torch.cross(w, tp, dim=-1) + (w * tp).sum(-1, keepdim=True) * w / (1 + c).clamp(min=1e-9)[..., None]
    tw = torch.rad2deg(torch.atan2((torch.cross(tpa, tg, dim=-1) * b).sum(-1).abs(), (tpa * tg).sum(-1)))
    return geo, nrm, tw


def sixd_to_R(six):
    a0, a1 = six[..., :3], six[..., 3:]
    e0 = F.normalize(a0, dim=-1)
    a1 = a1 - (a1 * e0).sum(-1, keepdim=True) * e0
    e1 = F.normalize(a1, dim=-1)
    return torch.stack([e0, e1, torch.cross(e0, e1, dim=-1)], dim=-1)


def proj_so3(M):
    U, _, Vh = torch.linalg.svd(M)
    d = torch.det(U @ Vh)
    Dm = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    return U @ Dm @ Vh


def shift(x, dy, dx, dims=(1, 2)):
    """y[..., i, j, ...] = x[..., i+dy, j+dx, ...], zero outside."""
    y = torch.zeros_like(x)
    H, W = x.shape[dims[0]], x.shape[dims[1]]
    si = slice(max(dy, 0), H + min(dy, 0)); di = slice(max(-dy, 0), H + min(-dy, 0))
    sj = slice(max(dx, 0), W + min(dx, 0)); dj = slice(max(-dx, 0), W + min(-dx, 0))
    idx_s = [slice(None)] * x.dim(); idx_d = [slice(None)] * x.dim()
    idx_s[dims[0]] = si; idx_s[dims[1]] = sj; idx_d[dims[0]] = di; idx_d[dims[1]] = dj
    y[tuple(idx_d)] = x[tuple(idx_s)]
    return y


def stencil_ok(Q):
    """Q (B,G,G) part map, 0 = invalid. Central differences valid: x+-1 col and x+-1 row share Q(x) > 0."""
    ok = Q > 0
    for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        ok &= shift(Q, dy, dx) == Q
    return ok


def ls_jacobian(P, Q, r):
    """Least-squares plane fit of the template point map over a (2r+1)^2 window, same-part pixels only, free
    intercept. P (B,3,G,G) float, Q (B,G,G) part map (0 = invalid). Returns J (B,G,G,3,2) float64 = dP/d(col,row)."""
    P = P.double()
    B, _, G, _ = P.shape
    n = torch.zeros(B, G, G, dtype=torch.float64, device=P.device)
    Sd = torch.zeros(B, G, G, 2, dtype=torch.float64, device=P.device)
    Sdd = torch.zeros(B, G, G, 2, 2, dtype=torch.float64, device=P.device)
    SP = torch.zeros(B, G, G, 3, dtype=torch.float64, device=P.device)
    SPd = torch.zeros(B, G, G, 3, 2, dtype=torch.float64, device=P.device)
    Pp = P.permute(0, 2, 3, 1)                                              # (B,G,G,3)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            w = ((Q > 0) & (shift(Q, dy, dx) == Q)).double()
            dP = (shift(Pp, dy, dx) - Pp) * w[..., None]
            dv = torch.tensor([dx, dy], dtype=torch.float64, device=P.device)
            n += w
            Sd += w[..., None] * dv
            Sdd += w[..., None, None] * (dv[:, None] * dv[None, :])
            SP += dP
            SPd += dP[..., :, None] * dv
    nn_ = n.clamp(min=1)[..., None]
    md = Sd / nn_; mP = SP / nn_
    Cdd = Sdd / nn_[..., None] - md[..., :, None] * md[..., None, :]
    CPd = SPd / nn_[..., None] - mP[..., :, None] * md[..., None, :]
    Ci, det = inv2(Cdd)
    J = CPd @ Ci
    return J, det.abs() > 1e-9


# ================================================================================================ disambiguation
def part_consensus(Rp, Rm, key, gen, n_hyp=64, thr_deg=20.0, iters=3):
    """GT-free rigid-part rule. Rp,Rm (n,3,3); key (n,) group id (image*32+part). Returns labels (n,) bool
    (True = choose Rm)."""
    lab = torch.zeros(len(key), dtype=torch.bool, device=key.device)
    cthr = 1 + 2 * math.cos(math.radians(thr_deg))
    order = torch.argsort(key, stable=True)
    ks, counts = torch.unique_consecutive(key[order], return_counts=True)
    starts = torch.cumsum(counts, 0) - counts
    for st, ct in zip(starts.tolist(), counts.tolist()):
        idx = order[st:st + ct]
        C = torch.stack([Rp[idx], Rm[idx]], 1).float()                     # (m,2,3,3)
        m = len(idx)
        h = torch.randperm(m, generator=gen, device="cpu")[:n_hyp].to(C.device)
        H = C[h].reshape(-1, 3, 3)
        tr = torch.einsum("hij,mkij->hmk", H, C)
        score = (tr.max(-1).values > cthr).sum(1)
        Rref = H[score.argmax()]
        for _ in range(iters):
            t = torch.einsum("ij,mkij->mk", Rref, C)
            best = t.argmax(-1)
            ok = t.max(-1).values > cthr
            if ok.any():
                Rref = proj_so3(C[torch.arange(m, device=C.device), best][ok].sum(0))
        t = torch.einsum("ij,mkij->mk", Rref, C)
        lab[idx] = t[:, 1] > t[:, 0]
    return lab


NB8 = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def icm(Rp_g, Rm_g, lab_g, valid_g, part_g, sweeps=6):
    """Checkerboard ICM: each pixel takes the candidate with the larger tr(R_k^T sum_nb C_nb) (== smaller summed
    chordal distance) over its 8 same-part valid neighbours' current choices."""
    B, G, _ = lab_g.shape
    ii = torch.arange(G, device=lab_g.device)
    color = ((ii[:, None] + ii[None, :]) % 2)[None].expand(B, G, G)
    nbm = [(valid_g & shift(valid_g, dy, dx) & (shift(part_g, dy, dx) == part_g)) for dy, dx in NB8]
    for _ in range(sweeps):
        for c in (0, 1):
            Ch = torch.where(lab_g[..., None, None], Rm_g, Rp_g)
            S = torch.zeros_like(Ch)
            for (dy, dx), m in zip(NB8, nbm):
                S += m[..., None, None] * shift(Ch, dy, dx)
            sp = (Rp_g * S).sum((-1, -2)); sm = (Rm_g * S).sum((-1, -2))
            upd = valid_g & (color == c) & (S.abs().sum((-1, -2)) > 0)
            new = torch.where(sm > sp, torch.ones_like(lab_g), torch.where(sp > sm, torch.zeros_like(lab_g), lab_g))
            lab_g = torch.where(upd, new, lab_g)
    return lab_g


# ================================================================================================ assets
def bake_normals(K=512, uv_asset="smplx_face_uv_v2.npy"):
    """Per-part KxK texture of the template NORMAL over the chart, rasterised exactly like orient/csc.py build()
    (same texel centres, same first-containing-triangle rule, same clipped projection outside the chart). Also
    re-bakes XYZ to prove the rasteriser matches the stored CSC texture."""
    from scipy.spatial import cKDTree
    from orient.csc import _bary2d
    face_part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy").astype(int)
    face_uv = np.load(f"{ASSETS}/{uv_asset}").astype(np.float64)
    faces = np.load(f"{ASSETS}/smplx_faces.npy")
    vt = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float64)
    nrest = vertex_normals(vt, faces)
    g = (np.arange(K) + 0.5) / K
    gu, gv = np.meshgrid(g, g, indexing="xy")
    P = np.stack([gu.ravel(), gv.ravel()], 1)
    texn = np.zeros((N_PARTS, 3, K, K), np.float32); texx = np.zeros((N_PARTS, 3, K, K), np.float32)
    for p in range(1, N_PARTS + 1):
        sel = face_part == p
        tuv = face_uv[sel]; txyz = vt[faces[sel]]; tn = nrest[faces[sel]]
        tree = cKDTree(tuv.mean(1)); kq = min(32, len(tuv))
        _, cand = tree.query(P, k=kq); cand = cand.reshape(len(P), kq)
        done = np.zeros(len(P), bool); bc = np.zeros((len(P), 3)); ti = np.zeros(len(P), int)
        for j in range(kq):
            t = cand[:, j]; b = _bary2d(P, tuv[t]); ins = (b >= -1e-9).all(1); take = ins & ~done
            bc[take] = b[take]; ti[take] = t[take]; done |= take
        miss = ~done
        if miss.any():
            t = cand[miss, 0]; b = np.clip(_bary2d(P[miss], tuv[t]), 0, 1); b /= np.clip(b.sum(1, keepdims=True), 1e-12, None)
            bc[miss] = b; ti[miss] = t
        texx[p - 1] = (bc[:, :, None] * txyz[ti]).sum(1).reshape(K, K, 3).transpose(2, 0, 1)
        texn[p - 1] = (bc[:, :, None] * tn[ti]).sum(1).reshape(K, K, 3).transpose(2, 0, 1)
        print(f"  part {p:2d} baked", flush=True)
    ref = np.load(f"{ASSETS}/dp24_uv2xyz_K{K}.npz")
    dx = np.abs(ref["tex"] - texx).max()
    print(f"[bake] re-baked XYZ vs stored dp24_uv2xyz_K{K}.npz: max |diff| = {dx*1000:.6f} mm")
    assert dx < 1e-5, "re-bake does not match the stored CSC texture"
    os.makedirs(os.path.dirname(NTEX), exist_ok=True)
    np.savez_compressed(NTEX, tex=texn, cov=ref["cov"], amb=ref["amb"], K=K)
    print(f"[bake] wrote {NTEX}")


# ================================================================================================ data
class Dec(DPDecode):
    """DPDecode + the extras E1 needs (uid, shard, key, exact face/bary, gt_box). Standard fields are produced by
    the unmodified parent, so the sample stream is identical to build_loader's."""
    def __call__(self, sample):
        jpg, png, meta, rotnpy, url, key = sample
        out = super().__call__((jpg, png, meta, rotnpy))
        m = json.loads(meta)
        out["uid"] = m["uid"]; out["shard"] = os.path.basename(url); out["key"] = key; out["seq"] = m.get("seq")
        if self.gt512:
            face, bary = unpack_rgba(np.array(Image.open(io.BytesIO(png))))
            out["face"] = torch.from_numpy(face); out["bary"] = torch.from_numpy(bary)
            out["gt_box"] = torch.tensor(m["gt_box"], dtype=torch.float64)
        return out


def collate(samples):
    b = {}
    for k in samples[0]:
        v = [s[k] for s in samples]
        if isinstance(v[0], torch.Tensor) and all(x.shape == v[0].shape for x in v):
            b[k] = torch.stack(v)
        else:
            b[k] = v
    return b


def make_loader(shards, dec, batch, workers):
    """== data.build_loader(train=False) with two extra tuple keys (url, key); same shard split + order."""
    ds = wds.WebDataset(shards, shardshuffle=False, nodesplitter=wds.split_by_node,
                        workersplitter=wds.split_by_worker, empty_check=False, handler=wds.warn_and_continue)
    ds = (ds.to_tuple("rgb.jpg", "corr.png", "meta.json", "rot.npy", "__url__", "__key__",
                      handler=wds.warn_and_continue)
          .map(dec, handler=wds.warn_and_continue)
          .batched(batch, collation_fn=collate, partial=False))
    return wds.WebLoader(ds, batch_size=None, num_workers=workers, pin_memory=True,
                         persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


def s1_loader():
    shards = sorted(glob.glob(f"{SHARDS_512}/val/shard-*.tar"))
    dec = Dec(input_size=512, train=False, orient=True, uv_asset="smplx_face_uv_v2.npy", gt512=True)
    return make_loader(shards, dec, 16, 6)


def s2_native_loader():
    allsh = sorted(glob.glob(f"{HOLD}/*.tar"))
    sub = list(np.random.RandomState(0).permutation(allsh)[:59])
    dec = Dec(input_size=512, train=False, orient=True)                    # indomain_probe's decode
    return make_loader(sub, dec, 16, 2)


_P3D = {}


def bedlam_K(seq, uid):
    if seq not in _P3D:
        with np.load(f"{POSED3D}/{seq}.npz", allow_pickle=True) as z:
            _P3D[seq] = ({str(u): i for i, u in enumerate(z["uid"])}, z["cam_int"].copy())
    idx, K = _P3D[seq]
    return K[idx[uid]].astype(np.float64)


def canvas_K(K, gt_box, G=512):
    gx0, gy0, side = [float(x) for x in gt_box]
    s = G / side
    return np.array([K[0, 0] * s, K[1, 1] * s, (K[0, 2] - gx0) * s, (K[1, 2] - gy0) * s])   # fx', fy', cx', cy'


# ================================================================================================ accumulator
class Acc:
    def __init__(self):
        self.h = {}          # (method, metric, group) -> hist
        self.img = {}        # method -> list of (n_img, 3) per-image sums
        self.cnt = []        # per-image Omega counts
        self.extra = {}      # scalar counters
        self.sub = {}        # per-pixel subsample (method -> list)
        self.sub_meta = []

    def add_counts(self, k, v):
        self.extra[k] = self.extra.get(k, 0) + v

    def add(self, name, errs, img, groups, nimg, sub_mask=None):
        """errs: dict metric -> (n,) float64 over Omega; img (n,) local image idx; groups: dict gname -> mask."""
        for mi, mt in enumerate(METRICS):
            e = errs[mt]
            b = (e / 180.0 * NBIN).long().clamp(0, NBIN - 1)
            for gname, gm in groups.items():
                key = (name, mt, gname)
                hh = torch.bincount(b[gm], minlength=NBIN).cpu().numpy().astype(np.int64)
                self.h[key] = self.h.get(key, 0) + hh
        s = torch.zeros(nimg, 3, dtype=torch.float64, device=img.device)
        for mi, mt in enumerate(METRICS):
            s[:, mi].index_add_(0, img, errs[mt])
        self.img.setdefault(name, []).append(s.cpu().numpy())
        if sub_mask is not None:
            self.sub.setdefault(name, []).append(
                torch.stack([errs[m][sub_mask] for m in METRICS], 1).cpu().numpy().astype(np.float16))

    def save(self, path, meta):
        keys = sorted(self.h)
        np.savez_compressed(
            path,
            hist=np.stack([self.h[k] for k in keys]).astype(np.int32),
            hist_keys=np.array(["|".join(k) for k in keys]),
            img_methods=np.array(sorted(self.img)),
            img_sums=np.stack([np.concatenate(self.img[m]) for m in sorted(self.img)]),
            img_cnt=np.concatenate(self.cnt),
            sub_methods=np.array(sorted(self.sub)),
            sub_err=np.stack([np.concatenate(self.sub[m]) for m in sorted(self.sub)]) if self.sub else np.zeros(0),
            sub_meta=np.concatenate(self.sub_meta) if self.sub_meta else np.zeros(0),
            meta=json.dumps(meta), extra=json.dumps(self.extra))


# ================================================================================================ model
def load_model(ckpt, dev):
    from model import DINOv3DPT, backbone_from_state_dict, default_layers
    sd = torch.load(ckpt, map_location="cpu")["model"]
    bb = backbone_from_state_dict(sd)
    m = DINOv3DPT(backbone=bb, layers=default_layers(bb), pretrained=False, decoupled=True, orient=True,
                  rot_per_part=sd["rot_head.2.weight"].shape[0] != 6).to(dev).eval()
    m.load_state_dict(sd)
    return m


def gather_rot6d(rot6d, part):
    if rot6d.shape[1] == 6:
        return rot6d
    B, _, H, W = rot6d.shape
    r = rot6d.view(B, N_PARTS, 6, H, W)
    idx = (part - 1).clamp(min=0)
    return r.gather(1, idx[:, None, None].expand(B, 1, 6, H, W)).squeeze(1)


# ================================================================================================ core E1/E6 batch
class Ctx:
    def __init__(self, dev):
        from orient.csc import CSC
        self.dev = dev
        self.csc = CSC(512).to(dev)
        self.ntex = CSC(512, path=NTEX).to(dev)                    # same sampler, normal texture
        self.faces = torch.from_numpy(np.load(f"{ASSETS}/smplx_faces.npy").astype(np.int64)).to(dev)
        vt = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float64)
        self.vt = torch.from_numpy(vt).float().to(dev)
        self.nrest = torch.from_numpy(vertex_normals(vt, np.load(f"{ASSETS}/smplx_faces.npy"))).float().to(dev)
        self.f2p = torch.from_numpy(np.load(f"{ASSETS}/smplx_face_to_dp24.npy").astype(np.int64)).to(dev)
        self.hidden = torch.from_numpy(np.load(f"{ASSETS}/smplx_hidden_faces.npy")).to(dev)
        self.gen = torch.Generator(device="cpu").manual_seed(0)


def dist_to_silhouette(body):
    from scipy.ndimage import distance_transform_edt
    return np.stack([distance_transform_edt(m) for m in body.cpu().numpy()])


@torch.no_grad()
def eval_gt512_batch(model, ctx, b, Ks, acc, img0, repro=None):
    """One batch on the fixed 512 canvas. Ks (B,4) canvas fx',fy',cx',cy'. repro: list to append eval_512's
    per-batch median (only meaningful for S1)."""
    dev = ctx.dev
    img = b["img"].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(img)
    face = b["face"].to(dev).long(); bary = b["bary"].to(dev).float()
    part = b["part"].to(dev); fg = b["fg"].to(dev)                          # fg = face>=0 (eval_512: no hidden mask)
    theta = b["theta"].to(dev).float()
    B, G, _ = part.shape
    grid = F.affine_grid(theta, (B, 1, G, G), align_corners=False)
    W = lambda x: F.grid_sample(x.float(), grid, mode="bilinear", padding_mode="border", align_corners=False)
    pl = W(out["part"]); uvp = W(out["uv"]).view(B, N_PARTS, 2, G, G); rp = W(out["rot6d"])
    pid = pl.argmax(1) + 1
    Rg = sixd_to_R(b["rot6d"].to(dev).float())                              # (B,G,G,3,3)
    Rh = sixd_to_R(gather_rot6d(rp, pid).permute(0, 2, 3, 1))
    Rho = sixd_to_R(gather_rot6d(rp, part.clamp(min=1)).permute(0, 2, 3, 1))
    if repro is not None:                                                    # eval_512, verbatim statistic
        tr = torch.einsum("...ij,...ij->...", Rh, Rg)
        e = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))
        repro.append(float(e[fg].median()))
        tro = torch.einsum("...ij,...ij->...", Rho, Rg)
        repro_pool = torch.rad2deg(torch.arccos(((tr - 1) / 2).clamp(-1, 1)))[fg]
        acc.add_counts("repro_pool_sum", float(repro_pool.double().sum())); acc.add_counts("repro_pool_n", int(fg.sum()))
    # ---- GT correspondence: exact template point + normal (float32 from face+bary; the loader's xyz is f16)
    fsafe = face.clamp(min=0)
    tv = ctx.faces[fsafe]                                                    # (B,G,G,3)
    Pg = (bary[..., None] * ctx.vt[tv]).sum(-2)                              # (B,G,G,3)
    Ng = F.normalize((bary[..., None] * ctx.nrest[tv]).sum(-2), dim=-1)
    hid = ctx.hidden[fsafe] & fg
    body = fg                                                                # target person's full silhouette
    fgv = fg & ~hid                                                          # supervised pixels
    Qg = torch.where(fgv, part, torch.zeros_like(part))
    # ---- predicted correspondence through the chart
    sel = uvp.gather(1, (pid - 1)[:, None, None].expand(B, 1, 2, G, G)).squeeze(1)
    Pp = torch.stack([ctx.csc(sel[i], pid[i]) for i in range(B)])            # (B,3,G,G)
    Np = F.normalize(torch.stack([ctx.ntex(sel[i], pid[i]) for i in range(B)]), dim=1).permute(0, 2, 3, 1)
    Qp = torch.where(fgv, pid, torch.zeros_like(pid))                        # neighbours: GT person px, same pred part
    okg = stencil_ok(Qg); okp = stencil_ok(Qp)
    acc.add_counts("n_fg", int(fg.sum())); acc.add_counts("n_nohidden", int(fgv.sum()))
    acc.add_counts("n_gt_stencil", int((fgv & okg).sum())); acc.add_counts("n_both_stencil", int((fgv & okg & okp).sum()))
    # ---- pixel rays (canvas intrinsics)
    Kt = torch.as_tensor(np.asarray(Ks), dtype=torch.float64, device=dev)   # (B,4)
    jj = torch.arange(G, device=dev, dtype=torch.float64)
    xt = -(jj[None, None, :] + 0.5 - Kt[:, 2, None, None]) / Kt[:, 0, None, None]    # (B,1,G)
    yt = -(jj[None, :, None] + 0.5 - Kt[:, 3, None, None]) / Kt[:, 1, None, None]    # (B,G,1)
    xt = xt.expand(B, G, G); yt = yt.expand(B, G, G)
    fxg = Kt[:, 0, None, None].expand(B, G, G); fyg = Kt[:, 1, None, None].expand(B, G, G)
    bidx = torch.arange(B, device=dev)[:, None, None].expand(B, G, G)
    # ---- solve every (corr, r, cam) on its own stencil-valid set, collect candidates
    cands = {}
    for corr, P_, N_, Q_, ok_ in (("pred", Pp, Np, Qp, okp), ("gt", Pg.permute(0, 3, 1, 2), Ng, Qg, okg)):
        for r in R_LIST:
            J, okJ = ls_jacobian(P_, Q_, r)
            v = ok_ & okJ
            for cam in ("persp", "weak"):
                Rp_, Rm_, oks, ndd = solve_frames(J[v], N_[v].double(), xt[v], yt[v], fxg[v], fyg[v], weak=(cam == "weak"))
                vv = v.clone(); vv[v] = oks
                cands[(corr, r, cam)] = (vv, Rp_[oks].float(), Rm_[oks].float(), ndd[oks], Q_)
            del J
    # ---- Omega: supervised, both stencils, all solves finite
    om = fgv & okg & okp
    for k, (vv, *_rest) in cands.items():
        om &= vv
    acc.add_counts("n_omega", int(om.sum()))
    nimg = B
    im = bidx[om] - 0                                                        # local image index
    cnt = torch.bincount(im, minlength=nimg).cpu().numpy()
    acc.cnt.append(cnt)
    RgO = Rg[om]; NgO = Ng[om]
    dist = torch.from_numpy(dist_to_silhouette(body)).to(dev)[om]
    gp = part[om]
    groups = {"all": torch.ones_like(gp, dtype=torch.bool)}
    for gname, ps in GROUPS.items():
        groups[gname] = torch.isin(gp, torch.tensor(ps, device=dev))
    for dn, lo, hi in DBINS:
        groups[dn] = (dist >= lo) & (dist < hi)
    ii = torch.arange(G, device=dev)
    subm_grid = ((ii[:, None] % 8 == 0) & (ii[None, :] % 8 == 0))[None].expand(B, G, G)
    subm = subm_grid[om]
    rows = ii[:, None].expand(G, G)[None].expand(B, G, G)[om]; cols = ii[None, :].expand(G, G)[None].expand(B, G, G)[om]
    acc.sub_meta.append(np.stack([(im + img0).cpu().numpy(), rows.cpu().numpy(), cols.cpu().numpy(),
                                  gp.cpu().numpy(), dist.cpu().numpy()], 1)[subm.cpu().numpy()].astype(np.float32))
    SUBKEEP = {"H", "H-orc"} | {f"D-{c}-{d}-persp-r{r}" for c in ("pred", "gt") for d in ("orc", "part", "smooth") for r in R_LIST}

    def score(name, R):
        g_, n_, t_ = err_deg(R, RgO, NgO)
        acc.add(name, {"geo": g_, "normal": n_, "twist": t_}, im, groups, nimg, subm if name in SUBKEEP else None)

    score("H", Rh[om]); score("H-orc", Rho[om])
    for (corr, r, cam), (vv, Rp_, Rm_, ndd, Q_) in cands.items():
        # grid scatter of the candidates on this method's valid set
        Rp_g = torch.zeros(B, G, G, 3, 3, device=dev); Rm_g = torch.zeros_like(Rp_g)
        Rp_g[vv] = Rp_; Rm_g[vv] = Rm_
        tag = f"{corr}-{cam}-r{r}"
        # ambiguity + facing, on Omega
        trpm = (Rp_g[om] * Rm_g[om]).sum((-1, -2))
        acc.add_counts(f"amb30_{tag}", int((trpm < ANG30).sum()))
        nd = torch.zeros(B, G, G, dtype=torch.float64, device=dev); nd[vv] = ndd
        acc.add_counts(f"backfacing_{tag}", int((nd[om] > 0).sum()))
        # oracle
        tp = (Rp_g[om] * RgO).sum((-1, -2)); tm = (Rm_g[om] * RgO).sum((-1, -2))
        score(f"D-{corr}-orc-{cam}-r{r}", torch.where((tm > tp)[:, None, None], Rm_g[om], Rp_g[om]))
        # part consensus (GT-free; groups = image x this correspondence's part)
        key = (bidx[vv] * 32 + Q_[vv]).long()
        lab = part_consensus(Rp_, Rm_, key, ctx.gen)
        lab_g = torch.zeros(B, G, G, dtype=torch.bool, device=dev); lab_g[vv] = lab
        score(f"D-{corr}-part-{cam}-r{r}", torch.where(lab_g[om][:, None, None], Rm_g[om], Rp_g[om]))
        lab_s = icm(Rp_g, Rm_g, lab_g, vv, Q_)
        score(f"D-{corr}-smooth-{cam}-r{r}", torch.where(lab_s[om][:, None, None], Rm_g[om], Rp_g[om]))
        del Rp_g, Rm_g
    return B


# ================================================================================================ floor diagnostic
def posed_cam_verts_bedlam(seq, uid, dev):
    """PyTorch3D camera-space posed vertices, EXACTLY as orient/rot6d.vert_rot6d_for_body builds them."""
    sys.path.insert(0, os.path.expanduser("~/codes/wan/bedlam"))
    from render_correspondence import pose_instance, world_to_cam
    if seq not in _P3D:
        bedlam_K(seq, uid)
    with np.load(f"{POSED3D}/{seq}.npz", allow_pickle=True) as z:
        r = _P3D[seq][0][uid]
        vw, _ = pose_instance(z["pose_world"][r], z["betas"][r], z["trans_world"][r], str(z["gender"][r]), dev)
        vc = world_to_cam(vw, z["cam_ext"][r], dev).float()
    return vc * torch.tensor([-1.0, -1.0, 1.0], device=dev)


def sqrtm2(Sm):
    """Symmetric PSD (...,2,2) -> (sqrt, inverse sqrt, eigenvalues asc). Closed form (cuSOLVER's batched eigh
    rejects ~1M-matrix batches): sqrt(S) = (S + sqrt(det) I) / sqrt(tr + 2 sqrt(det))."""
    a, b, c = Sm[..., 0, 0], Sm[..., 0, 1], Sm[..., 1, 1]
    det = (a * c - b * b).clamp(min=0); tr = a + c
    sd = det.sqrt()
    I2 = torch.eye(2, dtype=Sm.dtype, device=Sm.device)
    sq = (Sm + sd[..., None, None] * I2) / (tr + 2 * sd).clamp(min=1e-30).sqrt()[..., None, None]
    isq, _ = inv2(sq)
    disc = ((a - c) ** 2 + 4 * b * b).sqrt()
    lam = torch.stack([(tr - disc) / 2, (tr + disc) / 2], -1).clamp(min=1e-30)
    return sq, isq, lam


@torch.no_grad()
def floor_batch(ctx, b, Ks, Vc, acc, r=3):
    """GT-only decomposition of the derivation floor, on the GT correspondence. Vc: list of (V,3) p3d camera verts.
    Per pixel (GT-stencil-valid, non-hidden):
      Fp   = J_cam (T^T J_tmpl)^-1 : template tangent coords -> camera 3D (J_cam, J_tmpl by the same LS stencil)
      kappa = sqrt(lmax/lmin) of Fp^T Fp  (local anisotropy of the template->posed map; 1 = conformal)
      R3d  = polar frame of Fp: [q1 q2 q1xq2][T1 T2 N]^T, Q = Fp (Fp^T Fp)^-1/2  -> 'definitional' floor vs R_gt
      Dstr = derivation from the image Jacobian with the TRUE stretch removed (J -> T S T^T J), oracle-disambiguated
      D    = the plain derivation (same stencil), oracle-disambiguated
      true-tilt ambiguity: geodesic between R_gt and its exact Necker partner (solve on the ideal J built from R_gt)
      perspective linearisation check: |A_n - (1/z) M Fp| / |A_n|."""
    dev = ctx.dev
    face = b["face"].to(dev).long(); bary = b["bary"].to(dev).float(); part = b["part"].to(dev); fg = b["fg"].to(dev)
    B, G, _ = part.shape
    Rg = sixd_to_R(b["rot6d"].to(dev).float())
    fsafe = face.clamp(min=0)
    tv = ctx.faces[fsafe]
    Pg = (bary[..., None] * ctx.vt[tv]).sum(-2)
    Ng = F.normalize((bary[..., None] * ctx.nrest[tv]).sum(-2), dim=-1)
    Xc = torch.stack([(bary[i][..., None] * Vc[i][tv[i]]).sum(-2) for i in range(B)])      # (B,G,G,3) camera
    hid = ctx.hidden[fsafe] & fg
    fgv = fg & ~hid
    Qg = torch.where(fgv, part, torch.zeros_like(part))
    ok = stencil_ok(Qg)
    Jt, okt = ls_jacobian(Pg.permute(0, 3, 1, 2), Qg, r)
    Jc, okc = ls_jacobian(Xc.permute(0, 3, 1, 2), Qg, r)
    v = ok & okt & okc
    Kt = torch.as_tensor(np.asarray(Ks), dtype=torch.float64, device=dev)
    jj = torch.arange(G, device=dev, dtype=torch.float64)
    xt = (-(jj[None, None, :] + 0.5 - Kt[:, 2, None, None]) / Kt[:, 0, None, None]).expand(B, G, G)
    yt = (-(jj[None, :, None] + 0.5 - Kt[:, 3, None, None]) / Kt[:, 1, None, None]).expand(B, G, G)
    fxg = Kt[:, 0, None, None].expand(B, G, G); fyg = Kt[:, 1, None, None].expand(B, G, G)
    Jt_, Jc_, N_, Rg_, xt_, yt_, fx_, fy_ = Jt[v], Jc[v], Ng[v].double(), Rg[v].double(), xt[v], yt[v], fxg[v], fyg[v]
    z_ = Xc[v][:, 2].double()
    T1, T2 = tangent_basis(N_)
    T = torch.stack([T1, T2], -1)
    B2 = T.transpose(-1, -2) @ Jt_
    B2i, detB = inv2(B2)
    Fp = Jc_ @ B2i                                                          # (n,3,2)
    Sq, Sqi, lam = sqrtm2(Fp.transpose(-1, -2) @ Fp)
    kappa = (lam[:, 1] / lam[:, 0]).sqrt()
    Q = Fp @ Sqi
    Tf = torch.stack([T1, T2, N_], -1)
    R3d = torch.stack([Q[..., 0], Q[..., 1], torch.cross(Q[..., 0], Q[..., 1], dim=-1)], -1) @ Tf.transpose(-1, -2)
    # perspective linearisation check
    one = torch.ones_like(xt_)
    Mm = torch.stack([torch.stack([one, 0 * one, -xt_], -1), torch.stack([0 * one, one, -yt_], -1)], 1)
    An = torch.stack([-B2i[:, 0] / fx_[:, None], -B2i[:, 1] / fy_[:, None]], 1)
    lin = (An - (Mm @ Fp) / z_[:, None, None]).flatten(1).norm(dim=1) / An.flatten(1).norm(dim=1)

    def orc(Jin):
        Rp_, Rm_, oks, _ = solve_frames(Jin, N_, xt_, yt_, fx_, fy_)
        tp = (Rp_ * Rg_).sum((-1, -2)); tm = (Rm_ * Rg_).sum((-1, -2))
        return torch.where((tm > tp)[:, None, None], Rm_, Rp_), oks

    Rd, ok1 = orc(Jt_)
    Rs, ok2 = orc(T @ Sq @ B2)                                              # true stretch removed
    # exact Necker partner of the TRUE rotation (ideal, conformal J built from R_gt)
    Aid = (Mm @ Rg_ @ T)
    Acid = torch.stack([-Aid[:, 0] * fx_[:, None], -Aid[:, 1] * fy_[:, None]], 1)
    Rp_, Rm_, ok3, _ = solve_frames(T @ inv2(Acid)[0], N_, xt_, yt_, fx_, fy_)
    gp_, _, _ = err_deg(Rp_, Rg_, N_); gm_, _, _ = err_deg(Rm_, Rg_, N_)
    partner = torch.where((gp_ < gm_)[:, None, None], Rm_, Rp_)
    gpart, _, _ = err_deg(partner, Rg_, N_)
    good = ok1 & ok2 & ok3 & torch.isfinite(kappa) & torch.isfinite(R3d).all((-1, -2))
    out = {"kappa": kappa, "lin": lin, "amb_true": gpart}
    for nm, R in (("D", Rd), ("Dstr", Rs), ("R3d", R3d)):
        g_, n_, t_ = err_deg(R, Rg_, N_)
        out[nm + "_geo"] = g_; out[nm + "_normal"] = n_; out[nm + "_twist"] = t_
    gpv = part[v]
    for k in out:
        out[k] = out[k][good].float().cpu().numpy()
    out["part"] = gpv[good].cpu().numpy().astype(np.int8)
    acc.append(out)


def cmd_floor(a):
    """GT-only floor decomposition (checkpoint-independent) on S1 (posed BEDLAM2 bodies) or S2 (MAMMA vertices3d)."""
    dev = "cuda"
    ctx = Ctx(dev)
    res = []
    n = 0
    t0 = time.time()
    if a.split == "S1":
        for bi, b in enumerate(s1_loader()):
            Ks = [canvas_K(bedlam_K(b["seq"][k], b["uid"][k]), b["gt_box"][k].numpy()) for k in range(len(b["uid"]))]
            Vc = [posed_cam_verts_bedlam(b["seq"][k], b["uid"][k], dev) for k in range(len(b["uid"]))]
            floor_batch(ctx, b, Ks, Vc, res, r=a.r)
            n += len(b["uid"])
            print(f"  floor S1 n={n} ({time.time()-t0:.0f}s)", flush=True)
            if n >= (a.limit or 512):
                break
    else:
        rows = json.load(open(f"{OUT}/s2_samples.json"))
        rows = rows[: (a.limit or len(rows))]
        V = {}
        for src in sorted({r["src"] for r in rows}):
            want = {r["pyd"]: r for r in rows if r["src"] == src}
            with tarfile.open(src) as t:
                for m in t:
                    for w in want:
                        if m.name.endswith(w):
                            V[(src, w)] = pickle.load(t.extractfile(m))
        for idx, b, miss in s2_gt512_batches(rows):
            Ks = [canvas_K(np.asarray(rows[i]["K"]), b["gt_box"][k].numpy()) for k, i in enumerate(idx)]
            Vc = []
            for i in idx:
                d = V[(rows[i]["src"], rows[i]["pyd"])]
                pid = int(rows[i]["uid"].split("/")[-1])
                vv = torch.as_tensor(np.asarray(d[pid]["vertices3d"], np.float64), device=dev).float()
                Vc.append(vv * torch.tensor([-1.0, -1.0, 1.0], device=dev))          # OpenCV -> PyTorch3D (= _FLIP)
            floor_batch(ctx, b, Ks, Vc, res, r=a.r)
            n += len(idx)
            print(f"  floor S2 n={n} ({time.time()-t0:.0f}s)", flush=True)
    cat = {k: np.concatenate([o[k] for o in res]) for k in res[0]}
    path = f"{OUT}/floor_{a.split}_r{a.r}{'_lim' + str(a.limit) if a.limit else ''}.npz"
    np.savez_compressed(path, **cat)
    L = [f"FLOOR {a.split} r={a.r}: {n} images, {len(cat['kappa'])} px"]
    for nm, lab in (("R3d", "per-pixel polar frame of the TRUE posed surface (definition floor)"),
                    ("Dstr", "derivation, TRUE stretch removed, oracle sign"),
                    ("D", "derivation from GT correspondence, oracle sign (= E1 floor)")):
        s = f"  {lab:66s}"
        for mt in METRICS:
            e = cat[f"{nm}_{mt}"]
            s += f" | {mt} {e.mean():6.2f} {np.median(e):6.2f} {np.percentile(e, 90):6.2f}"
        L.append(s)
    k = cat["kappa"]
    L.append(f"  anisotropy kappa = sqrt(lmax/lmin): median {np.median(k):.3f}  p10 {np.percentile(k,10):.3f}  "
             f"p90 {np.percentile(k,90):.3f}  share>1.1 {100*(k>1.1).mean():.1f}%  share>1.25 {100*(k>1.25).mean():.1f}%")
    for lo, hi in ((1, 1.05), (1.05, 1.1), (1.1, 1.2), (1.2, 1.4), (1.4, 99)):
        m = (k >= lo) & (k < hi)
        L.append(f"    kappa [{lo:.2f},{hi:.2f}): {100*m.mean():5.1f}% px | D geo {cat['D_geo'][m].mean():6.2f} normal "
                 f"{cat['D_normal'][m].mean():6.2f} twist {cat['D_twist'][m].mean():5.2f} | Dstr geo {cat['Dstr_geo'][m].mean():5.2f}")
    L.append(f"  perspective linearisation |A_n-(1/z)M Fp|/|A_n|: median {np.median(cat['lin']):.2e} p99 {np.percentile(cat['lin'],99):.2e}")
    at = cat["amb_true"]
    L.append(f"  TRUE-geometry ambiguity (R_gt vs its exact Necker partner): >30 deg on {100*(at>30).mean():.1f}% of px, "
             f"median {np.median(at):.1f} deg")
    txt = "\n".join(L)
    print(txt)
    open(path.replace(".npz", ".txt"), "w").write(txt + "\n")


# ================================================================================================ commands
def cmd_selftest():
    torch.manual_seed(0)
    dev = "cuda"
    n = 200000
    q = F.normalize(torch.randn(n, 4, dtype=torch.float64, device=dev), dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    N = F.normalize(torch.randn(n, 3, dtype=torch.float64, device=dev), dim=-1)
    T1, T2 = tangent_basis(N)
    Tm = torch.stack([T1, T2], -1)
    fx = torch.full((n,), 900.0, dtype=torch.float64, device=dev) * (1 + 0.2 * torch.rand(n, device=dev, dtype=torch.float64))
    fy = fx * (1 + 0.002 * torch.randn(n, device=dev, dtype=torch.float64))
    zc = 2 + 8 * torch.rand(n, device=dev, dtype=torch.float64)
    for case in ("weak", "persp"):
        if case == "weak":
            xt = torch.zeros(n, device=dev, dtype=torch.float64); yt = torch.zeros_like(xt)
        else:
            xt = (torch.rand(n, device=dev, dtype=torch.float64) - 0.5) * 1.2
            yt = (torch.rand(n, device=dev, dtype=torch.float64) - 0.5) * 1.2
        one = torch.ones_like(xt)
        Mm = torch.stack([torch.stack([one, 0 * one, -xt], -1), torch.stack([0 * one, one, -yt], -1)], 1)
        An = (1 / zc)[:, None, None] * (Mm @ R @ Tm)                          # (n,2,2) tangent -> normalised
        Ac = torch.stack([-An[:, 0] * fx[:, None], -An[:, 1] * fy[:, None]], 1)
        J = Tm @ torch.linalg.inv(Ac)                                         # dP/d(canvas px)
        # rotate the tangent basis the solver uses: it must not matter -> give it a different N-consistent J
        Rp, Rm, ok, ndd = solve_frames(J, N, xt, yt, fx, fy, weak=(case == "weak"))
        gp, _, _ = err_deg(Rp, R, N); gm, _, _ = err_deg(Rm, R, N)
        best = torch.minimum(gp, gm)[ok]
        facing = ((R @ N[..., None])[..., 0] * F.normalize(torch.stack([xt, yt, one], -1), dim=-1)).sum(-1)
        print(f"[selftest] {case:5s}: ok {ok.float().mean():.4f} | best-of-2 geodesic: max {best.max():.2e} deg, "
              f"p99.9 {best.quantile(0.999):.2e} | other solution median {torch.maximum(gp, gm)[ok].median():.1f} deg "
              f"| n.d sign agrees with truth {((ndd > 0) == (facing > 0))[ok].float().mean():.4f}")
        assert best.quantile(0.999) < 1e-3, "solver failed"
        if case == "persp":
            Rp2, Rm2, ok2, _ = solve_frames(J, N, xt, yt, fx, fy, weak=True)
            g2 = torch.minimum(err_deg(Rp2, R, N)[0], err_deg(Rm2, R, N)[0])[ok2]
            print(f"[selftest] persp data solved with WEAK perspective: best-of-2 median {g2.median():.2f}, "
                  f"p90 {g2.quantile(0.9):.2f} deg (|x~|,|y~| up to 0.6)")
    # metric split
    ang = torch.rand(n, device=dev, dtype=torch.float64) * 170

    def rot_about(axis, deg):
        a = F.normalize(axis, dim=-1); t = torch.deg2rad(deg)[:, None, None]
        K = torch.zeros(len(a), 3, 3, dtype=a.dtype, device=a.device)
        K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -a[:, 2], a[:, 1], -a[:, 0]
        K = K - K.transpose(1, 2)
        return torch.eye(3, dtype=a.dtype, device=a.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K @ K

    g, nr, tw = err_deg(R @ rot_about(N, ang), R, N)
    print(f"[selftest] pure twist: |geo-ang| max {(g-ang).abs().max():.1e}, normal max {nr.max():.1e}, |twist-ang| max {(tw-ang).abs().max():.1e}")
    g, nr, tw = err_deg(R @ rot_about(T1, ang), R, N)
    print(f"[selftest] pure swing: |geo-ang| max {(g-ang).abs().max():.1e}, |normal-ang| max {(nr-ang).abs().max():.1e}, twist max {tw.max():.1e}")
    a2 = torch.rand(n, device=dev, dtype=torch.float64) * 60
    g, nr, tw = err_deg(R @ rot_about(T1, a2) @ rot_about(N, ang), R, N)
    print(f"[selftest] swing(a2) o twist(ang): |normal-a2| max {(nr-a2).abs().max():.1e}, |twist-ang| max {(tw-ang).abs().max():.1e}")
    # template/normals identical to the GT generator's
    from orient.rot6d import get_template
    vt_g, f_g, n_g = get_template("cpu")
    vt = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float64); fa = np.load(f"{ASSETS}/smplx_faces.npy")
    print(f"[selftest] dp_assets template vs GT generator template: max |dv| {np.abs(vt - vt_g).max():.2e} m, "
          f"faces equal {np.array_equal(fa, f_g)}, max |dn| {np.abs(vertex_normals(vt, fa) - n_g).max():.2e}")


def cmd_s2list():
    """Replicate indomain_probe's holdout sample stream (no model) -> the 640 uids + their MAMMA intrinsics."""
    ld = s2_native_loader()
    rows = []
    for i, b in enumerate(ld):
        if i >= 40:
            break
        for k in range(len(b["uid"])):
            rows.append({"shard": b["shard"][k], "key": b["key"][k], "uid": b["uid"][k], "seq": b["seq"][k]})
    print(f"[s2list] {len(rows)} samples from {len(set(r['shard'] for r in rows))} shards")
    # MAMMA source intrinsics
    need = {}
    for r in rows:
        chunk = r["shard"][:-4].rsplit("_", 1)[1]
        ds = r["shard"][:-4][: -(len(r["seq"]) + len(chunk) + 2)]
        _, cam, frame, _ = r["uid"].split("/")
        src = f"{MSRC}/{ds}/{r['seq']}/{chunk}.tar"
        need.setdefault(src, set()).add(f"/{r['seq']}/{cam}/{frame}.data.pyd")
        r["src"] = src; r["pyd"] = f"/{r['seq']}/{cam}/{frame}.data.pyd"
    Kmap = {}
    for src, want in need.items():
        with tarfile.open(src) as t:
            for m in t:
                for w in want:
                    if m.name.endswith(w):
                        d = pickle.load(t.extractfile(m))
                        Kmap[(src, w)] = np.asarray(d[sorted(d.keys())[0]]["cam_int"], np.float64).tolist()
        print(f"  {os.path.basename(os.path.dirname(src))}/{os.path.basename(src)}: {sum(1 for (s, _) in Kmap if s == src)}/{len(want)} frames")
    for r in rows:
        r["K"] = Kmap[(r["src"], r["pyd"])]
    json.dump(rows, open(f"{OUT}/s2_samples.json", "w"))
    print(f"[s2list] wrote {OUT}/s2_samples.json")


@torch.no_grad()
def s2_native_repro(model, dev, rows):
    """indomain_probe.evalc, verbatim math, on the same stream; also checks the stream is the recorded one."""
    ld = s2_native_loader()
    geo = np.zeros(2); geo_e = np.zeros(2); k = 0
    for i, b in enumerate(ld):
        if i >= 40:
            break
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(b["img"].to(dev))
        for bi in range(len(b["uid"])):
            assert b["uid"][bi] == rows[k]["uid"] and b["shard"][bi] == rows[k]["shard"], "stream mismatch"
            k += 1
            part, fg = b["part"][bi].to(dev), b["fg"][bi].to(dev)
            H, W = part.shape
            if not fg.any():
                continue
            rg = b["rot6d"][bi].to(dev).float(); rfg = b["rot_fg"][bi].to(dev)
            if rfg.any():
                Rh_, Rw_ = rfg.shape
                r_pp = F.interpolate(out["rot6d"][bi:bi + 1].float(), (Rh_, Rw_), mode="bilinear", align_corners=False)
                Rg = sixd_to_R(rg[rfg])

                def geo_of(part_map):
                    pm = F.interpolate(part_map[None, None].float(), (Rh_, Rw_), mode="nearest")[0, 0].long()
                    Rp = sixd_to_R(gather_rot6d(r_pp, pm[None])[0].permute(1, 2, 0)[rfg])
                    tr = torch.matmul(Rp.transpose(1, 2), Rg).diagonal(dim1=1, dim2=2).sum(1)
                    return (torch.arccos(((tr - 1) / 2).clamp(-1, 1)).sum() * 180 / np.pi).item()

                geo[0] += geo_of(part); geo[1] += rfg.sum().item()
                pred_full = F.interpolate(out["part"][bi:bi + 1].float(), (H, W), mode="bilinear",
                                          align_corners=False)[0].argmax(0) + 1
                geo_e[0] += geo_of(pred_full); geo_e[1] += rfg.sum().item()
    return {"n": k, "geo_oracle": geo[0] / geo[1], "geo_e2e": geo_e[0] / geo_e[1]}


def s2_gt512_batches(rows):
    """The recorded uids, decoded from the gt512 MAMMA shards (same basenames)."""
    dec = Dec(input_size=512, train=False, orient=True, uv_asset="smplx_face_uv_v2.npy", gt512=True)
    by_shard = {}
    for i, r in enumerate(rows):
        by_shard.setdefault(r["shard"], {})[r["key"]] = i
    samples = [None] * len(rows)
    for sh, keys in by_shard.items():
        parts = {}
        with tarfile.open(f"{M512}/{sh}") as t:
            for m in t:
                kk, ext = m.name.split(".", 1)
                if kk in keys:
                    parts.setdefault(kk, {})[ext] = t.extractfile(m).read()
        for kk, i in keys.items():
            if kk in parts and len(parts[kk]) == 4:
                p = parts[kk]
                samples[i] = dec((p["rgb.jpg"], p["corr.png"], p["meta.json"], p["rot.npy"], f"{M512}/{sh}", kk))
    miss = [rows[i]["uid"] for i, s in enumerate(samples) if s is None]
    keep = [i for i, s in enumerate(samples) if s is not None]
    for j in range(0, len(keep), 16):
        idx = keep[j:j + 16]
        yield idx, collate([samples[i] for i in idx]), miss


def cmd_run(a):
    dev = "cuda"
    ctx = Ctx(dev)
    ck = f"{RUNS}/{a.ckpt}"
    model = load_model(ck, dev)
    tag = a.ckpt.replace("/", "_").replace(".pt", "")
    acc = Acc()
    t0 = time.time()
    meta = {"split": a.split, "ckpt": ck, "protocol": f"{OUT}/PROTOCOL.txt"}
    uids = []
    if a.split == "S1":
        repro = []
        n = 0
        for bi, b in enumerate(s1_loader()):
            Ks = [canvas_K(bedlam_K(b["seq"][k], b["uid"][k]), b["gt_box"][k].numpy()) for k in range(len(b["uid"]))]
            eval_gt512_batch(model, ctx, b, Ks, acc, n, repro)
            uids += b["uid"]
            n += len(b["uid"])
            print(f"  batch {bi+1}  n={n}  eval_512 running geo_e2e {np.mean(repro):.3f}  ({time.time()-t0:.0f}s)", flush=True)
            if n >= (a.limit or 512):
                break
        meta["repro_eval512_geo_e2e"] = float(np.mean(repro)); meta["repro_batch_medians"] = repro
        meta["n_images"] = n
    else:
        rows = json.load(open(f"{OUT}/s2_samples.json"))
        if not a.limit:
            meta["repro_indomain_probe"] = s2_native_repro(model, dev, rows)
            print(f"  native probe repro: {meta['repro_indomain_probe']}  ({time.time()-t0:.0f}s)", flush=True)
        n = 0
        for idx, b, miss in s2_gt512_batches(rows[: (a.limit or len(rows))]):
            Ks = [canvas_K(np.asarray(rows[i]["K"]), b["gt_box"][k].numpy()) for k, i in enumerate(idx)]
            eval_gt512_batch(model, ctx, b, Ks, acc, n, None)
            uids += b["uid"]
            n += len(idx)
            print(f"  gt512 n={n} ({time.time()-t0:.0f}s)", flush=True)
        meta["n_images"] = n; meta["missing_in_gt512"] = miss
        hp = acc.extra
    meta["uids"] = uids
    meta["seconds"] = time.time() - t0
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{a.split}_{tag}{'_lim' + str(a.limit) if a.limit else ''}.npz"
    acc.save(path, meta)
    print(f"[run] wrote {path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "bake", "s2list", "run", "report", "floor"])
    ap.add_argument("--split", choices=["S1", "S2"])
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--r", type=int, default=3)
    a = ap.parse_args()
    if a.cmd == "floor":
        cmd_floor(a)
    elif a.cmd == "selftest":
        cmd_selftest()
    elif a.cmd == "bake":
        bake_normals()
    elif a.cmd == "s2list":
        cmd_s2list()
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "report":
        from orient.eval_frames_e1e6_report import main as rep
        rep()
