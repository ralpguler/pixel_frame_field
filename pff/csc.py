"""Canonical Surface Coordinates: a DIFFERENTIABLE (part, u, v) -> template-space XYZ map.

Why: the UV loss is chart-relative. Every DP part's chart is PCA-projected and renormalised to [0,1]
by ITS OWN span, and those spans differ by 8x (torso_back 76 cm per uv-unit vs the smallest chart 9.5 cm).
So `smooth_l1` on UV silently weights the same surface error 8x differently across parts, and it is
discontinuous across chart seams (6.1% of template vertices touch >1 chart). Mapping (part,uv) -> XYZ
re-expresses the same prediction in METRES on the template, where the target is continuous everywhere.

How: bake, per part, a KxK texture of template XYZ over that part's UV chart (barycentric interpolation
of the chart triangles from smplx_face_uv + smplx_faces + v_template), then read it with grid_sample.
Bilinear on a barycentrically-rasterised texture reproduces barycentric interpolation and is
differentiable in (u,v) with an exact analytic Jacobian. Texels outside a chart are filled by
clipped-barycentric projection onto the nearest chart triangle, so X is finite and smooth everywhere
and never reads garbage.

  build:  python -m pff.csc --build -K 512     # writes assets/dp24_uv2xyz_K512.npz (the training table)
  check:  python -m pff.csc --check -K 512     # the three validation gates
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .assets import ASSETS
from .geometry import N_PARTS


def _bary2d(P, T):
    """Barycentric coords of P (M,2) in triangles T (M,3,2)."""
    v0 = T[:, 1] - T[:, 0]; v1 = T[:, 2] - T[:, 0]; v2 = P - T[:, 0]
    d00 = (v0 * v0).sum(1); d01 = (v0 * v1).sum(1); d11 = (v1 * v1).sum(1)
    d20 = (v2 * v0).sum(1); d21 = (v2 * v1).sum(1)
    den = d00 * d11 - d01 * d01
    den = np.where(np.abs(den) < 1e-20, 1e-20, den)
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    return np.stack([1 - v - w, v, w], 1)


def build(K=512, out=None, uv_asset="smplx_face_uv_v2.npy"):
    """Rasterise per-part XYZ over the chart.

    Also emits AMB (24,K,K): texels claimed by MORE THAN ONE triangle of the part. The chart is not
    injective everywhere (head: SMPL-X hides 1088 eyeball + 756 interior faces inside the part, and the
    ear/jaw genuinely folds; hands keep ~1% residual after the sheet split). At those texels (part,uv)
    does not determine a surface point, so the loss MUST skip them rather than train against an
    arbitrary branch. Returns tex (24,3,K,K) f32, cov (24,K,K) bool, amb (24,K,K) bool."""
    from scipy.spatial import cKDTree
    face_part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy").astype(int)      # (20908,) 1..24
    face_uv = np.load(f"{ASSETS}/{uv_asset}").astype(np.float64)             # (20908,3,2) corner UV
    faces = np.load(f"{ASSETS}/smplx_faces.npy")                             # (20908,3)
    vt = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float64)        # (10475,3) canonical

    g = (np.arange(K) + 0.5) / K                                             # texel CENTRES
    gu, gv = np.meshgrid(g, g, indexing="xy")                                # gu along W, gv along H
    P = np.stack([gu.ravel(), gv.ravel()], 1)                                # (K*K,2)

    tex = np.zeros((N_PARTS, 3, K, K), np.float32)
    cov = np.zeros((N_PARTS, K, K), bool)
    amb = np.zeros((N_PARTS, K, K), bool)
    for p in range(1, N_PARTS + 1):
        sel = face_part == p
        tuv = face_uv[sel]                                                   # (T,3,2)
        txyz = vt[faces[sel]]                                                # (T,3,3)
        tree = cKDTree(tuv.mean(1))
        kq = min(32, len(tuv))
        _, cand = tree.query(P, k=kq)
        cand = cand.reshape(len(P), kq)
        done = np.zeros(len(P), bool)
        nhit = np.zeros(len(P), int)                                          # how many triangles claim it
        bc = np.zeros((len(P), 3)); ti = np.zeros(len(P), int)
        for j in range(kq):                                                  # first triangle that CONTAINS it
            t = cand[:, j]
            b = _bary2d(P, tuv[t])
            ins = (b >= -1e-9).all(1)
            nhit += ins
            take = ins & ~done
            bc[take] = b[take]; ti[take] = t[take]; done |= take
        miss = ~done                                                         # outside the chart: clip onto
        if miss.any():                                                       # the nearest triangle
            t = cand[miss, 0]
            b = np.clip(_bary2d(P[miss], tuv[t]), 0, 1)
            b /= np.clip(b.sum(1, keepdims=True), 1e-12, None)
            bc[miss] = b; ti[miss] = t
        X = (bc[:, :, None] * txyz[ti]).sum(1)                               # (K*K,3) barycentric XYZ
        tex[p - 1] = X.reshape(K, K, 3).transpose(2, 0, 1).astype(np.float32)
        cov[p - 1] = done.reshape(K, K)
        bad = (nhit > 1).reshape(K, K) | ~done.reshape(K, K)
        d = bad.copy()                                                       # dilate by 1: the bilinear
        d[1:] |= bad[:-1]; d[:-1] |= bad[1:]                                 # stencil is 2x2, so a sample
        d[:, 1:] |= bad[:, :-1]; d[:, :-1] |= bad[:, 1:]                     # next to a bad texel blends it
        amb[p - 1] = d
        inside = done.mean()
        print(f"  part {p:2d}: {sel.sum():5d} faces | coverage {100*inside:5.1f}% | "
              f"unusable {100*d.mean():5.1f}% of the chart (ambiguous or boundary)", flush=True)

    out = out or f"{ASSETS}/dp24_uv2xyz_K{K}.npz"
    np.savez_compressed(out, tex=tex, cov=cov, amb=amb, K=K)
    print(f"[csc] wrote {out}  ({os.path.getsize(out)/1e6:.1f} MB) | unusable {100*amb[cov].mean():.1f}% "
          f"of covered chart", flush=True)
    return tex, cov, amb


class CSC(nn.Module):
    """(part, uv) -> canonical template XYZ, differentiable in uv.

    part: (...,) long in 1..24   uv: (2, ...) or (B,2,...) in [0,1]
    Sampling is ONE 3D grid_sample into a (1,3,24,K,K) volume: the z coordinate selects the part slice
    exactly (texel centre -> zero interpolation in z), x/y bilinearly interpolate the chart.

    MUST run in fp32. Under bf16 autocast a uv in [0,1] resolves to ~1/256 -- one texel -- and the
    gradient signal is destroyed. forward() disables autocast itself.
    """

    def __init__(self, K=512, path=None):
        super().__init__()
        path = path or f"{ASSETS}/dp24_uv2xyz_K{K}.npz"
        d = np.load(path)
        tex = torch.from_numpy(d["tex"]).float()                             # (24,3,K,K)
        self.K = int(d["K"])
        # -> (1, 3, 24, K, K): N=1, C=3, D=parts, H=v, W=u
        self.register_buffer("vol", tex.permute(1, 0, 2, 3).unsqueeze(0).contiguous())
        self.register_buffer("cov", torch.from_numpy(d["cov"]))
        amb = torch.from_numpy(d["amb"]).float()[:, None]                    # (24,1,K,K)
        self.register_buffer("ambvol", amb.permute(1, 0, 2, 3).unsqueeze(0).contiguous())  # (1,1,24,K,K)

    def _grid(self, uv, part):
        x = uv[0] * 2 - 1                                                    # align_corners=False: u=(j+.5)/K -> texel j
        y = uv[1] * 2 - 1
        # z hits the centre of slice (part-1) exactly: x_pix = ((z+1)*D - 1)/2 = part-1
        z = (2.0 * part.float() - 1.0) / N_PARTS - 1.0
        return torch.stack([x, y, z], -1)[None, None]                        # (1,1,H,W,3)

    def forward(self, uv, part):
        """uv (2,H,W) in [0,1], part (H,W) long 1..24  ->  X (3,H,W) in metres."""
        with torch.autocast(device_type=uv.device.type, enabled=False):
            X = F.grid_sample(self.vol, self._grid(uv.to(self.vol.dtype), part), mode="bilinear",
                              padding_mode="border", align_corners=False)    # (1,3,1,H,W)
            return X[0, :, 0]                                                # (3,H,W)

    @torch.no_grad()
    def unambiguous(self, uv, part):
        """True where (part,uv) DOES determine a surface point (nearest lookup into the amb texture)."""
        with torch.autocast(device_type=uv.device.type, enabled=False):
            a = F.grid_sample(self.ambvol, self._grid(uv.to(self.ambvol.dtype), part), mode="nearest",
                              padding_mode="border", align_corners=False)
            return a[0, 0, 0] < 0.5                                          # (H,W) bool


def xyz_grad_loss(Xp, Xg, m, scales=4, clip=0.05):
    """Multi-scale image-space gradient matching (MiDaS/MegaDepth-style), L1, in metres.

    Only edges whose BOTH pixels are foreground count. Gradients are clipped to `clip` m/px on both
    sides so self-occlusion boundaries (arm over torso: |dX| can reach 0.5 m between adjacent pixels)
    cannot dominate the mean -- the term is meant to shape the SMOOTH surface structure.
    """
    L = Xp.new_zeros(()); n = 0
    for _ in range(scales):
        if Xp.shape[-1] < 4 or Xp.shape[-2] < 4:
            break
        for d in (-1, -2):                                                   # x then y
            gp = torch.diff(Xp, dim=d); gg = torch.diff(Xg, dim=d)
            n_ = m.shape[d] - 1                                               # Xp is (3,H,W), m is (H,W):
            me = m.narrow(d, 0, n_) & m.narrow(d, 1, n_)                      # SAME negative dim. both ends valid.
            if not me.any():
                continue
            gp = gp.permute(1, 2, 0)[me]; gg = gg.permute(1, 2, 0)[me]
            if clip:
                gp = gp.clamp(-clip, clip); gg = gg.clamp(-clip, clip)
            L = L + (gp - gg).abs().mean(); n += 1
        Xp = Xp[:, ::2, ::2]; Xg = Xg[:, ::2, ::2]; m = m[::2, ::2]
    return L / max(n, 1)


# ------------------------------------------------------------------ validation gates
def _check(K=512, uv_asset="smplx_face_uv_v2.npy"):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    face_part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy").astype(int)
    face_uv = np.load(f"{ASSETS}/{uv_asset}").astype(np.float64)
    faces = np.load(f"{ASSETS}/smplx_faces.npy")
    vt = np.load(f"{ASSETS}/smplx_v_template.npy").astype(np.float64)
    csc = CSC(K).to(dev)

    # GATE 1 -- round trip. Points built exactly the way data.py does (random face + random barycentric).
    rng = np.random.RandomState(0)
    M = 200000
    f = rng.randint(0, len(faces), M)
    b = rng.dirichlet([1, 1, 1], M)
    uv = (b[:, :, None] * face_uv[f]).sum(1)                                 # (M,2)  <- data.py:133
    xyz = (b[:, :, None] * vt[faces[f]]).sum(1)                              # (M,3)  <- exact target
    part = face_part[f]
    U = torch.from_numpy(uv.T.astype(np.float32)).to(dev)[:, :, None]        # (2,M,1)
    P = torch.from_numpy(part).long().to(dev)[:, None]                       # (M,1)
    X = csc(U, P)[:, :, 0].T.double().cpu().numpy()
    ok = csc.unambiguous(U, P)[:, 0].cpu().numpy()                           # chart determines the point?
    e = np.linalg.norm(X - xyz, axis=1) * 1000                               # mm
    print(f"GATE 1  round-trip vs exact barycentric XYZ ({M} pts, K={K}, {uv_asset}):")
    print(f"   UNAMBIGUOUS ({100*ok.mean():.1f}% of samples -- these are what the loss trains on):")
    print(f"      median {np.median(e[ok]):.3f} mm | p99 {np.percentile(e[ok],99):.3f} mm | max {e[ok].max():.3f} mm")
    print(f"   ambiguous   ({100*(~ok).mean():.1f}% -- MASKED OUT of the loss):")
    if (~ok).any():
        print(f"      median {np.median(e[~ok]):.3f} mm | max {e[~ok].max():.3f} mm  (the chart folds here)")
    g1 = e[ok].max() < 1.0
    print(f"      -> {'PASS' if g1 else 'FAIL'} (need max < 1 mm on the unambiguous set)")
    for nmp, ps in (("hands", (3, 4)), ("feet", (5, 6)), ("head", (23, 24))):
        m = np.isin(part, ps)
        print(f"      {nmp:6s}: {100*ok[m].mean():5.1f}% usable | err on usable: "
              f"median {np.median(e[m & ok]):.3f} mm, max {e[m & ok].max():.3f} mm")

    # GATE 2 -- autograd vs finite differences. The interpolant is piecewise-LINEAR, so a coarse step
    # straddles texel kinks and FD != analytic by construction. Use float64 on CPU with a sub-texel step.
    c64 = CSC(K).double()
    n = 4096
    U = torch.from_numpy(uv[:n].T).double()[:, :, None].requires_grad_(True)
    P = torch.from_numpy(part[:n]).long()[:, None]
    g = torch.randn(3, n, 1, dtype=torch.float64)
    (c64(U, P) * g).sum().backward()
    ana = U.grad.clone()
    h = 1e-7                                                                  # 2.6e-5 of a texel
    num = torch.zeros_like(ana)
    with torch.no_grad():
        for c in range(2):
            Up = U.detach().clone(); Up[c] += h
            Um = U.detach().clone(); Um[c] -= h
            num[c] = ((c64(Up, P) - c64(Um, P)) * g).sum(0) / (2 * h)
    d = (ana - num).abs()
    rel = (d.median() / num.abs().median()).item()
    print(f"GATE 2  autograd vs finite-diff d(X)/d(uv) (float64, h={h}): median rel err {rel:.2e} "
          f"-> {'PASS' if rel < 1e-4 else 'FAIL'}")

    # GATE 3 -- GT uv through the loss must be ~0 on the unambiguous set
    H = 256
    f = rng.randint(0, len(faces), H * H)
    b = rng.dirichlet([1, 1, 1], H * H)
    uv = (b[:, :, None] * face_uv[f]).sum(1).T.reshape(2, H, H)
    xyz = (b[:, :, None] * vt[faces[f]]).sum(1).T.reshape(3, H, H)
    U = torch.from_numpy(uv.astype(np.float32)).to(dev)
    Xg = torch.from_numpy(xyz.astype(np.float32)).to(dev)
    P = torch.from_numpy(face_part[f].reshape(H, H)).long().to(dev)
    m = csc.unambiguous(U, P)
    Xp = csc(U, P)
    l1 = (Xp - Xg).abs().permute(1, 2, 0)[m].mean().item() * 1000
    lg = xyz_grad_loss(Xp, Xg, m).item() * 1000
    print(f"GATE 3  GT uv through the loss: L_xyz {l1:.4f} mm | L_grad {lg:.4f} mm "
          f"-> {'PASS' if l1 < 1.0 else 'FAIL'}  (L_grad is nonzero by design: neighbouring pixels here are")
    print(f"        random faces, not a rendered surface, so the GT gradient is huge and both sides clip)")
    return g1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("-K", type=int, default=512)
    a = ap.parse_args()
    if a.build:
        build(a.K)
    if a.check:
        _check(a.K)
