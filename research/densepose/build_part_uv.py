"""Build per-part DensePose UV charts on the SMPL-X mesh (the IUV "U,V" target).

For each of the 24 parts we flatten that part's canonical SMPL-X vertices to 2D by
PCA (top-2 principal axes) and normalize to [0,1]^2 -> a smooth, deterministic
chart per part. We then store, per face, the chart-UV of its 3 corner vertices,
computed in *that face's* part chart (so barycentric interpolation at train time
never mixes charts across a part boundary).

Output: ~/data/bedlam2/dp_assets/smplx_face_uv.npy   (20908, 3, 2) float32 in [0,1]
        ~/data/bedlam2/dp_assets/dp24_charts.npz      per-part (mean, axes, lo, hi)

These are charts built directly on SMPL-X (not the original DensePose charts, which
need the gated SMPL<->SMPL-X transfer). Same IUV representation; swappable later.
"""
import os
import sys

import numpy as np

sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
from render_correspondence import get_smplx

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")


def main():
    m = get_smplx("neutral", 16, "cpu")
    vt = m.v_template.detach().cpu().numpy().astype(np.float64)   # (10475,3)
    faces = m.faces_tensor.detach().cpu().numpy()                 # (20908,3)
    vert_part = np.load(f"{ASSETS}/smplx_vert_to_dp24.npy")       # (10475,)
    face_part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy")       # (20908,)

    F = len(faces)
    face_uv = np.zeros((F, 3, 2), np.float32)
    charts = {}
    for p in range(1, 25):
        vsel = np.where(vert_part == p)[0]
        X = vt[vsel]
        mu = X.mean(0)
        Xc = X - mu
        # top-2 principal axes
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        axes = Vt[:2]                                   # (2,3)
        # normalization range from ALL faces assigned to this part (so corner
        # verts that bleed in from neighbor parts still land inside [0,1])
        fsel = np.where(face_part == p)[0]
        vids = faces[fsel].reshape(-1)
        q = (vt[vids] - mu) @ axes.T                    # (3*nf, 2)
        lo, hi = q.min(0), q.max(0)
        span = np.where(hi - lo > 1e-6, hi - lo, 1.0)
        charts[p] = (mu, axes, lo, span)
        uv = np.clip((q - lo) / span, 0, 1).reshape(len(fsel), 3, 2)
        face_uv[fsel] = uv.astype(np.float32)

    np.save(f"{ASSETS}/smplx_face_uv.npy", face_uv)
    np.savez(f"{ASSETS}/dp24_charts.npz",
             **{f"p{p}_{k}": np.asarray(v) for p in charts
                for k, v in zip(("mu", "axes", "lo", "span"), charts[p])})

    # report
    print(f"wrote smplx_face_uv.npy  shape={face_uv.shape}")
    bg = face_uv[face_part == 0]
    print(f"U range [{face_uv[...,0].min():.3f},{face_uv[...,0].max():.3f}]  "
          f"V range [{face_uv[...,1].min():.3f},{face_uv[...,1].max():.3f}]")
    # per-part chart area coverage sanity (mean UV spread per part should be ~full)
    for p in (1, 2, 15, 23):
        fsel = face_part == p
        u = face_uv[fsel, :, 0]; v = face_uv[fsel, :, 1]
        print(f"  part {p:2d}: U[{u.min():.2f},{u.max():.2f}] V[{v.min():.2f},{v.max():.2f}]")


if __name__ == "__main__":
    main()
