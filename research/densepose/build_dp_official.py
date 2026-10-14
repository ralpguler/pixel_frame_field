"""Transfer the OFFICIAL detectron2 DensePose IUV (24 parts + per-part UV charts,
defined on the SMPL mesh) onto the SMPL-X mesh used by our renders.

Inputs (in dp_assets/):
  UV_Processed.mat        official DensePose: SMPL faces, part-per-face, per-vertex UV
  smplx_to_smpl.pkl       official SMPL-X<->SMPL transfer ('matrix' 6890x10475)

Method: reconstruct SMPL template in the SMPL-X frame (smpl_v = matrix @ smplx_v),
then for each SMPL-X vertex find the closest SMPL face and read its DensePose part +
barycentric-interpolated UV. Aggregate to per-SMPL-X-face part (majority) and per-
corner UV. Outputs (overwrite the PCA versions; dataloader/model unchanged):
  smplx_vert_to_dp24.npy (10475,)  smplx_face_to_dp24.npy (20908,)
  smplx_vert_uv.npy (10475,2)      smplx_face_uv.npy (20908,3,2)
  dp24_lr_swap.npy (25,)           dp24_names.json
"""
import json
import os
import pickle
import sys

import numpy as np
import scipy.io as sio
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
from render_correspondence import get_smplx

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")

# Official DensePose-24 names (Detectron2 DensePose chart order, I = 1..24).
DP24_NAMES = {
    0: "background", 1: "torso_back", 2: "torso_front", 3: "hand_right",
    4: "hand_left", 5: "foot_left", 6: "foot_right", 7: "upperleg_right_back",
    8: "upperleg_left_back", 9: "upperleg_right_front", 10: "upperleg_left_front",
    11: "lowerleg_right_back", 12: "lowerleg_left_back", 13: "lowerleg_right_front",
    14: "lowerleg_left_front", 15: "upperarm_left_front", 16: "upperarm_right_front",
    17: "upperarm_left_back", 18: "upperarm_right_back", 19: "lowerarm_left_front",
    20: "lowerarm_right_front", 21: "lowerarm_left_back", 22: "lowerarm_right_back",
    23: "head_right", 24: "head_left",
}
# L<->R swap of part ids under horizontal flip (official DensePose Index symmetry).
SWAP_PAIRS = [(3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14),
              (15, 16), (17, 18), (19, 20), (21, 22), (23, 24)]


def closest_bary(p, tri):
    """p (...,3), tri (...,3,3) -> clamped barycentric (...,3) and dist (...)."""
    a, b, c = tri[..., 0, :], tri[..., 1, :], tri[..., 2, :]
    v0, v1, v2 = b - a, c - a, p - a
    d00 = (v0 * v0).sum(-1); d01 = (v0 * v1).sum(-1); d11 = (v1 * v1).sum(-1)
    d20 = (v2 * v0).sum(-1); d21 = (v2 * v1).sum(-1)
    den = d00 * d11 - d01 * d01 + 1e-12
    v = (d11 * d20 - d01 * d21) / den
    w = (d00 * d21 - d01 * d20) / den
    bc = np.stack([1 - v - w, v, w], -1)
    bc = np.clip(bc, 0, None)
    bc = bc / bc.sum(-1, keepdims=True).clip(1e-9)
    proj = (bc[..., None] * tri).sum(-2)
    return bc, np.linalg.norm(p - proj, axis=-1)


def main():
    m = get_smplx("neutral", 16, "cpu")
    smplx_v = m.v_template.detach().cpu().numpy().astype(np.float64)     # (10475,3)
    smplx_faces = m.faces_tensor.detach().cpu().numpy()                 # (20908,3)

    mat = pickle.load(open(f"{ASSETS}/smplx_to_smpl.pkl", "rb"),
                      encoding="latin1")["matrix"]                      # (6890,10475)
    smpl_v = mat @ smplx_v                                              # (6890,3)

    uvp = sio.loadmat(f"{ASSETS}/UV_Processed.mat")
    all_v = uvp["All_vertices"].reshape(-1).astype(np.int64) - 1        # (7829,) ->6890
    all_f = uvp["All_Faces"].astype(np.int64) - 1                       # (13774,3)->7829
    fpart = uvp["All_FaceIndices"].reshape(-1).astype(np.int64)         # (13774,) 1..24
    uvn = np.stack([uvp["All_U_norm"].reshape(-1),
                    uvp["All_V_norm"].reshape(-1)], -1)                 # (7829,2)
    smpl_faces = all_v[all_f]                                           # (13774,3) ->6890
    smpl_fuv = uvn[all_f]                                              # (13774,3,2) chart-consistent

    tri = smpl_v[smpl_faces]                                           # (13774,3,3)
    cent = tri.mean(1)
    tree = cKDTree(cent)

    def transfer_uv(points, part_of_point):
        """For each point, project onto the nearest SMPL face RESTRICTED to its part
        and read barycentric-interpolated chart UV. Guarantees chart consistency."""
        uv = np.zeros((len(points), 2), np.float32)
        for p in range(1, 25):
            sel = np.where(part_of_point == p)[0]
            pf = np.where(fpart == p)[0]                               # SMPL faces in chart p
            if len(sel) == 0 or len(pf) == 0:
                continue
            k = min(8, len(pf))
            _, cnd = cKDTree(cent[pf]).query(points[sel], k=k)
            cnd = cnd[:, None] if k == 1 else cnd
            gids = pf[cnd]                                             # (n,k) global SMPL face
            bc, dist = closest_bary(points[sel][:, None, :].repeat(k, 1), tri[gids])
            ar = np.arange(len(sel)); bk = dist.argmin(1)
            fb = gids[ar, bk]; bcb = bc[ar, bk]
            uv[sel] = (bcb[..., None] * smpl_fuv[fb]).sum(1)
        return uv

    # per-SMPL-X-VERTEX part (nearest SMPL face) -> for viz + hflip-consistency
    _, nfv = tree.query(smplx_v, k=1)
    vert_part = fpart[nfv].astype(np.int16)                           # (10475,)
    vert_uv = transfer_uv(smplx_v, vert_part).astype(np.float32)

    # per-SMPL-X-FACE part from face centroid (stable at boundaries), then each face's
    # 3 corner verts get UV within THAT face's chart -> chart-consistent corner UVs.
    cent_x = smplx_v[smplx_faces].mean(1)                             # (20908,3)
    _, nff = tree.query(cent_x, k=1)
    face_part = fpart[nff].astype(np.int64)                           # (20908,)
    corner_pts = smplx_v[smplx_faces].reshape(-1, 3)                  # (3*20908,3)
    corner_part = np.repeat(face_part, 3)
    corner_uv = transfer_uv(corner_pts, corner_part)
    smplx_face_uv = corner_uv.reshape(len(smplx_faces), 3, 2).astype(np.float32)
    dist = np.zeros((1, 1))                                           # (reporting placeholder)

    swap = np.arange(25, dtype=np.int8)
    for i, j in SWAP_PAIRS:
        swap[i], swap[j] = j, i

    np.save(f"{ASSETS}/smplx_vert_to_dp24.npy", vert_part.astype(np.int8))
    np.save(f"{ASSETS}/smplx_face_to_dp24.npy", face_part.astype(np.int16))
    np.save(f"{ASSETS}/smplx_vert_uv.npy", vert_uv)
    np.save(f"{ASSETS}/smplx_face_uv.npy", smplx_face_uv)
    np.save(f"{ASSETS}/dp24_lr_swap.npy", swap)
    json.dump(DP24_NAMES, open(f"{ASSETS}/dp24_names.json", "w"), indent=1)

    print(f"SMPL recon from SMPL-X: {smpl_v.shape}; chart-consistent per-face UV")
    print("=== per-part SMPL-X vertex counts (official DensePose-24) ===")
    for idx in range(1, 25):
        print(f"  {idx:2d} {DP24_NAMES[idx]:22s} verts={int((vert_part==idx).sum()):5d}"
              f"  faces={int((face_part==idx).sum()):5d}")
    print("U range", vert_uv[:, 0].min(), vert_uv[:, 0].max(),
          "V range", vert_uv[:, 1].min(), vert_uv[:, 1].max())


if __name__ == "__main__":
    main()
