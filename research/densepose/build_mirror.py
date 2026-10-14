"""Build an exact left-right mirror map on the SMPL-X mesh so horizontal-flip
augmentation is correct for BOTH part labels and UV.

vert_mirror[v]   = vertex whose canonical position is the mirror (-x,y,z) of v
face_mirror[f]   = the face made of the mirrored vertices of face f
corner_perm[f,k] = position within face_mirror[f] of the mirror of corner k of f
                   (so barycentric weights can be re-ordered onto the mirror face)

Output: ~/data/bedlam2/dp_assets/smplx_mirror.npz {vert_mirror, face_mirror, corner_perm}
"""
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
from render_correspondence import get_smplx

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")


def main():
    m = get_smplx("neutral", 16, "cpu")
    vt = m.v_template.detach().cpu().numpy()
    faces = m.faces_tensor.detach().cpu().numpy()
    F = len(faces)

    tree = cKDTree(vt)
    mirror_pos = vt.copy(); mirror_pos[:, 0] *= -1
    dist, vert_mirror = tree.query(mirror_pos)         # nearest vert to (-x,y,z)
    print(f"vert_mirror: max match dist={dist.max():.5f} (should be ~0)")

    key2f = {frozenset(int(x) for x in faces[f]): f for f in range(F)}
    face_mirror = np.arange(F)
    corner_perm = np.tile([0, 1, 2], (F, 1))
    miss = 0
    for f in range(F):
        a, b, c = faces[f]
        ma, mb, mc = vert_mirror[a], vert_mirror[b], vert_mirror[c]
        g = key2f.get(frozenset((int(ma), int(mb), int(mc))))
        if g is None:
            miss += 1
            continue
        face_mirror[f] = g
        gv = list(int(x) for x in faces[g])
        corner_perm[f] = [gv.index(int(x)) for x in (ma, mb, mc)]
    print(f"face_mirror: {miss}/{F} faces had no exact mirror (kept identity)")

    np.savez(f"{ASSETS}/smplx_mirror.npz",
             vert_mirror=vert_mirror.astype(np.int32),
             face_mirror=face_mirror.astype(np.int32),
             corner_perm=corner_perm.astype(np.int8))
    print(f"wrote {ASSETS}/smplx_mirror.npz")


if __name__ == "__main__":
    main()
