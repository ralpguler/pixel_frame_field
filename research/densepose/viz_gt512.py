"""Visualise the rebuilt fixed-canvas 512 GT, and what the OLD pipeline was actually feeding the model.

FIG 1  the new GT, per sample:
   network input | GT canvas (image warped through theta) | part | IUV | canonical XYZ | surface normal
   Everything after the first panel lives in the GT canvas and is used VERBATIM -- never resampled.

FIG 2  the bug that was fixed. The old loader bilinearly resized the 6D ROTATION field to ROT_CAP=384.
   Averaging the 6D vectors of two DIFFERENT surfaces across an occlusion boundary -- an arm in front of
   a torso -- and Gram-Schmidt-ing the blend produces a frame belonging to neither; where the blend lands
   near zero the Gram-Schmidt runs on noise and the rotation is arbitrary. Reproduced here on the new
   canvas (512 -> 384 bilinear, vs an exact 384 resample of face+barycentrics):
   exact normals | old-pipeline normals | geodesic error (deg)
   The error map lights up exactly on the occlusion boundaries MAMMA exists to teach.

  python viz_gt512.py --n 5
"""
import argparse
import glob
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import webdataset as wds

from data import DPDecode, ASSETS, _PAL
from orient.rot6d import sixd_to_R_np

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--out", default=os.path.expanduser("~/data/dp_viz_compare/gt512"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

vt = np.load(f"{ASSETS}/smplx_v_template.npy")
LO, SP = vt.min(0), (vt.max(0) - vt.min(0))
faces = np.load(f"{ASSETS}/smplx_faces.npy")


def n_rest():
    n = np.zeros_like(vt)
    p = vt[faces]
    fn = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    for k in range(3):
        np.add.at(n, faces[:, k], fn)
    return n / np.clip(np.linalg.norm(n, axis=1, keepdims=True), 1e-9, None)


NR = n_rest()
dec = DPDecode(input_size=512, train=False, orient=True, gt512=True, uv_asset="smplx_face_uv_v2.npy")


def normals(rot6, face, bary, fg):
    """GT rotation field -> the outward surface normal in camera space (R @ n_rest), as RGB."""
    out = np.zeros(fg.shape + (3,), np.uint8)
    if not fg.any():
        return out
    R = sixd_to_R_np(rot6[fg].astype(np.float64))                 # (n,3,3) rest -> camera
    nr = (bary[fg][:, :, None] * NR[faces[face[fg]]]).sum(1)      # rest normal at the pixel
    nc = np.einsum("nij,nj->ni", R, nr)
    nc /= np.clip(np.linalg.norm(nc, axis=1, keepdims=True), 1e-9, None)
    out[fg] = ((nc * 0.5 + 0.5) * 255).astype(np.uint8)
    return out


def geo_map(A, B, fg):
    d = np.zeros(fg.shape, np.float32)
    if fg.any():
        Ra, Rb = sixd_to_R_np(A[fg].astype(np.float64)), sixd_to_R_np(B[fg].astype(np.float64))
        tr = np.einsum("nij,nij->n", Ra, Rb)
        d[fg] = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
    return d


def heat(v, fg, vmax):
    t = np.clip(v / vmax, 0, 1)
    c = (np.stack([np.clip(1.5 - np.abs(4 * t - 3), 0, 1),
                   np.clip(1.5 - np.abs(4 * t - 2), 0, 1),
                   np.clip(1.5 - np.abs(4 * t - 1), 0, 1)], -1) * 255).astype(np.uint8)
    c[~fg] = 18
    return c


def lab(im, s):
    im = np.ascontiguousarray(im)
    cv2.putText(im, s, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(im, s, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 1, cv2.LINE_AA)
    return im


SRC = [("BEDLAM", sorted(glob.glob(os.path.expanduser("~/data/bedlam2/dp_shards_512/train/*.tar")))[:2]),
       ("MAMMA", [t for t in sorted(glob.glob(os.path.expanduser(
           "~/data/datasets/dp_shards_512_mamma/train/*.tar"))) if "contact" in t][:2])]

r1, r2 = [], []
for name, tars in SRC:
    ds = wds.WebDataset(tars, empty_check=False).to_tuple("rgb.jpg", "corr.png", "meta.json", "rot.npy")
    got = 0
    for sm in ds:
        o = dec(sm)
        fg = o["fg"].numpy()
        if fg.mean() < 0.06:
            continue
        from data import unpack_rgba
        from PIL import Image
        import io as _io
        face, bary = unpack_rgba(np.array(Image.open(_io.BytesIO(sm[1]))))
        img = (o["img"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        g = F.affine_grid(o["theta"][None], (1, 3, 512, 512), align_corners=False)
        wimg = (F.grid_sample(o["img"][None], g, align_corners=False)[0]
                .permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        part = o["part"].numpy(); uv = o["uv"].numpy(); xyz = o["xyz"].numpy().astype(np.float32)
        rot6 = o["rot6d"].numpy().astype(np.float32)

        pc = _PAL[np.clip(part, 0, 24)]; pc[~fg] = 18
        iuv = np.zeros((512, 512, 3), np.uint8)
        iuv[fg] = np.stack([uv[0][fg] * 255, uv[1][fg] * 255, np.full(fg.sum(), 150)], 1).astype(np.uint8)
        noc = np.zeros((512, 512, 3), np.uint8)
        noc[fg] = (np.clip((xyz[fg] - LO) / SP, 0, 1) * 255).astype(np.uint8)
        nrm = normals(rot6, face, bary, fg)

        r1.append(np.concatenate([lab(img, f"{name} input"), lab(wimg, "GT canvas"), lab(pc, "part"),
                                  lab(iuv, "IUV"), lab(noc, "canonical XYZ"), lab(nrm, "normal")], 1))

        # ---- FIG 2: what the OLD loader did to the rotation labels ----
        R = 384
        old = np.concatenate([cv2.resize(rot6[..., :3], (R, R), interpolation=cv2.INTER_LINEAR),
                              cv2.resize(rot6[..., 3:], (R, R), interpolation=cv2.INTER_LINEAR)], -1)
        fr = cv2.resize(face.astype(np.float32), (R, R), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        br = cv2.resize(bary, (R, R), interpolation=cv2.INTER_NEAREST)
        efg = fr >= 0
        # EXACT 6D at 384: recompute from face+bary (never blends two surfaces) -- the correct way
        ex = np.zeros((R, R, 6), np.float32)
        v6 = np.load(_io.BytesIO(sm[3])).astype(np.float32)      # per-vertex rot6d table
        if efg.any():
            ex[efg] = (br[efg][:, :, None] * v6[faces[fr[efg]]]).sum(1)
        d = geo_map(old, ex, efg)
        r2.append(np.concatenate([
            lab(cv2.resize(normals(ex, fr, br, efg), (512, 512), interpolation=cv2.INTER_NEAREST),
                f"{name}: EXACT normals"),
            lab(cv2.resize(normals(old, fr, br, efg), (512, 512), interpolation=cv2.INTER_NEAREST),
                "OLD pipeline (bilinear rot)"),
            lab(cv2.resize(heat(d, efg, 30.0), (512, 512), interpolation=cv2.INTER_NEAREST),
                f"error (0-30 deg) | >30deg on {100*(d[efg]>30).mean():.1f}% px")], 1))
        got += 1
        if got >= a.n // 2 + a.n % 2:
            break

cv2.imwrite(f"{a.out}/fig1_new_gt.png", np.concatenate(r1, 0)[..., ::-1])
cv2.imwrite(f"{a.out}/fig2_old_rot_bug.png", np.concatenate(r2, 0)[..., ::-1])
print(f"wrote {a.out}/fig1_new_gt.png   ({len(r1)} samples)")
print(f"wrote {a.out}/fig2_old_rot_bug.png ({len(r2)} samples)")
