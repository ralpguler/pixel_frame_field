"""Visualize what the combined BEDLAM2+MAMMA training dataloader yields: per sample,
RGB | GT IUV (part) | GT normal map (R_gt . N_template). Pull a few from each source (labeled)."""
import os, glob, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from data import build_loader, SHARDS_ROT, SHARDS_ROT_MAMMA
from model import sixd_to_R
from orient.template_normals import TemplateNormals, normal_to_rgb
from orient.rot6d import get_template

A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cpu"); faces = np.asarray(faces); n_rest = np.asarray(n_rest)
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, faces, n_rest)


def denorm(img):  # img (3,H,W) -> HWC uint8
    x = img.permute(1, 2, 0).numpy()
    if x.max() > 1.5 or x.min() < -0.5:            # imagenet-normalized
        x = x * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    return np.clip(x, 0, 1)


def gt_normal(gt):
    import cv2
    part, uv, fg, rot6d, rot_fg = gt
    Rh, Rw = rot6d.shape[:2]                         # rot GT res (capped, e.g. 384)
    part_r = cv2.resize(part.numpy().astype(np.uint8), (Rw, Rh), interpolation=cv2.INTER_NEAREST).astype(int)
    U = cv2.resize(uv[0].numpy(), (Rw, Rh)); V = cv2.resize(uv[1].numpy(), (Rw, Rh))
    fgm = rot_fg.numpy() > 0
    nt, _ = TN(part_r, U, V)                         # rest-pose normal from GT part,U,V @ rot res
    R = sixd_to_R(rot6d.float()).numpy()             # (Rh,Rw,3,3)
    ncam = np.einsum("hwij,hwj->hwi", R, nt)
    return normal_to_rgb(ncam, fgm)


def iuv_rgb(gt):
    part = gt[0].numpy(); fg = (gt[2].numpy() > 0)
    import matplotlib.cm as cm
    out = (cm.hsv(np.clip(part, 0, 24) / 24.0)[..., :3])
    out[~fg] = 1.0
    return out


rows = []
for label, shards in [("BEDLAM2", sorted(glob.glob(f"{SHARDS_ROT}/train/shard-*.tar"))[:3]),
                      ("MAMMA", sorted(glob.glob(f"{SHARDS_ROT_MAMMA}/train/*.tar"))[:3])]:
    ld = build_loader("train", batch=3, workers=2, input_size=512, orient=True, shards=shards)
    b = next(iter(ld))
    for i in range(3):
        rows.append((label, denorm(b["img"][i]), iuv_rgb(b["gt"][i]), gt_normal(b["gt"][i])))

fig, ax = plt.subplots(len(rows), 3, figsize=(9, 3 * len(rows)))
for r, (lab, rgb, iuv, nrm) in enumerate(rows):
    ax[r, 0].imshow(rgb); ax[r, 0].set_ylabel(lab, fontsize=12)
    ax[r, 1].imshow(iuv); ax[r, 2].imshow(nrm)
    if r == 0:
        ax[r, 0].set_title("RGB (loader input)"); ax[r, 1].set_title("GT IUV (part)"); ax[r, 2].set_title("GT normal (R.N)")
    for c in range(3): ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
out = os.path.expanduser("~/data/datasets/dp_shards_rot_mamma/loader_viz.png")
plt.tight_layout(); fig.savefig(out, dpi=90, bbox_inches="tight"); print("wrote", out)
