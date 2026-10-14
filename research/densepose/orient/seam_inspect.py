"""High-res seam inspector: for ONE held-out instance render large panels —
RGB | GT normal | Pred normal (pred UV, seam-free kNN N_template) | GEODESIC ERROR (R_pred vs R_gt).
The geodesic-error panel uses NO N_template, so if a seam shows there it's a real rotation discontinuity;
if it's clean at island boundaries, the (former) normal seam was purely N_template reconstruction.
Usage: seam_inspect.py [--ckpt ...] [--idx N]
"""
import os, io, glob, argparse, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from data import unpack_rgba, STORED_EXPAND
from model import DINOv3DPT, part_pred, gather_rot6d
from orient.rot6d import get_template, sixd_to_R_np, per_pixel_R
from orient.template_normals import TemplateNormals, normal_to_rgb, denoise_part
import webdataset as wds

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_scratch_bedlam_orient/ckpt_072000.pt"))
ap.add_argument("--idx", type=int, default=6)
ap.add_argument("--knn", type=int, default=4)
a = ap.parse_args()
dev = "cuda"; Z = 512
v_template, faces, n_rest = get_template("cuda")
A = os.path.expanduser("~/data/bedlam2/dp_assets")
vpart = np.load(f"{A}/smplx_vert_to_dp24.npy").astype(int); vuv = np.load(f"{A}/smplx_vert_uv.npy")
TREES = {p: (np.where(vpart == p)[0], cKDTree(vuv[vpart == p])) for p in range(1, 25) if (vpart == p).any()}
INTERP3D = {p: LinearNDInterpolator(vuv[vpart == p], v_template[vpart == p]) for p in range(1, 25) if (vpart == p).sum() >= 4}
GTREE = cKDTree(v_template)
fuv = np.load(f"{A}/smplx_face_uv.npy")              # (F,3,2) per-face-corner UV (handles straddling faces)
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, faces, n_rest)      # GT-faithful, seam-free N_template
_sd = torch.load(a.ckpt, map_location="cpu")["model"]
_rpp = _sd["rot_head.2.weight"].shape[0] != 6                          # auto-detect per-part vs single
m = DINOv3DPT(backbone="vit_large_patch16_dinov3", layers=(4, 11, 17, 23), pretrained=False, decoupled=True, orient=True, rot_per_part=_rpp).to(dev).eval()
m.load_state_dict(_sd)


@torch.no_grad()
def predict(crop, cs):
    img = torch.from_numpy(cv2.resize(crop, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)                    # predicted part @ Z
    pS = cv2.resize(part.astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int)
    r, c = np.indices(pS.shape); idx = np.clip(pS - 1, 0, 23)
    # resize the per-part VOLUMES first, then gather at cs. Gathering then bilinear-resizing the gathered
    # map blends adjacent parts' chart coords / rotations across every boundary -> seam at each part edge.
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])  # (24,2,cs,cs)
    U = uvv[idx, 0, r, c]; Vv = uvv[idx, 1, r, c]
    rot = o["rot6d"].float()[0]                                         # (6 or 144, Z, Z)
    if rot.shape[0] == 6:                                               # single shared head
        rt = rot.permute(1, 2, 0).cpu().numpy()
        six = np.concatenate([cv2.resize(rt[..., :3], (cs, cs)), cv2.resize(rt[..., 3:], (cs, cs))], -1)
    else:                                                              # per-part: resize 24x6 volume, gather
        rot = rot.view(24, 6, Z, Z).cpu().numpy()
        rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
        six = rotv[idx, :, r, c]                                        # (cs,cs,6)
    R = sixd_to_R_np(six)
    return pS, U, Vv, R


def nt_iuv(part, U, V, knn):
    fgp = part > 0; P = np.zeros((*part.shape, 3), np.float32)
    for p, (vidx, tree) in TREES.items():
        s = part == p
        if not s.any():
            continue
        q = np.stack([U[s], V[s]], 1); pts = INTERP3D[p](q) if p in INTERP3D else np.full((len(q), 3), np.nan, np.float32)
        bad = np.isnan(pts[:, 0])
        if bad.any():
            _, nn = tree.query(q[bad]); pts[bad] = v_template[vidx[nn]]
        P[s] = pts
    nt = np.zeros((*part.shape, 3), np.float32)
    if fgp.any():
        d, vn = GTREE.query(P[fgp], k=knn)
        if knn == 1:
            nt[fgp] = n_rest[vn]
        else:
            w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)
            nn = (w[..., None] * n_rest[vn]).sum(1); nt[fgp] = nn / np.clip(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-9, None)
    return nt, fgp


def ncam(R, nt):
    nc = np.einsum("hwij,hwj->hwi", R, nt)
    return nc / np.clip(np.linalg.norm(nc, axis=-1, keepdims=True), 1e-9, None)


def nrm(R, nt, mask):
    return normal_to_rgb(ncam(R, nt), mask)         # conventional view-space (facing=blue, up=green)


def nangle(a, b, mm):
    d = np.clip((a * b).sum(-1), -1, 1); r = np.degrees(np.arccos(d)); r[~mm] = np.nan; return r


shards = sorted(glob.glob(os.path.expanduser("~/data/bedlam2/dp_shards_rot/val/*.tar")))
k = 0
for s in wds.WebDataset(shards, shardshuffle=False, empty_check=False, handler=wds.warn_and_continue):
    rgb = cv2.imdecode(np.frombuffer(s["rgb.jpg"], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    face, bary = unpack_rgba(np.array(__import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(s["corr.png"]))))
    vr = np.load(io.BytesIO(s["rot.npy"])).astype(np.float32)
    S0 = rgb.shape[0]; cs = max(8, min(int(round(1.3 / STORED_EXPAND * S0)), S0)); o = (S0 - cs) // 2
    rc = np.ascontiguousarray(rgb[o:o + cs, o:o + cs]); fw = np.ascontiguousarray(face[o:o + cs, o:o + cs]); bw = np.ascontiguousarray(bary[o:o + cs, o:o + cs])
    mask = fw >= 0
    if mask.mean() < 0.03:
        continue
    if k != a.idx:
        k += 1; continue
    tv = faces[np.where(mask, fw, 0)]; ntg = (bw[..., None] * n_rest[tv]).sum(2); ntg /= np.clip(np.linalg.norm(ntg, axis=-1, keepdims=True), 1e-9, None)
    R_gt = per_pixel_R(fw, bw, vr, faces, mask)
    pS, U, Vv, R_pr = predict(rc, cs)
    ntPR, mPR = TN(pS, U, Vv)                            # predicted N_template (face-bary, projection fix)
    # camera normals + errors
    ng = ncam(R_gt, ntg); npr = ncam(R_pr, ntPR)
    nerr = nangle(npr, ng, mask & mPR)                  # normal error (uses pred IUV + pred R)
    rel = np.matmul(np.transpose(R_pr, (0, 1, 3, 2)), R_gt); tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    geo = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))); geo[~mask] = np.nan    # rotation-only error
    fig, ax = plt.subplots(1, 5, figsize=(26, 6.5))
    ax[0].imshow(rc); ax[0].set_title("RGB")
    ax[1].imshow(nrm(R_gt, ntg, mask)); ax[1].set_title("GT normal")
    ax[2].imshow(nrm(R_pr, ntPR, mPR)); ax[2].set_title("PRED normal (camera)")
    hn = ax[3].imshow(nerr, cmap="turbo", vmin=0, vmax=45); ax[3].set_title(f"NORMAL err° (mean {np.nanmean(nerr):.1f})")
    hg = ax[4].imshow(geo, cmap="turbo", vmin=0, vmax=45); ax[4].set_title(f"ROTATION err° (mean {np.nanmean(geo):.1f})")
    for h, x in [(hn, ax[3]), (hg, ax[4])]:
        fig.colorbar(h, ax=[x], fraction=0.046)
    for x in ax: x.axis("off")
    out = os.path.expanduser("~/codes/wan/densepose/dpeval/seam_inspect.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out, "| normal-err %.1f  rotation-err %.1f" % (np.nanmean(nerr), np.nanmean(geo)))
    break
