"""Rotation PREDICTION vs GT visualization on held-out BEDLAM (dp_shards_rot/val), + per-pixel error.

Per test instance (val-style 1.3x crop): run the orientation model, reconstruct GT R from the stored
vert_rot6d, and render aligned panels:
  RGB | GT normal (R_gt·N) | Pred normal (R_pred·N) | geodesic error (deg) heatmap |
      | GT twist (deg) | Pred twist (deg)
Normal = the 2-DoF facing; twist = the gauge-free 3rd DoF; error = arccos((tr(R_predᵀR_gt)-1)/2).
-> dpeval/rot_pred_viz/index.html
"""
import os, io, glob, json, html, argparse, numpy as np, cv2, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from data import unpack_rgba, STORED_EXPAND
from model import DINOv3DPT, part_pred, gather_rot6d
from orient.rot6d import get_template, sixd_to_R_np, per_pixel_R
from orient.template_normals import TemplateNormals, normal_to_rgb, denoise_part
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_scratch_bedlam_orient/ckpt_038000.pt"))
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--n", type=int, default=8)
ap.add_argument("--z", type=int, default=512)
ap.add_argument("--denoise", type=int, default=0)   # part-map majority-filter window (0=off; coherent part errors -> doesn't help, can erode thin parts)
a = ap.parse_args()
dev = "cuda"; Z = a.z
OUT = os.path.expanduser("~/codes/wan/densepose/dpeval/rot_pred_viz"); os.makedirs(OUT + "/imgs", exist_ok=True)
v_template, faces, n_rest = get_template("cuda")
_A = os.path.expanduser("~/data/bedlam2/dp_assets")
_vpart = np.load(f"{_A}/smplx_vert_to_dp24.npy").astype(int)
_vuv = np.load(f"{_A}/smplx_vert_uv.npy")
TREES = {p: (np.where(_vpart == p)[0], cKDTree(_vuv[_vpart == p])) for p in range(1, 25) if (_vpart == p).any()}
# per-part UV->3D (barycentric, continuous within a chart) + GLOBAL 3D vertex tree -> seam-free N_template
INTERP3D = {p: LinearNDInterpolator(_vuv[_vpart == p], v_template[_vpart == p])
            for p in range(1, 25) if (_vpart == p).sum() >= 4}
GTREE = cKDTree(v_template)
_fuv = np.load(f"{_A}/smplx_face_uv.npy")            # (F,3,2) per-face-corner UV (handles straddling faces)
_fpart = np.load(f"{_A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(_fuv, _fpart, faces, n_rest)    # GT-faithful, seam-free N_template

_sd = torch.load(a.ckpt, map_location="cpu")["model"]
_rpp = _sd["rot_head.2.weight"].shape[0] != 6                          # auto-detect per-part vs single
m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
              pretrained=False, decoupled=True, orient=True, rot_per_part=_rpp).to(dev).eval()
m.load_state_dict(_sd)
print("loaded", a.ckpt, "rot_per_part" if _rpp else "rot_single", flush=True)


def twist_deg(R, nt, mask):
    out = np.full(R.shape[:2], np.nan, np.float32)
    if not mask.any():
        return out
    q = Rotation.from_matrix(R[mask].reshape(-1, 3, 3)).as_quat()
    proj = (q[:, :3] * nt[mask]).sum(1)
    tau = 2.0 * np.arctan2(proj, q[:, 3])
    out[mask] = np.degrees((tau + np.pi) % (2 * np.pi) - np.pi)
    return out


@torch.no_grad()
def predict(crop_rgb, cs):
    """Return PREDICTED part (cs,cs), U,V (cs,cs), R (cs,cs,3,3) — fully from the model."""
    img = torch.from_numpy(cv2.resize(crop_rgb, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)
    if a.denoise > 0:
        part = denoise_part(part, a.denoise)
    partS = cv2.resize(part.astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int)
    r, c = np.indices(partS.shape); idx = np.clip(partS - 1, 0, 23)
    # resize per-part VOLUMES, then gather at cs (gather-then-resize blends charts/rotations at seams)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])
    US = uvv[idx, 0, r, c]; VS = uvv[idx, 1, r, c]
    rot = o["rot6d"].float()[0]
    if rot.shape[0] == 6:
        rt = rot.permute(1, 2, 0).cpu().numpy()
        six = np.concatenate([cv2.resize(rt[..., :3], (cs, cs)), cv2.resize(rt[..., 3:], (cs, cs))], -1)
    else:
        rot = rot.view(24, 6, Z, Z).cpu().numpy()
        rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
        six = rotv[idx, :, r, c]
    R = sixd_to_R_np(six)
    return partS, US, VS, R


def nt_from_iuv(part, U, V):
    """Seam-free N_template from predicted (part,U,V): -> continuous 3D canonical point (per-part
    barycentric; adjacent UV islands meet in 3D, so no chart-boundary jump) -> GLOBAL nearest template
    vertex -> rest normal. (Per-chart nearest-vertex jumps between UV islands -> the bad seams.)"""
    fgp = part > 0
    P = np.zeros((*part.shape, 3), np.float32)
    for p, (vidx, tree) in TREES.items():
        s = part == p
        if not s.any():
            continue
        q = np.stack([U[s], V[s]], 1)
        pts = INTERP3D[p](q) if p in INTERP3D else np.full((len(q), 3), np.nan, np.float32)
        bad = np.isnan(pts[:, 0])
        if bad.any():                                         # UV outside the chart hull -> nearest vertex
            _, nn = tree.query(q[bad]); pts[bad] = v_template[vidx[nn]]
        P[s] = pts
    nt = np.zeros((*part.shape, 3), np.float32)
    if fgp.any():
        d, vn = GTREE.query(P[fgp], k=4)                      # k-NN inverse-distance weighted ->
        w = 1.0 / (d + 1e-6); w /= w.sum(1, keepdims=True)    # CONTINUOUS in P (no Voronoi step / thin seam)
        nn = (w[..., None] * n_rest[vn]).sum(1)
        nt[fgp] = nn / np.clip(np.linalg.norm(nn, axis=-1, keepdims=True), 1e-9, None)
    return nt, fgp


def normal_rgb(R, nt, mask):
    nc = np.einsum("hwij,hwj->hwi", R, nt)          # camera-space normal
    return normal_to_rgb(nc, mask)                  # conventional view-space map (facing=blue, up=green)


shards = sorted(glob.glob(os.path.expanduser("~/data/bedlam2/dp_shards_rot/val/*.tar")))
import webdataset as wds
ds = wds.WebDataset(shards, shardshuffle=False, empty_check=False, handler=wds.warn_and_continue)
rows = []
for s in ds:
    if len(rows) >= a.n:
        break
    try:
        rgb = cv2.imdecode(np.frombuffer(s["rgb.jpg"], np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
        face, bary = unpack_rgba(np.array(__import__("PIL.Image", fromlist=["Image"]).open(io.BytesIO(s["corr.png"]))))
        vr = np.load(io.BytesIO(s["rot.npy"])).astype(np.float32)
        S0 = rgb.shape[0]
        cs = int(round((1.3 / STORED_EXPAND) * S0)); cs = max(8, min(cs, S0))
        x0 = (S0 - cs) // 2; y0 = (S0 - cs) // 2                          # centered val crop
        rc = np.ascontiguousarray(rgb[y0:y0 + cs, x0:x0 + cs])
        fw = np.ascontiguousarray(face[y0:y0 + cs, x0:x0 + cs])
        bw = np.ascontiguousarray(bary[y0:y0 + cs, x0:x0 + cs])
        mask = fw >= 0
        if mask.mean() < 0.03:
            continue
        # per-pixel template normal
        tv = faces[np.where(mask, fw, 0)]
        nt = (bw[..., None] * n_rest[tv]).sum(2)
        nt = nt / np.clip(np.linalg.norm(nt, axis=-1, keepdims=True), 1e-9, None)
        R_gt = per_pixel_R(fw, bw, vr, faces, mask)                       # (cs,cs,3,3)
        part_pr, U_pr, V_pr, R_pr = predict(rc, cs)
        nt_pr, mask_pr = TN(part_pr, U_pr, V_pr)                          # face+bary N_template (pred IUV)
        # geodesic error (deg)
        rel = np.matmul(np.transpose(R_pr, (0, 1, 3, 2)), R_gt)
        tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
        err = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))); err[~mask] = np.nan
        meanerr = float(np.nanmean(err))
        # panels
        fig, ax = plt.subplots(1, 6, figsize=(20, 3.6))
        ax[0].imshow(rc); ax[0].set_title("RGB", fontsize=9)
        ax[1].imshow(normal_rgb(R_gt, nt, mask)); ax[1].set_title("GT normal", fontsize=9)
        ax[2].imshow(normal_rgb(R_pr, nt_pr, mask_pr)); ax[2].set_title("Pred normal (pred UV)", fontsize=9)
        he = ax[3].imshow(err, cmap="turbo", vmin=0, vmax=45); ax[3].set_title(f"error° (mean {meanerr:.1f})", fontsize=9)
        fig.colorbar(he, ax=[ax[3]], fraction=0.046, pad=0.02)
        tg = twist_deg(R_gt, nt, mask); tp = twist_deg(R_pr, nt_pr, mask_pr)
        ax[4].imshow(tg, cmap="twilight", vmin=-180, vmax=180); ax[4].set_title("GT twist°", fontsize=9)
        tw = ax[5].imshow(tp, cmap="twilight", vmin=-180, vmax=180); ax[5].set_title("Pred twist°", fontsize=9)
        fig.colorbar(tw, ax=[ax[5]], fraction=0.046, pad=0.02)
        for x in ax: x.axis("off")
        fn = f"imgs/{len(rows):02d}.png"; fig.savefig(OUT + "/" + fn, dpi=70, bbox_inches="tight"); plt.close(fig)
        rows.append((fn, meanerr, json.loads(s["meta.json"]).get("uid", s["__key__"])))
        print("rendered", fn, "mean err %.1f deg" % meanerr, flush=True)
    except Exception as e:
        print("skip", s.get("__key__", "?"), repr(e), flush=True)

H = ['<html><head><meta charset=utf-8><style>body{font-family:sans-serif;background:#fff}img{width:100%;max-width:1500px}'
     '.it{border-bottom:1px solid #ccc;padding:6px 0}</style></head><body>']
H.append(f"<h2>Rotation prediction vs GT — held-out BEDLAM ({os.path.basename(a.ckpt)})</h2>")
H.append("<p>RGB | <b>GT normal</b> (R·N) | <b>Pred normal</b> | <b>geodesic error</b> (deg, turbo 0-45) | "
         "<b>GT twist</b> | <b>Pred twist</b> (deg, cyclic). Normal = facing (2 DoF); twist = gauge-free 3rd DoF.</p>")
mErr = np.mean([r[1] for r in rows]) if rows else float("nan")
H.append(f"<p>mean geodesic error over shown instances: <b>{mErr:.1f}°</b></p>")
for fn, e, uid in rows:
    H.append(f"<div class=it>{html.escape(str(uid))} — mean err {e:.1f}°<br><img src='{fn}'></div>")
H.append("</body></html>")
open(OUT + "/index.html", "w").write("\n".join(H))
print(f"wrote {OUT}/index.html ({len(rows)} instances, mean err {mErr:.1f} deg)", flush=True)
