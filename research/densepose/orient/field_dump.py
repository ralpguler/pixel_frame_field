"""FORWARD per-pixel field dump for the direction-viz beauty contest.

Every visualization in the contest composes with the prediction f: pixel -> (template point, frame);
none of them inverts it (no template->pixel search, so none of the blinking/density pathology of the
tracked-point overlays). This script runs the model once per frame and stores everything those
renderers need, per pixel of the tracked crop:

  mask     model's own foreground
  part     DP part 1..24 (0=bg), majority-denoised
  uv       predicted chart UV (2,H,W)
  n_cam    camera-space outward surface normal  = R_pixel @ n_template(part,uv)
  t_cam    camera-space combing tangent         = R_pixel @ t_template(part,uv)
  twist    the gauge DoF as in the overlay viz (hue channel)

t_template is the gradient of geodesic-distance-from-the-crown, projected on the tangent plane --
the body-intrinsic "downhill" direction that flows down the torso AND out along the arms (gravity
stalls on horizontal limbs; this was established in the flow-mode work). Camera basis is OpenCV-ish:
x right, y DOWN in image terms after the y-flip used by all our overlays (image dir of camera v is
(v[0], -v[1])); camera-facing normals have n_cam[2] < 0.

Usage:
  field_dump.py CLIP --out DIR --video --maxframes 150      # per-frame NPZ + JPG + meta.json
  field_dump.py CLIP --out DIR --frame 75                   # single frame
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model import DINOv3DPT, part_pred, backbone_from_state_dict, default_layers
from orient.detect_track import detect_persons
from orient.rot6d import get_template, sixd_to_R_np
from orient.template_normals import TemplateNormals, denoise_part

ap = argparse.ArgumentParser()
D = os.path.expanduser("~/data/bedlam2/dp_runs")
A = os.path.expanduser("~/data/bedlam2/dp_assets")
ap.add_argument("clip")
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default=f"{D}/dp_pp_llrd_1e3_mamma/ckpt_120000.pt")
ap.add_argument("--frame", type=int, default=75)
ap.add_argument("--video", action="store_true")
ap.add_argument("--maxframes", type=int, default=150)
ap.add_argument("--smooth", type=int, default=15)
ap.add_argument("--exp", type=float, default=1.3)
ap.add_argument("--uv-asset", default="smplx_face_uv.npy",
                help="chart asset MATCHING THE CKPT's training charts: smplx_face_uv.npy for models "
                     "trained on v1 (e.g. ckpt_120000), smplx_face_uv_v2.npy for the dp_csc_* runs -- "
                     "a mismatch corrupts the template-normal lookup on hands/feet")
ap.add_argument("--tfield", default="smooth", choices=["smooth", "geo"],
                help="template combing: 'smooth' = globally-optimal direction field "
                     "(smplx_tangent_smooth.npy; ~46 singular pts at soles/fingertips/ears, no cut-locus "
                     "creases); 'geo' = old geodesic-from-crown gradient (a critical point at EVERY "
                     "extremity + C0 creases -- its defects read as model artifacts)")
a = ap.parse_args()
dev = "cuda"; Z = 512
os.makedirs(a.out, exist_ok=True)

v_template, faces, n_rest = get_template("cuda")
vt = v_template.detach().cpu().numpy() if torch.is_tensor(v_template) else np.asarray(v_template)
FA = np.asarray(faces)
nrest = n_rest.detach().cpu().numpy() if torch.is_tensor(n_rest) else np.asarray(n_rest)
fuv = np.load(f"{A}/{a.uv_asset}"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)


def geodesic_tangent():
    """Per-vertex unit tangent = grad(geodesic distance from crown), projected on the tangent plane.
    Same field the flow-mode viz validated: down the torso/legs, OUT along the arms to the fingertips."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    E = set()
    for t in FA:
        for i in range(3):
            u_, v_ = int(t[i]), int(t[(i + 1) % 3]); E.add((min(u_, v_), max(u_, v_)))
    ei = np.array(sorted(E))
    w = np.linalg.norm(vt[ei[:, 0]] - vt[ei[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[ei[:, 0], ei[:, 1]], np.r_[ei[:, 1], ei[:, 0]])),
                   shape=(len(vt),) * 2)
    crown = int(np.argmax(vt[:, 1]))                        # y-up SMPL template: crown = highest vertex
    geo = dijkstra(G.tocsr(), indices=crown)
    geo[~np.isfinite(geo)] = geo[np.isfinite(geo)].max()
    p0, p1, p2 = vt[FA[:, 0]], vt[FA[:, 1]], vt[FA[:, 2]]
    fnv = np.cross(p1 - p0, p2 - p0)
    A2 = np.linalg.norm(fnv, axis=1, keepdims=True)
    un = fnv / np.clip(A2, 1e-12, None)
    g0, g1, g2 = geo[FA[:, 0]], geo[FA[:, 1]], geo[FA[:, 2]]
    fgrad = (g0[:, None] * np.cross(un, p2 - p1) + g1[:, None] * np.cross(un, p0 - p2)
             + g2[:, None] * np.cross(un, p1 - p0)) / np.clip(A2, 1e-12, None)
    T = np.zeros_like(vt)
    for k in range(3):
        np.add.at(T, FA[:, k], fgrad * (A2 / 2))            # area-weighted face->vertex average
    T -= (T * nrest).sum(1, keepdims=True) * nrest          # project onto the tangent plane
    T /= np.clip(np.linalg.norm(T, axis=1, keepdims=True), 1e-9, None)
    return T


T_tmpl = np.load(f"{A}/smplx_tangent_smooth.npy").astype(np.float64) if a.tfield == "smooth" \
    else geodesic_tangent()
TN_nrm = TemplateNormals(fuv, fpart, FA, nrest)
TN_tan = TemplateNormals(fuv, fpart, FA, T_tmpl)

sd = torch.load(a.ckpt, map_location="cpu")["model"]
rpp = sd["rot_head.2.weight"].shape[0] != 6
m = DINOv3DPT(backbone=backbone_from_state_dict(sd), layers=default_layers(backbone_from_state_dict(sd)), pretrained=False,
              decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
m.load_state_dict(sd)


@torch.no_grad()
def predict(crop, cs):
    img = torch.from_numpy(cv2.resize(crop, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = cv2.resize(part_pred(o)[0].cpu().numpy().astype(np.uint8), (cs, cs),
                      interpolation=cv2.INTER_NEAREST).astype(int)
    part = denoise_part(part)
    r, c = np.indices(part.shape); idx = np.clip(part - 1, 0, 23)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])
    US = uvv[idx, 0, r, c]; VS = uvv[idx, 1, r, c]
    rot = o["rot6d"].float()[0].view(24, 6, Z, Z).cpu().numpy()
    rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
    six = rotv[idx, :, r, c]; R = sixd_to_R_np(six)
    return part, US, VS, R


def detect_box(rgb):
    dets = sorted(detect_persons(rgb, thr=0.5), key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    if not dets:
        return None
    x0, y0, x1, y1 = dets[0][:4]
    return np.array([(x0+x1)/2, (y0+y1)/2, max(x1-x0, y1-y0)*a.exp/2])


def smooth_boxes(B, win):
    F = len(B)
    for c in range(3):
        col = B[:, c]; mnan = np.isnan(col)
        if mnan.all():
            return None
        col[mnan] = np.interp(np.flatnonzero(mnan), np.flatnonzero(~mnan), col[~mnan])
    if win > 1:
        r = int(win) // 2
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / (win / 4.0)) ** 2); k /= k.sum()
        Bp = np.pad(B, ((r, r), (0, 0)), mode="edge")
        B = np.stack([np.convolve(Bp[:, c], k, mode="valid")[:F] for c in range(3)], 1)
    B[:, 2] = np.median(B[:, 2])
    return B


def dump_frame(rgb, box, fi):
    cx, cy, s = box
    H, W = rgb.shape[:2]
    side = int(2*s); X0, Y0 = int(cx-s), int(cy-s)
    pad = ((max(0, -Y0), max(0, Y0+side-H)), (max(0, -X0), max(0, X0+side-W)), (0, 0))
    crop = np.pad(rgb, pad, mode="edge")[max(0, Y0):Y0+side+pad[0][0], max(0, X0):X0+side+pad[1][0]]
    cs = crop.shape[0]
    part, US, VS, R = predict(crop, cs)
    nt, fg = TN_nrm(part, US, VS)
    tt, _ = TN_tan(part, US, VS)
    tt -= (tt * nt).sum(-1, keepdims=True) * nt             # orthonormal on the TEMPLATE...
    tt /= np.clip(np.linalg.norm(tt, axis=-1, keepdims=True), 1e-9, None)
    n_cam = np.einsum("hwij,hwj->hwi", R, nt).astype(np.float32)
    t_cam = np.einsum("hwij,hwj->hwi", R, tt).astype(np.float32)   # ...hence orthonormal in camera
    twist = ((np.arctan2(R[..., 1, 0], R[..., 0, 0]) + np.pi) / (2*np.pi)).astype(np.float32)
    np.savez_compressed(f"{a.out}/f_{fi:04d}.npz",
                        mask=fg, part=part.astype(np.uint8),
                        uv=np.stack([US, VS]).astype(np.float16),
                        n_cam=n_cam.astype(np.float16), t_cam=t_cam.astype(np.float16),
                        twist=twist.astype(np.float16))
    cv2.imwrite(f"{a.out}/f_{fi:04d}.jpg", crop[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cs, float(fg.mean())


cap = cv2.VideoCapture(a.clip)
fps = cap.get(5) or 16
frames_idx = list(range(min(a.maxframes, int(cap.get(7))))) if a.video else [a.frame]

B = np.full((len(frames_idx), 3), np.nan)
for j, fi in enumerate(frames_idx):
    cap.set(1, fi); ok, bgr = cap.read()
    if not ok:
        frames_idx = frames_idx[:j]; B = B[:j]; break
    b = detect_box(np.ascontiguousarray(bgr[..., ::-1]))
    if b is not None:
        B[j] = b
Bs = smooth_boxes(B, a.smooth if len(frames_idx) > 1 else 1)
if Bs is None:
    raise SystemExit("no person detected anywhere")
print(f"[box] {len(frames_idx)} frames, constant crop side={2*Bs[0,2]:.0f}px", flush=True)

done = []
for j, fi in enumerate(frames_idx):
    cap.set(1, fi); ok, bgr = cap.read()
    if not ok:
        break
    cs, fgm = dump_frame(np.ascontiguousarray(bgr[..., ::-1]), Bs[j], fi)
    done.append(fi)
    if j % 10 == 0:
        print(f"frame {fi}: side={cs} fg={100*fgm:.1f}%", flush=True)
cap.release()
json.dump({"clip": os.path.basename(a.clip), "frames": done, "fps": fps,
           "side": int(2*Bs[0, 2]), "ckpt": a.ckpt},
          open(f"{a.out}/meta.json", "w"))
print(f"DUMP DONE -> {a.out} ({len(done)} frames)", flush=True)
