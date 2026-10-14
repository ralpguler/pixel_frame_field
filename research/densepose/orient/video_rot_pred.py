"""Oriented-DensePose on VIDEO: per frame, detect people, predict IUV + rotation, and composite the
camera-space surface ORIENTATION. No GT on real video -> we show predictions only:
  [ original | predicted normal (R·N) | predicted twist (deg) ]
N_template per pixel comes from the model's OWN predicted IUV (part,U,V -> nearest SMPL-X vertex ->
rest normal); R from the predicted 6D. Output mp4 (H.264).

Usage: video_rot_pred.py IN.mp4 OUT.mp4 [--ckpt ...] [--max-frames N]
"""
import os, sys, argparse, numpy as np, cv2, torch, imageio
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model import DINOv3DPT, part_pred, gather_rot6d
from orient.template_normals import TemplateNormals, normal_to_rgb
from orient.rot6d import get_template, sixd_to_R_np
from orient.detect_track import detect_persons, Tracker   # RT-DETRv2 + IoU/EMA tracking (demo detector)

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_120k/ckpt_120000.pt"))
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--stride", type=int, default=1)        # subsample frames (e.g. 3DPW 30fps -> stride 2)
ap.add_argument("--fps", type=float, default=30.0)      # output fps when input is an image folder
ap.add_argument("--det-thr", type=float, default=0.4)
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cuda")                      # (10475,3) rest normals, faces, template
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, np.asarray(faces), np.asarray(n_rest))   # GT-faithful seam-free N_template

_sd = torch.load(a.ckpt, map_location="cpu")["model"]
_rpp = _sd["rot_head.2.weight"].shape[0] != 6                          # auto-detect per-part vs single
m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
              pretrained=False, decoupled=True, orient=True, rot_per_part=_rpp).to(dev).eval()
m.load_state_dict(_sd); print("loaded", a.ckpt, "rot_per_part" if _rpp else "rot_single", flush=True)


@torch.no_grad()
def infer(crop):
    img = torch.from_numpy(crop).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)                   # (Z,Z) 0..24, faithful predicted part
    pt = torch.from_numpy(part).to(dev)[None]
    uv = o["uv"].float()[0].view(24, 2, Z, Z)
    sel = (pt[0] - 1).clamp(0)
    U = uv[:, 0].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    V = uv[:, 1].gather(0, sel[None])[0].clamp(0, 1).cpu().numpy()
    six = gather_rot6d(o["rot6d"], pt).float()[0].permute(1, 2, 0).cpu().numpy()  # gather by pred part -> (Z,Z,6)
    return part, U, V, six


def orient_maps(part, U, V, six):
    """-> normal RGB (Z,Z,3 uint8), twist deg (Z,Z), fg mask. N_template via face-bary (TemplateNormals)."""
    R = sixd_to_R_np(six)                                             # (Z,Z,3,3)
    nt, mask = TN(part, U, V)                                         # (Z,Z,3) template normal, fg
    ncam = np.einsum("hwij,hwj->hwi", R, nt)                          # camera normal
    nrgb = normal_to_rgb(ncam, mask)                                  # conventional (facing=blue, up=green)
    # twist about predicted N_template
    tw = np.full((Z, Z), np.nan, np.float32)
    if mask.any():
        q = Rotation.from_matrix(R[mask].reshape(-1, 3, 3)).as_quat()
        proj = (q[:, :3] * nt[mask]).sum(1)
        tau = 2.0 * np.arctan2(proj, q[:, 3])
        tw[mask] = np.degrees((tau + np.pi) % (2 * np.pi) - np.pi)
    return (nrgb * 255).astype(np.uint8), tw, mask


def twist_rgb(tw, mask):
    import matplotlib.cm as cm
    t = np.clip((tw + 180) / 360, 0, 1); rgb = (cm.twilight(t)[..., :3] * 255).astype(np.uint8)
    rgb[~mask] = 255
    return rgb


import matplotlib.cm as _cm
_PAL = (_cm.hsv(np.linspace(0, 1, 25))[:, :3] * 255).astype(np.uint8)   # 24 distinct part colors (idx0=bg)


def iuv_rgb(part, U, V):
    """DensePose IUV viz: each pixel colored by its part (the dense correspondence / chart), brightness
    modulated by U and V so the continuous within-part coordinates are visible too."""
    mask = part > 0
    img = _PAL[np.clip(part, 0, 24)].astype(np.float32)
    img *= (0.45 + 0.4 * U + 0.15 * V)[..., None]          # texture by continuous U,V within each chart
    out = np.clip(img, 0, 255).astype(np.uint8); out[~mask] = 255
    return out


import glob as _glob
is_dir = os.path.isdir(a.inp)
if is_dir:                                                            # image-folder input (e.g. 3DPW)
    frames = sorted(_glob.glob(os.path.join(a.inp, "*.jpg")) + _glob.glob(os.path.join(a.inp, "*.png")))
    frames = frames[::a.stride]
    N = len(frames); fps = a.fps / a.stride
    H, W = cv2.imread(frames[0]).shape[:2]
else:
    cap = cv2.VideoCapture(a.inp); fps = (cap.get(5) or 16) / a.stride
    W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7)) // a.stride
if a.max_frames:
    N = min(N, a.max_frames)
vw = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
                        macro_block_size=16, ffmpeg_log_level="error")
tracker = Tracker()
i = 0
while i < N:
    if is_dir:
        bgr = cv2.imread(frames[i]); ok = bgr is not None
    else:
        for _ in range(a.stride):
            ok, bgr = cap.read()
            if not ok:
                break
    if not ok:
        break
    rgb = bgr[..., ::-1].copy()
    boxp = rgb.copy(); iuvov = rgb.copy(); nov = rgb.copy()
    for x0, y0, x1, y1 in tracker.update(detect_persons(rgb, thr=a.det_thr, dev=dev)):
        cv2.rectangle(boxp, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)   # show detection+tracking
        bw, bh = x1 - x0, y1 - y0; cx, cy = x0 + bw / 2, y0 + bh / 2; side = EXP * max(bw, bh); S = int(round(side))
        ix0, iy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        canv = np.zeros((S, S, 3), np.uint8)
        ax0, ay0 = max(0, -ix0), max(0, -iy0); bx0, by0 = max(0, ix0), max(0, iy0)
        bx1, by1 = min(W, ix0 + S), min(H, iy0 + S)
        canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
        part, U, V, six = infer(cv2.resize(canv, (Z, Z)))
        nrgb, tw, mask = orient_maps(part, U, V, six)
        irgb = iuv_rgb(part, U, V)                                                     # IUV (UV correspondence)
        nS = cv2.resize(nrgb, (S, S), interpolation=cv2.INTER_NEAREST)
        iS = cv2.resize(irgb, (S, S), interpolation=cv2.INTER_NEAREST)
        mS = cv2.resize(mask.astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST) > 0
        sx, sy = bx0 - ix0, by0 - iy0; ph, pw = by1 - by0, bx1 - bx0
        fg = mS[sy:sy + ph, sx:sx + pw]
        iuvov[by0:by1, bx0:bx1][fg] = iS[sy:sy + ph, sx:sx + pw][fg]
        nov[by0:by1, bx0:bx1][fg] = nS[sy:sy + ph, sx:sx + pw][fg]
    vw.append_data(np.ascontiguousarray(np.concatenate([boxp, iuvov, nov], 1)))
    i += 1
    if i % 25 == 0:
        print(f"  {i}/{N}", flush=True)
if not is_dir:
    cap.release()
vw.close()
print(f"wrote {a.out} ({i} frames, {3*W}x{H} @ {fps:.0f}fps) [orig | IUV | pred normal]", flush=True)
