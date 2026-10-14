"""Per-frame correspondence visualization as VIDEO: for each frame, paint every foreground pixel's
RGB onto the SMPL-X template via the predicted (part, U, V) using BARYCENTRIC interpolation
(same as dp_surface_paint.py), then render the template from a FIXED frontal orthographic view.
Body pose is fixed across frames, so a temporally STABLE painted texture = consistent correspondences;
flicker = correspondence jitter. Output: side-by-side [input crop | frontal painted template] mp4.

Usage: video_template_paint.py IN.mp4 OUT.mp4 [--ckpt ...] [--max-frames N]
"""
import os, sys, argparse, numpy as np, cv2, torch, imageio
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DINOv3DPT, part_pred
import smplx
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_ft_joint/ckpt_003000.pt"))
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--res", type=int, default=720)
a = ap.parse_args()
dev = "cuda"; Z = 512; A = os.path.expanduser("~/data/bedlam2/dp_assets")

# SMPL-X template + per-vertex part/UV
sm = smplx.create(os.path.expanduser("~/data/smplx/models_lockedhead"), model_type="smplx",
                  gender="neutral", num_betas=16, use_pca=False)
V = sm.v_template.detach().cpu().numpy().astype(np.float32)               # (10475,3)
vpart = np.load(f"{A}/smplx_vert_to_dp24.npy").astype(int)
vuv = np.load(f"{A}/smplx_vert_uv.npy")
trees = {p: (np.where(vpart == p)[0], cKDTree(vuv[vpart == p])) for p in range(1, 25) if (vpart == p).any()}
interp = {p: LinearNDInterpolator(vuv[vpart == p], V[vpart == p]) for p in trees if (vpart == p).sum() >= 4}

# fixed frontal orthographic projection of the template (x right, y up->flip, z depth toward camera)
R = a.res
xmn, xmx = V[:, 0].min(), V[:, 0].max(); ymn, ymx = V[:, 1].min(), V[:, 1].max()
pad = 0.12 * max(xmx - xmn, ymx - ymn)
span = max(xmx - xmn, ymx - ymn) + 2 * pad
PX = ((V[:, 0] - (xmn + xmx) / 2) / span + 0.5) * R
PY = (0.5 - (V[:, 1] - (ymn + ymx) / 2) / span) * R                       # flip y: head up
PX = np.clip(PX.astype(int), 0, R - 1); PY = np.clip(PY.astype(int), 0, R - 1)
ZB = V[:, 2]                                                              # larger z = nearer camera (front)
order_static = np.argsort(ZB)                                            # back -> front

det = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT).eval().to(dev)
m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
              pretrained=False, decoupled=True).to(dev).eval()
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
print(f"loaded {a.ckpt}", flush=True)


def largest_person(rgb):
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().to(dev) / 255
    with torch.no_grad():
        o = det([t])[0]
    bs = [b.cpu().numpy() for b, l, s in zip(o["boxes"], o["labels"], o["scores"])
          if int(l) == 1 and float(s) > 0.85]
    return max(bs, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bs else None


def crop_square(rgb, box, exp=1.3):
    x0, y0, x1, y1 = box; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = int(exp * max(x1 - x0, y1 - y0))
    H, W = rgb.shape[:2]; sx0, sy0 = int(cx - s / 2), int(cy - s / 2); canv = np.zeros((s, s, 3), np.uint8)
    ax0, ay0 = max(0, -sx0), max(0, -sy0); bx0, by0 = max(0, sx0), max(0, sy0)
    bx1, by1 = min(W, sx0 + s), min(H, sy0 + s)
    canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
    return cv2.resize(canv, (Z, Z))


@torch.no_grad()
def infer(crop):
    img = torch.from_numpy(crop).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)
    uv = o["uv"].float()[0].cpu().numpy().reshape(24, 2, Z, Z)
    return part, uv


# faint static template (gray) as the backdrop body shape, z-buffered
def base_canvas():
    cv = np.zeros((R, R, 3), np.uint8)
    cv[PY[order_static], PX[order_static]] = (60, 60, 60)
    return cv


def paint_frame(crop, part, uv):
    """Map each fg pixel -> per-vertex color (nearest vertex in its part's UV chart), z-buffer to frontal."""
    vcol = np.zeros((len(V), 3), np.float32); vcnt = np.zeros(len(V), np.float32)
    for p, (vidx, tree) in trees.items():
        sel = part == p
        if not sel.any():
            continue
        q = np.stack([uv[p - 1, 0][sel], uv[p - 1, 1][sel]], 1)
        rgb = crop[sel].astype(np.float32)
        _, nn = tree.query(q)                          # nearest template vertex in this part
        gv = vidx[nn]
        np.add.at(vcol, gv, rgb); np.add.at(vcnt, gv, 1.0)
    painted = vcnt > 0
    vcol[painted] /= vcnt[painted, None]
    cv = base_canvas()
    pv = np.where(painted)[0]
    pv = pv[np.argsort(ZB[pv])]                         # back->front so front wins
    cv[PY[pv], PX[pv]] = vcol[pv].astype(np.uint8)
    # dilate the painted cloud slightly for visibility
    cv = cv2.dilate(cv, np.ones((2, 2), np.uint8))
    return cv


cap = cv2.VideoCapture(a.inp); fps = cap.get(5) or 16; N = int(cap.get(7))
if a.max_frames:
    N = min(N, a.max_frames)
vw = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8,
                        pixelformat="yuv420p", macro_block_size=16, ffmpeg_log_level="error")
i = 0
while i < N:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = bgr[..., ::-1].copy()
    box = largest_person(rgb)
    cropV = cv2.resize(rgb, (R, R)) if box is None else None
    if box is not None:
        crop = crop_square(rgb, box); part, uv = infer(crop)
        tmpl = paint_frame(crop, part, uv)
        left = cv2.resize(crop, (R, R))
    else:
        tmpl = base_canvas(); left = cv2.resize(rgb, (R, R))
    vw.append_data(np.ascontiguousarray(np.concatenate([left, tmpl], 1)))
    i += 1
    if i % 25 == 0:
        print(f"  {i}/{N}", flush=True)
cap.release(); vw.close()
print(f"wrote {a.out} ({i} frames, {2*R}x{R} @ {fps:.0f}fps)", flush=True)
