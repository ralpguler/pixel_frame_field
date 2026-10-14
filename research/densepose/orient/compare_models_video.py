"""Side-by-side VIDEO comparison of two oriented-DensePose checkpoints on real clips (3DPW etc.).
One detector pass per frame; both models predict IUV+rotation on each tracked person. Composite:
  [ original+boxes | A: normal | B: normal | A-vs-B disagreement (geodesic deg) ]
Purpose: temporal check of whether model B's rotations are broken/flickery vs model A.

Usage: compare_models_video.py IN_DIR_OR_MP4 OUT.mp4 --a A.pt --b B.pt --la LA --lb LB [--stride N --max-frames N]
"""
import os, sys, glob, argparse, numpy as np, cv2, torch, imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from model import DINOv3DPT, part_pred
from orient.template_normals import TemplateNormals, normal_to_rgb
from orient.rot6d import get_template, sixd_to_R_np
from orient.detect_track import detect_persons, Tracker

ap = argparse.ArgumentParser()
D = os.path.expanduser("~/data/bedlam2/dp_runs")
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--a", default=f"{D}/dp_scratch_bedlam_orient_pp_llrd_1e3/ckpt_050000.pt")
ap.add_argument("--b", default=f"{D}/dp_pp_llrd_1e3_mamma/ckpt_050000.pt")
ap.add_argument("--la", default="BEDLAM 50k"); ap.add_argument("--lb", default="+MAMMA 50k")
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--stride", type=int, default=1)
ap.add_argument("--fps", type=float, default=30.0)
ap.add_argument("--det-thr", type=float, default=0.5)
ap.add_argument("--nms-iou", type=float, default=0.6)   # suppress overlapping duplicate person boxes
ap.add_argument("--min-frac", type=float, default=0.04) # drop boxes whose max side < this * frame height
ap.add_argument("--no-track", action="store_true")      # use per-frame NMS'd dets directly (no EMA tracker)
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
A = os.path.expanduser("~/data/bedlam2/dp_assets")
v_template, faces, n_rest = get_template("cuda")
fuv = np.load(f"{A}/smplx_face_uv.npy"); fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
TN = TemplateNormals(fuv, fpart, np.asarray(faces), np.asarray(n_rest))


def load(ckpt):
    sd = torch.load(ckpt, map_location="cpu")["model"]
    rpp = sd["rot_head.2.weight"].shape[0] != 6
    m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
                  pretrained=False, decoupled=True, orient=True, rot_per_part=rpp).to(dev).eval()
    m.load_state_dict(sd); print("loaded", ckpt, "per_part" if rpp else "single", flush=True); return m


mA, mB = load(a.a), load(a.b)


@torch.no_grad()
def predict(m, crop_rgb, cs):
    img = torch.from_numpy(cv2.resize(crop_rgb, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o = m(img)
    part = part_pred(o)[0].cpu().numpy().astype(int)
    partS = cv2.resize(part.astype(np.uint8), (cs, cs), interpolation=cv2.INTER_NEAREST).astype(int)
    r, c = np.indices(partS.shape); idx = np.clip(partS - 1, 0, 23)
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
    R = sixd_to_R_np(six); nt, mask = TN(partS, US, VS)
    return nt, mask, R


def _iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0); inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(dets, iou_thr, min_side):
    """Greedy NMS on [x0,y0,x1,y1,score] person boxes + drop tiny boxes -> clean, non-duplicate people."""
    dets = [d for d in dets if max(d[2] - d[0], d[3] - d[1]) >= min_side]
    dets = sorted(dets, key=lambda d: -d[4]); keep = []
    for d in dets:
        if all(_iou(d, k) < iou_thr for k in keep):
            keep.append(d)
    return keep


def crop_box(rgb, box):
    """Square crop of EXP*box centered on the box; returns (crop, X0, Y0) where (X0,Y0) is the crop's
    top-left in ORIGINAL coords (may be negative — the crop is edge-padded there)."""
    x0, y0, x1, y1 = box[:4]; cx, cy = (x0 + x1) / 2, (y0 + y1) / 2; s = max(x1 - x0, y1 - y0) * EXP / 2
    side = int(2 * s); X0, Y0 = int(cx - s), int(cy - s); X1, Y1 = X0 + side, Y0 + side   # exactly square
    H, W = rgb.shape[:2]
    pad = ((max(0, -Y0), max(0, Y1 - H)), (max(0, -X0), max(0, X1 - W)), (0, 0))
    crop = np.pad(rgb, pad, mode="edge")[max(0, Y0):Y1 + pad[0][0], max(0, X0):X1 + pad[1][0]]
    return crop, X0, Y0


def paste(canvas, tile, mask, X0, Y0):
    """Composite tile[mask] onto canvas mapping crop pixel (i,j) -> original (Y0+i, X0+j); handles
    negative X0/Y0 (edge-padded boxes) and overflow via intersected src/dst slices."""
    H, W = canvas.shape[:2]; cs = tile.shape[0]
    dr0, dr1 = max(0, Y0), min(H, Y0 + cs); dc0, dc1 = max(0, X0), min(W, X0 + cs)
    if dr1 <= dr0 or dc1 <= dc0:
        return
    sr0, sr1 = dr0 - Y0, dr1 - Y0; sc0, sc1 = dc0 - X0, dc1 - X0
    msub = mask[sr0:sr1, sc0:sc1]
    canvas[dr0:dr1, dc0:dc1][msub] = tile[sr0:sr1, sc0:sc1][msub]


def render_normal(rgb, boxes, m):
    canvas = np.full_like(rgb, 255)
    for box in boxes:
        crop, X0, Y0 = crop_box(rgb, box); cs = crop.shape[0]
        if cs < 48 or crop.shape[0] != crop.shape[1]:
            continue
        nt, mask, R = predict(m, crop, cs)
        nrm = (normal_to_rgb(np.einsum("hwij,hwj->hwi", R, nt), mask) * 255).astype(np.uint8)
        paste(canvas, nrm, mask, X0, Y0)
    return canvas


def render_disagree(rgb, boxes):
    canvas = np.full(rgb.shape[:2], np.nan, np.float32)
    for box in boxes:
        crop, X0, Y0 = crop_box(rgb, box); cs = crop.shape[0]
        if cs < 48 or crop.shape[0] != crop.shape[1]:
            continue
        _, mA_, RA = predict(mA, crop, cs); _, mB_, RB = predict(mB, crop, cs)
        both = mA_ & mB_
        rel = np.matmul(np.transpose(RA, (0, 1, 3, 2)), RB)
        tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
        d = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        paste(canvas, d, both, X0, Y0)
    import matplotlib.cm as cm
    out = (cm.turbo(np.clip(np.nan_to_num(canvas, nan=0) / 45.0, 0, 1))[..., :3] * 255).astype(np.uint8)
    out[np.isnan(canvas)] = 255
    return out


is_dir = os.path.isdir(a.inp)
if is_dir:
    frames = sorted(glob.glob(os.path.join(a.inp, "*.jpg")) + glob.glob(os.path.join(a.inp, "*.png")))
    frames = frames[::a.stride]
    if a.max_frames:
        frames = frames[:a.max_frames]
    reader = (cv2.imread(f)[..., ::-1] for f in frames); fps = a.fps
else:
    cap = cv2.VideoCapture(a.inp); fps = (cap.get(5) or a.fps)
    def _gen():
        i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % a.stride == 0:
                yield fr[..., ::-1]
            i += 1
    reader = _gen()

trk = Tracker()
wr = imageio.get_writer(a.out, fps=fps / (1 if is_dir else a.stride), codec="libx264", quality=7,
                        pixelformat="yuv420p", macro_block_size=8)
n = 0
for rgb in reader:
    if a.max_frames and n >= a.max_frames:
        break
    rgb = np.ascontiguousarray(rgb)
    dets = nms(detect_persons(rgb, thr=a.det_thr), a.nms_iou, a.min_frac * rgb.shape[0])
    boxes = dets if a.no_track else trk.update(dets)
    orig = rgb.copy()
    for b in boxes:
        cv2.rectangle(orig, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    nA = render_normal(rgb, boxes, mA); nB = render_normal(rgb, boxes, mB); dis = render_disagree(rgb, boxes)

    def lab(im, t):
        im = im.copy(); cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        cv2.putText(im, t, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1); return im
    panel = np.concatenate([lab(orig, "input"), lab(nA, a.la), lab(nB, a.lb), lab(dis, "A vs B deg")], axis=1)
    wr.append_data(panel); n += 1
    if n % 20 == 0:
        print("frame", n, flush=True)
wr.close()
print("wrote", a.out, n, "frames", flush=True)
