"""Run the DensePose model on a video: per frame, detect people (fasterrcnn) -> DensePose per box ->
composite the 24-part IUV overlay. Output = side-by-side [original | densepose] mp4.

Usage: video_densepose.py IN.mp4 OUT.mp4 [--ckpt ...] [--max-frames N] [--uv]
"""
import os, sys, argparse, numpy as np, cv2, torch, imageio
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DINOv3DPT, part_pred
from data import _PAL
from orient.detect_track import detect_persons, Tracker   # RT-DETRv2 + letterbox + IoU/EMA tracking

ap = argparse.ArgumentParser()
ap.add_argument("inp"); ap.add_argument("out")
ap.add_argument("--ckpt", default=os.path.expanduser("~/data/bedlam2/dp_runs/dp_ft_joint/ckpt_003000.pt"))
ap.add_argument("--backbone", default="vit_large_patch16_dinov3")
ap.add_argument("--layers", default="4,11,17,23")
ap.add_argument("--max-frames", type=int, default=0)
ap.add_argument("--det-thr", type=float, default=0.4)
ap.add_argument("--uv", action="store_true", help="overlay UV instead of part colors")
a = ap.parse_args()
dev = "cuda"; Z = 512; EXP = 1.3
PAL = _PAL

m = DINOv3DPT(backbone=a.backbone, layers=tuple(int(x) for x in a.layers.split(",")),
              pretrained=False, decoupled=True).to(dev).eval()
m.load_state_dict(torch.load(a.ckpt, map_location="cpu")["model"])
print(f"loaded {a.ckpt}", flush=True)


@torch.no_grad()
def densepose_overlay(rgb, person_boxes):
    H, W = rgb.shape[:2]
    ov = rgb.copy()
    for x0, y0, x1, y1 in person_boxes:
        bw, bh = x1 - x0, y1 - y0
        cx, cy = x0 + bw / 2, y0 + bh / 2; side = EXP * max(bw, bh); S = int(round(side))
        ix0, iy0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        canv = np.zeros((S, S, 3), np.uint8)
        ax0, ay0 = max(0, -ix0), max(0, -iy0); bx0, by0 = max(0, ix0), max(0, iy0)
        bx1, by1 = min(W, ix0 + S), min(H, iy0 + S)
        canv[ay0:ay0 + (by1 - by0), ax0:ax0 + (bx1 - bx0)] = rgb[by0:by1, bx0:bx1]
        img = torch.from_numpy(cv2.resize(canv, (Z, Z))).permute(2, 0, 1)[None].float().to(dev) / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = m(img)
        part = part_pred(o)[0]
        if a.uv:
            uv = o["uv"].float()[0].view(24, 2, Z, Z); sel = (part - 1).clamp(0)
            U = uv[:, 0].gather(0, sel[None])[0].clamp(0, 1); V = uv[:, 1].gather(0, sel[None])[0].clamp(0, 1)
            col = np.stack([(U * 255).byte().cpu().numpy(), (V * 255).byte().cpu().numpy(),
                            np.full((Z, Z), 128, np.uint8)], -1)
        else:
            col = PAL[part.cpu().numpy().astype(np.uint8)]
        partS = cv2.resize(part.cpu().numpy().astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        colS = cv2.resize(col, (S, S), interpolation=cv2.INTER_NEAREST)
        sx, sy = bx0 - ix0, by0 - iy0; ph, pw = by1 - by0, bx1 - bx0
        pp = partS[sy:sy + ph, sx:sx + pw]; fg = pp > 0
        reg = ov[by0:by1, bx0:bx1]; cc = colS[sy:sy + ph, sx:sx + pw]
        reg[fg] = (0.45 * reg[fg] + 0.55 * cc[fg]).astype(np.uint8); ov[by0:by1, bx0:bx1] = reg
    return ov


cap = cv2.VideoCapture(a.inp)
fps = cap.get(cv2.CAP_PROP_FPS) or 16
W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7))
if a.max_frames:
    N = min(N, a.max_frames)
# H.264/yuv420p via imageio-ffmpeg (OpenCV's mp4v writer produces green-corrupt streams)
vw = imageio.get_writer(a.out, fps=fps, codec="libx264", quality=8,
                        pixelformat="yuv420p", macro_block_size=16, ffmpeg_log_level="error")
tracker = Tracker()
i = 0
while i < N:
    ok, bgr = cap.read()
    if not ok:
        break
    rgb = bgr[..., ::-1].copy()
    pboxes = tracker.update(detect_persons(rgb, thr=a.det_thr, dev=dev))
    ov = densepose_overlay(rgb, pboxes)
    vw.append_data(np.ascontiguousarray(np.concatenate([rgb, ov], 1)))   # RGB for imageio
    i += 1
    if i % 50 == 0:
        print(f"  {i}/{N}", flush=True)
cap.release(); vw.close()
print(f"wrote {a.out} ({i} frames, {W*2}x{H} @ {fps:.0f}fps)", flush=True)
