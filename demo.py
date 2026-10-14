"""Demo: pixel frame fields on an image or a video.

  python demo.py photo.jpg out.png --ckpt checkpoints/pff_dp_w_MF_3k.pt [--box X Y W H]
  python demo.py clip.mp4 out.mp4  --ckpt checkpoints/pff_dp_w_MF_3k.pt [--max-frames 200]
  python demo.py clip.mp4 fields/  --ckpt checkpoints/pff_dp_w_MF_3k.pt --dump      # per-frame field files

Each output frame has four panels: the input with the person boxes | body parts | chart coordinates (U red,
V green) | the camera-space surface normal R n, with n the template's rest normal at the predicted (part, U, V)
and R the predicted rotation. People come from RT-DETRv2 (pff.detect, with a tracker on video) unless --box is
given.

--dump writes, for the main person of every frame (largest box, smoothed over time, one crop size for the clip),
f_<frame>.npz with mask, part, uv, n_cam, t_cam, twist (the format of research/densepose/orient/field_dump.py,
which the HTML viewers in viz/ read), f_<frame>.jpg (the crop) and meta.json.
"""
import argparse
import json
import os

import cv2
import numpy as np

from pff.infer import Z, Predictor
from pff.template import Template, denoise_part, iuv_rgb, normal_to_rgb, sixd_to_R_np, uv_rgb

VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


def panels(rgb, res, tmpl, boxes):
    """Four panels side by side for one frame."""
    part, fg = res["part"], res["part"] > 0
    show = rgb.copy()
    for b in boxes:
        x, y, w, h = map(int, b[:4])
        cv2.rectangle(show, (x, y), (x + w, y + h), (0, 255, 0), 2)
    p_iuv, p_uv, p_n = rgb.copy(), rgb.copy(), rgb.copy()
    if fg.any():
        p_iuv[fg] = iuv_rgb(part, res["U"], res["V"])[fg]
        p_uv[fg] = uv_rgb(part, res["U"], res["V"])[fg]
        nt, _ = tmpl.normals(part, res["U"], res["V"])
        n = np.einsum("hwij,hwj->hwi", res["R"], nt)
        p_n[fg] = normal_to_rgb(n, fg)[fg]
    return np.concatenate([show, p_iuv, p_uv, p_n], 1)


def xyxy_to_xywh(b):
    return [b[0], b[1], b[2] - b[0], b[3] - b[1], b[4] if len(b) > 4 else 1.0]


def frames_of(path, stride=1, max_frames=0):
    cap = cv2.VideoCapture(path)
    fps = (cap.get(cv2.CAP_PROP_FPS) or 25.0) / stride
    i = n = 0
    while True:
        ok, bgr = cap.read()
        if not ok or (max_frames and n >= max_frames):
            break
        if i % stride == 0:
            yield fps, np.ascontiguousarray(bgr[..., ::-1]); n += 1
        i += 1
    cap.release()


def run(a):
    pred, tmpl = Predictor(a.ckpt, a.device), Template()
    det = None
    if not a.box:
        from pff.detect import Tracker, detect_persons
        det = lambda rgb: detect_persons(rgb, thr=a.det_thr, device=pred.device)      # noqa: E731
    if not a.inp.lower().endswith(VIDEO_EXT):
        rgb = cv2.imread(a.inp)[..., ::-1].copy()
        boxes = [a.box + [1.0]] if a.box else [xyxy_to_xywh(b) for b in det(rgb)]
        res = pred.predict_image(rgb, boxes)
        cv2.imwrite(a.out, panels(rgb, res, tmpl, boxes)[..., ::-1])
        print(f"wrote {a.out} ({len(boxes)} people)")
        return
    from imageio_ffmpeg import write_frames
    tracker = None if a.box else Tracker()
    wr = None
    for k, (fps, rgb) in enumerate(frames_of(a.inp, a.stride, a.max_frames)):
        boxes = [a.box + [1.0]] if a.box else [xyxy_to_xywh(b) for b in tracker.update(det(rgb))]
        frame = panels(rgb, pred.predict_image(rgb, boxes), tmpl, boxes)
        if wr is None:
            h, w = frame.shape[:2]
            wr = write_frames(a.out, (w - w % 2, h - h % 2), fps=fps, codec="libx264", pix_fmt_out="yuv420p",
                              macro_block_size=2, output_params=["-crf", "20"])
            wr.send(None)
        wr.send(np.ascontiguousarray(frame[:h - h % 2, :w - w % 2]))
        if k % 25 == 0:
            print(f"  frame {k}", flush=True)
    if wr is not None:
        wr.close()
    print(f"wrote {a.out}")


# ------------------------------------------------------------------ field dump (viewer format)
def predict_fields(pred, crop):
    """research field_dump.predict: the 512 prediction resampled to the crop size, part majority-denoised,
    U, V and the 6D read at the denoised part, rotation re-orthonormalised."""
    import torch
    from pff.geometry import part_pred
    cs = crop.shape[0]
    img = torch.from_numpy(cv2.resize(crop, (Z, Z))).permute(2, 0, 1)[None].float().to(pred.device) / 255
    with torch.no_grad():
        if pred.device.startswith("cuda"):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = pred.model(img)
        else:
            o = pred.model(img)
    part = cv2.resize(part_pred(o)[0].cpu().numpy().astype(np.uint8), (cs, cs),
                      interpolation=cv2.INTER_NEAREST).astype(int)
    part = denoise_part(part)
    r, c = np.indices(part.shape); idx = np.clip(part - 1, 0, 23)
    uv = o["uv"].float()[0].view(24, 2, Z, Z).clamp(0, 1).cpu().numpy()
    uvv = np.stack([[cv2.resize(uv[p, k], (cs, cs)) for k in range(2)] for p in range(24)])
    rot = o["rot6d"].float()[0].view(24, 6, Z, Z).cpu().numpy()
    rotv = np.stack([[cv2.resize(rot[p, k], (cs, cs)) for k in range(6)] for p in range(24)])
    return part, uvv[idx, 0, r, c], uvv[idx, 1, r, c], sixd_to_R_np(rotv[idx, :, r, c])


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


def dump(a):
    from pff.detect import detect_persons
    pred, tmpl = Predictor(a.ckpt, a.device), Template()
    os.makedirs(a.out, exist_ok=True)
    video = a.inp.lower().endswith(VIDEO_EXT)
    frames = list(frames_of(a.inp, a.stride, a.max_frames)) if video else \
        [(25.0, cv2.imread(a.inp)[..., ::-1].copy())]
    B = np.full((len(frames), 3), np.nan)                            # centre x, centre y, half side
    for j, (_, rgb) in enumerate(frames):
        if a.box:
            x, y, w, h = a.box; dets = [[x, y, x + w, y + h, 1.0]]
        else:
            dets = detect_persons(rgb, thr=0.5, device=pred.device)
        if dets:
            x0, y0, x1, y1 = max(dets, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))[:4]
            B[j] = [(x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * a.exp / 2]
    B = smooth_boxes(B, a.smooth if len(frames) > 1 else 1)
    if B is None:
        raise SystemExit("no person detected")
    done = []
    for j, (fps, rgb) in enumerate(frames):
        cx, cy, s = B[j]
        H, W = rgb.shape[:2]
        side = int(2 * s); X0, Y0 = int(cx - s), int(cy - s)
        pad = ((max(0, -Y0), max(0, Y0 + side - H)), (max(0, -X0), max(0, X0 + side - W)), (0, 0))
        crop = np.pad(rgb, pad, mode="edge")[max(0, Y0):Y0 + side + pad[0][0], max(0, X0):X0 + side + pad[1][0]]
        part, U, V, R = predict_fields(pred, np.ascontiguousarray(crop))
        n_cam, t_cam, fg = tmpl.frames(part, U, V, R)
        twist = (np.arctan2(R[..., 1, 0], R[..., 0, 0]) + np.pi) / (2 * np.pi)
        np.savez_compressed(f"{a.out}/f_{j:04d}.npz", mask=fg, part=part.astype(np.uint8),
                            uv=np.stack([U, V]).astype(np.float16), n_cam=n_cam.astype(np.float16),
                            t_cam=t_cam.astype(np.float16), twist=twist.astype(np.float16))
        cv2.imwrite(f"{a.out}/f_{j:04d}.jpg", crop[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        done.append(j)
    json.dump({"clip": os.path.basename(a.inp), "frames": done, "fps": frames[0][0], "side": int(2 * B[0, 2]),
               "ckpt": os.path.basename(a.ckpt)}, open(f"{a.out}/meta.json", "w"))
    print(f"wrote {len(done)} frames to {a.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inp"); ap.add_argument("out")
    ap.add_argument("--ckpt", default="checkpoints/pff_dp_w_MF_3k.pt")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                    help="one person box instead of the detector")
    ap.add_argument("--det-thr", type=float, default=0.4)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dump", action="store_true", help="write per-frame field files to the folder OUT")
    ap.add_argument("--exp", type=float, default=1.3, help="--dump: crop side / box side")
    ap.add_argument("--smooth", type=int, default=15, help="--dump: box smoothing window (frames)")
    a = ap.parse_args()
    a.box = list(a.box) if a.box else None
    dump(a) if a.dump else run(a)


if __name__ == "__main__":
    main()
