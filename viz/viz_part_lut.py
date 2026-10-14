"""Validate the DensePose-24 SMPL-X LUT visually: pose instances from posed3d,
rasterize the face index, color by face_to_dp24, composite onto the real movie
frame. Output ~/data/bedlam2/dp_assets/dp24_lut_check/index.html.

Usage: python viz_part_lut.py [--seqs 6] [--per-seq 1]
"""
import os  # noqa: E402
import sys  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _env  # noqa: E402  (repository paths: viz/_env.py)
_env.take_out_arg()
import argparse
import glob
import json
import os
import random
import sys
import tarfile
import tempfile

import numpy as np
import torch
from PIL import Image, ImageDraw

import imageio.v3 as iio

from render_correspondence import rasterize_frame

ROOT = _env.data("bedlam2")
CORR = f"{ROOT}/correspondence"
MOVIES = f"{ROOT}/downloader"
ASSETS = _env.ASSETS
OUT = _env.out("part_lut_check")

# 25-color palette (0=background black), visually distinct.
_PAL = np.array([
    [0, 0, 0], [178, 34, 34], [255, 99, 71], [30, 144, 255], [0, 191, 255],
    [50, 205, 50], [34, 139, 34], [255, 215, 0], [218, 165, 32],
    [255, 140, 0], [255, 200, 0], [148, 0, 211], [186, 85, 211],
    [199, 21, 133], [255, 105, 180], [0, 206, 209], [64, 224, 208],
    [0, 128, 128], [72, 209, 204], [255, 0, 255], [238, 130, 238],
    [139, 0, 139], [218, 112, 214], [240, 230, 140], [189, 183, 107],
], np.uint8)


def get_frame(seq, sub, frame, cache):
    key = (seq, sub)
    if key not in cache:
        with tarfile.open(f"{MOVIES}/{seq}_mp4.tar") as t:
            m = next((x for x in t.getmembers() if x.name.endswith(f"{sub}.mp4")), None)
            if m is None:
                cache[key] = None; return None
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(t.extractfile(m).read()); tmp.close()
            cache[key] = tmp.name
    if cache[key] is None:
        return None
    for i, fr in enumerate(iio.imiter(cache[key])):
        if i == frame:
            return fr
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=6)
    ap.add_argument("--per-seq", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    face2part = np.load(f"{ASSETS}/smplx_face_to_dp24.npy")   # (20908,)
    face_uv = np.load(f"{ASSETS}/smplx_face_uv.npy")          # (20908,3,2)
    names = json.load(open(f"{ASSETS}/dp24_names.json"))

    p3ds = sorted(glob.glob(f"{CORR}/posed3d/*.npz"))
    rng = random.Random(1); rng.shuffle(p3ds)
    cache = {}; cards = []
    for npz in p3ds:
        if len(cards) >= args.seqs:
            break
        seq = os.path.basename(npz)[:-4]
        with np.load(npz, allow_pickle=True) as z:
            d = {k: z[k] for k in ("uid", "pose_world", "betas", "trans_world",
                                   "cam_int", "cam_ext", "gender")}
        # group instances by (sub, frame); prefer frames with >=2 people
        groups = {}
        for r, uid in enumerate(d["uid"]):
            sub, fn, _ = str(uid).rsplit("_", 2)
            groups.setdefault((sub, int(fn)), []).append(r)
        multi = sorted(groups.items(), key=lambda kv: -len(kv[1]))
        done = 0
        for (sub, fn), rows in multi:
            if done >= args.per_seq:
                break
            rgb = get_frame(seq, sub, fn, cache)
            if rgb is None:
                continue
            H, W = rgb.shape[:2]
            K = d["cam_int"][rows[0]]
            insts = [dict(pose=d["pose_world"][r], betas=d["betas"][r],
                          trans=d["trans_world"][r], ext=d["cam_ext"][r],
                          gender=str(d["gender"][r])) for r in rows]
            face_idx, bary, inst_id, _, _, _ = rasterize_frame(insts, K, H, W, "cuda")
            fg = face_idx >= 0
            part = np.zeros((H, W), np.int32)
            part[fg] = face2part[face_idx[fg]]
            over = rgb.copy()
            col = _PAL[part]
            over[fg] = (0.35 * over[fg] + 0.65 * col[fg]).astype(np.uint8)
            # UV overlay: per-pixel uv = bary . face_uv[face]; show R=U,G=V
            uv = np.zeros((H, W, 2), np.float32)
            fuv = face_uv[face_idx[fg]]                       # (n,3,2)
            uv[fg] = (bary[fg][:, :, None] * fuv).sum(1)
            uvimg = rgb.copy()
            uvc = np.zeros((H, W, 3), np.uint8)
            uvc[..., 0] = (uv[..., 0] * 255).astype(np.uint8)
            uvc[..., 1] = (uv[..., 1] * 255).astype(np.uint8)
            uvc[..., 2] = np.where(fg, 128, 0).astype(np.uint8)
            uvimg[fg] = (0.25 * uvimg[fg] + 0.75 * uvc[fg]).astype(np.uint8)
            tag = f"{seq}_{sub}_{fn:04d}"
            comp = np.concatenate([rgb, over, uvimg], 1)
            Image.fromarray(comp).save(f"{OUT}/{tag}.png")
            cards.append((tag, seq, len(rows)))
            done += 1
    for v in cache.values():
        if v:
            os.unlink(v)

    # legend
    leg = "".join(
        f"<span style='background:rgb({_PAL[i,0]},{_PAL[i,1]},{_PAL[i,2]});"
        f"color:#fff;padding:2px 5px;margin:1px;display:inline-block'>"
        f"{i} {names[str(i)]}</span>" for i in range(1, 25))
    html = ["<html><head><style>body{font:12px monospace}img{max-width:100%;"
            "border:1px solid #ccc;margin:3px 0}h3{background:#eef;padding:4px}"
            "</style></head><body>",
            "<h1>DensePose-24 SMPL-X IUV check</h1>",
            "<p>Left = movie frame. Middle = part (I) coloring. Right = UV (R=U,G=V). "
            "Check parts are anatomically right and UV gradients are smooth per part.</p>",
            f"<div>{leg}</div>"]
    for tag, seq, n in cards:
        html.append(f"<h3>{seq} — {n} people</h3><img src='{tag}.png'>")
    html.append("</body></html>")
    open(f"{OUT}/index.html", "w").write("\n".join(html))
    sz = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e6
    print(f"dp24 LUT check: {OUT} ({len(cards)} frames, {sz:.0f} MB)")


if __name__ == "__main__":
    main()
