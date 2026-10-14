"""Build a downloadable visualization bundle of BEDLAM2 correspondence renders.

For several multi-person frames: original RGB, overview (face-id surface color |
barycentric | instance ids), round-trip (face coloring composited back onto the
original), and per-instance crops (rgb | face color | barycentric). Writes PNGs
+ an index.html for offline review.

Usage: python make_viz_bundle.py
Output: ~/data/bedlam2/correspondence_viz/  (rsync this folder)
"""
import os
import subprocess

import numpy as np
import torch
from PIL import Image

import imageio_ffmpeg
from render_correspondence import (crops_from_frame, pose_instance,
                                    rasterize_frame)

ROOT = os.path.expanduser("~/data/bedlam2")
LAB = f"{ROOT}/ground_truth/bedlam2_labels_processed"
OUT = f"{ROOT}/correspondence_viz"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
DEV = "cuda"

# (sequence, list of unique-frame indices to render) — multi-person + occlusion
TARGETS = [
    ("20240506_10_200_hdri", [0, 40, 80]),
]


def extract(seq, sub, fn):
    mp4 = f"{ROOT}/sample_mp/{seq}/mp4/{sub}.mp4"
    out = f"/tmp/{sub}_{fn}.png"
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-vf", f"select='eq(n\\,{fn})'",
                    "-frames:v", "1", out, "-loglevel", "error"], check=True)
    return np.array(Image.open(out).convert("RGB"))


def surface_colors():
    v, f = pose_instance(np.zeros(165), np.zeros(16), np.zeros(3), "neutral", DEV)
    fc = v[f.long()].mean(1).detach().cpu().numpy()
    lo, hi = fc.min(0), fc.max(0)
    return (fc - lo) / (hi - lo)


def colorize(face, fcolors):
    h, w = face.shape
    img = np.zeros((h, w, 3), np.float32)
    fg = face >= 0
    img[fg] = fcolors[face[fg]]
    return (img * 255).astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    fcolors = surface_colors()
    cards = []
    for seq, frames in TARGETS:
        d = np.load(f"{LAB}/{seq}.npz", allow_pickle=True)
        uniq = sorted(set(d["imgname"].tolist()))
        for fi in frames:
            target = uniq[fi]
            rows = np.where(d["imgname"] == target)[0]
            sub = target.split("/")[0]
            fn = int(target.split("_")[-1].split(".")[0])
            rgb = extract(seq, sub, fn)
            H, W = rgb.shape[:2]
            K = d["cam_int"][rows[0]]
            insts = [dict(pose=d["pose_world"][r], betas=d["shape"][r],
                          trans=d["trans_world"][r], ext=d["cam_ext"][r],
                          gender=str(d["gender"][r])) for r in rows]
            face, bary, inst, meta, _, _ = rasterize_frame(insts, K, H, W, DEV)
            fcol = colorize(face, fcolors)
            bimg = np.zeros((H, W, 3), np.uint8); fg = face >= 0
            bimg[fg] = (bary[fg] * 255).astype(np.uint8)
            rng = np.random.default_rng(0)
            pal = (rng.random((len(insts), 3)) * 255).astype(np.uint8)
            icol = np.zeros((H, W, 3), np.uint8)
            for i in range(len(insts)):
                icol[inst == i] = pal[i]
            tag = f"{seq}_{fn:04d}"
            H4 = lambda a: np.concatenate(a, 1)
            Image.fromarray(H4([rgb, fcol, bimg, icol])).save(f"{OUT}/{tag}_overview.png")
            # roundtrip
            crops = crops_from_frame(face, bary, inst, len(insts))
            rt = rgb.copy()
            crop_imgs = []
            for i, c in enumerate(crops):
                if c is None:
                    continue
                x0, y0, x1, y1 = c["bbox"]
                fcc = colorize(c["face"], fcolors)
                vis = c["face"] >= 0
                b2 = c["bary2"].astype(np.float32)
                b3 = np.concatenate([b2, (1 - b2.sum(-1, keepdims=True))], -1)
                bcc = np.zeros((*c["face"].shape, 3), np.uint8)
                bcc[vis] = (b3[vis] * 255).astype(np.uint8)
                rt[y0:y1, x0:x1][vis] = fcc[vis]
                if c["area"] > 3000 and len(crop_imgs) < 3:
                    ci = H4([rgb[y0:y1, x0:x1], fcc, bcc])
                    cp = f"{tag}_crop{i}.png"; Image.fromarray(ci).save(f"{OUT}/{cp}")
                    crop_imgs.append(cp)
            Image.fromarray(rt).save(f"{OUT}/{tag}_roundtrip.png")
            cards.append((tag, len(rows), crop_imgs))

    html = ["<html><head><style>body{font:13px monospace}img{max-width:100%;"
            "border:1px solid #ccc;margin:4px 0}h2{background:#eef;padding:6px}"
            "</style></head><body><h1>BEDLAM2 dense correspondence — multi-person"
            " renders</h1><p>Per frame: <b>overview</b> = original | face-index "
            "(smooth surface color) | barycentric | instance-ids. <b>roundtrip</b>"
            " = face coloring composited back onto original. <b>crops</b> = rgb | "
            "face | barycentric.</p>"]
    for tag, npx, crops in cards:
        html.append(f"<h2>{tag} — {npx} people</h2>")
        html.append(f"overview:<br><img src='{tag}_overview.png'>")
        html.append(f"<br>roundtrip:<br><img src='{tag}_roundtrip.png'>")
        for cp in crops:
            html.append(f"<br><img src='{cp}'>")
    html.append("</body></html>")
    open(f"{OUT}/index.html", "w").write("\n".join(html))
    sz = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e6
    print(f"viz bundle: {OUT} ({len(cards)} frames, {sz:.0f} MB)")


if __name__ == "__main__":
    main()
