"""Dataset-wide verification: sample saved instances from MANY sequences, read
their stored RGBA correspondence PNG, decode it, and composite back onto the
actual movie frame to confirm alignment held across the whole dataset (not just
the one HDRI sequence we eyeballed).

Usage: python make_verify_bundle.py [--per-seq 2] [--seqs 14]
Output: ~/data/bedlam2/correspondence_verify/index.html
"""
import argparse
import glob
import json
import os
import random
import subprocess
import tarfile
import tempfile

import numpy as np
import torch
from PIL import Image

import imageio.v3 as iio
import imageio_ffmpeg
from encode_correspondence import unpack_rgba
from render_correspondence import pose_instance

ROOT = os.path.expanduser("~/data/bedlam2")
CORR = f"{ROOT}/correspondence"
MOVIES = f"{ROOT}/downloader"
OUT = f"{ROOT}/correspondence_verify"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def surface_colors():
    v, f = pose_instance(np.zeros(165), np.zeros(16), np.zeros(3), "neutral", "cuda")
    fc = v[f.long()].mean(1).detach().cpu().numpy()
    lo, hi = fc.min(0), fc.max(0)
    return (fc - lo) / (hi - lo)


def get_frame(seq, sub, frame, cache):
    """Extract one RGB frame from the sequence movie tar (cached per sub)."""
    key = (seq, sub)
    if key not in cache:
        tar = f"{MOVIES}/{seq}_mp4.tar"
        with tarfile.open(tar) as t:
            m = next((x for x in t.getmembers()
                      if x.name.endswith(f"{sub}.mp4")), None)
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
    ap.add_argument("--per-seq", type=int, default=2)
    ap.add_argument("--seqs", type=int, default=14)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    fcolors = surface_colors()
    rng = random.Random(0)

    anns = sorted(glob.glob(f"{CORR}/annotations/*.jsonl"))
    rng.shuffle(anns)
    anns = anns[:args.seqs]
    cards = []
    framecache = {}
    for ann in anns:
        seq = os.path.basename(ann)[:-6]
        recs = [json.loads(l) for l in open(ann)]
        if not recs:
            continue
        # prefer multi-person frames: pick records, group by frame, sample
        rng.shuffle(recs)
        picked = recs[:args.per_seq]
        for rec in picked:
            rgb = get_frame(seq, rec["sub"], rec["frame"], framecache)
            if rgb is None:
                continue
            png = np.array(Image.open(f"{CORR}/{rec['png']}"))
            face, bary = unpack_rgba(png)
            x0, y0, x1, y1 = rec["bbox"]
            vis = face >= 0
            col = np.zeros((*face.shape, 3), np.uint8)
            col[vis] = (fcolors[face[vis]] * 255).astype(np.uint8)
            over = rgb.copy()
            crop = over[y0:y1, x0:x1]
            if crop.shape[:2] != face.shape:  # safety on edge clamps
                continue
            crop[vis] = (0.45 * crop[vis] + 0.55 * col[vis]).astype(np.uint8)
            # side-by-side: original frame (bbox drawn) | overlay
            tag = f"{seq}_{rec['sub']}_{rec['frame']}_{rec['instance']}"
            from PIL import ImageDraw
            orig = Image.fromarray(rgb); dr = ImageDraw.Draw(orig)
            dr.rectangle([x0, y0, x1, y1], outline=(0, 255, 0), width=3)
            comp = np.concatenate([np.array(orig), over], 1)
            Image.fromarray(comp).save(f"{OUT}/{tag}.png")
            cards.append((tag, seq, rec))
    # cleanup temp mp4s
    for v in framecache.values():
        if v:
            os.unlink(v)

    html = ["<html><head><style>body{font:12px monospace}img{max-width:100%;"
            "border:1px solid #ccc;margin:3px 0}h3{background:#eef;padding:4px;"
            "margin:10px 0 2px}</style></head><body>",
            "<h1>BEDLAM2 correspondence — dataset-wide verification</h1>",
            "<p>Each row: a randomly-sampled SAVED instance from a different "
            "sequence. Left = movie frame (green box = stored bbox). Right = the "
            "stored RGBA correspondence decoded and composited back. Alignment "
            "of the colored body onto the person verifies the on-disk data.</p>"]
    for tag, seq, rec in cards:
        html.append(f"<h3>{seq} | frame {rec['frame']} inst {rec['instance']} "
                    f"| {rec['gender']} | area {rec['area']}px</h3>")
        html.append(f"<img src='{tag}.png'>")
    html.append("</body></html>")
    open(f"{OUT}/index.html", "w").write("\n".join(html))
    sz = sum(os.path.getsize(f"{OUT}/{f}") for f in os.listdir(OUT)) / 1e6
    print(f"verify bundle: {OUT} ({len(cards)} instances from {len(anns)} seqs, {sz:.0f} MB)")


if __name__ == "__main__":
    main()
