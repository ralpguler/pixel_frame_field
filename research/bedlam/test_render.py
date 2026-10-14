"""Visual test of BEDLAM2 correspondence rendering on a multi-person frame.

Produces, into bedlam/test_out/:
  1. validate_proj.png  - GT proj_verts vs our projected verts (camera sanity)
  2. overview.png       - original | face-id colored | barycentric | instance ids
  3. crops_<i>.png      - per-instance: rgb crop | face coloring | barycentric
  4. roundtrip.png      - crops composited back onto the original by bbox

Usage: python test_render.py [--seq 20240506_10_200_hdri] [--frame 0]
"""
import argparse
import os
import subprocess

import numpy as np
import torch
from PIL import Image

import imageio_ffmpeg
from render_correspondence import (crops_from_frame, get_smplx, pose_instance,
                                    project_opencv, rasterize_frame)

ROOT = os.path.expanduser("~/data/bedlam2")
OUT = os.path.dirname(__file__) + "/test_out"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def extract_frame(seq, sub_seq, frame_idx):
    """imgname 'seq_000000/seq_000000_0000.png' -> RGB frame from the mp4."""
    mp4 = f"{ROOT}/sample_mp/{seq}/mp4/{sub_seq}.mp4"
    out = f"/tmp/{sub_seq}_{frame_idx}.png"
    subprocess.run([FFMPEG, "-y", "-i", mp4, "-vf",
                    f"select='eq(n\\,{frame_idx})'", "-frames:v", "1", out,
                    "-loglevel", "error"], check=True)
    return np.array(Image.open(out).convert("RGB"))


def face_surface_colors(faces, verts_template):
    """Smooth per-face color = normalized face-centroid position (DensePose-like
    continuous surface coordinate). Computed once on a T-pose mesh."""
    fc = verts_template[faces].mean(1)            # (F,3)
    lo, hi = fc.min(0).values, fc.max(0).values
    return ((fc - lo) / (hi - lo)).cpu().numpy()  # (F,3) in [0,1]


def colorize(face_idx, inst_id, fcolors):
    h, w = face_idx.shape
    img = np.zeros((h, w, 3), np.float32)
    fg = face_idx >= 0
    img[fg] = fcolors[face_idx[fg]]
    return (img * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", default="20240506_10_200_hdri")
    ap.add_argument("--frame", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda"

    d = np.load(f"{ROOT}/ground_truth/bedlam2_labels_processed/{args.seq}.npz",
                allow_pickle=True)
    img_all = d["imgname"]
    # pick the target frame's imgname (frame-th unique, or use --frame as index)
    uniq = sorted(set(img_all.tolist()))
    target = uniq[args.frame]
    rows = np.where(img_all == target)[0]
    print(f"frame {target}: {len(rows)} people")

    sub_seq = target.split("/")[0]
    fnum = int(target.split("_")[-1].split(".")[0])
    rgb = extract_frame(args.seq, sub_seq, fnum)
    H, W = rgb.shape[:2]
    K = d["cam_int"][rows[0]]

    ext = d["cam_ext"][rows[0]]
    instances = [dict(pose=d["pose_world"][r], betas=d["shape"][r],
                      trans=d["trans_world"][r], gender=str(d["gender"][r]))
                 for r in rows]

    # --- camera validation: our projection of instance 0 vs GT proj_verts ---
    from render_correspondence import world_to_cam
    v0w, _ = pose_instance(instances[0]["pose"], instances[0]["betas"],
                           instances[0]["trans"], instances[0]["gender"], dev)
    v0 = world_to_cam(v0w, ext, dev)
    proj = project_opencv(v0, K).cpu().numpy()
    gtp = d["proj_verts"][rows[0]][:, :2]
    fig = Image.fromarray(rgb).copy()
    import PIL.ImageDraw as ImageDraw
    dr = ImageDraw.Draw(fig)
    for x, y in proj[::20]:
        dr.ellipse([x-2, y-2, x+2, y+2], fill=(0, 255, 0))      # ours green
    for x, y in gtp[::5]:
        dr.ellipse([x-1, y-1, x+1, y+1], fill=(255, 0, 0))      # GT red
    fig.save(f"{OUT}/validate_proj.png")
    err = np.linalg.norm(proj[:gtp.shape[0]] - gtp, axis=1) if proj.shape[0] >= gtp.shape[0] else None
    print(f"camera check saved; (green=ours, red=GT proj_verts)")

    # --- rasterize the whole frame ---
    face_idx, bary, inst_id, meta, verts, faces = rasterize_frame(
        instances, K, ext, H, W, dev)
    # smooth surface colors from a template (betas=0) mesh
    tmpl, ftens = pose_instance(np.zeros(165), np.zeros(16), np.zeros(3),
                                "neutral", dev)
    fcolors = face_surface_colors(ftens.cpu().numpy(), tmpl.detach())

    face_col = colorize(face_idx, inst_id, fcolors)
    bary_img = np.zeros((H, W, 3), np.uint8)
    fgm = face_idx >= 0
    bary_img[fgm] = (bary[fgm] * 255).astype(np.uint8)
    # instance id colored
    rng = np.random.default_rng(0)
    inst_pal = (rng.random((len(instances), 3)) * 255).astype(np.uint8)
    inst_col = np.zeros((H, W, 3), np.uint8)
    for i in range(len(instances)):
        inst_col[inst_id == i] = inst_pal[i]

    def hstack(imgs):
        return np.concatenate(imgs, axis=1)
    Image.fromarray(hstack([rgb, face_col, bary_img, inst_col])).save(
        f"{OUT}/overview.png")
    print(f"overview saved; visible coverage "
          f"{(fgm.sum()/(H*W)):.1%} of frame")

    # --- per-instance crops + round-trip ---
    crops = crops_from_frame(face_idx, bary, inst_id, len(instances))
    rt = rgb.copy()
    for i, c in enumerate(crops):
        if c is None:
            continue
        x0, y0, x1, y1 = c["bbox"]
        rgb_crop = rgb[y0:y1, x0:x1]
        fc_crop = colorize(c["face"], None, fcolors)
        b = np.zeros((*c["face"].shape, 3), np.uint8)
        vis = c["face"] >= 0
        b3 = np.concatenate([c["bary2"].astype(np.float32),
                             (1 - c["bary2"].sum(-1, keepdims=True)).astype(np.float32)], -1)
        b[vis] = (b3[vis] * 255).astype(np.uint8)
        Image.fromarray(hstack([rgb_crop, fc_crop, b])).save(f"{OUT}/crop_{i}.png")
        # composite face coloring back onto original at the bbox (visible only)
        rt[y0:y1, x0:x1][vis] = fc_crop[vis]
    Image.fromarray(rt).save(f"{OUT}/roundtrip.png")
    sizes = [(i, c["bbox"], c["area"]) for i, c in enumerate(crops) if c]
    print(f"{len(sizes)} visible instances; bboxes/areas:")
    for i, bb, a in sizes:
        print(f"  inst {i}: bbox {bb} area {a}px")


if __name__ == "__main__":
    main()
