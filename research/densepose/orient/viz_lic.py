"""BEAUTY CONTEST #4 -- Line Integral Convolution of the predicted tangent field.

The body pixels get combed: white noise is smeared along streamlines of the image-projected
predicted tangent (R @ t_template), so the *image itself* becomes the direction field. Dense,
per-pixel, forward-only. Colorized by twist (cyclic twilight) or by the camera normal, and
shaded by a fixed key light so the combing reads as fur on a 3D body, not flat streaks.

  python -m orient.viz_lic --dump DIR --out DIR [--frame 75 | --video] [--color twist]
"""
import argparse
import os

import cv2
import numpy as np

from orient.beauty_common import (bilinear, erode_mask, feather, field2d, frames_of, fps_of,
                                  load, plasma, to8, tonemap, twilight, vignette, write_mp4)

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=int, default=-1, help="index into the dump's frame list; -1 = middle")
ap.add_argument("--video", action="store_true")
ap.add_argument("--maxframes", type=int, default=0)
ap.add_argument("--L", type=int, default=14, help="streamline half-length (px)")
ap.add_argument("--passes", type=int, default=2)
ap.add_argument("--color", default="twist", choices=["twist", "normal", "mono"])
ap.add_argument("--dark", type=float, default=0.11)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def lic_once(tex, dir2d, mask):
    ys, xs = np.nonzero(mask)
    P0 = np.stack([xs, ys], 1).astype(np.float32) + 0.5
    acc = tex[ys, xs].copy(); wsum = np.ones(len(ys), np.float32)
    for sgn in (1.0, -1.0):
        P = P0.copy()
        for i in range(1, a.L + 1):
            v = bilinear(dir2d, P)
            n = np.linalg.norm(v, axis=1, keepdims=True)
            P = P + sgn * v / np.clip(n, 1e-6, None)
            w = 0.5 * (1 + np.cos(np.pi * i / (a.L + 1)))      # Hann falloff
            acc += w * bilinear(tex, P); wsum += w
    out = np.zeros_like(tex)
    out[ys, xs] = acc / wsum
    return out


def render(d, noise):
    mask = erode_mask(d["mask"], 1)
    t2d = field2d(d["t_cam"]) * mask[..., None]
    n = np.linalg.norm(t2d, axis=-1, keepdims=True)
    dir2d = t2d / np.clip(n, 1e-6, None) * (n > 0.03)
    lic = noise
    for _ in range(a.passes):
        lic = lic_once(lic, dir2d, mask)
    v = lic[mask]
    lum = np.zeros_like(lic)
    lum[mask] = 0.5 + (v - v.mean()) / (5.0 * v.std() + 1e-9)
    lum = np.clip(lum, 0, 1) ** 1.4
    # key light on the predicted camera normal (camera-facing normals have z<0)
    Ldir = np.array([-0.42, 0.5, -0.76]); Ldir /= np.linalg.norm(Ldir)
    lam = np.clip((d["n_cam"] * Ldir).sum(-1), 0, 1)
    shade = 0.28 + 0.72 * lam
    if a.color == "twist":
        col = twilight(d["twist"])
    elif a.color == "normal":
        col = d["n_cam"] * np.array([1, -1, -1], np.float32) * 0.5 + 0.5
    else:
        col = np.full((*mask.shape, 3), 0.85, np.float32)
    fmask = feather(mask, 1.6)[..., None]
    body = col * (shade * lum * 2.1)[..., None]
    hdr = d["img"] * a.dark * vignette(mask) * (1 - 0.85 * fmask) + body * fmask
    return to8(tonemap(hdr))


frames = frames_of(a.dump)
noise = None
if a.video:
    if a.maxframes:
        frames = frames[:a.maxframes]
    outf = []
    for k, fi in enumerate(frames):
        d = load(a.dump, fi)
        if noise is None:
            rng = np.random.default_rng(0)
            noise = rng.random(d["mask"].shape).astype(np.float32)
        outf.append(render(d, noise))
        if k % 10 == 0:
            print(f"frame {k}/{len(frames)}", flush=True)
    write_mp4(outf, f"{a.out}/lic_{a.color}.mp4", fps_of(a.dump))
    print(f"wrote {a.out}/lic_{a.color}.mp4", flush=True)
else:
    fi = frames[len(frames) // 2 if a.frame < 0 else a.frame]
    d = load(a.dump, fi)
    rng = np.random.default_rng(0)
    noise = rng.random(d["mask"].shape).astype(np.float32)
    fr = render(d, noise)
    cv2.imwrite(f"{a.out}/lic_{a.color}_f{fi:04d}.png", fr[..., ::-1])
    print(f"wrote {a.out}/lic_{a.color}_f{fi:04d}.png", flush=True)
print("LIC DONE", flush=True)
