"""BEAUTY CONTEST #1 -- wind-map particle advection of the predicted tangent field.

Particles live on the body pixels, advected along the image-projected predicted tangent
(R_pixel @ t_template), leaving fading additive HDR trails over the darkened video -- the
earth.nullschool look, except the wind is our frame field flowing over a moving person.
Fully forward: no template->pixel search anywhere. Uniform density is maintained the honest
way (respawn at a fresh uniform body pixel; never silently drop).

Speed is modulated by the projected tangent's length: where the tangent points at the camera
(foreshortened), particles slow down -- the 3D field reads through the 2D motion.

  python -m orient.viz_wind --dump ~/data/dp_viz_compare/beauty/15731089 --out DIR [--color twist]
"""
import argparse
import os

import cv2
import numpy as np

from orient.beauty_common import (bilinear, bloom, erode_mask, feather, field2d, frames_of,
                                  fps_of, load, plasma, to8, tonemap, vignette, write_mp4)

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--density", type=float, default=35.0, help="body px per particle")
ap.add_argument("--substeps", type=int, default=4)
ap.add_argument("--decay", type=float, default=0.90, help="trail energy kept per frame")
ap.add_argument("--speed", type=float, default=0.011, help="px per frame as fraction of crop side")
ap.add_argument("--color", default="twist", choices=["twist", "dir", "white"])
ap.add_argument("--dark", type=float, default=0.10)
ap.add_argument("--gain", type=float, default=0.9, help="splat energy")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--maxframes", type=int, default=0)
ap.add_argument("--stills", default="40,75,110", help="frame indices also saved as PNG")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = np.random.default_rng(a.seed)

frames = frames_of(a.dump)
if a.maxframes:
    frames = frames[:a.maxframes]
d0 = load(a.dump, frames[0])
H, W = d0["mask"].shape
SPEED = a.speed * W
N = max(400, int(d0["mask"].sum() / a.density))
LIFE_LO, LIFE_HI = 26, 60
print(f"[wind] {len(frames)} frames {W}x{H} | {N} particles | speed {SPEED:.1f}px/f", flush=True)


def spawn(k, mask, twist):
    ys, xs = np.nonzero(mask)
    j = rng.integers(0, len(ys), k)
    P = np.stack([xs[j] + rng.random(k), ys[j] + rng.random(k)], 1).astype(np.float32)
    life = rng.integers(LIFE_LO, LIFE_HI, k).astype(np.float32)
    return P, life


P, life = spawn(N, erode_mask(d0["mask"]), d0["twist"])
age = rng.integers(0, LIFE_HI, N).astype(np.float32)          # desync so deaths don't pulse
E = np.zeros((H, W, 3), np.float32)                            # HDR trail energy
out_frames = []
stills = set(int(s) for s in a.stills.split(",") if s)

for k, fi in enumerate(frames):
    d = load(a.dump, fi)
    mask = erode_mask(d["mask"], 2)
    t2d = field2d(d["t_cam"]) * mask[..., None]
    twist = d["twist"]
    E *= a.decay
    for _ in range(a.substeps):
        v = bilinear(t2d, P)                                   # projected tangent at particles
        sp = np.linalg.norm(v, axis=1)
        ok = sp > 0.05
        dirn = v / np.clip(sp, 1e-6, None)[:, None]
        step = dirn * (SPEED / a.substeps) * (0.35 + 0.65 * np.clip(sp, 0, 1))[:, None]
        # color per particle, sampled where it is NOW
        if a.color == "twist":
            col = plasma(bilinear(twist, P))
        elif a.color == "dir":
            hue = (np.arctan2(dirn[:, 1], dirn[:, 0]) / (2 * np.pi) + 0.5)
            col = plasma(hue)
        else:
            col = np.ones((len(P), 3), np.float32)
        col = col * (0.35 + 0.65 * np.clip(sp, 0, 1))[:, None] # foreshortened -> dimmer too
        # splat 3 samples along the substep segment (continuous trails via decay + substeps)
        for f in (0.0, 0.34, 0.67):
            Q = P + step * f
            xi = np.clip(Q[:, 0].astype(int), 0, W - 1); yi = np.clip(Q[:, 1].astype(int), 0, H - 1)
            np.add.at(E, (yi[ok], xi[ok]), col[ok] * (a.gain / (a.substeps * 3)))
        P = P + step
        inb = bilinear(mask.astype(np.float32), P) > 0.5
        dead = ~inb | ~ok
        if dead.any():
            Pn, ln = spawn(int(dead.sum()), mask, twist)
            P[dead] = Pn; life[dead] = ln; age[dead] = 0
    age += 1
    old = age > life
    if old.any():
        Pn, ln = spawn(int(old.sum()), mask, twist)
        P[old] = Pn; life[old] = ln; age[old] = 0
    # bright particle heads
    Ehead = E.copy()
    xi = np.clip(P[:, 0].astype(int), 0, W - 1); yi = np.clip(P[:, 1].astype(int), 0, H - 1)
    if a.color == "twist":
        hcol = plasma(bilinear(twist, P))
    else:
        hcol = np.ones((len(P), 3), np.float32)
    np.add.at(Ehead, (yi, xi), hcol * 1.15)
    Esoft = cv2.GaussianBlur(Ehead, (0, 0), 0.8)
    hdr = d["img"] * a.dark * vignette(mask) + bloom(Esoft, thr=0.35)
    fr = to8(tonemap(hdr))
    out_frames.append(fr)
    if k in stills:
        cv2.imwrite(f"{a.out}/wind_{a.color}_f{k:03d}.png", fr[..., ::-1])
    if k % 20 == 0:
        print(f"frame {k}/{len(frames)} trailE={E.mean():.4f}", flush=True)

if len(frames) > 1:
    write_mp4(out_frames, f"{a.out}/wind_{a.color}.mp4", fps_of(a.dump))
    print(f"wrote {a.out}/wind_{a.color}.mp4", flush=True)
print("WIND DONE", flush=True)
