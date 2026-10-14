"""BEAUTY CONTEST #2 -- pen-and-ink cross-hatching driven by the predicted frame field.

Artists hatch along surface orientation and cross-hatch into shadow; here the network's per-pixel
frame does both: stroke DIRECTION = image-projected predicted tangent (streamline through each
seed), CROSS strokes = predicted bitangent, and stroke DENSITY = Lambert shading of the predicted
normal. Foreshortening is honest -- a tangent pointing at the camera shrinks its stroke to a dot.
Strokes sit on a fixed jittered image grid (deterministic per cell) so video is watchable without
tracking anything.

Styles: ink (dark strokes on paper, faint photo ghost)  |  neon (glowing strokes on darkened photo)

  python -m orient.viz_hatch --dump DIR --out DIR [--frame -1 | --video] [--style ink]
"""
import argparse
import os

import cv2
import numpy as np

from orient.beauty_common import (bilinear, bloom, erode_mask, feather, field2d, frames_of,
                                  fps_of, load, to8, tonemap, write_mp4)

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=int, default=-1)
ap.add_argument("--video", action="store_true")
ap.add_argument("--maxframes", type=int, default=0)
ap.add_argument("--style", default="ink", choices=["ink", "neon"])
ap.add_argument("--spacing", type=float, default=0.0095, help="hatch grid spacing as fraction of side")
ap.add_argument("--slen", type=float, default=2.4, help="stroke length in units of spacing")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SS = 3                                                          # supersampling


def streamline(seed, dir2d, half, h=1.5):
    """Integrate ±half px along dir2d from seed -> polyline points."""
    pts = [seed]
    for sgn in (1.0, -1.0):
        P = seed.copy()[None]
        d_prev = None
        for _ in range(int(half / h)):
            v = bilinear(dir2d, P)
            nv = np.linalg.norm(v)
            if nv < 0.03:
                break
            v = v / nv
            if d_prev is not None and (v * d_prev).sum() < 0:
                v = -v
            P = P + sgn * v * h
            d_prev = v
            if sgn > 0:
                pts.append(P[0].copy())
            else:
                pts.insert(0, P[0].copy())
    return np.array(pts)


def render(d, rng_grid):
    mask = erode_mask(d["mask"], 1)
    H, W = mask.shape
    sp = max(5, int(a.spacing * W))
    t2d = field2d(d["t_cam"]); b2d = field2d(np.cross(d["n_cam"], d["t_cam"]))
    fore_t = np.linalg.norm(t2d, axis=-1)
    dir_t = t2d / np.clip(fore_t, 1e-6, None)[..., None]
    fore_b = np.linalg.norm(b2d, axis=-1)
    dir_b = b2d / np.clip(fore_b, 1e-6, None)[..., None]
    Ldir = np.array([-0.42, 0.5, -0.76]); Ldir /= np.linalg.norm(Ldir)
    tone = np.clip((d["n_cam"] * Ldir).sum(-1), 0, 1) ** 1.35   # 1 = lit, 0 = shadow; gamma spreads layers
    # deterministic jittered grid (stable across frames)
    gy, gx = np.mgrid[sp // 2:H:sp, sp // 2:W:sp]
    cells = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)
    jit = (np.stack([np.sin(cells[:, 0] * 12.9898 + cells[:, 1] * 78.233),
                     np.sin(cells[:, 0] * 39.346 + cells[:, 1] * 11.135)], 1) * 43758.5453) % 1.0
    seeds = cells + (jit - 0.5) * sp * 0.9
    on = bilinear(mask.astype(np.float32), seeds) > 0.5
    seeds = seeds[on]
    s_tone = bilinear(tone, seeds)
    s_ft = bilinear(fore_t, seeds); s_fb = bilinear(fore_b, seeds)
    ink = np.zeros((H * SS, W * SS), np.float32)
    half0 = a.slen * sp / 2

    def put(sel, dirfield, fore, alpha, thick=2):
        for i in np.nonzero(sel)[0]:
            half = half0 * (0.35 + 0.65 * min(fore[i], 1.0))
            pl = streamline(seeds[i], dirfield, half)
            if len(pl) < 2:
                continue
            cv2.polylines(ink, [np.round(pl * SS).astype(np.int32)], False, alpha,
                          thickness=thick, lineType=cv2.LINE_AA)

    put(s_tone < 0.92, dir_t, s_ft, 0.78)                       # layer 1: everywhere, along tangent
    off = seeds + np.stack([jit[on][:, 1], -jit[on][:, 0]], 1) * sp * 0.5
    seeds2 = off; s2_on = bilinear(mask.astype(np.float32), seeds2) > 0.5
    sv = seeds; seeds = seeds2                                  # reuse put() on offset seeds
    put(s2_on & (bilinear(tone, seeds2) < 0.60), dir_t, bilinear(fore_t, seeds2), 0.62)
    seeds = sv
    put(s_tone < 0.38, dir_b, s_fb, 0.68)                       # cross-hatch into shadow
    put(s_tone < 0.20, dir_t, s_ft, 0.8, thick=3)               # darkest: double down
    # occluding contour from the mask
    cts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    cv2.polylines(ink, [(c[:, 0] * SS).astype(np.int32) for c in cts], True, 0.95, thickness=3,
                  lineType=cv2.LINE_AA)
    ink = cv2.resize(ink, (W, H), interpolation=cv2.INTER_AREA)
    if a.style == "ink":
        rngp = np.random.default_rng(7)
        paper = 0.96 - 0.05 * cv2.GaussianBlur(rngp.random((H, W)).astype(np.float32), (0, 0), 1.2)
        yy, xx = np.mgrid[0:H, 0:W]
        vig = 1 - 0.25 * (((xx / W - 0.5) ** 2 + (yy / H - 0.5) ** 2) * 2.2)
        ghost = d["img"].mean(-1) * 0.18
        base = np.clip(paper * vig - ghost * feather(mask, 8), 0, 1)
        col = np.array([0.10, 0.11, 0.19], np.float32)          # blue-black ink
        outp = base[..., None] * (1 - np.clip(ink, 0, 1)[..., None]) \
            + col * np.clip(ink, 0, 1)[..., None] * base[..., None]
        return to8(outp)
    from orient.beauty_common import vignette
    hdr = d["img"] * 0.13 * vignette(mask) + bloom(np.clip(ink, 0, 1.6)[..., None]
                                                   * np.array([0.55, 0.85, 1.0], np.float32) * 1.6, thr=0.4)
    return to8(tonemap(hdr))


frames = frames_of(a.dump)
if a.video:
    if a.maxframes:
        frames = frames[:a.maxframes]
    outf = []
    for k, fi in enumerate(frames):
        outf.append(render(load(a.dump, fi), None))
        if k % 10 == 0:
            print(f"frame {k}/{len(frames)}", flush=True)
    write_mp4(outf, f"{a.out}/hatch_{a.style}.mp4", fps_of(a.dump))
    print(f"wrote {a.out}/hatch_{a.style}.mp4", flush=True)
else:
    fi = frames[len(frames) // 2 if a.frame < 0 else a.frame]
    fr = render(load(a.dump, fi), None)
    cv2.imwrite(f"{a.out}/hatch_{a.style}_f{fi:04d}.png", fr[..., ::-1])
    print(f"wrote {a.out}/hatch_{a.style}_f{fi:04d}.png", flush=True)
print("HATCH DONE", flush=True)
