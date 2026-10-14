"""BEAUTY CONTEST #3 (stage 1) -- bake the predicted G-buffer for the Blender satin relight.

Writes, per frame, 16-bit PNGs in BLENDER camera convention (x right, y up, camera-facing = +z;
ours is camera-facing = -z, so z is flipped):
  nrm_XXXX.png   (n+1)/2      predicted camera-space surface normal
  tan_XXXX.png   (t+1)/2      predicted camera-space tangent (drives the anisotropic highlight;
                              this is where the TWIST DoF becomes visible -- no normal map has it)
  msk_XXXX.png   foreground

blender_satin.py then builds a camera-facing plane whose Principled BSDF reads these maps and
renders a real light sweep with Cycles.

  python -m orient.viz_satin_bake --dump DIR --out DIR [--maxframes 150]
"""
import argparse
import os

import cv2
import numpy as np

from orient.beauty_common import erode_mask, feather, frames_of, load

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--maxframes", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

FLIP = np.array([1.0, 1.0, -1.0], np.float32)


def png16(v):
    return np.clip((v * 0.5 + 0.5) * 65535, 0, 65535).astype(np.uint16)


frames = frames_of(a.dump)
if a.maxframes:
    frames = frames[:a.maxframes]
for k, fi in enumerate(frames):
    d = load(a.dump, fi)
    m = erode_mask(d["mask"], 1)
    n = d["n_cam"] * FLIP
    t = d["t_cam"] * FLIP
    # fill background with camera-facing normal / zero tangent so the mask edge doesn't ring
    n[~m] = [0, 0, 1]; t[~m] = [1, 0, 0]
    cv2.imwrite(f"{a.out}/nrm_{fi:04d}.png", png16(n)[..., ::-1])
    cv2.imwrite(f"{a.out}/tan_{fi:04d}.png", png16(t)[..., ::-1])
    cv2.imwrite(f"{a.out}/msk_{fi:04d}.png", (feather(m, 1.2) * 255).astype(np.uint8))
    # metallic map: full metal on the body, low on the head (23/24) so the face stays photographic
    met = np.where(np.isin(d["part"], (23, 24)), 0.22, 1.0).astype(np.float32)
    met[~m] = 0
    cv2.imwrite(f"{a.out}/met_{fi:04d}.png", (cv2.GaussianBlur(met, (0, 0), 1.5) * 255).astype(np.uint8))
    if k % 20 == 0:
        print(f"baked {k}/{len(frames)}", flush=True)
print(f"BAKE DONE -> {a.out} ({len(frames)} frames)", flush=True)
