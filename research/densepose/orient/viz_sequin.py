"""ROUND 2 PBR #1 -- sequins: body-anchored micro-mirrors, oriented by the predicted frame.

Sequins are anchored in CANONICAL space -- a jittered grid in each part's (fold-free) chart UV,
with per-part pitch chosen so sequins are the same PHYSICAL size everywhere -- and rendered as a
forward pullback: each pixel knows its predicted (part,uv), hence which sequin disc it lies on.
Each sequin carries a fixed random cant in the TANGENT plane, so its mirror normal is
    n_s = normalize(n_cam + cant_u * t_cam + cant_v * b_cam),
which co-rotates with the FULL predicted R (twist included). A sequin flashes exactly when the
sweeping light crosses its mirror direction, so the sparkle timing pattern across the video is a
physical readout of the rotation field -- wrong rotations would flash chaotically.

  python -m orient.viz_sequin --dump DIR --out DIR [--video] [--frame -1] [--mm 9]
"""
import argparse
import os

import cv2
import numpy as np

from orient.beauty_common import (bloom, erode_mask, feather, frames_of, fps_of, load, to8,
                                  tonemap, vignette, write_mp4)

ap = argparse.ArgumentParser()
ap.add_argument("--dump", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--frame", type=int, default=-1)
ap.add_argument("--video", action="store_true")
ap.add_argument("--maxframes", type=int, default=0)
ap.add_argument("--mm", type=float, default=15.0, help="sequin diameter (mm on the template)")
ap.add_argument("--cant", type=float, default=0.18, help="random mirror cant strength")
ap.add_argument("--power", type=float, default=1200.0, help="glint sharpness")
ap.add_argument("--dark", type=float, default=0.12)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

A = os.path.expanduser("~/data/bedlam2/dp_assets")
fuv = np.load(f"{A}/smplx_face_uv_v2.npy")
fpart = np.load(f"{A}/smplx_face_to_dp24.npy").astype(int)
vt = np.load(f"{A}/smplx_v_template.npy")
FA = np.load(f"{A}/smplx_faces.npy").astype(int)

# physical scale of each chart: mm per UV unit (median 3D/UV edge-length ratio)
SCALE = np.ones(25, np.float32) * 1000
for p in range(1, 25):
    fm = fpart == p
    if not fm.any():
        continue
    e3d = np.linalg.norm(vt[FA[fm][:, 1]] - vt[FA[fm][:, 0]], axis=1) * 1000
    euv = np.linalg.norm(fuv[fm][:, 1] - fuv[fm][:, 0], axis=1)
    SCALE[p] = np.median(e3d / np.clip(euv, 1e-9, None))


def hash2(ix, iy, p, salt=0.0):
    h = np.sin(ix * 127.1 + iy * 311.7 + p * 74.7 + salt * 269.5) * 43758.5453
    return h - np.floor(h)


def render(d, k, NF):
    mask = erode_mask(d["mask"], 1)
    H, W = mask.shape
    part = d["part"].astype(np.int32)
    U, Vv = d["uv"][0].astype(np.float32), d["uv"][1].astype(np.float32)
    n = d["n_cam"]; t = d["t_cam"]; b = np.cross(n, t)
    pitch = (a.mm / SCALE)[part]                                # per-pixel UV pitch of its part grid
    ix = np.floor(U / pitch); iy = np.floor(Vv / pitch)
    jx = hash2(ix, iy, part, 1.0) - 0.5; jy = hash2(ix, iy, part, 2.0) - 0.5
    cu = (ix + 0.5 + 0.6 * jx) * pitch; cv = (iy + 0.5 + 0.6 * jy) * pitch
    du = (U - cu) * SCALE[part]; dv = (Vv - cv) * SCALE[part]   # offset from sequin centre, in mm
    r = np.hypot(du, dv)
    disc = mask & (r < a.mm * 0.44)
    # per-sequin identity -> cant + colour + material
    h1 = hash2(ix, iy, part, 3.0); h2 = hash2(ix, iy, part, 4.0); h3 = hash2(ix, iy, part, 5.0)
    ang = h1 * 2 * np.pi; mag = a.cant * (0.4 + 0.6 * h2)
    ns = n + (mag * np.cos(ang))[..., None] * t + (mag * np.sin(ang))[..., None] * b
    ns /= np.clip(np.linalg.norm(ns, axis=-1, keepdims=True), 1e-9, None)
    gold = h3 > 0.55
    col = np.where(gold[..., None], np.array([1.0, 0.78, 0.38], np.float32),
                   np.array([0.85, 0.88, 0.95], np.float32))
    # lights: one sweeping key + one static fill (view = (0,0,-1), ortho)
    v = np.array([0, 0, -1.0], np.float32)
    az = np.radians(-55 + 110 * (k / max(NF - 1, 1))); el = np.radians(18)
    l1 = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)], np.float32)
    l2 = np.array([-0.5, 0.6, -0.63], np.float32); l2 /= np.linalg.norm(l2)
    hdr = d["img"] * a.dark * vignette(mask)
    lam = np.clip((n * l1).sum(-1), 0, 1)
    base = col * (0.11 + 0.30 * lam)[..., None]                 # sequin body when not flashing
    rim = (1 - np.abs(-n[..., 2])) ** 3 * 0.10
    E = np.zeros((H, W, 3), np.float32)
    for l, gain in ((l1, 34.0), (l2, 9.0)):
        hv = l + v; hv = hv / np.linalg.norm(hv)
        spec = np.clip((ns * hv).sum(-1), 0, 1) ** a.power
        E += (spec * gain)[..., None] * col
    body = np.where(disc[..., None], base + rim[..., None] + E,
                    d["img"] * 0.35 * np.array([0.8, 0.8, 0.9], np.float32))  # fabric between sequins
    fm = feather(mask, 1.4)[..., None]
    hdr = hdr * (1 - 0.9 * fm) + body * fm
    return to8(tonemap(bloom(hdr, thr=0.5)))


frames = frames_of(a.dump)
if a.video:
    if a.maxframes:
        frames = frames[:a.maxframes]
    out = []
    for k, fi in enumerate(frames):
        out.append(render(load(a.dump, fi), k, len(frames)))
        if k % 20 == 0:
            print(f"frame {k}/{len(frames)}", flush=True)
    write_mp4(out, f"{a.out}/sequin.mp4", fps_of(a.dump))
    print(f"wrote {a.out}/sequin.mp4", flush=True)
else:
    kk = len(frames) // 2 if a.frame < 0 else a.frame
    fr = render(load(a.dump, frames[kk]), kk, len(frames))
    cv2.imwrite(f"{a.out}/sequin_f{frames[kk]:04d}.png", fr[..., ::-1])
    print(f"wrote {a.out}/sequin_f{frames[kk]:04d}.png", flush=True)
print("SEQUIN DONE", flush=True)
