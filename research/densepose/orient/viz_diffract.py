"""ROUND 2 PBR #2 -- diffraction grating: the holographic suit.

Micro-grooves run ALONG the predicted tangent, so the grating vector is the predicted bitangent
b = n x t. Stam's diffraction model (SIGGRAPH 99): with H = l + v (unnormalised), the m-th order
reflects wavelength  lambda = d * (H . b_hat) / m.  Where that lands in [380,780] nm the pixel fires
that spectral colour -- rainbow bands whose ORIENTATION is set by the tangent field and which sweep
across the body as the light moves. The twist channel becomes literal colour physics: rotate the
grooves (twist) and the band pattern rotates with them, at fixed geometry and light.

  python -m orient.viz_diffract --dump DIR --out DIR [--video] [--frame -1] [--d 1.6]
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
ap.add_argument("--d", type=float, default=1.6, help="groove period (micrometres)")
ap.add_argument("--orders", type=int, default=3)
ap.add_argument("--gain", type=float, default=2.3)
ap.add_argument("--dark", type=float, default=0.10)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)


def wavelength_rgb(lam):
    """Approximate spectral colour for lam (nm), (...,) -> (...,3), zero outside 380-780."""
    l = lam
    r = np.zeros_like(l); g = np.zeros_like(l); b = np.zeros_like(l)
    def seg(m, lo, hi, fr, fg, fb):
        np.putmask(r, m, fr); np.putmask(g, m, fg); np.putmask(b, m, fb)
    m = (l >= 380) & (l < 440); seg(m, 380, 440, -(l - 440) / 60, 0.0, 1.0)
    m = (l >= 440) & (l < 490); seg(m, 440, 490, 0.0, (l - 440) / 50, 1.0)
    m = (l >= 490) & (l < 510); seg(m, 490, 510, 0.0, 1.0, -(l - 510) / 20)
    m = (l >= 510) & (l < 580); seg(m, 510, 580, (l - 510) / 70, 1.0, 0.0)
    m = (l >= 580) & (l < 645); seg(m, 580, 645, 1.0, -(l - 645) / 65, 0.0)
    m = (l >= 645) & (l <= 780); seg(m, 645, 780, 1.0, 0.0, 0.0)
    # intensity falloff at the ends of the visible range
    w = np.ones_like(l)
    m = (l >= 380) & (l < 420); np.putmask(w, m, 0.3 + 0.7 * (l - 380) / 40)
    m = (l > 700) & (l <= 780); np.putmask(w, m, 0.3 + 0.7 * (780 - l) / 80)
    out = np.stack([r, g, b], -1) * w[..., None]
    out[(l < 380) | (l > 780)] = 0
    return np.clip(out, 0, 1)


def render(d, k, NF):
    mask = erode_mask(d["mask"], 1)
    n = d["n_cam"]; t = d["t_cam"]
    tb = np.cross(n, t)                                          # grating vector (across the grooves)
    v = np.array([0, 0, -1.0], np.float32)
    az = np.radians(-60 + 120 * (k / max(NF - 1, 1))); el = np.radians(20)
    l = np.array([np.sin(az) * np.cos(el), np.sin(el), -np.cos(az) * np.cos(el)], np.float32)
    Hv = l + v                                                   # Stam: unnormalised half-vector
    u = (tb * Hv).sum(-1)                                        # component across the grooves
    E = np.zeros((*mask.shape, 3), np.float32)
    for m in range(1, a.orders + 1):
        lam = 1000.0 * a.d * np.abs(u) / m                       # micron * 1000 = nm
        E += wavelength_rgb(lam) * (a.gain / m)
    hn = Hv / np.linalg.norm(Hv)
    spec0 = np.clip((n * hn).sum(-1), 0, 1) ** 400               # zeroth order: mirror glint
    lam_l = np.clip((n * l).sum(-1), 0, 1)
    base = np.array([0.05, 0.055, 0.08], np.float32) * (0.4 + 0.6 * lam_l)[..., None] \
        + d["img"] * 0.10                                        # near-black vinyl base
    body = base + E * (0.25 + 0.75 * lam_l)[..., None] + spec0[..., None] * 3.0
    fm = feather(mask, 1.4)[..., None]
    hdr = d["img"] * a.dark * vignette(mask) * (1 - 0.9 * fm) + body * fm
    return to8(tonemap(bloom(hdr, thr=0.45)))


frames = frames_of(a.dump)
if a.video:
    if a.maxframes:
        frames = frames[:a.maxframes]
    out = []
    for k, fi in enumerate(frames):
        out.append(render(load(a.dump, fi), k, len(frames)))
        if k % 20 == 0:
            print(f"frame {k}/{len(frames)}", flush=True)
    write_mp4(out, f"{a.out}/diffract.mp4", fps_of(a.dump))
    print(f"wrote {a.out}/diffract.mp4", flush=True)
else:
    kk = len(frames) // 2 if a.frame < 0 else a.frame
    fr = render(load(a.dump, frames[kk]), kk, len(frames))
    cv2.imwrite(f"{a.out}/diffract_f{frames[kk]:04d}.png", fr[..., ::-1])
    print(f"wrote {a.out}/diffract_f{frames[kk]:04d}.png", flush=True)
print("DIFFRACT DONE", flush=True)
