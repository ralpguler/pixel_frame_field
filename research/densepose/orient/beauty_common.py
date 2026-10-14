"""Shared pieces for the direction-viz beauty contest renderers (wind / hatch / satin / LIC).

All renderers consume the FORWARD dump of field_dump.py. Conventions fixed there:
  camera x -> +image x, camera y -> -image y (up on screen), camera-facing normals have n_cam[2] < 0.
So the image-space 2D direction of a camera vector v is (v[0], -v[1]).
"""
import json
import os

import cv2
import numpy as np


def load(dump, fi):
    z = np.load(f"{dump}/f_{fi:04d}.npz")
    img = cv2.imread(f"{dump}/f_{fi:04d}.jpg")[..., ::-1].astype(np.float32) / 255
    d = {k: z[k].astype(np.float32) if z[k].dtype == np.float16 else z[k] for k in z.files}
    d["img"] = img
    d["mask"] = d["mask"].astype(bool)
    return d


def frames_of(dump):
    return json.load(open(f"{dump}/meta.json"))["frames"]


def fps_of(dump):
    return json.load(open(f"{dump}/meta.json"))["fps"]


def field2d(v):
    """camera 3-vector field (H,W,3) -> image-space 2D direction (H,W,2), y flipped to pixel-down."""
    return np.stack([v[..., 0], -v[..., 1]], -1)


def bilinear(F, P):
    """Sample F (H,W,C or H,W) at float pixel positions P (N,2) as (x,y). Clamped borders."""
    H, W = F.shape[:2]
    x = np.clip(P[:, 0], 0, W - 1.001); y = np.clip(P[:, 1], 0, H - 1.001)
    x0 = x.astype(int); y0 = y.astype(int); fx = x - x0; fy = y - y0
    if F.ndim == 2:
        F = F[..., None]
    out = (F[y0, x0] * ((1 - fx) * (1 - fy))[:, None] + F[y0, x0 + 1] * (fx * (1 - fy))[:, None]
           + F[y0 + 1, x0] * ((1 - fx) * fy)[:, None] + F[y0 + 1, x0 + 1] * (fx * fy)[:, None])
    return out[:, 0] if out.shape[1] == 1 else out


def erode_mask(m, px=2):
    return cv2.erode(m.astype(np.uint8), np.ones((2 * px + 1, 2 * px + 1), np.uint8)).astype(bool)


def feather(m, px=3):
    return cv2.GaussianBlur(m.astype(np.float32), (0, 0), px)


def vignette(mask, gain=0.75):
    """Radial falloff centred on the body (mask centroid), full brightness over the body extent.
    Buries the bright-studio washout AND the crop's edge-padding smears without touching the person."""
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.mean(), xs.mean()) if len(ys) else (H / 2, W / 2)
    rr = max(ys.ptp() if len(ys) else H, xs.ptp() if len(xs) else W, 1) * 0.72
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.hypot(yy - cy, xx - cx) / rr
    return np.clip(1 - gain * np.clip(d - 0.55, 0, None) ** 1.5, 0.06, 1).astype(np.float32)[..., None]


def bloom(hdr, sigmas=(3, 9, 27), gains=(0.5, 0.35, 0.25), thr=0.6):
    """Multi-scale Gaussian bloom on the HDR excess above thr."""
    exc = np.clip(hdr - thr, 0, None)
    out = hdr.copy()
    for s, g in zip(sigmas, gains):
        out += g * cv2.GaussianBlur(exc, (0, 0), s)
    return out


def tonemap(hdr):
    """Filmic-ish curve + gamma; expects linear-ish HDR >= 0."""
    x = np.clip(hdr, 0, None)
    y = x * (1.0 + x / 4.0) / (1.0 + x)
    return np.clip(y, 0, 1) ** (1 / 2.2)


def to8(x):
    return (np.clip(x, 0, 1) * 255).astype(np.uint8)


def write_mp4(frames, path, fps):
    import imageio
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wr = imageio.get_writer(path, fps=fps, codec="libx264", quality=8,
                            pixelformat="yuv420p", macro_block_size=8)
    for f in frames:
        wr.append_data(f)
    wr.close()


PLASMA = None


def plasma(t):
    """t in [0,1] (N,) -> RGB (N,3) float."""
    global PLASMA
    if PLASMA is None:
        import matplotlib.cm as cm
        PLASMA = cm.plasma(np.linspace(0, 1, 256))[:, :3].astype(np.float32)
    return PLASMA[np.clip((t * 255), 0, 255).astype(int)]


def twilight(t):
    import matplotlib.cm as cm
    return cm.twilight(np.clip(t, 0, 1))[..., :3].astype(np.float32)
