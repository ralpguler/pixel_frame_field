"""Collect the beauty-contest outputs into web-ready assets (body-cropped, downscaled).

Stills: crop to the body bbox (mask union + margin, square), resize to 640px, JPEG q88.
Videos: same fixed crop across the whole clip, width 512, mp4 q6 -> small enough to embed.

  python -m orient.beauty_report_assets --out ~/data/dp_viz_compare/beauty/report_assets
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--beauty", default=os.path.expanduser("~/data/dp_viz_compare/beauty"))
ap.add_argument("--out", default=os.path.expanduser("~/data/dp_viz_compare/beauty/report_assets"))
ap.add_argument("--size", type=int, default=640)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

CLIPS = ["15731089", "15153663", "28513779", "37828157"]


def body_box(dump):
    """Union of body masks (sampled frames) + margin -> square crop box."""
    fs = sorted(glob.glob(f"{dump}/f_*.npz"))
    U = None
    for f in fs[:: max(1, len(fs) // 10)]:
        m = np.load(f)["mask"]
        U = m if U is None else (U | m)
    ys, xs = np.nonzero(U)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    s = max(ys.ptp(), xs.ptp()) * 0.62 + 0.06 * U.shape[0]
    H, W = U.shape
    s = min(s, H / 2, W / 2)
    x0, y0 = int(np.clip(cx - s, 0, W - 2 * s)), int(np.clip(cy - s, 0, H - 2 * s))
    return x0, y0, int(2 * s)


for c in CLIPS:
    dump = f"{a.beauty}/{c}"
    if not os.path.exists(f"{dump}/meta.json"):
        continue
    x0, y0, s = body_box(dump)
    for p in sorted(glob.glob(f"{dump}/render/*.png")) + sorted(glob.glob(f"{dump}/f_0075.jpg")):
        im = cv2.imread(p)
        if im is None or im.shape[0] < s:
            continue
        crop = im[y0:y0 + s, x0:x0 + s]
        crop = cv2.resize(crop, (a.size, a.size), interpolation=cv2.INTER_AREA)
        name = os.path.basename(p).replace(".png", "").replace(".jpg", "_photo")
        cv2.imwrite(f"{a.out}/{c}_{name}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 88])
    for v in sorted(glob.glob(f"{dump}/render/*.mp4")):
        import imageio.v2 as imageio
        rd = imageio.get_reader(v)
        fps = rd.get_meta_data().get("fps", 16)
        wr = imageio.get_writer(f"{a.out}/{c}_{os.path.basename(v)}", fps=fps, codec="libx264",
                                quality=6, pixelformat="yuv420p", macro_block_size=8)
        for fr in rd:
            crop = fr[y0:y0 + s, x0:x0 + s]
            wr.append_data(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA))
        wr.close(); rd.close()
    print(f"{c}: crop=({x0},{y0},{s})", flush=True)

tot = sum(os.path.getsize(f) for f in glob.glob(f"{a.out}/*"))
print(f"ASSETS DONE -> {a.out} ({tot/1e6:.1f} MB)", flush=True)
