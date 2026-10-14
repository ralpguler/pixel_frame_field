"""Visualize the GT and what the dataloader feeds the model, the FAITHFUL way:
 - exact native-resolution GT (no label resampling): input | part | UV
 - the actual training input with sampled GT points scattered (colored by part),
   which is what the loss uses (predictions are grid_sampled at these points).

No nearest interpolation of labels anywhere.
"""
import io
import json
import os
import sys
import tarfile
import tempfile

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/bedlam")
sys.path.insert(0, "/home/u6ga/raguler.u6ga/codes/wan/densepose")
from encode_correspondence import iter_mp4_frames
from preprocess_shards import CORR, MOVIES, expand_box, make_sample
from data import DPDecode, dense_gt, _PAL

ASSETS = os.path.expanduser("~/data/bedlam2/dp_assets")
SEQ = os.environ.get("VIZ_SEQ", "20240506_10_200_hdri")
INPUT = 384


def gather(seq, n=8):
    """Return n (rgb_native, rgba_native) crops from real saved data."""
    recs = [json.loads(l) for l in open(f"{CORR}/annotations/{seq}.jsonl")]
    recs = [r for r in recs if max(r["bbox"][2] - r["bbox"][0],
                                   r["bbox"][3] - r["bbox"][1]) >= 60]
    recs = recs[:: max(1, len(recs) // (n * 3))]
    tar = tarfile.open(f"{MOVIES}/{seq}_mp4.tar")
    members = {os.path.basename(m.name): m for m in tar.getmembers()
               if m.name.endswith(".mp4")}
    cache, out = {}, []
    for r in recs:
        if len(out) >= n:
            break
        sub = r["sub"]
        if sub not in cache:
            m = members.get(f"{sub}.mp4")
            if m is None:
                continue
            tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
            tmp.write(tar.extractfile(m).read()); tmp.close(); cache[sub] = tmp.name
        fr = dict(iter_mp4_frames(cache[sub], [r["frame"]])).get(r["frame"])
        if fr is None:
            continue
        corr = np.array(Image.open(f"{CORR}/{r['png']}"))
        s = make_sample(fr, corr, r["bbox"], *expand_box(*r["bbox"]))
        if s:
            out.append(s)
    for v in cache.values():
        os.unlink(v)
    tar.close()
    return out


def panel_exact(rgb, rgba, h=384):
    """Exact native-res GT: [input | part | UV], composite downscaled only for display."""
    part, uv, fg = dense_gt(rgba)
    pov = rgb.copy(); pov[fg] = (0.45 * rgb[fg] + 0.55 * _PAL[part[fg]]).astype(np.uint8)
    uvi = np.zeros_like(rgb)
    uvi[..., 0] = (uv[..., 0] * 255).astype(np.uint8)
    uvi[..., 1] = (uv[..., 1] * 255).astype(np.uint8)
    uvi[fg, 2] = 128
    uvi = np.where(fg[..., None], uvi, rgb)
    col = np.concatenate([rgb, pov, uvi], 1)
    s = h / rgb.shape[0]
    return cv2.resize(col, (int(col.shape[1] * s), h), interpolation=cv2.INTER_AREA)


def panel_points(rgb, rgba, dec):
    """Actual training input + sampled GT points (colored by part) scattered on it."""
    jpg = cv2.imencode(".jpg", rgb[..., ::-1])[1].tobytes()
    b = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(b, "PNG")
    d = dec((jpg, b.getvalue(), b"{}"))
    img = (d["img"].permute(1, 2, 0).numpy() * 255).astype(np.uint8).copy()
    Z = img.shape[0]
    pts = d["pts"].numpy(); part = d["part"].numpy(); fg = d["fg"].numpy()
    xs = ((pts[:, 0] + 1) / 2 * Z).astype(int).clip(0, Z - 1)
    ys = ((pts[:, 1] + 1) / 2 * Z).astype(int).clip(0, Z - 1)
    scat = img.copy()
    for x, y, p, f in zip(xs, ys, part, fg):
        if f:
            cv2.circle(scat, (x, y), 1, tuple(int(c) for c in _PAL[p]), -1)
    return np.concatenate([img, scat], 1)


def main():
    samples = gather(SEQ, n=8)
    print(f"{SEQ}: {len(samples)} instances")
    # exact native-res GT
    rows = [panel_exact(rgb, rgba) for rgb, rgba in samples]
    Image.fromarray(np.concatenate(rows, 0)).save(f"{ASSETS}/dl_gt_exact.png")
    # training input + sampled points
    dec = DPDecode(input_size=INPUT, train=True, n_points=1500)
    rows = [panel_points(rgb, rgba, dec) for rgb, rgba in samples]
    Image.fromarray(np.concatenate(rows, 0)).save(f"{ASSETS}/dl_points.png")
    print("wrote dl_gt_exact.png (input|part|UV exact) and dl_points.png "
          "(input | sampled GT points) to", ASSETS)


if __name__ == "__main__":
    main()
