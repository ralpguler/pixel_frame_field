"""Preprocess BEDLAM2 correspondence -> WebDataset shards for DensePose training.

Uses the ALREADY-SAVED face index + barycentric crops (no re-rendering). For every
instance we store a canonical S×S sample:
  rgb.jpg   : expanded-box RGB crop (EXPAND× context, square) warped from the movie frame
  corr.png  : RGBA pack of face index + barycentric, the stored tight crop warped into the
              SAME expanded-box canvas with the SAME affine (so rgb and labels align).
              Part/UV are derived on-the-fly at train time via the SMPL-X face LUTs.
  meta.json : uid, seq, sub, frame, instance, box [x0,y0,side], gender

CPU-only (movie decode + image warps). Sharding: sequences seqs[SHARD::NUM_SHARDS]
(resumable via per-seq done markers). VAL_SEQS held out (sequence-disjoint).
"""
import io
import json
import os
import sys
import tarfile

import cv2
import numpy as np
from PIL import Image

import imageio.v3 as iio
import webdataset as wds


def iter_frames_mem(data, want):
    """Yield (frame_idx, rgb) for frames in `want`, decoding mp4 bytes from memory
    (no temp files -> robust to flaky node-local scratch)."""
    want = set(want)
    hi = max(want)
    for i, fr in enumerate(iio.imiter(data, plugin="pyav")):
        if i in want:
            yield i, fr
        if i >= hi:
            break

ROOT = os.path.expanduser("~/data/bedlam2")
CORR = f"{ROOT}/correspondence"
MOVIES = f"{ROOT}/downloader"
OUT = f"{ROOT}/dp_shards"

# Store crops at NATIVE resolution (no resampling) so training resolution stays a
# free dataloader knob. Only downscale crops whose long side exceeds MAX_SIDE.
EXPAND = float(os.environ.get("DP_EXPAND", 2.0))     # generous context (box-jitter headroom)
MAX_SIDE = int(os.environ.get("DP_MAX_SIDE", 1280))  # cap stored long side; never upscale
MIN_SIDE = int(os.environ.get("DP_MIN_SIDE", 28))
PER_SHARD = int(os.environ.get("DP_PER_SHARD", 4000))
BG = (255, 255, 0, 0)        # FACE_BG sentinel packing -> unpacks to face=-1

VAL_SEQS = {
    "20240502_5_200_citysample_dolly",
    "20240617_10_500_hdri",
    "20240711_5-10_250_busstation_orbit_zoom",
    "20240611_5_250_archmodelsvol8_dolly",
}


def expand_box(x0, y0, x1, y1):
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = EXPAND * max(x1 - x0, y1 - y0)
    return cx - side / 2, cy - side / 2, side


def make_sample(frame_rgb, corr_rgba, bbox, bx0, by0, side):
    """Warp movie frame + stored tight corr crop into one square canvas at NATIVE
    resolution (cz = min(side, MAX_SIDE); downscale only when side > MAX_SIDE).
    frame_rgb (Hf,Wf,3); corr_rgba (h,w,4) tight crop located at bbox in the frame."""
    cz = int(round(min(side, MAX_SIDE)))
    s = cz / side                                      # <=1 (never upscale)
    x0, y0 = bbox[0], bbox[1]
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    Mrgb = np.array([[s, 0, -bx0 * s], [0, s, -by0 * s]], np.float32)
    rgb = cv2.warpAffine(frame_rgb, Mrgb, (cz, cz), flags=interp,
                         borderMode=cv2.BORDER_REPLICATE)
    # corr tight-crop pixel (u,v)->frame (x0+u,y0+v)->canvas; same s & origin
    Mcorr = np.array([[s, 0, (x0 - bx0) * s], [0, s, (y0 - by0) * s]], np.float32)
    rgba = cv2.warpAffine(corr_rgba, Mcorr, (cz, cz), flags=cv2.INTER_NEAREST,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=BG)
    if not (rgba[..., 0] != 255).any() and not (rgba[..., 1] != 255).any():
        return None                                   # nothing visible -> skip
    return rgb, rgba


def process_sequence(seq, writers):
    ann_path = f"{CORR}/annotations/{seq}.jsonl"
    if not os.path.exists(ann_path):
        return 0
    split = "val" if seq in VAL_SEQS else "train"
    wr = writers[split]
    groups = {}
    for line in open(ann_path):
        r = json.loads(line)
        x0, y0, x1, y1 = r["bbox"]
        if max(x1 - x0, y1 - y0) < MIN_SIDE:
            continue
        groups.setdefault((r["sub"], r["frame"]), []).append(r)

    tar = tarfile.open(f"{MOVIES}/{seq}_mp4.tar")
    members = {os.path.basename(m.name): m for m in tar.getmembers()
               if m.name.endswith(".mp4")}
    n = 0
    for sub in sorted({s for s, _ in groups}):
        m = members.get(f"{sub}.mp4")
        if m is None:
            print(f"  {seq}/{sub}: no mp4 member", flush=True); continue
        want = sorted({fn for (s2, fn) in groups if s2 == sub})
        try:
            data = tar.extractfile(m).read()
            frame_iter = list(iter_frames_mem(data, want))
        except Exception as e:
            print(f"  {seq}/{sub}: decode failed {e!r}", flush=True); continue
        for fn, frame in frame_iter:
            for r in groups.get((sub, fn), []):
                png = f"{CORR}/{r['png']}"
                if not os.path.exists(png):
                    continue
                corr = np.array(Image.open(png))               # (h,w,4)
                out = make_sample(frame, corr, r["bbox"], *expand_box(*r["bbox"]))
                if out is None:
                    continue
                rgb, rgba = out
                jpg = cv2.imencode(".jpg", rgb[..., ::-1],
                                   [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                pbuf = io.BytesIO(); Image.fromarray(rgba, "RGBA").save(pbuf, "PNG")
                bx0, by0, side = expand_box(*r["bbox"])
                meta = {"uid": r["uid"], "seq": seq, "sub": sub, "frame": fn,
                        "instance": r["instance"], "box": [bx0, by0, side],
                        "gender": r["gender"]}
                wr.write({"__key__": r["uid"].replace("/", "_"),
                          "rgb.jpg": jpg, "corr.png": pbuf.getvalue(),
                          "meta.json": json.dumps(meta).encode()})
                n += 1
    tar.close()
    return n


def main():
    shard = int(os.environ.get("PEXELS_SHARD", 0))
    nshard = int(os.environ.get("PEXELS_NUM_SHARDS", 1))
    for d in ("train", "val", "_done"):
        os.makedirs(f"{OUT}/{d}", exist_ok=True)
    import glob
    seqs = sorted(os.path.basename(p)[:-6]
                  for p in glob.glob(f"{CORR}/annotations/*.jsonl"))
    seqs = seqs[shard::nshard]
    writers = {
        "train": wds.ShardWriter(f"{OUT}/train/shard-{shard:03d}-%06d.tar",
                                 maxcount=PER_SHARD, verbose=0),
        "val": wds.ShardWriter(f"{OUT}/val/shard-{shard:03d}-%06d.tar",
                               maxcount=PER_SHARD, verbose=0),
    }
    total = 0
    for seq in seqs:
        done = f"{OUT}/_done/{seq}.json"
        if os.path.exists(done):
            print(f"skip {seq} (done)", flush=True); continue
        n = process_sequence(seq, writers)
        json.dump({"seq": seq, "n": n}, open(done, "w"))
        total += n
        print(f"[shard {shard}] {seq}: {n} samples (total {total})", flush=True)
    for w in writers.values():
        w.close()
    print(f"[shard {shard}] DONE total={total}", flush=True)


if __name__ == "__main__":
    main()
