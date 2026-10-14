"""Re-render the BEDLAM2 correspondence GT into a FIXED 512x512 crop canvas, per instance.

WHY. Today the correspondence is rasterised at the source frame's resolution (1280x720) and then cropped,
which welds the LABEL resolution to the IMAGE resolution. A BEDLAM2 person is only ~239 px tall, so the
loss ends up supervised on a ~311 px crop -- 61% of the network's own 512^2 output -- and every head then
needs a different resampling hack to cope (ROT_CAP, CSC_CAP, cv2.resize of 6D rotations, ...). That last
one was a real bug: bilinearly resizing rotations AVERAGES two different surfaces across an occlusion
boundary; 3.2% of MAMMA's rotation labels were wrong by >30 deg, up to a 180 deg flip.

The label is ANALYTIC -- it comes from rasterising a mesh with a known camera -- so its resolution is a
free parameter. Render it once, per instance, into a fixed 512^2 canvas, and the GT is never resampled
again: no caps, no cv2.resize of labels, no nearest-vs-bilinear debate. The prediction is warped INTO this
GT crop space (legitimate: the network's output is an inherently smooth field, so warping only EVALUATES
it), and every loss/threshold/gather/pool happens there.

Augmentation still works: the GT crop is a FIXED expansion of the tight body box, and every jittered image
crop is a WIDER expansion of the same box, so the foreground is inside both and the prediction has full
support over the GT crop.

Reads correspondence/posed3d/<seq>.npz (SMPL-X params + per-instance camera) and annotations/<seq>.jsonl
(tight bboxes). Does NOT touch the mp4s -- the RGB in the existing shards stays as it is.

  python render_gt512.py --shard 0 --nshards 16
"""
import argparse
import io
import json
import os
import tarfile
import time
from collections import defaultdict

import cv2
import numpy as np
import torch

from encode_correspondence import pack_rgba
from render_correspondence import _rasterize, pose_instance, world_to_cam

ROOT = os.path.expanduser("~/data/bedlam2")
CORR = f"{ROOT}/correspondence"
OUT = f"{ROOT}/corr512"
CANVAS = 512
GT_EXPAND = 1.2          # tight-box expansion of the GT crop. MUST be <= the tightest train image crop
                         # (data.py jitter_expand=(1.2,1.4)) so the GT is always inside the network's view.


def crop_K(K, bbox, canvas=CANVAS, expand=GT_EXPAND):
    """Intrinsics of the square GT crop, as seen by the frame camera. Returns (K', origin, side)."""
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = float(expand * max(x1 - x0, y1 - y0))
    gx0, gy0 = cx - side / 2.0, cy - side / 2.0
    s = canvas / side
    Kp = K.copy().astype(np.float32)
    Kp[0, 0] *= s; Kp[1, 1] *= s
    Kp[0, 2] = (K[0, 2] - gx0) * s
    Kp[1, 2] = (K[1, 2] - gy0) * s
    return Kp, (gx0, gy0), side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda"
    os.makedirs(OUT, exist_ok=True)

    seqs = sorted(f[:-4] for f in os.listdir(f"{CORR}/posed3d") if f.endswith(".npz"))
    mine = seqs[a.shard::a.nshards]
    print(f"[gt512] shard {a.shard}/{a.nshards}: {len(mine)}/{len(seqs)} seqs", flush=True)

    for seq in mine:
        dst = f"{OUT}/{seq}.tar"
        if os.path.exists(dst + ".done"):
            print(f"  skip {seq} (done)", flush=True); continue
        with np.load(f"{CORR}/posed3d/{seq}.npz", allow_pickle=True) as z:
            P = {k: z[k] for k in z.keys()}
        bbox = {}
        for ln in open(f"{CORR}/annotations/{seq}.jsonl"):
            r = json.loads(ln); bbox[r["uid"]] = r["bbox"]

        # group instance rows by frame: every person in the frame must be rasterised together or
        # inter-person occlusion -- the whole reason MAMMA exists -- silently breaks.
        frames = defaultdict(list)
        for i, uid in enumerate(P["uid"]):
            sub, fn, _ = str(uid).rsplit("_", 2)[0], str(uid).split("_")[-2], None
            frames[(sub, fn)].append(i)

        t0 = time.time(); n = 0
        tmp = dst + ".part"
        with tarfile.open(tmp, "w") as tf:
            for (sub, fn), rows in sorted(frames.items()):
                # pose every person ONCE per frame; only the crop camera changes per instance
                V, F, foff, meta = [], [], [], []
                off_v = off_f = 0
                for r in rows:
                    vw, f = pose_instance(P["pose_world"][r], P["betas"][r], P["trans_world"][r],
                                          str(P["gender"][r]), dev)
                    v = world_to_cam(vw, P["cam_ext"][r], dev)
                    V.append(v); F.append(f + off_v); foff.append(off_f)
                    meta.append(f.shape[0])
                    off_v += v.shape[0]; off_f += f.shape[0]
                verts = torch.cat(V, 0); faces = torch.cat(F, 0)
                starts = torch.tensor(foff, device=dev)

                for k, r in enumerate(rows):
                    uid = str(P["uid"][r])
                    if uid not in bbox:
                        continue
                    Kp, org, side = crop_K(P["cam_int"][r], bbox[uid])
                    gface, bary = _rasterize(verts, faces, Kp, CANVAS, CANVAS)
                    fg = gface >= 0
                    which = torch.bucketize(gface[fg].long(), starts, right=True) - 1
                    mine_ = torch.zeros_like(gface, dtype=torch.bool)
                    mine_[fg] = which == k                       # other people OCCLUDE, then become bg
                    face = torch.where(mine_, gface - starts[k], torch.full_like(gface, -1))
                    b2 = bary[..., :2].clone(); b2[~mine_] = 0
                    rgba = pack_rgba(face.cpu().numpy().astype(np.int32),
                                     b2.cpu().numpy().astype(np.float16))
                    # cv2 is 2x faster than PIL at the same output size (8.4 vs 18.3 ms, both 108 KB).
                    # NOTE cv2 wants BGRA: our channels are (fid_hi, fid_lo, bary_u, bary_v) -- an opaque
                    # payload, not a colour -- so we must NOT let cv2 swizzle them. Encode as-is by
                    # passing a 4ch array and reversing the first 3 to cancel cv2's BGR<->RGB swap.
                    ok, enc = cv2.imencode(".png", rgba[..., [2, 1, 0, 3]],
                                           [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    assert ok
                    data = enc.tobytes()
                    ti = tarfile.TarInfo(f"{uid}.png"); ti.size = len(data)
                    tf.addfile(ti, io.BytesIO(data))
                    # the loader needs the GT crop's placement to warp the prediction into it
                    j = json.dumps({"uid": uid, "gt_box": [org[0], org[1], side], "canvas": CANVAS,
                                    "expand": GT_EXPAND}).encode()
                    tj = tarfile.TarInfo(f"{uid}.json"); tj.size = len(j)
                    tf.addfile(tj, io.BytesIO(j))
                    n += 1
                if a.limit and n >= a.limit:
                    break
        os.replace(tmp, dst)
        open(dst + ".done", "w").write(str(n))
        print(f"  {seq}: {n} instances in {time.time()-t0:.0f}s "
              f"({1000*(time.time()-t0)/max(n,1):.1f} ms/inst)", flush=True)
    print("[gt512] SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
