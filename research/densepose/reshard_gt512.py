"""Join the new fixed-canvas 512 correspondence onto the existing shards.

The RGB and the per-vertex rotation table are unchanged and are copied through as raw bytes (no decode,
no re-encode) -- only `corr.png` is replaced and `meta.json` gains the GT crop's placement.

The result is a shard whose GT is a FIXED 512x512 crop of the person that the loader NEVER resamples.
Both boxes live in SOURCE-FRAME pixel coordinates, so the loader can build the exact affine that warps a
(jittered) image crop into GT space:
    meta["box"]    = [bx0, by0, side]   the stored 2.0x-context RGB canvas   (unchanged)
    meta["gt_box"] = [gx0, gy0, side]   the 1.2x-context 512^2 GT canvas     (new)

  python reshard_gt512.py --shard 0 --nshards 32
"""
import argparse
import io
import json
import os
import tarfile
from collections import OrderedDict

import webdataset as wds

ROOT = os.path.expanduser("~/data/bedlam2")
SRC = os.environ.get("RS_SRC", f"{ROOT}/dp_shards_rot")
CORR512 = os.environ.get("RS_CORR", f"{ROOT}/corr512")
DST = os.environ.get("RS_DST", f"{ROOT}/dp_shards_512")


class SeqTars:
    """LRU of open per-seq correspondence tars. Shards are written seq-major, so 3 is plenty."""

    def __init__(self, cap=3):
        self.cap = cap
        self.o = OrderedDict()

    def get(self, seq):
        if seq in self.o:
            self.o.move_to_end(seq)
            return self.o[seq]
        p = f"{CORR512}/{seq}.tar"
        if not os.path.exists(p):
            self.o[seq] = None
            return None
        tf = tarfile.open(p)
        names = {m.name: m for m in tf.getmembers()}
        self.o[seq] = (tf, names)
        if len(self.o) > self.cap:
            k, v = self.o.popitem(last=False)
            if v:
                v[0].close()
        return self.o[seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--nshards", type=int, default=32)
    a = ap.parse_args()

    jobs = []
    for split in ("train", "val"):
        for t in sorted(os.listdir(f"{SRC}/{split}")):
            if t.endswith(".tar"):
                jobs.append((split, t))
    mine = jobs[a.shard::a.nshards]
    print(f"[reshard] task {a.shard}/{a.nshards}: {len(mine)}/{len(jobs)} shards", flush=True)

    tars = SeqTars()
    n_ok = n_miss = 0
    for split, t in mine:
        out = f"{DST}/{split}/{t}"
        if os.path.exists(out + ".done"):
            print(f"  skip {split}/{t}", flush=True); continue
        os.makedirs(f"{DST}/{split}", exist_ok=True)
        tmp = out + ".part"
        ds = wds.WebDataset([f"{SRC}/{split}/{t}"], empty_check=False,
                            handler=wds.warn_and_continue).to_tuple(
            "__key__", "rgb.jpg", "meta.json", "rot.npy", handler=wds.warn_and_continue)
        k_ok = k_miss = 0
        with wds.TarWriter(tmp) as w:
            for key, rgb, mj, rot in ds:
                m = json.loads(mj)
                st = tars.get(m["seq"])
                if st is None:
                    k_miss += 1; continue
                tf, names = st
                pk, jk = f"{m['uid']}.png", f"{m['uid']}.json"
                if pk not in names or jk not in names:
                    k_miss += 1; continue
                png = tf.extractfile(names[pk]).read()
                g = json.loads(tf.extractfile(names[jk]).read())
                m["gt_box"] = g["gt_box"]; m["canvas"] = g["canvas"]; m["gt_expand"] = g["expand"]
                w.write({"__key__": key, "rgb.jpg": rgb, "corr.png": png,
                         "meta.json": json.dumps(m).encode(), "rot.npy": rot})
                k_ok += 1
        os.replace(tmp, out)
        open(out + ".done", "w").write(f"{k_ok} {k_miss}")
        n_ok += k_ok; n_miss += k_miss
        print(f"  {split}/{t}: {k_ok} written, {k_miss} missing", flush=True)
    print(f"[reshard] TASK DONE: {n_ok} written, {n_miss} missing "
          f"({100*n_miss/max(n_ok+n_miss,1):.2f}%)", flush=True)


if __name__ == "__main__":
    main()
