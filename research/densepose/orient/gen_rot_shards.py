"""Augment dp_shards -> dp_shards_rot by adding per-instance `rot.npy` (vert_rot6d (10475,6) f16).

Streams each existing shard (rgb.jpg + corr.png + meta.json), looks up the instance's posed SMPL-X
params in posed3d/<renderjob>.npz by uid, computes vert_rot6d (pose-matched to the rasterized
face+bary), and writes a self-contained shard with rgb+corr+rot+meta. Shard-by-tar for a Slurm array;
resumable (skips shards whose output already exists). Samples with no posed3d match are dropped.

Usage: gen_rot_shards.py --split train --shard <i> --nshards <N>
"""
import os, io, sys, glob, json, argparse, numpy as np
import webdataset as wds
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from orient.rot6d import vert_rot6d_for_body, vert_rot6d_for_body_gpu  # noqa: E402

ROOT = os.path.expanduser("~/data/bedlam2")
IN = f"{ROOT}/dp_shards"
OUT = f"{ROOT}/dp_shards_rot"
P3D = f"{ROOT}/correspondence/posed3d"

_CACHE = {}


def posed3d(seq):
    """Return (uid->row dict, npz) for a renderjob, cached."""
    if seq not in _CACHE:
        p = f"{P3D}/{seq}.npz"
        if not os.path.exists(p):
            _CACHE[seq] = (None, None)
        else:
            d = np.load(p, allow_pickle=True)
            _CACHE[seq] = ({u: i for i, u in enumerate(d["uid"])}, d)
        if len(_CACHE) > 6:                          # tiny LRU to bound memory
            _CACHE.pop(next(iter(_CACHE)))
    return _CACHE[seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--device", default="cpu")          # cpu (frees GPUs) or cuda
    a = ap.parse_args()
    gen = (lambda **kw: vert_rot6d_for_body_gpu(device="cuda", **kw)) if a.device == "cuda" \
        else (lambda **kw: vert_rot6d_for_body(device="cpu", **kw))
    os.makedirs(f"{OUT}/{a.split}", exist_ok=True)
    tars = sorted(glob.glob(f"{IN}/{a.split}/shard-*.tar"))
    mine = tars[a.shard::a.nshards]
    print(f"[{a.split}] shard {a.shard}/{a.nshards}: {len(mine)} of {len(tars)} tars", flush=True)

    for tar in mine:
        out_tar = f"{OUT}/{a.split}/{os.path.basename(tar)}"
        if os.path.exists(out_tar):
            print(f"  skip {os.path.basename(tar)} (done)", flush=True); continue
        tmp = out_tar + ".tmp"
        n_in = n_out = 0
        with wds.TarWriter(tmp) as sink:
            for s in wds.WebDataset(tar, shardshuffle=False, empty_check=False,
                                    handler=wds.warn_and_continue):
                n_in += 1
                try:
                    meta = json.loads(s["meta.json"])
                    u2r, d = posed3d(meta["seq"])
                    if u2r is None or meta["uid"] not in u2r:
                        continue
                    k = u2r[meta["uid"]]
                    vr = gen(pose_world=d["pose_world"][k], betas=d["betas"][k],
                             trans_world=d["trans_world"][k], ext=d["cam_ext"][k],
                             gender=str(d["gender"][k]))
                    buf = io.BytesIO(); np.save(buf, vr)
                    sink.write({"__key__": s["__key__"], "rgb.jpg": s["rgb.jpg"],
                                "corr.png": s["corr.png"], "meta.json": s["meta.json"],
                                "rot.npy": buf.getvalue()})
                    n_out += 1
                except Exception as e:
                    print(f"    sample err {s.get('__key__','?')}: {e!r}", flush=True)
        os.rename(tmp, out_tar)
        print(f"  {os.path.basename(tar)}: {n_out}/{n_in} written", flush=True)


if __name__ == "__main__":
    main()
