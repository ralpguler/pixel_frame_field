"""Process a shard of BEDLAM2 sequences into dense correspondence + posed 3D.

Shard by sequence: worker `shard` handles seqs[shard::nshards]. Resumable:
a sequence whose annotations jsonl + posed3d npz both exist is skipped.

Env: PEXELS_SHARD / PEXELS_NUM_SHARDS (reused sharding convention), or --shard/-n.
"""
import argparse
import os
import time

from encode_correspondence import process_sequence

LABELS = os.path.expanduser("~/data/bedlam2/labels_npz")          # extracted npz
MOVIES = os.path.expanduser("~/data/bedlam2/downloader")          # <seq>_mp4.tar
OUT = os.path.expanduser("~/data/bedlam2/correspondence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int,
                    default=int(os.environ.get("PEXELS_SHARD", 0)))
    ap.add_argument("--nshards", type=int,
                    default=int(os.environ.get("PEXELS_NUM_SHARDS", 1)))
    ap.add_argument("--limit-frames", type=int, default=None)
    args = ap.parse_args()

    seqs = sorted(f[:-4] for f in os.listdir(LABELS) if f.endswith(".npz"))
    mine = seqs[args.shard::args.nshards]
    print(f"shard {args.shard}/{args.nshards}: {len(mine)} of {len(seqs)} seqs")

    for seq in mine:
        done_ann = f"{OUT}/annotations/{seq}.jsonl"
        done_p3d = f"{OUT}/posed3d/{seq}.npz"
        if os.path.exists(done_ann) and os.path.exists(done_p3d):
            print(f"  skip {seq} (done)"); continue
        tar = f"{MOVIES}/{seq}_mp4.tar"
        if not os.path.exists(tar):
            print(f"  MISSING movie tar for {seq}"); continue
        t0 = time.time()
        try:
            n = process_sequence(seq, LABELS, tar, OUT,
                                 limit_frames=args.limit_frames)
            print(f"  {seq}: {n} instances in {time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            print(f"  {seq} FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    main()
