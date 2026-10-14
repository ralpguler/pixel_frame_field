"""Pick a held-out MAMMA-syn eval split: ~2% of SEQUENCES per sub-dataset, deterministic.

Shards are named <seq>_<idx>.tar with the sequence id embedded, so holding out whole sequences
is exact (no temporal leakage into training). Selection = md5(seq) mod 50 == 0, i.e. every
sub-dataset keeps ~98% of its sequences in training and contributes to the holdout.

Writes (dp_assets):
  mamma_holdout_seqs.txt    one seq name per line -> data.py build_loader(mamma_holdout=...)
  mamma_holdout_shards.txt  full shard paths      -> eval probes (--shards)

  python -m orient.build_mamma_holdout
"""
import glob
import hashlib
import os
import re
from collections import defaultdict

A = os.path.expanduser("~/data/bedlam2/dp_assets")
TRAIN = os.path.expanduser("~/data/datasets/dp_shards_512_mamma/train")

shards = sorted(glob.glob(f"{TRAIN}/*.tar"))
seq_shards = defaultdict(list)
for s in shards:
    seq_shards[re.sub(r"_\d+\.tar$", "", os.path.basename(s))].append(s)

held = sorted(q for q in seq_shards
              if int(hashlib.md5(q.encode()).hexdigest(), 16) % 50 == 0)
per_sub = defaultdict(lambda: [0, 0])
for q in seq_shards:
    sub = q.split("_")[0]
    per_sub[sub][0] += 1
    per_sub[sub][1] += q in held

with open(f"{A}/mamma_holdout_seqs.txt", "w") as f:
    f.write("\n".join(held) + "\n")
hshards = [s for q in held for s in seq_shards[q]]
with open(f"{A}/mamma_holdout_shards.txt", "w") as f:
    f.write("\n".join(hshards) + "\n")

for sub, (n, h) in sorted(per_sub.items()):
    print(f"{sub:15s} seqs {n:5d}  held {h:3d}")
print(f"TOTAL held {len(held)}/{len(seq_shards)} seqs, "
      f"{len(hshards)}/{len(shards)} shards ({100 * len(hshards) / len(shards):.1f}%)")
