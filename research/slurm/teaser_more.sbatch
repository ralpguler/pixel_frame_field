#!/bin/bash
#SBATCH --job-name=teaser-more
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Template-vs-image teaser (approved look: raw photo dimmed 0.45, saturated thick tripods) on ~70 NEW pexels frames:
# single-frame field dumps with the shipped model dp_w_MF@3k -> orient/teaser_auto.py -> overview sheets.
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
cd $HOME/codes/wan/densepose
PY=$HOME/codes/wan/.venv/bin/python
CK=$HOME/data/bedlam2/dp_runs/dp_w_MF/ckpt_003000.pt
V=$HOME/data/dp_viz_compare; OUT=$V/teasers_more; mkdir -p $OUT
$PY - <<'PYEOF' > $OUT/candidates.txt
import glob, os, random, cv2
used = {os.path.basename(p).split("_")[0] for p in glob.glob(os.path.expanduser("~/data/dp_viz_compare/beauty/*_wMF"))}
pool = [("D", p) for p in glob.glob(os.path.expanduser("~/data/datasets/pexels_dance/raw/*.mp4"))] + \
       [("H", p) for p in glob.glob(os.path.expanduser("~/data/datasets/pexels_hifps/clips/*_conformed16.mp4"))]
pool = [(t, p) for t, p in pool if os.path.basename(p).split(".")[0].split("_")[0] not in used]
random.Random(7).shuffle(pool)
for t, p in pool[:70]:
    n = int(cv2.VideoCapture(p).get(7)); print(os.path.basename(p).split(".")[0].split("_")[0], p, 75 if n >= 150 else max(0, n // 2))
PYEOF
echo "$(wc -l < $OUT/candidates.txt) candidates"
while read id clip f; do
  [ -f $V/beauty/${id}_wMF/meta.json ] && continue
  $PY -m orient.field_dump $clip --out $V/beauty/${id}_wMF --frame $f --ckpt $CK --uv-asset smplx_face_uv_v2.npy > $OUT/dump_${id}.log 2>&1 \
    && echo "$(date +%T) dumped $id f$f" || echo "$(date +%T) DUMP FAIL $id"
done < $OUT/candidates.txt
$PY - <<'PYEOF' > $OUT/todo.txt
import json, os
for ln in open(os.path.expanduser("~/data/dp_viz_compare/teasers_more/candidates.txt")):
    id_, clip, f = ln.split(); m = os.path.expanduser(f"~/data/dp_viz_compare/beauty/{id_}_wMF/meta.json")
    if os.path.exists(m) and (json.load(open(m)).get("side") or 0) >= 600: print(id_, f)
PYEOF
echo "$(wc -l < $OUT/todo.txt) with person crop >= 600 px"
one() { id=$1; f=$2; V=$HOME/data/dp_viz_compare
  $HOME/codes/wan/.venv/bin/python -m orient.teaser_auto --dump $V/beauty/${id}_wMF --frame $f --tag ${id}_tsr --fig $V/teasers_more/${id}.png > $V/teasers_more/${id}.log 2>&1 \
    && echo "$(date +%T) ok $id" || echo "$(date +%T) FAIL $id"; }
export -f one
cat $OUT/todo.txt | xargs -P 2 -L 1 bash -c 'one $0 $1'
$PY - <<'PYEOF'
from PIL import Image
import glob, os
O = os.path.expanduser("~/data/dp_viz_compare/teasers_more"); fs = sorted(glob.glob(f"{O}/[0-9]*.png")); cols, W = 5, 800
for s in range(0, len(fs), 20):
    tiles = [Image.open(f).convert("RGB") for f in fs[s:s + 20]]; tiles = [t.resize((W, int(t.height * W / t.width)), Image.LANCZOS) for t in tiles]
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    sheet = Image.new("RGB", (cols * W + 30 * (cols - 1), sum(max(t.height for t in r) for r in rows) + 30 * len(rows)), "white"); y = 0
    for r in rows:
        for k, t in enumerate(r): sheet.paste(t, (k * (W + 30), y))
        y += max(t.height for t in r) + 30
    sheet.save(f"{O}/sheet_{s // 20 + 1}.jpg", quality=85); print("sheet", s // 20 + 1, len(tiles))
PYEOF
echo "TEASER-MORE DONE $(date +%T)"
