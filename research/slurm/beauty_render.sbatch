#!/bin/bash
#SBATCH --job-name=beauty-render
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Direction-viz BEAUTY CONTEST renders, all CPU. Order: quick stills first (inspectable early),
# then the hero-clip videos, then the Blender Cycles satin sweep.
set -uo pipefail
cd ~/codes/wan/densepose
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
PY=$HOME/codes/wan/.venv/bin/python
B=$HOME/data/dp_viz_compare/beauty
SIF=$HOME/data/containers/blender.sif
HERO=$B/15731089

echo "=== stage 1: stills, all clips ==="
for C in 15731089 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] || continue
  $PY -m orient.viz_lic   --dump $B/$C --out $B/$C/render --color twist &
  $PY -m orient.viz_lic   --dump $B/$C --out $B/$C/render --color normal &
  $PY -m orient.viz_hatch --dump $B/$C --out $B/$C/render --style ink &
  $PY -m orient.viz_hatch --dump $B/$C --out $B/$C/render --style neon &
  wait
done

echo "=== stage 2: satin still test (look check before the full sweep) ==="
$PY -m orient.viz_satin_bake --dump $HERO --out $HERO/bake
for C in 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] && $PY -m orient.viz_satin_bake --dump $B/$C --out $B/$C/bake &
done
wait
apptainer exec -B /lus/lfs1aip2,/projects $SIF blender --background --python orient/blender_satin.py -- \
  $HERO/bake $HERO $HERO/render --samples 48 --still 75

echo "=== stage 3: hero videos (wind / lic / hatch) ==="
$PY -m orient.viz_wind  --dump $HERO --out $HERO/render --color twist
$PY -m orient.viz_wind  --dump $HERO --out $HERO/render --color white &
$PY -m orient.viz_lic   --dump $HERO --out $HERO/render --color twist --video &
$PY -m orient.viz_hatch --dump $HERO --out $HERO/render --style ink --video &
wait

echo "=== stage 4: satin full sweep + encode ==="
apptainer exec -B /lus/lfs1aip2,/projects $SIF blender --background --python orient/blender_satin.py -- \
  $HERO/bake $HERO $HERO/render/satin_seq --samples 48
$PY - <<'EOF'
import glob, os, json, cv2, numpy as np, imageio.v2 as imageio
B = os.path.expanduser("~/data/dp_viz_compare/beauty/15731089")
fps = json.load(open(f"{B}/meta.json"))["fps"]
fr = sorted(glob.glob(f"{B}/render/satin_seq/satin_*.png"))
wr = imageio.get_writer(f"{B}/render/satin.mp4", fps=fps, codec="libx264",
                        quality=8, pixelformat="yuv420p", macro_block_size=8)
for f in fr:
    x = imageio.imread(f)[..., :3].astype(np.float32) / 255
    exc = np.clip(x - 0.72, 0, None)          # subtle bloom on the sheen
    x += 0.5 * cv2.GaussianBlur(exc, (0, 0), 6) + 0.3 * cv2.GaussianBlur(exc, (0, 0), 18)
    wr.append_data((np.clip(x, 0, 1) * 255).astype(np.uint8))
wr.close()
print("satin.mp4:", len(fr), "frames")
EOF

echo "=== stage 5: satin stills for the other clips ==="
for C in 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] && apptainer exec -B /lus/lfs1aip2,/projects $SIF blender --background --python orient/blender_satin.py -- \
    $B/$C/bake $B/$C $B/$C/render --samples 64 --still 0
done
echo "BEAUTY RENDER ALL DONE"
