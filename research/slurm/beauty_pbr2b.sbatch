#!/bin/bash
#SBATCH --job-name=beauty-pbr2b
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Round-2 PBR, final pass with the tuned looks: hero videos + other-clip stills + assets.
set -uo pipefail
cd ~/codes/wan/densepose
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
PY=$HOME/codes/wan/.venv/bin/python
B=$HOME/data/dp_viz_compare/beauty
BL="apptainer exec -B /lus/lfs1aip2,/projects $HOME/data/containers/blender.sif blender --background --python"
HERO=$B/15731089

echo "=== hero videos ==="
$PY -m orient.viz_sequin   --dump $HERO --out $HERO/render --video &
$PY -m orient.viz_diffract --dump $HERO --out $HERO/render --video &
wait
$BL orient/blender_satin.py -- $HERO/bake $HERO $HERO/render/chrome_seq --samples 48 --look chrome
$PY - <<'EOF'
import glob, os, json, cv2, numpy as np, imageio.v2 as imageio
B = os.path.expanduser("~/data/dp_viz_compare/beauty/15731089")
fps = json.load(open(f"{B}/meta.json"))["fps"]
fr = sorted(glob.glob(f"{B}/render/chrome_seq/chrome_*.png"))
wr = imageio.get_writer(f"{B}/render/chrome.mp4", fps=fps, codec="libx264",
                        quality=8, pixelformat="yuv420p", macro_block_size=8)
for f in fr:
    x = imageio.imread(f)[..., :3].astype(np.float32) / 255
    exc = np.clip(x - 0.72, 0, None)
    x += 0.5 * cv2.GaussianBlur(exc, (0, 0), 6) + 0.3 * cv2.GaussianBlur(exc, (0, 0), 18)
    wr.append_data((np.clip(x, 0, 1) * 255).astype(np.uint8))
wr.close()
print("chrome.mp4:", len(fr), "frames")
EOF

echo "=== other-clip stills ==="
for C in 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] || continue
  $PY -m orient.viz_sequin   --dump $B/$C --out $B/$C/render &
  $PY -m orient.viz_diffract --dump $B/$C --out $B/$C/render &
  wait
  $BL orient/blender_satin.py -- $B/$C/bake $B/$C $B/$C/render --samples 64 --still 0 --look chrome
done

$PY -m orient.beauty_report_assets
echo "BEAUTY PBR2B ALL DONE"
