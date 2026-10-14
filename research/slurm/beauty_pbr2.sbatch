#!/bin/bash
#SBATCH --job-name=beauty-pbr2
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Round-2 PBR: sequins + diffraction (numpy spectral shaders) + chrome (Cycles, Nishita sky),
# plus a satin re-render -- the G-buffers are re-baked first from the CURRENT smooth-field dumps
# (the v3 satin had silently reused stale geodesic-tangent bakes).
set -uo pipefail
cd ~/codes/wan/densepose
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
PY=$HOME/codes/wan/.venv/bin/python
B=$HOME/data/dp_viz_compare/beauty
BL="apptainer exec -B /lus/lfs1aip2,/projects $HOME/data/containers/blender.sif blender --background --python"
HERO=$B/15731089

echo "=== stage 1: re-bake ALL G-buffers from the smooth-field dumps ==="
for C in 15731089 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] && $PY -m orient.viz_satin_bake --dump $B/$C --out $B/$C/bake &
done
wait

echo "=== stage 2: look tests first (hero stills) ==="
$BL orient/blender_satin.py -- $HERO/bake $HERO $HERO/render --samples 64 --still 75 --look chrome
$PY -m orient.viz_sequin   --dump $HERO --out $HERO/render &
$PY -m orient.viz_diffract --dump $HERO --out $HERO/render &
wait

echo "=== stage 3: hero videos ==="
$PY -m orient.viz_sequin   --dump $HERO --out $HERO/render --video &
$PY -m orient.viz_diffract --dump $HERO --out $HERO/render --video &
wait
$BL orient/blender_satin.py -- $HERO/bake $HERO $HERO/render/chrome_seq --samples 48 --look chrome
$BL orient/blender_satin.py -- $HERO/bake $HERO $HERO/render/satin_seq --samples 48 --look satin
$PY - <<'EOF'
import glob, os, json, cv2, numpy as np, imageio.v2 as imageio
B = os.path.expanduser("~/data/dp_viz_compare/beauty/15731089")
fps = json.load(open(f"{B}/meta.json"))["fps"]
for look in ("chrome", "satin"):
    fr = sorted(glob.glob(f"{B}/render/{look}_seq/{look}_*.png"))
    wr = imageio.get_writer(f"{B}/render/{look}.mp4", fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p", macro_block_size=8)
    for f in fr:
        x = imageio.imread(f)[..., :3].astype(np.float32) / 255
        exc = np.clip(x - 0.72, 0, None)
        x += 0.5 * cv2.GaussianBlur(exc, (0, 0), 6) + 0.3 * cv2.GaussianBlur(exc, (0, 0), 18)
        wr.append_data((np.clip(x, 0, 1) * 255).astype(np.uint8))
    wr.close()
    print(look, len(fr), "frames")
EOF

echo "=== stage 4: stills for the other clips ==="
for C in 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] || continue
  $PY -m orient.viz_sequin   --dump $B/$C --out $B/$C/render &
  $PY -m orient.viz_diffract --dump $B/$C --out $B/$C/render &
  wait
  $BL orient/blender_satin.py -- $B/$C/bake $B/$C $B/$C/render --samples 64 --still 0 --look chrome
  $BL orient/blender_satin.py -- $B/$C/bake $B/$C $B/$C/render --samples 64 --still 0 --look satin
done

echo "=== stage 5: report assets ==="
$PY -m orient.beauty_report_assets
echo "BEAUTY PBR2 ALL DONE"
