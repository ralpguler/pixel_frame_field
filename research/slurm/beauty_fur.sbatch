#!/bin/bash
#SBATCH --job-name=beauty-fur
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# FUR: full-clip strand bake + Cycles hero videos in both shaders + other-clip stills + assets.
set -uo pipefail
cd ~/codes/wan/densepose
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES="" MPLBACKEND=Agg
PY=$HOME/codes/wan/.venv/bin/python
B=$HOME/data/dp_viz_compare/beauty
BL="apptainer exec -B /lus/lfs1aip2,/projects $HOME/data/containers/blender.sif blender --background --python"
HERO=$B/15731089

echo "=== bake strands (all clips, final params) ==="
for C in 15731089 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] && $PY -m orient.viz_fur_bake --dump $B/$C --out $B/$C/fur &
done
wait

echo "=== hero videos, both shaders ==="
$BL orient/blender_fur.py -- $HERO/fur $HERO $HERO/render/fur_seq --samples 48 --shader hair
$BL orient/blender_fur.py -- $HERO/fur $HERO $HERO/render/fur_seq --samples 48 --shader principled
$PY - <<'EOF'
import glob, os, json, cv2, numpy as np, imageio.v2 as imageio
B = os.path.expanduser("~/data/dp_viz_compare/beauty/15731089")
fps = json.load(open(f"{B}/meta.json"))["fps"]
for pat, name in (("fur_0", "fur_hair"), ("fur_p_0", "fur_tinsel")):
    fr = sorted(glob.glob(f"{B}/render/fur_seq/{pat}*.png"))
    wr = imageio.get_writer(f"{B}/render/{name}.mp4", fps=fps, codec="libx264",
                            quality=8, pixelformat="yuv420p", macro_block_size=8)
    for f in fr:
        x = imageio.imread(f)[..., :3].astype(np.float32) / 255
        exc = np.clip(x - 0.75, 0, None)
        x += 0.4 * cv2.GaussianBlur(exc, (0, 0), 5)
        wr.append_data((np.clip(x, 0, 1) * 255).astype(np.uint8))
    wr.close()
    print(name, len(fr), "frames")
EOF

echo "=== other-clip stills (hair shader) ==="
for C in 15153663 28513779 37828157; do
  [ -f $B/$C/meta.json ] && $BL orient/blender_fur.py -- $B/$C/fur $B/$C $B/$C/render \
    --samples 64 --still 0 --shader hair
done

$PY -m orient.beauty_report_assets
echo "BEAUTY FUR ALL DONE"
