#!/bin/bash
#SBATCH --job-name=beauty-objects
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
set -uo pipefail
cd ~/codes/wan/densepose
export MPLBACKEND=Agg CUDA_VISIBLE_DEVICES=""
PY=$HOME/codes/wan/.venv/bin/python
H=$HOME/data/dp_viz_compare/beauty/15731089
apptainer exec -B /lus/lfs1aip2,/projects $HOME/data/containers/blender.sif blender --background \
  --python orient/blender_objects.py -- $H/objects $H $H/render/objects_seq --samples 48
$PY - <<'PYEOF'
import glob, os, json, imageio.v2 as imageio
B = os.path.expanduser("~/data/dp_viz_compare/beauty/15731089")
fps = json.load(open(f"{B}/meta.json"))["fps"]
fr = sorted(glob.glob(f"{B}/render/objects_seq/objects_*.png"))
wr = imageio.get_writer(f"{B}/render/objects.mp4", fps=fps, codec="libx264",
                        quality=8, pixelformat="yuv420p", macro_block_size=8)
for f in fr:
    wr.append_data(imageio.imread(f)[..., :3])
wr.close()
print("objects.mp4:", len(fr), "frames")
PYEOF
echo "OBJECTS VIDEO DONE"
