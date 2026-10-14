#!/bin/bash
#SBATCH --job-name=beauty-dump
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Per-pixel FORWARD field dump for the direction-viz beauty contest:
# hero dance clip (150 frames) + still frame 75 of the three other multi-viewer clips.
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
cd $HOME/codes/wan/densepose
PY=$HOME/codes/wan/.venv/bin/python
CLIPS=$HOME/data/datasets/pexels_hifps/clips
OUT=$HOME/data/dp_viz_compare/beauty

$PY -u orient/field_dump.py $CLIPS/15731089_conformed16.mp4 \
    --out $OUT/15731089 --video --maxframes 150
for C in 15153663 28513779 37828157; do
  $PY -u orient/field_dump.py $CLIPS/${C}_conformed16.mp4 --out $OUT/$C --frame 75
done
echo "BEAUTY DUMP ALL DONE"
