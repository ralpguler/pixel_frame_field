#!/bin/bash
#SBATCH --job-name=ff-reproj
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Re-export the 150-frame tracked framefield (bg + per-frame points/quaternions) with the FIXED
# orthonormal template tangent, for the in-browser reprojection viewer.
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
cd $HOME/codes/wan/densepose
VDIR=/lus/lfs1aip2/projects/u6ga/rguler/data/dp_viz_compare/framefield_video_fixed
mkdir -p $VDIR
$HOME/codes/wan/.venv/bin/python -u orient/framefield_overlay.py \
  $HOME/data/datasets/pexels_hifps/clips/15731089_conformed16.mp4 \
  --n 340 --export-blender $VDIR --video --maxframes 150
echo "REPROJ EXPORT DONE -> $VDIR"
