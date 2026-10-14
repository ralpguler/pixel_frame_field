#!/bin/bash
#SBATCH --job-name=cface
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
~/codes/wan/.venv/bin/python -u orient/field_dump.py \
  $HOME/data/datasets/pexels_hifps/clips/28513779_conformed16.mp4 \
  --out $HOME/data/dp_viz_compare/beauty/28513779_cscC --frame 75 \
  --ckpt $HOME/data/bedlam2/dp_runs/dp_csc_C/ckpt_025000.pt --uv-asset smplx_face_uv_v2.npy
echo "C FACE DUMP DONE"
