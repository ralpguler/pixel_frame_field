#!/bin/bash
#SBATCH --job-name=cdump-v2
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0 MPLBACKEND=Agg
~/codes/wan/.venv/bin/python -u orient/field_dump.py \
  $HOME/data/datasets/pexels_hifps/clips/15731089_conformed16.mp4 \
  --out $HOME/data/dp_viz_compare/beauty/15731089_cscC --video --maxframes 150 \
  --ckpt $HOME/data/bedlam2/dp_runs/dp_csc_C/ckpt_025000.pt --uv-asset smplx_face_uv_v2.npy
echo "C DUMP V2 DONE"
