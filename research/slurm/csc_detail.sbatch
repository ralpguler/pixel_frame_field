#!/bin/bash
#SBATCH --job-name=csc-detail
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
D=$HOME/data/bedlam2/dp_runs
~/codes/wan/.venv/bin/python -m orient.eval_csc_detail --n 512 --ckpt \
  $D/dp_csc_A/ckpt_025000.pt $D/dp_csc_C/ckpt_025000.pt \
  $D/dp_csc_D/ckpt_025000.pt $D/dp_csc_E/ckpt_025000.pt
