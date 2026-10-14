#!/bin/bash
#SBATCH --job-name=bedlam-corr
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
set -uo pipefail
cd ~/codes/wan/bedlam
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export TMPDIR=/local/user/$(id -u)/bedlam_scratch  # node-local for tar extract
mkdir -p "$TMPDIR"
export PEXELS_SHARD=${SLURM_ARRAY_TASK_ID:-0}
export PEXELS_NUM_SHARDS=${PEXELS_NUM_SHARDS:-$((SLURM_ARRAY_TASK_MAX + 1))}
~/codes/wan/.venv/bin/python run_correspondence.py
echo "[$(date +%F\ %T)] shard $PEXELS_SHARD done"
