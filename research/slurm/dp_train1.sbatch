#!/bin/bash
#SBATCH --job-name=dp-train1
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=1:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16
GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/dp-train1-${SLURM_JOB_ID}.gpu
( while true; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw --format=csv,noheader,nounits; sleep 10; done ) > "$GPU_LOG" 2>&1 &
SMPID=$!
~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=1 train.py "$@"
kill $SMPID 2>/dev/null
echo "[$(date +%F\ %T)] dp-train1 done"
