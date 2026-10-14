#!/bin/bash
#SBATCH --job-name=dp-train
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Single full GH200 node, 4-GPU DDP via torchrun.
# Usage: sbatch dp_train.sbatch [extra train.py args]
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=16
export TMPDIR=/local/user/$(id -u)/dp_scratch
mkdir -p "$TMPDIR"

# sample all-4-GPU utilization + memory + power every 15s (node-local nvidia-smi)
GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/dp-train-${SLURM_JOB_ID}.gpu
( while true; do
    date +%T
    nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw \
      --format=csv,noheader,nounits
    sleep 15
  done ) > "$GPU_LOG" 2>&1 &
SMPID=$!

# single-node: run torchrun directly (no srun -> avoids cpu-bind conflict; torchrun
# spawns the 4 local ranks itself). srun is only needed for multi-node.
~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=4 train.py "$@"
kill $SMPID 2>/dev/null
echo "[$(date +%F\ %T)] dp-train done"
