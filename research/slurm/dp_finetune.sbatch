#!/bin/bash
#SBATCH --job-name=dp-ft
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Fine-tune BEDLAM-pretrained DensePose on real DensePose-COCO. 4-GPU DDP via torchrun.
# Usage: sbatch dp_finetune.sbatch --data coco|joint --name <run> [extra finetune.py args]
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=16
export TMPDIR=/local/user/$(id -u)/dp_scratch
mkdir -p "$TMPDIR"

GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/dp-ft-${SLURM_JOB_ID}.gpu
( while true; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,power.draw \
      --format=csv,noheader,nounits; sleep 15; done ) > "$GPU_LOG" 2>&1 &
SMPID=$!

~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=4 finetune.py "$@"
kill $SMPID 2>/dev/null
echo "[$(date +%F\ %T)] dp-ft done"
