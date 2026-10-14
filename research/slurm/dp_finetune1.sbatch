#!/bin/bash
#SBATCH --job-name=dp-ft1
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=110G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Single-GPU fine-tune (backfills easier than a 4-GPU full-node ask). Usage: sbatch dp_finetune1.sbatch [finetune.py args]
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home
export TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=16
export TMPDIR=/local/user/$(id -u)/dp_scratch
mkdir -p "$TMPDIR"
~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=1 finetune.py "$@"
echo "[$(date +%F\ %T)] dp-ft1 done"
