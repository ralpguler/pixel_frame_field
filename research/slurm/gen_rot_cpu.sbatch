#!/bin/bash
#SBATCH --job-name=gen-rot
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
# CPU-only generation of dp_shards_rot (NO GPU -> leaves GPUs for training; schedules easily).
# JOB ARRAY: each task handles tar-slice [task::nshards]. Usage: sbatch --array=0-63 gen_rot_cpu.sbatch <split>
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=""
SPLIT=${1:-train}
NSH=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
~/codes/wan/.venv/bin/python orient/gen_rot_shards.py --split "$SPLIT" --device cpu \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSH"
echo "[$(date +%F\ %T)] gen-rot-cpu $SPLIT task $SLURM_ARRAY_TASK_ID done"
