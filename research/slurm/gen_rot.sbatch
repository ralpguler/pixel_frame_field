#!/bin/bash
#SBATCH --job-name=gen-rot
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=110G
#SBATCH --time=8:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
# Generate dp_shards_rot (add per-instance vert_rot6d). JOB ARRAY: 1 GPU/task, SLURM binds the GPU
# (no manual CUDA_VISIBLE_DEVICES). Each array task handles tar-slice [task::nshards].
# Usage: sbatch --array=0-15 gen_rot.sbatch <split>     (nshards = array size)
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home
SPLIT=${1:-train}
NSH=$(( SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1 ))
~/codes/wan/.venv/bin/python orient/gen_rot_shards.py --split "$SPLIT" \
    --shard "$SLURM_ARRAY_TASK_ID" --nshards "$NSH"
echo "[$(date +%F\ %T)] gen-rot $SPLIT task $SLURM_ARRAY_TASK_ID done"
