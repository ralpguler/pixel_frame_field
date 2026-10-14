#!/bin/bash
#SBATCH --job-name=dp-preproc
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
# CPU-only (no re-render): uses saved face+bary crops. PEXELS_NUM_SHARDS = array size.
# Usage: sbatch --array=0-63 --export=ALL,PEXELS_NUM_SHARDS=64 dp_preprocess.sbatch
set -uo pipefail
cd ~/codes/wan/densepose
export TMPDIR=/local/user/$(id -u)/dp_scratch
mkdir -p "$TMPDIR"
export PEXELS_SHARD=${SLURM_ARRAY_TASK_ID:-0}
export PEXELS_NUM_SHARDS=${PEXELS_NUM_SHARDS:-$((SLURM_ARRAY_TASK_MAX + 1))}
~/codes/wan/.venv/bin/python preprocess_shards.py || { echo "FAILED shard $PEXELS_SHARD"; exit 1; }
echo "[$(date +%F\ %T)] dp-preproc shard $PEXELS_SHARD done"
