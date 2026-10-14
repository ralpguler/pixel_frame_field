#!/bin/bash
#SBATCH --job-name=mamma512
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=16:00:00
#SBATCH --array=0-67%32
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1
for _c in "/local/user/$(id -u)/m512" "/tmp/$(id -un)/m512" "/dev/shm/$(id -un)/m512"; do
  if mkdir -p "$_c" 2>/dev/null && [ -w "$_c" ]; then export TMPDIR="$_c"; break; fi
done
LIST=~/codes/wan/slurm/mamma_shards.txt
OUT=~/data/datasets/dp_shards_512_mamma/train
mkdir -p "$OUT"
N=$(wc -l < "$LIST"); TASKS=68; per=$(( (N + TASKS - 1) / TASKS ))
start=$(( SLURM_ARRAY_TASK_ID * per ))
echo "task $SLURM_ARRAY_TASK_ID: shards $((start+1))..$((start+per)) of $N"
sed -n "$((start+1)),$((start+per))p" "$LIST" | while read tar; do
  name=$(echo "$tar" | sed -E 's#.*/mammasyn/##; s#/#_#g; s#\.tar$##')
  if [ -f "$OUT/$name.tar" ]; then echo "skip $name"; continue; fi
  ~/codes/wan/.venv/bin/python orient/mamma_to_dp_shards.py --shard "$tar" --out "$OUT" --outname "$name" --gt512 2>&1 | tail -1 || echo "FAIL $name"
done
echo "task $SLURM_ARRAY_TASK_ID DONE"
