#!/bin/bash
#SBATCH --job-name=gt512rs
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --array=0-31
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
# Join the fixed-canvas 512 correspondence onto the existing shards. Pure byte-shuffle: rgb.jpg and
# rot.npy are copied through without decoding, only corr.png is replaced and meta.json gains gt_box.
# CPU-only. Submitted with --dependency=afterok on the render array so it fires unattended.
set -uo pipefail
cd ~/codes/wan/densepose
for _c in "/local/user/$(id -u)/gt512rs" "/tmp/$(id -un)/gt512rs" "/dev/shm/$(id -un)/gt512rs"; do
  if mkdir -p "$_c" 2>/dev/null && [ -w "$_c" ]; then export TMPDIR="$_c"; break; fi
done
[ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ] || { echo "FATAL: no writable TMPDIR"; exit 1; }

echo "[reshard] task ${SLURM_ARRAY_TASK_ID}/32 on $(hostname)"
~/codes/wan/.venv/bin/python -u reshard_gt512.py --shard "${SLURM_ARRAY_TASK_ID}" --nshards 32
echo "[reshard] TASK ${SLURM_ARRAY_TASK_ID} EXIT=$?"
