#!/bin/bash
#SBATCH --job-name=gt512
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --array=0-21
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%A_%a.out
# Re-render the BEDLAM2 correspondence GT into a FIXED 512x512 per-instance crop canvas.
#
# Validated bit-exact against the ORIGINAL rasteriser output before launching (fg IoU 1.0000, 100%
# identical face ids, median surface difference 0.0000 mm). Note the crop camera is MORE accurate than
# the current shard label, which is a cv2.warpAffine-NEAREST resample carrying ~0.5 px of quantisation.
#
# Gains: BEDLAM person 239 -> 427 px of label (2.8x more label pixels on the body), and the GT is then
# NEVER resampled again -- no ROT_CAP, no CSC_CAP, no cv2.resize of rotations (which was averaging two
# different surfaces across occlusion boundaries: 3.2% of rotation labels wrong by >30 deg, up to 180 deg).
set -uo pipefail
cd ~/codes/wan/bedlam
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=8

for _c in "/local/user/$(id -u)/gt512" "/tmp/$(id -un)/gt512" "/dev/shm/$(id -un)/gt512"; do
  if mkdir -p "$_c" 2>/dev/null && [ -w "$_c" ]; then export TMPDIR="$_c"; break; fi
done
[ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ] || { echo "FATAL: no writable TMPDIR"; exit 1; }

echo "[gt512] task ${SLURM_ARRAY_TASK_ID}/22 on $(hostname)  TMPDIR=$TMPDIR"
~/codes/wan/.venv/bin/python -u render_gt512.py --shard "${SLURM_ARRAY_TASK_ID}" --nshards 22
echo "[gt512] TASK ${SLURM_ARRAY_TASK_ID} DONE"
