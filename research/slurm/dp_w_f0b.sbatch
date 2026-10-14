#!/bin/bash
#SBATCH --job-name=dp-w-F0b
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=440G
#SBATCH --time=08:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# F0: joint BEDLAM(+MAMMA)+COCO fine-tune from dp_csc_C -- the quick winner candidate.
# Proven ft recipe (flat 0.1 backbone, lr 5e-5, the one that took dp_v3 0.624 -> 0.738),
# + CSC stays on for synthetic batches, + hidden-geometry mask (also fixes eye/lip speckle).
# COCO UV loss auto-skips hands/feet (v1-chart targets vs v2-chart model; part CE keeps them).
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16
for _c in "/local/user/$(id -u)/dp_scratch" "/tmp/$(id -un)/dp_scratch" "/dev/shm/$(id -un)/dp_scratch"; do
  if mkdir -p "$_c" 2>/dev/null && [ -w "$_c" ]; then export TMPDIR="$_c"; break; fi
done
[ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ] || { echo "FATAL: no writable TMPDIR"; exit 1; }

GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/dp_w_F0b-${SLURM_JOB_ID}.gpu
( while true; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits; sleep 60; done ) > "$GPU_LOG" 2>&1 &
SMPID=$!

# OOM postmortem (job 5679209, killed @7300/10000): joint mode runs TWO loader stacks
# (dense gt512 + COCO) x 4 ranks; 12+12 workers each blew the 400G cgroup. Halve workers,
# and resume from the latest checkpoint instead of restarting.
RUNDIR=~/data/bedlam2/dp_runs/dp_w_F0b
latest=$(ls -t "$RUNDIR"/ckpt_*.pt "$RUNDIR"/final.pt 2>/dev/null | head -1)
if [ -z "$latest" ]; then INITARG="--init $HOME/data/bedlam2/dp_runs/dp_csc_C/ckpt_025000.pt"
else INITARG="--resume $latest"; fi
echo "[F0] $INITARG"

~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=4 finetune.py \
  --data joint --coco-period 5 --orient --rot-per-part \
  --lr 5e-5 --bb-mult 0.1 --batch 32 --w-rot 1.0 --steps 10000 --warmup 200 \
  --workers 6 \
  --gt512 --uv-asset smplx_face_uv_v2.npy \
  --w-xyz 2.0 --w-xyzgrad 5.0 --csc-clip 0.20 --csc-gather gt \
  --hidden-mask --mamma-holdout ~/data/bedlam2/dp_assets/mamma_holdout_seqs.txt \
  --save-every 1000 --val-every 500 \
  --name dp_w_F0b $INITARG

kill $SMPID 2>/dev/null
echo "[F0 $(date +%F\ %T)] done"
