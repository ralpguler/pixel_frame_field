#!/bin/bash
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Canonical-Surface-Coordinate ablation, fine-tuned from the 120k checkpoint on CLEAN GT:
#   * the fixed 512^2 per-instance GT canvas (--gt512): the GT is rasterised straight into it and NEVER
#     resampled. Kills ROT_CAP/CSC_CAP and the bilinear-rotation bug (3.2% of MAMMA rotation labels were
#     wrong by >30 deg, up to 180 deg, right at the occlusion boundaries MAMMA exists to teach).
#   * the FOLD-FREE v2 charts: the old charts were non-injective on hands/feet (55% of samples shared a
#     (part,uv) with another triangle, up to 149 mm apart).
# Both change the GT, so the baseline MUST be retrained -- A is what isolates "clean GT" from "new loss".
#
#   RUN=A   baseline: v2 charts, existing losses only
#   RUN=C   + L_xyz + L_grad, UV gathered by the GT part      (the loss itself)
#   RUN=D   + L_xyz + L_grad, UV gathered by argmax(pred)     (training matches inference)
#   RUN=E   + L_xyz + L_grad, SOFT top-3 part mixture          (metric gradient into the PART logits --
#           measured: given the GT part, uv already nails the occlusion step to 1.3mm; the blur and the
#           local noise are entirely part-classification artifacts, so this is the run that targets them)
#
# usage: sbatch --export=ALL,RUN=C slurm/dp_csc.sbatch
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16
# torchrun writes its worker logs under TMPDIR. /local/user/<uid> exists on the LOGIN node but NOT on the
# compute nodes (and /local is not writable there), so the old `mkdir -p` failed silently under
# `set -uo pipefail` and torchrun died with PermissionError. Pick the first path we can actually WRITE to.
for _c in "/local/user/$(id -u)/dp_scratch" "/tmp/$(id -un)/dp_scratch" "/dev/shm/$(id -un)/dp_scratch"; do
  if mkdir -p "$_c" 2>/dev/null && [ -w "$_c" ]; then export TMPDIR="$_c"; break; fi
done
[ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ] || { echo "FATAL: no writable TMPDIR"; exit 1; }
echo "[csc] TMPDIR=$TMPDIR (on $(hostname))"

# Derive RUN from the job name (dp-csc-X) so we do not silently depend on --export propagating an env
# var: if it did not, every job would build the SAME run and be indistinguishable in squeue.
RUN=${RUN:-${SLURM_JOB_NAME##*-}}
case "$RUN" in A|C|D|E) ;; *) echo "FATAL: bad RUN='$RUN' (job name '$SLURM_JOB_NAME')"; exit 1 ;; esac
NAME=dp_csc_${RUN}
RUNDIR=~/data/bedlam2/dp_runs/$NAME
INIT=~/data/bedlam2/dp_runs/dp_pp_llrd_1e3_mamma/ckpt_120000.pt

case "$RUN" in
  A) CSC_ARGS="--w-xyz 0    --w-xyzgrad 0" ;;
  C) CSC_ARGS="--w-xyz 2.0  --w-xyzgrad 5.0 --csc-clip 0.20 --csc-gather gt" ;;
  D) CSC_ARGS="--w-xyz 2.0  --w-xyzgrad 5.0 --csc-clip 0.20 --csc-gather pred" ;;
  E) CSC_ARGS="--w-xyz 2.0  --w-xyzgrad 5.0 --csc-clip 0.20 --csc-gather soft" ;;
  *) echo "bad RUN=$RUN"; exit 1 ;;
esac

# resume-chain if this run already has a checkpoint
latest=$(ls -t "$RUNDIR"/ckpt_*.pt "$RUNDIR"/final.pt 2>/dev/null | head -1)
if [ -z "$latest" ]; then INITARG="--init $INIT"; else INITARG="--resume $latest"; fi
echo "[csc $(date +%F\ %T)] RUN=$RUN NAME=$NAME $INITARG"
echo "[csc] args: $CSC_ARGS"

GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/${NAME}-${SLURM_JOB_ID}.gpu
( while true; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits; sleep 60; done ) > "$GPU_LOG" 2>&1 &
SMPID=$!

~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=4 finetune.py \
  --data bedlam --orient --rot-per-part --llrd --llrd-decay 0.75 \
  --lr 3e-4 --batch 32 --w-rot 1.0 --steps 25000 --warmup 500 \
  --save-every 2500 --val-every 2500 \
  --gt512 --uv-asset smplx_face_uv_v2.npy $CSC_ARGS \
  --name "$NAME" $INITARG

kill $SMPID 2>/dev/null
echo "[csc $(date +%F\ %T)] RUN=$RUN DONE"
