#!/bin/bash
#SBATCH --job-name=dp-v21-M
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=440G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# V-JEPA 2.1 BACKBONE arm (2026-09-15, Riza): the W-M winner recipe with ONLY the backbone swapped DINOv3 ViT-L -> V-JEPA 2.1
# ViT-L (EMA encoder, image patchifier, taps 5/11/17/23). Compare against dp_w_M at matched steps, then F0-style joint COCO
# ft (as dp_w_MF) vs dp_w_MF@3k. From scratch to 120k via resume-chaining (submit 3 with afterany).
# Shared recipe = everything the evidence supports: per-part rotation head, LLRD base 3e-4
# (the real-domain sweet spot: LLRD structure +1.0 COCO AP, 1e-3 base LR -3.4), clean gt512,
# fold-free v2 charts, CSC-C loss, hidden-geometry mask, MAMMA sequence holdout.
# Arm from job name (submit with --job-name=dp-w-M|dp-w-B|dp-w-J):
#   M = BEDLAM2+MAMMA 50/50          (presumed winner base)
#   B = BEDLAM2 only (--no-mamma)    (the fair MAMMA control at the right LR)
#   J = joint: COCO interleaved all the way (tests 2-stage vs joint-throughout)
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16
# resume preloads optimizer states BEFORE the first backward -> peak = activations + opt states
# together; rank0 OOM'd by ~200MB with 5GB fragmented-reserved (job 5679215). Defragment:
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for _c in "/local/user/$(id -u)/dp_scratch" "/tmp/$(id -un)/dp_scratch" "/dev/shm/$(id -un)/dp_scratch"; do
  if mkdir -p "$_c" 2>/dev/null && [ -w "$_c" ]; then export TMPDIR="$_c"; break; fi
done
[ -n "${TMPDIR:-}" ] && [ -w "$TMPDIR" ] || { echo "FATAL: no writable TMPDIR"; exit 1; }

ARM=${SLURM_JOB_NAME##*-}
case "$ARM" in
  M) EXTRA="--data bedlam" ;;
  B) EXTRA="--data bedlam --no-mamma" ;;
  # J runs TWO loader stacks (dense + COCO): halve workers or the cgroup OOMs (F0 postmortem)
  J) EXTRA="--data joint --coco-period 5 --workers 6" ;;   # 20% COCO (Riza, 2026-07-16)
  *) echo "FATAL: bad ARM '$ARM' (job name '$SLURM_JOB_NAME')"; exit 1 ;;
esac
NAME=dp_v21_${ARM}
RUNDIR=~/data/bedlam2/dp_runs/$NAME
latest=$(ls -t "$RUNDIR"/ckpt_*.pt "$RUNDIR"/final.pt 2>/dev/null | head -1)
if [ -z "$latest" ]; then INITARG="--init none"; else INITARG="--resume $latest"; fi
echo "[v21 $(date +%F\ %T)] ARM=$ARM NAME=$NAME $INITARG"

GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/${NAME}-${SLURM_JOB_ID}.gpu
( while true; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits; sleep 60; done ) > "$GPU_LOG" 2>&1 &
SMPID=$!

~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=4 finetune.py \
  $EXTRA --backbone vjepa2_1_vit_large --layers 5,11,17,23 \
  --orient --rot-per-part --llrd --llrd-decay 0.75 \
  --lr 3e-4 --batch 32 --w-rot 1.0 --steps 120000 --warmup 500 \
  --gt512 --uv-asset smplx_face_uv_v2.npy \
  --w-xyz 2.0 --w-xyzgrad 5.0 --csc-clip 0.20 --csc-gather gt \
  --hidden-mask --mamma-holdout ~/data/bedlam2/dp_assets/mamma_holdout_seqs.txt \
  --save-every 2000 --val-every 2000 \
  --name "$NAME" $INITARG

kill $SMPID 2>/dev/null
echo "[v21 $(date +%F\ %T)] done"
