#!/bin/bash
#SBATCH --job-name=dp-mamma
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --mem=440G
#SBATCH --time=16:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Combined BEDLAM2 + MAMMA-syn (~50/50 via data.py) oriented-DensePose training, winning config
# (per-part rotation + LLRD lr1e-3), from scratch to 120k via resume-chaining. Submit several afterany.
set -uo pipefail
cd ~/codes/wan/densepose
export HF_HOME=~/data/hf_home TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=16
export TMPDIR=/local/user/$(id -u)/dp_scratch; mkdir -p "$TMPDIR"
NAME=dp_pp_llrd_1e3_mamma
RUNDIR=~/data/bedlam2/dp_runs/$NAME
latest=$(ls -t "$RUNDIR"/ckpt_*.pt "$RUNDIR"/final.pt 2>/dev/null | head -1)
if [ -z "$latest" ]; then INITARG="--init none"; else INITARG="--resume $latest"; fi
echo "[chain $(date +%F\ %T)] NAME=$NAME  $INITARG"
GPU_LOG=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/dp-mamma-${SLURM_JOB_ID}.gpu
( while true; do date +%T; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 30; done ) > "$GPU_LOG" 2>&1 &
SMPID=$!
~/codes/wan/.venv/bin/torchrun --standalone --nproc_per_node=4 finetune.py \
  --data bedlam --orient --rot-per-part --llrd --llrd-decay 0.75 \
  --lr 1e-3 --batch 32 --w-rot 1.0 --steps 120000 --warmup 500 --save-every 2000 --val-every 2000 \
  --name "$NAME" $INITARG
kill $SMPID 2>/dev/null
echo "[chain $(date +%F\ %T)] done"
