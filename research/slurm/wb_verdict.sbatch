#!/bin/bash
#SBATCH --job-name=wb-verdict
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Verdict battery for the completed from-scratch arms W-M (BEDLAM+MAMMA) and W-J (20% COCO
# anchored): COCO GT-box GPS (v2 remap), BEDLAM-val eval_512, held-out MAMMA probe.
# Key question: does W-J sit OFF the F0/F0b frontier (COCO ~0.72+ at holdout <9)?
set -uo pipefail
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs
ARGS="--backbone vit_large_patch16_dinov3 --layers 4,11,17,23 --input 512"
RM="--uv-remap $HOME/data/bedlam2/dp_assets/uv_v2_to_v1.npy"
HOLD=$HOME/data/datasets/dp_shards_rot_mamma/holdout_links

cd ~/codes/wan/densepose/dpeval
for R in dp_w_B; do
  echo "===== COCO: $R/final ====="
  $PY run_eval.py $ARGS --ckpt $D/$R/final.pt $RM
done

cd ~/codes/wan/densepose
for R in dp_w_B; do
  echo "===== eval_512: $R ====="
  $PY -m orient.eval_512 --ckpt $D/$R/ckpt_120000.pt --n 512
  echo "===== holdout probe: $R ====="
  $PY -m orient.indomain_probe --run $R --ckpts 120000 --shards "$HOLD/*.tar" --nshards 59 --batches 40
done
echo "WMJ VERDICT DONE"
