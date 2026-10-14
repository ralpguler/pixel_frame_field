#!/bin/bash
#SBATCH --job-name=coco-w-F0
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# COCO GT-box GPS of the F0 joint fine-tune (final + midpoint), v2->v1 remap (model stays v2-chart).
set -uo pipefail
cd ~/codes/wan/densepose/dpeval
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs/dp_w_F0
ARGS="--backbone vit_large_patch16_dinov3 --layers 4,11,17,23 --input 512"
RM="--uv-remap $HOME/data/bedlam2/dp_assets/uv_v2_to_v1.npy"
for CK in ckpt_005000.pt final.pt; do
  echo "===== dp_w_F0/$CK ====="
  $PY run_eval.py $ARGS --ckpt $D/$CK $RM
done
echo "F0 COCO DONE"
