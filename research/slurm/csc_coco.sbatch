#!/bin/bash
#SBATCH --job-name=csc-coco
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=05:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# Official COCO minival GPS eval (GT boxes) for baseline + A/C/D/E.
# face_uv_v2 ckpts get the exact v2->v1 chart remap on hands/feet (COCO GT is v1-chart);
# C is also run WITHOUT the remap to quantify the chart handicap.
set -uo pipefail
cd ~/codes/wan/densepose/dpeval
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs
ARGS="--backbone vit_large_patch16_dinov3 --layers 4,11,17,23 --input 512"
RM="--uv-remap $HOME/data/bedlam2/dp_assets/uv_v2_to_v1.npy"
echo "===== baseline ckpt_120000 (v1 charts, no remap) ====="
$PY run_eval.py $ARGS --ckpt $D/dp_pp_llrd_1e3_mamma/ckpt_120000.pt
for R in A C D E; do
  echo "===== dp_csc_$R (remap) ====="
  $PY run_eval.py $ARGS --ckpt $D/dp_csc_$R/ckpt_025000.pt $RM
done
echo "===== dp_csc_C WITHOUT remap (chart-handicap probe) ====="
$PY run_eval.py $ARGS --ckpt $D/dp_csc_C/ckpt_025000.pt
echo "CSC COCO DONE"
