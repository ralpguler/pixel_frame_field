#!/bin/bash
#SBATCH --job-name=coco-lineage
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=04:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# COCO minival GT-box GPS on the orient-recipe ablation ckpts at MATCHED step 58k.
# Decomposes the dp_v2_L512 (0.633) -> dp_pp_llrd_1e3_mamma (0.565) drop:
# single-head/flat vs +per-part head vs +LLRD(3e-4) vs +LLRD lr1e-3.
# All 4 trained on v1 charts, BEDLAM-only, hflip off -> no uv remap needed.
set -uo pipefail
cd ~/codes/wan/densepose/dpeval
export HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1 CUDA_VISIBLE_DEVICES=0
PY=$HOME/codes/wan/.venv/bin/python
D=$HOME/data/bedlam2/dp_runs
ARGS="--backbone vit_large_patch16_dinov3 --layers 4,11,17,23 --input 512"
for R in dp_scratch_bedlam_orient dp_scratch_bedlam_orient_pp \
         dp_scratch_bedlam_orient_pp_llrd dp_scratch_bedlam_orient_pp_llrd_1e3; do
  echo "===== $R @58k ====="
  $PY run_eval.py $ARGS --ckpt $D/$R/ckpt_058000.pt
done
echo "COCO LINEAGE DONE"
