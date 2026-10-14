#!/bin/bash
#SBATCH --job-name=dpw3-mamma
#SBATCH --account=brics.u6ga
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/home/u6ga/raguler.u6ga/codes/wan/slurm/logs/%x-%j.out
# 3DPW test with +MAMMA-120k correspondence vs pure-BEDLAM-120k (baseline + pure-BEDLAM refined cached).
# 3DPW has 10968 two-person frames (62% of instances) -> the RIGHT benchmark for MAMMA's person-person
# occlusion advantage (RICH was 100% single-person). Extract MAMMA corr, refine (--tag _mamma), then
# STRATIFY the result by frame person-count (single vs 2-person) to isolate any occlusion-subset gain.
set -uo pipefail
CHMR=$HOME/codes/CameraHMR
CHMR_PY=$CHMR/.venv/bin/python
WAN_PY=$HOME/codes/wan/.venv/bin/python
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$HOME/data/hf_home TORCH_COMPILE_DISABLE=1
MAMMA_CKPT=$HOME/data/bedlam2/dp_runs/dp_pp_llrd_1e3_mamma/ckpt_120000.pt
CORR=$CHMR/dp_corr_3dpw_mamma.npz

echo "==================== STEP 1: DensePose corr with +MAMMA-120k ===================="
cd "$HOME/codes/wan/densepose"
$WAN_PY -u extract_dp_corr_3dpw.py --batch 16 --ckpt "$MAMMA_CKPT" --out "$CORR" || { echo "EXTRACT FAILED"; exit 1; }
[ -s "$CORR" ] || { echo "no corr npz"; exit 1; }

echo "==================== STEP 2: refinement + 3-way table (+MAMMA corr) ===================="
cd "$CHMR"
$CHMR_PY -u refine_3dpw.py --mode full --only both --lam_rot 0.003 --corr "$CORR" --tag _mamma || { echo "REFINE FAILED"; exit 1; }

echo "==================== STEP 3: stratify by person-count (single vs 2-person) ===================="
$CHMR_PY -u stratify_3dpw.py || { echo "STRATIFY FAILED"; exit 1; }
echo "==================== 3DPW +MAMMA PIPELINE DONE ===================="
